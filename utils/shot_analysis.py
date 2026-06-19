"""
utils/shot_analysis.py

Detects the moment of the shots and classifies forehand/backhand starting from
the ball CSV produced by ballTracking/BallTracking.py (frame,x,y,w,h,cx,cy,area)
and from the players CSV.

Shot detection (on the ball track smoothed with Savitzky-Golay):
  1. candidates = persistent sign reversal of the vertical velocity vy
     (in image: a ball traveling toward the far player has vy<0, toward the
     near player vy>0; a shot reverses the direction) COMBINED with peaks of
     the acceleration magnitude |dv| (abrupt change of velocity);
  2. a candidate is a SHOT only if the ball is close to a player's bounding
     box (bounces also reverse vy but happen far from the players);
  3. minimum gap between consecutive shots (default 0.5 s), keeping the
     candidate with the larger acceleration.

Before any velocity/speed is computed the ball track is DESPIKED: a position
that jumps far from its local neighbours for a single frame and then returns is
a tracking outlier (a confident YOLO mis-detection — net cord, line, shoe) and
is removed (set NaN, then refilled by interpolation), because one such spike
would otherwise produce a huge spurious speed (see _despike_track).

Forehand / backhand at the shot frame:
  - the player's court side (near/far) from his feet point projected in
    meters: near = we see him from behind, his right is the image right;
    far = we see him from the front, his right is the image LEFT;
  - forehand if the ball is on the dominant-hand side, backhand otherwise;
    for left-handers (--p1-hand/--p2-hand left) the reasoning is inverted;
  - if the ball is almost on the body axis the shot is marked "unknown".

Shot type (drive / slice / dropshot) from the OUTGOING ball PACE measured just
after contact. Pace is SCALE-FREE: 100 * (ball pixels/frame just after contact)
/ (striker's on-screen box height). We deliberately do NOT use a court-metre
km/h: the ball is airborne at contact and the ground-plane homography turns its
arc into large fake distances (worst at the far baseline, where the court is a
few pixels deep), so homography km/h scatters by court end, not by how hard the
ball was hit. Dividing the pixel speed by the striker's box height cancels
perspective, making near/far players comparable.
  - pace >= --drive-thr            -> "drive"   (normal groundstroke)
  - --dropshot-thr <= pace < drive -> "slice"   (medium-pace control shot)
  - pace < --dropshot-thr          -> "dropshot"
  - pace unavailable               -> "unknown"
  Pace alone cannot detect backspin, so "slice" means a medium-pace control shot
  inferred from speed; the numeric ball_pace is also emitted. Defaults fit
  ~1080p/30fps footage — CALIBRATE --drive-thr / --dropshot-thr against the PNGs.

Use (from project root, after generating the ball CSV with BallTracking):
    python utils/shot_analysis.py --ball outputs/ball_clip2.csv
    python utils/shot_analysis.py --ball outputs/ball_clip2.csv \\
        --p1-hand right --p2-hand left
    python utils/shot_analysis.py --ball outputs/ball_clip2.csv \\
        --drive-thr 15 --dropshot-thr 7

Validation of the logic without the YOLO model (synthetic trajectory):
    python utils/shot_analysis.py --self-test
"""

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter

_FT = 0.3048
W_m = 27.0 * _FT
L_m = 78.0 * _FT
NET = L_m / 2.0


# ── loading ────────────────────────────────────────────────────────────────────

def _despike_track(full: pd.DataFrame, win: int = 5,
                   mad_mult: float = 4.0, px_gate: float = 40.0) -> int:
    """
    Remove single-frame position outliers ("spike then return") from the
    realigned ball track IN PLACE, setting the offending cx/cy to NaN so the
    caller's interpolation refills them. Returns how many points were removed.

    Why this is needed: the ball CSV interpolates dropouts but never rejects a
    confident YOLO mis-detection (a net cord, a line marking, an opponent's
    shoe). Such a point survives as a real sample, and because ball SPEED is a
    frame-to-frame difference, a single 1-frame jump produces a huge spurious
    speed that would wreck the drive/slice/dropshot thresholds. Savitzky-Golay
    smoothing only BLENDS the outlier into its neighbours — it must be removed,
    not smoothed.

    Rule (per frame i, on the RAW realigned cx,cy, BEFORE smoothing):
      - centred rolling MEDIAN of cx and cy over `win` frames;
      - residual = distance from (cx,cy) to that local median;
      - i is a spike iff its residual exceeds BOTH
            (a) px_gate pixels                       (absolute floor), AND
            (b) mad_mult * 1.4826 * rolling MAD       (adaptive, scale-free).
    Keying on the distance to the local MEDIAN (not the raw frame-to-frame
    delta) is what separates a one-frame jump-and-return from a genuinely fast
    ball: a sustained fast ball keeps a small residual because the median
    tracks it, whereas a lone outlier spikes at exactly one index. Requiring
    BOTH gates means a real fast ball (~15-40 px/frame at 1080p/30fps) is never
    clipped while a ~60 px island is.
    """
    med_cx = full["cx"].rolling(win, center=True, min_periods=3).median()
    med_cy = full["cy"].rolling(win, center=True, min_periods=3).median()
    resid = np.hypot(full["cx"] - med_cx, full["cy"] - med_cy)
    mad = resid.rolling(win, center=True, min_periods=3).apply(
        lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))), raw=True)
    scale = 1.4826 * mad
    spike = (resid > px_gate) & (resid > mad_mult * scale)
    spike = spike.fillna(False).values
    n = int(spike.sum())
    if n:
        full.loc[spike, "cx"] = np.nan
        full.loc[spike, "cy"] = np.nan
    return n


def load_ball_track(ball_csv: str, despike_px: float = 40.0,
                    despike_mad: float = 4.0, despike_win: int = 5) -> pd.DataFrame:
    """
    Load the ball CSV, realign it over all frames, REMOVE single-frame position
    outliers (see _despike_track), interpolate short gaps and smooth cx/cy.
    Return a DataFrame indexed by frame with columns cx, cy (smoothed),
    NaN where the ball was never seen.
    """
    df = pd.read_csv(ball_csv)
    if df.empty:
        raise ValueError(f"Empty ball CSV: {ball_csv}")
    full = df.set_index("frame")[["cx", "cy"]].reindex(
        range(int(df["frame"].min()), int(df["frame"].max()) + 1))
    # Despike BEFORE interpolation, so the rolling median sees genuine ball
    # samples as neighbours; the interpolation then refills both the original
    # dropouts and the removed spikes.
    n_spk = _despike_track(full, win=despike_win, mad_mult=despike_mad,
                           px_gate=despike_px)
    if n_spk:
        print(f"  Despike: removed {n_spk} single-frame ball outlier(s) "
              f"before smoothing.")
    full = full.interpolate(limit=8, limit_area="inside")

    win = 9
    for col in ("cx", "cy"):
        vals = full[col].values.astype(float)
        ok = ~np.isnan(vals)
        if ok.sum() > win:
            sm = vals.copy()
            sm[ok] = savgol_filter(vals[ok], win, 2)
            full[col] = sm
        else:
            print(f"  Warning: skipping Savitzky-Golay smoothing of '{col}' "
                  f"(only {int(ok.sum())} valid points, need > {win}).")
    return full


def load_player_boxes(players_csv: str) -> dict:
    """{frame: {pid: (x, y, w, h)}} from the player tracker CSV."""
    df = pd.read_csv(players_csv)
    boxes = defaultdict(dict)
    for r in df.itertuples():
        boxes[int(r.frame)][int(r.player_id)] = (
            float(r.x), float(r.y), float(r.w), float(r.h))
    return boxes


# ── shot detection ─────────────────────────────────────────────────────────────

def _nearest_box(boxes, frame, pid, radius=3):
    """
    Box of player ``pid`` in the temporally nearest frame within ±radius, or
    None. Distance grows outward from ``frame`` (d = 0, 1, 2, …). Tie-break at
    equal distance |d|: prefer the LATER frame (frame + d) over the earlier one
    (frame - d) — a small, documented, deterministic rule so the result no
    longer silently depends on the probe order.
    """
    for d in range(radius + 1):
        for f in (frame + d, frame - d):   # later frame first on ties
            if f in boxes and pid in boxes[f]:
                return boxes[f][pid]
    return None


def _ball_near_player(ball_xy, box, expand_w=0.9, expand_h=0.55):
    """True if the ball is inside the player's expanded box."""
    x, y, w, h = box
    bx, by = ball_xy
    return (x - expand_w * w <= bx <= x + w + expand_w * w
            and y - expand_h * h <= by <= y + h + expand_h * h)


def _gradient_runs(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    np.gradient applied independently over each contiguous run where ``valid``
    is True, leaving every invalid (NaN) sample as NaN.

    Running np.gradient over a NaN-holed array contaminates the two neighbours
    of every gap (a single 1-frame ball dropout would poison the velocity on
    both sides and can hide a real shot next to it). By splitting the track
    into maximal valid runs and differentiating each run on its own, gaps stay
    isolated and the samples around them keep clean derivatives.
    """
    out = np.full(len(values), np.nan, dtype=float)
    i = 0
    n = len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        # contiguous valid run [i, j)
        if j - i >= 2:
            out[i:j] = np.gradient(values[i:j].astype(float))
        else:
            # a lone valid sample has no defined gradient -> leave NaN
            out[i:j] = np.nan
        i = j
    return out


def detect_hits(track: pd.DataFrame, boxes: dict, fps: float,
                min_gap_s: float = 0.5, vy_min: float = 0.5,
                acc_thr: float = 1.5, win: int = 4,
                reversal_look: int = 6,
                reversal_vy_frac: float = 0.5) -> list:
    """
    Return [(frame, player_id, acc_strength), ...] of the detected shots.

    Shot detection is BB-ENTRY-TRIGGERED: exactly one shot is registered per
    OUTSIDE->INSIDE transition of the (smoothed) ball center into a player's
    asymmetrically expanded box (_ball_near_player, expand_w=0.9/expand_h=0.55).
    A single physical contact therefore can no longer split into two hits the
    way the old vy-flip / acc-peak candidate spray + min_gap/chain_gap merge did.

    Pipeline:
      1. Entry scan per player: record the frame the ball crosses from outside
         to inside the expanded box (NaN ball or missing box counts as OUTSIDE).
      2. Same-player refractory merge: a brief edge-graze exit + re-entry of one
         contact (e.g. 800/812, 901/907 on the real clip) collapses into one
         entry. Different players are never merged here (alternation is the norm).
      3. Contact resolution: search FORWARD from each entry for the FIRST
         persistent vy sign reversal (the racket-imparted direction change); fall
         back to closest-approach-to-box-center if the window has no reversal.
      4. Cross-player overlap dedup: if two opposite-player entries resolve to the
         same contact frame (expanded boxes overlap), keep the nearer player only.
      5. Confirm-or-reject: keep the single resolved frame iff it has a real
         direction change (reversal), an acceleration burst, or the ball is still
         geometrically inside the box (soft contact). Pure boolean filter — it can
         only drop the one chosen frame, never add a second.

    vy_min, acc_thr, win, reversal_look, reversal_vy_frac keep their meaning from
    the original detector (see _persistent_reversal).
    """
    frames = track.index.values
    cx = track["cx"].values
    cy = track["cy"].values
    valid = ~(np.isnan(cx) | np.isnan(cy))

    # Differentiate over contiguous valid runs only (see _gradient_runs), so a
    # 1-frame ball dropout does not poison its neighbours' velocity/acceleration.
    vx = _gradient_runs(cx, valid)
    vy = _gradient_runs(cy, valid)
    valid_v = ~(np.isnan(vx) | np.isnan(vy))
    acc = np.hypot(_gradient_runs(vx, valid_v), _gradient_runs(vy, valid_v))

    rev_floor = reversal_vy_frac * vy_min
    min_valid = 2

    def _persistent_reversal(i):
        lo = max(0, i - reversal_look)
        hi = min(len(vy), i + 1 + reversal_look)
        if np.count_nonzero(valid_v[lo:i]) < min_valid:
            return False
        if np.count_nonzero(valid_v[i + 1:hi]) < min_valid:
            return False
        b = np.nanmean(vy[lo:i])
        a = np.nanmean(vy[i + 1:hi])
        if np.isnan(b) or np.isnan(a):
            return False
        return (b * a < 0) and (abs(a) >= rev_floor)

    n = len(frames)
    min_gap = int(min_gap_s * fps)
    # Same-player re-entry refractory: above the ~12-frame edge-graze doubles
    # (800/812, 901/907) but below the >=29-frame real inter-shot spacing.
    # Decoupled from min_gap so it gates re-entry dedup only.
    merge_gap = max(min_gap, int(round(0.45 * fps)))
    # Entry->contact lag budget. Real data shows the true vy reversal can lag the
    # entry by 17-21 frames (e.g. entry 642 -> reversal ~659), so ~0.8 s is needed;
    # max(min_gap, ...) keeps a sane floor at low fps.
    W = max(min_gap, int(round(0.8 * fps)))

    # ---- STAGE 1: OUTSIDE->INSIDE expanded-BB entry events per player --------
    raw = []   # (entry_index, pid)  -- store index for direct vy/acc access
    for pid in (1, 2):
        prev_inside = False
        for i in range(n):
            f = int(frames[i])
            inside = False
            if valid[i]:
                box = _nearest_box(boxes, f, pid)
                if box is not None and _ball_near_player((cx[i], cy[i]), box):
                    inside = True
            if inside and not prev_inside:
                raw.append((i, pid))
            prev_inside = inside
    raw.sort()

    # ---- STAGE 2: collapse same-player re-entries within merge_gap -----------
    merged = []
    for ei, pid in raw:
        f = int(frames[ei])
        if (merged and merged[-1][1] == pid
                and f - int(frames[merged[-1][0]]) <= merge_gap):
            continue                      # same physical contact, keep earliest
        merged.append((ei, pid))

    # ---- STAGE 3: resolve ONE contact frame per merged entry -----------------
    def _resolve(ei, pid):
        box0 = _nearest_box(boxes, int(frames[ei]), pid)
        bxc = box0[0] + box0[2] / 2.0 if box0 is not None else cx[ei]
        byc = box0[1] + box0[3] / 2.0 if box0 is not None else cy[ei]
        # Reversal CENTERS can reach ei+W (the full lag budget); the validity
        # guard below keeps the +/-win mean windows in range, so there is no
        # off-by-win clipping of the search.
        hi_center = min(n - win - 1, ei + W)
        for i in range(max(ei, win), hi_center + 1):
            if not valid[i - win:i + win + 1].all():
                continue
            before = np.nanmean(vy[i - win:i])
            after = np.nanmean(vy[i + 1:i + 1 + win])
            if before * after < 0:                  # FIRST persistent reversal
                return i, True
        # Fallback: no reversal in-window -> closest approach to box center.
        cand = [(i, np.hypot(cx[i] - bxc, cy[i] - byc))
                for i in range(ei, min(n, ei + W + 1)) if valid[i]]
        if not cand:
            return None
        return min(cand, key=lambda t: t[1])[0], False

    resolved = []   # (entry_index, pid, contact_index, have_rev)
    for ei, pid in merged:
        r = _resolve(ei, pid)
        if r is None:
            continue
        c, have_rev = r
        resolved.append((ei, pid, c, have_rev))

    # ---- STAGE 3b: cross-player overlap dedup (resolved contacts coincide) ---
    # With expand_w=0.9 the two expanded boxes can overlap; a single contact in
    # the overlap zone produces an entry for BOTH players. If two opposite-player
    # entries resolve to within merge_gap frames, keep the player whose box centre
    # is nearest the ball at the contact frame.
    def _center_dist(pp, idx):
        b = _nearest_box(boxes, int(frames[idx]), pp)
        if b is None:
            return float("inf")
        return np.hypot(cx[idx] - (b[0] + b[2] / 2.0),
                        cy[idx] - (b[1] + b[3] / 2.0))

    resolved.sort(key=lambda t: t[2])
    deduped = []
    for ei, pid, c, have_rev in resolved:
        if deduped:
            _, ppid, pc, _ = deduped[-1]
            if pid != ppid and abs(c - pc) <= merge_gap:
                if _center_dist(pid, c) < _center_dist(ppid, pc):
                    deduped[-1] = (ei, pid, c, have_rev)
                continue
        deduped.append((ei, pid, c, have_rev))

    # ---- STAGE 4 + 5: confirm-or-reject (boolean only), then emit ------------
    hits = []
    for ei, pid, c, have_rev in deduped:
        acc_ok = (not np.isnan(acc[c]) and acc[c] >= acc_thr
                  and acc[c] == np.nanmax(acc[max(0, c - win):c + win + 1])
                  and _persistent_reversal(c))
        geom_ok = False
        if not have_rev:
            box = _nearest_box(boxes, int(frames[c]), pid)
            if box is not None and _ball_near_player((cx[c], cy[c]), box):
                geom_ok = True            # soft contact: ball still in the box
        if not (have_rev or acc_ok or geom_ok):
            continue                      # stray / edge-graze entry: drop
        strength = float(acc[c]) if not np.isnan(acc[c]) else 0.0
        hits.append((int(frames[c]), pid, strength))

    hits.sort()                           # ascending-frame contract for analyze_shots
    return hits


# ── forehand / backhand classification ─────────────────────────────────────────

def classify_stroke(ball_cx, player_box, player_side, hand,
                    deadband_frac=0.12):
    """
    player_side : "near" | "far" (relative to the camera)
    hand        : "right" | "left"
    Return "forehand", "backhand" or "unknown" (ball on the body axis).
    """
    x, y, w, h = player_box
    player_cx = x + w / 2.0
    db = ball_cx - player_cx
    if abs(db) < deadband_frac * w:
        return "unknown"
    # the dominant hand is toward image-right if (near and right) or (far and left-handed)
    dominant_is_image_right = (player_side == "near") == (hand == "right")
    return "forehand" if (db > 0) == dominant_is_image_right else "backhand"


# ── shot-type (drive / slice / dropshot) from outgoing ball speed ────────────────
#
# IMPORTANT — why we do NOT use court-metre km/h here:
# Projecting the ball through the court (ground-plane) homography to get a km/h
# speed does not work for SHOT speed. At contact the ball is airborne and its
# image motion is dominated by its arc (height), which the ground homography
# turns into large fake ground distances — worst at the far baseline, where the
# court is compressed into a handful of pixels (~0.12 m per pixel vs ~0.025 m
# near the camera). The result is speeds that scatter by which side of the net a
# player stands on rather than by how hard the ball was hit (the far player P2
# came out almost random). Instead we use a SCALE-FREE proxy: the outgoing ball
# pixel-speed normalised by the player's on-screen box HEIGHT. The box height is
# a stand-in for a fixed real-world length (~player height) at that depth, so
# dividing by it cancels perspective and makes near/far players directly
# comparable. This is a relative "shot pace" index, not km/h.

def _outgoing_pace(track, boxes, f, pid, look=6, skip=1, min_pairs=3):
    """
    Scale-free outgoing ball pace at contact frame f for player `pid`.

    Returns 100 * (median per-frame outgoing pixel speed) / (player box height),
    i.e. the ball's outgoing image speed expressed as a percentage of the
    striker's on-screen height per frame. Perspective cancels (both numerator and
    the height reference scale the same way with depth), so the value reflects
    how hard the ball was struck regardless of court end. Returns NaN if fewer
    than `min_pairs` valid consecutive ball samples or no player box is found.

    The contact frame itself is skipped (`skip`): it is the noisiest point (ball
    deforming / occluded by the racket). The MEDIAN over the window rejects the
    odd jittery pair.
    """
    box = _nearest_box(boxes, f, pid)
    if box is None or box[3] <= 0:
        return float("nan")
    lo = f + skip
    hi = min(int(track.index.max()), lo + look)
    seg = track.loc[lo:hi]
    cx, cy = seg["cx"].values, seg["cy"].values
    steps = []
    for k in range(len(cx) - 1):
        if (np.isnan(cx[k]) or np.isnan(cy[k])
                or np.isnan(cx[k + 1]) or np.isnan(cy[k + 1])):
            continue
        steps.append(np.hypot(cx[k + 1] - cx[k], cy[k + 1] - cy[k]))
    if len(steps) < min_pairs:
        return float("nan")
    return float(100.0 * np.median(steps) / box[3])


def classify_shot_type(pace, drive_thr, dropshot_thr):
    """
    Speed-tier label from the scale-free outgoing pace (see _outgoing_pace):
        pace >= drive_thr                  -> "drive"   (normal groundstroke)
        dropshot_thr <= pace < drive_thr   -> "slice"   (medium control shot)
        pace < dropshot_thr                -> "dropshot"
        NaN / unavailable                  -> "unknown"

    NOTE: pace alone cannot detect backspin, so "slice" here means a
    medium-pace control shot INFERRED from speed, not a confirmed sliced stroke.
    The numeric pace is always emitted alongside so the call can be re-judged.
    """
    if pace is None or np.isnan(pace):
        return "unknown"
    if pace >= drive_thr:
        return "drive"
    if pace >= dropshot_thr:
        return "slice"
    return "dropshot"


def analyze_shots(track, boxes, conv, fps, hands, min_gap_s=0.5,
                  vy_min=0.5, acc_thr=1.5, win=4,
                  reversal_look=6, reversal_vy_frac=0.5,
                  drive_thr=15.0, dropshot_thr=4.0, pace_look=6):
    """Full pipeline: detect shots and classify them. Return a DataFrame."""
    # Diagnose a frame-numbering mismatch between the ball CSV and the players
    # CSV: if the two frame ranges don't overlap, _nearest_box can never match,
    # detect_hits yields nothing and the user only sees "No shots detected"
    # with no explanation. Warn explicitly instead.
    if len(track.index) and boxes:
        b0, b1 = int(track.index.min()), int(track.index.max())
        p0, p1 = min(boxes), max(boxes)
        if b1 < p0 or p1 < b0:
            print(f"  Warning: ball frames [{b0},{b1}] don't overlap player "
                  f"frames [{p0},{p1}] — frame numbering mismatch between the "
                  f"ball CSV and the players CSV; no shots can be detected.")

    hits = detect_hits(track, boxes, fps, min_gap_s=min_gap_s,
                       vy_min=vy_min, acc_thr=acc_thr, win=win,
                       reversal_look=reversal_look,
                       reversal_vy_frac=reversal_vy_frac)
    n_dropped = 0
    rows = []
    for f, pid, strength in hits:
        box = _nearest_box(boxes, f, pid)
        if box is None:
            continue
        # median of the ball position over ±2 frames (reduces noise)
        sel = track.loc[max(track.index.min(), f - 2): f + 2]
        # Guard against an all-NaN window (the ball was never seen around this
        # frame): np.nanmedian would return NaN and contaminate
        # classify_stroke / to_meters. Skip (and count) such shots instead.
        cx_win = sel["cx"].values
        cy_win = sel["cy"].values
        if np.all(np.isnan(cx_win)) or np.all(np.isnan(cy_win)):
            n_dropped += 1
            continue
        ball_cx = float(np.nanmedian(cx_win))
        ball_cy = float(np.nanmedian(cy_win))

        feet = (box[0] + box[2] / 2.0, box[1] + box[3])
        y_m = conv.to_meters(*feet)[1]
        side = "near" if y_m > NET else "far"
        stroke = classify_stroke(ball_cx, box, side, hands[pid])

        # NOTE: to_meters uses the ground-plane (court) homography, but at the
        # moment of contact the ball is airborne (~ racket height). Projecting
        # an above-ground point through a ground homography is only approximate
        # — treat ball_x_m / ball_y_m as indicative, not exact, court coords.
        bx_m, by_m = conv.to_meters(ball_cx, ball_cy)

        # Outgoing ball PACE just after contact -> drive / slice / dropshot.
        # Scale-free (pixel speed / striker box height); see _outgoing_pace for
        # why court-metre km/h is unusable for an airborne ball. Thresholds are
        # tunable and can be calibrated against the saved PNGs.
        pace = _outgoing_pace(track, boxes, f, pid, look=pace_look)
        shot_type = classify_shot_type(pace, drive_thr, dropshot_thr)

        rows.append({
            "frame": f, "time_s": round(f / fps, 2), "player_id": pid,
            "side": side, "hand": hands[pid], "stroke": stroke,
            "shot_type": shot_type,
            "ball_pace": (round(pace, 1) if not np.isnan(pace) else None),
            "ball_cx": round(ball_cx, 1), "ball_cy": round(ball_cy, 1),
            "player_cx": round(box[0] + box[2] / 2.0, 1),
            "ball_x_m": round(bx_m, 2), "ball_y_m": round(by_m, 2),
        })
    if n_dropped:
        print(f"  Note: skipped {n_dropped} candidate shot(s) with an all-NaN "
              f"ball window (ball not seen around the contact frame).")
    return pd.DataFrame(rows)


# ── output ─────────────────────────────────────────────────────────────────────

def save_shot_frames(shots: pd.DataFrame, video_path: str, boxes: dict,
                     out_dir: Path) -> None:
    """Verification PNGs: shot frame with player box and ball highlighted."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  (video not available, skipping PNGs: {video_path})")
        return
    for r in shots.itertuples():
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame)
        ok, fr = cap.read()
        if not ok:
            continue
        box = _nearest_box(boxes, r.frame, r.player_id)
        if box:
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(fr, (int(r.ball_cx), int(r.ball_cy)), 9, (0, 255, 255), 2)
        label = f"P{r.player_id} {r.stroke}/{r.shot_type} (frame {r.frame})"
        cv2.putText(fr, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 3)
        path = out_dir / f"shot_f{r.frame:05d}_P{r.player_id}_{r.stroke}.png"
        cv2.imwrite(str(path), fr)
    cap.release()
    print(f"  Shot PNGs saved in {out_dir}")


def print_summary(shots: pd.DataFrame, hands: dict) -> None:
    print("\n" + "=" * 56)
    print("  SHOT ANALYSIS  -  SUMMARY")
    print("=" * 56)
    if shots.empty:
        print("  No shots detected.")
        return
    for r in shots.itertuples():
        pace = f"{r.ball_pace:5.1f}" if r.ball_pace is not None else "  n/a"
        print(f"  frame {r.frame:5d} ({r.time_s:6.2f}s)  "
              f"P{r.player_id} ({r.side:4s}, {r.hand:5s})  ->  "
              f"{r.stroke:8s} [{r.shot_type:8s} pace {pace}]")
    print()
    for pid, sub in shots.groupby("player_id"):
        n_fh = (sub["stroke"] == "forehand").sum()
        n_bh = (sub["stroke"] == "backhand").sum()
        n_uk = (sub["stroke"] == "unknown").sum()
        print(f"  P{pid} ({hands[pid]:5s}): {len(sub)} shots  "
              f"-  forehands {n_fh}, backhands {n_bh}, unknown {n_uk}")
    print()
    for st in ("drive", "slice", "dropshot", "unknown"):
        print(f"  shot_type {st:9s}: {(shots['shot_type'] == st).sum()}")
    print()


# ── self-test with synthetic trajectory ────────────────────────────────────────

def _synthetic_court_csv(path: str) -> None:
    """Synthetic court: homography from 4 arbitrary px corners + ITF proportions."""
    corners_m = np.array([[0, 0], [W_m, 0], [0, L_m], [W_m, L_m]],
                         dtype=np.float32)                 # TL TR BL BR
    corners_px = np.array([[700, 300], [1200, 300], [400, 860], [1500, 860]],
                          dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners_m, corners_px)

    def proj(xm, ym):
        p = H @ np.array([xm, ym, 1.0])
        return p[0] / p[2], p[1] / p[2]

    svc_t, svc_b = 18.0 * _FT, L_m - 18.0 * _FT
    pts = {"TL": (0, 0), "TR": (W_m, 0), "BL": (0, L_m), "BR": (W_m, L_m),
           "STL": (0, svc_t), "STR": (W_m, svc_t),
           "SBL": (0, svc_b), "SBR": (W_m, svc_b)}
    with open(path, "w", newline="") as f:
        f.write("label,x,y\n")
        for lab, (xm, ym) in pts.items():
            x, y = proj(xm, ym)
            f.write(f"{lab},{x:.1f},{y:.1f}\n")


def self_test() -> None:
    """
    Synthetic rally: 4 shots with known ball side + mid-court bounces that
    must NOT be detected as shots. Checks frame, player and classification,
    including the left-handed case.
    """
    rng = np.random.default_rng(3)
    p1 = (900, 700, 100, 190)     # near, feet y=890, cx=950
    p2 = (920, 200, 60, 120)      # far,  feet y=320, cx=950

    # (shot_frame, player, ball offset relative to center, expected for right-hander)
    plan = [(20, 1, +70, "forehand"),
            (60, 2, -45, "forehand"),
            (100, 1, -70, "backhand"),
            (140, 2, +45, "backhand")]

    contact = {}
    for f, pid, dx, _ in plan:
        box = p1 if pid == 1 else p2
        cx_c = box[0] + box[2] / 2.0 + dx
        cy_c = box[1] + box[3] * (0.45 if pid == 1 else 0.5)
        contact[f] = (cx_c, cy_c)

    # trajectory: linear segments between contacts + bounce (cusp) at mid-court
    frames = np.arange(0, 171)
    cx = np.full(len(frames), np.nan)
    cy = np.full(len(frames), np.nan)
    keys = sorted(contact)
    segs = [(0, (950.0, 520.0), keys[0], contact[keys[0]])]
    segs += [(keys[i], contact[keys[i]], keys[i + 1], contact[keys[i + 1]])
             for i in range(len(keys) - 1)]
    segs += [(keys[-1], contact[keys[-1]], 170, (950.0, 520.0))]
    for f0, (x0, y0), f1, (x1, y1) in segs:
        for f in range(f0, f1 + 1):
            s = (f - f0) / max(1, f1 - f0)
            # bounce cusp at s=0.65 (deviation toward image bottom)
            bounce = 35.0 * max(0.0, 1.0 - abs(s - 0.65) / 0.18)
            cx[f] = x0 + s * (x1 - x0)
            cy[f] = y0 + s * (y1 - y0) + bounce
    cx += rng.normal(0, 0.8, len(frames))
    cy += rng.normal(0, 0.8, len(frames))

    # Inject a single-frame position outlier on a non-contact frame: the despike
    # in load_ball_track must remove it and the interpolation must restore a
    # value close to the true trajectory.
    SPIKE_F = 45
    cx_true, cy_true = float(cx[SPIKE_F]), float(cy[SPIKE_F])
    cx[SPIKE_F] += 120.0
    cy[SPIKE_F] -= 90.0

    with tempfile.TemporaryDirectory() as tmp:
        ball_csv = os.path.join(tmp, "ball.csv")
        pd.DataFrame({"frame": frames,
                      "x": cx - 5, "y": cy - 5, "w": 10, "h": 10,
                      "cx": cx, "cy": cy, "area": 100}).to_csv(
            ball_csv, index=False)
        players_csv = os.path.join(tmp, "players.csv")
        rows = []
        for f in frames:
            for pid, box in ((1, p1), (2, p2)):
                x, y, w, h = box
                rows.append([f, pid, x, y, w, h, x + w / 2, y + h / 2, w * h])
        pd.DataFrame(rows, columns=["frame", "player_id", "x", "y", "w", "h",
                                    "cx", "cy", "area"]).to_csv(
            players_csv, index=False)
        court_csv = os.path.join(tmp, "court.csv")
        _synthetic_court_csv(court_csv)

        track = load_ball_track(ball_csv)
        boxes = load_player_boxes(players_csv)
        conv = CourtConverter(court_csv)

        failures = []

        # Despike: the injected single-frame outlier at SPIKE_F must be removed
        # and refilled close to the true trajectory.
        got_cx = track.loc[SPIKE_F, "cx"]
        got_cy = track.loc[SPIKE_F, "cy"]
        if abs(got_cx - cx_true) > 8 or abs(got_cy - cy_true) > 8:
            failures.append(
                f"despike: outlier at frame {SPIKE_F} not removed "
                f"(got ({got_cx:.1f},{got_cy:.1f}), "
                f"true ~({cx_true:.1f},{cy_true:.1f}))")

        # Self-calibrate the two shot-type thresholds from the realized synthetic
        # outgoing PACE at the actually DETECTED hit frames, so the test asserts
        # the FASTEST detected shot is a "drive" and the SLOWEST a "dropshot"
        # without depending on absolute values. The synthetic rally naturally
        # spans a wide pace range (short fast segment vs long slow segment), so
        # no trajectory overwrite is needed — keeping detection untouched. Pace
        # needs the player id, so pair each detected frame with its pid.
        det = detect_hits(track, boxes, 30.0)
        pace = {f: _outgoing_pace(track, boxes, f, pid)
                for f, pid, _ in det}
        pace = {f: v for f, v in pace.items() if not np.isnan(v)}
        if len(pace) < 2:
            failures.append(f"shot-type setup: <2 measurable paces ({pace})")
            f_fast = f_slow = None
            drive_thr, dropshot_thr = 15.0, 7.0
        else:
            f_fast = max(pace, key=pace.get)
            f_slow = min(pace, key=pace.get)
            v_fast, v_slow = pace[f_fast], pace[f_slow]
            # drive_thr just below the fastest; dropshot_thr just above the
            # slowest -> fastest classifies "drive", slowest "dropshot".
            drive_thr = v_slow + 0.75 * (v_fast - v_slow)
            dropshot_thr = v_slow + 0.25 * (v_fast - v_slow)

        for hands, expect_fn in (
            ({1: "right", 2: "right"}, lambda _, exp: exp),
            ({1: "right", 2: "left"},
             lambda pid, exp: exp if pid == 1 else
             ("backhand" if exp == "forehand" else "forehand")),
        ):
            shots = analyze_shots(track, boxes, conv, fps=30.0, hands=hands,
                                  drive_thr=drive_thr, dropshot_thr=dropshot_thr)
            tag = f"hands={hands}"
            if len(shots) != len(plan):
                failures.append(f"{tag}: expected {len(plan)} shots, "
                                f"detected {len(shots)}")
                continue
            for (f_exp, pid_exp, _, stroke_exp), r in zip(
                    plan, shots.itertuples()):
                stroke_exp = expect_fn(pid_exp, stroke_exp)
                if abs(r.frame - f_exp) > 4:
                    failures.append(f"{tag}: shot at frame {r.frame}, "
                                    f"expected ~{f_exp}")
                if r.player_id != pid_exp:
                    failures.append(f"{tag}: frame {r.frame} assigned to "
                                    f"P{r.player_id}, expected P{pid_exp}")
                if r.stroke != stroke_exp:
                    failures.append(f"{tag}: frame {r.frame} P{pid_exp} "
                                    f"{r.stroke}, expected {stroke_exp}")

            types = {r.frame: r.shot_type for r in shots.itertuples()}
            if f_fast is not None and types.get(f_fast) != "drive":
                failures.append(f"{tag}: fastest shot (frame {f_fast}) expected "
                                f"shot_type 'drive', got {types.get(f_fast)} "
                                f"(types {types})")
            if f_slow is not None and types.get(f_slow) != "dropshot":
                failures.append(f"{tag}: slowest shot (frame {f_slow}) expected "
                                f"shot_type 'dropshot', got {types.get(f_slow)} "
                                f"(types {types})")

    print("SELF-TEST shot_analysis")
    print(f"  planned shots : {[(f, p, s) for f, p, _, s in plan]}")
    if failures:
        print("  FAIL:")
        for msg in failures:
            print("   -", msg)
        raise SystemExit(1)
    print("  PASS: 4/4 shots detected and classified correctly "
          "(including the left-handed case); mid-court bounces ignored; "
          "single-frame outlier despiked; fast shot -> drive, slow -> dropshot.")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shot detection and forehand/backhand classification "
                    "from the ball tracking")
    parser.add_argument("--ball", default="outputs/ball_clip2.csv",
                        help="ball CSV generated by BallTracking.py")
    parser.add_argument("--players", default="outputs/players_Input_video2.csv")
    parser.add_argument("--court",
                        default="outputs/court_coordinates/Input_video2_court.csv")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="source video for the verification PNGs")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--p1-hand", choices=["right", "left"],
                        default="right", dest="p1_hand",
                        help="dominant hand of player 1 (default right)")
    parser.add_argument("--p2-hand", choices=["right", "left"],
                        default="right", dest="p2_hand",
                        help="dominant hand of player 2 (default right)")
    parser.add_argument("--min-gap", type=float, default=0.5, dest="min_gap",
                        help="minimum distance in seconds between two shots")
    # Detection thresholds (previously hardcoded in detect_hits). Defaults are
    # identical to the old hardcoded values, so behaviour is unchanged unless
    # the user overrides them.
    parser.add_argument("--vy-min", type=float, default=0.5, dest="vy_min",
                        help="min |mean vy| (px/frame) before/after for a "
                             "vy sign reversal to count as a candidate "
                             "(default 0.5)")
    parser.add_argument("--acc-thr", type=float, default=1.5, dest="acc_thr",
                        help="threshold (px/frame^2) on acceleration-magnitude "
                             "peaks (default 1.5)")
    parser.add_argument("--win", type=int, default=4, dest="win",
                        help="half-window (frames) for the before/after vy "
                             "means and the local acceleration-peak test "
                             "(default 4)")
    parser.add_argument("--reversal-look", type=int, default=6,
                        dest="reversal_look",
                        help="forward/backward half-window (frames) used to "
                             "confirm an acceleration-only candidate's vertical "
                             "reversal (default 6); not applied to candidates "
                             "that already pass the vy sign-reversal test")
    parser.add_argument("--reversal-vy-frac", type=float, default=0.5,
                        dest="reversal_vy_frac",
                        help="relaxed outgoing |mean vy| floor for that "
                             "confirmation, as a fraction of --vy-min "
                             "(default 0.5); pass 0 to require only a sign "
                             "change (closest to the pre-gate behaviour)")
    # Shot-type (drive/slice/dropshot) thresholds on the scale-free outgoing
    # PACE = 100 * (ball px/frame just after contact) / (striker box height).
    # This avoids the court homography, which cannot measure an airborne ball's
    # speed (see _outgoing_pace); the pace is comparable across near/far players.
    # Defaults fit ~1080p/30fps footage; calibrate against the saved PNGs.
    parser.add_argument("--drive-thr", type=float, default=15.0,
                        dest="drive_thr",
                        help="outgoing pace at/above which a shot is a 'drive'; "
                             "below it and above --dropshot-thr it is a 'slice' "
                             "(default 15); calibrate against the PNGs")
    parser.add_argument("--dropshot-thr", type=float, default=4.0,
                        dest="dropshot_thr",
                        help="outgoing pace below which a shot is a 'dropshot' "
                             "(default 4); calibrate against the PNGs")
    parser.add_argument("--pace-look", type=int, default=6, dest="pace_look",
                        help="frames after contact used to measure the outgoing "
                             "ball pace (default 6)")
    parser.add_argument("--despike-px", type=float, default=40.0,
                        dest="despike_px",
                        help="absolute floor (px) for removing single-frame "
                             "ball-position outliers before smoothing "
                             "(default 40)")
    parser.add_argument("--despike-mad", type=float, default=4.0,
                        dest="despike_mad",
                        help="adaptive multiple of the local MAD for the same "
                             "outlier removal; a point is dropped only if it "
                             "exceeds BOTH --despike-px and this (default 4)")
    parser.add_argument("--output", default="outputs/shot_analysis")
    parser.add_argument("--no-frames", action="store_true",
                        help="do not save the shot PNGs")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="validate the logic on a synthetic trajectory")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not os.path.exists(args.ball):
        raise SystemExit(
            f"Ball CSV not found: {args.ball}\n"
            "Generate it first with the ball tracker (needs the YOLO model):\n"
            "    python ballTracking/BallTracking.py")

    hands = {1: args.p1_hand, 2: args.p2_hand}
    track = load_ball_track(args.ball, despike_px=args.despike_px,
                            despike_mad=args.despike_mad)
    boxes = load_player_boxes(args.players)
    conv = CourtConverter(args.court)

    shots = analyze_shots(track, boxes, conv, args.fps, hands,
                          min_gap_s=args.min_gap, vy_min=args.vy_min,
                          acc_thr=args.acc_thr, win=args.win,
                          reversal_look=args.reversal_look,
                          reversal_vy_frac=args.reversal_vy_frac,
                          drive_thr=args.drive_thr,
                          dropshot_thr=args.dropshot_thr,
                          pace_look=args.pace_look)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "shots.csv"
    shots.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(shots)} shots)")

    if not args.no_frames and not shots.empty:
        save_shot_frames(shots, args.video, boxes, out_dir)

    print_summary(shots, hands)


if __name__ == "__main__":
    main()
