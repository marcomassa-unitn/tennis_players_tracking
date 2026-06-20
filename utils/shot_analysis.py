"""
utils/shot_analysis.py

Detects the moment of the shots and classifies forehand/backhand starting from
the ball CSV produced by tracking/BallTracking.py (frame,x,y,w,h,cx,cy,area)
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

Forehand / backhand at the shot frame:
  - the player's court side (near/far) from his feet point projected in
    meters: near = we see him from behind, his right is the image right;
    far = we see him from the front, his right is the image LEFT;
  - forehand if the ball is on the dominant-hand side, backhand otherwise;
    for left-handers (--p1-hand/--p2-hand left) the reasoning is inverted;
  - if the ball is almost on the body axis the shot is marked "unknown".

Shot TYPE (flat / slice / dropshot / lob), orthogonal to forehand/backhand,
from the OUTGOING ball trajectory shape (the frames right after contact):
  - the angled camera makes a NEAR-player pixel speed ~3x a FAR-player one for
    the same physical shot, so raw pixel speed is never compared across sides.
    Perspective is handled by (a) a scale-free "pace" = pixel speed / ball
    bbox-height, (b) court-meters speed via the homography for the far split,
    and (c) dimensionless trajectory-SHAPE features (diefrac = how fast the ball
    stops travelling, reach, and bowback = how much the ball arcs back);
  - dropshot = the far ball dies quickly (low diefrac + low late speed); lob =
    a near ball that arcs up and returns (tiny diefrac, slow, short reach);
    slice = a floated/decelerating ball (bowback, low pace); else flat;
  - CAVEAT: the court homography is a GROUND plane while the ball is airborne at
    contact, so the meters-based speeds are approximate — every shot-type
    threshold is a CLI flag (see --type-self-test) and may need per-camera
    retuning; the scale-free features (pace/diefrac/reach/bowback) do not.

Use (from project root, after generating the ball CSV with BallTracking):
    python utils/shot_analysis.py --ball outputs/ball_coordinates/ball_Input_video2.csv
    python utils/shot_analysis.py --ball outputs/ball_coordinates/ball_Input_video2.csv \\
        --p1-hand right --p2-hand left

Validation of the logic without the YOLO model (synthetic trajectory):
    python utils/shot_analysis.py --self-test

Validation of the shot-type classifier against the labelled ground truth
(needs the real ball_Input_video2.csv + shots.csv):
    python utils/shot_analysis.py --type-self-test
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

def load_ball_track(ball_csv: str) -> pd.DataFrame:
    """
    Load the ball CSV, realign it over all frames and smooth cx/cy.
    Return a DataFrame indexed by frame with columns cx, cy (smoothed) and a
    boolean `interp` column, NaN cx/cy where the ball was never seen.

    `interp` marks frames that are interpolated straight-line FILLS rather than
    real ball detections, so detect_hits can ignore the frozen-BB seam where a
    long fill meets real motion (which otherwise looks like a contact). It comes
    from the producer's `interpolated` column (BallTracking) when present; on an
    older CSV that lacks the column it is all-False (the seam guard is then a
    no-op — a geometric reconstruction is unreliable after the smoothing below,
    so we deliberately do not attempt it).
    """
    df = pd.read_csv(ball_csv)
    if df.empty:
        raise ValueError(f"Empty ball CSV: {ball_csv}")
    idx = range(int(df["frame"].min()), int(df["frame"].max()) + 1)
    # `h` (ball bbox height) is carried alongside cx/cy as the SCALE proxy used by
    # the shot-type classifier: the ball looks bigger near the camera, so dividing
    # a pixel speed by the local median(h) yields a perspective-robust ("scale-free")
    # pace. h is interpolated the same way as cx/cy but is NOT Savitzky-Golay
    # smoothed below (only a local median of it is ever used). Older CSVs without an
    # `h` column degrade gracefully: the column is all-NaN and the meters/shape
    # features (which don't need h) still work; pace/reach become NaN -> unknown.
    cols = ["cx", "cy"] + (["h"] if "h" in df.columns else [])
    full = df.set_index("frame")[cols].reindex(idx)
    full = full.interpolate(limit=8, limit_area="inside")
    if "h" not in full.columns:
        full["h"] = np.nan

    if "interpolated" in df.columns:
        interp = (df.set_index("frame")["interpolated"]
                  .reindex(idx).fillna(0).to_numpy().astype(bool))
    else:
        interp = np.zeros(len(list(idx)), dtype=bool)

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
    full["interp"] = interp
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

    vy_min  : minimum magnitude (px/frame) of the mean velocity before/after for
              a sign reversal to count as a candidate
    acc_thr : threshold (px/frame^2) on the peaks of the acceleration magnitude
    win     : half-window (frames) used for the before/after velocity means and
              the local acceleration-peak test
    reversal_look    : forward/backward half-window (frames) used ONLY to confirm
              an acceleration-only candidate (acc_peak True, flip False): the
              outgoing vy must keep the opposite sign to the incoming vy over this
              window. Tolerates a short ball dropout right after contact via
              nanmean + a minimum valid-sample count. NOT applied to candidates
              that already pass `flip`. Longer than `win` so it checks the
              sustained outgoing direction.
    reversal_vy_frac : relaxed magnitude floor for that confirmation, as a
              fraction of vy_min (effective floor = reversal_vy_frac * vy_min),
              applied to the OUTGOING (forward) mean only. Kept below vy_min so
              the confirmation is genuinely looser than `flip` and does not
              collapse the acceptance rule back to `flip`.
    """
    frames = track.index.values
    cx = track["cx"].values
    cy = track["cy"].values
    valid = ~(np.isnan(cx) | np.isnan(cy))

    # Differentiate over contiguous valid runs only, so a 1-frame ball dropout
    # does not contaminate the velocity/acceleration of its neighbours and hide
    # a shot next to the gap (see _gradient_runs).
    vx = _gradient_runs(cx, valid)
    vy = _gradient_runs(cy, valid)
    valid_v = ~(np.isnan(vx) | np.isnan(vy))
    acc = np.hypot(_gradient_runs(vx, valid_v), _gradient_runs(vy, valid_v))

    # Independent post-contact reversal confirmation, used ONLY to gate
    # acceleration-only candidates (acc_peak True, flip False). Deliberately
    # LOOSER than `flip`: a relaxed magnitude floor (reversal_vy_frac * vy_min)
    # applied to the outgoing mean only, over a longer look-ahead than `win`,
    # and tolerant of a short ball dropout right after contact (nanmean over the
    # window, requiring at least `min_valid` real samples on each side). If this
    # were identical to `flip`, `flip or (acc_peak and reversal)` would collapse
    # to `flip` and silently drop every acc-only candidate.
    rev_floor = reversal_vy_frac * vy_min
    min_valid = 2   # minimum real vy samples required in each window

    def _persistent_reversal(i):
        lo = max(0, i - reversal_look)
        hi = min(len(vy), i + 1 + reversal_look)
        if np.count_nonzero(valid_v[lo:i]) < min_valid:
            return False
        if np.count_nonzero(valid_v[i + 1:hi]) < min_valid:
            return False
        b = np.nanmean(vy[lo:i])          # incoming, excludes contact frame i
        a = np.nanmean(vy[i + 1:hi])      # outgoing, excludes contact frame i
        if np.isnan(b) or np.isnan(a):
            return False
        # genuine, sustained sign flip; magnitude floor on the OUTGOING side only
        return (b * a < 0) and (abs(a) >= rev_floor)

    candidates = {}   # idx -> strength
    for i in range(win, len(frames) - win):
        if not valid[i - win:i + win + 1].all():
            continue
        before = np.nanmean(vy[i - win:i])
        after = np.nanmean(vy[i + 1:i + 1 + win])
        flip = (before * after < 0
                and min(abs(before), abs(after)) >= vy_min)
        acc_peak = (acc[i] >= acc_thr
                    and acc[i] == np.nanmax(acc[i - win:i + win + 1]))
        # acc-only candidates (acc_peak but not flip) now require an independent,
        # relaxed post-contact reversal confirmation. flip candidates unchanged.
        if flip or (acc_peak and _persistent_reversal(i)):
            candidates[i] = max(candidates.get(i, 0.0), float(acc[i]))

    # player proximity filter
    near_player = {}
    for i, strength in candidates.items():
        f = int(frames[i])
        ball_xy = (cx[i], cy[i])
        best = None
        for pid in (1, 2):
            box = _nearest_box(boxes, f, pid)
            if box is not None and _ball_near_player(ball_xy, box):
                bx_c = box[0] + box[2] / 2.0
                by_c = box[1] + box[3] / 2.0
                d = np.hypot(ball_xy[0] - bx_c, ball_xy[1] - by_c)
                if best is None or d < best[1]:
                    best = (pid, d)
        if best is not None:
            near_player[i] = (best[0], strength)

    # Same-player WIDE-gap clustering: one shot per physical contact.
    #
    # A single contact produces TWO reversals a few frames apart (an early
    # approach-side reversal as the ball nears the player, plus the true contact
    # reversal), and the old narrow min_gap (~15 frames) let those survive as TWO
    # shots whenever they were 10-29 frames apart (the observed double spacing).
    # Distinct same-player rally shots are >= ~64 frames apart on real footage, so
    # a wider merge_gap collapses each double into one while never merging two
    # genuine shots. Opposite-player candidates are NEVER merged (the rally
    # alternates), so the player id is part of the cluster key.
    merge_gap = max(int(min_gap_s * fps), 30)
    interp = (track["interp"].to_numpy().astype(bool)
              if "interp" in track.columns else np.zeros(len(frames), bool))

    def _fill_run(i, step):
        # Length of the consecutive interpolated-fill run adjacent to index i.
        c, k = 0, i + step
        while 0 <= k < len(interp) and interp[k]:
            c += 1
            k += step
        return c

    # Build same-player clusters (frame order), then keep the STRONGEST member of
    # each (peak acceleration = the most physical contact).
    clusters = []   # each: list of array-indices i
    for i in sorted(near_player):
        pid = near_player[i][0]
        f = int(frames[i])
        if (clusters
                and near_player[clusters[-1][-1]][0] == pid
                and f - int(frames[clusters[-1][-1]]) <= merge_gap):
            clusters[-1].append(i)
        else:
            clusters.append([i])

    hits = []
    for cl in clusters:
        best = max(cl, key=lambda j: near_player[j][1])   # strongest member
        # Frozen-BB seam guard: a candidate sitting at the end of a long
        # interpolated straight fill (incoming fill >= 10 frames) with no real
        # ball arriving after it (outgoing fill == 0) is the interpolation kink,
        # not a contact (the frame-49 false positive). No-op when `interp` is
        # all-False (legacy CSV without the producer `interpolated` column).
        if _fill_run(best, -1) >= 10 and _fill_run(best, +1) == 0:
            continue
        f = int(frames[best])
        pid, strength = near_player[best]
        hits.append((f, pid, strength))
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


# ── shot-type (flat / slice / dropshot / lob) classification ────────────────────
#
# Orthogonal to forehand/backhand: classifies HOW the ball was struck from the
# shape of its OUTGOING trajectory (the frames right after contact).
#
# The footage is shot from an angled camera, so a raw pixel speed is ~3x larger
# for the NEAR player than the FAR player for the same physical shot. We therefore
# never compare a raw pixel speed across sides; perspective is handled three ways:
#   * scale-free `pace`   = 100 * mean(px step speed) / median(ball bbox h); the
#                           ball's apparent size h shrinks with distance exactly as
#                           its apparent speed does, so the ratio is scale-free.
#   * meters `peak_m`/`tail_m` via the court homography (× fps) — physical units,
#     used only for the FAR flat/slice split. CAVEAT: the homography is a GROUND
#     plane but the ball is airborne at contact, so these m/s are approximate —
#     hence every threshold is exposed as a CLI flag for per-camera re-derivation.
#   * dimensionless SHAPE: `diefrac` (2nd-half / 1st-half outgoing path length — how
#     fast the ball stops travelling), `reach` (path length / h), and `bb`
#     (bowback: fraction of the vertical range the ball travels back after an
#     interior vertical extreme — ~0 for a flat drive, high for a lofted arc).
#
# Thresholds were fit against a 23-shot ground truth and sit on wide plateaus
# (see --type-self-test, which reproduces 23/23 and is stable for K in 22..30).

# Default thresholds (also the CLI-flag defaults in main()). Plateau centers.
SHOT_TYPE_PARAMS = {
    "k": 26,                 # outgoing kinematics window (frames after contact)
    "w30": 30,               # outgoing window for the path-length ratios
    "drop_diefrac": 0.50,    # FAR dropshot: 2nd/1st half path-length ratio below this
    "drop_tail_m": 10.0,     # FAR dropshot: late-window speed (m/s) below this
    "lob_diefrac": 0.25,     # NEAR lob: ball arcs up then returns -> tiny diefrac
    "lob_pace": 95.0,        # NEAR lob: scale-free pace below this
    "lob_reach": 25.0,       # NEAR lob: short scale-free penetration
    "nslice_bb": 0.40,       # NEAR slice: bowback at/above this (floated, not driven)
    "nslice_diefrac": 0.70,  # NEAR slice: ball loses pace (separates slice from flat)
    "nslice_peak_m": 39.0,   # NEAR slice: peak speed (m/s) below this (corroborator)
    "far_peak_m": 37.0,      # FAR flat if peak speed (m/s) at/above this, else slice
}


def _pathlen(cx: np.ndarray, cy: np.ndarray) -> float:
    """Total polyline length of an (cx, cy) trajectory, ignoring NaN steps."""
    return float(np.nansum(np.hypot(np.diff(cx), np.diff(cy))))


def _bowback(cy: np.ndarray) -> float:
    """
    Bowback: after the ball's first INTERIOR vertical extreme in the outgoing
    window, the fraction of the total vertical range it travels back. ~0 for a
    monotone drive (the ball keeps going one way); -> 1 for a lofted/floated arc
    (it rises to an apex then comes back down, or vice-versa). Scale-free.
    """
    n = len(cy)
    if n < 6:
        return 0.0
    rng = float(np.nanmax(cy) - np.nanmin(cy))
    if rng < 20:          # essentially flat in y: no meaningful arc
        return 0.0
    imin = int(np.nanargmin(cy))
    imax = int(np.nanargmax(cy))
    # interior extreme only (1 <= idx <= n-3): an extreme at the very edge is the
    # window boundary, not a real turn-around of the trajectory.
    up_down = (np.nanmax(cy[imin:]) - cy[imin]) / rng if 1 <= imin <= n - 3 else 0.0
    down_up = (cy[imax] - np.nanmin(cy[imax:])) / rng if 1 <= imax <= n - 3 else 0.0
    return float(max(up_down, down_up))


def _shot_type_features(track, conv, i: int, fps: float,
                        k: int, w30: int) -> dict:
    """
    Outgoing-trajectory features for the contact at integer array-index ``i`` into
    the (already-smoothed) ball track. Returns a dict of pace, peak_m, tail_m,
    apex, rise, bb, diefrac, reach, vy_end (any may be NaN on a missing window).
    """
    cx = track["cx"].values
    cy = track["cy"].values
    h = track["h"].values if "h" in track.columns else np.full(len(cx), np.nan)

    def _mean(a):
        a = a[~np.isnan(a)]
        return float(np.mean(a)) if len(a) else np.nan

    cyo = cy[i:i + k + 1]
    cxo = cx[i:i + k + 1]
    hc = np.nanmedian(h[max(0, i - 3):i + 4])    # local scale proxy

    # scale-free pixel pace
    spx = np.hypot(np.diff(cxo), np.diff(cyo))
    pace = 100.0 * _mean(spx) / hc if hc and not np.isnan(hc) else np.nan

    # perspective-corrected speed in m/s (ground-plane homography; approximate)
    pm = conv.to_meters_batch(np.column_stack([cxo, cyo]))
    dm = np.diff(pm, axis=0)
    spm = np.hypot(dm[:, 0], dm[:, 1]) * fps
    peak_m = float(np.nanmax(spm)) if np.any(~np.isnan(spm)) else np.nan
    tail_m = _mean(spm[18:26])                   # late-window speed

    apex = float(cyo[0] - np.nanmin(cyo)) if np.any(~np.isnan(cyo)) else np.nan
    rise = _mean(np.diff(cyo[:6]))               # initial vertical velocity sign
    bb = _bowback(cyo)

    # half-vs-half path-length ratio over the (slightly longer) W30 window
    cyo30 = cy[i:i + w30 + 1]
    cxo30 = cx[i:i + w30 + 1]
    half = w30 // 2
    p1 = _pathlen(cxo30[:half + 1], cyo30[:half + 1])
    p2 = _pathlen(cxo30[half:], cyo30[half:])
    diefrac = p2 / p1 if p1 > 1e-6 else np.nan
    reach = _pathlen(cxo30, cyo30) / hc if hc and not np.isnan(hc) else np.nan
    vy_end = abs(_mean(np.diff(cyo[k - 5:k + 1])))

    return dict(pace=pace, peak_m=peak_m, tail_m=tail_m, apex=apex, rise=rise,
                bb=bb, diefrac=diefrac, reach=reach, vy_end=vy_end)


def classify_shot_type(feats: dict, side: str, params: dict = None) -> str:
    """
    flat | slice | dropshot | lob | unknown, from the outgoing-trajectory features
    and the striking player's court side. First matching rule wins. Returns
    "unknown" when the discriminating features could not be computed (missing
    outgoing window) so a bad shot never silently masquerades as a confident type.
    """
    p = params or SHOT_TYPE_PARAMS
    df = feats.get("diefrac")
    peak_m = feats.get("peak_m")

    # The shape signals (diefrac for dropshot/lob, peak_m for the far split) are
    # the load-bearing discriminators; if they're NaN we can't classify safely.
    if df is None or np.isnan(df):
        return "unknown"

    # R1 DROPSHOT (far): the ball's path collapses in the 2nd half AND the late
    # speed is low. diefrac is scale-free and immune to where the apex falls; the
    # tail_m AND-gate protects a fast far-flat whose diefrac happens to be middling.
    tail_m = feats.get("tail_m")
    if (side == "far" and df < p["drop_diefrac"]
            and tail_m is not None and not np.isnan(tail_m)
            and tail_m < p["drop_tail_m"]):
        return "dropshot"

    if side == "near":
        # R2a LOB: arcs up then returns -> tiny diefrac; triple-confirmed by a slow
        # pace and a short reach (lob is rare, so it is gated three independent ways).
        pace = feats.get("pace")
        reach = feats.get("reach")
        if (df < p["lob_diefrac"]
                and pace is not None and not np.isnan(pace) and pace < p["lob_pace"]
                and reach is not None and not np.isnan(reach)
                and reach < p["lob_reach"]):
            return "lob"
        # R2b NEAR SLICE: bows back, loses pace, and is not a fast drive.
        bb = feats.get("bb", 0.0)
        if (bb >= p["nslice_bb"] and df < p["nslice_diefrac"]
                and peak_m is not None and not np.isnan(peak_m)
                and peak_m < p["nslice_peak_m"]):
            return "slice"
        # R2c default
        return "flat"

    # R3 FAR branch: a fast drive is flat, a floated ball is a slice.
    if peak_m is None or np.isnan(peak_m):
        return "unknown"
    return "flat" if peak_m >= p["far_peak_m"] else "slice"


# ── overhead (serve / smash) detection ──────────────────────────────────────────
#
# ORTHOGONAL and ADDITIVE to shot_type: serve/smash go in a SEPARATE `overhead`
# column and NEVER overwrite shot_type (flat/slice/dropshot/lob/unknown). The
# classifier is deliberately CONSERVATIVE — its first priority is ZERO false
# positives on groundstrokes/lobs, so when anything is uncertain it returns "".
#
# Geometry recap (image y grows DOWNWARD; court meters via the ground homography):
#   * an overhead contact has the ball clearly ABOVE the player's head
#     (ball_cy < box_top - margin*h) and roughly over the body (x-aligned). This
#     gate is geometrically rare for groundstrokes (ball ~ waist/shoulder, off to
#     the side), so it is what protects the existing shot types.
#   * a SERVE is the first shot of the clip, the server stationary behind his own
#     baseline, the ball tossed up locally (slow, near-side incoming).
#   * a SMASH is hit off an opponent's ball that came fast from across the net and
#     is then DRIVEN downward (outgoing arcs down/fast, unlike a lob which arcs up).

OVERHEAD_PARAMS = {
    # Common overhead gate
    "above_head_frac": 0.25,   # ball must be above box_top by this fraction of box h
    "x_align_frac": 0.60,      # |ball_cx - player_cx| must be below this fraction of box w
    # Serve gate
    "stationary_mps": 1.0,     # server feet speed (m/s) must be below this (~standing)
    "behind_baseline_m": 1.5,  # server feet within this many m BEHIND his own baseline
    "serve_incoming_slow_mps": 6.0,  # incoming ("toss") ball speed (m/s) must be below this
    "serve_toss_local_m": 6.0,       # pre-contact ball must be on the server's OWN side of the
                                     # net, within this margin past the net line (a toss stays on
                                     # the server's half; a returned/across-net ball does not).
                                     # NB the ground homography distorts the airborne toss, so this
                                     # is a generous same-half check, not a tight distance.
    # Smash gate
    "smash_incoming_fast_mps": 9.0,  # incoming ball speed (m/s) must be at/above this
    "smash_down_rise": 0.5,    # outgoing must descend: mean initial vertical step (px/frame) >= this
                               # (image y grows down, so a positive `rise` means the ball goes DOWN)
    "smash_min_apex": 6.0,     # outgoing apex (px the ball rises above contact) must be SMALL
                               # (a driven smash barely rises; a lob's apex is large) — upper bound
}


def _box_feet(box):
    """Feet (mid-bottom) pixel point of a player box (x, y, w, h)."""
    x, y, w, h = box
    return (x + w / 2.0, y + h)


def _player_speed_mps(boxes, conv, f, pid, fps, dt=3):
    """
    Player feet speed in m/s around frame ``f`` for player ``pid``: take the feet
    point at f-dt and f+dt (via _nearest_box, robust to a few absent frames),
    project both through the court homography to meters and divide the travelled
    distance by the elapsed time. Returns None if either box is missing or the
    projection is not finite.
    """
    b0 = _nearest_box(boxes, f - dt, pid)
    b1 = _nearest_box(boxes, f + dt, pid)
    if b0 is None or b1 is None:
        return None
    try:
        x0, y0 = conv.to_meters(*_box_feet(b0))
        x1, y1 = conv.to_meters(*_box_feet(b1))
    except Exception:
        return None
    if not (np.isfinite(x0) and np.isfinite(y0)
            and np.isfinite(x1) and np.isfinite(y1)):
        return None
    dist = float(np.hypot(x1 - x0, y1 - y0))
    elapsed = (2 * dt) / float(fps)
    if elapsed <= 0:
        return None
    return dist / elapsed


def _incoming_ball_features(track, conv, i, fps, win=10):
    """
    Features of the ball BEFORE contact at array-index ``i`` (the window
    [i-win:i]) — the incoming mirror of the OUTGOING _shot_type_features. Used to
    tell a serve (slow, locally-tossed ball) from a smash (fast ball arriving from
    across the net).

    Returns a dict:
      in_speed_m : incoming ball speed in m/s (median of per-step homography
                   speeds over the window), NaN if the pre-contact track is too
                   short/holey.
      in_y_m     : the ball's court y (meters) ~win frames before contact, used to
                   place the ball's origin (own side vs across the net). NaN if
                   unavailable.
      in_x_m     : the ball's court x (meters) ~win frames before contact. NaN if
                   unavailable.
    """
    out = dict(in_speed_m=np.nan, in_y_m=np.nan, in_x_m=np.nan)
    lo = max(0, i - win)
    if i - lo < 3:
        return out
    cx = track["cx"].values[lo:i + 1]
    cy = track["cy"].values[lo:i + 1]
    pm = conv.to_meters_batch(np.column_stack([cx, cy]))
    dm = np.diff(pm, axis=0)
    spm = np.hypot(dm[:, 0], dm[:, 1]) * fps
    spm = spm[~np.isnan(spm)]
    if len(spm):
        out["in_speed_m"] = float(np.nanmedian(spm))
    # earliest valid pre-contact ball court position (origin of the incoming ball)
    for k in range(len(pm)):
        if np.all(np.isfinite(pm[k])):
            out["in_x_m"] = float(pm[k, 0])
            out["in_y_m"] = float(pm[k, 1])
            break
    return out


def classify_overhead(box, ball_cx, ball_cy, player_x_m, player_y_m, side,
                      stroke, in_feats, out_feats, player_speed,
                      is_first_shot, params=None):
    """
    Classify an overhead shot as "serve", "smash" or "" (none). CONSERVATIVE:
    returns "" the moment any required signal is missing or out of range, so a
    groundstroke/lob is never mislabelled. serve and smash are mutually exclusive.

    box                : player bbox (x, y, w, h) px at contact
    ball_cx, ball_cy   : ball pixel position at contact (image y grows downward)
    player_x_m,_y_m    : player feet court position (meters)
    side               : "near" | "far"
    stroke             : forehand/backhand/unknown (corroborates a serve, not required)
    in_feats           : dict from _incoming_ball_features
    out_feats          : dict from _shot_type_features (outgoing shape)
    player_speed       : player feet speed (m/s) from _player_speed_mps, or None
    is_first_shot      : True iff this is the first detected shot of the clip
    """
    p = params or OVERHEAD_PARAMS
    if box is None:
        return ""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return ""
    player_cx = x + w / 2.0
    box_top = y

    # ── Common overhead gate: ball clearly above head AND horizontally aligned ──
    # (image y grows downward, so "above" means a SMALLER y than box_top).
    if not (ball_cy < box_top - p["above_head_frac"] * h):
        return ""
    if not (abs(ball_cx - player_cx) < p["x_align_frac"] * w):
        return ""

    in_speed = in_feats.get("in_speed_m") if in_feats else None
    in_y = in_feats.get("in_y_m") if in_feats else None

    # ── SERVE: first shot, stationary, behind own baseline, slow local toss ──
    if is_first_shot:
        # behind own baseline: near baseline at y_m = L_m, far baseline at y_m = 0
        if side == "near":
            behind = player_y_m >= L_m - p["behind_baseline_m"]
        else:
            behind = player_y_m <= 0.0 + p["behind_baseline_m"]
        stationary = (player_speed is not None
                      and player_speed < p["stationary_mps"])
        # the toss is slow and originates on the server's OWN side of the net, not
        # from across it (a smash's incoming ball comes from the opponent's half).
        slow = (in_speed is not None and not np.isnan(in_speed)
                and in_speed < p["serve_incoming_slow_mps"])
        if in_y is not None and not np.isnan(in_y):
            if side == "near":   # near half is y_m > NET
                local = in_y > NET - p["serve_toss_local_m"]
            else:                # far half is y_m < NET
                local = in_y < NET + p["serve_toss_local_m"]
        else:
            local = False
        if behind and stationary and slow and local:
            return "serve"

    # ── SMASH: opponent ball arrives fast, driven DOWNWARD on the way out ──
    # (Must NOT be a serve case: a smash is generally not the first shot and/or
    # the player is not parked behind his own baseline.)
    behind_own = False
    if side == "near":
        behind_own = player_y_m >= L_m - p["behind_baseline_m"]
    else:
        behind_own = player_y_m <= 0.0 + p["behind_baseline_m"]
    if is_first_shot and behind_own:
        return ""   # looks like a serve setup, not a smash — stay conservative

    rise = out_feats.get("rise") if out_feats else None     # +ve = ball goes DOWN
    apex = out_feats.get("apex") if out_feats else None      # px the ball rose above contact
    fast_incoming = (in_speed is not None and not np.isnan(in_speed)
                     and in_speed >= p["smash_incoming_fast_mps"])
    downward = (rise is not None and not np.isnan(rise)
                and rise >= p["smash_down_rise"])
    low_apex = (apex is not None and not np.isnan(apex)
                and apex <= p["smash_min_apex"])
    if fast_incoming and downward and low_apex:
        return "smash"

    return ""


def analyze_shots(track, boxes, conv, fps, hands, min_gap_s=0.5,
                  vy_min=0.5, acc_thr=1.5, win=4,
                  reversal_look=6, reversal_vy_frac=0.5,
                  type_params=None):
    """Full pipeline: detect shots and classify them. Return a DataFrame."""
    tp = type_params or SHOT_TYPE_PARAMS
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
    first_frame = hits[0][0] if hits else None
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
        # Player court position (feet) in meters — used by the shot hitmap to
        # place the marker where the PLAYER hit the ball (not where the ball was).
        player_x_m, player_y_m = conv.to_meters(*feet)
        y_m = player_y_m
        side = "near" if y_m > NET else "far"
        stroke = classify_stroke(ball_cx, box, side, hands[pid])

        # Shot TYPE (flat/slice/dropshot/lob) from the OUTGOING trajectory shape.
        # The contact frame `f` maps to the integer array-index `i` into the
        # smoothed track via the (contiguous) frame index.
        i = int(f - track.index.min())
        feats = _shot_type_features(track, conv, i, fps, tp["k"], tp["w30"])
        shot_type = classify_shot_type(feats, side, tp)

        # Overhead (serve/smash) — ADDITIVE, separate column, never overwrites
        # shot_type. Wrapped so missing/short data yields "" and never raises.
        try:
            in_feats = _incoming_ball_features(track, conv, i, fps)
            player_speed = _player_speed_mps(boxes, conv, f, pid, fps)
            overhead = classify_overhead(
                box, ball_cx, ball_cy, player_x_m, player_y_m, side,
                stroke, in_feats, feats, player_speed,
                is_first_shot=(first_frame is not None and f == first_frame))
        except Exception:
            overhead = ""

        # NOTE: to_meters uses the ground-plane (court) homography, but at the
        # moment of contact the ball is airborne (~ racket height). Projecting
        # an above-ground point through a ground homography is only approximate
        # — treat ball_x_m / ball_y_m as indicative, not exact, court coords.
        bx_m, by_m = conv.to_meters(ball_cx, ball_cy)
        rows.append({
            "frame": f, "time_s": round(f / fps, 2), "player_id": pid,
            "side": side, "hand": hands[pid], "stroke": stroke,
            "shot_type": shot_type,
            "overhead": overhead,
            "ball_pace": (round(feats["pace"], 1)
                          if feats["pace"] is not None
                          and not np.isnan(feats["pace"]) else None),
            "ball_cx": round(ball_cx, 1), "ball_cy": round(ball_cy, 1),
            "player_cx": round(box[0] + box[2] / 2.0, 1),
            "player_x_m": round(player_x_m, 2),
            "player_y_m": round(player_y_m, 2),
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
        # shot_type may be absent on a legacy DataFrame; guard with getattr.
        shot_type = getattr(r, "shot_type", "")
        label = (f"P{r.player_id} {r.stroke} {shot_type} (frame {r.frame})"
                 if shot_type else f"P{r.player_id} {r.stroke} (frame {r.frame})")
        cv2.putText(fr, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 3)
        type_tag = f"_{shot_type}" if shot_type else ""
        path = (out_dir /
                f"shot_f{r.frame:05d}_P{r.player_id}_{r.stroke}{type_tag}.png")
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
    has_type = "shot_type" in shots.columns
    for r in shots.itertuples():
        type_str = f"  [{r.shot_type}]" if has_type else ""
        print(f"  frame {r.frame:5d} ({r.time_s:6.2f}s)  "
              f"P{r.player_id} ({r.side:4s}, {r.hand:5s})  ->  {r.stroke}{type_str}")
    print()
    for pid, sub in shots.groupby("player_id"):
        n_fh = (sub["stroke"] == "forehand").sum()
        n_bh = (sub["stroke"] == "backhand").sum()
        n_uk = (sub["stroke"] == "unknown").sum()
        print(f"  P{pid} ({hands[pid]:5s}): {len(sub)} shots  "
              f"-  forehands {n_fh}, backhands {n_bh}, unknown {n_uk}")
    if has_type:
        print()
        for t in ("flat", "slice", "dropshot", "lob", "unknown"):
            n = (shots["shot_type"] == t).sum()
            if n:
                print(f"  shot type {t:9s}: {n}")
    if "overhead" in shots.columns:
        print()
        for t in ("serve", "smash"):
            n = (shots["overhead"] == t).sum()
            if n:
                print(f"  overhead  {t:9s}: {n}")
    print()


# ── shot hitmap (player positions on the minimap, coloured by shot type) ────────
#
# "Combined category": one colour-label per shot. A special shot_type
# (slice/dropshot/lob/serve) wins; otherwise we fall back to the forehand/backhand
# stroke. `serve` is not produced by the classifier yet but is already in the
# palette so it is coloured correctly as soon as it is added. The marker SHAPE
# encodes the player (P1 = circle, P2 = triangle).
SHOT_CATEGORY_COLORS = {
    "forehand": "#2ca02c",   # green
    "backhand": "#ff7f0e",   # orange
    "slice":    "#9467bd",   # purple
    "dropshot": "#d62728",   # red
    "lob":      "#8c564b",   # brown
    "serve":    "#17becf",   # cyan
    "smash":    "#d4af37",   # gold
    "unknown":  "#7f7f7f",   # grey
}
_CATEGORY_FALLBACK_COLOR = "#111111"
_SPECIAL_SHOT_TYPES = ("slice", "dropshot", "lob", "serve")
_PLAYER_MARKERS = {1: "o", 2: "^"}
_PLAYER_MARKER_FALLBACK = "s"


def shot_category(stroke, shot_type, overhead=""):
    """Single colour-category for a shot. A non-empty `overhead` (serve/smash)
    WINS over everything; else a special shot_type (slice/dropshot/lob/serve)
    wins; otherwise the forehand/backhand stroke."""
    ov = (str(overhead) if overhead is not None else "").strip().lower()
    if ov in ("serve", "smash"):
        return ov
    st = (str(shot_type) if shot_type is not None else "").strip().lower()
    if st in _SPECIAL_SHOT_TYPES:
        return st
    sk = (str(stroke) if stroke is not None else "").strip().lower()
    return sk if sk else "unknown"


def save_shot_hitmap(shots: pd.DataFrame, out_dir,
                     title: str = "Shot hitmap (player position)") -> None:
    """Plot every shot at the PLAYER's court position (feet, in meters) on the
    minimap, coloured by combined shot category, with the marker shape encoding
    the player. Reuses the court drawing from player_analysis to avoid
    duplicating the court geometry."""
    out_dir = Path(out_dir)
    if shots is None or shots.empty:
        print("  Hitmap skipped: no shots.")
        return
    if "player_x_m" not in shots.columns or "player_y_m" not in shots.columns:
        print("  Hitmap skipped: shots.csv has no player_x_m/player_y_m columns "
              "(regenerate it with the current shot_analysis).")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from utils.player_analysis import _draw_court

    fig, ax = plt.subplots(figsize=(5.2, 9.0))
    # Same look as heatmap_combined.png: dark background so the white court
    # lines and the yellow net stand out clearly. _draw_court() sets a green
    # facecolor and turns the axis OFF (which hides the patch, leaving white
    # lines invisible on a white page); we re-enable a dark patch so the field
    # reads as the dark court used by the heatmap figures.
    fig.patch.set_facecolor("#1a1a2e")
    _draw_court(ax)
    ax.set_facecolor("#1a1a2e")
    ax.patch.set_visible(True)

    present_cats, present_players = [], []
    for r in shots.itertuples():
        xm = getattr(r, "player_x_m", float("nan"))
        ym = getattr(r, "player_y_m", float("nan"))
        if xm is None or ym is None or np.isnan(xm) or np.isnan(ym):
            continue
        cat = shot_category(getattr(r, "stroke", None),
                            getattr(r, "shot_type", None),
                            getattr(r, "overhead", ""))
        color = SHOT_CATEGORY_COLORS.get(cat, _CATEGORY_FALLBACK_COLOR)
        pid = int(getattr(r, "player_id", 0))
        marker = _PLAYER_MARKERS.get(pid, _PLAYER_MARKER_FALLBACK)
        ax.scatter(xm, ym, c=color, marker=marker, s=150,
                   edgecolors="white", linewidths=0.9, zorder=5, alpha=0.95)
        if cat not in present_cats:
            present_cats.append(cat)
        if pid not in present_players:
            present_players.append(pid)

    if not present_cats:
        print("  Hitmap skipped: no shot has a valid player position.")
        plt.close(fig)
        return

    # Legend 1: shot category -> colour
    cat_handles = [Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=SHOT_CATEGORY_COLORS.get(
                              c, _CATEGORY_FALLBACK_COLOR),
                          markeredgecolor="black", markersize=10, label=c)
                   for c in present_cats]
    # Legend 2: player -> marker shape
    plr_handles = [Line2D([0], [0],
                          marker=_PLAYER_MARKERS.get(p, _PLAYER_MARKER_FALLBACK),
                          linestyle="", markerfacecolor="white",
                          markeredgecolor="black", markersize=10, label=f"P{p}")
                   for p in sorted(present_players)]

    leg1 = ax.legend(handles=cat_handles, title="Shot type",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, framealpha=0.6, facecolor="#333",
                     edgecolor="none", labelcolor="white")
    leg1.get_title().set_color("white")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=plr_handles, title="Player",
                     loc="lower left", bbox_to_anchor=(1.02, 0.0),
                     fontsize=8, framealpha=0.6, facecolor="#333",
                     edgecolor="none", labelcolor="white")
    leg2.get_title().set_color("white")

    ax.set_title(title, color="white")
    path = out_dir / "shot_hitmap.png"
    # bbox_extra_artists ensures the out-of-axes legends are not clipped by
    # bbox_inches="tight"; facecolor keeps the dark background in the saved PNG.
    fig.savefig(path, dpi=140, bbox_inches="tight",
                bbox_extra_artists=(leg1, leg2),
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


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
        for hands, expect_fn in (
            ({1: "right", 2: "right"}, lambda _, exp: exp),
            ({1: "right", 2: "left"},
             lambda pid, exp: exp if pid == 1 else
             ("backhand" if exp == "forehand" else "forehand")),
        ):
            shots = analyze_shots(track, boxes, conv, fps=30.0, hands=hands)
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

    print("SELF-TEST shot_analysis")
    print(f"  planned shots : {[(f, p, s) for f, p, _, s in plan]}")
    if failures:
        print("  FAIL:")
        for msg in failures:
            print("   -", msg)
        raise SystemExit(1)
    print("  PASS: 4/4 shots detected and classified correctly "
          "(including the left-handed case); mid-court bounces ignored.")


# ── labelled shot-type validation (real Input_video2 ground truth) ──────────────

# Ground truth for the 23 detected shots of Input_video2, provided by the user.
# A "dropshot/slice" label is genuinely ambiguous (slow but not dropshot-slow) and
# counts as correct for EITHER dropshot or slice.
_TYPE_GROUND_TRUTH = {
    84: "flat", 109: "flat", 152: "flat", 188: "flat", 225: "flat",
    252: "dropshot", 322: "flat", 349: "slice", 386: "lob", 432: "flat",
    469: "flat", 503: "slice", 554: "flat", 594: "flat", 629: "flat",
    660: "flat", 696: "flat", 740: "dropshot/slice", 775: "flat", 812: "flat",
    858: "dropshot/slice", 912: "slice", 950: "flat",
}
_TYPE_TARGET_ACC = 0.90


def type_self_test(ball_csv: str = None, players_csv: str = None,
                   court_csv: str = None, fps: float = 30.0) -> None:
    """
    Validate the shot-TYPE classifier against the labelled Input_video2 ground
    truth, end-to-end through the real analyze_shots pipeline (not a re-impl).

    Each ground-truth frame is matched to the nearest detected shot within ±3
    frames; extra detections (e.g. a trailing shot the user did not label) are
    ignored. Fails with a non-zero exit if accuracy < 90 %.
    """
    base = Path(__file__).resolve().parent.parent / "outputs"
    ball_csv = ball_csv or str(base / "ball_coordinates" / "ball_Input_video2.csv")
    players_csv = players_csv or str(
        base / "player_coordinates" / "players_Input_video2.csv")
    court_csv = court_csv or str(
        base / "court_coordinates" / "Input_video2_court.csv")

    for label, path in (("ball", ball_csv), ("players", players_csv),
                        ("court", court_csv)):
        if not os.path.exists(path):
            raise SystemExit(
                f"TYPE-SELF-TEST: missing {label} CSV: {path}\n"
                "  This test needs the real Input_video2 outputs "
                "(run the tracking pipeline first).")

    track = load_ball_track(ball_csv)
    boxes = load_player_boxes(players_csv)
    conv = CourtConverter(court_csv)
    # default hands (the ground truth was labelled on the default right/right run)
    shots = analyze_shots(track, boxes, conv, fps, {1: "right", 2: "right"})

    if shots.empty or "shot_type" not in shots.columns:
        raise SystemExit("TYPE-SELF-TEST: no shots / no shot_type column produced.")

    det = {int(r.frame): r.shot_type for r in shots.itertuples()}
    det_frames = sorted(det)

    def _nearest(g):
        best = None
        for d in det_frames:
            if abs(d - g) <= 3 and (best is None or abs(d - g) < abs(best - g)):
                best = d
        return best

    print("TYPE-SELF-TEST shot_analysis")
    print(f"  {'frame':>5} {'GT':>14} {'PRED':>9}  ok")
    correct = 0
    missing = 0
    rows = []
    for g in sorted(_TYPE_GROUND_TRUTH):
        gt = _TYPE_GROUND_TRUTH[g]
        d = _nearest(g)
        if d is None:
            pred = "(not detected)"
            ok = False
            missing += 1
        else:
            pred = det[d]
            ok = (pred in ("dropshot", "slice") if gt == "dropshot/slice"
                  else pred == gt)
        correct += ok
        rows.append((g, gt, pred, ok))
        print(f"  {g:>5} {gt:>14} {pred:>9}  {'Y' if ok else 'N'}")

    n = len(_TYPE_GROUND_TRUTH)
    acc = correct / n
    print(f"\n  ACCURACY: {correct}/{n} = {acc:.3f}  (target >= {_TYPE_TARGET_ACC:.2f})")
    if missing:
        print(f"  Note: {missing} ground-truth shot(s) had no detection within "
              f"±3 frames (detection drift, counted as wrong).")
    wrong = [(g, gt, pred) for g, gt, pred, ok in rows if not ok]
    if acc < _TYPE_TARGET_ACC:
        print("  FAIL: below target. Misclassified:")
        for g, gt, pred in wrong:
            print(f"   - frame {g}: expected {gt}, got {pred}")
        raise SystemExit(1)
    print("  PASS:", "all correct." if not wrong else
          f"{len(wrong)} miss(es) but above the {_TYPE_TARGET_ACC:.0%} target: "
          f"{[(g, gt, pred) for g, gt, pred in wrong]}")


# ── overhead (serve / smash) self-test (synthetic) ──────────────────────────────

def overhead_self_test() -> None:
    """
    Validate classify_overhead on four SYNTHETIC scenarios built on the same
    synthetic court as self_test():
      (a) SERVE  — near player stationary behind his baseline, ball tossed above
          the head, slow & local incoming  -> expect "serve".
      (b) SMASH  — near player mid-court, fast ball descending from across the
          net, low-apex downward outgoing  -> expect "smash".
      (c) GROUNDSTROKE — ball at waist height beside the body -> expect "".
      (d) LOB    — ball struck low and lofted up on the way out -> expect "".
    Drives the real helpers (_incoming_ball_features, _player_speed_mps) and
    classify_overhead end-to-end; asserts all four.
    """
    # Same synthetic homography as _synthetic_court_csv: meters -> pixels.
    corners_m = np.array([[0, 0], [W_m, 0], [0, L_m], [W_m, L_m]],
                         dtype=np.float32)
    corners_px = np.array([[700, 300], [1200, 300], [400, 860], [1500, 860]],
                          dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners_m, corners_px)

    def m2px(xm, ym):
        p = H @ np.array([xm, ym, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def build_track(seg, frames=80, hpx=12.0):
        """seg: list of (frame, cx, cy); linearly interpolated track DataFrame."""
        idx = np.arange(frames)
        cx = np.full(frames, np.nan)
        cy = np.full(frames, np.nan)
        for (f0, x0, y0), (f1, x1, y1) in zip(seg[:-1], seg[1:]):
            for f in range(f0, f1 + 1):
                s = (f - f0) / max(1, f1 - f0)
                cx[f] = x0 + s * (x1 - x0)
                cy[f] = y0 + s * (y1 - y0)
        return pd.DataFrame({"cx": cx, "cy": cy, "h": hpx}, index=idx)

    with tempfile.TemporaryDirectory() as tmp:
        court_csv = os.path.join(tmp, "court.csv")
        _synthetic_court_csv(court_csv)
        conv = CourtConverter(court_csv)
        fps = 30.0
        failures = []

        # ── (a) SERVE ──────────────────────────────────────────────────────────
        # Near player parked just behind his own baseline (y ~ L_m). Box drawn so
        # the head (box_top) is well below the tossed ball.
        sx_m, sy_m = W_m / 2.0, L_m - 0.3       # behind near baseline
        feet_px = m2px(sx_m, sy_m)
        bw, bh = 90.0, 200.0
        box = (feet_px[0] - bw / 2.0, feet_px[1] - bh, bw, bh)
        head_y = box[1]
        ball_cx = box[0] + bw / 2.0             # aligned over the body
        ball_cy = head_y - 0.6 * bh             # well above the head (toss apex)
        # Stationary boxes around the contact (identical -> ~0 m/s).
        boxes = {f: {1: box} for f in range(40)}
        i = 20
        # Toss: a slow, local up-then-down ball arc, all near the server side.
        # The ball drifts only slightly over the pre-contact window (a real toss
        # is slow), staying just above the head, so incoming speed is small/local.
        toss = [(i - 10, ball_cx, ball_cy + 18),
                (i, ball_cx, ball_cy),
                (i + 10, ball_cx + 2, ball_cy + 18)]
        track = build_track(toss, frames=40)
        in_feats = _incoming_ball_features(track, conv, i, fps)
        out_feats = _shot_type_features(track, conv, i, fps,
                                        SHOT_TYPE_PARAMS["k"], SHOT_TYPE_PARAMS["w30"])
        spd = _player_speed_mps(boxes, conv, i, 1, fps)
        px_m, py_m = conv.to_meters(*_box_feet(box))
        side = "near" if py_m > NET else "far"
        ov = classify_overhead(box, ball_cx, ball_cy, px_m, py_m, side,
                               "unknown", in_feats, out_feats, spd,
                               is_first_shot=True)
        if ov != "serve":
            failures.append(f"(a) SERVE: got {ov!r}, expected 'serve' "
                            f"(speed={spd}, in_speed={in_feats['in_speed_m']:.2f})")

        # ── (b) SMASH ──────────────────────────────────────────────────────────
        # Near player mid-court; ball arrives FAST from the far side and is driven
        # steeply DOWN with a small apex.
        mx_m, my_m = W_m / 2.0, NET + 2.0       # near side, mid-court
        feet_px = m2px(mx_m, my_m)
        box = (feet_px[0] - bw / 2.0, feet_px[1] - bh, bw, bh)
        head_y = box[1]
        ball_cx = box[0] + bw / 2.0
        ball_cy = head_y - 0.5 * bh             # above the head at contact
        boxes = {f: {1: box} for f in range(80)}
        i = 30
        # Incoming: ball comes from across the net (far baseline px), fast.
        far_px = m2px(W_m / 2.0, 3.0)
        # Outgoing: driven DOWN (cy increases, image-down) and forward, low apex.
        seg = [(i - 12, far_px[0], far_px[1]),
               (i, ball_cx, ball_cy),
               (i + 26, ball_cx + 40, ball_cy + 260)]
        track = build_track(seg, frames=80)
        in_feats = _incoming_ball_features(track, conv, i, fps)
        out_feats = _shot_type_features(track, conv, i, fps,
                                        SHOT_TYPE_PARAMS["k"], SHOT_TYPE_PARAMS["w30"])
        spd = _player_speed_mps(boxes, conv, i, 1, fps)
        px_m, py_m = conv.to_meters(*_box_feet(box))
        side = "near" if py_m > NET else "far"
        ov = classify_overhead(box, ball_cx, ball_cy, px_m, py_m, side,
                               "forehand", in_feats, out_feats, spd,
                               is_first_shot=False)
        if ov != "smash":
            failures.append(
                f"(b) SMASH: got {ov!r}, expected 'smash' "
                f"(in_speed={in_feats['in_speed_m']:.2f}, "
                f"rise={out_feats['rise']:.2f}, apex={out_feats['apex']:.2f})")

        # ── (c) GROUNDSTROKE ────────────────────────────────────────────────────
        # Ball at ~waist height, off to the side -> overhead gate fails -> "".
        feet_px = m2px(W_m / 2.0, NET + 4.0)
        box = (feet_px[0] - bw / 2.0, feet_px[1] - bh, bw, bh)
        ball_cx = box[0] + bw + 30              # well off to the side
        ball_cy = box[1] + 0.6 * bh             # waist height (inside the box)
        boxes = {f: {1: box} for f in range(80)}
        i = 30
        seg = [(i - 12, ball_cx - 120, ball_cy + 10),
               (i, ball_cx, ball_cy),
               (i + 26, ball_cx + 220, ball_cy - 20)]
        track = build_track(seg, frames=80)
        in_feats = _incoming_ball_features(track, conv, i, fps)
        out_feats = _shot_type_features(track, conv, i, fps,
                                        SHOT_TYPE_PARAMS["k"], SHOT_TYPE_PARAMS["w30"])
        spd = _player_speed_mps(boxes, conv, i, 1, fps)
        px_m, py_m = conv.to_meters(*_box_feet(box))
        side = "near" if py_m > NET else "far"
        ov = classify_overhead(box, ball_cx, ball_cy, px_m, py_m, side,
                               "forehand", in_feats, out_feats, spd,
                               is_first_shot=False)
        if ov != "":
            failures.append(f"(c) GROUNDSTROKE: got {ov!r}, expected '' (no overhead)")

        # ── (d) LOB ─────────────────────────────────────────────────────────────
        # Ball struck low (waist) and lofted UP on the way out -> gate fails on
        # height AND the smash 'downward' test would fail anyway -> "".
        feet_px = m2px(W_m / 2.0, NET + 4.0)
        box = (feet_px[0] - bw / 2.0, feet_px[1] - bh, bw, bh)
        ball_cx = box[0] + bw / 2.0
        ball_cy = box[1] + 0.5 * bh             # waist height, NOT above head
        boxes = {f: {1: box} for f in range(80)}
        i = 30
        # Outgoing arcs UP then down (lofted) — apex large, rise negative.
        seg = [(i - 12, ball_cx - 60, ball_cy + 30),
               (i, ball_cx, ball_cy),
               (i + 13, ball_cx + 120, ball_cy - 220),
               (i + 26, ball_cx + 240, ball_cy - 40)]
        track = build_track(seg, frames=80)
        in_feats = _incoming_ball_features(track, conv, i, fps)
        out_feats = _shot_type_features(track, conv, i, fps,
                                        SHOT_TYPE_PARAMS["k"], SHOT_TYPE_PARAMS["w30"])
        spd = _player_speed_mps(boxes, conv, i, 1, fps)
        px_m, py_m = conv.to_meters(*_box_feet(box))
        side = "near" if py_m > NET else "far"
        ov = classify_overhead(box, ball_cx, ball_cy, px_m, py_m, side,
                               "forehand", in_feats, out_feats, spd,
                               is_first_shot=False)
        if ov != "":
            failures.append(f"(d) LOB: got {ov!r}, expected '' (no overhead)")

    print("OVERHEAD-SELF-TEST shot_analysis")
    if failures:
        print("  FAIL:")
        for msg in failures:
            print("   -", msg)
        raise SystemExit(1)
    print("  PASS: 4/4 — serve and smash detected; groundstroke and lob "
          "correctly produce no overhead (zero false positives).")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shot detection and forehand/backhand classification "
                    "from the ball tracking")
    parser.add_argument("--ball", default=None,
                        help="ball CSV generated by BallTracking.py (defaults to "
                             "outputs/ball_coordinates/ball_<video name>.csv)")
    parser.add_argument("--players", default=None,
                        help="player tracking CSV (defaults to "
                             "outputs/player_coordinates/players_<video name>.csv)")
    parser.add_argument("--court", default=None,
                        help="court keypoints CSV (defaults to "
                             "outputs/court_coordinates/<video name>_court.csv)")
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
    # Shot-TYPE (flat/slice/dropshot/lob) thresholds. Defaults are SHOT_TYPE_PARAMS,
    # fit against a 23-shot ground truth (see --type-self-test) and sitting on wide
    # plateaus. The meters-based cuts (--drop-tail-m, --*-peak-m) assume the video
    # fps and an accurate court homography, so they may need per-camera retuning;
    # the scale-free cuts (diefrac, pace, reach, bb) do not.
    parser.add_argument("--k-window", type=int, default=SHOT_TYPE_PARAMS["k"],
                        dest="k", help="outgoing kinematics window in frames "
                        "(default %(default)s)")
    parser.add_argument("--w30-window", type=int, default=SHOT_TYPE_PARAMS["w30"],
                        dest="w30", help="outgoing window for path-length ratios "
                        "(default %(default)s)")
    parser.add_argument("--drop-diefrac", type=float,
                        default=SHOT_TYPE_PARAMS["drop_diefrac"], dest="drop_diefrac",
                        help="FAR dropshot: 2nd/1st-half path-length ratio below "
                        "this (default %(default)s)")
    parser.add_argument("--drop-tail-m", type=float,
                        default=SHOT_TYPE_PARAMS["drop_tail_m"], dest="drop_tail_m",
                        help="FAR dropshot: late-window speed (m/s) below this "
                        "(default %(default)s)")
    parser.add_argument("--lob-diefrac", type=float,
                        default=SHOT_TYPE_PARAMS["lob_diefrac"], dest="lob_diefrac",
                        help="NEAR lob: path-length ratio below this "
                        "(default %(default)s)")
    parser.add_argument("--lob-pace", type=float,
                        default=SHOT_TYPE_PARAMS["lob_pace"], dest="lob_pace",
                        help="NEAR lob: scale-free pace below this "
                        "(default %(default)s)")
    parser.add_argument("--lob-reach", type=float,
                        default=SHOT_TYPE_PARAMS["lob_reach"], dest="lob_reach",
                        help="NEAR lob: scale-free reach below this "
                        "(default %(default)s)")
    parser.add_argument("--nslice-bb", type=float,
                        default=SHOT_TYPE_PARAMS["nslice_bb"], dest="nslice_bb",
                        help="NEAR slice: bowback at/above this "
                        "(default %(default)s)")
    parser.add_argument("--nslice-diefrac", type=float,
                        default=SHOT_TYPE_PARAMS["nslice_diefrac"],
                        dest="nslice_diefrac",
                        help="NEAR slice: path-length ratio below this "
                        "(default %(default)s)")
    parser.add_argument("--nslice-peak-m", type=float,
                        default=SHOT_TYPE_PARAMS["nslice_peak_m"], dest="nslice_peak_m",
                        help="NEAR slice: peak speed (m/s) below this "
                        "(default %(default)s)")
    parser.add_argument("--far-peak-m", type=float,
                        default=SHOT_TYPE_PARAMS["far_peak_m"], dest="far_peak_m",
                        help="FAR flat if peak speed (m/s) at/above this, else "
                        "slice (default %(default)s)")
    parser.add_argument("--output", default="outputs/shot_analysis")
    parser.add_argument("--no-frames", action="store_true",
                        help="do not save the shot PNGs")
    parser.add_argument("--no-hitmap", action="store_true", dest="no_hitmap",
                        help="do not save the shot hitmap (player positions on "
                             "the minimap, coloured by shot type)")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="validate the logic on a synthetic trajectory")
    parser.add_argument("--type-self-test", action="store_true",
                        dest="type_self_test",
                        help="validate shot-type classification against the "
                        "embedded 23-shot ground truth for Input_video2")
    parser.add_argument("--overhead-self-test", action="store_true",
                        dest="overhead_self_test",
                        help="validate serve/smash (overhead) detection on "
                        "synthetic serve/smash/groundstroke/lob scenarios")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.type_self_test:
        type_self_test()
        return
    if args.overhead_self_test:
        overhead_self_test()
        return

    type_params = {
        "k": args.k, "w30": args.w30,
        "drop_diefrac": args.drop_diefrac, "drop_tail_m": args.drop_tail_m,
        "lob_diefrac": args.lob_diefrac, "lob_pace": args.lob_pace,
        "lob_reach": args.lob_reach, "nslice_bb": args.nslice_bb,
        "nslice_diefrac": args.nslice_diefrac, "nslice_peak_m": args.nslice_peak_m,
        "far_peak_m": args.far_peak_m,
    }

    # Derive the input CSV paths from the video name when not given explicitly,
    # matching the producers' defaults (players_<stem>.csv in player_coordinates/,
    # ball_<stem>.csv in ball_coordinates/, <stem>_court.csv in court_coordinates/).
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    if args.ball is None:
        args.ball = os.path.join("outputs", "ball_coordinates",
                                 f"ball_{video_stem}.csv")
    if args.players is None:
        args.players = os.path.join("outputs", "player_coordinates",
                                    f"players_{video_stem}.csv")
    if args.court is None:
        args.court = os.path.join("outputs", "court_coordinates",
                                  f"{video_stem}_court.csv")

    if not os.path.exists(args.ball):
        raise SystemExit(
            f"Ball CSV not found: {args.ball}\n"
            "Generate it first with the ball tracker (needs the YOLO model):\n"
            "    python tracking/BallTracking.py")

    hands = {1: args.p1_hand, 2: args.p2_hand}
    track = load_ball_track(args.ball)
    boxes = load_player_boxes(args.players)
    conv = CourtConverter(args.court)

    shots = analyze_shots(track, boxes, conv, args.fps, hands,
                          min_gap_s=args.min_gap, vy_min=args.vy_min,
                          acc_thr=args.acc_thr, win=args.win,
                          reversal_look=args.reversal_look,
                          reversal_vy_frac=args.reversal_vy_frac,
                          type_params=type_params)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "shots.csv"
    shots.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(shots)} shots)")

    if not args.no_frames and not shots.empty:
        save_shot_frames(shots, args.video, boxes, out_dir)

    if not args.no_hitmap:
        save_shot_hitmap(shots, out_dir)

    print_summary(shots, hands)


if __name__ == "__main__":
    main()
