"""
Shot detection + forehand/backhand/type classification from the ball CSV
(tracking/ball_tracking.py: frame,x,y,w,h,cx,cy,area) and the players CSV.

Detection (ball track smoothed with Savitzky-Golay):
  1. candidates = persistent vy sign reversal (image: vy<0 toward the far
     player, vy>0 toward the near; a shot flips it) AND/OR an acceleration-
     magnitude peak (abrupt velocity change);
  2. a candidate is a SHOT only if the ball is near a player box — bounces also
     reverse vy but happen far from the players;
  3. enforce a minimum gap (default 0.5 s), keeping the stronger acceleration.

Forehand / backhand at contact:
  - court side from the feet point in meters: near = seen from behind, his
    right is image-right; far = seen from the front, his right is image-LEFT;
  - forehand if the ball is on the dominant-hand side; inverted for left-handers
    (--p1-hand/--p2-hand left); "unknown" if the ball sits on the body axis.

Shot TYPE (flat/slice/dropshot/lob), orthogonal to fore/back, from the OUTGOING
trajectory shape. The angled camera makes near-player pixel speed ~3x the far
for the same shot, so raw pixel speed is never compared across sides;
perspective is handled by (a) scale-free pace = pixel speed / ball bbox-height,
(b) court-meters speed via the homography (far split only), and (c)
dimensionless shape features (diefrac, reach, bowback). dropshot = far ball dies
fast (low diefrac + low late speed); lob = near ball arcs up and returns;
slice = floated/decelerating ball; else flat.
CAVEAT: the homography is a GROUND plane but the ball is airborne at contact, so
meters speeds are approximate — every meters threshold is a CLI flag (see
--type-self-test) and may need per-camera retuning; the scale-free ones do not.
These rules are the fallback: a shot whose bounce is known gets its type from the
flight to that bounce (classify_flight).

Use (from project root, after the ball CSV exists):
    python analysis/shot_analysis.py --ball outputs/Input_video2/ball_coordinates/ball_Input_video2.csv
    python analysis/shot_analysis.py --ball outputs/Input_video2/ball_coordinates/ball_Input_video2.csv \\
        --p1-hand right --p2-hand left
    python analysis/shot_analysis.py --events outputs/Input_video2/shot_analysis/events.csv \\
        --real-ball outputs/Input_video2/ball_coordinates/ball_Input_video2.csv
    python analysis/shot_analysis.py --self-test        # synthetic, no YOLO
    python analysis/shot_analysis.py --type-self-test   # vs labelled ground truth
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

# Court dimensions shared via utils/court_geometry.
from utils.court_geometry import _FT, NET, SVC_T, L_m, W_m
from utils.paths import clip_paths

# ── loading ────────────────────────────────────────────────────────────────────

def load_ball_track(ball_csv: str) -> pd.DataFrame:
    """
    Realign the ball CSV over every frame and Savitzky-Golay smooth cx/cy.

    Returns a frame-indexed DataFrame with smoothed cx, cy (NaN where the ball
    was never seen) and a boolean `interp` column. `interp` flags interpolated
    straight-line fills (not real detections) so detect_hits can skip the
    frozen-BB seam where a long fill meets real motion (it mimics a contact).
    Sourced from the producer's `interpolated` column; absent on legacy CSVs
    -> all-False (seam guard becomes a no-op; we don't reconstruct it, as that
    is unreliable after the smoothing below).
    """
    df = pd.read_csv(ball_csv)
    if df.empty:
        raise ValueError(f"Empty ball CSV: {ball_csv}")
    idx = range(int(df["frame"].min()), int(df["frame"].max()) + 1)
    # `h` (ball bbox height) is the SCALE proxy for the shot-type classifier:
    # the ball looks bigger near the camera, so px-speed / local median(h) is a
    # perspective-robust ("scale-free") pace. Interpolated like cx/cy but NOT
    # smoothed (only its local median is used). Legacy CSVs lack it -> all-NaN
    # column; meters/shape features still work, pace/reach become NaN -> unknown.
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


def load_real_detections(ball_csv: str) -> pd.DataFrame:
    """Frames with a real ball detection (frame, cx, cy), without the filled gaps."""
    df = pd.read_csv(ball_csv)
    if "interpolated" in df.columns:
        df = df[df["interpolated"] == 0]
    df = df.dropna(subset=["cx", "cy"])
    return df[["frame", "cx", "cy"]].sort_values("frame").reset_index(drop=True)


def ball_track_for_shots(ball_csv: str) -> str:
    """The ball CSV that shot detection should read for `ball_csv`.

    Shot detection was designed and validated on the track of the original
    single-pass, whole-frame detector. The default ball track now comes from a
    denser multi-scale detector and a different tracker, which changes the
    timing of the contact candidates, so the whole-frame track written next to
    it (ball_wholeframe_<video>.csv) is used when it exists.
    """
    folder, name = os.path.split(ball_csv)
    if name.startswith("ball_") and not name.startswith(("ball_wholeframe_", "ball_candidates_")):
        alt = os.path.join(folder, "ball_wholeframe_" + name[len("ball_"):])
        if os.path.exists(alt):
            return alt
    return ball_csv


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
    Box of player ``pid`` in the nearest frame within ±radius, or None.

    Probes outward (d = 0, 1, 2, …); on a tie in |d| prefers the later frame
    (deterministic, so the result doesn't depend on probe order).
    """
    for d in range(radius + 1):
        for f in (frame + d, frame - d):   # later frame first on ties
            if f in boxes and pid in boxes[f]:
                return boxes[f][pid]
    return None


def _ball_near_player(ball_xy, box, reach_x=0.8, reach_y=0.7):
    """True if the ball is within racket reach of the player's box.

    The reach is a multiple of the box HEIGHT (arm + racket is about 0.8 x a
    person's height): the box width changes a lot with the pose, the height
    much less.
    """
    x, y, w, h = box
    bx, by = ball_xy
    return (x - reach_x * h <= bx <= x + w + reach_x * h
            and y - reach_y * h <= by <= y + h + reach_y * h)


def _gradient_runs(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    np.gradient per contiguous valid run, leaving NaN samples NaN.

    Differentiating across a NaN gap contaminates both neighbours (a 1-frame
    dropout would poison the velocity on each side and could hide an adjacent
    shot); splitting into maximal valid runs keeps gaps isolated.
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
        # valid run [i, j)
        if j - i >= 2:
            out[i:j] = np.gradient(values[i:j].astype(float))
        else:
            # lone sample has no gradient
            out[i:j] = np.nan
        i = j
    return out


def detect_hits(track: pd.DataFrame, boxes: dict, fps: float,
                min_gap_s: float = 0.5, vy_min: float = 0.5,
                acc_thr: float = 1.5, win: int = 4,
                reversal_look: int = 6,
                reversal_vy_frac: float = 0.5) -> list:
    """
    Detected shots as [(frame, player_id, acc_strength), ...].

    vy_min  : min |mean vy| (px/frame) before/after for a sign flip to count.
    acc_thr : threshold (px/frame^2) on acceleration-magnitude peaks.
    win     : half-window (frames) for the before/after vy means and the local
              acceleration-peak test.
    reversal_look    : half-window (frames) used ONLY to confirm an acc-only
              candidate (acc_peak True, flip False): outgoing vy must hold the
              opposite sign over it. nanmean-tolerant of a short post-contact
              dropout; longer than `win` to check a sustained direction.
    reversal_vy_frac : relaxed outgoing |mean vy| floor for that confirmation
              (= reversal_vy_frac * vy_min). Kept below vy_min so confirmation
              stays genuinely looser than `flip`.
    """
    frames = track.index.values
    cx = track["cx"].values
    cy = track["cy"].values
    valid = ~(np.isnan(cx) | np.isnan(cy))

    # Per-run differentiation so a 1-frame dropout can't hide an adjacent shot.
    vx = _gradient_runs(cx, valid)
    vy = _gradient_runs(cy, valid)
    valid_v = ~(np.isnan(vx) | np.isnan(vy))
    acc = np.hypot(_gradient_runs(vx, valid_v), _gradient_runs(vy, valid_v))

    # Post-contact reversal confirmation, gating acc-only candidates only.
    # Deliberately looser than `flip` (relaxed floor on the outgoing mean,
    # longer look-ahead, dropout-tolerant) — if it equalled `flip`,
    # `flip or (acc_peak and reversal)` would collapse to `flip`.
    rev_floor = reversal_vy_frac * vy_min
    min_valid = 2   # real vy samples required per window

    def _persistent_reversal(i):
        lo = max(0, i - reversal_look)
        hi = min(len(vy), i + 1 + reversal_look)
        if np.count_nonzero(valid_v[lo:i]) < min_valid:
            return False
        if np.count_nonzero(valid_v[i + 1:hi]) < min_valid:
            return False
        b = np.nanmean(vy[lo:i])          # incoming (excludes contact frame i)
        a = np.nanmean(vy[i + 1:hi])      # outgoing (excludes contact frame i)
        if np.isnan(b) or np.isnan(a):
            return False
        # sustained flip; floor on the outgoing side only
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
        # acc-only candidates need the relaxed reversal confirmation; flip ones don't.
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
                bx_c = _box_cx(box)
                by_c = box[1] + box[3] / 2.0
                d = np.hypot(ball_xy[0] - bx_c, ball_xy[1] - by_c)
                if best is None or d < best[1]:
                    best = (pid, d)
        if best is not None:
            near_player[i] = (best[0], strength)

    # Same-player wide-gap clustering: one shot per physical contact.
    # A contact often yields TWO reversals 10-29 frames apart (approach-side +
    # true contact). Distinct same-player rally shots are >= ~64 frames apart, so
    # a wide merge_gap collapses each double without merging genuine shots.
    # Opposite-player candidates are never merged (rally alternates) -> pid in key.
    merge_gap = max(int(min_gap_s * fps), 30)
    interp = (track["interp"].to_numpy().astype(bool)
              if "interp" in track.columns else np.zeros(len(frames), bool))

    def _fill_run(i, step):
        # Length of the interpolated-fill run adjacent to index i in direction step.
        c, k = 0, i + step
        while 0 <= k < len(interp) and interp[k]:
            c += 1
            k += step
        return c

    # Cluster same-player candidates (frame order); the strongest member of each
    # cluster (peak acceleration) is the real contact.
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
        # Frozen-BB seam guard: a candidate at the end of a long fill (incoming
        # >= 10 frames) with no real ball after it (outgoing == 0) is the
        # interpolation kink, not a contact. No-op on legacy all-False `interp`.
        if _fill_run(best, -1) >= 10 and _fill_run(best, +1) == 0:
            continue
        f = int(frames[best])
        pid, strength = near_player[best]
        hits.append((f, pid, strength))
    return hits


# ── forehand / backhand classification ─────────────────────────────────────────

def classify_stroke(ball_cx, player_box, player_side, hand,
                    deadband_frac=0.10):
    """
    "forehand" / "backhand" / "unknown" (ball on the body axis).

    player_side : "near" | "far" relative to the camera.
    hand        : "right" | "left".
    The body-axis dead band is a fraction of the box height, which depends on
    the pose much less than the width does.
    """
    x, y, w, h = player_box
    player_cx = x + w / 2.0
    db = ball_cx - player_cx
    if abs(db) < deadband_frac * h:
        return "unknown"
    # dominant hand points image-right iff (near and right) or (far and left)
    dominant_is_image_right = (player_side == "near") == (hand == "right")
    return "forehand" if (db > 0) == dominant_is_image_right else "backhand"


# ── shot-type (flat / slice / dropshot / lob) classification ────────────────────
#
# Orthogonal to fore/back: HOW the ball was struck, from its OUTGOING trajectory
# shape. The angled camera makes near-player px speed ~3x the far for the same
# shot, so px speed is never compared across sides; perspective is handled by:
#   * scale-free `pace`   = 100 * mean(px step speed) / median(ball bbox h);
#                           apparent size and speed shrink with distance alike.
#   * meters `peak_m`/`tail_m` via the homography (× fps), FAR flat/slice split
#     only. CAVEAT: ground homography, airborne ball -> m/s approximate, so every
#     meters threshold is a CLI flag for per-camera re-derivation.
#   * dimensionless SHAPE: `diefrac` (2nd-/1st-half outgoing path length = how
#     fast the ball stops), `reach` (path length / h), `bb` (bowback: vertical-
#     range fraction travelled back after an interior extreme; ~0 flat, high arc).
#
# Thresholds fit against a 23-shot ground truth, on wide plateaus, before the
# contacts that were really bounces were corrected (see --type-self-test).

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
    Scale-free bowback: fraction of the vertical range the ball travels back
    after its first interior vertical extreme. ~0 for a monotone drive, -> 1 for
    a lofted/floated arc (rises to an apex then returns, or vice-versa).
    """
    n = len(cy)
    if n < 6:
        return 0.0
    rng = float(np.nanmax(cy) - np.nanmin(cy))
    if rng < 20:          # essentially flat in y
        return 0.0
    imin = int(np.nanargmin(cy))
    imax = int(np.nanargmax(cy))
    # interior extreme only (1 <= idx <= n-3): an edge extreme is the window
    # boundary, not a real turn-around.
    up_down = (np.nanmax(cy[imin:]) - cy[imin]) / rng if 1 <= imin <= n - 3 else 0.0
    down_up = (cy[imax] - np.nanmin(cy[imax:])) / rng if 1 <= imax <= n - 3 else 0.0
    return float(max(up_down, down_up))


def _shot_type_features(track, conv, i: int, fps: float,
                        k: int, w30: int) -> dict:
    """
    Outgoing-trajectory features for the contact at array-index ``i``.

    Returns pace, peak_m, tail_m, apex, rise, bb, diefrac, reach (any may be NaN
    on a missing window).
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

    # speed in m/s via the ground homography (airborne ball -> approximate)
    pm = conv.to_meters_batch(np.column_stack([cxo, cyo]))
    dm = np.diff(pm, axis=0)
    spm = np.hypot(dm[:, 0], dm[:, 1]) * fps
    peak_m = float(np.nanmax(spm)) if np.any(~np.isnan(spm)) else np.nan
    tail_m = _mean(spm[18:26])                   # late-window speed

    apex = float(cyo[0] - np.nanmin(cyo)) if np.any(~np.isnan(cyo)) else np.nan
    rise = _mean(np.diff(cyo[:6]))               # initial vertical step (sign = direction)
    bb = _bowback(cyo)

    # half-vs-half path-length ratio over the longer W30 window
    cyo30 = cy[i:i + w30 + 1]
    cxo30 = cx[i:i + w30 + 1]
    half = w30 // 2
    p1 = _pathlen(cxo30[:half + 1], cyo30[:half + 1])
    p2 = _pathlen(cxo30[half:], cyo30[half:])
    diefrac = p2 / p1 if p1 > 1e-6 else np.nan
    reach = _pathlen(cxo30, cyo30) / hc if hc and not np.isnan(hc) else np.nan

    return dict(pace=pace, peak_m=peak_m, tail_m=tail_m, apex=apex, rise=rise,
                bb=bb, diefrac=diefrac, reach=reach)


def classify_shot_type(feats: dict, side: str, params: dict = None) -> str:
    """
    flat | slice | dropshot | lob | unknown, from the outgoing features and the
    striker's court side. First matching rule wins; "unknown" when the
    discriminating features are missing, so a bad shot never poses as a type.
    """
    p = params or SHOT_TYPE_PARAMS
    df = feats.get("diefrac")
    peak_m = feats.get("peak_m")

    # diefrac (and peak_m for the far split) are load-bearing; NaN -> can't classify.
    if df is None or np.isnan(df):
        return "unknown"

    # R1 DROPSHOT (far): path collapses in the 2nd half AND late speed is low.
    # diefrac is scale-free; the tail_m AND-gate protects a fast far-flat with a
    # middling diefrac.
    tail_m = feats.get("tail_m")
    if (side == "far" and df < p["drop_diefrac"]
            and tail_m is not None and not np.isnan(tail_m)
            and tail_m < p["drop_tail_m"]):
        return "dropshot"

    if side == "near":
        # R2a LOB: arcs up then returns -> tiny diefrac; triple-gated (slow pace,
        # short reach) since lobs are rare and false positives costly.
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

    # R3 FAR: fast drive -> flat, floated -> slice.
    if peak_m is None or np.isnan(peak_m):
        return "unknown"
    return "flat" if peak_m >= p["far_peak_m"] else "slice"


# ── overhead (smash) detection ─────────────────────────────────────────────────
#
# ADDITIVE to shot_type: the `overhead` column holds "serve" (found from the
# server's motion, see detect_serves) or "smash" (from the ball, here), and never
# overwrites shot_type. Deliberately CONSERVATIVE: zero false positives on
# groundstrokes and lobs is the priority, so anything uncertain returns "".
#
# Geometry (image y grows DOWN; meters via the ground homography):
#   * overhead gate: ball clearly ABOVE the head (ball_cy < box_top - margin*h)
#     and roughly over the body. Rare for groundstrokes (ball ~ waist, off to the
#     side), which is what protects the existing shot types.
#   * SMASH: off an opponent ball that arrives DESCENDING, then DRIVEN downward
#     with a low apex (vs a lob, which arcs up).

OVERHEAD_PARAMS = {
    # Common overhead gate
    "above_head_frac": 0.25,   # ball must be above box_top by this fraction of box h
    "x_align_frac": 0.60,      # |ball_cx - player_cx| must be below this fraction of box w
    # Smash gate
    "smash_incoming_down_px": 5.0,   # incoming ball must be DESCENDING: median per-step
                                     # dcy (px/frame) at/above this (image y grows down)
    "smash_down_rise": 0.5,    # outgoing must descend: mean initial vertical step (px/frame) >= this
                               # (image y grows down, so a positive `rise` means the ball goes DOWN)
    "smash_min_apex": 6.0,     # outgoing apex (px the ball rises above contact) must be SMALL
                               # (a driven smash barely rises; a lob's apex is large) — upper bound
}


def _box_cx(box):
    """Horizontal centre (pixels) of a player box (x, y, w, h)."""
    return box[0] + box[2] / 2.0


def _box_feet(box):
    """Feet (mid-bottom) pixel point of a player box (x, y, w, h)."""
    x, y, w, h = box
    return (_box_cx(box), y + h)


def _incoming_ball_features(track, conv, i, fps, win=10):
    """
    Incoming-ball features over [i-win:i] — the mirror of _shot_type_features,
    used to tell a serve (slow, local toss) from a smash (fast, across-net).

    Returns:
      in_speed_m : incoming speed (m/s, median of per-step homography speeds),
                   NaN if the pre-contact track is too short/holey.
      in_y_m, in_x_m : ball court position (meters) ~win frames pre-contact,
                   placing the ball's origin (own side vs across net). NaN if
                   unavailable.
    """
    out = dict(in_speed_m=np.nan, in_y_m=np.nan, in_x_m=np.nan, in_vy_px=np.nan)
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
    # Incoming vertical direction in PIXELS (robust to homography distortion of
    # the airborne ball near the baseline). y grows down, so median dcy < 0 =
    # RISING (serve toss), > 0 = DESCENDING (ball falling for a smash).
    dcy = np.diff(cy)
    dcy = dcy[~np.isnan(dcy)]
    if len(dcy):
        out["in_vy_px"] = float(np.median(dcy))
    # earliest valid pre-contact court position (the incoming ball's origin)
    for k in range(len(pm)):
        if np.all(np.isfinite(pm[k])):
            out["in_x_m"] = float(pm[k, 0])
            out["in_y_m"] = float(pm[k, 1])
            break
    return out


def classify_overhead(box, ball_cx, ball_cy, in_feats, out_feats, params=None):
    """
    "smash" | "" (none). CONSERVATIVE: returns "" the moment any required signal
    is missing or out of range. Serves are found separately (detect_serves).

    box                : player bbox (x, y, w, h) px at contact.
    ball_cx, ball_cy   : ball px at contact (image y grows downward).
    in_feats           : from _incoming_ball_features.
    out_feats          : from _shot_type_features (outgoing shape).
    """
    p = params or OVERHEAD_PARAMS
    if box is None or np.isnan(ball_cx) or np.isnan(ball_cy):
        return ""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return ""

    # ── Overhead gate: ball clearly above head AND x-aligned over the body ──
    # (y grows down, so "above" = smaller y than the box top).
    if not (ball_cy < y - p["above_head_frac"] * h):
        return ""
    if not (abs(ball_cx - (x + w / 2.0)) < p["x_align_frac"] * w):
        return ""

    # ── SMASH: the ball arrives descending and is driven DOWNWARD on the way out ──
    rise = out_feats.get("rise") if out_feats else None     # +ve = ball goes DOWN
    apex = out_feats.get("apex") if out_feats else None      # px the ball rose above contact
    in_vy = in_feats.get("in_vy_px") if in_feats else None
    descending_in = (in_vy is not None and not np.isnan(in_vy)
                     and in_vy >= p["smash_incoming_down_px"])
    downward = (rise is not None and not np.isnan(rise)
                and rise >= p["smash_down_rise"])
    low_apex = (apex is not None and not np.isnan(apex)
                and apex <= p["smash_min_apex"])
    if descending_in and downward and low_apex:
        return "smash"
    return ""


# ── serve detection from the server's motion ───────────────────────────────────
#
# The ball alone is not enough to find a serve: the toss release looks like a
# contact (the ball reverses near the player), the contact frame is often
# missed, and the far player's toss is usually not detected at all. The server's
# motion is a clearer signature: standing near their own baseline at the start
# of a point, they raise arm and racket above the head (the box grows taller)
# and bring them down after the contact. The contact is the last height peak of
# that raised phase.

SERVE_PARAMS = {
    "raised_min": 1.15,      # box height / player's median height: raised phase
    "raised_peak": 1.20,     # the phase must reach at least this...
    "min_raised_s": 0.2,     # ...and last at least this long
    "max_dip_s": 0.2,        # shorter dips inside the phase are bridged (racket drop)
    "peak_tol": 0.03,        # contact = last local max within this fraction of the max
    "near_baseline_m": 2.0,  # feet at most this far inside their own baseline
    "max_move_m": 1.5,       # feet displacement during the phase (server stands)
    "quiet_s": 2.5,          # no opponent shot in this time before the phase
    "next_shot_s": 1.0,      # the same player striking again this soon: not a serve
    "min_tail_s": 0.3,       # the clip must go on this long after the contact
}


def _px_per_metre(conv, x_m, y_m):
    """Image scale at a ground point; it also holds for heights above that point."""
    m_to_px = np.linalg.inv(conv._H)
    ends = np.array([[[x_m - 0.5, y_m]], [[x_m + 0.5, y_m]]], dtype=np.float64)
    a, b = cv2.perspectiveTransform(ends, m_to_px)[:, 0]
    return float(np.hypot(*(b - a)))


def _metric_heights(boxes, conv, pid):
    """Box height of player `pid` in metres-equivalent (px / local px-per-metre).

    Dividing by the image scale at the feet removes perspective, so heights at
    the near and far baselines are comparable. Returns (frames, heights, feet_m).
    """
    frames = sorted(f for f in boxes if pid in boxes[f])
    if not frames:
        return np.array([], int), np.array([]), np.empty((0, 2))
    b = np.array([boxes[f][pid] for f in frames], float)        # x, y, w, h
    feet_px = np.c_[b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3]]
    feet_m = conv.to_meters_batch(feet_px)
    m_to_px = np.linalg.inv(conv._H)
    left = cv2.perspectiveTransform((feet_m - [0.5, 0]).reshape(-1, 1, 2), m_to_px)[:, 0]
    right = cv2.perspectiveTransform((feet_m + [0.5, 0]).reshape(-1, 1, 2), m_to_px)[:, 0]
    ppm = np.hypot(*(right - left).T)
    return np.array(frames), b[:, 3] / np.maximum(ppm, 1e-6), feet_m


def detect_serves(boxes, conv, fps, hits, params=None):
    """Serves as [(contact_frame, pid, phase_start_frame), ...].

    `hits` (from detect_hits) is used to check that a serve starts a point (no
    opponent shot just before) and is answered by the opponent.
    """
    p = params or SERVE_PARAMS
    min_len = max(2, int(round(p["min_raised_s"] * fps)))
    max_dip = max(1, int(round(p["max_dip_s"] * fps)))
    serves = []
    for pid in (1, 2):
        frames, hm, feet_m = _metric_heights(boxes, conv, pid)
        if len(frames) < min_len:
            continue
        rel = hm / np.nanmedian(hm)
        raised = rel >= p["raised_min"]
        # Raised phases: runs of raised frames, bridging dips up to max_dip frames.
        phases, start, last = [], None, None
        for k, (f, up) in enumerate(zip(frames, raised)):
            if not up:
                continue
            if start is not None and f - frames[last] <= max_dip + 1:
                last = k
                continue
            if start is not None:
                phases.append((start, last))
            start = last = k
        if start is not None:
            phases.append((start, last))

        for a, b in phases:
            if frames[b] - frames[a] + 1 < min_len or rel[a:b + 1].max() < p["raised_peak"]:
                continue
            seg = rel[a:b + 1]
            top = seg.max() * (1 - p["peak_tol"])
            peaks = [k for k in range(len(seg))
                     if seg[k] >= top
                     and (k == 0 or seg[k] >= seg[k - 1])
                     and (k == len(seg) - 1 or seg[k] >= seg[k + 1])]
            contact_k = a + peaks[-1]
            contact, phase_start = int(frames[contact_k]), int(frames[a])
            # A phase cut by the end of the clip has no real peak.
            if frames[-1] - contact < p["min_tail_s"] * fps:
                continue

            # Near their own baseline (or behind it) and standing.
            y = feet_m[a, 1]
            near_own_baseline = (y >= L_m - p["near_baseline_m"] if y > NET
                                 else y <= p["near_baseline_m"])
            settle = max(a, contact_k - min_len)                # before the jump into the court
            moved = float(np.hypot(*(feet_m[settle] - feet_m[a])))
            if not near_own_baseline or not np.isfinite(moved) or moved > p["max_move_m"]:
                continue
            # Start of a point: no opponent shot just before the phase...
            quiet_from = phase_start - p["quiet_s"] * fps
            if any(h_pid != pid and quiet_from <= h_f <= phase_start for h_f, h_pid, _ in hits):
                continue
            # ...and the next shot after the contact is the opponent's (the return).
            later = [(h_f, h_pid) for h_f, h_pid, _ in hits
                     if contact + 0.3 * fps < h_f <= contact + p["next_shot_s"] * fps]
            if later and min(later)[1] == pid:
                continue
            serves.append((contact, pid, phase_start))

    # A big swing just after a serve is the return, not another serve.
    kept = []
    for serve in sorted(serves):
        if not kept or serve[2] - kept[-1][0] > p["quiet_s"] * fps:
            kept.append(serve)
    return kept


# ── contacts that are really bounces ───────────────────────────────────────────
#
# detect_hits keeps the reversals close to a player, and a bounce in front of the
# receiver is one of them. The events read from the track (bounce_detection.py)
# tell the two apart, so such a contact moves to the hit that follows the bounce.

HIT_NEAR, BOUNCE_FAR, HIT_FAR, BOUNCE_NEAR = "hit_near", "bounce_far", "hit_far", "bounce_near"
HIT_OF_PLAYER = {1: HIT_NEAR, 2: HIT_FAR}
BOUNCE_OF_PLAYER = {1: BOUNCE_NEAR, 2: BOUNCE_FAR}

CONTACT_PARAMS = {
    "bounce_tol_s": 0.1,     # a contact this close to a bounce on its own side is that bounce
    "next_hit_s": 0.6,       # the hit comes at most this long after the bounce
}


def move_contacts_off_bounces(hits, events, fps, min_gap_s=0.5, params=None):
    """Move each hit sitting on a bounce of its own player to the hit after it.

    Without a following hit in `events` the contact stays where it is. Returns the
    hits and {moved frame: (cx, cy)}, the ball at those contacts: the track used
    for the shots is often a straight fill there, cutting the corner of the hit.
    """
    p = params or CONTACT_PARAMS
    moved, ball_at = [], {}
    for f, pid, strength in hits:
        bounces = events[(events["event"] == BOUNCE_OF_PLAYER[pid])
                         & ((events["frame"] - f).abs() <= p["bounce_tol_s"] * fps)]
        if not bounces.empty:
            t = bounces["frame"].iloc[0]
            after = events[(events["event"] == HIT_OF_PLAYER[pid]) & (events["frame"] > t)
                           & (events["frame"] <= t + p["next_hit_s"] * fps)]
            if not after.empty:
                hit = after.loc[np.hypot(after["jump_up_ms"], after["jump_side_ms"]).idxmax()]
                f = int(round(hit["frame"]))
                ball_at[f] = (float(hit["cx"]), float(hit["cy"]))
        moved.append((f, pid, strength))

    kept = []
    for hit in sorted(moved):
        if not any(k[1] == hit[1] and hit[0] - k[0] < min_gap_s * fps for k in kept):
            kept.append(hit)
    return kept, ball_at


# ── shot type from the flight ──────────────────────────────────────────────────
#
# Once a shot's bounce is known its type comes from the flight itself, which is
# physics rather than image shape. A rally ball crosses in under a second; of the
# slower ones a dropshot lands short, a lob flies high, and a slice floats low.
# A volley is told by what came before: the ball never touched the ground.

FLIGHT_PARAMS = {
    "slow_flight_s": 1.05,   # a longer flight is no longer a rally ball
    "drop_depth_m": 4.5,     # a dropshot lands this close to the net...
    "lob_apex_m": 2.0,       # ...a lob flies above this height (the net is 0.91 m)
    "late_bounce_s": 0.1,    # a bounce may follow the next contact by this much
}


def shot_flight(contact, pid, feet_m, next_contact, events, real_ball, conv, fps,
                params=None):
    """Flight of a shot to its bounce, or None if no bounce was found.

    {"time_s", "depth_m" (landing point to net), "apex_m"}. The height of the ball
    is its distance in the image from the point below it, taken on the straight
    path from the striker's feet to the landing point.
    """
    p = params or FLIGHT_PARAMS
    bounces = events[(events["event"] == BOUNCE_OF_PLAYER[3 - pid])
                     & (events["frame"] > contact)
                     & (events["frame"] < next_contact + p["late_bounce_s"] * fps)]
    if bounces.empty:
        return None
    bounce = bounces.iloc[0]
    seen = real_ball[real_ball["frame"].between(contact, bounce["frame"])]
    if seen.empty:
        return None
    start = np.asarray(feet_m, float)
    end = np.array([bounce["x_m"], bounce["y_m"]], float)
    m_to_px = np.linalg.inv(conv._H)
    heights = []
    for frame, cy in zip(seen["frame"], seen["cy"]):
        below = start + (frame - contact) / (bounce["frame"] - contact) * (end - start)
        _, v, w = m_to_px @ np.array([below[0], below[1], 1.0])
        heights.append((v / w - cy) / _px_per_metre(conv, *below))
    return {"time_s": (bounce["frame"] - contact) / fps,
            "depth_m": abs(bounce["y_m"] - NET),
            "apex_m": max(heights)}


def classify_flight(flight, params=None):
    """flat / dropshot / lob / slice from shot_flight."""
    p = params or FLIGHT_PARAMS
    if flight["time_s"] < p["slow_flight_s"]:
        return "flat"
    if flight["depth_m"] <= p["drop_depth_m"]:
        return "dropshot"
    return "lob" if flight["apex_m"] >= p["lob_apex_m"] else "slice"


def is_volley(contact, pid, y_m, previous, events, fps, params=None):
    """True for a shot played inside the service line, before the ball bounced.

    `previous` is the shot before, (frame, player, ...) or None. The service line
    matters: a bounce hidden behind a player at the baseline also leaves no bounce
    in `events`, and nobody volleys from there.
    """
    p = params or FLIGHT_PARAMS
    if previous is None or previous[1] == pid or abs(y_m - NET) >= NET - SVC_T:
        return False
    landed = events[(events["event"] == BOUNCE_OF_PLAYER[pid])
                    & (events["frame"] > previous[0])
                    & (events["frame"] <= contact + p["late_bounce_s"] * fps)]
    return landed.empty


def analyze_shots(track, boxes, conv, fps, hands, min_gap_s=0.5,
                  vy_min=0.5, acc_thr=1.5, win=4,
                  reversal_look=6, reversal_vy_frac=0.5,
                  type_params=None, events=None, real_ball=None):
    """Detect and classify every shot; return a DataFrame.

    events    : hits and bounces from bounce_detection.py, to correct the contacts
                that are really bounces. Without them the contacts stay as detected.
    real_ball : real detections of the ball (load_real_detections). With `events`,
                the type of a shot with a known bounce comes from its flight.
    """
    tp = type_params or SHOT_TYPE_PARAMS
    # Warn on a ball/player frame-range mismatch: non-overlapping ranges mean
    # _nearest_box never matches and the user would just see "No shots detected".
    if len(track.index) and boxes:
        b0, b1 = int(track.index.min()), int(track.index.max())
        p0, p1 = min(boxes), max(boxes)
        if b1 < p0 or p1 < b0:
            print(f"  Warning: ball frames [{b0},{b1}] don't overlap player "
                  f"frames [{p0},{p1}] - frame numbering mismatch between the "
                  f"ball CSV and the players CSV; no shots can be detected.")

    hits = detect_hits(track, boxes, fps, min_gap_s=min_gap_s,
                       vy_min=vy_min, acc_thr=acc_thr, win=win,
                       reversal_look=reversal_look,
                       reversal_vy_frac=reversal_vy_frac)

    # Serves come from the server's motion (detect_serves). The server's own
    # "shots" during the toss are the toss release, not contacts: replace them
    # with the serve contact.
    serves = detect_serves(boxes, conv, fps, hits)
    serve_starts = {c: start for c, _, start in serves}
    for contact, pid, phase_start in serves:
        lo, hi = phase_start - 0.5 * fps, contact + 0.3 * fps
        hits = [h for h in hits if not (h[1] == pid and lo <= h[0] <= hi)]
        hits.append((contact, pid, 0.0))
    hits.sort()
    ball_at = {}
    if events is not None:
        hits, ball_at = move_contacts_off_bounces(hits, events, fps, min_gap_s)

    n_dropped = 0
    rows = []
    for k, (f, pid, _strength) in enumerate(hits):
        box = _nearest_box(boxes, f, pid)
        if box is None:
            continue
        is_serve = f in serve_starts
        # ±2-frame median ball position (denoise)
        sel = track.loc[max(track.index.min(), f - 2): f + 2]
        # An all-NaN window (ball never seen here) would give nanmedian = NaN and
        # poison classify_stroke / to_meters; skip and count instead. A serve is
        # kept anyway: it is found from the server's motion, and the far
        # player's toss is often not detected.
        cx_win = sel["cx"].values
        cy_win = sel["cy"].values
        if np.all(np.isnan(cx_win)) or np.all(np.isnan(cy_win)):
            if not is_serve:
                n_dropped += 1
                continue
            ball_cx = ball_cy = np.nan
        else:
            ball_cx = float(np.nanmedian(cx_win))
            ball_cy = float(np.nanmedian(cy_win))
        ball_cx, ball_cy = ball_at.get(f, (ball_cx, ball_cy))

        # A server is in the air at contact, and feet off the ground project to the
        # wrong place on the court: take the stance, before the jump.
        stance = _nearest_box(boxes, serve_starts[f], pid) if is_serve else None
        feet = _box_feet(stance or box)
        # Player feet in meters — the hitmap marks where the PLAYER struck, not the ball.
        player_x_m, player_y_m = conv.to_meters(*feet)
        side = "near" if player_y_m > NET else "far"
        stroke = ("unknown" if np.isnan(ball_cx)
                  else classify_stroke(ball_cx, box, side, hands[pid]))

        # Shot type from outgoing shape; contact frame `f` -> array-index `i`
        # (the frame index is contiguous).
        i = int(f - track.index.min())
        feats = _shot_type_features(track, conv, i, fps, tp["k"], tp["w30"])
        shot_type = classify_shot_type(feats, side, tp)
        flight = None
        if events is not None and real_ball is not None:
            next_contact = hits[k + 1][0] if k + 1 < len(hits) else np.inf
            flight = shot_flight(f, pid, (player_x_m, player_y_m), next_contact,
                                 events, real_ball, conv, fps)
        if flight is not None:
            shot_type = classify_flight(flight)
        if events is not None and not is_serve and is_volley(
                f, pid, player_y_m, hits[k - 1] if k else None, events, fps):
            shot_type = "volley"

        # Overhead (serve/smash): separate column; wrapped so short/missing data
        # yields "" rather than raising. Serves are already known; the ball-based
        # classifier is only asked about smashes.
        if is_serve:
            overhead = "serve"
        else:
            try:
                in_feats = _incoming_ball_features(track, conv, i, fps)
                overhead = classify_overhead(box, ball_cx, ball_cy, in_feats, feats)
            except Exception:
                overhead = ""

        # CAVEAT: to_meters uses the ground homography but the ball is airborne at
        # contact, so ball_x_m/ball_y_m are indicative, not exact.
        bx_m, by_m = conv.to_meters(ball_cx, ball_cy)
        rows.append({
            "frame": f, "time_s": round(f / fps, 2), "player_id": pid,
            "side": side, "hand": hands[pid], "stroke": stroke,
            "shot_type": shot_type,
            "overhead": overhead,
            "flight_s": round(flight["time_s"], 2) if flight else None,
            "landing_depth_m": round(flight["depth_m"], 1) if flight else None,
            "apex_m": round(flight["apex_m"], 1) if flight else None,
            "ball_pace": (round(feats["pace"], 1)
                          if feats["pace"] is not None
                          and not np.isnan(feats["pace"]) else None),
            "ball_cx": round(ball_cx, 1), "ball_cy": round(ball_cy, 1),
            "player_cx": round(_box_cx(box), 1),
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
        # shot_type absent on a legacy DataFrame
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
        for t in ("flat", "slice", "dropshot", "lob", "volley", "unknown"):
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
# Combined category = one colour per shot: a special shot_type
# (slice/dropshot/lob/serve) wins, else the forehand/backhand stroke. Marker
# SHAPE encodes the player (P1 = circle, P2 = triangle).
SHOT_CATEGORY_COLORS = {
    "forehand": "#2ca02c",   # green
    "backhand": "#ff7f0e",   # orange
    "slice":    "#9467bd",   # purple
    "dropshot": "#d62728",   # red
    "lob":      "#8c564b",   # brown
    "volley":   "#e377c2",   # pink
    "serve":    "#17becf",   # cyan
    "smash":    "#d4af37",   # gold
    "unknown":  "#7f7f7f",   # grey
}
_CATEGORY_FALLBACK_COLOR = "#111111"
_SPECIAL_SHOT_TYPES = ("slice", "dropshot", "lob", "volley", "serve")
_PLAYER_MARKERS = {1: "o", 2: "^"}
_PLAYER_MARKER_FALLBACK = "s"


def shot_category(stroke, shot_type, overhead=""):
    """Colour category for a shot: overhead (serve/smash) wins, else a special
    shot_type (slice/dropshot/lob/volley/serve), else the forehand/backhand stroke."""
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
    """Plot each shot at the PLAYER's feet (meters) on the minimap, coloured by
    combined category, marker shape per player. Reuses player_analysis's court
    drawing."""
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

    from analysis.player_analysis import _draw_court

    fig, ax = plt.subplots(figsize=(5.2, 9.0))
    # Dark background to match heatmap_combined.png. _draw_court() turns the axis
    # OFF (hiding the patch -> white lines invisible on white), so re-enable a
    # dark patch below.
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
    """Write a synthetic court CSV: homography from 4 px corners + ITF proportions."""
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
    Synthetic 4-shot rally with mid-court bounces that must NOT be detected.
    Checks frame, player and stroke, including the left-handed case.
    """
    rng = np.random.default_rng(3)
    p1 = (900, 700, 100, 190)     # near, feet y=890, cx=950
    p2 = (920, 200, 60, 120)      # far,  feet y=320, cx=950

    # (shot_frame, player, ball offset from center, expected stroke for a right-hander)
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

    # linear segments between contacts, with a bounce cusp at mid-court
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
            # bounce cusp at s=0.65 (dips toward image bottom)
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

# User-provided ground truth for the 23 detected shots of Input_video2. A
# "dropshot/slice" label is ambiguous and counts correct for EITHER.
_TYPE_GROUND_TRUTH = {
    84: "flat", 115: "flat", 152: "flat", 188: "flat", 225: "flat",
    262: "dropshot", 322: "flat", 349: "slice", 389: "lob", 432: "flat",
    469: "flat", 503: "slice", 554: "flat", 594: "flat", 629: "flat",
    660: "flat", 696: "flat", 740: "dropshot/slice", 785: "flat", 828: "flat",
    861: "dropshot/slice", 912: "slice", 950: "flat",
}
_TYPE_TARGET_ACC = 0.90


def type_self_test(ball_csv: str = None, players_csv: str = None,
                   court_csv: str = None, fps: float = 30.0) -> None:
    """
    Validate the shot-TYPE classifier against the Input_video2 ground truth,
    end-to-end through the real analyze_shots pipeline.

    Each GT frame is matched to the nearest detection within ±3 frames; extra
    detections are ignored. Exits non-zero if accuracy < 90 %.
    """
    paths = clip_paths("Input_video2",
                       str(Path(__file__).resolve().parent.parent / "outputs"))
    ball_csv = ball_csv or ball_track_for_shots(paths["ball_csv"])
    players_csv = players_csv or paths["players_csv"]
    court_csv = court_csv or paths["court_csv"]

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
    from analysis.bounce_detection import clip_events
    real_csv = paths["ball_csv"]
    events = clip_events(real_csv, boxes, conv, fps)
    # GT was labelled on the default right/right run
    shots = analyze_shots(track, boxes, conv, fps, {1: "right", 2: "right"}, events=events,
                          real_ball=load_real_detections(real_csv))

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
              f"+/-3 frames (detection drift, counted as wrong).")
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
    Validate serve and smash detection on synthetic scenarios (same court as
    self_test):
      (a) SERVE — near player at their baseline raises arm and racket (taller box),
          dips, peaks at contact, lowers -> one serve at the last peak.
      (b) NO SERVE — the same motion right after an opponent shot (a rally).
      (c) SMASH — mid-court, fast descending ball, low-apex downward -> "smash".
      (d) GROUNDSTROKE — ball at waist beside the body -> "".
      (e) LOB — struck low and lofted up -> "".
    Drives the real helpers end-to-end.
    """
    # Same homography as _synthetic_court_csv: meters -> pixels.
    corners_m = np.array([[0, 0], [W_m, 0], [0, L_m], [W_m, L_m]],
                         dtype=np.float32)
    corners_px = np.array([[700, 300], [1200, 300], [400, 860], [1500, 860]],
                          dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners_m, corners_px)

    def m2px(xm, ym):
        p = H @ np.array([xm, ym, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def build_track(seg, frames=80, hpx=12.0):
        """Linearly interpolate seg [(frame, cx, cy), ...] into a track DataFrame."""
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
        bw, bh = 90.0, 200.0

        # ── (a) SERVE from the server's motion ─────────────────────────────────
        # Near player just behind their baseline; far player standing still.
        feet = m2px(W_m / 2.0, L_m + 0.3)
        far_feet = m2px(W_m / 2.0, -0.5)
        heights = {f: bh for f in range(150)}
        heights.update({f: 1.4 * bh for f in range(40, 62)})      # toss: arm up
        heights.update({f: 1.05 * bh for f in range(62, 66)})     # racket drop
        heights.update({66: 1.6 * bh, 67: 1.7 * bh, 68: 1.3 * bh})  # contact at 67
        boxes = {f: {1: (feet[0] - bw / 2, feet[1] - h, bw, h),
                     2: (far_feet[0] - 20, far_feet[1] - 90, 40, 90)}
                 for f, h in heights.items()}
        toss_release = [(41, 1, 2.0)]          # the ball leaving the hand looks like a hit
        serves = detect_serves(boxes, conv, fps, toss_release + [(100, 2, 3.0)])
        if [(c, pid) for c, pid, _ in serves] != [(67, 1)]:
            failures.append(f"(a) SERVE: got {serves}, expected one serve by P1 at frame 67")

        # ── (b) NO SERVE in a rally ────────────────────────────────────────────
        serves = detect_serves(boxes, conv, fps, [(25, 2, 3.0)])     # opponent just hit
        if serves:
            failures.append(f"(b) RALLY: got {serves}, expected no serve")

        # ── (c) SMASH ──────────────────────────────────────────────────────────
        # Near player mid-court; ball arrives DESCENDING and is driven steeply DOWN.
        feet_px = m2px(W_m / 2.0, NET + 2.0)
        box = (feet_px[0] - bw / 2.0, feet_px[1] - bh, bw, bh)
        ball_cx = box[0] + bw / 2.0
        ball_cy = box[1] - 0.5 * bh             # above the head at contact
        i = 30
        seg = [(i - 12, ball_cx - 30, ball_cy - 140),
               (i, ball_cx, ball_cy),
               (i + 26, ball_cx + 40, ball_cy + 260)]
        track = build_track(seg, frames=80)
        in_feats = _incoming_ball_features(track, conv, i, fps)
        out_feats = _shot_type_features(track, conv, i, fps,
                                        SHOT_TYPE_PARAMS["k"], SHOT_TYPE_PARAMS["w30"])
        ov = classify_overhead(box, ball_cx, ball_cy, in_feats, out_feats)
        if ov != "smash":
            failures.append(
                f"(c) SMASH: got {ov!r}, expected 'smash' "
                f"(rise={out_feats['rise']:.2f}, apex={out_feats['apex']:.2f})")

        # ── (d) GROUNDSTROKE ────────────────────────────────────────────────────
        # Ball at waist, off to the side -> overhead gate fails -> "".
        feet_px = m2px(W_m / 2.0, NET + 4.0)
        box = (feet_px[0] - bw / 2.0, feet_px[1] - bh, bw, bh)
        ball_cx = box[0] + bw + 30              # off to the side
        ball_cy = box[1] + 0.6 * bh             # waist height (inside the box)
        seg = [(i - 12, ball_cx - 120, ball_cy + 10),
               (i, ball_cx, ball_cy),
               (i + 26, ball_cx + 220, ball_cy - 20)]
        track = build_track(seg, frames=80)
        in_feats = _incoming_ball_features(track, conv, i, fps)
        out_feats = _shot_type_features(track, conv, i, fps,
                                        SHOT_TYPE_PARAMS["k"], SHOT_TYPE_PARAMS["w30"])
        ov = classify_overhead(box, ball_cx, ball_cy, in_feats, out_feats)
        if ov != "":
            failures.append(f"(d) GROUNDSTROKE: got {ov!r}, expected '' (no overhead)")

        # ── (e) LOB ─────────────────────────────────────────────────────────────
        # Struck low and lofted UP -> gate fails on height -> "".
        ball_cx = box[0] + bw / 2.0
        ball_cy = box[1] + 0.5 * bh             # waist height, NOT above head
        seg = [(i - 12, ball_cx - 60, ball_cy + 30),
               (i, ball_cx, ball_cy),
               (i + 13, ball_cx + 120, ball_cy - 220),
               (i + 26, ball_cx + 240, ball_cy - 40)]
        track = build_track(seg, frames=80)
        in_feats = _incoming_ball_features(track, conv, i, fps)
        out_feats = _shot_type_features(track, conv, i, fps,
                                        SHOT_TYPE_PARAMS["k"], SHOT_TYPE_PARAMS["w30"])
        ov = classify_overhead(box, ball_cx, ball_cy, in_feats, out_feats)
        if ov != "":
            failures.append(f"(e) LOB: got {ov!r}, expected '' (no overhead)")

    print("OVERHEAD-SELF-TEST shot_analysis")
    if failures:
        print("  FAIL:")
        for msg in failures:
            print("   -", msg)
        raise SystemExit(1)
    print("  PASS: 5/5 - serve found from the server's motion (none in a rally), smash "
          "detected; groundstroke and lob produce no overhead (zero false positives).")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shot detection and forehand/backhand classification "
                    "from the ball tracking")
    parser.add_argument("--ball", default=None,
                        help="ball CSV generated by ball_tracking.py (defaults to "
                             "outputs/<video name>/ball_coordinates/ball_<video name>.csv)")
    parser.add_argument("--players", default=None,
                        help="player tracking CSV (defaults to "
                             "outputs/<video name>/player_coordinates/players_<video name>.csv)")
    parser.add_argument("--court", default=None,
                        help="court keypoints CSV (defaults to "
                             "outputs/<video name>/court_coordinates/<video name>_court.csv)")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="source video for the verification PNGs")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--p1-hand", choices=["right", "left"],
                        default="right", dest="p1_hand",
                        help="dominant hand of player 1 (default right)")
    parser.add_argument("--p2-hand", choices=["right", "left"],
                        default="right", dest="p2_hand",
                        help="dominant hand of player 2 (default right)")
    parser.add_argument("--events", default=None,
                        help="events CSV from analysis/bounce_detection.py, to correct "
                             "the contacts that are really bounces")
    parser.add_argument("--real-ball", default=None, dest="real_ball",
                        help="ball CSV with the real detections (ball_<video name>.csv); "
                             "with --events, shot types come from the flight to the bounce")
    parser.add_argument("--min-gap", type=float, default=0.5, dest="min_gap",
                        help="minimum distance in seconds between two shots")
    # Detection thresholds (px/frame units, tuned on the demo clip).
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
    # Shot-TYPE thresholds (defaults = SHOT_TYPE_PARAMS). The meters-based cuts
    # (--drop-tail-m, --*-peak-m) assume the video fps and an accurate homography
    # and may need per-camera retuning; the scale-free cuts do not.
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
    parser.add_argument("--output", default=None,
                        help="output directory (default: outputs/<video name>/shot_analysis)")
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

    # Missing paths come from the clip's folder. Its bounces go with its own ball
    # track, so they are only picked up when --ball is not overridden.
    paths = clip_paths(args.video)
    if args.ball is None:
        args.ball = ball_track_for_shots(paths["ball_csv"])
        if args.events is None and os.path.exists(paths["events_csv"]):
            args.events, args.real_ball = paths["events_csv"], paths["ball_csv"]
    args.players = args.players or paths["players_csv"]
    args.court = args.court or paths["court_csv"]
    args.output = args.output or paths["shots_dir"]

    if not os.path.exists(args.ball):
        raise SystemExit(
            f"Ball CSV not found: {args.ball}\n"
            "Generate it first with the ball tracker (needs the YOLO model):\n"
            "    python tracking/ball_tracking.py")

    hands = {1: args.p1_hand, 2: args.p2_hand}
    track = load_ball_track(args.ball)
    boxes = load_player_boxes(args.players)
    conv = CourtConverter(args.court)

    shots = analyze_shots(track, boxes, conv, args.fps, hands,
                          min_gap_s=args.min_gap, vy_min=args.vy_min,
                          acc_thr=args.acc_thr, win=args.win,
                          reversal_look=args.reversal_look,
                          reversal_vy_frac=args.reversal_vy_frac,
                          type_params=type_params,
                          events=pd.read_csv(args.events) if args.events else None,
                          real_ball=(load_real_detections(args.real_ball)
                                     if args.real_ball else None))

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
