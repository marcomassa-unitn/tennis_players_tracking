"""Court keypoint detection by fitting the ITF court model to the white lines.

1. Line pixels: a pixel belongs to a court line if it is bright, unsaturated and
   brighter than both of its neighbours `tau` px away, horizontally or
   vertically. This keeps thin lines -- dim far lines included -- and rejects
   large bright areas such as banners.
2. Candidate lines: standard Hough transform on the line pixels, split into
   near-horizontal (baselines, service lines) and oblique (sidelines) lines.
3. Model fitting: every pair of horizontal and pair of oblique candidates,
   assigned in order to two horizontal and two vertical lines of the court model
   (doubles court: 4 horizontal lines, 4 sidelines), defines a homography. Each
   one is scored by projecting the WHOLE court model and counting how much of it
   lands on line pixels. A hypothesis built on a banner edge or on writing next
   to the court projects the rest of the court onto empty ground and scores low.
4. Refinement: each model line is refitted on the line pixels around its
   projection, and the homography is re-estimated from all line intersections.
5. Several frames are tried (the first one may be a black fade-in or occluded),
   and the best-scoring fit is kept. If no frame matches the model well enough,
   the run fails instead of saving wrong keypoints.

The 8 labelled singles keypoints (TL, TR, BL, BR, STL, STR, SBL, SBR) are then
projected from the fitted model and saved to
outputs/<video name>/court_coordinates/<video name>_court.csv, together with an overlay image
(<video name>_court.png) to check the result by eye.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Make the project root importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_geometry import _REAL_WORLD, ALLEY, SVC_B, SVC_T, L_m, W_m
from utils.court_overlay import MODEL_SEGMENTS, draw_court_overlay
from utils.paths import clip_paths
from utils.video import safe_fps

# ── Court model (metres, origin at the far-left singles corner) ───────────────
H_LINES = np.array([0.0, SVC_T, SVC_B, L_m])        # far baseline ... near baseline
V_LINES = np.array([-ALLEY, 0.0, W_m, W_m + ALLEY])  # doubles L, singles L/R, doubles R
MODEL_STEP_M = 0.2                                   # sampling step along model lines

# ── Line-pixel detector (values for 1080p, scaled with the frame height) ──────
LINE_TAU_PX = 8          # max half-width of a line
LINE_MIN_LUMA = 110      # min brightness of a line pixel
LINE_MIN_CONTRAST = 18   # min brightness gap to the neighbours at +-tau
LINE_MAX_SAT = 170       # court lines are white: reject saturated colours (on clay
                         # they are dusty orange: 90 lost half of them)

# ── Hough candidates ──────────────────────────────────────────────────────────
HOUGH_VOTES_FRAC = 0.12  # accumulator threshold as a fraction of frame height
MAX_CANDIDATES = 12      # per orientation (hypotheses grow as n^4)
HORIZONTAL_DEG = 20      # lines within this angle of horizontal are "horizontal"

# ── Fit acceptance ────────────────────────────────────────────────────────────
MATCH_RADIUS_PX = 3      # a model point "hits" a line pixel within this distance
MIN_MATCH_RATIO = 0.6    # min fraction of visible model points on line pixels
GOOD_MATCH_RATIO = 0.9   # stop trying further frames once this is reached
MIN_MODEL_POINTS = 200   # min model points inside the frame (court big enough)
MIN_COURT_HEIGHT = 0.15  # baseline to baseline, at least this fraction of the frame
                         # height: a model squeezed onto the net tape or a banner
                         # matches a white line, but is no court
FRAMES_TO_TRY = 6        # frames sampled over the first half of the clip

# ── Playback ──────────────────────────────────────────────────────────────────
PLAYBACK_DELAY_BASE_MS = 500   # per-frame waitKey delay is base / fps (ms)


def _sample_model(step=MODEL_STEP_M):
    pts = []
    for (x0, y0), (x1, y1) in MODEL_SEGMENTS:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
        t = np.linspace(0.0, 1.0, n)
        pts.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0)])
    return np.vstack(pts)


MODEL_POINTS = _sample_model()


# ── 1. Line pixels ────────────────────────────────────────────────────────────

def line_pixel_mask(frame):
    """uint8 mask (255 = court-line pixel) of thin, bright, unsaturated structures."""
    h = frame.shape[0]
    scale = h / 1080
    tau = max(3, int(round(LINE_TAU_PX * scale)))
    luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1]

    def brighter_than(shift, axis):
        return (luma - np.roll(luma, shift, axis=axis)) > LINE_MIN_CONTRAST

    horizontal = brighter_than(tau, 1) & brighter_than(-tau, 1)
    vertical = brighter_than(tau, 0) & brighter_than(-tau, 0)
    mask = (luma > LINE_MIN_LUMA) & (sat < LINE_MAX_SAT) & (horizontal | vertical)
    # np.roll wraps around: drop the borders it contaminates.
    mask[:tau], mask[-tau:], mask[:, :tau], mask[:, -tau:] = False, False, False, False
    return mask.astype(np.uint8) * 255


# ── 2. Candidate lines ────────────────────────────────────────────────────────

def _line_angle_deg(line):
    a, b, _ = line
    return np.degrees(np.arctan2(b, a)) % 180


def candidate_lines(mask):
    """Strongest distinct Hough lines as (horizontal, oblique) lists of (a, b, c),
    with a x + b y = c. Horizontal ones are sorted top to bottom, oblique ones
    left to right."""
    h, w = mask.shape
    found = cv2.HoughLines(mask, 1, np.pi / 360, threshold=int(HOUGH_VOTES_FRAC * h))
    if found is None:
        return [], []
    horizontal, oblique = [], []
    for rho, theta in found[:, 0, :]:          # strongest first
        line = (float(np.cos(theta)), float(np.sin(theta)), float(rho))
        is_horizontal = abs(np.degrees(theta) - 90) < HORIZONTAL_DEG
        bucket = horizontal if is_horizontal else oblique
        if len(bucket) >= MAX_CANDIDATES or _is_duplicate(line, bucket, is_horizontal, w, h):
            continue
        bucket.append(line)
        if len(horizontal) >= MAX_CANDIDATES and len(oblique) >= MAX_CANDIDATES:
            break
    horizontal.sort(key=lambda ln: (ln[2] - ln[0] * w / 2) / ln[1])  # y at mid-width
    oblique.sort(key=lambda ln: (ln[2] - ln[1] * 0.6 * h) / ln[0])  # x at 0.6 h
    return horizontal, oblique


def _is_duplicate(line, kept, is_horizontal, w, h, max_deg=2.5, max_px=12):
    a, b, c = line
    for a2, b2, c2 in kept:
        d_angle = abs(_line_angle_deg(line) - _line_angle_deg((a2, b2, c2)))
        if min(d_angle, 180 - d_angle) > max_deg:
            continue
        if is_horizontal:
            gap = abs((c - a * w / 2) / b - (c2 - a2 * w / 2) / b2)
        else:
            gap = abs((c - b * h / 2) / a - (c2 - b2 * h / 2) / a2)
        if gap < max_px:
            return True
    return False


def _intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return np.array([(c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det])


# ── 3. Model fitting ──────────────────────────────────────────────────────────

def _square_to_quad(quads):
    """Homographies mapping the unit square (0,0),(1,0),(1,1),(0,1) onto each
    quad of `quads` (N, 4, 2), in closed form (Heckbert, 1989)."""
    x0, y0 = quads[:, 0, 0], quads[:, 0, 1]
    x1, y1 = quads[:, 1, 0], quads[:, 1, 1]
    x2, y2 = quads[:, 2, 0], quads[:, 2, 1]
    x3, y3 = quads[:, 3, 0], quads[:, 3, 1]
    sx, sy = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    dx1, dx2, dy1, dy2 = x1 - x2, x3 - x2, y1 - y2, y3 - y2
    den = dx1 * dy2 - dx2 * dy1
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    g = (sx * dy2 - dx2 * sy) / den
    k = (dx1 * sy - sx * dy1) / den
    H = np.zeros((len(quads), 3, 3))
    H[:, 0] = np.c_[x1 - x0 + g * x1, x3 - x0 + k * x3, x0]
    H[:, 1] = np.c_[y1 - y0 + g * y1, y3 - y0 + k * y3, y0]
    H[:, 2] = np.c_[g, k, np.ones_like(g)]
    return H


def _hypotheses(horizontal, oblique, w, h):
    """All (image quad, model line indices) pairs that pass basic sanity checks,
    as homographies model metres -> image pixels."""
    nh, nv = len(horizontal), len(oblique)
    X = np.full((nh, nv, 2), np.nan)
    for i, lh in enumerate(horizontal):
        for j, lv in enumerate(oblique):
            p = _intersect(lh, lv)
            if p is not None and -0.3 * w < p[0] < 1.3 * w and -0.3 * h < p[1] < 1.3 * h:
                X[i, j] = p
    min_gap = 20 * h / 1080
    quads = []
    for i in range(nh):
        for i2 in range(i + 1, nh):
            for j in range(nv):
                for j2 in range(j + 1, nv):
                    q = np.array([X[i, j], X[i, j2], X[i2, j2], X[i2, j]])
                    if np.isnan(q).any():
                        continue
                    if q[1, 0] - q[0, 0] < min_gap or q[2, 0] - q[3, 0] < min_gap:
                        continue
                    if min(q[2, 1], q[3, 1]) - max(q[0, 1], q[1, 1]) < min_gap:
                        continue
                    quads.append(q)
    if not quads:
        return np.empty((0, 3, 3))
    # Each image quad against each ordered choice of 2 horizontal x 2 vertical
    # model lines (same top-to-bottom / left-to-right order as in the image).
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    model = np.array([(a, b, c, d) for a, b in pairs for c, d in pairs])
    quads = np.repeat(np.array(quads), len(model), axis=0)
    model = np.tile(model, (len(quads) // len(model), 1))
    ya, yb = H_LINES[model[:, 0]], H_LINES[model[:, 1]]
    xa, xb = V_LINES[model[:, 2]], V_LINES[model[:, 3]]
    to_unit = np.zeros((len(model), 3, 3))       # model rectangle -> unit square
    to_unit[:, 0, 0] = 1 / (xb - xa)
    to_unit[:, 0, 2] = -xa / (xb - xa)
    to_unit[:, 1, 1] = 1 / (yb - ya)
    to_unit[:, 1, 2] = -ya / (yb - ya)
    to_unit[:, 2, 2] = 1
    return _square_to_quad(quads) @ to_unit


def _score(Hs, dist, radius):
    """(score, model points on lines, model points in frame) per homography."""
    h, w = dist.shape
    P = np.c_[MODEL_POINTS, np.ones(len(MODEL_POINTS))].T
    proj = Hs @ P
    z = proj[:, 2]
    in_front = z > 1e-6
    x = proj[:, 0] / np.where(in_front, z, 1)
    y = proj[:, 1] / np.where(in_front, z, 1)
    inside = in_front & (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    xi = np.clip(x, 0, w - 1).astype(np.int32)
    yi = np.clip(y, 0, h - 1).astype(np.int32)
    hit = inside & (dist[yi, xi] <= radius)
    n_in, n_hit = inside.sum(1), hit.sum(1)
    return n_hit - 0.5 * (n_in - n_hit), n_hit, n_in


def fit_court_model(mask, chunk=3000):
    """Best homography (court metres -> pixels) for a line-pixel mask, with its
    match ratio, or (None, 0.0) when no hypothesis is plausible."""
    h, w = mask.shape
    horizontal, oblique = candidate_lines(mask)
    if len(horizontal) < 2 or len(oblique) < 2:
        return None, 0.0
    Hs = _hypotheses(horizontal, oblique, w, h)
    if not len(Hs):
        return None, 0.0
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    radius = max(2.0, MATCH_RADIUS_PX * h / 1080)
    best = (-np.inf, None, 0, 0)
    for s in range(0, len(Hs), chunk):
        score, n_hit, n_in = _score(Hs[s:s + chunk], dist, radius)
        score = np.where(_court_height(Hs[s:s + chunk]) >= MIN_COURT_HEIGHT * h, score, -np.inf)
        k = int(np.argmax(score))
        if score[k] > best[0]:
            best = (score[k], Hs[s + k], n_hit[k], n_in[k])
    score, H, n_hit, n_in = best
    if not np.isfinite(score) or n_in < MIN_MODEL_POINTS:
        return None, 0.0
    return H, n_hit / n_in


def _court_height(Hs):
    """Image height (px) from the far to the near baseline, per homography."""
    P = np.array([[W_m / 2, 0.0, 1.0], [W_m / 2, L_m, 1.0]]).T
    proj = Hs @ P
    z = proj[:, 2]
    y = proj[:, 1] / np.where(np.abs(z) > 1e-9, z, 1e-9)
    ok = (z > 1e-6).all(axis=1)
    return np.where(ok, np.abs(y[:, 1] - y[:, 0]), 0.0)


# ── 4. Refinement ─────────────────────────────────────────────────────────────

def _model_line(seg):
    (x0, y0), (x1, y1) = seg
    a, b = y0 - y1, x1 - x0
    return (a, b, a * x0 + b * y0)


def refine_homography(H, mask, iterations=3):
    """Refit every model line on the line pixels around its projection and
    re-estimate H from the intersections of the refitted lines."""
    h = mask.shape[0]
    ys, xs = np.nonzero(mask)
    pixels = np.c_[xs, ys].astype(np.float64)
    band = max(4.0, 6.0 * h / 1080)
    for _ in range(iterations):
        fitted = {}
        for k, seg in enumerate(MODEL_SEGMENTS):
            p0, p1 = cv2.perspectiveTransform(np.array([[seg[0]], [seg[1]]], np.float64), H)[:, 0]
            length = np.hypot(*(p1 - p0))
            if length < 1:
                continue
            u = (p1 - p0) / length
            rel = pixels - p0
            along = rel @ u
            across = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
            near = (across < band) & (along > -5) & (along < length + 5)
            if near.sum() < 30:
                continue
            vx, vy, cx, cy = cv2.fitLine(pixels[near].astype(np.float32),
                                         cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            fitted[k] = (vy, -vx, vy * cx - vx * cy)
        src, dst = [], []
        for i in fitted:
            for j in fitted:
                if j <= i:
                    continue
                m = _intersect(_model_line(MODEL_SEGMENTS[i]), _model_line(MODEL_SEGMENTS[j]))
                p = _intersect(fitted[i], fitted[j])
                if m is not None and p is not None:
                    src.append(m)
                    dst.append(p)
        if len(src) < 4:
            break
        H_new, _ = cv2.findHomography(np.array(src), np.array(dst), 0)
        if H_new is None:
            break
        H = H_new
    return H


def detect_court(frame):
    """(H court metres -> pixels, match ratio) for one frame; H is None on failure."""
    mask = line_pixel_mask(frame)
    H, ratio = fit_court_model(mask)
    if H is None:
        return None, 0.0
    return refine_homography(H, mask), ratio


def keypoints_from_homography(H):
    """The 8 labelled singles keypoints {label: (x, y)} in pixels."""
    labels = list(_REAL_WORLD)
    pts = np.array([[_REAL_WORLD[k]] for k in labels], dtype=np.float64)
    px = cv2.perspectiveTransform(pts, H)[:, 0]
    return {k: (float(x), float(y)) for k, (x, y) in zip(labels, px)}


def save_court(H, frame, csv_path):
    """Write the 8 keypoints of H to `csv_path` and an overlay PNG next to it."""
    keypoints = keypoints_from_homography(H)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "x", "y"])
        for label in ("TL", "TR", "BL", "BR", "STL", "STR", "SBL", "SBR"):
            x, y = keypoints[label]
            writer.writerow([label, f"{x:.1f}", f"{y:.1f}"])
    overlay, _ = draw_court_overlay(frame, csv_path)
    cv2.imwrite(os.path.splitext(csv_path)[0] + ".png", overlay)
    return keypoints


def court_match_ratio(frame, H):
    """Share of the visible court model of H lying on the line pixels of `frame`.

    The cheap check of a known homography (one mask, one projection): the live
    pipeline runs it now and then to notice that the camera moved.
    """
    mask = line_pixel_mask(frame)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    radius = max(2.0, MATCH_RADIUS_PX * frame.shape[0] / 1080)
    _, n_hit, n_in = _score(np.asarray(H, float)[None], dist, radius)
    return float(n_hit[0] / n_in[0]) if n_in[0] else 0.0


# ── Court tracker ─────────────────────────────────────────────────────────────

class CourtTracker:
    """Detect the court on a few frames of the video and save its keypoints."""

    def __init__(self, video_path, no_display=False, output_dir="outputs",
                 frames_to_try=FRAMES_TO_TRY):
        self.video_path = video_path
        self.no_display = no_display
        self.output_dir = output_dir
        self.frames_to_try = frames_to_try

    def _best_fit(self, cap):
        """(H, ratio, frame index, frame) of the best fit over the sampled frames."""
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last = max(1, total // 2)
        indices = sorted({int(round(i * last / max(1, self.frames_to_try - 1)))
                          for i in range(self.frames_to_try)})
        best = (None, 0.0, -1, None)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            H, ratio = detect_court(frame)
            print(f"  frame {idx:5d}: " + (f"{ratio:.0%} of the court model on detected lines"
                                           if H is not None else "no court found"))
            if H is not None and ratio > best[1]:
                best = (H, ratio, idx, frame)
            if best[1] >= GOOD_MATCH_RATIO:
                break
        return best

    def run(self):
        """Detect the court, write the 8 keypoints to CSV plus an overlay image,
        and (unless no_display) play the video with the court drawn on it.
        Returns the keypoint dict."""
        csv_path = clip_paths(self.video_path, self.output_dir)["court_csv"]

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        try:
            print(f"Fitting the court model on {self.video_path}...")
            H, ratio, frame_idx, frame = self._best_fit(cap)
            if H is None or ratio < MIN_MATCH_RATIO:
                raise RuntimeError(
                    f"Court not found in {self.video_path}: best fit matched only "
                    f"{ratio:.0%} of the court model (need {MIN_MATCH_RATIO:.0%}). "
                    "The clip needs a fixed broadcast view of the whole court.")

            keypoints = save_court(H, frame, csv_path)
            print(f"Court fitted on frame {frame_idx} ({ratio:.0%} match). "
                  f"Saved keypoints to '{csv_path}' and its overlay next to it.")

            if not self.no_display:
                self._play_back(cap, csv_path)
            return keypoints
        finally:
            cap.release()
            if not self.no_display:     # headless OpenCV builds raise here
                cv2.destroyAllWindows()

    def _play_back(self, cap, csv_path):
        """Play the video with the fitted court drawn on every frame; q quits."""
        delay = max(1, int(PLAYBACK_DELAY_BASE_MS / safe_fps(cap.get(cv2.CAP_PROP_FPS))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            overlay, _ = draw_court_overlay(frame, csv_path)
            cv2.imshow("Court detection (q to quit)", cv2.resize(overlay, (1280, 720)))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Court keypoint detection by fitting the ITF court model to "
                    "the white court lines")
    parser.add_argument("--video", default="data/Input_video2.mp4")
    parser.add_argument("--no-display", action="store_true",
                        help="skip the video playback with the fitted court")
    parser.add_argument("--output", default="outputs", dest="output_dir",
                        help="base output directory (default: outputs)")
    args = parser.parse_args()

    CourtTracker(video_path=args.video, no_display=args.no_display,
                 output_dir=args.output_dir).run()
