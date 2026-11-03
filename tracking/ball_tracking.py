"""Ball tracking: YOLO candidates at full resolution, then a physical tracker.

Stage 1 -- detection (needs ultralytics + the YOLOv8 weights; GPU recommended).
    Each frame is split into overlapping TILE x TILE tiles at native
    resolution, and the detector runs on all tiles plus the whole frame in one
    batch. Whole-frame inference shrinks a 1080p frame to 640 px and the ~15 px
    ball to ~5 px; on the tiles it keeps its real size. The whole frame is kept
    too, because the model was trained on whole downscaled frames and some
    (motion-blurred) balls are only found that way. Duplicates are merged with
    NMS, and every box above CANDIDATE_CONF is kept,
    with the fraction of its pixels that moved since the previous frame, in
    outputs/ball_coordinates/ball_candidates_<video name>.csv.

Stage 2 -- tracking (CPU, no model needed).
    Still candidates are discarded (as the original motion gate did). A
    constant-velocity Kalman filter follows the ball through the rest: a
    candidate can only continue the track if it is physically reachable from
    the last position (MAX_SPEED_PX); among those, confidence, motion and
    agreement with the predicted position decide. The track restarts from the
    strongest moving candidate after LOST_AFTER_S without the ball. Isolated
    detections that jump away and back are dropped, and gaps up to MAX_GAP_S are
    filled with a shape-preserving cubic (PCHIP) through the real detections;
    longer gaps and the clip edges are left empty.

Output: outputs/ball_coordinates/ball_<video name>.csv with
frame, x, y, w, h, cx, cy, area, interpolated.

Usage (from project root):
    python tracking/ball_tracking.py --video data/Input_video2.mp4             # both stages
    python tracking/ball_tracking.py --video data/Input_video2.mp4 --detect-only   # stage 1 (Colab)
    python tracking/ball_tracking.py --video data/Input_video2.mp4 \\
        --candidates outputs/ball_coordinates/ball_candidates_Input_video2.csv    # stage 2 only
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Make the project root importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.video import check_full_read, safe_fps

# ── Stage 1: detection ────────────────────────────────────────────────────────
TILE = 640                 # tile size = the detector's input size: no downscaling
TILE_MIN_OVERLAP = 48      # px; > 3x the ball, so a ball on a seam is whole in a tile
CANDIDATE_CONF = 0.15      # keep every box above this; the tracker decides later
NMS_IOU = 0.3              # merge duplicates of the same ball from overlapping tiles
MOTION_THRESH = 15         # per-pixel intensity change (0-255) that counts as motion

# ── Stage 2: tracking (pixel values for 1080p, scaled with the frame height) ──
MIN_TRACK_CONF = 0.25      # candidates below this are ignored by the tracker
MIN_MOTION = 0.05          # ...and so are still ones (balls held by ball kids or in
                           # pockets): at least this fraction of the box must move
ACQUIRE_CONF = 0.5         # starting a track needs a confident candidate
MAX_SPEED_PX = 150         # per frame: farther candidates cannot be the same ball
SPIKE_SPEED_PX = 80        # a detection this fast in and out...
SPIKE_DETOUR = 3.0         # ...on a path this much longer than its neighbours' gap
                           # is a one-frame false positive
LOST_AFTER_S = 0.33        # without the ball this long, the track restarts
ACCEL_STD_PX = 12          # Kalman process noise: hits and bounces change velocity
MEAS_STD_PX = 3            # Kalman measurement noise: detection jitter
MAX_GAP_S = 0.33           # gaps up to this long are filled; longer ones stay empty

CANDIDATE_COLUMNS = ["frame", "x1", "y1", "x2", "y2", "conf", "motion"]
BALL_COLUMNS = ["frame", "x", "y", "w", "h", "cx", "cy", "area", "interpolated"]


# ═════════════════════════════ Stage 1: detection ═════════════════════════════

def tile_origins(width, height, tile=TILE, min_overlap=TILE_MIN_OVERLAP):
    """Top-left corners of tile x tile windows covering a width x height frame."""
    def axis(length):
        if length <= tile:
            return [0]
        n = int(np.ceil((length - tile) / (tile - min_overlap))) + 1
        return [int(round(v)) for v in np.linspace(0, length - tile, n)]
    return [(x, y) for y in axis(height) for x in axis(width)]


def motion_fraction(prev_gray, gray, box):
    """Fraction of the box pixels whose intensity changed since the previous frame."""
    h, w = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if prev_gray is None or x2 <= x1 or y2 <= y1:
        return 0.0
    diff = cv2.absdiff(prev_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2])
    return float((diff > MOTION_THRESH).mean())


class BallDetector:
    """Runs the YOLO ball detector on full-resolution tiles of every frame."""

    def __init__(self, model_path):
        from ultralytics import YOLO  # only stage 1 needs ultralytics
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        """[(x1, y1, x2, y2, conf), ...] for one frame: tiles + whole frame, merged."""
        h, w = frame.shape[:2]
        origins = tile_origins(w, h)
        tiles = [frame[y:y + TILE, x:x + TILE] for x, y in origins]
        if len(origins) > 1:                 # the whole frame, downscaled as in training
            tiles.append(frame)
            origins.append((0, 0))
        results = self.model.predict(tiles, imgsz=TILE, conf=CANDIDATE_CONF, verbose=False)
        boxes, scores = [], []
        for (ox, oy), r in zip(origins, results):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                boxes.append([float(x1 + ox), float(y1 + oy), float(x2 - x1), float(y2 - y1)])
                scores.append(float(c))
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, CANDIDATE_CONF, NMS_IOU)
        return [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2],
                 boxes[i][1] + boxes[i][3], scores[i]) for i in np.array(keep).ravel()]

    def detect_candidates(self, video_path, out_csv):
        """Write every candidate of every frame to `out_csv`; return the frame count."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Detecting ball candidates in {total or '?'} frames "
              f"({len(tile_origins(int(cap.get(3)), int(cap.get(4))))} tiles + the whole frame, per frame)...")
        rows, prev_gray, idx = [], None, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for x1, y1, x2, y2, conf in self.detect_frame(frame):
                    rows.append([idx, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                                 round(conf, 4),
                                 round(motion_fraction(prev_gray, gray, (x1, y1, x2, y2)), 3)])
                prev_gray = gray
                idx += 1
                if idx % 100 == 0:
                    print(f"  {idx} frames")
        finally:
            cap.release()
        check_full_read(idx, total, video_path)
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(CANDIDATE_COLUMNS)
            wr.writerows(rows)
        print(f"Saved {len(rows)} candidates to {out_csv}")
        return idx


# ═════════════════════════════ Stage 2: tracking ══════════════════════════════

class _Kalman:
    """Constant-velocity Kalman filter on the ball centre (pixels, 1 step = 1 frame)."""

    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def __init__(self, xy, accel_std, meas_std):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([meas_std ** 2, meas_std ** 2, 50.0 ** 2, 50.0 ** 2])
        # Piecewise-constant random acceleration: Q = G G^T sigma_a^2.
        G = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]])
        self.Q = G @ G.T * accel_std ** 2
        self.R = np.eye(2) * meas_std ** 2

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, xy):
        y = np.asarray(xy, float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


def track_candidates(cands, n_frames, fps, frame_height=1080):
    """Pick the ball among per-frame candidates.

    cands: DataFrame with CANDIDATE_COLUMNS. Returns {frame: (x1, y1, x2, y2)}
    of the real detections kept by the tracker.
    """
    s = frame_height / 1080
    max_speed = MAX_SPEED_PX * s
    lost_after = max(2, int(round(LOST_AFTER_S * fps)))
    cands = cands[(cands.conf >= MIN_TRACK_CONF) & (cands.motion >= MIN_MOTION)]
    by_frame = {f: g.to_numpy() for f, g in cands.groupby("frame")}

    track = {}
    kf, last_xy, last_frame = None, None, None
    for f in range(n_frames):
        rows = by_frame.get(f)
        missed = f - last_frame if last_frame is not None else None
        if kf is not None and missed > lost_after:
            kf = None                                   # lost: restart from scratch
        pred = kf.predict() if kf is not None else None
        if rows is None:
            continue
        centres = np.c_[(rows[:, 1] + rows[:, 3]) / 2, (rows[:, 2] + rows[:, 4]) / 2]
        conf, motion = rows[:, 5], rows[:, 6]
        if kf is None:
            ok = conf >= ACQUIRE_CONF
            if not ok.any():
                continue
            k = int(np.argmax(np.where(ok, conf, -1)))
            kf = _Kalman(centres[k], ACCEL_STD_PX * s, MEAS_STD_PX * s)
        else:
            # Physically reachable from the last detection...
            reach = np.hypot(*(centres - last_xy).T) <= max_speed * missed
            if not reach.any():
                continue
            # ...then prefer confident candidates near the prediction.
            spread = max(20 * s, np.sqrt(np.trace(kf.P[:2, :2])))
            d_pred = np.hypot(*(centres - pred).T) / spread
            score = np.where(reach, conf + 0.2 * motion - 0.5 * d_pred, -np.inf)
            k = int(np.argmax(score))
            kf.update(centres[k])
        track[f] = tuple(rows[k, 1:5])
        last_xy, last_frame = centres[k], f
    return track


def drop_outliers(track, frame_height=1080, window=3, passes=3):
    """Remove isolated detections that jump away and straight back.

    A detection f between neighbours a and b (each within `window` frames) is
    dropped when it is physically unreachable from both while a and b agree,
    or when it is a fast spike: faster than SPIKE_SPEED_PX on both sides with
    an out-and-back path more than SPIKE_DETOUR times the direct a -> b
    distance. A racket contact is also V-shaped, but far slower.
    """
    s = frame_height / 1080
    max_speed, spike_speed = MAX_SPEED_PX * s, SPIKE_SPEED_PX * s
    track = dict(track)
    for _ in range(passes):                   # dropping one may expose the next
        frames = sorted(track)
        centre = {f: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                  for f, b in track.items()}

        def dist(u, v, centre=centre):
            return float(np.hypot(*(centre[v] - centre[u])))

        drop = set()
        for a, f, b in zip(frames, frames[1:], frames[2:]):
            if f - a > window or b - f > window:
                continue
            v_af, v_fb, v_ab = dist(a, f) / (f - a), dist(f, b) / (b - f), dist(a, b) / (b - a)
            unreachable = v_af > max_speed and v_fb > max_speed and v_ab <= max_speed
            spike = (v_af > spike_speed and v_fb > spike_speed
                     and dist(a, f) + dist(f, b) > SPIKE_DETOUR * max(dist(a, b), 1.0))
            if unreachable or spike:
                drop.add(f)
        if not drop:
            break
        track = {f: b for f, b in track.items() if f not in drop}
    return track


def fill_gaps(track, fps):
    """Rows (frame, x1, y1, x2, y2, interpolated) with short gaps PCHIP-filled."""
    frames = np.array(sorted(track))
    if len(frames) == 0:
        return []
    boxes = np.array([track[f] for f in frames], float)
    rows = [(int(f), *boxes[i], 0) for i, f in enumerate(frames)]
    if len(frames) < 2:
        return rows
    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    cx = PchipInterpolator(frames, (boxes[:, 0] + boxes[:, 2]) / 2)
    cy = PchipInterpolator(frames, (boxes[:, 1] + boxes[:, 3]) / 2)
    for a, b, ba, bb in zip(frames[:-1], frames[1:], boxes[:-1], boxes[1:]):
        if not 0 < b - a - 1 <= max_gap:
            continue
        for f in range(a + 1, b):
            t = (f - a) / (b - a)
            w = ba[2] - ba[0] + t * ((bb[2] - bb[0]) - (ba[2] - ba[0]))
            h = ba[3] - ba[1] + t * ((bb[3] - bb[1]) - (ba[3] - ba[1]))
            x, y = float(cx(f)), float(cy(f))
            rows.append((f, x - w / 2, y - h / 2, x + w / 2, y + h / 2, 1))
    rows.sort()
    return rows


def write_ball_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(BALL_COLUMNS)
        for frame, x1, y1, x2, y2, interp in rows:
            x, y, w, h = int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))
            wr.writerow([frame, x, y, w, h, round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
                         w * h, interp])


def track_from_candidates(candidates_csv, video_path, out_csv):
    """Stage 2: candidates CSV -> ball CSV. Returns a short summary dict."""
    cap = cv2.VideoCapture(video_path)
    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    cands = pd.read_csv(candidates_csv)
    n_frames = max(n_frames, int(cands.frame.max()) + 1 if len(cands) else 0)

    track = track_candidates(cands, n_frames, fps, height)
    clean = drop_outliers(track, height)
    rows = fill_gaps(clean, fps)
    write_ball_csv(rows, out_csv)
    summary = {"frames": n_frames, "detected": len(clean),
               "outliers_dropped": len(track) - len(clean),
               "interpolated": sum(r[5] for r in rows)}
    print(f"Ball found on {summary['detected']} of {n_frames} frames "
          f"({summary['detected'] / max(n_frames, 1):.0%}), "
          f"{summary['outliers_dropped']} outliers dropped, "
          f"{summary['interpolated']} gap frames filled -> {out_csv}")
    return summary


def render_video(video_path, ball_csv, out_path):
    """Write a copy of the video with the ball box drawn (real = green, filled = yellow)."""
    balls = pd.read_csv(ball_csv).set_index("frame")
    cap = cv2.VideoCapture(video_path)
    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    writer, idx = None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in balls.index:
                r = balls.loc[idx]
                color = (0, 255, 255) if r.interpolated else (0, 255, 0)
                cv2.rectangle(frame, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), color, 2)
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print(f"Saved annotated video to {out_path}")


def default_paths(video, output_dir="outputs"):
    stem = os.path.splitext(os.path.basename(video))[0]
    base = os.path.join(output_dir, "ball_coordinates")
    return (os.path.join(base, f"ball_candidates_{stem}.csv"),
            os.path.join(base, f"ball_{stem}.csv"))


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "ball_tracker.pt")

    parser = argparse.ArgumentParser(
        description="Ball tracking: YOLO candidates on full-resolution tiles, then a "
                    "Kalman tracker with outlier removal and gap filling")
    parser.add_argument("--video", default="data/Input_video2.mp4", help="input video")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="YOLO ball-detector weights (.pt), needed by stage 1")
    parser.add_argument("--detect-only", action="store_true", dest="detect_only",
                        help="run stage 1 only and write the candidates CSV")
    parser.add_argument("--candidates", default=None,
                        help="existing candidates CSV: run stage 2 only (no model needed)")
    parser.add_argument("--csv", default=None,
                        help="output ball CSV (default: outputs/ball_coordinates/ball_<video>.csv)")
    parser.add_argument("--video-out", default=None, dest="video_out",
                        help="also write an annotated copy of the video here")
    args = parser.parse_args()

    cand_csv, ball_csv = default_paths(args.video)
    ball_csv = args.csv or ball_csv
    if args.candidates:
        cand_csv = args.candidates
    else:
        if not os.path.exists(args.model):
            raise SystemExit(f"Model not found: {args.model}\n"
                             "Place the YOLO weights there, pass --model, or run stage 2 "
                             "only with --candidates <csv>.")
        BallDetector(args.model).detect_candidates(args.video, cand_csv)
    if not args.detect_only:
        track_from_candidates(cand_csv, args.video, ball_csv)
        if args.video_out:
            render_video(args.video, ball_csv, args.video_out)
