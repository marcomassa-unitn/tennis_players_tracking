"""Visual check of the court keypoints and of the homography built from them.

Draws, on a frame of the original video:
  - the 8 detected keypoints (red), labelled;
  - every painted line of the ITF court model (yellow), doubles alleys
    included, reprojected into the image through the homography fitted on
    those 8 keypoints.

If the yellow lines sit on the painted white lines -- including the doubles
sidelines and the centre service line, which the 8 singles keypoints do not
pin down -- the keypoints and the homography are correct. The reprojection
error only measures how consistent the 8 points are with each other, so the
visual check is the one that matters.

Usage (from project root, after court_tracking.py has written the CSV):
    python utils/court_overlay.py --video data/Input_video2.mp4
    python utils/court_overlay.py --video data/input4.mp4 --frame 100

Without --frame, the first frame that is not black is used.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter
from utils.court_geometry import _REAL_WORLD, ALLEY, CL_X, SVC_B, SVC_T, L_m, W_m

# Painted lines of the court in metres (origin at the far-left singles corner).
# Shared with tracking/court_tracking.py, which fits exactly this model.
MODEL_SEGMENTS = [
    ((-ALLEY, 0.0), (W_m + ALLEY, 0.0)),    # far baseline
    ((-ALLEY, L_m), (W_m + ALLEY, L_m)),    # near baseline
    ((-ALLEY, 0.0), (-ALLEY, L_m)),         # left doubles sideline
    ((W_m + ALLEY, 0.0), (W_m + ALLEY, L_m)),  # right doubles sideline
    ((0.0, 0.0), (0.0, L_m)),               # left singles sideline
    ((W_m, 0.0), (W_m, L_m)),               # right singles sideline
    ((0.0, SVC_T), (W_m, SVC_T)),           # far service line
    ((0.0, SVC_B), (W_m, SVC_B)),           # near service line
    ((CL_X, SVC_T), (CL_X, SVC_B)),         # centre service line
]

_LINE_COLOR = (0, 230, 255)    # yellow (BGR)
_POINT_COLOR = (0, 0, 255)     # red


def _to_px(H_inv, pts_m):
    """Project (N, 2) court-metre points into the image."""
    pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)


def _detected_keypoints(court_csv):
    import csv
    with open(court_csv, newline="") as f:
        return {r["label"].strip(): (float(r["x"]), float(r["y"]))
                for r in csv.DictReader(f)}


def draw_court_overlay(frame, court_csv):
    """Return (annotated frame, per-label reprojection errors in px)."""
    H_inv = np.linalg.inv(CourtConverter(court_csv)._H)
    detected = _detected_keypoints(court_csv)
    out = frame.copy()
    scale = max(1.0, out.shape[1] / 1280)          # keep strokes visible on 4K
    thick = int(round(2 * scale))

    for a, b in MODEL_SEGMENTS:
        p, q = _to_px(H_inv, [a, b])
        cv2.line(out, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)),
                 _LINE_COLOR, thick, cv2.LINE_AA)

    errors = {}
    for label, (x, y) in detected.items():
        if label in _REAL_WORLD:
            rx, ry = _to_px(H_inv, [_REAL_WORLD[label]])[0]
            errors[label] = float(np.hypot(rx - x, ry - y))
        c = (int(round(x)), int(round(y)))
        cv2.circle(out, c, int(8 * scale), _POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, c, int(8 * scale), (255, 255, 255), thick, cv2.LINE_AA)
        off = (-int(60 * scale), -int(14 * scale)) if x < out.shape[1] / 2 else (int(14 * scale), -int(14 * scale))
        org = (c[0] + off[0], c[1] + off[1])
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_DUPLEX, 0.8 * scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_DUPLEX, 0.8 * scale, (255, 255, 255), thick, cv2.LINE_AA)

    if errors:
        vals = list(errors.values())
        text = (f"red = detected keypoints, yellow = court model via homography | "
                f"reprojection error: mean {np.mean(vals):.1f}px, max {np.max(vals):.1f}px")
        org = (int(12 * scale), out.shape[0] - int(14 * scale))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out, errors


def _read_frame(video_path, frame_idx=None, min_brightness=40):
    """Frame `frame_idx`, or the first frame that is not black (fade-ins)."""
    cap = cv2.VideoCapture(video_path)
    try:
        if frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            return frame if ok else None
        frame = None
        for _ in range(300):
            ok, candidate = cap.read()
            if not ok:
                break
            frame = candidate
            if frame.mean() >= min_brightness:
                break
        return frame
    finally:
        cap.release()


def save_court_overlay(video_path, court_csv, out_path, frame_idx=None):
    """Draw the overlay on a frame of the video, save it, report errors."""
    frame = _read_frame(video_path, frame_idx)
    if frame is None:
        raise RuntimeError(f"Cannot read a frame of {video_path}")

    image, errors = draw_court_overlay(frame, court_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, image)
    if errors:
        worst = max(errors, key=errors.get)
        print(f"Court overlay saved to '{out_path}' (keypoint reprojection error: "
              f"mean {np.mean(list(errors.values())):.1f}px, "
              f"max {errors[worst]:.1f}px at {worst}).")
    else:
        print(f"Court overlay saved to '{out_path}'.")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Draw detected court keypoints and the reprojected court model "
                    "on a video frame, to check the keypoints and the homography.")
    parser.add_argument("--video", default="data/Input_video2.mp4")
    parser.add_argument("--court", default=None,
                        help="court CSV (default: outputs/court_coordinates/<video name>_court.csv)")
    parser.add_argument("--frame", type=int, default=None,
                        help="frame to draw on (default: first non-black frame)")
    parser.add_argument("--output", default=None,
                        help="image path (default: next to the court CSV, <video name>_court.png)")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    court = args.court or os.path.join("outputs", "court_coordinates", f"{stem}_court.csv")
    if not os.path.exists(court):
        raise SystemExit(f"Court CSV not found: {court}\n"
                         f"Run: python tracking/court_tracking.py --video {args.video} --no-display")
    out = args.output or os.path.splitext(court)[0] + ".png"
    save_court_overlay(args.video, court, out, args.frame)


if __name__ == "__main__":
    main()
