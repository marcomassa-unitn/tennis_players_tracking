"""Ball tracking against the ground truth: offline, real-time, and the detector alone.

For every clip with a ball ground truth (ground_truth/ball_<clip>_gt.csv) it scores,
with evaluate_tracking.ball_metrics (a real detection within --radius px):

  offline     the offline tracker (YOLOv8x tiles + motion blobs + whole-clip path)
  wholeframe  the original single-pass detector
  realtime    the causal track of realtime_pipeline.py (fixed lag)
  rt_cands    the real-time candidates through the OFFLINE path search: tells the
              detector's share of a difference from the tracker's
  live        the zero-lag Kalman estimate that steers the crops
  oracle      with --model: the detector alone, on crops placed around the true
              ball (as in TRACKING), replayed with the online background

Recall and precision come with 95 % Wilson intervals: with ~100 labelled frames
per clip, a few points of difference are noise.

Usage (from project root):
    python evaluation/benchmark_ball.py
    python evaluation/benchmark_ball.py --model models/ball_motion.onnx --clips Input_video2
"""

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evaluate_tracking import ball_metrics, load_ball_gt, load_ball_pred
from tracking.ball_model import CROP, load_backend
from tracking.ball_online import CropMaker, court_roi
from tracking.ball_tracking import track_from_candidates
from tracking.player_tracking import PROC_HEIGHT
from training.build_ball_dataset import find_video
from utils.background import OnlineBackground, camera_shift
from utils.court_converter import CourtConverter
from utils.paths import GROUND_TRUTH_DIR, OUTPUT_ROOT, clip_paths
from utils.video import safe_fps


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score(gt, pred_csv, radius):
    if not pred_csv or not os.path.exists(pred_csv):
        return None
    s, _ = ball_metrics(gt, load_ball_pred(pred_csv), radius)
    s["recall_ci"] = wilson(s["tp"], s["tp"] + s["fn"])
    s["precision_ci"] = wilson(s["tp"], s["tp"] + s["fp"])
    return s


def oracle(video, gt, court_csv, backend, radius, seed=0):
    """The detector alone on crops around the true ball (and around nothing)."""
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(video)
    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = (int(round(W * PROC_HEIGHT / H)), PROC_HEIGHT)
    bg = OnlineBackground(fps, (W, H), proc, roi=court_roi(CourtConverter(court_csv), W, H),
                          threaded=False)
    crops = CropMaker(bg.scale)
    hann = cv2.createHanningWindow(proc, cv2.CV_32F)
    tp = fn = fp = neg = 0
    prev, prev_shift, f = None, (0.0, 0.0), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, proc, interpolation=cv2.INTER_AREA)
        snap = bg.snapshot
        shift = camera_shift(snap.small_gray, small, hann) if snap is not None else (0.0, 0.0)
        if snap is not None and prev is not None and f in gt:
            g = gt[f]
            dx, dy = shift[0] / crops.scale, shift[1] / crops.scale
            if g:
                ox, oy = rng.normal(0, 40, 2)
                x0, y0 = crops.clamp(snap, g[0] - dx + ox - CROP / 2, g[1] - dy + oy - CROP / 2)
            else:
                x0, y0 = crops.clamp(snap, rng.uniform(snap.roi[0], snap.roi[2]) - CROP / 2,
                                     rng.uniform(snap.roi[1], snap.roi[3]) - CROP / 2)
            img = crops.channels(snap, frame, prev, x0, y0, shift, prev_shift)
            dets = backend.detect([img])[0]
            best = max(dets, key=lambda d: d[4]) if dets else None
            if g:
                hit = best is not None and np.hypot((best[0] + best[2]) / 2 + x0 + dx - g[0],
                                                    (best[1] + best[3]) / 2 + y0 + dy - g[1]) <= radius
                tp, fn = tp + hit, fn + (not hit)
            else:
                neg += 1
                fp += best is not None
        bg.feed(f, frame, small, shift=shift)
        prev, prev_shift, f = frame, shift, f + 1
    cap.release()
    return {"recall": tp / max(tp + fn, 1), "recall_ci": wilson(tp, tp + fn), "tp": tp, "fn": fn,
            "empty_crops": neg, "false_alarm_rate": fp / neg if neg else float("nan")}


def main():
    parser = argparse.ArgumentParser(description="Ball tracking benchmark against the ground truth")
    parser.add_argument("--clips", default=None, help="comma-separated (default: every clip with GT)")
    parser.add_argument("--videos", default="data", help="folder(s) of the clips, comma-separated")
    parser.add_argument("--outputs", default=OUTPUT_ROOT)
    parser.add_argument("--realtime", default=os.path.join(OUTPUT_ROOT, "realtime"))
    parser.add_argument("--model", default=None, help="detector for the oracle-crop test")
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--csv", default=os.path.join(OUTPUT_ROOT, "realtime", "benchmark_ball.csv"))
    args = parser.parse_args()

    gts = sorted(glob.glob(os.path.join(GROUND_TRUTH_DIR, "ball_*_gt.csv")))
    clips = [Path(g).name[len("ball_"):-len("_gt.csv")] for g in gts]
    if args.clips:
        clips = [c for c in clips if c in args.clips.split(",")]
    backend = load_backend(args.model) if args.model else None
    rows = []
    for clip in clips:
        video = find_video(clip, args.videos.split(",")) or os.path.join("data", clip + ".mp4")
        gt = load_ball_gt(os.path.join(GROUND_TRUTH_DIR, f"ball_{clip}_gt.csv"))
        off, rt = clip_paths(video, args.outputs), clip_paths(video, args.realtime)
        tracks = {"offline": off["ball_csv"], "wholeframe": off["ball_wholeframe_csv"],
                  "realtime": rt["ball_csv"],
                  "live": os.path.join(rt["clip_dir"], "realtime", f"ball_live_{clip}.csv")}
        if os.path.exists(rt["ball_candidates_csv"]) and os.path.exists(video):
            tmp = os.path.join(tempfile.mkdtemp(), "rt_cands.csv")
            track_from_candidates(rt["ball_candidates_csv"], video, tmp)
            tracks["rt_cands"] = tmp
        for name, path in tracks.items():
            s = score(gt, path, args.radius)
            if s is None:
                continue
            rows.append({"clip": clip, "track": name, "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                         "recall": s["recall"], "recall_lo": s["recall_ci"][0],
                         "recall_hi": s["recall_ci"][1], "precision": s["precision"],
                         "precision_lo": s["precision_ci"][0],
                         "median_err_px": s["median_err_px"]})
        if backend is not None and os.path.exists(video):
            o = oracle(video, gt, off["court_csv"], backend, args.radius)
            rows.append({"clip": clip, "track": "oracle", "tp": o["tp"], "fn": o["fn"],
                         "recall": o["recall"], "recall_lo": o["recall_ci"][0],
                         "recall_hi": o["recall_ci"][1],
                         "false_alarm_rate": o["false_alarm_rate"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Nothing to score: no ground truth or no tracks.")
    with pd.option_context("display.width", 160, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
