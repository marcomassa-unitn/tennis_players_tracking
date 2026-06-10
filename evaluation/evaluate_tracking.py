"""
evaluation/evaluate_tracking.py

Quantitative evaluation of the tracking pipeline against manual ground
truth created with evaluation/annotate.py.

Player tracking (when --gt is given):
  - GT boxes and predicted boxes are matched per frame by best IoU
    (all assignments are enumerated — at most 2 players per side).
  - mean / median IoU over matched pairs
  - precision and recall at the chosen IoU threshold (default 0.5)
  - centre error in pixels and feet-point error in metres (if a court
    CSV is given the homography converts both feet points)
  - ID switches: how often the predicted id assigned to the same GT
    player changes between consecutive annotated frames
  - per-frame detail saved to outputs/evaluation/player_eval.csv

Court keypoints (when --court-gt is given):
  - per-label pixel error between detected and annotated keypoints
  - error in metres through the GT homography

Usage (from project root):
    python evaluation/evaluate_tracking.py \\
        --gt   outputs/ground_truth/players_clip2_gt.csv \\
        --pred outputs/players_clip2.csv \\
        --court outputs/court_coordinates/input_video_court.csv \\
        --court-gt   outputs/ground_truth/input_video2_court_gt.csv \\
        --court-pred outputs/court_coordinates/input_video_court.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter, _REAL_WORLD


# ── helpers ────────────────────────────────────────────────────────────────────

def load_boxes(path):
    """CSV with frame,player_id,x,y,w,h[,...] → {frame: {pid: (x,y,w,h)}}."""
    boxes = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            boxes[int(row["frame"])][int(row["player_id"])] = (
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            )
    return boxes


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center(box):
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0])


def feet(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h)


def match_boxes(gt, pred):
    """
    Best assignment GT→pred by total IoU (≤2 boxes per side: enumerate).
    Returns list of (gt_id, pred_id, iou).
    """
    gt_ids, pred_ids = list(gt), list(pred)
    if not gt_ids or not pred_ids:
        return []
    best, best_score = [], -1.0
    k = min(len(gt_ids), len(pred_ids))
    for perm in permutations(pred_ids, k):
        pairs = list(zip(gt_ids, perm))
        score = sum(iou(gt[g], pred[p]) for g, p in pairs)
        if score > best_score:
            best_score = score
            best = [(g, p, iou(gt[g], pred[p])) for g, p in pairs]
    return best


# ── player evaluation ──────────────────────────────────────────────────────────

def evaluate_players(gt_path, pred_path, court_path, iou_thr, out_dir):
    gt_all = load_boxes(gt_path)
    pred_all = load_boxes(pred_path)
    conv = CourtConverter(court_path) if court_path else None

    rows = []
    n_gt = n_pred = n_tp = 0
    ious, errs_px, errs_m = [], [], []
    last_assign = {}          # gt_id -> pred_id on the previous annotated frame
    id_switches = 0

    for frame in sorted(gt_all):
        gt = gt_all[frame]
        pred = pred_all.get(frame, {})
        n_gt += len(gt)
        n_pred += len(pred)

        for g, p, ov in match_boxes(gt, pred):
            err_px = float(np.linalg.norm(center(gt[g]) - center(pred[p])))
            err_m = ""
            if conv is not None:
                fm_gt = np.array(conv.to_meters(*feet(gt[g])))
                fm_pr = np.array(conv.to_meters(*feet(pred[p])))
                err_m = float(np.linalg.norm(fm_gt - fm_pr))
                errs_m.append(err_m)
                err_m = round(err_m, 3)
            ious.append(ov)
            errs_px.append(err_px)
            if ov >= iou_thr:
                n_tp += 1
            if g in last_assign and last_assign[g] != p:
                id_switches += 1
            last_assign[g] = p
            rows.append([frame, g, p, round(ov, 4), round(err_px, 2), err_m])

    os.makedirs(out_dir, exist_ok=True)
    detail = os.path.join(out_dir, "player_eval.csv")
    with open(detail, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "gt_id", "pred_id", "iou",
                     "center_err_px", "feet_err_m"])
        wr.writerows(rows)

    print("\n" + "=" * 56)
    print("  PLAYER TRACKING EVALUATION")
    print("=" * 56)
    print(f"  annotated frames      : {len(gt_all)}")
    print(f"  GT boxes / pred boxes : {n_gt} / {n_pred}")
    if ious:
        print(f"  mean IoU (matched)    : {np.mean(ious):.3f}")
        print(f"  median IoU            : {np.median(ious):.3f}")
        print(f"  recall  @IoU>={iou_thr}   : {n_tp / n_gt:.3f}")
        print(f"  precision @IoU>={iou_thr} : {n_tp / max(1, n_pred):.3f}")
        print(f"  centre error (px)     : mean {np.mean(errs_px):.1f}, "
              f"median {np.median(errs_px):.1f}")
        if errs_m:
            print(f"  feet error (m)        : mean {np.mean(errs_m):.2f}, "
                  f"median {np.median(errs_m):.2f}")
        print(f"  ID switches           : {id_switches}")
    else:
        print("  no overlapping frames between GT and predictions!")
    print(f"  per-frame detail      : {detail}")


# ── court evaluation ───────────────────────────────────────────────────────────

def load_court(path):
    pts = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pts[row["label"].strip()] = (float(row["x"]), float(row["y"]))
    return pts


def evaluate_court(gt_path, pred_path, out_dir):
    gt = load_court(gt_path)
    pred = load_court(pred_path)
    conv = CourtConverter(gt_path)   # homography from the GT keypoints

    print("\n" + "=" * 56)
    print("  COURT KEYPOINT EVALUATION")
    print("=" * 56)
    print(f"  {'label':<6s} {'err px':>8s} {'err m':>8s}")

    errs_px, errs_m = [], []
    rows = []
    for label in ("TL", "TR", "BL", "BR", "STL", "STR", "SBL", "SBR"):
        if label not in gt or label not in pred:
            continue
        e_px = float(np.hypot(gt[label][0] - pred[label][0],
                              gt[label][1] - pred[label][1]))
        # project the detected point with the GT homography and compare with
        # the real-world ITF position of that keypoint
        pm = np.array(conv.to_meters(*pred[label]))
        e_m = float(np.linalg.norm(pm - np.array(_REAL_WORLD[label])))
        errs_px.append(e_px)
        errs_m.append(e_m)
        rows.append([label, round(e_px, 2), round(e_m, 3)])
        print(f"  {label:<6s} {e_px:8.1f} {e_m:8.2f}")

    if errs_px:
        print(f"  {'mean':<6s} {np.mean(errs_px):8.1f} {np.mean(errs_m):8.2f}")

    os.makedirs(out_dir, exist_ok=True)
    detail = os.path.join(out_dir, "court_eval.csv")
    with open(detail, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "err_px", "err_m"])
        wr.writerows(rows)
    print(f"  per-label detail      : {detail}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate player tracking and court detection vs GT")
    parser.add_argument("--gt",
                        help="GT player boxes CSV (evaluation/annotate.py boxes)")
    parser.add_argument("--pred", default="outputs/players_clip2.csv",
                        help="tracker output CSV")
    parser.add_argument("--court", default=None,
                        help="court CSV used to convert errors to metres")
    parser.add_argument("--iou-thr", type=float, default=0.5, dest="iou_thr")
    parser.add_argument("--court-gt", dest="court_gt",
                        help="GT court keypoints CSV (annotate.py court)")
    parser.add_argument("--court-pred", dest="court_pred",
                        default="outputs/court_coordinates/Input_video2_court.csv",
                        help="detected court keypoints CSV")
    parser.add_argument("--output", default="outputs/evaluation")
    args = parser.parse_args()

    if not args.gt and not args.court_gt:
        parser.error("nothing to do: pass --gt and/or --court-gt")

    if args.gt:
        evaluate_players(args.gt, args.pred, args.court,
                         args.iou_thr, args.output)
    if args.court_gt:
        evaluate_court(args.court_gt, args.court_pred, args.output)
    print()


if __name__ == "__main__":
    main()
