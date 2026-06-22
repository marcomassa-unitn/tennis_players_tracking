"""
evaluation/evaluate_tracking.py

Quantitative tracking metrics against manual ground truth from
evaluation/annotate.py.

Player tracking (--gt): per-frame best-IoU matching of GT vs predicted
boxes, mean/median IoU, precision/recall at the IoU threshold, centre
error (px) and feet-point error (m, via the court homography), and an
ID-switch count. Per-frame detail goes to outputs/evaluation/player_eval.csv.

Court keypoints (--court-gt): per-label pixel error and metric error
through the GT homography.

Usage (from project root):
    python evaluation/evaluate_tracking.py \\
        --gt   outputs/ground_truth/players_clip2_gt.csv \\
        --pred outputs/player_coordinates/players_Input_video2.csv \\
        --court outputs/court_coordinates/Input_video2_court.csv \\
        --court-gt   outputs/ground_truth/input_video2_court_gt.csv \\
        --court-pred outputs/court_coordinates/Input_video2_court.csv
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

def load_boxes(path, valid_ids=None):
    """
    Read frame,player_id,x,y,w,h CSV into {frame: {pid: (x,y,w,h)}}.

    `valid_ids` (e.g. {1, 2} for GT) gates player_id: out-of-set rows are
    warned and skipped so a typo can't introduce a spurious third identity.
    """
    boxes = defaultdict(dict)
    skipped = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pid = int(row["player_id"])
            if valid_ids is not None and pid not in valid_ids:
                if skipped < 10:
                    print(f"  [warn] {path}: skipping row with player_id="
                          f"{pid} (expected one of {sorted(valid_ids)}), "
                          f"frame {row['frame']}")
                skipped += 1
                continue
            boxes[int(row["frame"])][pid] = (
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            )
    if skipped:
        print(f"  [warn] {path}: skipped {skipped} row(s) with "
              f"unexpected player_id")
    return boxes


def iou(a, b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center(box):
    """Box centroid as a 2-vector."""
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0])


def feet(box):
    """Bottom-centre (feet) point of the box, not its centroid."""
    x, y, w, h = box
    return (x + w / 2.0, y + h)


def match_boxes(gt, pred):
    """
    Brute-force the GT→pred assignment maximising summed IoU.

    Exhaustive permutation search is fine since each side has ≤2 boxes.
    Returns (gt_id, pred_id, iou) triples for the winning assignment.
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
    """
    Score predicted player boxes against GT, per annotated frame.

    Caveat on "ID switches": predicted ids are area-ordered connected
    components, not a re-id ground truth, so this counts assignment
    instability (the pred id matched to a GT player flipping between
    consecutive annotated frames), not true identity-swap errors.
    """
    # annotate.py labels P1=NEAR, P2=FAR, so only ids {1,2} are legitimate.
    gt_all = load_boxes(gt_path, valid_ids={1, 2})
    pred_all = load_boxes(pred_path)
    conv = CourtConverter(court_path) if court_path else None

    rows = []
    n_gt = n_pred = n_tp = 0
    ious, errs_px, errs_m = [], [], []
    last_assign = {}          # gt_id -> pred_id on the previous annotated frame
    id_switches = 0           # assignment flips, not re-id errors (see docstring)

    for frame in sorted(gt_all):
        gt = gt_all[frame]
        pred = pred_all.get(frame, {})
        n_gt += len(gt)
        n_pred += len(pred)

        for g, p, ov in match_boxes(gt, pred):
            err_px = float(np.linalg.norm(center(gt[g]) - center(pred[p])))
            err_m_val = None
            err_m = ""
            if conv is not None:
                fm_gt = np.array(conv.to_meters(*feet(gt[g])))
                fm_pr = np.array(conv.to_meters(*feet(pred[p])))
                err_m_val = float(np.linalg.norm(fm_gt - fm_pr))
                err_m = round(err_m_val, 3)
            ious.append(ov)
            # Gate position-error aggregation on IoU>=thr: match_boxes returns
            # a best assignment even for poor/zero overlaps, which would
            # otherwise pollute the centre/feet error means. precision/recall
            # use the same gate.
            if ov >= iou_thr:
                n_tp += 1
                errs_px.append(err_px)
                if err_m_val is not None:
                    errs_m.append(err_m_val)
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
        if errs_px:
            print(f"  centre error (px)     : mean {np.mean(errs_px):.1f}, "
                  f"median {np.median(errs_px):.1f}  (IoU>={iou_thr} only)")
        else:
            print(f"  centre error (px)     : n/a (no matches at IoU>={iou_thr})")
        if errs_m:
            print(f"  feet error (m)        : mean {np.mean(errs_m):.2f}, "
                  f"median {np.median(errs_m):.2f}  (IoU>={iou_thr} only)")
        print(f"  ID switches           : {id_switches}")
    else:
        print("  no overlapping frames between GT and predictions!")
    print(f"  per-frame detail      : {detail}")


# ── court evaluation ───────────────────────────────────────────────────────────

def load_court(path):
    """Read a court keypoints CSV into {label: (x, y)}."""
    pts = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pts[row["label"].strip()] = (float(row["x"]), float(row["y"]))
    return pts


def evaluate_court(gt_path, pred_path, out_dir):
    """Report per-keypoint pixel and metric error of detected vs GT court."""
    gt = load_court(gt_path)
    pred = load_court(pred_path)
    conv = CourtConverter(gt_path)   # homography built from the GT keypoints

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
        # metric error: detected point under the GT homography vs the
        # keypoint's real-world ITF position
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
    """Parse args and run player and/or court evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate player tracking and court detection vs GT")
    parser.add_argument("--gt",
                        help="GT player boxes CSV (evaluation/annotate.py boxes)")
    parser.add_argument("--pred",
                        default="outputs/player_coordinates/players_Input_video2.csv",
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
