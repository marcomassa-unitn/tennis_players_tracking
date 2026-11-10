"""Build the YOLO dataset of the motion ball detector from the clips.

Every clip is replayed causally with the same online background as the live
pipeline (utils/background.py), so the model trains on the backgrounds it will
see. On the labelled frames, CROP x CROP crops at full resolution are cut and
stored as 3-channel PNGs (diff t-1, diff t, gray: see tracking/ball_online.py),
with YOLO labels.

Labels:
  pseudo  the real detections (interpolated == 0) of the current tracker,
          outputs/<clip>/ball_coordinates/ball_<clip>.csv, kept only inside runs of
          at least MIN_RUN detections; positions found only by the motion blobs
          (no detector candidate there) need MIN_MOTION_RUN.
  manual  ground_truth/ball_<clip>_gt.csv (frame, visible, x, y), from
          `python evaluation/annotate.py ball --frames-from ...`. They override the
          pseudo-labels; visible = 0 frames give crops without a ball.

Crops: most positives have the ball near the centre (as in TRACKING), the others
anywhere (as in SEARCH). Negatives come only from frames where the ball position is
known, so a crop never holds an unlabelled ball: around the players' rackets, on
moving blobs that are not the ball, and anywhere on the court.

Splits are by clip (TEST clips are never written: they are for evaluation).

Usage (from project root):
    python training/build_ball_dataset.py                       # every clip
    python training/build_ball_dataset.py --clips input3 --preview
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracking.ball_model import CROP
from tracking.ball_online import CropMaker, court_roi, pack_channels
from tracking.ball_tracking import ball_sized_blobs
from utils.background import OnlineBackground, camera_shift
from utils.court_converter import CourtConverter
from utils.paths import GROUND_TRUTH_DIR, OUTPUT_ROOT, clip_paths
from utils.video import safe_fps

PROC_HEIGHT = 540          # same processing scale as the player tracker

TEST_CLIPS = ("Input_video2", "input_video", "input11")
VAL_CLIPS = ("input4", "input7")
TAIL_VAL = {"input10": 0.75}   # this clip's frames from this fraction on go to val
SPLIT_GAP_S = 1.0              # frames dropped between the two parts of such a clip

MIN_RUN = 5                # pseudo-labels: runs of at least this many detections...
MIN_MOTION_RUN = 8         # ...or this many for positions only the blobs found
RUN_GAP = 2                # frames apart within a run
MATCH_PX = 2.0             # a ball row comes from a detector candidate this close
BOX_RANGE = (10, 40)       # px at 1080p: box sides clamped to this
MANUAL_BOX = 16            # px at 1080p: box of a manual click

STRIDE = 1                 # one pseudo-labelled frame in this many
MANUAL_COPIES = 4          # crops per manually labelled frame
CENTRED_SHARE = 0.7        # positives with the ball near the centre...
CENTRED_SIGMA = 40         # ...within this (px, normal)...
SEARCH_RANGE = 150         # ...the others within this (px, uniform)
NEG_PER_POS = 1.0          # negatives per positive
NEG_MARGIN = 20            # px: a negative crop keeps the ball this far outside
MIN_VISIBLE = 0.5          # a positive box must keep this share inside the crop
DIFF_JITTER = (0.7, 1.3)   # augmentation: gain on the differences...
GAMMA_JITTER = (0.8, 1.25)  # ...and gamma on the grey image


def find_video(stem, video_dirs):
    for d in video_dirs:
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    return None


def _runs(frames, gap=RUN_GAP):
    """Lists of frames, consecutive within `gap`."""
    runs, cur = [], []
    for f in frames:
        if cur and f - cur[-1] > gap:
            runs.append(cur)
            cur = []
        cur.append(f)
    if cur:
        runs.append(cur)
    return runs


def pseudo_labels(ball_csv, candidates_csv, height):
    """{frame: (cx, cy, w, h)} of the trustworthy real detections of the tracker."""
    s = height / 1080
    ball = pd.read_csv(ball_csv)
    ball = ball[ball["interpolated"] == 0].dropna(subset=["cx", "cy"])
    cands = pd.read_csv(candidates_csv) if os.path.exists(candidates_csv) else None
    by_frame = {} if cands is None else {
        f: np.c_[(g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2] for f, g in cands.groupby("frame")}
    detector = {}
    for r in ball.itertuples():
        c = by_frame.get(int(r.frame))
        detector[int(r.frame)] = bool(c is not None and len(c) and
                                      np.hypot(c[:, 0] - r.cx, c[:, 1] - r.cy).min() <= MATCH_PX)
    rows = {int(r.frame): r for r in ball.itertuples()}
    labels = {}
    for run in _runs(sorted(rows)):
        if len(run) < MIN_RUN:
            continue
        long_run = len(run) >= MIN_MOTION_RUN
        for f in run:
            if not detector[f] and not long_run:
                continue
            r = rows[f]
            w = float(np.clip(r.w, BOX_RANGE[0] * s, BOX_RANGE[1] * s))
            h = float(np.clip(r.h, BOX_RANGE[0] * s, BOX_RANGE[1] * s))
            labels[f] = (float(r.cx), float(r.cy), w, h)
    return labels


def refine_box_from_diff(diff, x, y, default):
    """(w, h) of the difference blob under a click at (x, y) of `diff`, or `default`."""
    r = 24
    h_img, w_img = diff.shape
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    win = diff[y0:min(h_img, int(y) + r), x0:min(w_img, int(x) + r)]
    if not win.size:
        return default
    mask = (win >= 30).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    k = lab[min(int(y) - y0, lab.shape[0] - 1), min(int(x) - x0, lab.shape[1] - 1)]
    if k == 0 or not (4 <= stats[k, 4] <= 400):
        return default
    return float(stats[k, 2]), float(stats[k, 3])


def manual_labels(gt_csv):
    """{frame: (cx, cy) or None (no ball)} from an annotation file."""
    if not os.path.exists(gt_csv):
        return {}
    gt = pd.read_csv(gt_csv)
    return {int(r.frame): ((float(r.x), float(r.y)) if int(r.visible) else None)
            for r in gt.itertuples()}


def split_of(clip, frame, n_frames, fps):
    if clip in TEST_CLIPS:
        return None
    if clip in VAL_CLIPS:
        return "val"
    if clip in TAIL_VAL:
        cut = TAIL_VAL[clip] * n_frames
        if frame >= cut:
            return "val"
        return None if frame >= cut - SPLIT_GAP_S * fps else "train"
    return "train"


class ClipWriter:
    """Cuts and writes the crops of one clip."""

    def __init__(self, out_dir, clip, rng, preview=0):
        self.out_dir, self.clip, self.rng = out_dir, clip, rng
        self.count = {"train": [0, 0], "val": [0, 0]}     # positives, negatives
        self.preview = preview
        self.thumbs = []

    def write(self, split, frame, k, crop, box):
        """box: (cx, cy, w, h) in crop px, or None."""
        name = f"{self.clip}_{frame:05d}_{k}"
        img_dir = os.path.join(self.out_dir, "images", split)
        lab_dir = os.path.join(self.out_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        cv2.imwrite(os.path.join(img_dir, name + ".png"), crop)      # PNG: lossless
        with open(os.path.join(lab_dir, name + ".txt"), "w") as f:
            if box is not None:
                cx, cy, w, h = box
                n = crop.shape[0]
                f.write(f"0 {cx / n:.6f} {cy / n:.6f} {w / n:.6f} {h / n:.6f}\n")
        self.count[split][0 if box is not None else 1] += 1
        if len(self.thumbs) < self.preview:
            t = crop.copy()
            if box is not None:
                cx, cy, w, h = box
                cv2.rectangle(t, (int(cx - w / 2) - 2, int(cy - h / 2) - 2),
                              (int(cx + w / 2) + 2, int(cy + h / 2) + 2), (0, 0, 255), 1)
            self.thumbs.append(t[..., ::-1])   # on screen: ball at t green, at t-1 red

    def save_preview(self):
        if not self.thumbs:
            return
        cols = 8
        rows = (len(self.thumbs) + cols - 1) // cols
        n = self.thumbs[0].shape[0]
        sheet = np.zeros((rows * n, cols * n, 3), np.uint8)
        for i, t in enumerate(self.thumbs):
            r, c = divmod(i, cols)
            sheet[r * n:(r + 1) * n, c * n:(c + 1) * n] = t
        path = os.path.join(self.out_dir, f"preview_{self.clip}.jpg")
        cv2.imwrite(path, sheet)
        print(f"  preview -> {path}")


def _augment(rng, gray, d_t, d_tm1):
    g = rng.uniform(*DIFF_JITTER)
    d_t = cv2.convertScaleAbs(d_t, alpha=g)
    d_tm1 = cv2.convertScaleAbs(d_tm1, alpha=g)
    gamma = rng.uniform(*GAMMA_JITTER)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, lut), d_t, d_tm1


def build_clip(clip, video, outputs, gt_dir, out_dir, rng, stride=STRIDE, preview=0):
    paths = clip_paths(video, outputs)
    for key in ("ball_csv", "players_csv", "court_csv"):
        if not os.path.exists(paths[key]):
            print(f"  {clip}: skipped, {paths[key]} not found")
            return None
    cap = cv2.VideoCapture(video)
    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = H / 1080
    proc = (int(round(W * PROC_HEIGHT / H)), PROC_HEIGHT)

    labels = {f: ("pseudo", v) for f, v in
              pseudo_labels(paths["ball_csv"], paths["ball_candidates_csv"], H).items()}
    manual = manual_labels(os.path.join(gt_dir, f"ball_{clip}_gt.csv"))
    for f, v in manual.items():
        labels[f] = ("manual", v)
    players = pd.read_csv(paths["players_csv"])
    boxes = {f: g[["x", "y", "w", "h"]].to_numpy(float) for f, g in players.groupby("frame")}

    roi = court_roi(CourtConverter(paths["court_csv"]), W, H)
    bg = OnlineBackground(fps, (W, H), proc, roi=roi, threaded=False)
    crops = CropMaker(bg.scale)
    hann = cv2.createHanningWindow(proc, cv2.CV_32F)
    writer = ClipWriter(out_dir, clip, rng, preview)
    print(f"  {clip}: {len(labels)} labelled frames "
          f"({sum(1 for k, _ in labels.values() if k == 'manual')} manual), ROI {roi}")

    prev, prev_shift, f = None, (0.0, 0.0), 0
    pseudo_seen = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, proc, interpolation=cv2.INTER_AREA)
            snap = bg.snapshot
            shift = camera_shift(snap.small_gray, small, hann) if snap is not None else (0.0, 0.0)
            pl = boxes.get(f, ())
            lab = labels.get(f)
            split = split_of(clip, f, n_frames, fps)
            if snap is not None and lab is not None and split is not None and prev is not None:
                kind, value = lab
                take = True
                if kind == "pseudo":
                    take = pseudo_seen % stride == 0
                    pseudo_seen += 1
                if take:
                    _write_frame(writer, crops, snap, frame, prev, shift, prev_shift, f, split,
                                 kind, value, pl, rng, s)
            bg.feed(f, frame, small, players=pl, shift=shift)
            prev, prev_shift, f = frame, shift, f + 1
    finally:
        cap.release()
    writer.save_preview()
    return writer.count


def _write_frame(writer, crops, snap, frame, prev, shift, prev_shift, f, split, kind, value,
                 players, rng, s):
    rx0, ry0, rx1, ry1 = snap.roi
    dx, dy = shift[0] / crops.scale, shift[1] / crops.scale
    if value is None:                         # manual "no ball": negatives anywhere
        ball = None
        n_pos = 0
    else:
        cx, cy = value[0], value[1]
        if len(value) == 4:
            bw, bh = value[2], value[3]
        else:
            d, _ = crops.diff(snap, frame, *crops.clamp(snap, cx - dx - CROP / 2,
                                                        cy - dy - CROP / 2), shift)
            o = crops.clamp(snap, cx - dx - CROP / 2, cy - dy - CROP / 2)
            bw, bh = refine_box_from_diff(d, cx - dx - o[0], cy - dy - o[1],
                                          (MANUAL_BOX * s, MANUAL_BOX * s))
        ball = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)     # frame px
        if not (rx0 <= cx - dx < rx1 and ry0 <= cy - dy < ry1):
            return
        n_pos = MANUAL_COPIES if kind == "manual" else 1

    def cut(o):
        d_t, img = crops.diff(snap, frame, o[0], o[1], shift)
        d_tm1, _ = crops.diff(snap, prev, o[0], o[1], prev_shift)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pack_channels(*_augment(rng, gray, d_t, d_tm1))

    k = 0
    for _ in range(n_pos):
        if rng.random() < CENTRED_SHARE:
            ox, oy = rng.normal(0, CENTRED_SIGMA * s, 2)
        else:
            ox, oy = rng.uniform(-SEARCH_RANGE * s, SEARCH_RANGE * s, 2)
        o = crops.clamp(snap, cx - dx + ox - CROP / 2, cy - dy + oy - CROP / 2)
        bx0, by0 = ball[0] - dx - o[0], ball[1] - dy - o[1]
        bx1, by1 = ball[2] - dx - o[0], ball[3] - dy - o[1]
        cx0, cy0 = max(bx0, 0), max(by0, 0)
        cx1, cy1 = min(bx1, CROP), min(by1, CROP)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < MIN_VISIBLE * (bx1 - bx0) * (by1 - by0):
            continue
        writer.write(split, f, k, cut(o), ((cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0))
        k += 1

    n_neg = int(NEG_PER_POS * max(n_pos, 1) + rng.random())
    blobs = None
    for _ in range(n_neg):
        for _attempt in range(6):
            kind_neg = rng.choice(["racket", "blob", "court"], p=[0.4, 0.3, 0.3])
            if kind_neg == "racket" and len(players):
                x, y, w, h = players[rng.integers(len(players))]
                px, py = x + w / 2 + rng.normal(0, 0.5 * h), y + 0.3 * h + rng.normal(0, 0.3 * h)
            elif kind_neg == "blob":
                if blobs is None:
                    region = frame[ry0:ry1, rx0:rx1]
                    background = snap.full
                    blobs = [((a + c) / 2 + rx0, (b + d) / 2 + ry0) for a, b, c, d in
                             ball_sized_blobs(region, background, frame.shape[0])] \
                        if background.shape == region.shape else []
                if not blobs:
                    continue
                px, py = blobs[rng.integers(len(blobs))]
            else:
                px, py = rng.uniform(rx0, rx1), rng.uniform(ry0, ry1)
            o = crops.clamp(snap, px - dx - CROP / 2, py - dy - CROP / 2)
            if ball is not None:
                fx0, fy0 = o[0] + dx - NEG_MARGIN * s, o[1] + dy - NEG_MARGIN * s
                fx1, fy1 = fx0 + CROP + 2 * NEG_MARGIN * s, fy0 + CROP + 2 * NEG_MARGIN * s
                if ball[2] > fx0 and ball[0] < fx1 and ball[3] > fy0 and ball[1] < fy1:
                    continue
            writer.write(split, f, k, cut(o), None)
            k += 1
            break


def write_yaml(out_dir):
    path = os.path.join(out_dir, "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\n"
                "names:\n  0: ball\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Build the motion ball detector dataset")
    parser.add_argument("--videos", default="data", help="folder(s) with the clips, comma-separated")
    parser.add_argument("--outputs", default=OUTPUT_ROOT,
                        help="results of the offline pipeline (ball, players, court CSVs)")
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_DIR, dest="gt_dir")
    parser.add_argument("--out", default="datasets/ball_motion")
    parser.add_argument("--clips", default=None, help="comma-separated clip names (default: all)")
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", action="store_true", help="save a contact sheet per clip")
    args = parser.parse_args()

    video_dirs = args.videos.split(",")
    if args.clips:
        clips = args.clips.split(",")
    else:
        clips = sorted({Path(p).parent.parent.name for p in
                        glob.glob(os.path.join(args.outputs, "*", "ball_coordinates", "ball_*.csv"))})
    rng = np.random.default_rng(args.seed)
    total = {"train": [0, 0], "val": [0, 0]}
    for clip in clips:
        if clip in TEST_CLIPS:
            print(f"  {clip}: test clip, not written")
            continue
        video = find_video(clip, video_dirs)
        if video is None:
            print(f"  {clip}: skipped, no video in {video_dirs}")
            continue
        count = build_clip(clip, video, args.outputs, args.gt_dir, args.out, rng,
                           args.stride, 32 if args.preview else 0)
        if count:
            for k in total:
                total[k][0] += count[k][0]
                total[k][1] += count[k][1]
            print(f"  {clip}: " + ", ".join(f"{k} {v[0]}+/{v[1]}-" for k, v in count.items()))
    print("Total: " + ", ".join(f"{k} {v[0]} positives / {v[1]} negatives" for k, v in total.items()))
    print(f"Dataset config -> {write_yaml(args.out)}")


if __name__ == "__main__":
    main()
