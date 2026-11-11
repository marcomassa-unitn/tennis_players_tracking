"""Calibration of a fixed camera before the match: the court and the empty court.

A live system with a fixed camera can learn both before play starts, during the
warm-up for instance, so the analysis tracks from its very first frame. On a
recorded clip the same is done here on the clip itself:

  court       the court model fitted on a few frames (tracking/court_tracking.py)
  background  the per-pixel median of frames spread over the clip, aligned to
              each other and without the dark ones, as in the offline player
              tracker, but at full resolution so the ball keeps its size

Both are written to <output>/<clip>/calibration/ (court.csv, court.png,
background.png) and reused by later runs. The live background still refreshes
itself from the frames as they come.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from tracking.court_tracking import (
    FRAMES_TO_TRY,
    GOOD_MATCH_RATIO,
    MIN_MATCH_RATIO,
    detect_court,
    save_court,
)
from tracking.player_tracking import PROC_HEIGHT
from utils.background import OnlineBackground
from utils.paths import clip_paths
from utils.video import safe_fps

CALIBRATION_FRAMES = 41    # frames of the background median


def calibration_paths(video, output):
    d = os.path.join(clip_paths(video, output)["clip_dir"], "calibration")
    return {"dir": d, "court_csv": os.path.join(d, "court.csv"),
            "background": os.path.join(d, "background.png")}


def _frames(cap, indices):
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            yield int(idx), frame


def fit_court(cap, total):
    """(H court metres -> pixels, match ratio, frame) of the best of a few frames."""
    best = (None, 0.0, None)
    last = max(1, total // 2)
    for _, frame in _frames(cap, np.linspace(0, last, FRAMES_TO_TRY).round()):
        H, ratio = detect_court(frame)
        if H is not None and ratio > best[1]:
            best = (H, ratio, frame)
        if best[1] >= GOOD_MATCH_RATIO:
            break
    return best


def median_background(cap, total, size, n=CALIBRATION_FRAMES):
    """Full-resolution median background of n frames spread over the clip."""
    W, H = size
    proc = (int(round(W * PROC_HEIGHT / H)), PROC_HEIGHT) if H > PROC_HEIGHT else (W, H)
    bg = OnlineBackground(safe_fps(cap.get(cv2.CAP_PROP_FPS)), (W, H), proc,
                          roi=(0, 0, W, H), threaded=False)
    small, full = [], []
    for _, frame in _frames(cap, np.linspace(0, max(total - 1, 0), n).round()):
        small.append(cv2.resize(frame, proc, interpolation=cv2.INTER_AREA))
        full.append(frame)
    snap = bg._build(small, full, 0, align=True)
    if snap is None:
        raise RuntimeError("No usable frame for the background")
    return snap.full


def calibrate(video, output, force=False, frames=CALIBRATION_FRAMES):
    """Court CSV and background PNG of `video`, computed once and then reused."""
    paths = calibration_paths(video, output)
    if not force and os.path.exists(paths["court_csv"]) and os.path.exists(paths["background"]):
        print(f"  calibration reused from {paths['dir']}")
        return paths
    os.makedirs(paths["dir"], exist_ok=True)
    t = time.perf_counter()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        H, ratio, frame = fit_court(cap, total)
        if H is None or ratio < MIN_MATCH_RATIO:
            raise RuntimeError(f"Court not found in {video} (best match {ratio:.0%})")
        save_court(H, frame, paths["court_csv"])
        background = median_background(cap, total, size, frames)
        cv2.imwrite(paths["background"], background)       # PNG: lossless
    finally:
        cap.release()
    print(f"  calibrated in {time.perf_counter() - t:.1f}s: court {ratio:.0%} match, "
          f"background from {frames} frames -> {paths['dir']}")
    return paths
