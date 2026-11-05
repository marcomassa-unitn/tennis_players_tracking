#!/usr/bin/env python3
"""Show what placing the impact on the ball's flight line changes, side by side.

Left, the ball goes from bounce to the player's feet, so it turns at every bounce.
Right, it goes to the impact on the line of its flight, and runs straight through.
Both are drawn from the same shots.csv; no match footage is used, only the court.

Use (from project root, after the pipeline has run on the clip):
    python analysis/impact_comparison.py --video data/input7.mp4 --still 156

Writes impact_comparison.mp4 and .gif in the clip's shot_analysis folder.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.shot_analysis import shot_category, shot_side
from live_view import SS, BallMap, ball_ground_path, load_bounce_markers
from utils.paths import clip_paths

PANEL = (300, 640)      # width, height of each court
HEADER_H = 64
TRAIL_S = 2.5           # long enough to show the ball before and after a bounce
SLOW_DOWN = 2           # each frame is held this many frames in the video
GIF_STEP = 2            # the GIF keeps one frame every this many
BG = (46, 26, 26)       # BGR
TITLES = (("Before", "shots at the player's feet"), ("After", "impact on the flight line"))


def _markers(shots, ball_map, x, y):
    """[(frame, pid, category, side, (px, py)), ...] for the shots, at columns x, y."""
    return [(int(r.frame), int(r.player_id), shot_category(r.shot_type, r.overhead),
             shot_side(r.stroke, r.overhead), ball_map._to_px(getattr(r, x), getattr(r, y)))
            for r in shots.dropna(subset=[x, y]).itertuples()]


def _trail(path, frame, length):
    """The recent path up to `frame`, stopping at the first frame without one."""
    points = []
    for g in range(frame, frame - length, -1):
        if g not in path:
            break
        points.append(path[g])
    return points[::-1]


def _panel(ball_map, path, bounces, shots, frame, fps):
    img = ball_map.render(path.get(frame), _trail(path, frame, int(TRAIL_S * fps)),
                          [b[1:] for b in bounces if b[0] <= frame],
                          [m[1:] for m in shots if m[0] <= frame])
    return cv2.resize(img, PANEL, interpolation=cv2.INTER_AREA)


def _header(width, title, subtitle):
    head = np.full((HEADER_H, width, 3), BG, np.uint8)
    cv2.putText(head, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2,
                cv2.LINE_AA)
    cv2.putText(head, subtitle, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1,
                cv2.LINE_AA)
    return head


def _open_writer(path, fps, size):
    """H.264 if this OpenCV build has it, for sharing; MPEG-4 otherwise."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            print(f"  Codec: {codec}")
            return writer
    raise SystemExit(f"Cannot write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video comparing the ball's path before and after placing the impacts "
                    "on its flight line")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="clip to show; only its name is used, to find its results")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--last", type=int, default=None,
                        help="last frame to show (default: the last shot or bounce)")
    parser.add_argument("--still", type=int, default=None,
                        help="also save this frame as impact_comparison.png")
    args = parser.parse_args()

    paths = clip_paths(args.video)
    if not os.path.exists(paths["shots_csv"]):
        raise SystemExit(f"Shots not found: {paths['shots_csv']}\n"
                         f"Run: python pipeline.py --video {args.video}")
    shots = pd.read_csv(paths["shots_csv"])
    bounces = pd.read_csv(paths["bounces_csv"])

    ball_map = BallMap((PANEL[0] * SS, PANEL[1] * SS))
    bounds = (-10, 20, -10, 35)
    bounce_marks = load_bounce_markers(bounces, ball_map, bounds)
    before = ball_ground_path(shots.drop(columns=["impact_x_m", "impact_y_m"]), bounces, args.fps)
    after = ball_ground_path(shots, bounces, args.fps)
    versions = ((before, _markers(shots, ball_map, "player_x_m", "player_y_m")),
                (after, _markers(shots, ball_map, "impact_x_m", "impact_y_m")))

    gap = np.full((HEADER_H + PANEL[1], 8, 3), BG, np.uint8)
    headers = [_header(PANEL[0], *title) for title in TITLES]
    first = int(min(min(before), min(after)))
    last = args.last or int(max(max(before), max(after))) + int(args.fps)

    out = Path(paths["shots_dir"])
    size = (2 * PANEL[0] + gap.shape[1], HEADER_H + PANEL[1])
    writer = _open_writer(str(out / "impact_comparison.mp4"), args.fps, size)
    gif_frames = []
    for frame in range(first, last + 1):
        columns = [np.vstack([head, _panel(ball_map, path, bounce_marks, marks, frame, args.fps)])
                   for head, (path, marks) in zip(headers, versions)]
        canvas = np.hstack([columns[0], gap, columns[1]])
        for _ in range(SLOW_DOWN):
            writer.write(canvas)
        if (frame - first) % GIF_STEP == 0:
            small = cv2.resize(canvas, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(small[:, :, ::-1]))
        if frame == args.still:
            cv2.imwrite(str(out / "impact_comparison.png"), canvas)
    writer.release()

    palette = gif_frames[len(gif_frames) // 2].quantize(colors=64)
    gif_frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in gif_frames]
    gif_frames[0].save(out / "impact_comparison.gif", save_all=True, append_images=gif_frames[1:],
                       duration=int(1000 * GIF_STEP * SLOW_DOWN / args.fps), loop=0, optimize=True)
    print(f"Saved in {out}: impact_comparison.mp4 and .gif ({last - first + 1} frames)")


if __name__ == "__main__":
    main()
