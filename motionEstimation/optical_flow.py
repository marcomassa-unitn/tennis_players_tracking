"""
motionEstimation/optical_flow.py

Optical-flow motion estimation applied to the tracked players, as covered
in the course labs:

  - Farneback dense optical flow: the mean flow inside each tracked player
    bounding box gives a per-frame velocity estimate that is converted to
    km/h through the court homography and compared against the positional
    speed (frame-to-frame displacement of the projected feet point).
  - Lucas-Kanade pyramidal sparse flow on Shi-Tomasi corners: a short demo
    that tracks feature points and saves their trails.

Outputs (default outputs/motion_estimation/):
  - flow_speeds.csv        frame, player, flow vector, speed from flow vs
                           positional speed
  - flow_hsv_*.png         dense-flow colour coding (hue = direction,
                           value = magnitude)
  - flow_arrows_*.png      flow arrows on a sparse grid over the frame
  - lk_trails.png          Lucas-Kanade feature trails

Usage (from project root):
    python motionEstimation/optical_flow.py
    python motionEstimation/optical_flow.py --video data/Input_video2.mp4 \\
        --players outputs/players_clip2.csv \\
        --court outputs/court_coordinates/input_video_court.csv \\
        --frames 200 --output outputs/motion_estimation
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter

FB_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15,
                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


# ── helpers ────────────────────────────────────────────────────────────────────

def load_player_boxes(players_csv):
    """Return {frame: {player_id: (x, y, w, h)}} from the tracker CSV."""
    boxes = defaultdict(dict)
    with open(players_csv, newline="") as f:
        for row in csv.DictReader(f):
            boxes[int(row["frame"])][int(row["player_id"])] = (
                int(float(row["x"])), int(float(row["y"])),
                int(float(row["w"])), int(float(row["h"])),
            )
    return boxes


def mean_box_flow(flow, box, scale, noise_thr=0.3):
    """
    Mean flow vector (in ORIGINAL-resolution pixels) inside a player box.
    Only pixels that actually move (|flow| > noise_thr, in scaled px) are
    averaged, so the static background inside the box does not dilute the
    player's motion. Returns None if the box is empty.
    """
    x, y, w, h = box
    fh, fw = flow.shape[:2]
    xs, ys = int(x * scale), int(y * scale)
    ws, hs = max(1, int(w * scale)), max(1, int(h * scale))
    # Clip the box to the scaled-frame bounds so an out-of-frame box samples
    # the valid overlap only (instead of silently indexing wrong/empty rows).
    x0, y0 = max(0, xs), max(0, ys)
    x1, y1 = min(fw, xs + ws), min(fh, ys + hs)
    region = flow[y0: y1, x0: x1]
    if region.size == 0:
        return None
    mag = np.linalg.norm(region, axis=2)
    moving = mag > noise_thr
    sel = region[moving] if moving.any() else region.reshape(-1, 2)
    return sel.mean(axis=0) / scale


def flow_to_hsv(flow):
    """Standard HSV colour coding: hue = direction, value = magnitude."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(frame, flow, scale, grid=24, min_mag=0.6):
    """Flow arrows on a sparse grid (frame at original resolution)."""
    vis = frame.copy()
    h, w = flow.shape[:2]
    for ys in range(grid // 2, h, grid):
        for xs in range(grid // 2, w, grid):
            fx, fy = flow[ys, xs]
            if np.hypot(fx, fy) < min_mag:
                continue
            x0, y0 = int(xs / scale), int(ys / scale)
            x1 = int((xs + fx * 3) / scale)
            y1 = int((ys + fy * 3) / scale)
            cv2.arrowedLine(vis, (x0, y0), (x1, y1), (0, 255, 0), 1,
                            tipLength=0.3)
    return vis


# ── Lucas-Kanade demo ──────────────────────────────────────────────────────────

def lucas_kanade_demo(video, start, n_frames, out_path, display=False):
    """Track Shi-Tomasi corners with pyramidal LK and save the trails."""
    cap = cv2.VideoCapture(video)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read LK start frame")
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300,
                                      qualityLevel=0.01, minDistance=12,
                                      blockSize=7)
        # goodFeaturesToTrack returns None on a textureless start frame; in
        # that case there is nothing to track, so skip the demo gracefully
        # instead of crashing at len(pts).
        if pts is None or len(pts) == 0:
            print("  LK demo skipped: no corners found on the start frame.")
            return
        canvas = np.zeros_like(frame)
        colors = np.random.default_rng(7).integers(80, 255, (len(pts), 3))

        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            30, 0.01))
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok or pts is None or len(pts) == 0:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts,
                                                  None, **lk)
            good_new = nxt[st.flatten() == 1]
            good_old = pts[st.flatten() == 1]
            colors = colors[st.flatten() == 1]
            for (n, o, c) in zip(good_new, good_old, colors):
                cv2.line(canvas, tuple(o.ravel().astype(int)),
                         tuple(n.ravel().astype(int)),
                         tuple(int(v) for v in c), 1)
            prev_gray = gray
            pts = good_new.reshape(-1, 1, 2)
            if display:
                cv2.imshow("LK trails", cv2.add(frame, canvas))
                cv2.waitKey(15)

        cv2.imwrite(out_path, cv2.add(frame, canvas))
        print(f"  Saved: {out_path}  ({len(pts) if pts is not None else 0} "
              f"points still tracked)")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optical-flow motion estimation on tracked players")
    parser.add_argument("--video",   default="data/Input_video2.mp4")
    parser.add_argument("--players", default="outputs/players_Input_video2.csv")
    parser.add_argument("--court",
                        default="outputs/court_coordinates/Input_video2_court.csv")
    parser.add_argument("--fps",     type=float, default=30.0)
    parser.add_argument("--scale",   type=float, default=0.5,
                        help="Downscale factor for Farneback (default: 0.5)")
    parser.add_argument("--frames",  type=int, default=0,
                        help="Max frames to process (0 = whole video)")
    parser.add_argument("--vis-every", type=int, default=150, dest="vis_every",
                        help="Save flow visualizations every N frames "
                             "(default: 150)")
    parser.add_argument("--output",  default="outputs/motion_estimation")
    parser.add_argument("--no-lk",   action="store_true",
                        help="Skip the Lucas-Kanade demo")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    boxes = load_player_boxes(args.players)
    conv = CourtConverter(args.court)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    print(f"Farneback dense flow on {args.video} (scale {args.scale})")
    rows = []
    prev_gray = None
    prev_feet_m = {}          # player_id -> last projected feet point (m)
    prev_frame_idx = {}       # player_id -> frame of that point
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.frames and frame_idx >= args.frames):
                break
            small = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and frame_idx in boxes:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                    **FB_PARAMS)
                for pid, box in sorted(boxes[frame_idx].items()):
                    d = mean_box_flow(flow, box, args.scale)
                    if d is None:
                        continue
                    x, y, w, h = box
                    feet = np.array([x + w / 2.0, y + h], dtype=np.float64)
                    p1 = np.array(conv.to_meters(*(feet - d)))
                    p2 = np.array(conv.to_meters(*feet))
                    speed_flow = np.linalg.norm(p2 - p1) * args.fps * 3.6

                    speed_pos = np.nan
                    if (pid in prev_feet_m
                            and frame_idx - prev_frame_idx[pid] == 1):
                        speed_pos = (np.linalg.norm(p2 - prev_feet_m[pid])
                                     * args.fps * 3.6)
                    prev_feet_m[pid] = p2
                    prev_frame_idx[pid] = frame_idx

                    rows.append([frame_idx, pid,
                                 round(float(d[0]), 3), round(float(d[1]), 3),
                                 round(float(speed_flow), 3),
                                 round(float(speed_pos), 3)
                                 if not np.isnan(speed_pos) else ""])

                if args.vis_every and frame_idx % args.vis_every == 0:
                    hsv = flow_to_hsv(flow)
                    cv2.imwrite(os.path.join(
                        args.output, f"flow_hsv_{frame_idx:05d}.png"), hsv)
                    arrows = draw_flow_arrows(frame, flow, args.scale)
                    cv2.imwrite(os.path.join(
                        args.output, f"flow_arrows_{frame_idx:05d}.png"),
                        arrows)
                    if args.display:
                        cv2.imshow("Farneback flow", arrows)
                        cv2.waitKey(1)

            prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    csv_path = os.path.join(args.output, "flow_speeds.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "player_id", "flow_dx_px", "flow_dy_px",
                     "speed_flow_kmh", "speed_pos_kmh"])
        wr.writerows(rows)
    print(f"  Saved: {csv_path}  ({len(rows)} rows)")

    # summary: flow speed vs positional speed (tracking glitches above a
    # physiological 45 km/h are excluded from both, as in player_analysis).
    # Methodological note: this 45 km/h cap is a blunt outlier filter -- it
    # discards tracking glitches but will also clip any legitimately-noisy but
    # high flow-speed sample, so the reported means are slightly conservative.
    data = np.array([[r[1], r[4], r[5] if r[5] != "" else np.nan]
                     for r in rows], dtype=np.float64)
    data[:, 1][data[:, 1] > 45.0] = np.nan
    data[:, 2][data[:, 2] > 45.0] = np.nan
    print("\n  Flow-based vs positional speed (km/h, glitches excluded):")
    for pid in np.unique(data[:, 0]).astype(int):
        sub = data[data[:, 0] == pid]
        both = sub[~np.isnan(sub[:, 1]) & ~np.isnan(sub[:, 2])]
        mad = np.mean(np.abs(both[:, 1] - both[:, 2])) if len(both) else np.nan
        print(f"    P{pid}: mean flow {np.nanmean(sub[:, 1]):5.1f} | "
              f"mean positional {np.nanmean(sub[:, 2]):5.1f} | "
              f"mean |diff| {mad:4.1f}")

    if not args.no_lk:
        print("\nLucas-Kanade sparse-flow demo")
        lucas_kanade_demo(args.video, start=40, n_frames=90,
                          out_path=os.path.join(args.output, "lk_trails.png"),
                          display=args.display)

    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
