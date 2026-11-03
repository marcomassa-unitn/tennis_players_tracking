#!/usr/bin/env python3
"""Watch the player tracker at work on a clip, one processing stage per panel.

    top-left     the frame: tracked players (P1 green, P2 red), other candidates grey
    top-right    the background: per-pixel median of frames spread over the clip
    bottom-left  difference from the background (brighter = more different)
    bottom-right foreground: white = kept; coloured = removed by a filter
                 (red cast shadow, blue on-screen graphics, yellow court lines,
                 grey far off court)

The players CSV is not touched. If the clip has no court keypoints yet, the
court is detected first and saved, as tracking/court_tracking.py would. Keys in
the window: space / p pause and resume, n next frame while paused, q quit.

Usage (from the project root):
    python display_player_tracking.py --video data/input6.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Window handling is shared with the live viewer (DPI fix, size as a share of
# the screen, pause / step keys), so both tools behave the same way.
from live_view import (
    PAUSE_KEYS,
    _enable_dpi_awareness,
    _place_window,
    _wait_while_paused,
    _window_size,
)
from tracking.court_tracking import CourtTracker
from tracking.player_tracking import PlayerTracker

WINDOW = "Player tracking stages  (space: pause, n: next, q: quit)"
TILE_W = 960                  # width of each of the four panels (window is scaled)
REMOVED_COLORS = {            # foreground pixels removed by each filter (BGR)
    "shadow": (0, 0, 200),
    "graphics": (200, 80, 0),
    "lines": (0, 210, 230),
    "off_court": (110, 110, 110),
}


def _panel(img, size, title):
    """`img` resized to `size` with a title bar."""
    out = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (size[0], 34), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


class StageViewer:
    """on_frame callback for PlayerTracker.run(): shows the four panels."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.size = (TILE_W, int(round(TILE_W * tracker.H / tracker.W)))
        self.placed = False
        self.paused = False

    def __call__(self, frame_idx, frame, cands, detections):
        tr = self.tracker
        view = frame.copy()
        for c in cands:
            d = tr._to_frame_coords(c)
            cv2.rectangle(view, (int(d["x0"]), int(d["y0"])), (int(d["x1"]), int(d["y1"])),
                          (170, 170, 170), 2)
        for pid, color in ((1, (0, 255, 0)), (2, (0, 0, 255))):
            d = detections[pid].get(frame_idx)
            if d is not None:
                p0 = (int(d["x0"]), int(d["y0"]))
                cv2.rectangle(view, p0, (int(d["x1"]), int(d["y1"])), color, 4)
                cv2.putText(view, f"P{pid}", (p0[0], max(30, p0[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        dx, dy = (v / tr.scale for v in tr.shift)
        panels = [_panel(view, self.size,
                         f"frame {frame_idx}   camera shift ({dx:+.1f}, {dy:+.1f}) px")]

        layers = tr.layers
        if layers is None:
            blank = np.zeros((self.size[1], self.size[0], 3), np.uint8)
            panels += [_panel(blank, self.size, "frame too dark: skipped")] * 3
        else:
            bg = tr.backgrounds[layers["k"]][1]
            panels.append(_panel(bg, self.size, "background (median of the clip)"))
            heat = cv2.applyColorMap(np.clip(layers["diff"] * 3, 0, 255).astype(np.uint8),
                                     cv2.COLORMAP_INFERNO)
            panels.append(_panel(heat, self.size, "difference from the background"))
            fg = np.zeros(bg.shape, np.uint8)
            for name, color in REMOVED_COLORS.items():
                if name in layers:
                    fg[layers[name]] = color
            fg[layers["final"]] = (255, 255, 255)
            panels.append(_panel(fg, self.size, "foreground: white kept | removed: red shadow, "
                                                "blue graphics, yellow lines, grey off court"))
        mosaic = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])

        cv2.imshow(WINDOW, mosaic)
        if not self.placed:
            _place_window(WINDOW, *_window_size(mosaic.shape[1] / mosaic.shape[0]))
            self.placed = True
        key = cv2.waitKey(1) & 0xFF
        if key in PAUSE_KEYS or self.paused:
            action = _wait_while_paused(WINDOW, mosaic)
            self.paused = action == "step"
            return action == "quit"
        return key == ord("q")


def main():
    parser = argparse.ArgumentParser(
        description="Show the player tracker's processing stages on a clip (nothing is saved)")
    parser.add_argument("--video", default="data/Input_video2.mp4")
    parser.add_argument("--court", default=None,
                        help="court CSV (default: outputs/court_coordinates/<video name>_court.csv)")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    court = args.court or os.path.join("outputs", "court_coordinates", f"{stem}_court.csv")
    if not os.path.exists(court) and not args.court:
        # The tracker needs the court; detect it first (this writes the court
        # CSV and its overlay image, as tracking/court_tracking.py would).
        print(f"No court keypoints for {args.video}: detecting the court first...")
        try:
            CourtTracker(args.video, no_display=True, output_dir="outputs").run()
        except RuntimeError as exc:
            raise SystemExit(f"Court detection failed: {exc}") from exc
    tracker = PlayerTracker(args.video, os.devnull, court, display=False)
    tracker.record_layers = True

    _enable_dpi_awareness()                 # before the first window is created
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        tracker.run(save=False, on_frame=StageViewer(tracker))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
