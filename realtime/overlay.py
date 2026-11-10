"""Light live overlay: court, players, ball, model crops, event ticker, timings.

Drawn on a downscaled frame (DISPLAY_W wide) so it never costs more than a few ms;
the full compositor of live_view.py is for replaying finished results.
"""

from __future__ import annotations

import queue
import threading

import cv2
import numpy as np

from utils.court_overlay import MODEL_SEGMENTS

DISPLAY_W = 960
P_COLORS = {1: (0, 255, 0), 2: (0, 0, 255)}


class Overlay:
    def __init__(self, frame_w, frame_h, width=DISPLAY_W):
        self.k = width / frame_w
        self.size = (width, int(round(frame_h * self.k)))
        self.court = None
        self.ticker = []

    def set_court(self, H_m_to_px):
        segs = []
        for a, b in MODEL_SEGMENTS:
            p = cv2.perspectiveTransform(np.array([[a], [b]], np.float64), H_m_to_px)[:, 0]
            segs.append(tuple(map(tuple, np.round(p * self.k).astype(int))))
        self.court = segs

    def add_event(self, text):
        self.ticker = (self.ticker + [text])[-4:]

    def draw(self, frame, idx, players, live, trail, crops, crop_size, hud):
        k = self.k
        img = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.court:
            for p, q in self.court:
                cv2.line(img, p, q, (255, 255, 255), 1, cv2.LINE_AA)
        for x0, y0 in crops:
            cv2.rectangle(img, (int(x0 * k), int(y0 * k)),
                          (int((x0 + crop_size) * k), int((y0 + crop_size) * k)), (120, 120, 120), 1)
        for pid, (x, y, w, h) in players.items():
            c = P_COLORS.get(pid, (255, 255, 0))
            cv2.rectangle(img, (int(x * k), int(y * k)), (int((x + w) * k), int((y + h) * k)), c, 2)
            cv2.putText(img, f"P{pid}", (int(x * k), max(12, int(y * k) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        pts = [(int(cx * k), int(cy * k)) for cx, cy in trail]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (0, 200, 255), 2, cv2.LINE_AA)
        if live is not None:
            c = (0, 255, 255) if live["detected"] else (0, 140, 255)
            cv2.circle(img, (int(live["cx"] * k), int(live["cy"] * k)), 6, c, 2, cv2.LINE_AA)
        y = 18
        for line in hud:
            cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y += 17
        y = self.size[1] - 10
        for line in reversed(self.ticker):
            cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                        cv2.LINE_AA)
            y -= 18
        cv2.putText(img, f"{idx}", (self.size[0] - 60, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img


class VideoSink:
    """Writes overlay frames to a video file in its own thread."""

    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.q = queue.Queue(maxsize=32)
        self.thread = threading.Thread(target=self._run, name="video-out", daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            img = self.q.get()
            if img is None:
                break
            self.writer.write(img)
        self.writer.release()

    def write(self, img):
        self.q.put(img)

    def close(self):
        self.q.put(None)
        self.thread.join()
