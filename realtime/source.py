"""Frames for the live pipeline, decoded in their own thread.

clock="all"   every frame is delivered, as fast as the pipeline takes them: measures
              the throughput, and gives the same result on any machine.
clock="live"  frames arrive at the clip's frame rate, like a camera. If the pipeline
              is busy the older frames are dropped and only the newest one waits, so
              the output stays in step with the video (and the drops are counted).
"""

from __future__ import annotations

import queue
import threading
import time

import cv2

from utils.video import safe_fps


class FrameSource:
    def __init__(self, video, clock="all", max_frames=None, buffer=8):
        if clock not in ("all", "live"):
            raise ValueError(f"clock must be 'all' or 'live', not {clock!r}")
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        self.fps = safe_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.clock = clock
        self.max_frames = max_frames
        self.dropped = 0
        self.decoded = 0
        self._q = queue.Queue(maxsize=buffer if clock == "all" else 1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="decode", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _put_live(self, item):
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def _run(self):
        t0 = time.perf_counter()
        idx = 0
        try:
            while not self._stop.is_set():
                if self.max_frames is not None and idx >= self.max_frames:
                    break
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.decoded += 1
                if self.clock == "live":
                    wait = t0 + idx / self.fps - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    self._put_live((idx, frame, time.perf_counter()))
                else:
                    while not self._stop.is_set():
                        try:
                            self._q.put((idx, frame, time.perf_counter()), timeout=0.1)
                            break
                        except queue.Full:
                            continue
                idx += 1
        finally:
            self.cap.release()
            self._put_end()

    def _put_end(self):
        while True:
            try:
                self._q.put(None, timeout=0.1)
                return
            except queue.Full:
                if self.clock == "live":
                    try:
                        self._q.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass
                elif self._stop.is_set():
                    return

    def __iter__(self):
        """(frame index, BGR frame, decode time) until the end of the clip."""
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def stop(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2)
