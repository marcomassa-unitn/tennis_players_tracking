"""Per-stage timing of the live pipeline: timings.csv and summary.json."""

from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager

import numpy as np


class StageTimer:
    def __init__(self):
        self.rows = []              # (frame, stage, ms)
        self.frame_ms = []          # (frame, total ms, latency ms)
        self.frame_t = []           # perf_counter at the start of each frame
        self.t_ready = None         # when the startup (court, background) was over
        self._current = {}
        self.t_start = time.perf_counter()
        self.t_first = None         # first frame processed (after the startup)
        self.marks = {}

    @contextmanager
    def stage(self, frame, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t) * 1000.0
            self.rows.append((frame, name, ms))
            self._current[name] = ms

    def mark(self, name):
        """Record when something happened (seconds since the start)."""
        self.marks[name] = round(time.perf_counter() - self.t_start, 3)

    def ready(self):
        """The startup is over: steady-state speed is measured from here."""
        if self.t_ready is None:
            self.t_ready = time.perf_counter()
            self.mark("tracking_started")

    def end_frame(self, frame, t_begin, t_decoded):
        now = time.perf_counter()
        if self.t_first is None:
            self.t_first = t_begin
        self.frame_ms.append((frame, (now - t_begin) * 1000.0, (now - t_decoded) * 1000.0))
        self.frame_t.append(t_begin)
        last, self._current = self._current, {}
        return last

    def recent_fps(self, n=30):
        if len(self.frame_ms) < 2:
            return 0.0
        ms = [m for _, m, _ in self.frame_ms[-n:]]
        return 1000.0 / max(np.mean(ms), 1e-6)

    def summary(self, fps, dropped=0, extra=None):
        stages = {}
        for name in dict.fromkeys(n for _, n, _ in self.rows):
            ms = np.array([m for _, n, m in self.rows if n == name])
            stages[name] = {"calls": int(len(ms)), "mean_ms": round(float(ms.mean()), 2),
                            "p50_ms": round(float(np.percentile(ms, 50)), 2),
                            "p95_ms": round(float(np.percentile(ms, 95)), 2),
                            "max_ms": round(float(ms.max()), 2)}
        t_ready = self.t_ready if self.t_ready is not None else self.t_start
        steady = [k for k, t in enumerate(self.frame_t) if t >= t_ready]
        total = np.array([self.frame_ms[k][1] for k in steady]) if steady else np.zeros(1)
        lat = np.array([self.frame_ms[k][2] for k in steady]) if steady else np.zeros(1)
        span = self.frame_t[steady[-1]] - self.frame_t[steady[0]] if len(steady) > 1 else 0.0
        steady_fps = (len(steady) - 1) / span if span > 0 else 0.0
        wall = time.perf_counter() - (self.t_first or self.t_start)
        out = {
            "frames": len(self.frame_ms),
            "dropped_frames": int(dropped),
            "video_fps": fps,
            "throughput_fps": round(len(self.frame_ms) / max(wall, 1e-6), 2),
            "steady_frames": len(steady),
            "steady_fps": round(steady_fps, 2),
            "frame_ms": {"mean": round(float(total.mean()), 2),
                         "p50": round(float(np.percentile(total, 50)), 2),
                         "p95": round(float(np.percentile(total, 95)), 2)},
            "decode_to_output_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                                    "p95": round(float(np.percentile(lat, 95)), 2)},
            "real_time": bool(steady_fps >= fps),
            "marks_s": self.marks,
            "stages": stages,
        }
        if extra:
            out.update(extra)
        return out

    def save(self, out_dir, summary):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "timings.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "stage", "ms"])
            w.writerows((fr, n, round(m, 3)) for fr, n, m in self.rows)
        path = os.path.join(out_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path
