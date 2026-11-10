"""Bounces and shots while the clip plays: the offline analysis on a sliding window.

Both need some future. A bounce is the corner between two parabolas, known once
the second one has a few points; whether a near junction is a bounce or the hit
that follows is settled by the next event; a shot type reads the flight after the
contact, and a missing shot is only filled in once the next one is seen. Rather
than rewriting that logic causally, the live pipeline reruns the unchanged offline
functions (bounce_detection.detect_events, shot_analysis.analyze_shots,
detect_serves) every EVERY_S seconds on the last WINDOW_S seconds of final ball
and player positions, in a worker thread, and keeps only what is old enough to be
settled:

    events (hits, bounces)   EVENT_LAG_S behind the newest final ball position
    shots                    SHOT_LAG_S behind it

Inside the window the clip-wide medians of the shot analysis become window
medians. An event or shot is committed once; a later run that finds the same one
a frame or two away is recognised by DEDUP_FRAMES and ignored.
"""

from __future__ import annotations

import threading
import traceback
from collections import defaultdict

import pandas as pd

from analysis.bounce_detection import bounces_from_events, detect_events
from analysis.shot_analysis import (
    analyze_shots,
    ball_track_from_frame,
    detect_hits,
    detect_serves,
    real_detections_from_frame,
)
from tracking.ball_tracking import BALL_COLUMNS, fill_gaps

WINDOW_S = 10.0
EVERY_S = 1.0
EVENT_LAG_S = 1.0
SHOT_LAG_S = 4.0
DEDUP_FRAMES = 4
MIN_POINTS = 12            # real ball positions in the window before it is analysed


def ball_frame(final, fps):
    """DataFrame like a ball CSV (BALL_COLUMNS) from {frame: (x1, y1, x2, y2)}."""
    rows = []
    for frame, x1, y1, x2, y2, interp in fill_gaps(final, fps):
        x, y, w, h = int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))
        rows.append([frame, x, y, w, h, round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
                     w * h, interp])
    return pd.DataFrame(rows, columns=BALL_COLUMNS)


def player_boxes(rows):
    """{frame: {pid: (x, y, w, h)}} from PlayerTracker._fill_gaps rows, CSV-rounded."""
    boxes = defaultdict(dict)
    for r in rows:
        x, y = int(round(r["x0"])), int(round(r["y0"]))
        w, h = int(round(r["x1"] - r["x0"])), int(round(r["y1"] - r["y0"]))
        boxes[int(r["frame"])][int(r["pid"])] = (float(x), float(y), float(w), float(h))
    return boxes


def analyse_window(ball, boxes, conv, fps, hands):
    """(events, shots) of one window, with the offline functions."""
    real = real_detections_from_frame(ball)
    if len(real) < MIN_POINTS or not boxes:
        return None, None
    track = ball_track_from_frame(ball)
    serves = detect_serves(boxes, conv, fps, detect_hits(track, boxes, fps))
    events = detect_events(real, boxes, conv, fps, serves)
    shots = analyze_shots(track, boxes, conv, fps, hands, events=events, real_ball=real,
                          serves=serves)
    return events, shots


class OnlineAnalysis:
    """See the module docstring. maybe_run() once per frame; finish() at the end.

    fill_players(detections) -> rows turns the player detections into gap-filled
    rows (PlayerTracker._fill_gaps). threaded=False runs in the caller's thread.
    """

    def __init__(self, conv, fps, hands, fill_players, window_s=WINDOW_S, every_s=EVERY_S,
                 event_lag_s=EVENT_LAG_S, shot_lag_s=SHOT_LAG_S, threaded=True):
        self.conv, self.fps, self.hands = conv, fps, hands
        self.fill_players = fill_players
        self.window = int(round(window_s * fps))
        self.every = max(1, int(round(every_s * fps)))
        self.event_lag = int(round(event_lag_s * fps))
        self.shot_lag = int(round(shot_lag_s * fps))
        self.threaded = threaded
        self.events, self.shots = [], []     # committed rows (dicts)
        self._ev_done = self._sh_done = -10 ** 9   # frames up to these are decided
        self.runs = 0
        self.errors = 0
        self._last_run = None
        self._worker = None
        self._result = None
        self._lock = threading.Lock()
        self.new = []                        # rows committed by the last call (overlay)

    def _inputs(self, final, detections, lo, hi):
        ball = {f: b for f, b in final.items() if lo <= f <= hi}
        window = {pid: {f: r for f, r in d.items() if lo <= f <= hi}
                  for pid, d in detections.items()}
        return ball, window

    def _work(self, ball, window, upto):
        try:
            b = ball_frame(ball, self.fps)
            events, shots = analyse_window(b, player_boxes(self.fill_players(window)),
                                           self.conv, self.fps, self.hands)
        except Exception as exc:          # a bad window must not stop the live run
            self.errors += 1
            if self.errors == 1:
                traceback.print_exc()
                print(f"  online analysis failed on the window ending at frame {upto}: {exc}")
            events = shots = None
        with self._lock:
            self._result = (events, shots, upto)

    def maybe_run(self, upto, final, detections):
        """Commit a finished run, and start a new one every `every` frames."""
        self.new = []
        with self._lock:
            result, self._result = self._result, None
        if result is not None:
            self._commit(*result)
        if upto < 0 or (self._last_run is not None and upto - self._last_run < self.every):
            return self.new
        if self._worker is not None and self._worker.is_alive():
            return self.new
        self._last_run = upto
        ball, window = self._inputs(final, detections, upto - self.window, upto)
        self.runs += 1
        if self.threaded:
            self._worker = threading.Thread(target=self._work, args=(ball, window, upto),
                                            name="online-analysis", daemon=True)
            self._worker.start()
        else:
            self._work(ball, window, upto)
            with self._lock:
                result, self._result = self._result, None
            self._commit(*result)
        return self.new

    @staticmethod
    def _seen(committed, row, key):
        return any(abs(c["frame"] - row["frame"]) <= DEDUP_FRAMES and c[key] == row[key]
                   for c in committed)

    def _commit(self, events, shots, upto, final=False):
        """Commit the rows between the last boundary and the new one.

        Rows before the last boundary were decided by an earlier run (with more
        context on their left), so later runs cannot add to them.
        """
        ev_upto = upto if final else upto - self.event_lag
        sh_upto = upto if final else upto - self.shot_lag
        for table, rows, key, lo, hi, kind in (
                (self.events, events, "event", self._ev_done, ev_upto, "event"),
                (self.shots, shots, "player_id", self._sh_done, sh_upto, "shot")):
            if rows is None or not len(rows):
                continue
            for r in rows.to_dict("records"):
                if lo - DEDUP_FRAMES < r["frame"] <= hi and not self._seen(table, r, key):
                    table.append(r)
                    self.new.append((kind, r))
            table.sort(key=lambda r: r["frame"])
        self._ev_done = max(self._ev_done, ev_upto)
        self._sh_done = max(self._sh_done, sh_upto)

    def finish(self, upto, final, detections):
        """Last run, synchronous, committing everything up to `upto`."""
        if self._worker is not None:
            self._worker.join()
        with self._lock:
            result, self._result = self._result, None
        if result is not None:
            self._commit(*result)
        done = min((self._last_run or 0) - self.shot_lag, (self._last_run or 0) - self.event_lag)
        lo = max(0, done - self.window // 2) if self._last_run is not None else 0
        lo = min(lo, max(0, upto - self.window))
        ball, window = self._inputs(final, detections, lo, upto)
        b = ball_frame(ball, self.fps)
        events, shots = analyse_window(b, player_boxes(self.fill_players(window)),
                                       self.conv, self.fps, self.hands)
        self.new = []
        self._commit(events, shots, upto, final=True)

    def tables(self):
        """(events, bounces, shots) DataFrames of everything committed."""
        cols = ["frame", "cx", "cy", "x_m", "y_m", "jump_up_ms", "jump_side_ms",
                "fit_gap_px", "event"]
        events = pd.DataFrame(self.events, columns=cols if not self.events else None)
        if events.empty:
            events = pd.DataFrame(columns=cols)
        bounces = bounces_from_events(events) if len(events) else pd.DataFrame(
            columns=["frame", "cx", "cy", "x_m", "y_m", "jump_up_ms", "jump_side_ms",
                     "fit_gap_px", "side", "in_court", "reliable"])
        shots = pd.DataFrame(self.shots)
        return events, bounces, shots
