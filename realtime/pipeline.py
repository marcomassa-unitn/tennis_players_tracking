"""The whole analysis in one pass over the frames, as they arrive.

One decoding thread feeds one processing loop; the slow, occasional jobs run in
worker threads (court detection at startup, background refreshes, the windowed
bounce and shot analysis). Per frame, in order:

    1. players     PlayerTracker.process_frame against the online background
    2. ball        OnlineBallTracker (model crops) + OnlinePathDP (fixed lag)
    3. background  OnlineBackground.feed, with the players masked out
    4. analysis    OnlineAnalysis: bounces and shots, committed with a delay
    5. overlay     live view and/or video, with a timing HUD

Startup: the court is fitted on the first frames (or read from --court) and the
background needs WARMUP_S seconds; frames before that are counted but not tracked.

The results are the CSVs of the offline pipeline, in the same layout under
outputs/realtime/<clip>/, so live_view.py and evaluation/ read them unchanged:
court, players, ball (+ the candidates of the model), events, bounces, shots, and
realtime/timings.csv + summary.json.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from analysis.online_analysis import OnlineAnalysis
from realtime.overlay import Overlay, VideoSink
from realtime.source import FrameSource
from realtime.timing import StageTimer
from tracking.ball_model import CROP, load_backend
from tracking.ball_online import OnlineBallTracker, court_roi
from tracking.ball_tracking import (
    CANDIDATE_COLUMNS,
    OnlinePathDP,
    write_ball_csv,
)
from tracking.court_tracking import (
    GOOD_MATCH_RATIO,
    MIN_MATCH_RATIO,
    court_match_ratio,
    detect_court,
    save_court,
)
from tracking.player_tracking import PROC_HEIGHT, PlayerTracker
from utils.background import OnlineBackground, camera_shift
from utils.court_converter import CourtConverter
from utils.paths import clip_paths

COURT_TRY_EVERY_S = 0.5    # startup: a new frame for the court fit this often...
COURT_MAX_TRIES = 6        # ...this many times, keeping the best
COURT_CHECK_S = 5.0        # then the fit is checked this often...
COURT_LOST_CHECKS = 2      # ...and fitted again after this many bad checks
SHIFT_EVERY = 3            # camera shift measured once every this many frames
LUMA_STEP = 4              # frame brightness from every 4th pixel
PLAYER_RECENT = 3          # frames a player box stays "current" for the ball tracker
TRAIL = 20                 # final ball positions drawn on the overlay


def uses_gpu(backend):
    if backend.name == "onnx":
        return "CUDAExecutionProvider" in backend.session.get_providers()
    if backend.name == "ultralytics":
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    return False


class CourtWorker:
    """Fits the court on startup frames in a thread, then checks it now and then."""

    def __init__(self):
        self.best = (None, 0.0, None)      # (H, ratio, frame)
        self.tries = 0
        self.thread = None
        self.done = False
        self.check_ratio = None
        self.bad_checks = 0

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def offer(self, frame, wait=False):
        """Try a fit on `frame`, in the worker (or right here if `wait`)."""
        if self.busy() or self.done:
            return

        def work():
            H, ratio = detect_court(frame)
            self.tries += 1
            if H is not None and ratio > self.best[1]:
                self.best = (H, ratio, frame)
            if self.best[1] >= GOOD_MATCH_RATIO or (self.tries >= COURT_MAX_TRIES
                                                    and self.best[1] >= MIN_MATCH_RATIO):
                self.done = True
        if wait:
            work()
            return
        self.thread = threading.Thread(target=work, name="court-fit", daemon=True)
        self.thread.start()

    def check(self, frame, H):
        if self.busy():
            return

        def work():
            self.check_ratio = court_match_ratio(frame, H)
            self.bad_checks = self.bad_checks + 1 if self.check_ratio < MIN_MATCH_RATIO else 0
        self.thread = threading.Thread(target=work, name="court-check", daemon=True)
        self.thread.start()


class RealtimePipeline:
    def __init__(self, video, output="outputs/realtime", model=None, court_csv=None,
                 players_csv=None, clock="all", search=None, search_every=None,
                 display=False, video_out=None, max_frames=None, hands=("right", "right"),
                 shots=True, figures=False, device=None, log_every=150, background=None,
                 fallback=None):
        self.video = video
        self.paths = clip_paths(video, output)
        self.model = model
        self.court_csv_in = court_csv
        self.players_csv_in = players_csv          # ball-only mode: offline players
        self.clock = clock
        self.search = search
        self.search_every = search_every
        self.display = display
        self.video_out = video_out
        self.max_frames = max_frames
        self.hands = {1: hands[0], 2: hands[1]}
        self.with_shots = shots
        self.figures = figures
        self.device = device
        self.log_every = log_every
        self.background_png = background             # calibration: no warm-up
        self.fallback = fallback                     # colour detector by the players

    # ── setup ────────────────────────────────────────────────────────────────

    def _court_ready(self, H, frame):
        """Everything that needs the court: players, ROI, overlay, analysis."""
        p = self.paths
        os.makedirs(os.path.dirname(p["court_csv"]), exist_ok=True)
        if self.court_csv_in and os.path.abspath(self.court_csv_in) != os.path.abspath(p["court_csv"]):
            shutil.copy(self.court_csv_in, p["court_csv"])
        elif frame is not None:
            save_court(H, frame, p["court_csv"])
        self.H = H
        self.conv = CourtConverter(p["court_csv"])
        self.bg.set_roi(court_roi(self.conv, self.src.width, self.src.height))
        if self.players_csv_in:
            df = pd.read_csv(self.players_csv_in)
            self.offline_players = {f: {int(r.player_id): (r.x, r.y, r.w, r.h)
                                        for r in g.itertuples()} for f, g in df.groupby("frame")}
        else:
            os.makedirs(os.path.dirname(p["players_csv"]), exist_ok=True)
            pt = PlayerTracker(self.video, p["players_csv"], p["court_csv"], display=False)
            pt.shift_every, pt.luma_step, pt.shift_downscale = SHIFT_EVERY, LUMA_STEP, 2
            self.players = pt
        self.overlay.set_court(H)
        if self.with_shots and self.players is not None:
            self.analysis = OnlineAnalysis(self.conv, self.src.fps, self.hands,
                                           self.players._fill_gaps)
        self.timer.mark("court_ready")

    def _current_players(self, idx):
        """{pid: (x, y, w, h)} full-res boxes of the players at frame idx."""
        if self.offline_players is not None:
            return self.offline_players.get(idx, {})
        if self.players is None:
            return {}
        out = {}
        for pid, d in self.players.detections.items():
            for f in range(idx, idx - PLAYER_RECENT, -1):
                r = d.get(f)
                if r is not None:
                    out[pid] = (r["x0"], r["y0"], r["x1"] - r["x0"], r["y1"] - r["y0"])
                    break
        return out

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        self.timer = StageTimer()
        self.src = FrameSource(self.video, self.clock, self.max_frames)
        fps, W, H = self.src.fps, self.src.width, self.src.height
        proc = (int(round(W * PROC_HEIGHT / H)), PROC_HEIGHT) if H > PROC_HEIGHT else (W, H)
        self.bg = OnlineBackground(fps, (W, H), proc, roi=None, defer_roi=True)
        self.hann = cv2.createHanningWindow(proc, cv2.CV_32F)
        self.overlay = Overlay(W, H)
        self.players = None
        self.offline_players = None
        self.analysis = None
        self.H = None
        self.court = CourtWorker()

        backend = load_backend(self.model, device=self.device)
        gpu = uses_gpu(backend)
        search = self.search or ("tiles" if gpu else "blobs")
        search_every = self.search_every or (1 if gpu else 2)
        fallback = load_backend(self.fallback, device=self.device) if self.fallback else None
        self.ball = OnlineBallTracker(backend, self.bg.scale, H, search, search_every,
                                      fallback=fallback)
        self.path = OnlinePathDP(fps, H)
        candidates = []
        self.live_rows = []                 # zero-lag Kalman estimate, per frame
        trail = []
        sink = None
        if self.video_out:
            os.makedirs(os.path.dirname(self.video_out) or ".", exist_ok=True)
            sink = VideoSink(self.video_out, fps, self.overlay.size)
        print(f"Real-time pipeline on {self.video} ({W}x{H} @ {fps:g} fps, clock={self.clock})")
        print(f"  ball model: {self.model or 'none (difference blobs)'} [{backend.name}, "
              f"{'GPU' if gpu else 'CPU'}], search={search} every {search_every} frame(s)")

        if self.court_csv_in:
            conv = CourtConverter(self.court_csv_in)
            self._court_ready(np.linalg.inv(conv._H), None)
        if self.background_png:
            image = cv2.imread(self.background_png)
            if image is None:
                raise FileNotFoundError(f"Background not found: {self.background_png}")
            self.bg.seed(image)
            self.timer.mark("background_ready")
            print(f"  background from {self.background_png}: tracking from the first frame")

        court_every = max(1, int(round(COURT_TRY_EVERY_S * fps)))
        check_every = max(1, int(round(COURT_CHECK_S * fps)))
        last_court_try = -10 ** 9
        shift = (0.0, 0.0)
        boxes = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ball")
        stopped = False
        idx = -1
        self.src.start()
        try:
            for idx, frame, t_dec in self.src:
                t0 = time.perf_counter()
                tm = self.timer
                with tm.stage(idx, "resize"):
                    small = cv2.resize(frame, proc, interpolation=cv2.INTER_AREA)

                # Court: startup fit in a worker, then a check every few seconds.
                # With clock=all the fit blocks (it counts as startup time); live, it
                # runs beside the loop and the frames meanwhile go untracked.
                if self.H is None:
                    if not self.court.done and idx - last_court_try >= court_every                             and not self.court.busy():
                        last_court_try = idx
                        with tm.stage(idx, "court_fit"):
                            self.court.offer(frame, wait=self.clock == "all")
                    if self.court.done:
                        Hc, ratio, fr = self.court.best
                        print(f"  court fitted at {ratio:.0%} match (frame {idx})")
                        self._court_ready(Hc, fr)
                elif idx % check_every == 0:
                    if self.court.bad_checks >= COURT_LOST_CHECKS and "court_lost" not in tm.marks:
                        # The fixed-camera assumption broke (a cut, a replay): say so.
                        print(f"  WARNING frame {idx}: the court no longer matches "
                              f"({self.court.check_ratio:.0%}); results after it are unreliable")
                        tm.mark("court_lost")
                    self.court.check(frame, self.H)

                # Ball and players of this frame run side by side (OpenCV and
                # onnxruntime release the GIL); the ball uses the camera shift, the
                # player boxes and the difference of the previous frame.
                snap = self.bg.snapshot
                ball_job = None
                if snap is not None and snap.full is not None:
                    diff = self.players.last_diff if self.players is not None else None
                    ball_job = self.pool.submit(self._ball_step, idx, frame, snap, shift,
                                                list(boxes.values()), diff)
                if snap is not None and self.players is not None and self.H is not None:
                    with tm.stage(idx, "players"):
                        self.players.use_snapshot(snap)
                        self.players.process_frame(idx, frame, small, k=0)
                    shift = self.players.shift
                elif snap is not None and idx % SHIFT_EVERY == 0:
                    with tm.stage(idx, "shift"):
                        shift = camera_shift(snap.small_gray, small, self.hann)
                boxes = self._current_players(idx)

                live, rows = None, []
                if ball_job is not None:
                    with tm.stage(idx, "ball_wait"):
                        live, rows, ms = ball_job.result()
                    tm.rows.append((idx, "ball", ms))
                    with tm.stage(idx, "ball_path"):
                        finals = self.path.push(idx, rows)
                    candidates += rows
                    if live is not None:
                        self.live_rows.append((idx, round(live["cx"], 1), round(live["cy"], 1),
                                               int(not live["detected"])))
                    for b in finals.values():
                        trail = (trail + [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)])[-TRAIL:]

                with tm.stage(idx, "background"):
                    self.bg.feed(idx, frame, small, players=list(boxes.values()), shift=shift)
                if self.bg.snapshot is not None and "background_ready" not in tm.marks:
                    tm.mark("background_ready")
                if self.bg.snapshot is not None and self.H is not None:
                    tm.ready()

                if self.analysis is not None:
                    with tm.stage(idx, "analysis"):
                        new = self.analysis.maybe_run(self.path.finalised, self.path.final,
                                                      self.players.detections)
                    for kind, r in new:
                        if kind == "event":
                            self.overlay.add_event(f"{r['frame'] / fps:6.1f}s  {r['event']}")
                        else:
                            self.overlay.add_event(f"{r['frame'] / fps:6.1f}s  P{r['player_id']} "
                                                   f"{r['stroke']} {r['shot_type']}")

                if self.display or sink is not None:
                    with tm.stage(idx, "overlay"):
                        hud = [f"{tm.recent_fps():5.1f} fps  ball:{self.ball.state}  "
                               f"crops:{len(self.ball.last_crops)}",
                               "  ".join(f"{k} {v:.0f}" for k, v in tm._current.items())]
                        if self.H is None:
                            hud.append("fitting the court...")
                        elif snap is None:
                            hud.append("learning the background...")
                        img = self.overlay.draw(frame, idx, boxes, live, trail,
                                                self.ball.last_crops, CROP, hud)
                    if sink is not None:
                        sink.write(img)
                    if self.display:
                        cv2.imshow("Real-time pipeline (q: quit)", img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stopped = True
                tm.end_frame(idx, t0, t_dec)
                if self.log_every and idx % self.log_every == 0 and idx:
                    print(f"  frame {idx}: {tm.recent_fps():.1f} fps, ball {self.ball.state}, "
                          f"{len(self.path.final)} ball positions final")
                if stopped:
                    break
        finally:
            self.pool.shutdown(wait=True)
            self.src.stop()
            if sink is not None:
                sink.close()
            if self.display:
                cv2.destroyAllWindows()
        return self._finish(idx, candidates)

    def _ball_step(self, idx, frame, snap, shift, players, diff):
        t = time.perf_counter()
        live, rows = self.ball.step(idx, frame, snap, shift, players, diff)
        return live, rows, (time.perf_counter() - t) * 1000.0

    # ── outputs ──────────────────────────────────────────────────────────────

    def _finish(self, last, candidates):
        p = self.paths
        fps = self.src.fps
        self.timer.mark("end")
        self.path.flush()
        os.makedirs(os.path.dirname(p["ball_csv"]), exist_ok=True)
        write_ball_csv(self.path.rows(), p["ball_csv"])
        pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS).to_csv(p["ball_candidates_csv"],
                                                                   index=False)
        live_csv = os.path.join(p["clip_dir"], "realtime", f"ball_live_{p['stem']}.csv")
        os.makedirs(os.path.dirname(live_csv), exist_ok=True)
        pd.DataFrame(self.live_rows, columns=["frame", "cx", "cy", "interpolated"]).to_csv(
            live_csv, index=False)
        written = {"ball": p["ball_csv"], "ball_live": live_csv}
        if self.players is not None:
            self.players._write_csv(self.players._fill_gaps(self.players.detections))
            written["players"] = p["players_csv"]
        if self.H is not None:
            written["court"] = p["court_csv"]
        if self.analysis is not None:
            self.analysis.finish(self.path.finalised, self.path.final,
                                 self.players.detections)
            events, bounces, shots = self.analysis.tables()
            os.makedirs(p["shots_dir"], exist_ok=True)
            events.to_csv(p["events_csv"], index=False)
            bounces.to_csv(p["bounces_csv"], index=False)
            shots.to_csv(p["shots_csv"], index=False)
            written["shots"] = p["shots_csv"]
        if self.figures and self.players is not None:
            self._figures()

        extra = {
            "video": self.video,
            "ball_model": self.model or "diff",
            "ball_backend": self.ball.backend.name,
            "ball_search": self.ball.search,
            "ball_crops_per_frame": round(self.ball.n_crops / max(len(self.timer.frame_ms), 1), 2),
            "ball_fallback": self.fallback,
            "ball_fallback_crops_per_frame": round(
                self.ball.n_fallback / max(len(self.timer.frame_ms), 1), 3),
            "ball_positions_final": len(self.path.final),
            "ball_lag_frames": self.path.latency,
            "event_lag_s": None if self.analysis is None else self.analysis.event_lag / fps,
            "shot_lag_s": None if self.analysis is None else self.analysis.shot_lag / fps,
            "events": 0 if self.analysis is None else len(self.analysis.events),
            "shots": 0 if self.analysis is None else len(self.analysis.shots),
            "background_refreshes": self.bg._version,
            "frames_in_clip": last + 1,
        }
        summary = self.timer.summary(fps, self.src.dropped, extra)
        out = self.timer.save(os.path.join(p["clip_dir"], "realtime"), summary)
        written["timings"] = out
        self._print_summary(summary, written)
        return summary

    def _figures(self):
        from analysis import player_analysis
        from pipeline import _override_argv
        p = self.paths
        os.makedirs(p["analysis_dir"], exist_ok=True)
        with _override_argv(["player_analysis", "--players", p["players_csv"], "--court",
                             p["court_csv"], "--fps", str(self.src.fps), "--output",
                             p["analysis_dir"], "--no-animation"]):
            player_analysis.main()

    @staticmethod
    def _print_summary(s, written):
        verdict = "real time" if s["real_time"] else "NOT real time"
        print(f"\nProcessed {s['frames']} frames; after the startup {s['steady_fps']} fps "
              f"(video {s['video_fps']:g} fps: {verdict}, {s['dropped_frames']} dropped); "
              f"frame time p50 {s['frame_ms']['p50']} ms, p95 {s['frame_ms']['p95']} ms")
        for name, st in s["stages"].items():
            print(f"  {name:11s} mean {st['mean_ms']:7.2f} ms   p95 {st['p95_ms']:7.2f} ms"
                  f"   ({st['calls']} calls)")
        print(f"  startup: {s['marks_s']}")
        print(f"  ball: {s['ball_positions_final']} final positions, {s['ball_crops_per_frame']} "
              f"crops/frame; {s['events']} events, {s['shots']} shots")
        for k, v in written.items():
            print(f"  {k:8s} -> {v}")
