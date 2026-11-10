"""End to end: the real-time pipeline on a synthetic clip, without a model."""

import csv
import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest

from live_view import load_ball_boxes
from realtime.pipeline import RealtimePipeline
from tracking.court_tracking import keypoints_from_homography
from utils.court_overlay import MODEL_SEGMENTS
from utils.paths import clip_paths

W, H, FPS = 960, 540, 30
N_FRAMES = 150


def _homography():
    model = np.float32([[-1.37, 0], [9.6, 0], [9.6, 23.77], [-1.37, 23.77]])
    image = np.float32([[340, 120], [620, 120], [800, 460], [160, 460]])
    return cv2.getPerspectiveTransform(model, image)


def _box(Hm, x_m, y_m, height_m=1.8, width_m=0.6):
    feet = cv2.perspectiveTransform(np.float64([[[x_m, y_m]]]), Hm)[0, 0]
    side = cv2.perspectiveTransform(np.float64([[[x_m + 1.0, y_m]]]), Hm)[0, 0]
    ppm = np.hypot(*(side - feet))
    w, h = width_m * ppm, height_m * ppm
    return int(feet[0] - w / 2), int(feet[1] - h), int(feet[0] + w / 2), int(feet[1])


def _ball(t):
    """Ball centre (px): across the court and back, never near the players."""
    u = (t % 60) / 60.0
    x = 400 + 160 * (u if (t // 60) % 2 == 0 else 1 - u)
    return x, 260 + 40 * np.sin(np.pi * u)


@pytest.fixture
def clip(tmp_path):
    Hm = _homography()
    rng = np.random.default_rng(0)
    court = np.full((H, W, 3), (70, 140, 60), np.uint8)
    quad = cv2.perspectiveTransform(np.float64([[[0, 0]], [[8.23, 0]], [[8.23, 23.77]],
                                                [[0, 23.77]]]), Hm)
    cv2.fillPoly(court, [quad[:, 0].astype(np.int32)], (150, 90, 60))
    for a, b in MODEL_SEGMENTS:
        p = cv2.perspectiveTransform(np.float64([[a], [b]]), Hm)[:, 0]
        cv2.line(court, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (235, 235, 235), 2)
    texture = cv2.GaussianBlur(rng.normal(0, 20, (H, W)), (0, 0), 2)[..., None]
    court = np.clip(court + texture, 0, 255).astype(np.uint8)

    video = tmp_path / "rt_clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for t in range(N_FRAMES):
        frame = court.copy()
        phase = 2 * np.pi * t / N_FRAMES
        for (x_m, y_m), color in (((4.1 + 3 * np.sin(phase), 22.0), (180, 60, 230)),
                                  ((4.1 - 3 * np.sin(phase), 1.5), (40, 40, 200))):
            x0, y0, x1, y1 = _box(Hm, x_m, y_m)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
        bx, by = _ball(t)
        cv2.circle(frame, (int(bx), int(by)), 4, (60, 230, 230), -1)
        frame = np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()

    court_csv = tmp_path / "court.csv"
    with open(court_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "x", "y"])
        for label, (x, y) in keypoints_from_homography(Hm).items():
            w.writerow([label, x, y])
    return video, court_csv


def test_realtime_pipeline_writes_the_offline_outputs(clip, tmp_path):
    video, court_csv = clip
    out = tmp_path / "out"
    summary = RealtimePipeline(str(video), str(out), "diff", court_csv=str(court_csv),
                               log_every=0).run()
    p = clip_paths(str(video), str(out))

    players = pd.read_csv(p["players_csv"])
    assert list(players.columns)[-1] == "interpolated"
    assert {1, 2} <= set(players.player_id)

    ball = pd.read_csv(p["ball_csv"])
    assert list(ball.columns) == ["frame", "x", "y", "w", "h", "cx", "cy", "area",
                                  "interpolated"]
    real = ball[ball.interpolated == 0]
    assert len(real) >= 0.6 * (N_FRAMES - 60)           # found once the background is ready
    err = [np.hypot(r.cx - _ball(r.frame)[0], r.cy - _ball(r.frame)[1]) for r in real.itertuples()]
    assert np.median(err) < 3
    assert load_ball_boxes(p["ball_csv"])                # live_view reads it

    for key in ("events_csv", "bounces_csv", "shots_csv", "ball_candidates_csv"):
        assert os.path.exists(p[key]), key

    saved = json.loads((out / "rt_clip" / "realtime" / "summary.json").read_text())
    assert saved["frames"] == N_FRAMES == summary["frames"]
    assert {"players", "ball", "ball_path", "background"} <= set(saved["stages"])
    assert saved["ball_lag_frames"] == 16
    timings = pd.read_csv(out / "rt_clip" / "realtime" / "timings.csv")
    assert set(timings.columns) == {"frame", "stage", "ms"}
