import csv

import cv2
import numpy as np
import pandas as pd
import pytest

from tracking.court_tracking import keypoints_from_homography
from tracking.player_tracking import PlayerTracker
from utils.court_converter import CourtConverter
from utils.court_geometry import NET
from utils.court_overlay import MODEL_SEGMENTS

W, H, FPS = 960, 540, 30
N_FRAMES = 90
FADE_FRAMES = 6                 # black -> full brightness, like a broadcast fade-in
STILL = range(40, 52)           # both players stand still (e.g. before a serve)


def _court_homography():
    model = np.float32([[-1.37, 0], [9.6, 0], [9.6, 23.77], [-1.37, 23.77]])
    image = np.float32([[340, 120], [620, 120], [800, 460], [160, 460]])
    return cv2.getPerspectiveTransform(model, image)


def _player_box(Hm, x_m, y_m, height_m=1.8, width_m=0.6):
    """Image box of a standing player whose feet are at court point (x_m, y_m)."""
    feet = cv2.perspectiveTransform(np.float64([[[x_m, y_m]]]), Hm)[0, 0]
    side = cv2.perspectiveTransform(np.float64([[[x_m + 1.0, y_m]]]), Hm)[0, 0]
    ppm = np.hypot(*(side - feet))
    w, h = width_m * ppm, height_m * ppm
    return int(feet[0] - w / 2), int(feet[1] - h), int(feet[0] + w / 2), int(feet[1])


def _positions(t):
    """Feet positions (metres) of P1 (near) and P2 (far) at frame t."""
    t_eff = STILL.start if t in STILL else t
    phase = 2 * np.pi * t_eff / N_FRAMES
    p1 = (4.1 + 3.0 * np.sin(phase), 22.0 + 1.5 * np.cos(phase))
    p2 = (4.1 - 3.0 * np.sin(phase), 1.5 + 1.0 * np.cos(phase))
    return p1, p2


@pytest.fixture
def clip(tmp_path):
    Hm = _court_homography()
    court = np.full((H, W, 3), (70, 140, 60), np.uint8)              # green run-off
    quad = cv2.perspectiveTransform(np.float64([[[0, 0]], [[8.23, 0]], [[8.23, 23.77]], [[0, 23.77]]]), Hm)
    cv2.fillPoly(court, [quad[:, 0].astype(np.int32)], (150, 90, 60))  # blue court
    for a, b in MODEL_SEGMENTS:
        p = cv2.perspectiveTransform(np.float64([[a], [b]]), Hm)[:, 0]
        cv2.line(court, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (235, 235, 235), 2)

    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    rng = np.random.default_rng(0)
    for t in range(N_FRAMES):
        frame = court.copy()
        for (x_m, y_m), color in zip(_positions(t), ((180, 60, 230), (40, 40, 200))):
            x0, y0, x1, y1 = _player_box(Hm, x_m, y_m)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
        noise = rng.normal(0, 3, frame.shape)                        # sensor noise
        frame = np.clip(frame + noise, 0, 255)
        frame *= min(1.0, t / FADE_FRAMES)                           # fade-in
        writer.write(frame.astype(np.uint8))
    writer.release()

    court_csv = tmp_path / "court.csv"
    with open(court_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "x", "y"])
        for label, (x, y) in keypoints_from_homography(Hm).items():
            w.writerow([label, x, y])
    return video, court_csv


def test_tracks_both_players_from_fade_in_and_while_still(clip, tmp_path):
    video, court_csv = clip
    out = tmp_path / "players.csv"
    PlayerTracker(str(video), str(out), str(court_csv), display=False).run()
    d = pd.read_csv(out)
    assert list(d.columns)[-1] == "interpolated"

    real = d[d.interpolated == 0]
    usable = range(FADE_FRAMES // 2 + 1, N_FRAMES)
    for pid in (1, 2):
        frames = set(real[real.player_id == pid].frame)
        coverage = len(frames & set(usable)) / len(usable)
        assert coverage >= 0.95, (pid, coverage)
        assert set(STILL) <= frames, f"P{pid} lost while standing still"

    conv = CourtConverter(str(court_csv))
    feet_y = np.array([conv.to_meters(r.x + r.w / 2, r.y + r.h)[1] for r in d.itertuples()])
    assert np.all(feet_y[d.player_id.values == 1] > NET)      # P1 = near half
    assert np.all(feet_y[d.player_id.values == 2] < NET)      # P2 = far half


def test_missing_court_csv_is_reported(clip, tmp_path):
    video, _ = clip
    with pytest.raises(FileNotFoundError, match="court_tracking.py"):
        PlayerTracker(str(video), str(tmp_path / "p.csv"), str(tmp_path / "nope.csv"))
