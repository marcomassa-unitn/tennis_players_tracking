import numpy as np
import pandas as pd

from tracking.ball_tracking import (
    MAX_GAP_S,
    drop_outliers,
    fill_gaps,
    tile_origins,
    track_candidates,
)

FPS = 30
N = 120


def _true_path(f):
    """A ball crossing the court and bouncing twice (piecewise parabolas)."""
    x = 300 + 9.0 * f
    t = f % 40
    y = 300 + 0.35 * (t - 20) ** 2
    return x, y


def _box(x, y, r=7):
    return [x - r, y - r, x + r, y + r]


def _candidates(rng):
    rows = []
    dropouts = set(rng.choice(np.arange(5, N - 5), 15, replace=False)) | set(range(60, 75))
    for f in range(N):
        if f not in dropouts:
            x, y = _true_path(f)
            rows.append([f, *_box(x + rng.normal(0, 1), y + rng.normal(0, 1)), 0.8, 0.4])
        # A ball held still by a ball kid: confident but never moving.
        rows.append([f, *_box(1500, 900), 0.9, 0.0])
        if f in (30, 31, 90):
            # One-frame false positives far from the ball.
            rows.append([f, *_box(200, 950), 0.95, 0.5])
    return pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "motion"]), dropouts


def test_tracker_follows_ball_and_ignores_distractors():
    rng = np.random.default_rng(1)
    cands, dropouts = _candidates(rng)
    track = drop_outliers(track_candidates(cands, N, FPS))
    assert len(track) >= 0.95 * (N - len(dropouts))
    for f, (x1, y1, x2, y2) in track.items():
        x, y = _true_path(f)
        assert np.hypot((x1 + x2) / 2 - x, (y1 + y2) / 2 - y) < 5, f


def test_gap_filling_is_short_and_never_at_the_edges():
    track = {f: tuple(_box(*_true_path(f))) for f in range(10, 100) if not 40 <= f < 70}
    track.pop(20)                      # short gap: 1 frame
    rows = fill_gaps(track, FPS)
    frames = {r[0]: r for r in rows}
    assert frames[20][5] == 1          # filled and flagged
    x, y = _true_path(20)
    assert np.hypot((frames[20][1] + frames[20][3]) / 2 - x, (frames[20][2] + frames[20][4]) / 2 - y) < 3
    assert 30 * MAX_GAP_S < 30         # the 30-frame gap below is longer than MAX_GAP_S
    assert not any(40 <= f < 70 for f in frames)     # long gap left empty
    assert min(frames) == 10 and max(frames) == 99   # no extrapolation at the edges


def test_spike_is_dropped_but_fast_straight_motion_is_kept():
    track = {f: tuple(_box(100 + 10 * f, 500)) for f in range(20)}
    track[10] = tuple(_box(100 + 10 * 10, 500 - 130))     # out and back: a spike
    track[15] = tuple(_box(100 + 10 * 15 + 3, 500))       # small jitter: kept
    clean = drop_outliers(track)
    assert 10 not in clean and 15 in clean and len(clean) == 19


def test_tiles_cover_the_frame_with_overlap():
    origins = tile_origins(1920, 1080)
    assert len(origins) == 8
    xs = sorted({x for x, _ in origins})
    assert xs[0] == 0 and xs[-1] + 640 == 1920
    assert all(b - a <= 640 - 48 for a, b in zip(xs, xs[1:]))
