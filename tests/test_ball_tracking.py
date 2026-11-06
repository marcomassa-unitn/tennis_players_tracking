import numpy as np
import pandas as pd

from tracking.ball_tracking import (
    MAX_GAP_S,
    ball_sized_blobs,
    drop_outliers,
    drop_short_blob_runs,
    fill_gaps,
    tile_origins,
    track_candidates,
    wholeframe_track,
    within_reach,
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


def test_wholeframe_track_follows_the_original_rules():
    rows = [
        [0, *_box(100, 100), 0.70, 0.0, "frame"],    # first frame: no motion gate
        [1, *_box(110, 100), 0.90, 0.3, "tile"],     # tile detections are ignored
        [1, *_box(500, 500), 0.60, 0.3, "frame"],    # below the original 0.65
        [2, *_box(120, 100), 0.80, 0.0, "frame"],    # still: rejected
        [3, *_box(130, 100), 0.75, 0.4, "frame"],
        [3, *_box(900, 100), 0.95, 0.4, "frame"],    # most confident wins
    ]
    cands = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "motion", "source"])
    track = wholeframe_track(cands, 5)
    centre = {f: ((x1 + x2) / 2, interp) for f, x1, y1, x2, y2, interp in track}
    assert centre[0] == (100, 0) and centre[3] == (900, 0)
    assert centre[1][1] == 1 and centre[2][1] == 1           # filled with a straight line
    assert centre[1][0] == 100 + (900 - 100) / 3
    assert centre[4] == (900, 1)                             # edge filled with the last position
    assert wholeframe_track(cands.drop(columns="source"), 5) is None


def test_tiles_cover_the_frame_with_overlap():
    origins = tile_origins(1920, 1080)
    assert len(origins) == 8
    xs = sorted({x for x, _ in origins})
    assert xs[0] == 0 and xs[-1] + 640 == 1920
    assert all(b - a <= 640 - 48 for a, b in zip(xs, xs[1:]))


def _frame_with(*rects):
    frame = np.zeros((200, 300, 3), np.uint8)
    for x, y, w, h in rects:
        frame[y:y + h, x:x + w] = 255
    return frame


def test_only_ball_sized_blobs_are_kept():
    background = _frame_with()
    frame = _frame_with((50, 50, 4, 4),        # a ball
                        (150, 50, 30, 30),     # a person
                        (250, 150, 1, 2),      # noise
                        (50, 150, 40, 2))      # a court line
    assert ball_sized_blobs(frame, background) == [(50, 50, 54, 54)]


def test_a_blurred_ball_is_a_blob_too():
    assert ball_sized_blobs(_frame_with((50, 50, 12, 3)), _frame_with()) == [(50, 50, 62, 53)]


def test_a_blob_within_reach_of_a_player_is_their_racket():
    player = [(100.0, 100.0, 40.0, 100.0)]
    assert within_reach(120, 150, player)
    assert within_reach(50, 120, player)        # an arm's length to the side
    assert not within_reach(20, 120, player)
    assert not within_reach(120, 260, player)   # well below the feet


def test_a_stray_blob_on_the_path_is_dropped_and_a_series_is_kept():
    box = (10.0, 20.0, 14.0, 24.0)
    track = {1: (1.0, 1.0, 5.0, 5.0), 2: (2.0, 2.0, 6.0, 6.0), 3: (3.0, 3.0, 7.0, 7.0),
             20: box, 40: box, 41: box, 42: box}
    blobs = {(20, 10.0, 20.0), (40, 10.0, 20.0), (41, 10.0, 20.0), (42, 10.0, 20.0)}
    assert sorted(drop_short_blob_runs(track, blobs)) == [1, 2, 3, 40, 41, 42]


def test_a_short_run_of_the_detector_stays():
    track = {5: (1.0, 1.0, 5.0, 5.0)}
    assert drop_short_blob_runs(track, set()) == track
