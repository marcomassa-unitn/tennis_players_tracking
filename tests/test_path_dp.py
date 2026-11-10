import numpy as np
import pandas as pd

from tracking.ball_tracking import (
    CANDIDATE_COLUMNS,
    OnlinePathDP,
    drop_outliers,
    drop_short_blob_runs,
    track_candidates,
)


def _rally(n=240, seed=1):
    """Candidates of a ball bouncing around, plus distractors and dropouts."""
    rng = np.random.default_rng(seed)
    rows = []
    x, y, vx, vy = 300.0, 200.0, 9.0, -6.0
    for f in range(n):
        vy += 0.5
        x, y = x + vx, y + vy
        if y > 800:
            vy = -abs(vy) * 0.8
        if x < 100 or x > 1800:
            vx = -vx
        if rng.random() > 0.15:                               # the ball, most frames
            rows.append((f, x - 7, y - 7, x + 7, y + 7, rng.uniform(0.3, 0.9), 0.5, "tile"))
        for _ in range(rng.integers(0, 3)):                   # still or random distractors
            dx, dy = rng.uniform(100, 1800), rng.uniform(100, 900)
            rows.append((f, dx, dy, dx + 14, dy + 14, rng.uniform(0.15, 0.5),
                         rng.uniform(0, 0.3), "tile"))
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _offline(cands, n):
    return drop_outliers(drop_short_blob_runs(track_candidates(cands, n, 30.0), set()))


def _online(cands, n, lag):
    dp = OnlinePathDP(30.0, lag=lag)
    by = {f: [tuple(r) for r in g.itertuples(index=False)] for f, g in cands.groupby("frame")}
    for f in range(n):
        dp.push(f, by.get(f, []))
    dp.flush()
    return dp.final


def test_an_unbounded_lag_is_the_offline_tracker():
    cands = _rally()
    assert _online(cands, 240, lag=10 ** 6) == _offline(cands, 240)


def test_a_short_lag_agrees_with_the_offline_tracker_on_almost_every_frame():
    cands = _rally(seed=3)
    off, on = _offline(cands, 240), _online(cands, 240, lag=10)
    keys = set(off) | set(on)
    same = sum(1 for k in keys if off.get(k) == on.get(k))
    assert same >= 0.95 * len(keys)


def test_positions_are_final_after_the_fixed_lag():
    cands = _rally(seed=5)
    dp = OnlinePathDP(30.0, lag=10, clean_lag=6)
    by = {f: [tuple(r) for r in g.itertuples(index=False)] for f, g in cands.groupby("frame")}
    for f in range(100):
        out = dp.push(f, by.get(f, []))
        assert all(g <= f - dp.latency for g in out)
    assert dp.finalised == 100 - 1 - dp.latency
