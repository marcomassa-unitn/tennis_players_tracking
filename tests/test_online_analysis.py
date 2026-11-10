import numpy as np
import pandas as pd

from analysis.bounce_detection import BOUNCE_PARAMS, _fit_parabola, _position, segment_trajectory
from analysis.online_analysis import OnlineAnalysis, ball_frame, player_boxes


def _segment_reference(t, pts, fps, p=BOUNCE_PARAMS):
    """The original loop of segment_trajectory, one lstsq per window."""
    n = len(t)
    cap = p["residual_cap_px"]
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n + 1, int)
    is_piece = np.zeros(n + 1, bool)
    for j in range(1, n + 1):
        best[j] = best[j - 1] + cap * cap
        prev[j] = j - 1
        for i in range(max(0, j - p["max_piece"]), j - p["min_piece"] + 1):
            if t[j - 1] - t[i] > p["max_piece_s"] * fps:
                continue
            fit = _fit_parabola(t[i:j], pts[i:j])
            err = np.linalg.norm(pts[i:j] - _position(fit, t[i:j]), axis=1)
            cost = best[i] + (np.minimum(err, cap) ** 2).sum() + p["piece_penalty"]
            if cost < best[j]:
                best[j], prev[j], is_piece[j] = cost, i, True
    pieces, j = [], n
    while j > 0:
        if is_piece[j]:
            pieces.append((prev[j], j))
        j = prev[j]
    return pieces[::-1]


def test_vectorised_segmentation_gives_the_same_pieces():
    rng = np.random.default_rng(2)
    t, pts, f = [], [], 0.0
    for _ in range(6):                                  # flights joined by bounces / hits
        v = rng.uniform(-12, 12, 2)
        start = pts[-1] if pts else np.array([900.0, 500.0])
        for k in range(int(rng.integers(12, 40))):
            if rng.random() < 0.1:
                f += 1
                continue                                # a missed detection
            pts.append(start + v * k + np.array([0.0, 0.25 * k * k]) + rng.normal(0, 1.5, 2))
            t.append(f)
            f += 1
    t, pts = np.array(t), np.array(pts)
    pts[rng.integers(0, len(pts), 4)] += 60             # outliers
    assert segment_trajectory(t, pts, 30.0) == _segment_reference(t, pts, 30.0)


def test_ball_frame_has_the_columns_of_a_ball_csv():
    df = ball_frame({0: (10.0, 10.0, 24.0, 24.0), 2: (14.0, 10.0, 28.0, 24.0)}, 30.0)
    assert list(df.columns) == ["frame", "x", "y", "w", "h", "cx", "cy", "area", "interpolated"]
    assert df.frame.tolist() == [0, 1, 2] and df.interpolated.tolist() == [0, 1, 0]


def test_player_boxes_round_like_the_csv():
    rows = [{"frame": 3, "pid": 1, "x0": 10.4, "y0": 20.6, "x1": 50.2, "y1": 120.9}]
    assert player_boxes(rows) == {3: {1: (10.0, 21.0, 40.0, 100.0)}}


class _Fake(OnlineAnalysis):
    """OnlineAnalysis whose windows return canned events."""

    def __init__(self, events):
        super().__init__(None, 30.0, {1: "right", 2: "right"}, lambda d: [],
                         every_s=1.0, event_lag_s=1.0, shot_lag_s=4.0, threaded=False)
        self.canned = events

    def _work(self, ball, window, upto):
        ev = pd.DataFrame([e for e in self.canned if e["frame"] <= upto])
        self._result = (ev if len(ev) else None, None, upto)


def test_events_are_committed_once_after_their_lag():
    fake = _Fake([{"frame": 40.2, "event": "bounce_far"}, {"frame": 100.0, "event": "hit_far"}])
    committed_at = {}
    for upto in range(0, 200):
        fake.maybe_run(upto, {}, {1: {}, 2: {}})
        for e in fake.events:
            committed_at.setdefault(e["frame"], upto)
    assert sorted(committed_at) == [40.2, 100.0]
    assert committed_at[40.2] >= 40.2 + fake.event_lag
    assert committed_at[100.0] >= 100.0 + fake.event_lag


def test_the_same_event_moved_by_a_frame_is_not_committed_twice():
    fake = _Fake([])                                   # event lag: 30 frames
    fake._commit(pd.DataFrame([{"frame": 50.0, "event": "bounce_near"}]), None, 82)
    fake._commit(pd.DataFrame([{"frame": 51.5, "event": "bounce_near"},     # the same one
                               {"frame": 53.0, "event": "hit_near"},        # a new one
                               {"frame": 30.0, "event": "hit_far"}]),       # already decided
                 None, 84)
    assert [(e["frame"], e["event"]) for e in fake.events] == [(50.0, "bounce_near"),
                                                               (53.0, "hit_near")]
