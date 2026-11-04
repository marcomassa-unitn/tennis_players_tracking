import numpy as np
import pandas as pd
import pytest

from analysis.bounce_detection import (
    BOUNCE_FAR,
    BOUNCE_NEAR,
    HIT_FAR,
    HIT_NEAR,
    _px_per_metre,
    detect_bounces,
    detect_events,
    load_real_detections,
)
from analysis.shot_analysis import _synthetic_court_csv, shot_flight
from utils.court_converter import CourtConverter

FPS = 30.0
G = 9.81

# (frame, x, y, z) in metres: the ball flies freely between consecutive points.
RALLY = [
    (10.0, 4.0, 23.0, 1.0),     # near hit
    (40.4, 5.5, 6.0, 0.0),      # far bounce
    (52.0, 6.0, 0.5, 1.0),      # far hit
    (82.6, 2.5, 19.0, 0.0),     # near bounce
    (94.0, 2.0, 23.5, 1.0),     # near hit
    (124.3, 6.0, 4.0, 0.0),     # far bounce
    (136.0, 6.5, -0.5, 1.0),    # far hit
    (160.0, 4.0, 11.0, 1.5),
]
BOUNCES = [RALLY[1], RALLY[3], RALLY[5]]


@pytest.fixture(scope="module")
def conv(tmp_path_factory):
    path = tmp_path_factory.mktemp("court") / "court.csv"
    _synthetic_court_csv(str(path))
    return CourtConverter(str(path))


def _project(conv, x, y, z):
    """Image position of a ball at height z: the ground point, moved up the image."""
    m_to_px = np.linalg.inv(conv._H)
    u, v, w = m_to_px @ np.array([x, y, 1.0])
    return u / w, v / w - z * _px_per_metre(conv, x, y)


def _ball(conv, points, seed=0, noise=1.0):
    """Detections of a ball following `points`, one per frame, with pixel noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for (f0, x0, y0, z0), (f1, x1, y1, z1) in zip(points[:-1], points[1:]):
        span = (f1 - f0) / FPS
        v0 = (z1 - z0 + 0.5 * G * span ** 2) / span
        for f in range(int(np.ceil(f0)), int(np.ceil(f1))):
            s = (f - f0) / FPS
            r = s / span
            z = z0 + v0 * s - 0.5 * G * s ** 2
            u, v = _project(conv, x0 + r * (x1 - x0), y0 + r * (y1 - y0), z)
            rows.append((f, u + rng.normal(0, noise), v + rng.normal(0, noise)))
    return pd.DataFrame(rows, columns=["frame", "cx", "cy"])


def _boxes(conv, points, near_at, far_at):
    """Both players standing where they strike, on every frame of the clip."""
    def box(x, y, w, h):
        u, v = _project(conv, x, y, 0.0)
        return (u - w / 2, v - h, w, h)

    last = int(points[-1][0])
    out = {}
    for f in range(last + 1):
        near = next((p for p in near_at if p[0] >= f), near_at[-1])
        far = next((p for p in far_at if p[0] >= f), far_at[-1])
        out[f] = {1: box(near[1], near[2], 90, 200), 2: box(far[1], far[2], 40, 90)}
    return out


def _rally_boxes(conv):
    return _boxes(conv, RALLY, [RALLY[0], RALLY[4]], [RALLY[2], RALLY[6]])


def _assert_found(bounces, conv):
    assert len(bounces) == len(BOUNCES)
    for row, (frame, x, y, _) in zip(bounces.itertuples(), BOUNCES):
        assert abs(row.frame - frame) <= 1.0
        assert np.hypot(row.x_m - x, row.y_m - y) <= 0.5


def test_finds_every_bounce_of_a_rally(conv):
    bounces = detect_bounces(_ball(conv, RALLY), _rally_boxes(conv), conv, FPS)
    _assert_found(bounces, conv)
    assert list(bounces["side"]) == ["far", "near", "far"]
    assert bounces["in_court"].all()


def test_rally_order_separates_hits_from_bounces(conv):
    events = detect_events(_ball(conv, RALLY), _rally_boxes(conv), conv, FPS)
    assert BOUNCE_NEAR in set(events["event"])
    labels = list(events["event"])
    assert labels == [BOUNCE_FAR, HIT_FAR, BOUNCE_NEAR, HIT_NEAR, BOUNCE_FAR, HIT_FAR]


def test_survives_missing_frames_and_a_wrong_detection(conv):
    ball = _ball(conv, RALLY, seed=1)
    rng = np.random.default_rng(2)
    ball = ball[rng.random(len(ball)) > 0.15].reset_index(drop=True)
    ball.loc[len(ball) // 3, ["cx", "cy"]] += [220.0, -160.0]
    _assert_found(detect_bounces(ball, _rally_boxes(conv), conv, FPS), conv)


def test_a_volley_has_no_bounce(conv):
    volley = [(10.0, 4.0, 23.0, 1.0), (32.0, 5.0, 8.0, 1.2), (60.0, 2.5, 19.0, 0.0),
              (72.0, 2.0, 23.5, 1.0), (90.0, 3.0, 14.0, 1.5)]
    boxes = _boxes(conv, volley, [volley[0], volley[3]], [volley[1]])
    bounces = detect_bounces(_ball(conv, volley), boxes, conv, FPS)
    assert list(bounces["side"]) == ["near"]
    assert abs(bounces["frame"].iloc[0] - 60.0) <= 1.0


def test_a_free_flight_has_no_event(conv):
    flight = [(0.0, 4.0, 23.0, 1.0), (30.0, 5.0, 3.0, 0.8)]
    boxes = _boxes(conv, flight, [flight[0]], [flight[1]])
    assert detect_bounces(_ball(conv, flight), boxes, conv, FPS).empty


def test_a_serve_anchors_the_order_without_a_junction(conv):
    points = RALLY[1:]              # the track starts after the serve
    ball = _ball(conv, points)
    boxes = _boxes(conv, RALLY, [RALLY[0], RALLY[4]], [RALLY[2], RALLY[6]])
    bounces = detect_bounces(ball, boxes, conv, FPS, serves=[(10, 1, 0)])
    assert list(bounces["side"]) == ["near", "far"]


def test_flight_measures_time_depth_and_apex(conv):
    lob = [(10.0, 4.0, 23.0, 1.0), (55.0, 5.0, 2.0, 0.0), (70.0, 5.2, -1.0, 1.0)]
    span = 1.5
    v0 = (0.0 - 1.0 + 0.5 * G * span ** 2) / span
    events = pd.DataFrame({"frame": [55.0], "x_m": [5.0], "y_m": [2.0],
                           "event": [BOUNCE_FAR]})
    flight = shot_flight(10, 1, (4.0, 23.0), 70, events, _ball(conv, lob), conv, FPS)
    assert flight["time_s"] == pytest.approx(span)
    assert flight["depth_m"] == pytest.approx(11.8872 - 2.0)
    assert flight["apex_m"] == pytest.approx(1.0 + v0 ** 2 / (2 * G), abs=0.3)


def test_a_shot_without_a_bounce_has_no_flight(conv):
    events = pd.DataFrame({"frame": [90.0], "x_m": [5.0], "y_m": [2.0],
                           "event": [BOUNCE_FAR]})
    ball = _ball(conv, RALLY)
    assert shot_flight(10, 1, (4.0, 23.0), 60, events, ball, conv, FPS) is None
    assert shot_flight(10, 2, (4.0, 1.0), 200, events, ball, conv, FPS) is None


def test_load_real_detections_drops_the_filled_gaps(tmp_path):
    path = tmp_path / "ball.csv"
    pd.DataFrame({"frame": [1, 2, 3], "cx": [10.0, 20.0, 30.0], "cy": [5.0, 6.0, 7.0],
                  "interpolated": [0, 1, 0]}).to_csv(path, index=False)
    assert list(load_real_detections(str(path))["frame"]) == [1, 3]
