import numpy as np
import pandas as pd
import pytest

from analysis.impact_comparison import _trail
from live_view import PANEL_W, BallMap, ball_ground_path, compose, load_bounce_markers

FPS = 30.0


def _shots(*rows):
    return pd.DataFrame(rows, columns=["frame", "player_x_m", "player_y_m"])


def _bounces(*rows):
    return pd.DataFrame(rows, columns=["frame", "x_m", "y_m", "reliable"])


def test_the_ball_flies_straight_from_a_contact_to_its_bounce():
    path = ball_ground_path(_shots((10, 4.0, 24.0)), _bounces((40, 6.0, 4.0, True)), FPS)
    assert path[10] == pytest.approx((4.0, 24.0))
    assert path[25] == pytest.approx((5.0, 14.0))
    assert path[40] == pytest.approx((6.0, 4.0))
    assert 9 not in path and 41 not in path


def test_the_path_goes_through_the_impact_rather_than_the_feet():
    shots = pd.DataFrame({"frame": [10, 70], "player_x_m": [4.0, 9.0], "player_y_m": [24.0, 0.0],
                          "impact_x_m": [4.0, 7.0], "impact_y_m": [24.0, 0.0]})
    path = ball_ground_path(shots, _bounces((40, 5.5, 12.0, True)), FPS)
    assert path[70] == pytest.approx((7.0, 0.0))


def test_the_bounce_frame_can_fall_between_two_frames():
    path = ball_ground_path(_shots((10, 0.0, 0.0)), _bounces((20.5, 10.5, 0.0, True)), FPS)
    assert path[20] == pytest.approx((10.0, 0.0))
    assert 21 not in path


def test_no_path_between_two_points():
    path = ball_ground_path(_shots((10, 4.0, 24.0), (300, 4.0, 24.0)),
                            _bounces((40, 6.0, 4.0, True)), FPS)
    assert 41 not in path and 299 not in path


def test_a_shot_with_no_position_is_ignored():
    path = ball_ground_path(_shots((10, np.nan, np.nan), (20, 1.0, 1.0)),
                            _bounces((30, 1.0, 11.0, True)), FPS)
    assert 15 not in path and 25 in path


def test_bounces_keep_their_rally_order_when_one_is_off_limits():
    ball_map = BallMap((260, 540))
    bounces = _bounces((10, 99.0, 4.0, True), (20, 4.0, 6.0, False))
    markers = load_bounce_markers(bounces, ball_map, (-5, 15, -8, 32))
    assert [(m[0], m[1], m[3]) for m in markers] == [(20, 2, False)]


def test_the_ball_column_sits_between_the_video_and_the_panels():
    ball_map = BallMap((520, 1080))
    panel = ball_map.render((4.0, 6.0), [(4.0, 10.0), (4.0, 6.0)], [(1, (100, 200), True)],
                            [(1, "forehand", (120, 700))])
    assert panel.shape == (1080, 520, 3)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    side = np.zeros((540, 760, 3), np.uint8)
    canvas = compose(frame, side, side, 540, PANEL_W, 270, ball_panel=panel, ball_w=260)
    assert canvas.shape == (540, 960 + 260 + PANEL_W, 3)
    assert canvas[:, 960:960 + 260].any()
    without = compose(frame, side, side, 540, PANEL_W, 270)
    assert without.shape == (540, 960 + PANEL_W, 3)


def test_the_trail_stops_where_the_path_is_interrupted():
    path = {f: (float(f), 0.0) for f in (1, 2, 5, 6, 7)}
    assert _trail(path, 7, 10) == [(5.0, 0.0), (6.0, 0.0), (7.0, 0.0)]
    assert _trail(path, 7, 2) == [(6.0, 0.0), (7.0, 0.0)]
    assert _trail(path, 4, 10) == []
