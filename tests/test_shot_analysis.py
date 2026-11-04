"""Run the synthetic self-tests shipped with the shot classifier.

Each one raises SystemExit on failure, so a clean return means it passed.
"""

import numpy as np
import pandas as pd
import pytest

from analysis import shot_analysis
from analysis.shot_analysis import (
    BOUNCE_FAR,
    BOUNCE_NEAR,
    HIT_FAR,
    _synthetic_court_csv,
    classify_flight,
    detect_serves,
    is_volley,
    move_contacts_off_bounces,
)
from utils.court_converter import CourtConverter
from utils.court_geometry import NET, L_m, W_m


def test_stroke_self_test():
    shot_analysis.self_test()


def test_overhead_self_test():
    shot_analysis.overhead_self_test()


def _events(*rows):
    return pd.DataFrame(rows, columns=["frame", "cx", "cy", "jump_up_ms", "jump_side_ms", "event"])


def test_contact_on_a_bounce_moves_to_the_strongest_hit_after_it():
    events = _events((100.2, 500, 300, 10.0, 0.5, BOUNCE_FAR),
                     (104.0, 510, 290, 2.0, 1.0, HIT_FAR),
                     (112.4, 530, 280, 3.0, 9.0, HIT_FAR))
    hits, ball_at = move_contacts_off_bounces([(50, 1, 1.0), (101, 2, 1.0)], events, 30.0)
    assert hits == [(50, 1, 1.0), (112, 2, 1.0)]
    assert ball_at == {112: (530.0, 280.0)}


def test_contact_stays_without_a_following_hit_or_on_the_other_side():
    events = _events((100.0, 500, 300, 10.0, 0.5, BOUNCE_FAR))
    hits = [(100, 2, 1.0), (101, 1, 1.0)]
    assert move_contacts_off_bounces(hits, events, 30.0) == (hits, {})


def test_flight_gives_the_shot_type():
    def flight(time_s, depth_m, apex_m):
        return {"time_s": time_s, "depth_m": depth_m, "apex_m": apex_m}

    assert classify_flight(flight(0.8, 9.0, 1.4)) == "flat"
    assert classify_flight(flight(0.8, 3.0, 3.0)) == "flat"
    assert classify_flight(flight(1.3, 3.0, 2.3)) == "dropshot"
    assert classify_flight(flight(1.4, 10.0, 3.5)) == "lob"
    assert classify_flight(flight(1.3, 9.0, 1.3)) == "slice"


def _bounce_at(frame, event):
    return pd.DataFrame({"frame": [float(frame)], "event": [event]})


def test_a_shot_inside_the_service_line_before_the_bounce_is_a_volley():
    no_bounce = _bounce_at(90, BOUNCE_FAR)            # on the other side
    assert is_volley(120, 1, NET + 3.0, (90, 2), no_bounce, 30.0)


def test_not_a_volley_after_a_bounce_or_from_the_back():
    bounced = _bounce_at(110, BOUNCE_NEAR)
    assert not is_volley(120, 1, NET + 3.0, (90, 2), bounced, 30.0)
    empty = _bounce_at(10, BOUNCE_NEAR)
    assert not is_volley(120, 1, L_m, (90, 2), empty, 30.0)          # hidden bounce
    assert not is_volley(120, 1, NET + 3.0, (90, 1), empty, 30.0)    # own shot before
    assert not is_volley(120, 1, NET + 3.0, None, empty, 30.0)


@pytest.fixture
def conv(tmp_path):
    path = tmp_path / "court.csv"
    _synthetic_court_csv(str(path))
    return CourtConverter(str(path))


def _server_boxes(conv, pid, first, last, contact):
    """One player at their baseline, box raised up to a peak at `contact`."""
    m_to_px = np.linalg.inv(conv._H)
    u, v, w = m_to_px @ np.array([W_m / 2, L_m + 0.3 if pid == 1 else -0.3, 1.0])
    feet = (u / w, v / w)
    heights = {f: 200.0 for f in range(first, last + 1)}
    heights.update({f: 280.0 for f in range(contact - 20, contact)})
    if contact <= last:
        heights[contact] = 340.0
    return {f: {pid: (feet[0] - 45, feet[1] - h, 90.0, h)} for f, h in heights.items()}


def test_a_peak_cut_by_the_end_of_the_clip_is_not_a_serve(conv):
    assert [s[0] for s in detect_serves(_server_boxes(conv, 1, 0, 150, 67), conv, 30.0, [])] == [67]
    assert detect_serves(_server_boxes(conv, 1, 0, 150, 148), conv, 30.0, []) == []


def test_a_swing_just_after_a_serve_is_the_return(conv):
    boxes = _server_boxes(conv, 1, 0, 150, 40)
    for f, players in _server_boxes(conv, 2, 0, 150, 90).items():
        boxes[f].update(players)
    assert [s[:2] for s in detect_serves(boxes, conv, 30.0, [])] == [(40, 1)]
