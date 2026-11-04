"""Run the synthetic self-tests shipped with the shot classifier.

Each one raises SystemExit on failure, so a clean return means it passed.
"""

import pandas as pd

from analysis import shot_analysis
from analysis.shot_analysis import (
    BOUNCE_FAR,
    HIT_FAR,
    classify_flight,
    move_contacts_off_bounces,
)


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
