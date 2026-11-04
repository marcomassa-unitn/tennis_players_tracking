"""Run the synthetic self-tests shipped with the shot classifier.

Each one raises SystemExit on failure, so a clean return means it passed.
"""

import pandas as pd

from analysis import shot_analysis
from analysis.shot_analysis import BOUNCE_FAR, HIT_FAR, move_contacts_off_bounces


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
