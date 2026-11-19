import pytest

from analysis.shot_analysis import _synthetic_court_csv
from tracking.ball_online import OnlineBallTracker
from tracking.ball_reach import BallReach
from tracking.ball_tracking import OnlinePathDP
from utils.court_converter import CourtConverter

# The synthetic court: far baseline at y = 300 px (x 700-1200), near one at y = 860.


@pytest.fixture
def reach(tmp_path):
    path = tmp_path / "court.csv"
    _synthetic_court_csv(str(path))
    return BallReach(CourtConverter(str(path)), 30.0)


def test_a_ball_carries_on_its_motion(reach):
    assert reach.allowed((950, 400), (962, 380), 4, velocity=(3, -5))


def test_the_top_of_a_lob_cannot_be_on_the_near_player_four_frames_later(reach):
    assert not reach.allowed((950, 290), (950, 700), 4, velocity=(0, -3))


def test_the_longer_the_ball_is_unseen_the_farther_it_may_be(reach):
    assert reach.allowed((950, 290), (950, 700), 20, velocity=(0, -3))


def test_a_hit_sends_the_ball_wherever_it_can_reach(reach):
    near = (900, 650, 80, 200)                       # the near player, feet at y = 850
    on_racket, going_up = (930, 700), (950, 330)     # a lob, seen high 4 frames later
    assert reach.allowed(on_racket, going_up, 4, velocity=(0, 30), players=[near])
    assert not reach.allowed(on_racket, going_up, 4, velocity=(0, 30))   # nobody hit it


def test_with_no_motion_known_only_the_impossible_is_refused(reach):
    assert reach.allowed((950, 290), (950, 700), 4)                  # maybe high over the court
    assert not reach.allowed((700, 300), (1500, 860), 1)             # corner to corner in a frame


NEAR = 1


def _near_player(f):
    """The near player walking towards the camera, 5 px a frame."""
    return (1050.0, 760.0 + 5 * f, 100.0, 220.0)


def _lob_and_a_head():
    """A lob rising and coming back down, lost for a while; meanwhile the near
    player's head, detected with confidence, rides with him for 7 frames."""
    rows = []
    for f in list(range(0, 21)) + list(range(34, 61)):
        x, y = 950.0, 700 - 20 * f + 0.25 * f * f
        rows.append((f, x - 7, y - 7, x + 7, y + 7, 0.7, 0.5, "crop"))
    for f in range(24, 31):
        x, y, w, _ = _near_player(f)
        hx, hy = x + 0.5 * w, y + 3
        rows.append((f, hx - 7, hy - 7, hx + 7, hy + 7, 0.6, 0.5, "search"))
    return rows


def _final(dp, rows, last=70, players=False):
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r)
    for f in range(last):
        dp.push(f, by.get(f, []), {NEAR: _near_player(f)} if players else {})
    dp.flush()
    return dp.final


def test_the_path_does_not_take_a_player_s_head_for_the_lost_ball(reach):
    rows = _lob_and_a_head()
    plain = _final(OnlinePathDP(30.0), rows)
    assert any(f in plain for f in range(24, 31))             # linked: 95 px/frame or less
    checked = _final(OnlinePathDP(30.0, reach=reach), rows, players=True)
    assert not any(f in checked for f in range(24, 31))
    assert all(f in checked for f in range(34, 61))            # the ball, found again


def test_a_candidate_that_rides_with_a_player_is_part_of_him():
    dp = OnlinePathDP(30.0)
    head = [(f, 1093.0, 763.0 + 5 * f, 1107.0, 777.0 + 5 * f, 0.6, 0.5, "search")
            for f in range(6)]
    for f in range(6):
        dp.push(f, [head[f]], {NEAR: _near_player(f)})
    assert dp.n_riding == 3                                    # frames 3-5: after 3 in place
    crossing = OnlinePathDP(30.0)                              # a ball flying across him
    for f in range(6):
        x, y = 1055.0 + 18 * f, 800.0 + 5 * f
        crossing.push(f, [(f, x - 7, y - 7, x + 7, y + 7, 0.6, 0.5, "crop")], {NEAR: _near_player(f)})
    assert crossing.n_riding == 0


class _Nothing:
    name = "none"

    def detect(self, crops):
        return [[] for _ in crops]


def test_the_search_starts_again_only_where_the_lost_ball_can_be(reach):
    tracker = OnlineBallTracker(_Nothing(), 0.5, 1080)
    tracker.set_reach(reach)
    tracker.last_seen, tracker.last_vel = (10, 950.0, 290.0), (0.0, -3.0)
    assert not tracker._reachable(14, (943, 693, 957, 707, 0.9))     # the near player's head
    assert tracker._reachable(14, (943, 270, 957, 284, 0.9))         # near the top of the lob
    assert tracker.n_refused == 1


def _at(f, x, y, conf=0.6):
    return (f, x - 7, y - 7, x + 7, y + 7, conf, 0.3, "search")


def test_something_standing_still_is_not_the_ball():
    """A camera lens by the boards: never in the track, and dropped at once next time."""
    dp = OnlinePathDP(30.0)
    for f in range(30):
        dp.push(f, [_at(f, 1680.0 + 0.1 * f, 418.0)])
    for f in range(30, 60):
        dp.push(f, [])
    live = dp.still.n_dropped
    for f in range(60, 66):                                    # it comes back
        dp.push(f, [_at(f, 1683.0, 418.0)])
    assert dp.still.n_dropped - live == 4                      # from its 3rd frame
    dp.flush()
    assert not dp.final


def test_a_slow_ball_and_a_ball_held_by_a_player_are_not_still():
    slow = OnlinePathDP(30.0)
    for f in range(30):
        slow.push(f, [_at(f, 900.0, 300.0 + 2 * f)])          # the top of a lob, far away
    assert slow.still.n_dropped == 0
    held = OnlinePathDP(30.0)
    server = {NEAR: (1000.0, 700.0, 100.0, 220.0)}
    for f in range(30):
        held.push(f, [_at(f, 1060.0, 640.0)], server)          # over the head before a toss
    assert held.still.n_dropped == 0


def test_the_first_frames_of_a_still_thing_do_not_stay_in_the_track():
    dp = OnlinePathDP(30.0)
    for f in range(20):
        dp.push(f, [_at(f, 950.0, 400.0 + 15 * f)])           # the ball, then lost
    for f in range(20, 60):
        dp.push(f, [_at(f, 1680.0, 418.0)])                   # a lens, taken for it
    dp.flush()
    assert all(f in dp.final for f in range(16))
    assert not any(f in dp.final for f in range(20, 60))
