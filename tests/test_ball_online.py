import cv2
import numpy as np
import pytest

from tracking.ball_model import CROP, DiffBlobBackend, OnnxBackend
from tracking.ball_online import Kalman, OnlineBallTracker, pack_channels
from utils.background import OnlineBackground

W, H = 1280, 720
PROC = (640, 360)


def test_pack_channels_stores_bgr_so_the_network_sees_gray_diff_previous():
    gray = np.full((4, 4), 10, np.uint8)
    d_t = np.full((4, 4), 20, np.uint8)
    d_tm1 = np.full((4, 4), 30, np.uint8)
    img = pack_channels(gray, d_t, d_tm1, gain=1.0)
    assert tuple(img[0, 0]) == (30, 20, 10)                 # B, G, R as written to PNG
    x = OnnxBackend.preprocess([img])                       # what the network reads
    assert x.shape == (1, 3, 4, 4)
    assert np.allclose(x[0, :, 0, 0] * 255, (10, 20, 30))   # gray, diff t, diff t-1


def test_the_difference_gain_saturates():
    d = np.array([[100, 200]], np.uint8)
    img = pack_channels(np.zeros_like(d), d, d)
    assert tuple(img[0, :, 1]) == (200, 255)


def test_kalman_follows_a_parabola():
    kf = Kalman((100.0, 500.0), (8.0, -20.0))
    errs = []
    for t in range(1, 40):
        truth = (100 + 8 * t, 500 - 20 * t + 0.5 * t * t)
        pred = kf.predict()
        errs.append(np.hypot(*(pred - truth)))
        kf.update(truth)
    assert max(errs[10:]) < 2.0


def _clip(n=105):
    """A static court and a ball flying across it (frames, ball centres)."""
    rng = np.random.default_rng(0)
    court = np.full((H, W, 3), (150, 90, 60), np.uint8)
    texture = cv2.GaussianBlur(rng.normal(0, 25, (H, W)), (0, 0), 3)[..., None]  # static,
    court = np.clip(court + texture, 0, 255).astype(np.uint8)      # like a real court
    cv2.rectangle(court, (200, 150), (1080, 650), (235, 235, 235), 3)
    frames, centres = [], []
    for t in range(n):
        f = court.copy()
        x, y = 150 + 9 * t, 200 + 4 * t + 0.03 * (t - 60) ** 2
        cv2.circle(f, (int(x), int(y)), 6, (60, 230, 230), -1)
        f = np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8)
        frames.append(f)
        centres.append((x, y))
    return frames, centres


@pytest.mark.parametrize("search", ["tiles", "blobs"])
def test_tracker_finds_and_follows_the_ball(search):
    frames, centres = _clip()
    bg = OnlineBackground(30, (W, H), PROC, roi=(100, 100, 1180, 700), threaded=False,
                          warmup_s=1.0, refresh_s=1.0)
    tracker = OnlineBallTracker(DiffBlobBackend(), bg.scale, H, search=search)
    found = 0
    for i, f in enumerate(frames):
        snap = bg.snapshot
        live, rows = tracker.step(i, f, snap)
        bg.feed(i, f)
        if i >= 45 and live is not None and live["detected"]:
            found += np.hypot(live["cx"] - centres[i][0], live["cy"] - centres[i][1]) < 5
        assert all(r[7] in ("crop", "search") for r in rows)
    assert tracker.state == OnlineBallTracker.TRACKING
    assert found >= 0.9 * (len(frames) - 45)
    if search == "tiles":
        assert tracker.n_crops < 3 * len(frames)            # one crop per frame once locked


def test_crops_stay_inside_the_roi_at_the_edges():
    frames, _ = _clip(40)
    bg = OnlineBackground(30, (W, H), PROC, roi=(100, 100, 1180, 700), threaded=False,
                          warmup_s=0.5)
    for i, f in enumerate(frames):
        bg.feed(i, f)
    tracker = OnlineBallTracker(DiffBlobBackend(), bg.scale, H)
    snap = bg.snapshot
    for x, y in ((0, 0), (5000, 5000), (90, 650)):
        x0, y0 = tracker.crops.clamp(snap, x, y)
        assert 100 <= x0 <= 1180 - CROP and 100 <= y0 <= 700 - CROP
        lo, hi = snap.full_crop(x0, y0, CROP)
        assert lo.shape == hi.shape == (CROP, CROP, 3)


class _Nothing:
    name = "none"

    def detect(self, crops):
        return [[] for _ in crops]


class _YellowThing:
    """A colour detector that always sees something in the middle of its crop."""

    name = "colour"

    def detect(self, crops):
        return [[(150.0, 150.0, 166.0, 166.0, 0.9)] for _ in crops]


def test_the_colour_detector_is_not_used_to_search_by_the_players():
    frames, _ = _clip(40)
    bg = OnlineBackground(30, (W, H), PROC, roi=(100, 100, 1180, 700), threaded=False,
                          warmup_s=0.5)
    tracker = OnlineBallTracker(_Nothing(), bg.scale, H, search="tiles", fallback=_YellowThing())
    player = [(600.0, 300.0, 60.0, 150.0)]
    sources = set()
    for i, f in enumerate(frames):
        live, rows = tracker.step(i, f, bg.snapshot, players=player)
        bg.feed(i, f)
        sources |= {r[7] for r in rows}
    assert "colour" not in sources and tracker.n_fallback == 0   # yellow things by a player
    assert tracker.state == OnlineBallTracker.SEARCH


def test_the_ball_is_on_a_player_only_inside_the_grown_box():
    from tracking.ball_online import inside_player
    player = [(100.0, 100.0, 40.0, 100.0)]
    assert inside_player(120, 150, player)
    assert inside_player(104 - 0.1 * 40, 95, player)          # within the 15 % margin
    assert not inside_player(60, 150, player)                 # an arm's length away
    assert not inside_player(120, 260, player)


def _lob_clip(n=120):
    """A ball rising out of the top of the picture and falling back where it left."""
    rng = np.random.default_rng(1)
    court = np.full((H, W, 3), (150, 90, 60), np.uint8)
    texture = cv2.GaussianBlur(rng.normal(0, 25, (H, W)), (0, 0), 3)[..., None]
    court = np.clip(court + texture, 0, 255).astype(np.uint8)
    frames, centres = [], []
    for t in range(n):
        f = court.copy()
        x = 640 + 2 * t
        y = 700 - 26 * t + 0.22 * t * t        # out at the top from t ~ 41 to t ~ 77
        if y > 6:
            cv2.circle(f, (int(x), int(y)), 6, (60, 230, 230), -1)
        frames.append(np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8))
        centres.append((x, y))
    return frames, centres


def test_a_ball_lost_at_the_top_is_found_again_where_it_left():
    frames, centres = _lob_clip()
    bg = OnlineBackground(30, (W, H), PROC, roi=(0, 0, W, H), threaded=False, warmup_s=0.5)
    tracker = OnlineBallTracker(DiffBlobBackend(), bg.scale, H, search="tiles",
                                search_roi=(0, 200, W, H))     # the court: below y = 200
    states = []
    for i, f in enumerate(frames):
        live, rows = tracker.step(i, f, bg.snapshot)
        bg.feed(i, f)
        states.append((i, tracker.state, live))
        if any(r[7] == "high" for r in rows):
            break
    assert tracker.n_high_found == 1                           # found in the crops where it left
    i, state, live = states[-1]
    assert state == OnlineBallTracker.TRACKING and centres[i][1] < 200   # still above the court
    assert np.hypot(live["cx"] - centres[i][0], live["cy"] - centres[i][1]) < 8


def test_nothing_far_to_the_side_is_taken_for_a_ball_out_at_the_top():
    frames, centres = _lob_clip()
    for t in range(45, len(frames)):          # a spectator waving in the stands, far left
        cv2.circle(frames[t], (150 + 4 * (t % 2), 400 + t), 6, (60, 230, 230), -1)
    bg = OnlineBackground(30, (W, H), PROC, roi=(0, 0, W, H), threaded=False, warmup_s=0.5)
    tracker = OnlineBallTracker(DiffBlobBackend(), bg.scale, H, search="tiles",
                                search_roi=(0, 200, W, H))
    taken = []
    for i, f in enumerate(frames):
        live, rows = tracker.step(i, f, bg.snapshot)
        bg.feed(i, f)
        taken += [(r[1] + r[3]) / 2 for r in rows]
        if tracker.n_high_found:
            break
    assert tracker.n_off_corridor > 0                          # the spectator was seen...
    assert min(taken) > 400                                    # ...and never kept
    assert tracker.n_high_found == 1                           # the ball, where it left


def test_gpu_batches_are_padded_to_a_few_sizes():
    from tracking.ball_model import batch_bucket
    assert [batch_bucket(n) for n in (1, 2, 3, 5, 15, 16, 17, 40)] == [1, 2, 4, 8, 16, 16, 32, 32]


class _Faint:
    """Sees a ball at (100, 100) of the crop, but only with a low threshold."""

    name = "faint"

    def __init__(self):
        self.conf = 0.15

    def detect(self, crops):
        return [[(94.0, 94.0, 106.0, 106.0, 0.08)] if self.conf < 0.1 else [] for _ in crops]


def test_a_missed_frame_is_read_again_where_the_ball_must_be():
    tracker = OnlineBallTracker(_Faint(), 0.5, 1080)
    crop = np.zeros((CROP, CROP, 3), np.uint8)
    tracker.missed[10] = ((500, 300), (2.0, 0.0), crop, None)   # origin, camera offset
    assert tracker.refill(10, (602.0, 400.0)) == (596.0, 394.0, 608.0, 406.0)
    assert tracker.refill(10, (700.0, 400.0)) is None            # too far from the gap
    assert tracker.refill(11, (602.0, 400.0)) is None            # nothing kept for it
    assert tracker.backend.conf == 0.15                          # threshold restored


class _BallAndCap(DiffBlobBackend):
    """The moving ball (from the difference) plus a still white cap the model loves."""

    name = "ball+cap"

    def detect(self, crops):
        out = super().detect(crops)
        for dets, img in zip(out, crops):
            ys, xs = np.nonzero(img[..., 2] > 245)             # gray channel: the cap
            if len(xs):
                dets.append((xs.min() - 1.0, ys.min() - 1.0, xs.max() + 1.0, ys.max() + 1.0, 0.95))
        return out


def test_a_still_ball_lookalike_neither_starts_nor_holds_a_track():
    frames, centres = _clip()
    for f in frames:                                            # a ball kid's cap, never moving
        cv2.circle(f, (400, 460), 7, (255, 255, 255), -1)
    bg = OnlineBackground(30, (W, H), PROC, roi=(100, 100, 1180, 700), threaded=False,
                          warmup_s=1.0, refresh_s=1.0)
    tracker = OnlineBallTracker(_BallAndCap(), bg.scale, H, search="tiles")
    found = on_cap = 0
    for i, f in enumerate(frames):
        live, rows = tracker.step(i, f, bg.snapshot)
        bg.feed(i, f)
        if i >= 45 and live is not None and live["detected"]:
            found += np.hypot(live["cx"] - centres[i][0], live["cy"] - centres[i][1]) < 5
            on_cap += np.hypot(live["cx"] - 400, live["cy"] - 460) < 10
    assert on_cap == 0
    assert found >= 0.9 * (len(frames) - 45)
