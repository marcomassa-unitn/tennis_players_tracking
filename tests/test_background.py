import cv2
import numpy as np

from utils.background import (
    OnlineBackground,
    channel_max,
    jitter_bounds,
    jitter_diff,
    subpix_crop,
)

W, H = 320, 180
PROC = (160, 90)


def _court():
    img = np.full((H, W, 3), (150, 90, 60), np.uint8)
    cv2.line(img, (20, 150), (300, 150), (235, 235, 235), 2)
    return img


def test_channel_max_matches_numpy():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    assert np.array_equal(channel_max(img), img.max(axis=2))


def test_jitter_diff_ignores_one_pixel_shake_but_not_a_ball():
    bg = _court()
    lo, hi = jitter_bounds(bg, 1)
    shaken = np.roll(bg, 1, axis=0)                       # camera moved by one pixel
    assert jitter_diff(shaken, lo, hi).max() == 0
    ball = shaken.copy()
    cv2.circle(ball, (100, 60), 4, (60, 230, 230), -1)
    assert jitter_diff(ball, lo, hi)[60, 100] > 100


def test_subpix_crop_is_a_plain_slice_on_integer_origins():
    img = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    assert np.array_equal(subpix_crop(img, 4, 5, 10, 8), img[5:13, 4:14])
    assert subpix_crop(img, 4.5, 5.0, 10, 8).shape == (8, 10, 3)


def _feed(bg, frames, players=None):
    snap = None
    for i, f in enumerate(frames):
        small = cv2.resize(f, PROC, interpolation=cv2.INTER_AREA)
        snap = bg.feed(i, f, small, players=(players(i) if players else ()))
    return snap


def test_background_is_the_empty_court_after_the_warm_up():
    court = _court()
    frames = []
    for i in range(90):
        f = court.copy()
        x = 20 + 3 * i
        cv2.rectangle(f, (x, 60), (x + 15, 110), (40, 40, 200), -1)    # a moving player
        frames.append(f)
    bg = OnlineBackground(30, (W, H), PROC, roi=(0, 0, W, H), threaded=False,
                          warmup_s=1.0, ring=9, sample_every_s=0.2, refresh_s=1.0)
    assert _feed(bg, frames[:20]) is None                   # still warming up
    snap = _feed(bg, frames)
    assert snap is not None and snap.full.shape == court.shape
    assert np.abs(snap.full.astype(int) - court).max() <= 2
    assert snap.version >= 2                                # refreshed since the warm-up


def test_a_player_standing_still_stays_out_of_the_background():
    court = _court()
    box = (100, 60, 20, 50)                                 # x, y, w, h

    def frame(i):
        f = court.copy()
        x = box[0] if i >= 30 else 20 + 2 * i               # walks in, then stands still
        cv2.rectangle(f, (x, box[1]), (x + box[2], box[1] + box[3]), (40, 40, 200), -1)
        return f

    frames = [frame(i) for i in range(240)]
    def players(i):
        return [box] if i >= 30 else [(20 + 2 * i, *box[1:])]

    bg = OnlineBackground(30, (W, H), PROC, roi=(0, 0, W, H), threaded=False,
                          warmup_s=1.0, ring=9, sample_every_s=0.2, refresh_s=1.0)
    snap = _feed(bg, frames, players)
    centre = snap.full[box[1] + box[3] // 2, box[0] + box[2] // 2].astype(int)
    assert np.abs(centre - court[0, 0]).max() <= 2          # still the court colour

    plain = OnlineBackground(30, (W, H), PROC, roi=(0, 0, W, H), threaded=False,
                             warmup_s=1.0, ring=9, sample_every_s=0.2, refresh_s=1.0)
    snap = _feed(plain, frames)                              # without masking: a ghost
    centre = snap.full[box[1] + box[3] // 2, box[0] + box[2] // 2].astype(int)
    assert np.abs(centre - (40, 40, 200)).max() <= 2


def test_the_roi_can_come_after_the_warm_up():
    frames = [_court() for _ in range(60)]
    bg = OnlineBackground(30, (W, H), PROC, roi=None, defer_roi=True, threaded=False,
                          warmup_s=0.5, refresh_s=1.0)
    snap = _feed(bg, frames[:30])
    assert snap is not None and snap.full is None
    bg.set_roi((40, 30, 240, 150))
    snap = bg.feed(30, frames[30])
    assert snap.full is not None and snap.full.shape == (120, 200, 3)
    assert snap.roi == (40, 30, 240, 150)


def test_a_seeded_background_is_ready_at_the_first_frame():
    court = _court()
    bg = OnlineBackground(30, (W, H), PROC, roi=(40, 30, 240, 150), threaded=False,
                          ring=5, sample_every_s=0.1, refresh_s=0.2)
    snap = bg.seed(court)
    assert snap is not None and np.array_equal(snap.full, court[30:150, 40:240])
    frame = court.copy()
    cv2.rectangle(frame, (100, 60), (115, 110), (40, 40, 200), -1)      # someone passing by
    for i in range(30):
        snap = bg.feed(i, frame if i < 2 else court)
    assert np.abs(snap.full.astype(int) - court[30:150, 40:240]).max() <= 2
