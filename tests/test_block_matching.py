import cv2
import numpy as np
import pytest

from motion_estimation.block_matching import (
    full_search,
    motion_compensate,
    psnr,
    three_step_search,
)

BLOCK = 16
SHIFT_X, SHIFT_Y = 3, 2


@pytest.fixture
def frame_pair():
    rng = np.random.default_rng(0)
    noise = rng.uniform(0, 255, (128, 128)).astype(np.float32)
    # A smooth texture gives a well-behaved SAD surface for three-step search.
    prev = cv2.GaussianBlur(noise, (0, 0), 3)
    prev = cv2.normalize(prev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cur = np.roll(prev, (SHIFT_Y, SHIFT_X), axis=(0, 1))  # content moves right/down
    return prev, cur


@pytest.mark.parametrize("search_fn", [full_search, three_step_search])
def test_recovers_global_translation(frame_pair, search_fn):
    prev, cur = frame_pair
    u, v = search_fn(prev, cur, block=BLOCK, search=8)
    # Skip the border blocks touched by np.roll's wrap-around.
    assert np.all(u[1:-1, 1:-1] == SHIFT_X)
    assert np.all(v[1:-1, 1:-1] == SHIFT_Y)


def test_motion_compensation_beats_no_motion_baseline(frame_pair):
    prev, cur = frame_pair
    u, v = full_search(prev, cur, block=BLOCK, search=8)
    pred = motion_compensate(prev, u, v, BLOCK)
    inner = np.s_[BLOCK:-BLOCK, BLOCK:-BLOCK]
    np.testing.assert_array_equal(pred[inner], cur[inner])
    assert psnr(pred, cur) > psnr(prev, cur)


def test_psnr_identical_images_is_infinite():
    img = np.full((8, 8), 128, dtype=np.uint8)
    assert psnr(img, img) == float("inf")
