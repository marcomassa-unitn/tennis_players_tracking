import pytest

from utils.video import check_full_read, safe_fps


def test_safe_fps_rejects_invalid_values():
    assert safe_fps(25.0) == 25.0
    for bad in (0, -1, None, float("nan"), float("inf")):
        assert safe_fps(bad) == 30.0


def test_full_read_passes():
    check_full_read(1008, 1008, "clip.mp4")
    check_full_read(1007, 1008, "clip.mp4")  # within tolerance


def test_unknown_frame_count_is_not_checked():
    check_full_read(10, 0, "clip.mp4")


def test_truncated_read_raises():
    with pytest.raises(RuntimeError, match="frame 388 of 1008"):
        check_full_read(388, 1008, "clip.mp4")
