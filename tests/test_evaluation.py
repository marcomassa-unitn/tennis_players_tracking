import pytest

from evaluation.evaluate_tracking import iou


def test_iou_identical_boxes():
    assert iou((10, 10, 50, 80), (10, 10, 50, 80)) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_iou_partial_overlap():
    # Overlap 5x10 = 50, union 100 + 100 - 50 = 150.
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_iou_degenerate_box():
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
