import pytest

from evaluation.evaluate_tracking import bounce_metrics, iou


def test_iou_identical_boxes():
    assert iou((10, 10, 50, 80), (10, 10, 50, 80)) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_iou_partial_overlap():
    # Overlap 5x10 = 50, union 100 + 100 - 50 = 150.
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_iou_degenerate_box():
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_bounce_metrics_pairs_closest_first_and_counts_hidden():
    gt = {100: (500.0, 400.0), 110: (600.0, 400.0), 200: None, 300: (10.0, 10.0)}
    pred = [(100.5, 503.0, 404.0), (108.4, 600.0, 400.0), (201.0, 0.0, 0.0), (250.0, 0.0, 0.0)]
    summary, rows = bounce_metrics(gt, pred, tolerance=2)
    assert (summary["tp"], summary["fp"], summary["fn"]) == (3, 1, 1)
    assert summary["precision"] == pytest.approx(0.75)
    assert summary["recall"] == pytest.approx(0.75)
    assert summary["recall_visible"] == pytest.approx(2 / 3)
    assert summary["median_err_px"] == pytest.approx(2.5)
    assert [r[-1] for r in rows].count("FP") == 1
