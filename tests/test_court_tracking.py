import cv2
import numpy as np
import pytest

from tracking.court_tracking import CourtTracker, detect_court, keypoints_from_homography
from utils.court_geometry import _REAL_WORLD
from utils.court_overlay import MODEL_SEGMENTS

W, H = 960, 540


def _broadcast_homography():
    """Court metres -> pixels for a typical elevated view behind the near baseline."""
    model = np.float32([[-1.37, 0], [9.6, 0], [9.6, 23.77], [-1.37, 23.77]])
    image = np.float32([[340, 120], [620, 120], [800, 460], [160, 460]])
    return cv2.getPerspectiveTransform(model, image)


def _render(Hm, with_distractors=True):
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), (150, 90, 60), np.uint8)          # blue court
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    for a, b in MODEL_SEGMENTS:
        p = cv2.perspectiveTransform(np.float64([[a], [b]]), Hm)[:, 0]
        cv2.line(img, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)),
                 (235, 235, 235), 3, cv2.LINE_AA)
    if with_distractors:
        # A white banner edge above the court and writing below it: the cases that
        # fooled the previous detector.
        cv2.line(img, (250, 70), (710, 70), (240, 240, 240), 3)
        cv2.putText(img, "INDIAN   WELLS", (300, 515), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (240, 240, 240), 3)
    return img


def test_detects_court_keypoints_despite_distractors():
    Hm = _broadcast_homography()
    H_fit, ratio = detect_court(_render(Hm))
    assert H_fit is not None
    assert ratio > 0.9
    expected = keypoints_from_homography(Hm)
    found = keypoints_from_homography(H_fit)
    for label in _REAL_WORLD:
        assert found[label] == pytest.approx(expected[label], abs=1.5), label


def test_black_frame_has_no_court():
    H_fit, ratio = detect_court(np.zeros((H, W, 3), np.uint8))
    assert H_fit is None
    assert ratio == 0.0


def test_no_folder_is_created_for_a_video_that_cannot_be_opened(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot open video"):
        CourtTracker("data/nope.mp4", no_display=True, output_dir=str(tmp_path)).run()
    assert list(tmp_path.iterdir()) == []
