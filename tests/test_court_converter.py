import csv

import numpy as np
import pytest

from utils.court_converter import CourtConverter
from utils.court_geometry import _REAL_WORLD

# A plausible broadcast-style metres -> pixels homography (perspective view).
H_TRUE = np.array([[60.0, -8.0, 700.0],
                   [0.0, 18.0, 250.0],
                   [0.0, -0.012, 1.0]])


def _project(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    res = np.c_[pts, np.ones(len(pts))] @ H.T
    return res[:, :2] / res[:, 2:]


@pytest.fixture
def court_csv(tmp_path):
    path = tmp_path / "court.csv"
    labels = list(_REAL_WORLD)
    pixels = _project(H_TRUE, [_REAL_WORLD[k] for k in labels])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "x", "y"])
        for label, (x, y) in zip(labels, pixels):
            writer.writerow([label, x, y])
    return path


def test_corners_map_back_to_metres(court_csv):
    conv = CourtConverter(str(court_csv))
    for label, real in _REAL_WORLD.items():
        px = _project(H_TRUE, [real])[0]
        assert conv.to_meters(*px) == pytest.approx(real, abs=1e-4), label


def test_batch_matches_single_point(court_csv):
    conv = CourtConverter(str(court_csv))
    real = np.array([[1.0, 2.0], [4.1, 11.9], [7.5, 20.0]])
    px = _project(H_TRUE, real)
    np.testing.assert_allclose(conv.to_meters_batch(px), real, atol=1e-4)


def test_too_few_labels_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("label,x,y\nTL,0,0\nTR,1,0\nBL,0,1\n")
    with pytest.raises(ValueError):
        CourtConverter(str(path))
