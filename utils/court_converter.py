import csv

import cv2
import numpy as np

# ITF singles court — real-world positions in metres for each labelled corner.
# Origin = TL (top-left corner of the far baseline).
#   x : 0 (left sideline)  →  8.2296 m (right sideline)
#   y : 0 (far baseline)   →  23.7744 m (near baseline)
# Defined once in utils/court_geometry; re-imported here (and re-exported, since
# evaluation/evaluate_tracking.py imports _REAL_WORLD from this module).
from utils.court_geometry import _REAL_WORLD


class CourtConverter:
    """
    Converts pixel coordinates inside a tennis court video to real-world
    metres using a perspective homography computed from the 8 labelled
    court corners produced by court_tracking.py.

    Usage
    -----
        converter = CourtConverter("outputs/court_coordinates/match1_court.csv")
        x_m, y_m = converter.to_meters(850, 600)

        # batch — e.g. all ball positions from BallTracking
        positions_px = np.array([[850, 600], [920, 650], ...])   # shape (N, 2)
        positions_m  = converter.to_meters_batch(positions_px)   # shape (N, 2)
    """

    def __init__(self, court_csv_path: str):
        pixel_pts, real_pts = self._load(court_csv_path)
        self._H = self._compute_homography(pixel_pts, real_pts)

    # ── public ────────────────────────────────────────────────────────────────

    # Smallest homogeneous divisor magnitude we trust. Points whose projective
    # w-coordinate is ~0 lie on (or beyond) the horizon line of the court plane:
    # the perspective division would explode to ±inf there, so we clip the
    # divisor magnitude to this epsilon to keep the result finite/bounded.
    _W_EPS = 1e-9

    def to_meters(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert a single pixel position to court metres."""
        p = self._H @ np.array([x_px, y_px, 1.0], dtype=np.float64)
        # Guard the homogeneous divisor against ~0 (point near the horizon line)
        # to avoid ±inf; preserve the original sign so the projection direction
        # is kept.
        w = p[2]
        if abs(w) < self._W_EPS:
            w = self._W_EPS if w >= 0 else -self._W_EPS
        return float(p[0] / w), float(p[1] / w)

    def to_meters_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Convert an (N, 2) array of pixel positions to court metres.
        Returns an (N, 2) float64 array. Rows whose homogeneous divisor is ~0
        (points on/near the horizon line of the court plane) are returned as
        NaN instead of ±inf.
        """
        pts = np.asarray(points, dtype=np.float64)
        hom = np.column_stack([pts, np.ones(len(pts))])  # (N, 3)
        res = (self._H @ hom.T).T                         # (N, 3)
        w = res[:, 2:3]
        # Mark near-horizon rows (|w| ~ 0) and divide safely; those rows become
        # NaN rather than ±inf.
        bad = np.abs(w) < self._W_EPS                   # (N, 1)
        safe_w = np.where(bad, np.nan, w)
        out = res[:, :2] / safe_w
        out[bad[:, 0]] = np.nan
        return out

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: str):
        """Read the court CSV and return aligned pixel and real-world arrays."""
        pixel_pts, real_pts = [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                label = row["label"].strip()
                if label in _REAL_WORLD:
                    pixel_pts.append([float(row["x"]), float(row["y"])])
                    real_pts.append(_REAL_WORLD[label])
        if len(pixel_pts) < 4:
            raise ValueError(
                f"Need at least 4 known labels in {path}; found {len(pixel_pts)}."
            )
        return np.array(pixel_pts, dtype=np.float64), \
               np.array(real_pts,  dtype=np.float64)

    @staticmethod
    def _compute_homography(pixel_pts, real_pts) -> np.ndarray:
        # RANSAC (instead of a plain least-squares fit, method=0) makes the
        # solve robust to a single mislabelled / noisy keypoint: with the 8
        # court corners available, an outlier corner is rejected rather than
        # biasing the whole homography.
        H, _ = cv2.findHomography(
            pixel_pts, real_pts,
            method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if H is None:
            raise RuntimeError("Homography computation failed — check the court CSV.")
        return H.astype(np.float64)
