import csv

import cv2
import numpy as np

# ITF singles court corner positions in metres. Origin = far-baseline TL corner;
# x spans 0..8.2296 (left→right sideline), y spans 0..23.7744 (far→near baseline).
# Re-exported: evaluation/evaluate_tracking.py imports _REAL_WORLD from here.
from utils.court_geometry import _REAL_WORLD


class CourtConverter:
    """
    Map court-video pixels to real-world metres via a perspective homography
    fit to the 8 labelled court corners from court_tracking.py.

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

    # Floor on the homogeneous divisor |w|. w~0 means the point sits on the
    # court plane's horizon line, where perspective division blows up to ±inf;
    # clipping to this epsilon keeps the result bounded.
    _W_EPS = 1e-9

    def to_meters(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Project one pixel position to court metres."""
        p = self._H @ np.array([x_px, y_px, 1.0], dtype=np.float64)
        # Clamp |w| away from 0 (near-horizon point); keep its sign to preserve
        # projection direction.
        w = p[2]
        if abs(w) < self._W_EPS:
            w = self._W_EPS if w >= 0 else -self._W_EPS
        return float(p[0] / w), float(p[1] / w)

    def to_meters_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Project an (N, 2) pixel array to (N, 2) float64 court metres.

        Near-horizon rows (|w| ~ 0) come back as NaN rather than ±inf.
        """
        pts = np.asarray(points, dtype=np.float64)
        hom = np.column_stack([pts, np.ones(len(pts))])  # (N, 3)
        res = (self._H @ hom.T).T                         # (N, 3)
        w = res[:, 2:3]
        # Flag near-horizon rows so their division yields NaN, not ±inf.
        bad = np.abs(w) < self._W_EPS                   # (N, 1)
        safe_w = np.where(bad, np.nan, w)
        out = res[:, :2] / safe_w
        out[bad[:, 0]] = np.nan
        return out

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: str):
        """Return aligned (pixel, real-world) arrays for the CSV's known labels."""
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
        # RANSAC (not plain least-squares) lets the 8-corner fit reject one
        # mislabelled/noisy corner instead of letting it bias the homography.
        H, _ = cv2.findHomography(
            pixel_pts, real_pts,
            method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if H is None:
            raise RuntimeError("Homography computation failed — check the court CSV.")
        return H.astype(np.float64)
