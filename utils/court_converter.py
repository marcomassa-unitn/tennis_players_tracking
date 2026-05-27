import csv

import cv2
import numpy as np

# ITF singles court — real-world positions in metres for each labelled corner.
# Origin = TL (top-left corner of the far baseline).
#   x : 0 (left sideline)  →  8.2296 m (right sideline)
#   y : 0 (far baseline)   →  23.7744 m (near baseline)
_FT = 0.3048
_REAL_WORLD = {
    "TL":  (0.0,           0.0),
    "TR":  (27.0 * _FT,    0.0),
    "BL":  (0.0,           78.0 * _FT),
    "BR":  (27.0 * _FT,    78.0 * _FT),
    "STL": (0.0,           18.0 * _FT),
    "STR": (27.0 * _FT,    18.0 * _FT),
    "SBL": (0.0,           78.0 * _FT - 18.0 * _FT),
    "SBR": (27.0 * _FT,    78.0 * _FT - 18.0 * _FT),
}


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

    def to_meters(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert a single pixel position to court metres."""
        p = self._H @ np.array([x_px, y_px, 1.0], dtype=np.float64)
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_meters_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Convert an (N, 2) array of pixel positions to court metres.
        Returns an (N, 2) float64 array.
        """
        pts = np.asarray(points, dtype=np.float64)
        hom = np.column_stack([pts, np.ones(len(pts))])  # (N, 3)
        res = (self._H @ hom.T).T                         # (N, 3)
        return res[:, :2] / res[:, 2:3]

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
        return np.array(pixel_pts, dtype=np.float32), \
               np.array(real_pts,  dtype=np.float32)

    @staticmethod
    def _compute_homography(pixel_pts, real_pts) -> np.ndarray:
        H, _ = cv2.findHomography(pixel_pts, real_pts, method=0)
        if H is None:
            raise RuntimeError("Homography computation failed — check the court CSV.")
        return H.astype(np.float64)
