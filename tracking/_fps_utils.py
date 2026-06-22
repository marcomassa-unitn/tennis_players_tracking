"""
tracking/_fps_utils.py

Shared fps sanity-guard for the trackers. The same guard was written verbatim in
playerTracking.py, BallTracking.py and court_tracking.py; it lives here so both the
orchestrated runs (pipeline.py, which puts the project root on sys.path) and the
standalone runs (`python tracking/X.py`, where tracking/ is on sys.path) can import
it.
"""

import numpy as np


def safe_fps(fps, default=30.0):
    """Return ``fps`` unless it is missing / non-finite / non-positive, in which
    case return ``default``. Mirrors the guard previously inlined in each tracker:
    ``if not fps or not np.isfinite(fps) or fps <= 0: fps = 30``.
    """
    if not fps or not np.isfinite(fps) or fps <= 0:
        return default
    return fps
