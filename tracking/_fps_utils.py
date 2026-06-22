"""Shared fps sanity-guard for the trackers.

Hoisted here (out of playerTracking/BallTracking/court_tracking) so both
orchestrated runs (pipeline.py adds project root to sys.path) and standalone
runs (`python tracking/X.py`, with tracking/ on sys.path) can import it.
"""

import numpy as np


def safe_fps(fps, default=30.0):
    """Fall back to ``default`` when ``fps`` is falsy, non-finite, or non-positive."""
    if not fps or not np.isfinite(fps) or fps <= 0:
        return default
    return fps
