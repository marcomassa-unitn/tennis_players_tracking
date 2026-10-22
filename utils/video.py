"""Video helpers shared by the trackers."""

import numpy as np


def safe_fps(fps, default=30.0):
    """Fall back to ``default`` when ``fps`` is falsy, non-finite, or non-positive."""
    if not fps or not np.isfinite(fps) or fps <= 0:
        return default
    return fps
