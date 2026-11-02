"""Video helpers shared by the trackers."""

import numpy as np


def safe_fps(fps, default=30.0):
    """Fall back to ``default`` when ``fps`` is falsy, non-finite, or non-positive."""
    if not fps or not np.isfinite(fps) or fps <= 0:
        return default
    return fps


def check_full_read(frames_read, frame_count, video_path, tolerance=2):
    """Raise if decoding stopped well before the container's frame count.

    ``cv2.VideoCapture.read()`` returns False both at the end of the file and
    when a frame cannot be decoded (e.g. when the machine is out of memory), so
    without this check a truncated run would pass for a complete one.
    ``frame_count`` comes from CAP_PROP_FRAME_COUNT, which can be slightly off
    for some codecs, hence the small tolerance; 0 (unknown) disables the check.
    """
    if frame_count > 0 and frames_read < frame_count - tolerance:
        raise RuntimeError(
            f"Stopped reading {video_path} at frame {frames_read} of {frame_count}: "
            "the remaining frames could not be decoded (often caused by low "
            "memory). The output of this run would be incomplete, so it was "
            "not saved.")
