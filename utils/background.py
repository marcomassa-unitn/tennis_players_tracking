"""Background of a fixed broadcast camera, built from the past frames only.

The offline trackers take the median of frames spread over the whole clip. A live
pipeline cannot look ahead, so this module keeps the same median, but over the
frames seen so far:

1. Warm-up: the first WARMUP_S seconds, one frame every WARMUP_STEP, give the first
   background. Dark frames (fade-in) are skipped, every sample is aligned to the
   middle one, as in PlayerTracker._build_backgrounds.
2. Refresh: one sample per SAMPLE_EVERY_S goes into a ring of RING_SAMPLES, and the
   median is recomputed every REFRESH_S seconds in a worker thread. Before a sample
   enters the ring, the pixels around the players are replaced by the current
   background, so a player standing still (a server) never becomes part of it.
3. Each background is published as an immutable BgSnapshot, swapped in between two
   frames: a frame is always compared with one consistent background.

Two resolutions come from the same samples: the processing size of the player
tracker (whole frame, 540 px high) and the full resolution inside a region of
interest (the court), where the ball is a few pixels wide.

Helpers shared by the trackers live here too: translation, median stack, phase
correlation, and the jitter-tolerant difference used for both players and ball.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

# ── Online background ─────────────────────────────────────────────────────────
WARMUP_S = 2.0             # first background from this many seconds...
WARMUP_STEP = 3            # ...one frame every this many
RING_SAMPLES = 21          # samples in the running median (odd)
SAMPLE_EVERY_S = 1.0       # one sample per second after the warm-up
REFRESH_S = 5.0            # median recomputed this often
DARK_SAMPLE_RATIO = 0.6    # samples darker than this x the typical one are skipped
PLAYER_MARGIN = 0.25       # player boxes grown by this fraction before masking

# ── Camera motion (processing px) ─────────────────────────────────────────────
MAX_CAMERA_SHIFT_PX = 40
MIN_SHIFT_RESPONSE = 0.1


def channel_max(img):
    """Per-pixel maximum over the channels of a BGR image (cv2, ~6x numpy's max)."""
    if img.ndim == 2:
        return img
    b, g, r = cv2.split(img)
    return cv2.max(cv2.max(b, g), r)


def luma_median(img_bgr, step=1):
    """Median grey level, on every `step`-th pixel (1 = all of them)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if step > 1:
        gray = gray[::step, ::step]
    return float(np.median(gray))


def gray_f32(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def translate(img, dx, dy, interpolation=cv2.INTER_LINEAR):
    """Shift `img` by (-dx, -dy): undoes a camera translation of (dx, dy)."""
    if dx == 0 and dy == 0:
        return img
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=interpolation,
                          borderMode=cv2.BORDER_REPLICATE)


def median_stack(frames):
    """Per-pixel median of uint8 frames, computed in row strips to bound memory."""
    stack = np.stack(frames)
    out = np.empty(stack.shape[1:], np.uint8)
    for r in range(0, stack.shape[1], 64):
        out[r:r + 64] = np.median(stack[:, r:r + 64], axis=0).astype(np.uint8)
    return out


def camera_shift(ref_gray, img, hann, max_shift=MAX_CAMERA_SHIFT_PX,
                 min_response=MIN_SHIFT_RESPONSE):
    """(dx, dy) translation of `img` relative to `ref_gray` (float32), or (0, 0)."""
    (dx, dy), response = cv2.phaseCorrelate(ref_gray, gray_f32(img), hann)
    if response < min_response or np.hypot(dx, dy) > max_shift:
        return 0.0, 0.0
    return float(dx), float(dy)


def jitter_bounds(bg, radius):
    """(local min, local max) of a background over a (2 radius + 1)^2 window."""
    if radius <= 0:
        return bg, bg
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.erode(bg, k), cv2.dilate(bg, k)


def jitter_diff(img, lo, hi):
    """Distance of each pixel to the background's local [min, max] range, 0 inside.

    Single channel uint8: the maximum over the colour channels. A camera shaking by
    a pixel or two then never lights up court lines or banner text.
    """
    return channel_max(cv2.max(cv2.subtract(img, hi), cv2.subtract(lo, img)))


def subpix_crop(img, x0, y0, w, h):
    """img[y0:y0+h, x0:x0+w] for a sub-pixel origin; borders replicated."""
    if float(x0).is_integer() and float(y0).is_integer():
        x0, y0 = int(x0), int(y0)
        if 0 <= x0 and 0 <= y0 and x0 + w <= img.shape[1] and y0 + h <= img.shape[0]:
            return img[y0:y0 + h, x0:x0 + w]
    centre = (x0 + (w - 1) / 2.0, y0 + (h - 1) / 2.0)
    return cv2.getRectSubPix(img, (w, h), centre)


# ── Snapshot ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BgSnapshot:
    """One background, with everything the trackers derive from it.

    small_*: whole frame at processing size. full_*: the region of interest at full
    resolution, whose top-left corner in the frame is roi[:2].
    """
    version: int
    frame: int                     # last frame that went into it
    small: np.ndarray
    small_gray: np.ndarray         # float32, for phase correlation
    small_lo: np.ndarray
    small_hi: np.ndarray
    small_hsv: np.ndarray
    luma: float
    roi: tuple                     # (x0, y0, x1, y1) in full-res px
    full: np.ndarray | None
    full_lo: np.ndarray | None
    full_hi: np.ndarray | None
    extra: dict = field(default_factory=dict)   # e.g. the court-line mask

    def full_crop(self, x0, y0, size):
        """(lo, hi) of the full-res background for a size x size window at (x0, y0)."""
        rx, ry = self.roi[0], self.roi[1]
        lo = self.full_lo[y0 - ry:y0 - ry + size, x0 - rx:x0 - rx + size]
        hi = self.full_hi[y0 - ry:y0 - ry + size, x0 - rx:x0 - rx + size]
        return lo, hi


class OnlineBackground:
    """Running median background over past frames only. See the module docstring.

    proc_size: (w, h) of the processing scale. roi: (x0, y0, x1, y1) full-res region
    kept at full resolution (None = none). small_jitter / full_jitter: radius of the
    local min / max at each scale. extra_fn(small_bg) -> dict adds derived images
    to each snapshot (it runs in the worker). threaded=False refreshes in the
    caller's thread, so a run is deterministic (tests, dataset building).
    """

    def __init__(self, fps, frame_size, proc_size, roi=None, small_jitter=2,
                 full_jitter=1, warmup_s=WARMUP_S, warmup_step=WARMUP_STEP,
                 ring=RING_SAMPLES, sample_every_s=SAMPLE_EVERY_S, refresh_s=REFRESH_S,
                 extra_fn=None, threaded=True, align=True, defer_roi=False):
        self.fps = fps
        self.W, self.H = frame_size
        self.proc_size = proc_size
        self.scale = proc_size[1] / self.H
        if roi is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in roi)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(self.W, x1), min(self.H, y1)
            roi = (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
        self.roi = roi
        self.small_jitter, self.full_jitter = small_jitter, full_jitter
        self.warmup_frames = max(warmup_step, int(round(warmup_s * fps)))
        self.warmup_step = warmup_step
        self.ring_size = ring if ring % 2 else ring + 1
        self.sample_every = max(1, int(round(sample_every_s * fps)))
        self.refresh_every = max(1, int(round(refresh_s * fps)))
        self.extra_fn = extra_fn
        self.defer_roi = defer_roi      # no ROI yet: keep whole frames until set_roi()
        self.threaded = threaded
        self.align = align
        self.hann = cv2.createHanningWindow(proc_size, cv2.CV_32F)

        self.snapshot: BgSnapshot | None = None
        self._first = None              # first frame offered
        self._last_sample = None        # frame of the last sample (frames may be dropped)
        self._warm_small, self._warm_full = [], []
        self._ring_small, self._ring_full = [], []
        self._last_refresh = None
        self._worker: threading.Thread | None = None
        self._ready: BgSnapshot | None = None
        self._lock = threading.Lock()
        self._version = 0

    # ── feeding ──────────────────────────────────────────────────────────────

    @property
    def ready(self):
        return self.snapshot is not None

    def _roi_crop(self, frame):
        if self.roi is None:
            return frame.copy() if self.defer_roi else None
        x0, y0, x1, y1 = self.roi
        return frame[y0:y1, x0:x1].copy()

    def set_roi(self, roi):
        """Set the full-resolution region once it is known (the court found later).

        Samples already taken at full resolution are cropped to it; the samples
        without one keep the background at processing scale only until the ring
        has turned over.
        """
        x0, y0, x1, y1 = (int(round(v)) for v in roi)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1)
        if x1 <= x0 or y1 <= y0:
            return
        old = self.roi
        self.roi = (x0, y0, x1, y1)

        def recrop(img):
            if img is None:
                return None
            ox, oy = (old[0], old[1]) if old is not None else (0, 0)
            if old is None and img.shape[:2] != (self.H, self.W):
                return None
            return img[y0 - oy:y1 - oy, x0 - ox:x1 - ox].copy()
        self._warm_full = [recrop(i) for i in self._warm_full]
        self._ring_full = [recrop(i) for i in self._ring_full]
        if self.snapshot is not None and self._last_refresh is not None:
            self._last_refresh = -10 ** 9      # rebuild with the ROI at the next frame

    def seed(self, background, idx=0):
        """Start from a known background (full-res BGR): no warm-up.

        With a fixed camera the empty court can be learnt before the match (see
        realtime/calibration.py). The ring is filled with copies of it, so the
        running median moves away from it only once most of the ring holds new
        samples. Returns the first snapshot.
        """
        if background.shape[:2] != (self.H, self.W):
            raise ValueError(f"background is {background.shape[1]}x{background.shape[0]}, "
                             f"the video {self.W}x{self.H}")
        small = cv2.resize(background, self.proc_size, interpolation=cv2.INTER_AREA)
        full = self._roi_crop(background)
        self.snapshot = self._build([small], [full], idx, align=False)
        self._ring_small = [small] * self.ring_size
        self._ring_full = [full] * self.ring_size
        self._warm_small, self._warm_full = [], []
        self._first = self._last_sample = self._last_refresh = idx
        return self.snapshot

    def feed(self, idx, frame, small=None, players=(), shift=(0.0, 0.0)):
        """Offer frame `idx` (full res BGR) to the background; return the snapshot.

        small: the frame at processing size (computed if None). players: full-res
        (x, y, w, h) boxes to keep out of the samples. shift: camera translation of
        this frame relative to the current background (processing px), undone
        before the frame is sampled. Returns the snapshot to use for this frame
        (None during the warm-up).
        """
        with self._lock:
            if self._ready is not None:
                self.snapshot, self._ready = self._ready, None
        if small is None:
            small = cv2.resize(frame, self.proc_size, interpolation=cv2.INTER_AREA)

        if self._first is None:
            self._first = idx
        if self.snapshot is None:
            if self._last_sample is None or idx - self._last_sample >= self.warmup_step:
                self._last_sample = idx
                self._warm_small.append(small.copy())
                self._warm_full.append(self._roi_crop(frame))
            if idx + 1 - self._first >= self.warmup_frames and len(self._warm_small) >= 3:
                snap = self._build(self._warm_small, self._warm_full, idx, align=self.align)
                if snap is not None:
                    self.snapshot = snap
                    self._ring_small = list(self._warm_small[-self.ring_size:])
                    self._ring_full = list(self._warm_full[-self.ring_size:])
                    self._last_refresh = idx
                    self._warm_small, self._warm_full = [], []
            return self.snapshot

        if idx - self._last_sample >= self.sample_every:
            self._last_sample = idx
            self._push_sample(frame, small, players, shift)
        if idx - self._last_refresh >= self.refresh_every and not self._busy():
            self._last_refresh = idx
            self._start_refresh(idx)
        return self.snapshot

    def _push_sample(self, frame, small, players, shift):
        snap = self.snapshot
        dx, dy = shift
        s_al = translate(small, dx, dy) if (dx or dy) else small.copy()
        f_al = frame.copy() if (self.roi is None and self.defer_roi) else None
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            f_al = subpix_crop(frame, x0 + dx / self.scale, y0 + dy / self.scale,
                               x1 - x0, y1 - y0).copy()
        if len(players):
            small_mask = np.zeros(s_al.shape[:2], np.uint8)
            full_mask = (None if f_al is None or self.roi is None
                         else np.zeros(f_al.shape[:2], np.uint8))
            for x, y, w, h in players:
                mx, my = PLAYER_MARGIN * w, PLAYER_MARGIN * h
                a = (x - mx - dx / self.scale, y - my - dy / self.scale)
                b = (x + w + mx - dx / self.scale, y + h + my - dy / self.scale)
                cv2.rectangle(small_mask, (int(a[0] * self.scale), int(a[1] * self.scale)),
                              (int(b[0] * self.scale) + 1, int(b[1] * self.scale) + 1), 255, -1)
                if full_mask is not None:
                    rx, ry = self.roi[0], self.roi[1]
                    cv2.rectangle(full_mask, (int(a[0] - rx), int(a[1] - ry)),
                                  (int(b[0] - rx) + 1, int(b[1] - ry) + 1), 255, -1)
            np.copyto(s_al, snap.small, where=small_mask[..., None] > 0)
            if full_mask is not None and snap.full is not None and snap.full.shape == f_al.shape:
                np.copyto(f_al, snap.full, where=full_mask[..., None] > 0)
        self._ring_small.append(s_al)
        self._ring_full.append(f_al)
        if len(self._ring_small) > self.ring_size:
            self._ring_small.pop(0)
            self._ring_full.pop(0)

    # ── refreshing ───────────────────────────────────────────────────────────

    def _busy(self):
        return self._worker is not None and self._worker.is_alive()

    def _start_refresh(self, idx):
        small, full = list(self._ring_small), list(self._ring_full)
        if not self.threaded:
            snap = self._build(small, full, idx, align=False)
            if snap is not None:
                self.snapshot = snap
            return

        def work():
            snap = self._build(small, full, idx, align=False)
            if snap is not None:
                with self._lock:
                    self._ready = snap
        self._worker = threading.Thread(target=work, name="bg-refresh", daemon=True)
        self._worker.start()

    def wait(self):
        """Block until a pending refresh is done and swap it in (deterministic tests)."""
        if self._worker is not None:
            self._worker.join()
        with self._lock:
            if self._ready is not None:
                self.snapshot, self._ready = self._ready, None

    def _build(self, small_frames, full_frames, idx, align):
        """Median snapshot of the samples (dark ones skipped, aligned if asked)."""
        if not small_frames:
            return None
        lumas = np.array([luma_median(f, 4) for f in small_frames])
        keep = [k for k, lum in enumerate(lumas) if lum >= DARK_SAMPLE_RATIO * np.median(lumas)]
        if len(keep) % 2 == 0 and len(keep) > 1:
            keep = keep[1:]
        if not keep:
            return None
        small = [small_frames[k] for k in keep]
        full = [full_frames[k] for k in keep] if self.roi is not None else []
        if any(f is None for f in full):          # some samples predate the ROI
            full = [f for f in full if f is not None]
            if len(full) % 2 == 0 and len(full) > 1:
                full = full[1:]
            align = False
        if align and len(small) > 1:
            ref = gray_f32(small[len(small) // 2])
            shifts = [camera_shift(ref, fr, self.hann) for fr in small]
            small = [translate(fr, dx, dy) for fr, (dx, dy) in zip(small, shifts)]
            if full:
                full = [translate(fr, dx / self.scale, dy / self.scale)
                        for fr, (dx, dy) in zip(full, shifts)]
        bg = median_stack(small)
        lo, hi = jitter_bounds(bg, self.small_jitter)
        full_bg = full_lo = full_hi = None
        roi = self.roi if full else None
        if full:
            full_bg = median_stack(full)
            full_lo, full_hi = jitter_bounds(full_bg, self.full_jitter)
        self._version += 1
        extra = self.extra_fn(bg) if self.extra_fn is not None else {}
        return BgSnapshot(version=self._version, frame=idx, small=bg, small_gray=gray_f32(bg),
                          small_lo=lo, small_hi=hi,
                          small_hsv=cv2.cvtColor(bg, cv2.COLOR_BGR2HSV),
                          luma=luma_median(bg), roi=roi, full=full_bg,
                          full_lo=full_lo, full_hi=full_hi, extra=extra)
