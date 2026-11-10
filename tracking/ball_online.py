"""Causal ball tracking for the real-time pipeline: a small detector on crops.

The model reads three channels, built the same way for training
(training/build_ball_dataset.py) and here:

    gray t,   difference from the background at t,   the same at t - 1

The difference shows the ball as a small bright dot whatever the colour of the
court; the grey image tells the ball from a racket or a shoe; the previous
difference gives the direction of motion. Crops are CROP x CROP at full
resolution, so the ~15 px ball keeps its size.

OnlineBallTracker is a two-state machine:
  TRACKING  a constant-acceleration Kalman filter predicts the ball, and the model
            runs on one crop centred on the prediction. A detection is accepted
            inside the Mahalanobis gate, or anywhere in the crop if it is very
            confident (a racket hit changes the velocity at once). After
            MAX_COAST frames without the ball it goes to SEARCH.
  SEARCH    the model runs on crops over the court: every tile of the court region
            (GPU), or only the ball-sized blobs of the difference (CPU). Two
            consistent detections within ACQUIRE_FRAMES start a new track.

Every detection is also returned as a candidate row (ball_tracking.CANDIDATE_COLUMNS),
and the fixed-lag path search of ball_tracking.OnlinePathDP picks the ball among
them, as the offline tracker does over the whole clip.
"""

from __future__ import annotations

import cv2
import numpy as np

from tracking.ball_model import CROP, NMS_IOU, _nms
from tracking.ball_tracking import (
    BLOB_MAX_AREA,
    BLOB_MAX_ELONGATION,
    BLOB_MIN_AREA,
    MAX_SPEED_PX,
    court_pixels,
    motion_fraction,
    tile_origins,
    within_reach,
)
from utils.background import jitter_diff, subpix_crop

DIFF_GAIN = 2.0            # differences are small: x2 uses more of the 0-255 range
DIFF_THRESH = 30           # (before the gain) foreground for the blob proposals
SEARCH_OVERLAP = 32        # px between court tiles in SEARCH
MAX_BLOB_PROPOSALS = 4     # CPU search: crops around the most contrasted blobs
MAX_COAST = 3              # frames predicted without a detection before SEARCH
GATE_CHI2 = 13.8           # 2 dof, 99.9 %
MIN_GATE_PX = 25           # always accept a detection this close to the prediction
STRONG_CONF = 0.6          # accepted anywhere in the crop: a hit changed the velocity
ACQUIRE_CONF = 0.3         # SEARCH: detections weaker than this start nothing
ACQUIRE_FRAMES = 3         # ...two of them this close in time start a track
REACH_Q_GAIN = 10.0        # process noise x this near a player (contacts)
ROI_MARGIN = 64            # px around the court region kept at full resolution


def court_roi(conv, width, height, margin=ROI_MARGIN, min_size=CROP):
    """(x0, y0, x1, y1): where the ball can be, at least one crop wide and high.

    The court and its run-off (ball_tracking.court_pixels) plus `margin`, inside
    the frame. Training crops and live crops both come from this region.
    """
    x0, x1, y0, y1 = court_pixels(conv)
    roi = [int(max(0, x0 - margin)), int(max(0, y0 - margin)),
           int(min(width, x1 + margin)), int(min(height, y1 + margin))]
    for a, b, lim in ((0, 2, width), (1, 3, height)):
        if roi[b] - roi[a] < min_size:
            roi[a] = max(0, min(roi[a], lim - min_size))
            roi[b] = min(lim, roi[a] + min_size)
    return tuple(roi)


def pack_channels(gray, diff_t, diff_tm1, gain=DIFF_GAIN):
    """Model input as an OpenCV (BGR-order) image: (diff t-1, diff t, gray).

    Ultralytics and OnnxBackend flip it to RGB, so the network sees
    (gray, diff t, diff t-1).
    """
    if gain != 1.0:
        diff_t = cv2.convertScaleAbs(diff_t, alpha=gain)
        diff_tm1 = cv2.convertScaleAbs(diff_tm1, alpha=gain)
    return cv2.merge([diff_tm1, diff_t, gray])


class CropMaker:
    """Builds model crops of a frame against a background snapshot.

    Crops are placed in background coordinates (the snapshot's ROI), so the
    background is sliced exactly; the frame is sampled at the same place moved by
    the camera shift. `scale` converts the shift from processing to full-res px.
    """

    def __init__(self, scale, size=CROP):
        self.scale = scale
        self.size = size

    def clamp(self, snap, x0, y0):
        """Top-left corner of a crop moved inside the snapshot's ROI."""
        rx0, ry0, rx1, ry1 = snap.roi
        x0 = int(round(min(max(x0, rx0), rx1 - self.size)))
        y0 = int(round(min(max(y0, ry0), ry1 - self.size)))
        return max(x0, rx0), max(y0, ry0)

    def frame_crop(self, frame, x0, y0, shift):
        dx, dy = shift[0] / self.scale, shift[1] / self.scale
        return subpix_crop(frame, x0 + dx, y0 + dy, self.size, self.size)

    def diff(self, snap, frame, x0, y0, shift):
        """(difference, BGR crop) at background position (x0, y0)."""
        img = self.frame_crop(frame, x0, y0, shift)
        lo, hi = snap.full_crop(x0, y0, self.size)
        return jitter_diff(img, lo, hi), img

    def channels(self, snap, frame, prev, x0, y0, shift, prev_shift):
        """Model crop at (x0, y0): the current frame and the previous one."""
        d_t, img = self.diff(snap, frame, x0, y0, shift)
        d_tm1 = d_t if prev is None else self.diff(snap, prev, x0, y0, prev_shift)[0]
        return pack_channels(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), d_t, d_tm1)


class Kalman:
    """Constant-acceleration Kalman filter on the image position (px, frames)."""

    F = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5], [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], float)
    Hm = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], float)

    def __init__(self, xy, v=(0.0, 0.0), scale=1.0):
        s = scale
        self.x = np.array([xy[0], xy[1], v[0], v[1], 0.0, 0.0], float)
        self.P = np.diag([4.0, 4.0, 400.0, 400.0, 25.0, 25.0]) * s * s
        self.Q = np.diag([1.0, 1.0, 4.0, 4.0, 1.0, 1.0]) * s * s
        self.R = np.diag([4.0, 4.0]) * s * s

    def predict(self, q_gain=1.0):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q * q_gain
        return self.x[:2].copy()

    def mahalanobis2(self, z):
        S = self.Hm @ self.P @ self.Hm.T + self.R
        y = np.asarray(z, float) - self.x[:2]
        return float(y @ np.linalg.solve(S, y))

    def update(self, z):
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, float) - self.x[:2])
        self.P = (np.eye(6) - K @ self.Hm) @ self.P


class OnlineBallTracker:
    """See the module docstring. One call of step() per frame.

    backend: a tracking.ball_model backend. scale: processing -> full-res factor of
    the camera shift (OnlineBackground.scale). search: "tiles" (every court tile,
    for a GPU) or "blobs" (crops on the ball-sized blobs only, for a CPU).
    search_every: run SEARCH on one frame in this many.
    """

    TRACKING, SEARCH = "tracking", "search"

    def __init__(self, backend, scale, frame_height=1080, search="tiles", search_every=1,
                 crop=CROP):
        self.backend = backend
        self.crops = CropMaker(scale, crop)
        self.size = crop
        self.s = frame_height / 1080
        self.search = search
        self.search_every = max(1, int(search_every))
        self.state = self.SEARCH
        self.kf = None
        self.misses = 0
        self.tentative = []            # SEARCH: [(frame, cx, cy, conf)]
        self.prev = self.prev_gray = None
        self.prev_shift = (0.0, 0.0)
        self.prev_idx = None
        self.last_crops = []           # (x0, y0) of the crops of the last frame (overlay)
        self.n_crops = 0               # crops run through the model so far

    # ── proposals ────────────────────────────────────────────────────────────

    def _tiles(self, snap):
        rx0, ry0, rx1, ry1 = snap.roi
        return [(rx0 + x, ry0 + y) for x, y in
                tile_origins(rx1 - rx0, ry1 - ry0, self.size, SEARCH_OVERLAP)]

    def _blob_proposals(self, snap, frame, shift, players, small_diff=None):
        """Crops centred on the most contrasted ball-sized blobs of the ROI.

        small_diff: the difference at processing scale (background coordinates)
        that the player tracker has just computed; without it the difference is
        computed here over the ROI at full resolution (slower).
        """
        rx0, ry0, rx1, ry1 = snap.roi
        k = self.crops.scale                     # processing px per full-res px
        dx, dy = shift[0] / k, shift[1] / k
        if small_diff is not None:
            x0, y0 = int(rx0 * k), int(ry0 * k)
            diff = small_diff[y0:int(np.ceil(ry1 * k)), x0:int(np.ceil(rx1 * k))]
            to_full, ox, oy = 1.0 / k, x0 / k, y0 / k
            s2 = (self.s * k) ** 2
        else:
            img = subpix_crop(frame, rx0 + dx, ry0 + dy, rx1 - rx0, ry1 - ry0)
            diff = jitter_diff(img, snap.full_lo, snap.full_hi)
            to_full, ox, oy = 1.0, rx0, ry0
            s2 = self.s * self.s
        mask = (diff >= DIFF_THRESH).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        found = []
        for x, y, w, h, area in stats[1:n]:
            if not (max(1.0, BLOB_MIN_AREA * s2) <= area <= BLOB_MAX_AREA * s2):
                continue
            if max(w, h) > BLOB_MAX_ELONGATION * max(1, min(w, h)):
                continue
            cx = ox + (x + w / 2) * to_full + dx
            cy = oy + (y + h / 2) * to_full + dy
            if within_reach(cx, cy, players):
                continue
            found.append((float(diff[y:y + h, x:x + w].max()) * area, cx, cy))
        found.sort(reverse=True)
        origins = []
        for _, cx, cy in found[:MAX_BLOB_PROPOSALS]:
            o = self.crops.clamp(snap, cx - dx - self.size / 2, cy - dy - self.size / 2)
            if o not in origins:
                origins.append(o)
        return origins

    # ── detection ────────────────────────────────────────────────────────────

    def _detect(self, snap, frame, origins, shift):
        """Detections (x1, y1, x2, y2, conf) in frame px over the crops at `origins`."""
        if not origins:
            return []
        batch = [self.crops.channels(snap, frame, self.prev, x0, y0, shift, self.prev_shift)
                 for x0, y0 in origins]
        self.n_crops += len(batch)
        dx, dy = shift[0] / self.crops.scale, shift[1] / self.crops.scale
        boxes, scores = [], []
        for (x0, y0), dets in zip(origins, self.backend.detect(batch)):
            for x1, y1, x2, y2, conf in dets:
                boxes.append((x1 + x0 + dx, y1 + y0 + dy, x2 + x0 + dx, y2 + y0 + dy))
                scores.append(conf)
        if len(origins) > 1 and boxes:
            return _nms(np.array(boxes), scores, 0.0, NMS_IOU)
        return [(*b, c) for b, c in zip(boxes, scores)]

    # ── state machine ────────────────────────────────────────────────────────

    def _start(self, frame_idx, det, v=(0.0, 0.0)):
        self.kf = Kalman(((det[0] + det[2]) / 2, (det[1] + det[3]) / 2), v, self.s)
        self.state, self.misses, self.tentative = self.TRACKING, 0, []

    def _acquire(self, frame_idx, dets):
        """SEARCH: start a track on a confident detection or two consistent ones."""
        good = [d for d in dets if d[4] >= ACQUIRE_CONF]
        self.tentative = [t for t in self.tentative if frame_idx - t[0] <= ACQUIRE_FRAMES]
        if not good:
            return None
        best = max(good, key=lambda d: d[4])
        cx, cy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
        if best[4] >= STRONG_CONF:
            self._start(frame_idx, best)
            return best
        for f, tx, ty, _ in reversed(self.tentative):
            gap = frame_idx - f
            if 0 < gap and np.hypot(cx - tx, cy - ty) <= MAX_SPEED_PX * self.s * gap:
                self._start(frame_idx, best, ((cx - tx) / gap, (cy - ty) / gap))
                return best
        self.tentative.append((frame_idx, cx, cy, best[4]))
        return None

    def step(self, frame_idx, frame, snap, shift=(0.0, 0.0), players=(), small_diff=None):
        """Process one frame. Returns (live, candidates).

        live: {"cx", "cy", "detected", "state"} or None (no track). candidates:
        [(frame, x1, y1, x2, y2, conf, motion, source), ...] of this frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        live, rows = None, []
        if snap is not None and snap.full is not None:
            if self.state == self.TRACKING:
                gap = 1 if self.prev_idx is None else max(1, frame_idx - self.prev_idx)
                live, rows = self._track(frame_idx, frame, gray, snap, shift, players, gap)
            if self.state == self.SEARCH and frame_idx % self.search_every == 0:
                if self.search == "tiles":
                    origins = self._tiles(snap)
                else:
                    origins = self._blob_proposals(snap, frame, shift, players, small_diff)
                self.last_crops = origins
                dets = self._detect(snap, frame, origins, shift)
                rows += self._rows(frame_idx, dets, gray, "search")
                got = self._acquire(frame_idx, dets)
                if got is not None:
                    live = {"cx": (got[0] + got[2]) / 2, "cy": (got[1] + got[3]) / 2,
                            "detected": True, "state": self.state}
        self.prev, self.prev_gray, self.prev_shift = frame, gray, shift
        self.prev_idx = frame_idx
        return live, rows

    def _track(self, frame_idx, frame, gray, snap, shift, players, gap=1):
        near = within_reach(*self.kf.x[:2], players)
        for _ in range(gap):                    # frames dropped by a live source
            px, py = self.kf.predict(REACH_Q_GAIN if near else 1.0)
        dx, dy = shift[0] / self.crops.scale, shift[1] / self.crops.scale
        origin = self.crops.clamp(snap, px - dx - self.size / 2, py - dy - self.size / 2)
        self.last_crops = [origin]
        dets = self._detect(snap, frame, [origin], shift)
        rows = self._rows(frame_idx, dets, gray, "crop")

        chosen = None
        gated = []
        for d in dets:
            z = ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2)
            if (self.kf.mahalanobis2(z) <= GATE_CHI2
                    or np.hypot(z[0] - px, z[1] - py) <= MIN_GATE_PX * self.s):
                gated.append(d)
        if gated:
            chosen = max(gated, key=lambda d: d[4])
            self.kf.update(((chosen[0] + chosen[2]) / 2, (chosen[1] + chosen[3]) / 2))
        else:
            strong = [d for d in dets if d[4] >= STRONG_CONF]
            if strong:
                chosen = max(strong, key=lambda d: d[4])
                cx, cy = (chosen[0] + chosen[2]) / 2, (chosen[1] + chosen[3]) / 2
                last = self.kf.x[:2] - self.kf.x[2:4]       # position one frame ago
                self._start(frame_idx, chosen, (cx - last[0], cy - last[1]))
        if chosen is not None:
            self.misses = 0
        else:
            self.misses += 1
            if self.misses > MAX_COAST:
                self.state, self.kf = self.SEARCH, None
                return None, rows
        cx, cy = self.kf.x[:2]
        return {"cx": float(cx), "cy": float(cy), "detected": chosen is not None,
                "state": self.state}, rows

    def _rows(self, frame_idx, dets, gray, source):
        return [(frame_idx, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                 round(conf, 4), round(motion_fraction(self.prev_gray, gray, (x1, y1, x2, y2)), 3),
                 source) for x1, y1, x2, y2, conf in dets]
