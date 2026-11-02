"""Two-player tracking against a background built from the whole clip.

1. Background: the per-pixel median of frames sampled evenly over the clip
   (black / fade-in frames excluded). The camera is fixed and the players move,
   so the median is the empty court, and a player standing still (e.g. while
   serving) still differs from it. People who barely move, such as ball kids,
   fall into the background.
2. Foreground: the frame is gain-matched to the background (absorbs fade-ins
   and exposure drift) and compared with the background's local min / max over
   a few pixels, so the slight shake of broadcast cameras does not turn court
   lines into foreground; cast shadows (darker pixels with the background's
   hue) are removed.
3. Candidates: connected components, merged when they are fragments of the
   same person, kept only if their feet stand in the walkable area of the
   court (court homography from the court CSV).
4. Identity from the court half: P1 is the player in the near half, P2 the one
   in the far half. A player seen recently is followed by proximity; otherwise
   the largest plausible blob in its half is taken, so a lost player comes
   back as soon as it is visible again.
5. Short gaps (<= MAX_GAP_S) are filled by linear interpolation and flagged in
   the `interpolated` column.

Output: outputs/player_coordinates/players_<video name>.csv with
frame, player_id, x, y, w, h, cx, cy, area, interpolated (full-resolution px).
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Make the project root importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.court_converter import CourtConverter
from utils.court_geometry import NET, RUNOFF_BEHIND, RUNOFF_SIDE, L_m, W_m
from utils.video import check_full_read, safe_fps

# ── Processing scale ──────────────────────────────────────────────────────────
PROC_HEIGHT = 540          # frames are processed at this height (boxes scaled back)

# ── Background ────────────────────────────────────────────────────────────────
BG_SAMPLES = 41            # frames per median (odd)
BG_SEGMENT_S = 30.0        # clips longer than 2 segments get one median per segment
DARK_SAMPLE_RATIO = 0.6    # samples darker than this x the typical one are skipped

# ── Foreground ────────────────────────────────────────────────────────────────
DARK_FRAME_RATIO = 0.5     # frames darker than this x the background: no detection
DIFF_THRESH = 30           # per-channel difference for a foreground pixel
JITTER_PX = 2              # broadcast cameras shake by a pixel or two: a pixel is
                           # foreground only if it differs from EVERY background
                           # pixel within this radius (at PROC_HEIGHT), so court
                           # lines and banner text never light up
FROZEN_TOL = 3             # a pixel within this of the previous frame is "unchanged"
FROZEN_S = 0.5             # unchanged this long while differing from the
                           # background = TV graphic, never a filmed player
SHADOW_V_RATIO = (0.5, 0.95)   # shadow: darker than the background by this ratio...
SHADOW_MAX_DHUE = 12           # ...with about the same hue (OpenCV hue, 0-180)...
SHADOW_MAX_DSAT = 50           # ...and saturation

# ── Candidates ────────────────────────────────────────────────────────────────
# Sizes are in metres-equivalent: multiples of the local image scale (px per
# metre across the court at the blob's feet), so they hold near and far.
MIN_FRAGMENT_AREA = 6      # px at PROC_HEIGHT: smaller specks are ignored
MIN_BLOB_AREA = 40         # px at PROC_HEIGHT: a merged candidate must be bigger
MIN_PLAYER_HEIGHT_M = 0.6  # plausible player: taller than this...
MAX_PLAYER_SIZE_M = (3.0, 3.5)  # ...and at most this (width, height): rejects
                                # TV graphics and other large blobs
MERGE_GAP_M = (0.3, 0.6)   # fragments closer than (x, y) merge into one person
BODY_COLUMN_FRAC = 0.2     # box columns thinner than this x the tallest one...
SHADOW_BAND_FRAC = 0.25    # ...and lying in the bottom quarter of the box are a
                           # cast shadow on the ground, and are trimmed off
AREA_MARGIN_M = 0.5        # slack around the walkable area for a player's feet
OFF_COURT_MARGIN_M = 5.0   # raw blobs whose base is farther than this beyond the
                           # walkable area are graphics / crowd, never a player

# ── Identity ──────────────────────────────────────────────────────────────────
RECENT_S = 0.5             # a player seen within this time is followed by proximity
# Proximity gate on the feet point, in units of the box height of the player:
# pixel distances stay meaningful at the far end, where 1 px is many
# centimetres. Gate = GATE_H + GATE_H_PER_FRAME * frames since last seen.
GATE_H = 0.5
GATE_H_PER_FRAME = 0.3

# ── Gaps ──────────────────────────────────────────────────────────────────────
MAX_GAP_S = 0.5            # gaps up to this long are interpolated

COLUMNS = ["frame", "player_id", "x", "y", "w", "h", "cx", "cy", "area", "interpolated"]


def _luma_median(img_bgr):
    return float(np.median(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))


def _median_stack(frames):
    """Per-pixel median of uint8 frames, computed in row strips to bound memory."""
    stack = np.stack(frames)
    out = np.empty(stack.shape[1:], np.uint8)
    for r in range(0, stack.shape[1], 64):
        out[r:r + 64] = np.median(stack[:, r:r + 64], axis=0).astype(np.uint8)
    return out


class PlayerTracker:
    def __init__(self, video_path, csv_path, court_csv, display=True):
        self.video_path = video_path
        self.csv_path = csv_path
        self.display = display
        if not os.path.exists(court_csv):
            raise FileNotFoundError(
                f"Court CSV not found: {court_csv}\n"
                f"Run: python tracking/court_tracking.py --video {video_path} --no-display")
        self.conv = CourtConverter(court_csv)
        self.m_to_px = np.linalg.inv(self.conv._H)      # court metres -> pixels
        # Sign of the homogeneous w for image points on the court (net centre).
        centre_px = self.m_to_px @ np.array([W_m / 2, NET, 1.0])
        centre_px /= centre_px[2]
        self._w_sign = np.sign((self.conv._H @ centre_px)[2])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.scale = min(1.0, PROC_HEIGHT / self.H)      # full -> processing
        self.proc_size = (int(round(self.W * self.scale)), int(round(self.H * self.scale)))
        self.kernel_close = np.ones((7, 7), np.uint8)
        self.recent_frames = max(1, int(round(RECENT_S * self.fps)))
        self.max_gap = max(1, int(round(MAX_GAP_S * self.fps)))
        self.frozen_frames = max(2, int(round(FROZEN_S * self.fps)))
        self._prev_raw = None
        self._frozen = None

    # ── 1. background ────────────────────────────────────────────────────────

    def _build_backgrounds(self):
        """[(first frame, background BGR at processing scale), ...] per segment."""
        n = self.total_frames
        seg_len = int(BG_SEGMENT_S * self.fps)
        if n <= 2 * seg_len:
            bounds = [(0, n)]
        else:
            bounds = [(s, min(n, s + seg_len)) for s in range(0, n, seg_len)]
        wanted = set()
        for s, e in bounds:
            wanted.update(np.linspace(s, e - 1, min(BG_SAMPLES, e - s)).round().astype(int).tolist())

        samples = {i: [] for i in range(len(bounds))}
        cap = cv2.VideoCapture(self.video_path)
        try:
            for idx in range(n):
                if not cap.grab():
                    break
                if idx not in wanted:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                seg = next(k for k, (s, e) in enumerate(bounds) if s <= idx < e)
                samples[seg].append(cv2.resize(frame, self.proc_size, interpolation=cv2.INTER_AREA))
        finally:
            cap.release()

        backgrounds = []
        for k, (s, _) in enumerate(bounds):
            frames = samples[k]
            if not frames:
                continue
            lumas = np.array([_luma_median(f) for f in frames])
            keep = [f for f, lum in zip(frames, lumas) if lum >= DARK_SAMPLE_RATIO * np.median(lumas)]
            if len(keep) % 2 == 0 and len(keep) > 1:
                keep = keep[1:]
            backgrounds.append((s, _median_stack(keep)))
            samples[k] = None
        if not backgrounds:
            raise RuntimeError(f"Could not sample frames from {self.video_path}")
        self._bg_luma = [_luma_median(bg) for _, bg in backgrounds]
        self._bg_hsv = [cv2.cvtColor(bg, cv2.COLOR_BGR2HSV) for _, bg in backgrounds]
        k = np.ones((2 * JITTER_PX + 1, 2 * JITTER_PX + 1), np.uint8)
        self._bg_lo = [cv2.erode(bg, k) for _, bg in backgrounds]
        self._bg_hi = [cv2.dilate(bg, k) for _, bg in backgrounds]
        return backgrounds

    def _background_for(self, frame_idx):
        k = 0
        for i, (start, _) in enumerate(self.backgrounds):
            if frame_idx >= start:
                k = i
        return k

    # ── 2. foreground ────────────────────────────────────────────────────────

    def _frozen_pixels(self, raw):
        """Pixels unchanged (within FROZEN_TOL) for FROZEN_S: on-screen graphics.

        Graphics are composited on top of the video, so their pixels are exactly
        the same from frame to frame; anything filmed (players included) changes
        a little every frame through camera shake, noise and motion.
        """
        if self._prev_raw is None:
            self._frozen = np.zeros(raw.shape[:2], np.uint16)
        else:
            same = cv2.absdiff(raw, self._prev_raw).max(axis=2) <= FROZEN_TOL
            self._frozen = np.where(same, self._frozen + 1, 0).astype(np.uint16)
        self._prev_raw = raw
        return self._frozen >= self.frozen_frames

    def _foreground(self, small, k):
        """Binary foreground at processing scale, or None for a too-dark frame."""
        raw = small
        luma = _luma_median(small)
        if luma < DARK_FRAME_RATIO * self._bg_luma[k]:
            self._prev_raw = None
            return None
        gain = self._bg_luma[k] / max(luma, 1.0)
        if abs(gain - 1.0) > 0.02:
            small = cv2.convertScaleAbs(small, alpha=gain)

        # Distance to the background's local [min, max] range (0 inside it).
        diff = cv2.max(cv2.subtract(small, self._bg_hi[k]),
                       cv2.subtract(self._bg_lo[k], small)).max(axis=2)
        fg = diff > DIFF_THRESH

        # Cast shadows: darker than the background, same hue and saturation.
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)
        bg_hsv = self._bg_hsv[k].astype(np.int16)
        v_ratio = hsv[..., 2] / np.maximum(bg_hsv[..., 2], 1)
        dhue = np.abs(hsv[..., 0] - bg_hsv[..., 0])
        dhue = np.minimum(dhue, 180 - dhue)
        shadow = ((v_ratio >= SHADOW_V_RATIO[0]) & (v_ratio <= SHADOW_V_RATIO[1])
                  & (dhue <= SHADOW_MAX_DHUE)
                  & (np.abs(hsv[..., 1] - bg_hsv[..., 1]) <= SHADOW_MAX_DSAT))
        fg &= ~shadow

        fg &= ~self._frozen_pixels(raw)

        # No morphological opening: the legs of the far player are ~2 px wide at
        # PROC_HEIGHT and would vanish; specks are dropped by area instead.
        mask = self._drop_off_court(fg.astype(np.uint8) * 255)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

    def _drop_off_court(self, mask):
        """Remove raw blobs whose base lies far outside the walkable area.

        TV graphics and banners sit many metres beyond the court, but can be a
        few pixels from a player's head: dropping them BEFORE the closing step
        keeps them from fusing with the player. Any part of a player (even the
        head of one standing deep behind the baseline) has its base within
        OFF_COURT_MARGIN_M of the walkable area.
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return mask
        inv = 1.0 / self.scale
        base = np.c_[(stats[1:, 0] + stats[1:, 2] / 2) * inv,
                     (stats[1:, 1] + stats[1:, 3]) * inv, np.ones(n - 1)]
        proj = base @ self.conv._H.T
        # Points beyond the court's horizon line project "behind the camera":
        # their homogeneous w has the opposite sign of a point on the court.
        in_front = proj[:, 2] * self._w_sign > 1e-9
        xm = proj[:, 0] / np.where(in_front, proj[:, 2], 1.0)
        ym = proj[:, 1] / np.where(in_front, proj[:, 2], 1.0)
        margin = OFF_COURT_MARGIN_M
        on_court = ((xm >= -RUNOFF_SIDE - margin) & (xm <= W_m + RUNOFF_SIDE + margin)
                    & (ym >= -RUNOFF_BEHIND - margin) & (ym <= L_m + RUNOFF_BEHIND + margin))
        keep = np.r_[False, on_court & in_front]
        return np.where(keep[labels], 255, 0).astype(np.uint8)

    # ── 3. candidates ────────────────────────────────────────────────────────

    def _px_per_m(self, x_m, y_m):
        """Horizontal image scale (full-res px per metre) at a court point."""
        pts = np.array([[[x_m - 0.5, y_m]], [[x_m + 0.5, y_m]]], np.float64)
        p = cv2.perspectiveTransform(pts, self.m_to_px)[:, 0]
        return float(np.hypot(*(p[1] - p[0])))

    def _scale_at_feet(self, box):
        """(feet in metres, px per metre there) for a box dict, or (None, None)."""
        fx, fy = self.conv.to_meters((box["x0"] + box["x1"]) / 2, box["y1"])
        if not (np.isfinite(fx) and np.isfinite(fy)):
            return None, None
        return (fx, fy), self._px_per_m(fx, fy)

    def _trim_to_body(self, m, labels):
        """Trim a cast shadow off a candidate's box.

        A cast shadow is a thin strip on the ground, a few pixels tall at the
        bottom of the box, while the columns of the body hold many foreground
        pixels. Thin columns in the bottom band are dropped, keeping the run
        around the tallest column. Updates the box, the area and the pixel
        centroid in place (full-res px).
        """
        s, inv = self.scale, 1.0 / self.scale
        x0, y0 = int(round(m["x0"] * s)), int(round(m["y0"] * s))
        x1, y1 = int(round(m["x1"] * s)), int(round(m["y1"] * s))
        body = np.isin(labels[y0:y1, x0:x1], m["ids"])
        cols = body.sum(axis=0)
        if not cols.any():
            return
        # A column is shadow only if it is thin AND lies on the ground: all its
        # pixels in the bottom SHADOW_BAND_FRAC of the box. An outstretched arm
        # or racket is thin but high, and must stay in the box (shot detection
        # looks for the ball inside the player's box at contact).
        top = np.where(body.any(axis=0), body.argmax(axis=0), body.shape[0])
        on_ground = top >= (1.0 - SHADOW_BAND_FRAC) * body.shape[0]
        keep = (cols >= BODY_COLUMN_FRAC * cols.max()) | ~on_ground
        peak = int(np.argmax(cols))
        lo = peak
        while lo > 0 and keep[lo - 1]:
            lo -= 1
        hi = peak
        while hi < len(keep) - 1 and keep[hi + 1]:
            hi += 1
        body = body[:, lo:hi + 1]
        rows = np.nonzero(body.any(axis=1))[0]
        ys, xs = np.nonzero(body)
        m["x0"], m["x1"] = (x0 + lo) * inv, (x0 + hi + 1) * inv
        m["y0"], m["y1"] = (y0 + rows[0]) * inv, (y0 + rows[-1] + 1) * inv
        m["area"] = len(xs) * inv * inv
        m["cx"] = (x0 + lo + xs.mean() + 0.5) * inv
        m["cy"] = (y0 + ys.mean() + 0.5) * inv

    def _too_big(self, box, ppm):
        return (box["x1"] - box["x0"] > MAX_PLAYER_SIZE_M[0] * ppm
                or box["y1"] - box["y0"] > MAX_PLAYER_SIZE_M[1] * ppm)

    def _candidates(self, mask):
        """Merged, court-filtered candidates as dicts (full-res px + metres)."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        inv = 1.0 / self.scale
        comps = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_FRAGMENT_AREA:
                continue
            c = {"x0": x * inv, "y0": y * inv, "x1": (x + w) * inv, "y1": (y + h) * inv,
                 "area": area * inv * inv, "ids": [i]}
            feet, ppm = self._scale_at_feet(c)
            # Height only: a TV graphic is far too tall for a person, while a
            # player with a long cast shadow is wide but not tall.
            if feet is None or c["y1"] - c["y0"] > MAX_PLAYER_SIZE_M[1] * ppm:
                continue            # never let such a blob absorb a player
            c["ppm"] = ppm
            comps.append(c)
        if not comps:
            return []

        # Merge fragments of the same person (split legs, racket, torso/shorts),
        # largest first so small fragments attach to the body they belong to.
        comps.sort(key=lambda c: -c["area"])
        merged = []
        for c in comps:
            for m in merged:
                ppm = m["ppm"]
                gx, gy = MERGE_GAP_M[0] * ppm, MERGE_GAP_M[1] * ppm
                if (c["x0"] <= m["x1"] + gx and c["x1"] >= m["x0"] - gx
                        and c["y0"] <= m["y1"] + gy and c["y1"] >= m["y0"] - gy):
                    m["area"] += c["area"]
                    m["ids"] += c["ids"]
                    for key, fn in (("x0", min), ("y0", min), ("x1", max), ("y1", max)):
                        m[key] = fn(m[key], c[key])
                    break
            else:
                merged.append(dict(c, ids=list(c["ids"])))

        out = []
        for m in merged:
            if m["area"] < MIN_BLOB_AREA * inv * inv:
                continue
            self._trim_to_body(m, labels)
            feet, ppm = self._scale_at_feet(m)
            if feet is None:
                continue
            fx, fy = feet
            if not (-RUNOFF_SIDE - AREA_MARGIN_M <= fx <= W_m + RUNOFF_SIDE + AREA_MARGIN_M
                    and -RUNOFF_BEHIND - AREA_MARGIN_M <= fy <= L_m + RUNOFF_BEHIND + AREA_MARGIN_M):
                continue
            if m["y1"] - m["y0"] < MIN_PLAYER_HEIGHT_M * ppm or self._too_big(m, ppm):
                continue
            m["feet_m"] = feet
            out.append(m)
        return out

    # ── 4. identities ────────────────────────────────────────────────────────

    def _pick(self, cands, track, frame_idx):
        """Best candidate for one player given its track state, or None."""
        if not cands:
            return None
        last_frame, last = track
        missed = frame_idx - last_frame if last_frame is not None else None
        if missed is not None and missed <= self.recent_frames:
            lx, ly = (last["x0"] + last["x1"]) / 2, last["y1"]
            gate = (GATE_H + GATE_H_PER_FRAME * missed) * (last["y1"] - last["y0"])
            dists = [np.hypot((c["x0"] + c["x1"]) / 2 - lx, c["y1"] - ly) for c in cands]
            k = int(np.argmin(dists))
            return cands[k] if dists[k] <= gate else None
        return max(cands, key=lambda c: c["area"])

    # ── 5. gaps ──────────────────────────────────────────────────────────────

    def _fill_gaps(self, detections):
        """detections {pid: {frame: row}} -> sorted rows with short gaps interpolated."""
        rows = []
        for pid, by_frame in detections.items():
            frames = sorted(by_frame)
            for a, b in zip(frames, frames[1:]):
                rows.append(by_frame[a])
                gap = b - a - 1
                if 0 < gap <= self.max_gap:
                    ra, rb = by_frame[a], by_frame[b]
                    for f in range(a + 1, b):
                        t = (f - a) / (b - a)
                        rows.append({k: (ra[k] + t * (rb[k] - ra[k]))
                                     for k in ("x0", "y0", "x1", "y1", "cx", "cy", "area")}
                                    | {"frame": f, "pid": pid, "interpolated": 1})
            if frames:
                rows.append(by_frame[frames[-1]])
        rows.sort(key=lambda r: (r["frame"], r["pid"]))
        return rows

    def _write_csv(self, rows):
        tmp = self.csv_path + ".part"
        try:
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(COLUMNS)
                for r in rows:
                    x, y = int(round(r["x0"])), int(round(r["y0"]))
                    w_, h_ = int(round(r["x1"] - r["x0"])), int(round(r["y1"] - r["y0"]))
                    w.writerow([r["frame"], r["pid"], x, y, w_, h_,
                                round(r["cx"], 1), round(r["cy"], 1), int(round(r["area"])),
                                r.get("interpolated", 0)])
            os.replace(tmp, self.csv_path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        print(f"Building the background from {BG_SAMPLES} frames sampled over the clip...")
        self.backgrounds = self._build_backgrounds()

        detections = {1: {}, 2: {}}
        tracks = {1: (None, None), 2: (None, None)}
        frame_idx, stopped_by_user, dark_frames = 0, False, 0
        cap = cv2.VideoCapture(self.video_path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                small = cv2.resize(frame, self.proc_size, interpolation=cv2.INTER_AREA)
                mask = self._foreground(small, self._background_for(frame_idx))
                if mask is None:
                    dark_frames += 1
                else:
                    cands = self._candidates(mask)
                    halves = {1: [c for c in cands if c["feet_m"][1] > NET],
                              2: [c for c in cands if c["feet_m"][1] <= NET]}
                    for pid in (1, 2):
                        best = self._pick(halves[pid], tracks[pid], frame_idx)
                        if best is not None:
                            detections[pid][frame_idx] = best | {"frame": frame_idx, "pid": pid}
                            tracks[pid] = (frame_idx, best)

                if self.display and self._show(frame, mask, detections, frame_idx):
                    stopped_by_user = True
                    break
                frame_idx += 1
            if not stopped_by_user:
                check_full_read(frame_idx, self.total_frames, self.video_path)
        finally:
            cap.release()
            cv2.destroyAllWindows()

        rows = self._fill_gaps(detections)
        self._write_csv(rows)
        processed = max(frame_idx - dark_frames, 1)
        for pid in (1, 2):
            real = len(detections[pid])
            print(f"  P{pid}: detected on {real / processed:.0%} of the usable frames"
                  f" ({sum(1 for r in rows if r['pid'] == pid) - real} gap frames interpolated)")
        if dark_frames:
            print(f"  {dark_frames} frames skipped as too dark (fade-in / black frames).")
        if stopped_by_user:
            print(f"WARNING: stopped with 'q' at frame {frame_idx} of {self.total_frames}; "
                  f"{self.csv_path} covers only the frames processed so far.")
        print(f"Tracking saved to {self.csv_path}")

    def _show(self, frame, mask, detections, frame_idx):
        """Draw the current boxes; return True if the user pressed q."""
        for pid, color in ((1, (0, 255, 0)), (2, (0, 0, 255))):
            d = detections[pid].get(frame_idx)
            if d is not None:
                p0 = (int(d["x0"]), int(d["y0"]))
                cv2.rectangle(frame, p0, (int(d["x1"]), int(d["y1"])), color, 2)
                cv2.putText(frame, f"P{pid}", (p0[0], max(0, p0[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Player Tracker", cv2.resize(frame, (960, 540)))
        if mask is not None:
            cv2.imshow("FG mask", cv2.resize(mask, (480, 270)))
        return (cv2.waitKey(1) & 0xFF) == ord("q")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-player tracking: whole-clip median background + court-half identities")
    parser.add_argument("--video", default="data/Input_video2.mp4")
    parser.add_argument("--court", default=None,
                        help="court CSV (default: outputs/court_coordinates/<video name>_court.csv)")
    parser.add_argument("--csv", default=None,
                        help="output CSV path (defaults to "
                             "outputs/player_coordinates/players_<video name>.csv)")
    parser.add_argument("--no-display", action="store_true",
                        help="run headless (no OpenCV windows)")
    args = parser.parse_args()

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    if args.csv is None:
        args.csv = os.path.join("outputs", "player_coordinates", f"players_{video_stem}.csv")
    if args.court is None:
        args.court = os.path.join("outputs", "court_coordinates", f"{video_stem}_court.csv")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    PlayerTracker(args.video, args.csv, args.court, display=not args.no_display).run()
