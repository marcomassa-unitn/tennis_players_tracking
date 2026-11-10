"""Ball tracking: YOLO candidates at full resolution, then a physical tracker.

Stage 1 -- detection (needs ultralytics + the YOLOv8 weights; GPU recommended).
    Each frame is split into overlapping TILE x TILE tiles at native
    resolution, and the detector runs on all tiles plus the whole frame in one
    batch. Whole-frame inference shrinks a 1080p frame to 640 px and the ~15 px
    ball to ~5 px; on the tiles it keeps its real size. The whole frame is kept
    too, because the model was trained on whole downscaled frames and some
    (motion-blurred) balls are only found that way. Duplicates are merged with
    NMS, and every box above CANDIDATE_CONF is kept, with the fraction of its
    pixels that moved since the previous frame and its source ("tile" or
    "frame"), in outputs/<video name>/ball_coordinates/ball_candidates_<video name>.csv.

Stage 2 -- tracking (CPU, no model needed).
    Still candidates are discarded (as the original motion gate did). Among
    the rest, the ball is the best path over the whole clip (dynamic
    programming): confident, moving candidates are rewarded, links between
    them cost in proportion to the implied speed and to the frames skipped,
    physically impossible links are forbidden, and restarting a track has a
    fixed cost. Knowing the frames after a racket hit, the path picks the
    right candidate where a frame-by-frame filter would follow the old
    direction. Isolated detections that jump away and back are dropped, and
    gaps up to MAX_GAP_S are
    filled with a shape-preserving cubic (PCHIP) through the real detections;
    longer gaps and the clip edges are left empty.

Output: outputs/<video name>/ball_coordinates/ball_<video name>.csv with
frame, x, y, w, h, cx, cy, area, interpolated.

Stage 2 also adds candidates of its own, on the CPU, once the players and the court
are known: ball-sized blobs in the difference from the clip's median background.
They do not depend on how the ball looks, so they fill in where the detector does
not see it, and the path through the candidates keeps those that fly like a ball.

Also written, from the "frame" candidates only: ball_wholeframe_<video name>.csv,
the track of the original single-pass detector (whole frame, conf >= 0.65,
motion gate, highest confidence, straight-line fills), kept to compare with it.

Usage (from project root):
    python tracking/ball_tracking.py --video data/Input_video2.mp4             # both stages
    python tracking/ball_tracking.py --video data/Input_video2.mp4 --detect-only   # stage 1 (Colab)
    python tracking/ball_tracking.py --video data/Input_video2.mp4 \\
        --candidates outputs/Input_video2/ball_coordinates/ball_candidates_Input_video2.csv    # stage 2 only
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Make the project root importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.background import channel_max
from utils.background import median_stack as _median_stack
from utils.court_converter import CourtConverter
from utils.court_geometry import L_m, W_m
from utils.paths import OUTPUT_ROOT, clip_paths
from utils.video import check_full_read, safe_fps

# ── Stage 1: detection ────────────────────────────────────────────────────────
TILE = 640                 # tile size = the detector's input size: no downscaling
TILE_MIN_OVERLAP = 48      # px; > 3x the ball, so a ball on a seam is whole in a tile
CANDIDATE_CONF = 0.15      # keep every box above this; the tracker decides later
NMS_IOU = 0.3              # merge duplicates of the same ball from overlapping tiles
MOTION_THRESH = 15         # per-pixel intensity change (0-255) that counts as motion

# ── Stage 2: tracking (pixel values for 1080p, scaled with the frame height) ──
MIN_TRACK_CONF = 0.15      # candidates below this are ignored by the tracker
MIN_MOTION = 0.05          # ...and so are still ones (balls held by ball kids or in
                           # pockets): at least this fraction of the box must move
MAX_SPEED_PX = 150         # per frame: farther candidates cannot be the same ball
CONF_BIAS = 0.15           # a node is worth conf + MOTION_BONUS * motion - CONF_BIAS
MOTION_BONUS = 0.2
SPEED_COST = 0.004         # path cost per px/frame of speed between linked nodes...
SKIP_COST = 0.1            # ...plus this per frame skipped between them
MAX_LINK_S = 0.33          # nodes farther apart in time can only be joined by a restart
RESTART_COST = 1.0         # starting a new path segment
SPIKE_SPEED_PX = 80        # a detection this fast in and out...
SPIKE_DETOUR = 3.0         # ...on a path this much longer than its neighbours' gap
                           # is a one-frame false positive
MAX_GAP_S = 0.33           # gaps up to this long are filled; longer ones stay empty

# ── Motion candidates (pixel values for 1080p, scaled with the frame height) ──
BG_FRAMES = 41             # frames of the median background, evenly spread over the clip
DIFF_THRESH = 30           # colour difference from the background counting as foreground
BLOB_MIN_AREA = 6          # px: a ball, not noise...
BLOB_MAX_AREA = 260        # ...and not a person
BLOB_MAX_ELONGATION = 5.0  # a blurred ball is a short streak, a line is not
BLOB_MIN_MOTION = 0.2      # fraction of the blob changed since the previous frame
BLOB_CONF = 0.35           # the confidence a blob counts for in the path
BLOB_REACH = 0.6           # player heights around a player where a blob is their racket
COURT_SIDE_M = 3.0         # blobs count over the court, this far beyond the sidelines...
COURT_BACK_M = 3.0         # ...and the near baseline...
COURT_FAR_M = 1.0          # ...and the far one: boards start right behind it
MIN_BLOB_RUN = 3           # blobs alone must follow each other this many times...
BLOB_RUN_GAP = 2           # ...at most this many frames apart

CANDIDATE_COLUMNS = ["frame", "x1", "y1", "x2", "y2", "conf", "motion", "source"]
WHOLEFRAME_CONF = 0.65     # the original detector's confidence threshold
BALL_COLUMNS = ["frame", "x", "y", "w", "h", "cx", "cy", "area", "interpolated"]


# ═════════════════════════════ Stage 1: detection ═════════════════════════════

def tile_origins(width, height, tile=TILE, min_overlap=TILE_MIN_OVERLAP):
    """Top-left corners of tile x tile windows covering a width x height frame."""
    def axis(length):
        if length <= tile:
            return [0]
        n = int(np.ceil((length - tile) / (tile - min_overlap))) + 1
        return [int(round(v)) for v in np.linspace(0, length - tile, n)]
    return [(x, y) for y in axis(height) for x in axis(width)]


def motion_fraction(prev_gray, gray, box):
    """Fraction of the box pixels whose intensity changed since the previous frame."""
    h, w = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if prev_gray is None or x2 <= x1 or y2 <= y1:
        return 0.0
    diff = cv2.absdiff(prev_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2])
    return float((diff > MOTION_THRESH).mean())


class BallDetector:
    """Runs the YOLO ball detector on full-resolution tiles of every frame."""

    def __init__(self, model_path):
        from ultralytics import YOLO  # only stage 1 needs ultralytics
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        """[(x1, y1, x2, y2, conf, source), ...] for one frame.

        source "tile": full-resolution tiles, duplicates merged across tiles;
        source "frame": the whole frame in its own call, exactly as the original
        single-pass detector ran it (so its track can be rebuilt bit for bit).
        """
        h, w = frame.shape[:2]
        origins = tile_origins(w, h)
        out = []
        if len(origins) > 1:
            tiles = [frame[y:y + TILE, x:x + TILE] for x, y in origins]
            results = self.model.predict(tiles, imgsz=TILE, conf=CANDIDATE_CONF, verbose=False)
            boxes, scores = [], []
            for (ox, oy), r in zip(origins, results):
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(),
                                               r.boxes.conf.cpu().numpy()):
                    boxes.append([float(x1 + ox), float(y1 + oy), float(x2 - x1), float(y2 - y1)])
                    scores.append(float(c))
            if boxes:
                keep = cv2.dnn.NMSBoxes(boxes, scores, CANDIDATE_CONF, NMS_IOU)
                out += [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2],
                         boxes[i][1] + boxes[i][3], scores[i], "tile")
                        for i in np.array(keep).ravel()]
        r = self.model.predict(frame, conf=CANDIDATE_CONF, verbose=False)[0]
        out += [(*map(float, b), float(c), "frame")
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        return out

    def detect_candidates(self, video_path, out_csv):
        """Write every candidate of every frame to `out_csv`; return the frame count."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Detecting ball candidates in {total or '?'} frames "
              f"({len(tile_origins(int(cap.get(3)), int(cap.get(4))))} tiles + the whole frame, per frame)...")
        rows, prev_gray, idx = [], None, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for x1, y1, x2, y2, conf, source in self.detect_frame(frame):
                    rows.append([idx, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                                 round(conf, 4),
                                 round(motion_fraction(prev_gray, gray, (x1, y1, x2, y2)), 3),
                                 source])
                prev_gray = gray
                idx += 1
                if idx % 100 == 0:
                    print(f"  {idx} frames")
        finally:
            cap.release()
        check_full_read(idx, total, video_path)
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(CANDIDATE_COLUMNS)
            wr.writerows(rows)
        print(f"Saved {len(rows)} candidates to {out_csv}")
        return idx


# ══════════════════════════ Motion candidates (CPU) ═══════════════════════════

def clip_background(video_path, n=BG_FRAMES):
    """Median of frames spread over the clip: the empty court, since everyone moves."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    return _median_stack(frames)


def ball_sized_blobs(frame, background, frame_height=1080):
    """Boxes (x1, y1, x2, y2) of the ball-sized blobs differing from the background."""
    s2 = (frame_height / 1080) ** 2
    mask = (channel_max(cv2.absdiff(frame, background)) >= DIFF_THRESH).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:n]:
        if (BLOB_MIN_AREA * s2 <= area <= BLOB_MAX_AREA * s2
                and max(w, h) <= BLOB_MAX_ELONGATION * max(1, min(w, h))):
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def court_pixels(conv):
    """Pixel limits (x0, x1, y0, y1) of the court and the run-off where blobs count."""
    m_to_px = np.linalg.inv(conv._H)
    corners = [(-COURT_SIDE_M, -COURT_FAR_M), (W_m + COURT_SIDE_M, -COURT_FAR_M),
               (-COURT_SIDE_M, L_m + COURT_BACK_M), (W_m + COURT_SIDE_M, L_m + COURT_BACK_M)]
    pts = []
    for x, y in corners:
        u, v, w = m_to_px @ np.array([x, y, 1.0])
        pts.append((u / w, v / w))
    pts = np.array(pts)
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def within_reach(cx, cy, players):
    """True if (cx, cy) is on a player or within reach of them: their racket, say."""
    for x, y, w, h in players:
        reach = BLOB_REACH * h
        if x - reach <= cx <= x + w + reach and y - reach <= cy <= y + 1.2 * h:
            return True
    return False


def motion_candidates(video_path, players_csv, court_csv):
    """Candidates (CANDIDATE_COLUMNS) from the difference with the median background."""
    players = pd.read_csv(players_csv)
    boxes = {f: g[["x", "y", "w", "h"]].to_numpy(float) for f, g in players.groupby("frame")}
    x0, x1, y0, y1 = court_pixels(CourtConverter(court_csv))
    background = clip_background(video_path)

    rows = []
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    previous, f = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for box in ball_sized_blobs(frame, background, height):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1) or within_reach(cx, cy, boxes.get(f, ())):
                continue
            motion = motion_fraction(previous, gray, box)
            if motion >= BLOB_MIN_MOTION:
                rows.append((f, *box, BLOB_CONF, motion, "motion"))
        previous, f = gray, f + 1
    cap.release()
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def drop_short_blob_runs(track, blobs):
    """Remove the blobs of `track` not part of a run of detections.

    A ball leaves a series; a stray blob that happens to fit the path does not.
    `blobs` holds the (frame, x1, y1) of the motion candidates.
    """
    frames = sorted(track)
    track = dict(track)
    start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] - frames[i - 1] > BLOB_RUN_GAP:
            run = frames[start:i]
            if len(run) < MIN_BLOB_RUN:
                for f in run:
                    if (f, track[f][0], track[f][1]) in blobs:
                        del track[f]
            start = i
    return track


# ═════════════════════════════ Stage 2: tracking ══════════════════════════════

class PathDP:
    """The forward pass of the path search, one frame at a time.

    push() adds the candidates of a frame (rows of frame, x1, y1, x2, y2, conf,
    motion, already filtered) and updates the best score of a path ending at each;
    backtrack() follows the parents back from a node. track_candidates() pushes the
    whole clip and backtracks once; OnlinePathDP backtracks a little after every
    frame and commits what is old enough.
    """

    def __init__(self, fps, frame_height=1080):
        self.s = frame_height / 1080
        self.max_speed = MAX_SPEED_PX * self.s
        self.max_link = max(1, int(round(MAX_LINK_S * fps)))
        self.rows, self.pos, self.best, self.back = {}, {}, {}, {}
        self.run_best = (0.0, None)     # best score among nodes older than the link window
        self.pending = []               # frames whose nodes are still within the window

    def push(self, f, r):
        s = self.s
        while self.pending and self.pending[0] < f - self.max_link:
            g = self.pending.pop(0)
            k = int(np.argmax(self.best[g]))
            if self.best[g][k] > self.run_best[0]:
                self.run_best = (self.best[g][k], (g, k))
        pos_f = np.c_[(r[:, 1] + r[:, 3]) / 2, (r[:, 2] + r[:, 4]) / 2]
        gain = r[:, 5] + MOTION_BONUS * np.minimum(r[:, 6], 0.5) - CONF_BIAS
        n = len(pos_f)
        # Start (or restart after the window) a path at this node.
        score = np.full(n, self.run_best[0] - RESTART_COST)
        parent = [self.run_best[1]] * n
        for g in self.pending:          # link from candidates in the previous frames
            gap = f - g
            pg = self.pos[g]
            speed = np.hypot(pos_f[None, :, 0] - pg[:, None, 0],
                             pos_f[None, :, 1] - pg[:, None, 1]) / gap
            cost = SPEED_COST * speed / s + SKIP_COST * (gap - 1)
            cost[speed > self.max_speed] = np.inf
            via = self.best[g][:, None] - cost
            i = np.argmax(via, axis=0)
            better = via[i, np.arange(n)] > score
            score = np.where(better, via[i, np.arange(n)], score)
            parent = [(g, int(i[j])) if better[j] else parent[j] for j in range(n)]
        self.rows[f], self.pos[f] = r, pos_f
        self.best[f] = score + gain
        self.back[f] = parent
        self.pending.append(f)

    def best_end(self, frames=None):
        """The node (frame, k) with the highest score, among `frames` (default all)."""
        frames = list(self.best if frames is None else frames)
        if not frames:
            return None
        return max(((f, int(np.argmax(self.best[f]))) for f in frames),
                   key=lambda fk: self.best[fk[0]][fk[1]])

    def backtrack(self, node, stop_before=None):
        """{frame: (x1, y1, x2, y2)} of the path ending at `node`, back to `stop_before`."""
        track = {}
        while node is not None:
            f, k = node
            if stop_before is not None and f < stop_before:
                break
            track[int(f)] = tuple(self.rows[f][k, 1:5])
            node = self.back[f][k]
        return track

    def forget_before(self, frame):
        """Drop the nodes older than `frame`; their best score lives on in run_best."""
        for f in [f for f in self.best if f < frame and f not in self.pending]:
            del self.rows[f], self.pos[f], self.best[f], self.back[f]


def usable_candidates(cands):
    """The candidates the tracker reads: confident enough, and moving.

    Still candidates are not the ball in play (motion is undefined on the first
    frame, which has no previous frame to compare with).
    """
    first = int(cands.frame.min()) if len(cands) else 0
    keep = (cands.conf >= MIN_TRACK_CONF) & ((cands.motion >= MIN_MOTION) | (cands.frame == first))
    return cands[keep]


def track_candidates(cands, n_frames, fps, frame_height=1080):
    """Pick the ball among per-frame candidates with the best path over the clip.

    Offline dynamic programming (Viterbi-style) on a graph whose nodes are the
    candidates. A node earns its confidence (plus a motion bonus) minus
    CONF_BIAS; linking two nodes up to MAX_LINK_S apart costs SPEED_COST per
    px/frame of the implied speed plus SKIP_COST per skipped frame, and is
    forbidden above MAX_SPEED_PX. A path may also stop and restart elsewhere
    for RESTART_COST, which a few good detections easily pay for but a weak
    isolated object does not. Because the whole clip is known, a racket hit is
    resolved by the frames after it, which a frame-by-frame filter cannot use.

    cands: DataFrame with CANDIDATE_COLUMNS. Returns {frame: (x1, y1, x2, y2)}
    of the real detections on the best path.
    """
    numeric = ["frame", "x1", "y1", "x2", "y2", "conf", "motion"]
    dp = PathDP(fps, frame_height)
    for f, g in usable_candidates(cands).groupby("frame"):
        dp.push(f, g[numeric].to_numpy(float))
    end = dp.best_end()
    return {} if end is None else dp.backtrack(end)


class OnlinePathDP:
    """Fixed-lag version of the offline tracking, for the live pipeline.

    push() is called once per frame t, in order, with that frame's candidates. The
    best path so far is followed back and its node at frame t - lag, if any, is
    committed: by then the frames after a racket hit have settled the choice, and a
    committed frame is never revisited. The clean-up of the offline tracker (short
    blob runs, outliers) then runs on a rolling window and finalises the frame
    `clean_lag` frames older still. With the defaults a position is final 0.5 s
    after its frame at 30 fps.
    """

    WINDOW = 30                         # frames of context kept for the clean-up

    def __init__(self, fps, frame_height=1080, lag=10, clean_lag=6):
        self.dp = PathDP(fps, frame_height)
        self.fps, self.height = fps, frame_height
        self.lag, self.clean_lag = lag, clean_lag
        self.first = None
        self.last = -1
        self.decided = -1               # last frame whose path node is decided
        self.finalised = -1             # last frame finalised
        self.committed = {}             # frame -> box, from the path (recent only)
        self.blobs = set()              # (frame, x1, y1) of the motion candidates
        self.final = {}                 # frame -> box, after the clean-up

    @property
    def latency(self):
        return self.lag + self.clean_lag

    def push(self, frame, rows):
        """Add the candidates [(frame, x1, y1, x2, y2, conf, motion, source)] of `frame`.

        Returns the positions finalised by this call, {frame: box}.
        """
        if self.first is None:
            self.first = frame
        self.last = frame
        for r in rows:
            if r[7] == "motion":
                self.blobs.add((int(r[0]), float(r[1]), float(r[2])))
        good = [r[:7] for r in rows if r[5] >= MIN_TRACK_CONF
                and (r[6] >= MIN_MOTION or frame == self.first)]
        if good:
            self.dp.push(frame, np.array(good, float))
        self._decide(frame - self.lag)
        return self._finalise(frame - self.lag - self.clean_lag)

    def _decide(self, upto):
        if upto <= self.decided:
            return
        recent = [f for f in self.dp.best if f > self.decided - self.dp.max_link]
        end = self.dp.best_end(recent)
        if end is not None:
            for f, box in self.dp.backtrack(end, stop_before=self.decided + 1).items():
                if f <= upto:
                    self.committed[f] = box
        self.decided = upto
        self.dp.forget_before(self.decided - self.dp.max_link - 1)

    def _finalise(self, upto):
        if upto <= self.finalised:
            return {}
        lo = self.finalised - self.WINDOW
        window = {f: b for f, b in self.committed.items() if f >= lo}
        clean = drop_outliers(drop_short_blob_runs(window, self.blobs), self.height)
        out = {f: b for f, b in clean.items() if self.finalised < f <= upto}
        self.final.update(out)
        self.finalised = upto
        keep_from = self.finalised - self.WINDOW
        self.committed = {f: b for f, b in self.committed.items() if f >= keep_from}
        self.blobs = {b for b in self.blobs if b[0] >= keep_from}
        return dict(sorted(out.items()))

    def flush(self):
        """End of the clip: decide and finalise every frame left."""
        self._decide(self.last)
        return self._finalise(self.last)

    def rows(self):
        """Ball rows like fill_gaps() over everything finalised so far."""
        return fill_gaps(self.final, self.fps)


def drop_outliers(track, frame_height=1080, window=3, passes=3):
    """Remove isolated detections that jump away and straight back.

    A detection f between neighbours a and b (each within `window` frames) is
    dropped when it is physically unreachable from both while a and b agree,
    or when it is a fast spike: faster than SPIKE_SPEED_PX on both sides with
    an out-and-back path more than SPIKE_DETOUR times the direct a -> b
    distance. A racket contact is also V-shaped, but far slower.
    """
    s = frame_height / 1080
    max_speed, spike_speed = MAX_SPEED_PX * s, SPIKE_SPEED_PX * s
    track = dict(track)
    for _ in range(passes):                   # dropping one may expose the next
        frames = sorted(track)
        centre = {f: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                  for f, b in track.items()}

        def dist(u, v, centre=centre):
            return float(np.hypot(*(centre[v] - centre[u])))

        drop = set()
        for a, f, b in zip(frames, frames[1:], frames[2:]):
            if f - a > window or b - f > window:
                continue
            v_af, v_fb, v_ab = dist(a, f) / (f - a), dist(f, b) / (b - f), dist(a, b) / (b - a)
            unreachable = v_af > max_speed and v_fb > max_speed and v_ab <= max_speed
            spike = (v_af > spike_speed and v_fb > spike_speed
                     and dist(a, f) + dist(f, b) > SPIKE_DETOUR * max(dist(a, b), 1.0))
            if unreachable or spike:
                drop.add(f)
        if not drop:
            break
        track = {f: b for f, b in track.items() if f not in drop}
    return track


def fill_gaps(track, fps):
    """Rows (frame, x1, y1, x2, y2, interpolated) with short gaps PCHIP-filled."""
    frames = np.array(sorted(track))
    if len(frames) == 0:
        return []
    boxes = np.array([track[f] for f in frames], float)
    rows = [(int(f), *boxes[i], 0) for i, f in enumerate(frames)]
    if len(frames) < 2:
        return rows
    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    cx = PchipInterpolator(frames, (boxes[:, 0] + boxes[:, 2]) / 2)
    cy = PchipInterpolator(frames, (boxes[:, 1] + boxes[:, 3]) / 2)
    for a, b, ba, bb in zip(frames[:-1], frames[1:], boxes[:-1], boxes[1:]):
        if not 0 < b - a - 1 <= max_gap:
            continue
        for f in range(a + 1, b):
            t = (f - a) / (b - a)
            w = ba[2] - ba[0] + t * ((bb[2] - bb[0]) - (ba[2] - ba[0]))
            h = ba[3] - ba[1] + t * ((bb[3] - bb[1]) - (ba[3] - ba[1]))
            x, y = float(cx(f)), float(cy(f))
            rows.append((f, x - w / 2, y - h / 2, x + w / 2, y + h / 2, 1))
    rows.sort()
    return rows


def wholeframe_track(cands, n_frames):
    """The original detector's track, rebuilt from the "frame" candidates.

    Per frame, the most confident whole-frame box with conf >= WHOLEFRAME_CONF
    that moved (no motion gate on the first frame, which has no previous one);
    every gap filled with a straight line, the clip edges with the nearest
    position. Returns rows like fill_gaps(), or None without "frame" candidates.
    """
    if "source" not in cands.columns or not (cands.source == "frame").any():
        return None
    first = int(cands.frame.min())
    c = cands[(cands.source == "frame") & (cands.conf >= WHOLEFRAME_CONF)
              & ((cands.motion >= MIN_MOTION) | (cands.frame == first))]
    best = c.loc[c.groupby("frame").conf.idxmax()].set_index("frame")
    df = pd.DataFrame(np.nan, index=range(n_frames), columns=["x1", "y1", "x2", "y2"])
    df.loc[best.index, ["x1", "y1", "x2", "y2"]] = best[["x1", "y1", "x2", "y2"]].to_numpy()
    real = df.x1.notna().to_numpy()
    df = df.interpolate(limit_direction="both").bfill().ffill()
    if df.x1.isna().all():
        return []
    return [(f, *df.loc[f].to_numpy(), int(not real[f])) for f in range(n_frames)]


def write_ball_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(BALL_COLUMNS)
        for frame, x1, y1, x2, y2, interp in rows:
            x, y, w, h = int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))
            wr.writerow([frame, x, y, w, h, round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
                         w * h, interp])


def track_from_candidates(candidates_csv, video_path, out_csv, wholeframe_csv=None,
                          players_csv=None, court_csv=None):
    """Stage 2: candidates CSV -> ball CSV (and the whole-frame track, if asked).

    With the clip's players and court, motion candidates join the detector's.
    """
    cap = cv2.VideoCapture(video_path)
    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    cands = pd.read_csv(candidates_csv)
    n_frames = max(n_frames, int(cands.frame.max()) + 1 if len(cands) else 0)

    merged, blobs = cands, set()
    if players_csv and court_csv and os.path.exists(players_csv) and os.path.exists(court_csv):
        extra = motion_candidates(video_path, players_csv, court_csv)
        print(f"  {len(extra)} motion candidates added to {len(cands)} from the detector")
        blobs = set(zip(extra.frame, extra.x1.astype(float), extra.y1.astype(float)))
        merged = pd.concat([cands, extra], ignore_index=True).sort_values("frame", kind="stable")

    track = drop_short_blob_runs(track_candidates(merged, n_frames, fps, height), blobs)
    clean = drop_outliers(track, height)
    rows = fill_gaps(clean, fps)
    write_ball_csv(rows, out_csv)
    summary = {"frames": n_frames, "detected": len(clean),
               "outliers_dropped": len(track) - len(clean),
               "interpolated": sum(r[5] for r in rows)}
    print(f"Ball found on {summary['detected']} of {n_frames} frames "
          f"({summary['detected'] / max(n_frames, 1):.0%}), "
          f"{summary['outliers_dropped']} outliers dropped, "
          f"{summary['interpolated']} gap frames filled -> {out_csv}")
    if wholeframe_csv:
        legacy = wholeframe_track(cands, n_frames)
        if legacy is None:
            print("  (candidates have no whole-frame source column: whole-frame track "
                  "not written; re-run detection to get it)")
        else:
            write_ball_csv(legacy, wholeframe_csv)
            print(f"  whole-frame track for shot analysis -> {wholeframe_csv}")
    return summary


def render_video(video_path, ball_csv, out_path):
    """Write a copy of the video with the ball box drawn (real = green, filled = yellow)."""
    balls = pd.read_csv(ball_csv).set_index("frame")
    cap = cv2.VideoCapture(video_path)
    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    writer, idx = None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in balls.index:
                r = balls.loc[idx]
                color = (0, 255, 255) if r.interpolated else (0, 255, 0)
                cv2.rectangle(frame, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), color, 2)
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print(f"Saved annotated video to {out_path}")


def default_paths(video, output_dir=OUTPUT_ROOT):
    paths = clip_paths(video, output_dir)
    return paths["ball_candidates_csv"], paths["ball_csv"], paths["ball_wholeframe_csv"]


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "ball_tracker.pt")

    parser = argparse.ArgumentParser(
        description="Ball tracking: YOLO candidates on full-resolution tiles, then the "
                    "best path through them, outlier removal and gap filling")
    parser.add_argument("--video", default="data/Input_video2.mp4", help="input video")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="YOLO ball-detector weights (.pt), needed by stage 1")
    parser.add_argument("--detect-only", action="store_true", dest="detect_only",
                        help="run stage 1 only and write the candidates CSV")
    parser.add_argument("--candidates", default=None,
                        help="existing candidates CSV: run stage 2 only (no model needed)")
    parser.add_argument("--csv", default=None,
                        help="output ball CSV (default: outputs/<video name>/ball_coordinates/ball_<video>.csv)")
    parser.add_argument("--video-out", default=None, dest="video_out",
                        help="also write an annotated copy of the video here")
    args = parser.parse_args()

    cand_csv, ball_csv, wholeframe_csv = default_paths(args.video)
    clip = clip_paths(args.video)
    if args.csv:
        ball_csv = args.csv
        wholeframe_csv = os.path.join(os.path.dirname(args.csv) or ".",
                                      "wholeframe_" + os.path.basename(args.csv))
    if args.candidates:
        cand_csv = args.candidates
    else:
        if not os.path.exists(args.model):
            raise SystemExit(f"Model not found: {args.model}\n"
                             "Place the YOLO weights there, pass --model, or run stage 2 "
                             "only with --candidates <csv>.")
        BallDetector(args.model).detect_candidates(args.video, cand_csv)
    if not args.detect_only:
        track_from_candidates(cand_csv, args.video, ball_csv, wholeframe_csv,
                              clip["players_csv"], clip["court_csv"])
        if args.video_out:
            render_video(args.video, ball_csv, args.video_out)
