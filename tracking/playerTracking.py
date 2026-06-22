import argparse
import csv
import os

import cv2
import numpy as np

# Dual import so safe_fps resolves both standalone (tracking/ on sys.path) and
# orchestrated as tracking.playerTracking (project root on sys.path via pipeline.py).
try:
    from tracking._fps_utils import safe_fps
except ImportError:
    from _fps_utils import safe_fps


# Foreground intensity-difference threshold (0-255), fps-independent. 15
# reproduces the old 30fps behaviour (0.5 * 30).
DIFF_THRESH = 15

# Min connected-component area (px) to count as a player candidate; rejects noise.
MIN_COMPONENT_AREA = 300


class PlayerTracker:
    def __init__(self, video_path, csv_path, display=True):
        self.display = display
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        ret, frame = self.cap.read()
        if not ret:
            # Release before raising so the capture isn't leaked on failure.
            self.cap.release()
            raise RuntimeError("Cannot read first frame")
        self.H, self.W = frame.shape[:2]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._init_distance_params()

        self.ALPHA = 0.15

        self.warmup_seconds = 1.0

        # safe_fps rejects missing / non-finite / non-positive values.
        fps = safe_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self.WARMUP_FRAMES = int(self.warmup_seconds * fps)
        self.THRESH = DIFF_THRESH

        self.KERNEL_SMALL = np.ones((3, 3), np.uint8)
        self.KERNEL_BIG = np.ones((5, 5), np.uint8)

        # Per-method tracking state so the warmup static-median tracker can never
        # corrupt the reliable running-average ("bg update") one.
        #   prev_warmup : identities during the warmup window
        #   prev_bg     : identities under the running-average background
        # prev_centroids is just the ACTIVE slot for the current frame.
        self.prev_centroids = None
        self.prev_warmup = None
        self.prev_bg = None

        self.background = None
        # Player-free reference (temporal median of the first WARMUP_FRAMES);
        # built by the run() pre-pass. None until then.
        self.static_bg = None
        # waitKey() playback delay (ms) between displayed frames.
        self.frame_ms = 5

        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None

    def _open_csv(self):
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["frame", "player_id", "x", "y", "w", "h", "cx", "cy", "area"]
        )

    def _write_csv_rows(self, frame_idx, components):
        if self.csv_writer is None:
            return
        for player_id, (area, x, y, w, h, cx, cy) in enumerate(components, start=1):
            self.csv_writer.writerow(
                [frame_idx, player_id, x, y, w, h, cx, cy, area]
            )

    def _close_csv(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def _init_distance_params(self):
        diag = np.hypot(self.W, self.H)

        max_move_ratio = 0.05
        min_dist_ratio = 0.15

        self.MAX_MOVE = max_move_ratio * diag
        self.MIN_DIST = min_dist_ratio * self.W

    def _update_background(self, gray):
        if self.background is None:
            self.background = gray.astype("float")
        else:
            cv2.accumulateWeighted(gray, self.background, self.ALPHA)

    def _foreground_from_bg(self, gray, bg_uint8):
        """Binary foreground mask from |gray - bg|, thresholded and morphologically
        cleaned. Shared by the running-average and static (warmup) backgrounds."""
        diff = cv2.absdiff(gray, bg_uint8)

        _, fg = cv2.threshold(diff, self.THRESH, 255, cv2.THRESH_BINARY)
        fg = cv2.GaussianBlur(fg, (5, 5), 0)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.KERNEL_SMALL)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.KERNEL_BIG)

        return fg

    def _get_foreground_mask(self, gray):
        if self.background is None:
            return None

        bg_uint8 = cv2.convertScaleAbs(self.background)
        return self._foreground_from_bg(gray, bg_uint8)

    def _compute_static_background(self):
        """Build static_bg from the temporal median of the first WARMUP_FRAMES, then rewind.

        Lets the warmup window detect against a player-free court (median averages
        moving players out) instead of skipping detection while the running-average
        model converges. That model warms up in parallel and takes over unchanged at
        frame >= WARMUP_FRAMES.
        """
        frames = []
        for _ in range(max(self.WARMUP_FRAMES, 1)):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Rewind so the main loop starts from frame 0.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if frames:
            self.static_bg = np.median(
                np.stack(frames, axis=0), axis=0
            ).astype("uint8")


    def _find_components(self, mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        candidates = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            # Drop sub-threshold noise blobs.
            if area < MIN_COMPONENT_AREA:
                continue
            cx, cy = centroids[i]
            candidates.append([area, x, y, w, h, cx, cy])

        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates

    def _select_players(self, candidates):
        if not candidates:
            return []

        merge_allowed = True
        if self.prev_centroids is None:
            comps = candidates[:2]
        elif len(self.prev_centroids) == 2:
            comps = self._associate_two(candidates)
            # Comps are already bound to distinct identities; the merge rule would
            # delete the far player when centroids fall within MIN_DIST (this is
            # how P2 was being lost), so disable it here.
            merge_allowed = False
        else:
            comps = []
            used = set()

            for old_cx, old_cy in self.prev_centroids:
                best_idx = None
                best_dist = None
                for idx, cand in enumerate(candidates):
                    if idx in used:
                        continue
                    area, x, y, w, h, cx, cy = cand
                    dist = np.hypot(cx - old_cx, cy - old_cy)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                if best_idx is not None and best_dist <= self.MAX_MOVE:
                    comps.append(candidates[best_idx])
                    used.add(best_idx)

            # Borrow an unused candidate to reach two players, but only one within
            # MAX_MOVE of a previous centroid -- never teleport in a far blob.
            if len(comps) < 2 and self.prev_centroids:
                for idx, cand in enumerate(candidates):
                    if idx in used:
                        continue
                    _, _, _, _, _, cx, cy = cand
                    near = any(
                        np.hypot(cx - ocx, cy - ocy) <= self.MAX_MOVE
                        for ocx, ocy in self.prev_centroids
                    )
                    if not near:
                        continue
                    comps.append(cand)
                    used.add(idx)
                    if len(comps) == 2:
                        break

        # Collapse detections nearer than MIN_DIST to a single box; for >2,
        # keep the first and drop any that crowd it.
        if merge_allowed and len(comps) >= 2:
            _, _, _, _, _, cx1, cy1 = comps[0]
            merged = [comps[0]]
            for cand in comps[1:]:
                _, _, _, _, _, cx2, cy2 = cand
                if np.hypot(cx2 - cx1, cy2 - cy1) >= self.MIN_DIST:
                    merged.append(cand)
            comps = merged

        return comps

    def _associate_two(self, candidates):
        """Match the two prior identities to detections via the global
        minimum-total-displacement 2x2 assignment (respecting MAX_MOVE).

        Returns up to two components ordered to match self.prev_centroids so
        identities survive a crossing. Logs identity loss / re-acquisition.
        """
        (p0x, p0y), (p1x, p1y) = self.prev_centroids

        def d(px, py, cand):
            _, _, _, _, _, cx, cy = cand
            return np.hypot(cx - px, cy - py)

        # Allow ANY candidate (not just the two largest): the far player is small
        # and would be excluded if the near player's blob fragments into the two
        # biggest. Global min-total-displacement over distinct candidates keeps
        # identities stable across a crossing.
        comps = [None, None]
        if len(candidates) >= 2:
            best_cost = None
            best_pair = None
            # Ordered pairs (i != j): i -> identity 0, j -> identity 1.
            for i, ci in enumerate(candidates):
                di0 = d(p0x, p0y, ci)
                for j, cj in enumerate(candidates):
                    if i == j:
                        continue
                    cost = di0 + d(p1x, p1y, cj)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_pair = (ci, cj)
            ci, cj = best_pair
            if d(p0x, p0y, ci) <= self.MAX_MOVE:
                comps[0] = ci
            if d(p1x, p1y, cj) <= self.MAX_MOVE:
                comps[1] = cj
        else:
            # Single candidate: give it to the nearer identity (within MAX_MOVE);
            # the other stays unmatched.
            only = candidates[0]
            d0, d1 = d(p0x, p0y, only), d(p1x, p1y, only)
            slot = 0 if d0 <= d1 else 1
            if min(d0, d1) <= self.MAX_MOVE:
                comps[slot] = only

        # Log per-slot loss / re-acquisition (player_id == slot + 1).
        for slot in range(2):
            had = slot < len(self.prev_centroids)
            has = comps[slot] is not None
            if had and not has:
                print(f"[playerTracking] Identity lost for player {slot + 1} "
                      f"(no candidate within MAX_MOVE)")
            elif not had and has:
                print(f"[playerTracking] Identity re-acquired for player {slot + 1}")

        # Drop unmatched slots; order preserved to keep player_id stable.
        return [c for c in comps if c is not None]

    def _draw_players(self, frame, components):
        new_prev = []
        for player_id, (area, x, y, w, h, cx, cy) in enumerate(components, start=1):
            color = (0, 255, 0) if player_id == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)
            cv2.putText(frame, f"P{player_id}", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            new_prev.append((cx, cy))
        self.prev_centroids = new_prev

    def run(self):
        frame_idx = 0
        # Reads and rewinds the first WARMUP_FRAMES to build static_bg.
        self._compute_static_background()
        self._open_csv()

        # finally guarantees capture release + CSV close even on mid-loop error.
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self._update_background(gray)

                # The two foreground methods keep separate tracking state. Warmup
                # (static median bg) handles the first WARMUP_FRAMES; the
                # running-average method handles the rest and re-seeds from scratch
                # on takeover (prev_bg starts None), so a bad warmup seed can't make
                # it lose a player.
                warmup = frame_idx < self.WARMUP_FRAMES
                if warmup:
                    mask = (self._foreground_from_bg(gray, self.static_bg)
                            if self.static_bg is not None else None)
                    self.prev_centroids = self.prev_warmup
                else:
                    mask = self._get_foreground_mask(gray)
                    self.prev_centroids = self.prev_bg

                components = []
                if mask is not None:
                    candidates = self._find_components(mask)
                    components = self._select_players(candidates)

                if components:
                    self._draw_players(frame, components)
                    self._write_csv_rows(frame_idx, components)

                # Persist active state into its own slot so the trackers never
                # share identities.
                if warmup:
                    self.prev_warmup = self.prev_centroids
                else:
                    self.prev_bg = self.prev_centroids

                if self.display:
                    vis = cv2.resize(frame, (960, 540))
                    cv2.imshow("Player Tracker", vis)
                    if mask is not None:
                        cv2.imshow("FG mask", cv2.resize(mask, (480, 270)))

                    key = cv2.waitKey(self.frame_ms) & 0xFF
                    if key == ord("q"):
                        break

                frame_idx += 1
        finally:
            self.cap.release()
            self._close_csv()
            cv2.destroyAllWindows()

        print(f"Tracking saved to {self.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-player tracking via running-average background "
                    "subtraction + nearest-centroid association")
    parser.add_argument("--video", default="data/Input_video2.mp4")
    parser.add_argument("--csv",   default=None,
                        help="output CSV path (defaults to "
                             "outputs/player_coordinates/players_<video name>.csv)")
    parser.add_argument("--no-display", action="store_true",
                        help="run headless (no OpenCV windows)")
    args = parser.parse_args()

    # Default the CSV name to the video stem so different --video inputs don't
    # overwrite each other.
    if args.csv is None:
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        args.csv = os.path.join("outputs", "player_coordinates",
                                f"players_{video_stem}.csv")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    tracker = PlayerTracker(args.video, args.csv, display=not args.no_display)
    tracker.run()

