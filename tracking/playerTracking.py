import argparse
import csv
import os

import cv2
import numpy as np

# Shared fps guard. Dual import so it resolves both standalone
# (`python tracking/playerTracking.py`, tracking/ on sys.path) and orchestrated
# (imported as tracking.playerTracking, project root on sys.path via pipeline.py).
try:
    from tracking._fps_utils import safe_fps
except ImportError:
    from _fps_utils import safe_fps


# Foreground intensity-difference threshold (0-255). Decoupled from fps: an
# intensity threshold has nothing to do with frame rate. 15 preserves the prior
# 30fps behaviour (0.5 * 30 == 15) used in cv2.threshold().
DIFF_THRESH = 15

# Minimum connected-component area (in pixels) for a blob to be considered a
# player candidate. Filters out tiny noise components below this size.
MIN_COMPONENT_AREA = 300


class PlayerTracker:
    def __init__(self, video_path, csv_path, display=True):
        self.display = display
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        ret, frame = self.cap.read()
        if not ret:
            # Release the capture before raising so it isn't leaked when
            # construction fails on an unreadable first frame.
            self.cap.release()
            raise RuntimeError("Cannot read first frame")
        self.H, self.W = frame.shape[:2]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._init_distance_params()

        self.ALPHA = 0.15

        self.warmup_seconds = 1.0

        # Robust fps guard: reject missing / non-finite / non-positive values.
        fps = safe_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self.WARMUP_FRAMES = int(self.warmup_seconds * fps)
        # Intensity difference threshold no longer depends on fps (see DIFF_THRESH).
        self.THRESH = DIFF_THRESH

        self.KERNEL_SMALL = np.ones((3, 3), np.uint8)
        self.KERNEL_BIG = np.ones((5, 5), np.uint8)

        # Tracking state is kept SEPARATE per foreground method so the warmup
        # "plain" frame-diff (static median bg) can never corrupt the
        # running-average ("bg update") tracker, which is the reliable one.
        #   prev_warmup : identities tracked during the warmup window
        #   prev_bg     : identities tracked by the running-average background
        # self.prev_centroids is just the ACTIVE slot for the current frame.
        self.prev_centroids = None
        self.prev_warmup = None
        self.prev_bg = None

        self.background = None
        # Static, player-free reference background (temporal median of the first
        # WARMUP_FRAMES frames). Used for plain background subtraction during the
        # warmup window while the running-average model converges; None until the
        # pre-pass in run() builds it.
        self.static_bg = None
        # Playback delay (ms) passed to cv2.waitKey() between displayed frames.
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
        """Threshold + clean the absolute difference of `gray` against a uint8
        background image, returning the binary foreground mask. Shared by both
        the running-average (update) and the static (warmup) backgrounds."""
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
        """Pre-pass: build a static, player-free reference background from the
        temporal median of the first WARMUP_FRAMES frames, then rewind.

        The running-average background needs ~WARMUP_FRAMES to converge, so the
        original tracker simply skipped detection during that window. Instead we
        detect those early frames against this static median background (classic
        "plain" background subtraction): moving players are averaged out by the
        median, leaving the empty court. The running-average model is still
        warmed up in parallel and takes over unchanged once the warmup window
        ends, so frame >= WARMUP_FRAMES behaviour is preserved exactly.
        """
        frames = []
        for _ in range(max(self.WARMUP_FRAMES, 1)):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Rewind so the main loop processes the video from the very first frame.
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
            # Gate out tiny noise blobs below the minimum area threshold.
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
            # Stable two-player association: pick the 2x2 assignment with the
            # minimum total displacement so identities don't swap when the
            # players cross (greedy-nearest would otherwise flip them).
            comps = self._associate_two(candidates)
            # Each comp is already bound to a distinct tracked identity, so the
            # "too close -> merge" rule must NOT fire here: it would delete the
            # smaller (far) player whenever the two centroids are within
            # MIN_DIST, which is exactly how P2 was being lost.
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

            # Fallback to reach two players: only borrow an unused candidate
            # that is itself within MAX_MOVE of one of the previous centroids.
            # Never teleport in an arbitrary far blob.
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

        # "Two players too close -> merge": collapse to one box when two
        # detections are nearer than MIN_DIST. Handle the >2 case defensively
        # too (keep the first, drop any others that crowd it).
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
        """Associate exactly two previous identities to current detections by
        the minimum-total-displacement 2x2 assignment (respecting MAX_MOVE).

        Returns up to two components ordered to match self.prev_centroids, so
        comps[0] stays player 1 and comps[1] stays player 2 across crossings.
        Logs when an identity is lost / re-acquired.
        """
        (p0x, p0y), (p1x, p1y) = self.prev_centroids

        def d(px, py, cand):
            _, _, _, _, _, cx, cy = cand
            return np.hypot(cx - px, cy - py)

        # Each identity may match ANY candidate, not just the two largest by
        # area: the far player is small and would otherwise be excluded whenever
        # the near player's foreground fragments into the two biggest blobs.
        # Pick the global minimum-total-displacement assignment of the two
        # previous identities to two distinct candidates so identities don't
        # swap when the players cross.
        comps = [None, None]
        if len(candidates) >= 2:
            best_cost = None
            best_pair = None
            # Search all ordered pairs (i != j): i -> identity 0, j -> identity 1.
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
            # Single candidate: assign it to whichever identity it is closest to
            # (within MAX_MOVE); the other identity is left unmatched.
            only = candidates[0]
            d0, d1 = d(p0x, p0y, only), d(p1x, p1y, only)
            slot = 0 if d0 <= d1 else 1
            if min(d0, d1) <= self.MAX_MOVE:
                comps[slot] = only

        # Log identity loss / re-acquisition per slot (player_id == slot + 1).
        for slot in range(2):
            had = slot < len(self.prev_centroids)
            has = comps[slot] is not None
            if had and not has:
                print(f"[playerTracking] Identity lost for player {slot + 1} "
                      f"(no candidate within MAX_MOVE)")
            elif not had and has:
                print(f"[playerTracking] Identity re-acquired for player {slot + 1}")

        # Drop unmatched slots, preserving order so player_id mapping is stable.
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
        # Build the static warmup background before the main loop (this reads and
        # rewinds the first WARMUP_FRAMES frames).
        self._compute_static_background()
        self._open_csv()

        # try/finally guarantees the capture is released and the CSV is closed
        # even if an exception is raised mid-loop.
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self._update_background(gray)

                # The two foreground methods run on COMPLETELY SEPARATE tracking
                # state. The warmup "plain" frame-diff (static median bg) handles
                # only the first WARMUP_FRAMES; the running-average ("bg update")
                # method handles the rest and RE-SEEDS itself from scratch when it
                # takes over (prev_bg starts None, so both players are picked from
                # its own first mask). A bad warmup seed can therefore never make
                # the bg-update tracker lose a player.
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

                # Persist the active state back into its own slot so the warmup
                # and bg-update trackers never share identities.
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

    # Derive the CSV name from the input video when not given explicitly, so a
    # different --video produces a distinct output instead of overwriting the
    # previous run's file.
    if args.csv is None:
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        args.csv = os.path.join("outputs", "player_coordinates",
                                f"players_{video_stem}.csv")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    tracker = PlayerTracker(args.video, args.csv, display=not args.no_display)
    tracker.run()

