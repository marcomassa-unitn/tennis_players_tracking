import argparse
import csv
import os

import cv2
import numpy as np



class PlayerTracker:
    def __init__(self, video_path, csv_path, display=True):
        self.display = display
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")


        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")
        self.H, self.W = frame.shape[:2]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._init_distance_params()

        self.ALPHA = 0.15 

             
        self.warmup_seconds = 1.0   

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.WARMUP_FRAMES = int(self.warmup_seconds * fps)
        self.THRESH = int(0.5 *fps)

        self.KERNEL_SMALL = np.ones((3, 3), np.uint8)
        self.KERNEL_BIG = np.ones((5, 5), np.uint8)

        self.prev_centroids = None           

        self.background = None
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

    def _get_foreground_mask(self, gray):
        if self.background is None:
            return None

        bg_uint8 = cv2.convertScaleAbs(self.background)
        diff = cv2.absdiff(gray, bg_uint8)

        _, fg = cv2.threshold(diff, self.THRESH, 255, cv2.THRESH_BINARY)
        fg = cv2.GaussianBlur(fg, (5, 5), 0)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.KERNEL_SMALL)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.KERNEL_BIG)

        return fg 


    def _find_components(self, mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        candidates = []
        for i in range(1, num_labels):  
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            candidates.append([area, x, y, w, h, cx, cy])

        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates 

    def _select_players(self, candidates):
        if not candidates:
            return []

        if self.prev_centroids is None:
            comps = candidates[:2]
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

            if len(comps) < 2:
                for idx, cand in enumerate(candidates):
                    if idx in used:
                        continue
                    comps.append(cand)
                    if len(comps) == 2:
                        break

        if len(comps) == 2:
            _, _, _, _, _, cx1, cy1 = comps[0]
            _, _, _, _, _, cx2, cy2 = comps[1]
            dist = np.hypot(cx2 - cx1, cy2 - cy1)
            if dist < self.MIN_DIST:
                comps = [comps[0]]

        return comps
    
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
        return new_prev
    
    def run(self):
        frame_idx = 0
        self._open_csv()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._update_background(gray)

            mask = self._get_foreground_mask(gray)

            components = []
            if mask is not None and frame_idx >= self.WARMUP_FRAMES:
                candidates = self._find_components(mask)
                components = self._select_players(candidates)

            if components:
                self._draw_players(frame, components)
                self._write_csv_rows(frame_idx, components)

            if self.display:
                vis = cv2.resize(frame, (960, 540))
                cv2.imshow("Player Tracker", vis)
                if mask is not None:
                    cv2.imshow("FG mask", cv2.resize(mask, (480, 270)))

                key = cv2.waitKey(self.frame_ms) & 0xFF
                if key == ord("q"):
                    break

            frame_idx += 1

        self.cap.release()
        self._close_csv()
        cv2.destroyAllWindows()
        print(f"Tracking saved to {self.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-player tracking via running-average background "
                    "subtraction + nearest-centroid association")
    parser.add_argument("--video", default="data/Input_video2.mp4")
    parser.add_argument("--csv",   default="outputs/players_clip2.csv",
                        help="output CSV path")
    parser.add_argument("--no-display", action="store_true",
                        help="run headless (no OpenCV windows)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    tracker = PlayerTracker(args.video, args.csv, display=not args.no_display)
    tracker.run()

