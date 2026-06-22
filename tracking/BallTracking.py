import argparse
import csv
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Dual import: works standalone (tracking/ on sys.path) and orchestrated
# (imported as tracking.BallTracking via pipeline.py with project root on path).
try:
    from tracking._fps_utils import safe_fps
except ImportError:
    from _fps_utils import safe_fps


# Min YOLO confidence to accept a ball detection.
BALL_CONF = 0.65
# Per-pixel intensity-diff threshold (0-255) marking a pixel as moved.
MOTION_THRESH = 15


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def _has_motion(self, prev_gray, curr_gray, x1, y1, x2, y2, threshold=0.05):
        """True if at least `threshold` fraction of box pixels changed between frames."""
        H, W = curr_gray.shape[:2]
        # Clamp to frame bounds and check ordering so a bad/out-of-range box
        # never reaches cv2.absdiff().
        x1 = max(0, min(int(x1), W))
        x2 = max(0, min(int(x2), W))
        y1 = max(0, min(int(y1), H))
        y2 = max(0, min(int(y2), H))
        if x2 <= x1 or y2 <= y1:
            return False
        prev_slice = prev_gray[y1:y2, x1:x2]
        curr_slice = curr_gray[y1:y2, x1:x2]
        if prev_slice.size == 0 or curr_slice.size == 0:
            return False
        motion = cv2.absdiff(prev_slice, curr_slice)
        _, binary = cv2.threshold(motion, MOTION_THRESH, 255, cv2.THRESH_BINARY)
        total = binary.size
        if total == 0:
            return False
        return (binary.sum() / 255) / total >= threshold

    def detect_frame(self, frame, prev_gray=None):
        """Highest-confidence detection passing the motion gate as {1: [x1,y1,x2,y2]}, else {}."""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.model.predict(frame, conf=BALL_CONF, verbose=False)[0]
        for idx in results.boxes.conf.argsort(descending=True):
            x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx])
            if prev_gray is not None and not self._has_motion(prev_gray, curr_gray, x1, y1, x2, y2):
                continue
            return {1: [x1, y1, x2, y2]}, curr_gray
        return {}, curr_gray

    def detect_frames(self, frames):
        """Per-frame detection dicts for in-memory frame lists.

        Not used by run()/__main__, which stream frames one at a time.
        """
        ball_positions = []
        prev_gray = None
        for frame in frames:
            detection, prev_gray = self.detect_frame(frame, prev_gray)
            ball_positions.append(detection)
        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """Linearly fill detection gaps, extending past both ends.

        limit_direction="both" also propagates nearest values outward so
        leading/trailing gaps aren't left NaN and dropped downstream; bfill/ffill
        cover all-NaN edge rows.

        Returns (filled_positions, real_mask); real_mask[i] is True only where
        frame i held a genuine 4-value YOLO bbox before filling. Shot analysis
        needs this: a straight-line fill meeting real motion mimics a racket
        contact and would trigger a false shot.
        """
        real_mask = [len(p.get(1, [])) == 4 for p in ball_positions]
        positions_list = [p.get(1, []) for p in ball_positions]
        df = pd.DataFrame(positions_list, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate(limit_direction="both")
        df = df.bfill().ffill()
        return [{1: row} for row in df.to_numpy().tolist()], real_mask

    def draw_bboxes(self, frames, ball_positions):
        """Draw ball boxes on copies of the frames (originals untouched)."""
        output = []
        for frame, pos in zip(frames, ball_positions):
            out = frame.copy()
            bbox = pos.get(1)
            if bbox and not np.any(np.isnan(bbox)):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, "Ball", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            output.append(out)
        return output

    def _write_csv(self, csv_path, ball_positions, real_mask=None):
        # `interpolated` is the LAST column (0=real, 1=fill) so readers indexing
        # the first 8 columns stay unaffected. real_mask=None => all marked real.
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "x", "y", "w", "h", "cx", "cy", "area",
                             "interpolated"])
            for frame_idx, pos in enumerate(ball_positions):
                bbox = pos.get(1)
                if bbox is None or np.any(np.isnan(bbox)):
                    continue
                x1, y1, x2, y2 = bbox
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = w * h
                is_real = real_mask[frame_idx] if real_mask is not None else True
                writer.writerow([frame_idx, x, y, w, h, cx, cy, area,
                                 0 if is_real else 1])

    def run(self, video_path, output_path=None, csv_path=None, display=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video:", video_path)
            return

        # Rejects missing / non-finite / non-positive fps.
        fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pass 1: detect frame by frame, keeping only the tiny position dicts.
        # Frames are dropped immediately so memory stays bounded on long clips.
        print(f"Detecting ball in {total or '?'} frames...")
        ball_positions = []
        prev_gray = None
        # Release the capture even if detection raises.
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detection, prev_gray = self.detect_frame(frame, prev_gray)
                ball_positions.append(detection)
        finally:
            cap.release()

        ball_positions, real_mask = self.interpolate_ball_positions(ball_positions)

        if csv_path:
            self._write_csv(csv_path, ball_positions, real_mask)
            print(f"CSV saved to {csv_path}")

        if not output_path and not display:
            return

        # Pass 2: re-read the video and draw the (interpolated) boxes one frame
        # at a time, writing/showing on the fly instead of buffering all frames.
        cap = cv2.VideoCapture(video_path)
        writer = None
        idx = 0
        # Release capture AND writer even on error, so the .mp4 is finalized.
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Assumes both passes read the same frames in order (true for
                # CFR). The guard avoids IndexError; VFR input could desync.
                pos = ball_positions[idx] if idx < len(ball_positions) else {}
                out = self.draw_bboxes([frame], [pos])[0]

                if output_path:
                    if writer is None:
                        h, w = out.shape[:2]
                        writer = cv2.VideoWriter(
                            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    writer.write(out)

                if display:
                    cv2.imshow("Ball Tracker", out)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break

                idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
                print(f"Saved to {output_path}")
            if display:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models/ball_tracker.pt")

    parser = argparse.ArgumentParser(
        description="YOLO ball tracking -> annotated video + ball CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="path to the YOLO ball-tracker weights (.pt)")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="input video")
    parser.add_argument("--csv", default=None,
                        help="output ball CSV path (defaults to "
                             "outputs/ball_coordinates/ball_<video name>.csv)")
    parser.add_argument("--output", default="outputs/ball_tracking_output2.mp4",
                        help="annotated output video path")
    parser.add_argument("--no-video", action="store_true", dest="no_video",
                        help="skip writing the annotated output video (CSV only)")
    parser.add_argument("--no-display", action="store_true",
                        help="run headless (no OpenCV window)")
    args = parser.parse_args()

    # Default CSV name from the video stem so distinct --video inputs don't
    # overwrite each other (mirrors playerTracking.py).
    if args.csv is None:
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        args.csv = os.path.join("outputs", "ball_coordinates",
                                f"ball_{video_stem}.csv")

    output_path = None if args.no_video else args.output

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Model not found: {args.model}\n"
            f"Place the YOLO ball-tracker weights there or pass --model <path>.")

    for path in (args.csv, output_path):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    tracker = BallTracker(args.model)
    tracker.run(args.video, output_path=output_path, csv_path=args.csv,
                display=not args.no_display)
