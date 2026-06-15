import argparse
import csv
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def _has_motion(self, prev_gray, curr_gray, x1, y1, x2, y2, threshold=0.05):
        """Returns True if at least `threshold` fraction of pixels in the box moved."""
        motion = cv2.absdiff(prev_gray[y1:y2, x1:x2], curr_gray[y1:y2, x1:x2])
        _, binary = cv2.threshold(motion, 15, 255, cv2.THRESH_BINARY)
        total = binary.size
        if total == 0:
            return False
        return (binary.sum() / 255) / total >= threshold

    def detect_frame(self, frame, prev_gray=None):
        """Returns {1: [x1,y1,x2,y2]} for the highest-confidence detection with motion, or {}."""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.model.predict(frame, conf=0.65, verbose=False)[0]
        for idx in results.boxes.conf.argsort(descending=True):
            x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx])
            if prev_gray is not None and not self._has_motion(prev_gray, curr_gray, x1, y1, x2, y2):
                continue
            return {1: [x1, y1, x2, y2]}, curr_gray
        return {}, curr_gray

    def detect_frames(self, frames):
        """Returns list of per-frame detection dicts."""
        ball_positions = []
        prev_gray = None
        for frame in frames:
            detection, prev_gray = self.detect_frame(frame, prev_gray)
            ball_positions.append(detection)
        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """Fills gaps with linear interpolation + backward fill."""
        positions_list = [p.get(1, []) for p in ball_positions]
        df = pd.DataFrame(positions_list, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate()
        df = df.bfill()
        return [{1: row} for row in df.to_numpy().tolist()]

    def draw_bboxes(self, frames, ball_positions):
        """Draws ball bounding boxes on copies of the input frames."""
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

    def _write_csv(self, csv_path, ball_positions):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "x", "y", "w", "h", "cx", "cy", "area"])
            for frame_idx, pos in enumerate(ball_positions):
                bbox = pos.get(1)
                if bbox is None or np.any(np.isnan(bbox)):
                    continue
                x1, y1, x2, y2 = bbox
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = w * h
                writer.writerow([frame_idx, x, y, w, h, cx, cy, area])

    def run(self, video_path, output_path=None, csv_path=None, display=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video:", video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pass 1: detect the ball frame by frame, keeping only the tiny
        # per-frame positions. Frames are released immediately (never held
        # all at once) so memory stays bounded even on long 1080p clips.
        print(f"Detecting ball in {total or '?'} frames...")
        ball_positions = []
        prev_gray = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detection, prev_gray = self.detect_frame(frame, prev_gray)
            ball_positions.append(detection)
        cap.release()

        ball_positions = self.interpolate_ball_positions(ball_positions)

        if csv_path:
            self._write_csv(csv_path, ball_positions)
            print(f"CSV saved to {csv_path}")

        if not output_path and not display:
            return

        # Pass 2: re-read the video and draw the (interpolated) boxes one frame
        # at a time, writing/showing on the fly instead of buffering all frames.
        cap = cv2.VideoCapture(video_path)
        writer = None
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
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

        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved to {output_path}")
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "ball_tracker.pt")

    parser = argparse.ArgumentParser(
        description="YOLO ball tracking -> annotated video + ball CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="path to the YOLO ball-tracker weights (.pt)")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="input video")
    parser.add_argument("--csv", default="outputs/ball_clip2.csv",
                        help="output ball CSV path")
    parser.add_argument("--output", default="outputs/ball_tracking_output2.mp4",
                        help="annotated output video path")
    parser.add_argument("--no-video", action="store_true", dest="no_video",
                        help="skip writing the annotated output video (CSV only)")
    parser.add_argument("--no-display", action="store_true",
                        help="run headless (no OpenCV window)")
    args = parser.parse_args()

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
