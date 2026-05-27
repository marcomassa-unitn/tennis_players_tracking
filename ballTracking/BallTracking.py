import csv
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

    def run(self, video_path, output_path=None, csv_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video:", video_path)
            return

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        print(f"Detecting ball in {len(frames)} frames...")
        ball_positions = self.detect_frames(frames)
        ball_positions = self.interpolate_ball_positions(ball_positions)

        if csv_path:
            self._write_csv(csv_path, ball_positions)
            print(f"CSV saved to {csv_path}")

        annotated = self.draw_bboxes(frames, ball_positions)

        if output_path:
            h, w = annotated[0].shape[:2]
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
            )
            for f in annotated:
                writer.write(f)
            writer.release()
            print(f"Saved to {output_path}")

        for frame in annotated:
            cv2.imshow("Ball Tracker", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = BallTracker("models/ball_tracker.pt")
    tracker.run("data/input_video.mp4", output_path="outputs/ball_tracking_output1.mp4", csv_path="outputs/ball_clip1.csv")
