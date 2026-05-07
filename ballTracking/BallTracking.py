import cv2
import numpy as np


# La pallina viene rilevata calcolando la differenza tra frame consecutivi in scala di grigi
# e filtrando i blob in movimento che ricadono nel range cromatico giallo-verde della pallina.

class BallTracker:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.lower_green = np.array([15, 15, 60])
        self.upper_green = np.array([95, 255, 255])

    def find_candidates(self, frame, gray, prev_gray):
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, motion = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        combined = cv2.bitwise_and(motion, green_mask)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 5 or area > 150:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / h if h > 0 else 0
            if not (0.5 < ratio < 2.0):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < 0.6:
                continue
            cx, cy = x + w // 2, y + h // 2
            candidates.append((area, cx, cy, x, y, w, h))
        return candidates

    def detect(self, frame, gray, prev_gray):
        if prev_gray is None:
            return None

        candidates = self.find_candidates(frame, gray, prev_gray)

        if not candidates:
            return None

        best = min(candidates, key=lambda c: c[0])
        _, _, _, x, y, w, h = best
        return (x, y, w, h)

    def run(self):
        if not self.cap.isOpened():
            print("Cannot open video")
            return
        frame_ms = 30
        prev_gray = None
        while True:
            t0 = cv2.getTickCount()
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox = self.detect(frame, gray, prev_gray)

            if bbox is not None:
                x, y, w, h = bbox
                cx, cy = x + w // 2, y + h // 2
                half = max(w, h) // 2 + 4
                cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), 2)

            cv2.imshow("Ball Tracker", frame)
            prev_gray = gray
            elapsed_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
            delay = max(1, int(frame_ms - elapsed_ms))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = BallTracker("data/input_video.mp4")
    tracker.run()
