import cv2
import numpy as np


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
        return candidates, motion, green_mask, combined

    def detect(self, frame, gray, prev_gray):
        if prev_gray is None:
            return None, None, None, None

        candidates, motion, green_mask, combined = self.find_candidates(frame, gray, prev_gray)

        if not candidates:
            return None, motion, green_mask, combined

        best = min(candidates, key=lambda c: c[0])
        _, _, _, x, y, w, h = best
        return (x, y, w, h), motion, green_mask, combined

    def run(self):
        if not self.cap.isOpened():
            print("Cannot open video")
            return

        frame_ms = 30
        prev_gray = None
        frame_num = 0

        while True:
            t0 = cv2.getTickCount()
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_num += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox, motion, green_mask, combined = self.detect(frame, gray, prev_gray)

            debug_frame = frame.copy()

            if bbox is not None:
                x, y, w, h = bbox
                cx, cy = x + w // 2, y + h // 2
                half = max(w, h) // 2 + 4
                cv2.rectangle(debug_frame, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), 2)
                status = f"DETECTED ({cx},{cy})"
                status_color = (0, 255, 0)
            else:
                status = "NOT DETECTED"
                status_color = (0, 0, 255)

            # Disegna tutti i contorni in combined: giallo=passa i filtri, rosso=scartato
            if combined is not None:
                all_contours, _ = cv2.findContours(
                    combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in all_contours:
                    area = cv2.contourArea(c)
                    x2, y2, w2, h2 = cv2.boundingRect(c)
                    ratio = w2 / h2 if h2 > 0 else 0
                    perimeter = cv2.arcLength(c, True)
                    circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                    passed = (5 <= area <= 150 and 0.5 < ratio < 2.0 and circ >= 0.6)
                    box_color = (0, 255, 255) if passed else (0, 0, 255)
                    cv2.rectangle(debug_frame, (x2, y2), (x2 + w2, y2 + h2), box_color, 1)
                    cv2.putText(debug_frame, f"a={int(area)} c={circ:.2f}",
                                (x2, max(y2 - 4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            cv2.putText(debug_frame, f"Frame {frame_num} | {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Griglia 2x2: original+detection | motion | color | combined
            # Tutte ridimensionate a 480x270 per formare una finestra 960x540
            tile_w, tile_h = 960, 540

            def to_bgr(img):
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

            def tile(img):
                return cv2.resize(img, (tile_w // 2, tile_h // 2))

            top = np.hstack([
                tile(debug_frame),
                tile(to_bgr(motion) if motion is not None else np.zeros_like(debug_frame)),
            ])
            bottom = np.hstack([
                tile(to_bgr(green_mask) if green_mask is not None else np.zeros_like(debug_frame)),
                tile(to_bgr(combined) if combined is not None else np.zeros_like(debug_frame)),
            ])
            grid = np.vstack([top, bottom])

            # Etichette sui 4 quadranti
            for label, pos in [
                ("Original + detection", (10, 20)),
                ("Motion mask",          (tile_w // 2 + 10, 20)),
                ("Color mask",           (10, tile_h // 2 + 20)),
                ("Combined (AND)",       (tile_w // 2 + 10, tile_h // 2 + 20)),
            ]:
                cv2.putText(grid, label, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Ball Tracker - Debug", grid)

            prev_gray = gray
            elapsed_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
            delay = max(1, int(frame_ms - elapsed_ms))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = BallTracker("data/input_video2.mp4")
    tracker.run()