import cv2
import numpy as np

class CourtDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 50, 255])

    def get_court_lines(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)
        return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)

    def _cluster(self, lines_with_pos, threshold=40):
        """Merge line segments whose position (x or y avg) is within threshold pixels."""
        if not lines_with_pos:
            return []
        lines_with_pos.sort(key=lambda l: l[4])
        clusters = [[lines_with_pos[0]]]
        for l in lines_with_pos[1:]:
            if l[4] - clusters[-1][-1][4] < threshold:
                clusters[-1].append(l)
            else:
                clusters.append([l])
        merged = []
        for c in clusters:
            merged.append((
                int(np.mean([l[0] for l in c])),
                int(np.mean([l[1] for l in c])),
                int(np.mean([l[2] for l in c])),
                int(np.mean([l[3] for l in c])),
                float(np.mean([l[4] for l in c]))
            ))
        return merged

    def _intersect(self, l1, l2):
        """Compute the intersection point of two infinite lines defined by two endpoints each."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)))

    def find_singles_corners(self, lines, frame_shape):
        if lines is None:
            return None

        h, w = frame_shape[:2]
        center_x = w / 2

        h_raw, v_raw = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx == 0 and dy == 0:
                continue
            slope = dy / dx if dx != 0 else 999
            if slope < 0.5:
                h_raw.append((x1, y1, x2, y2, (y1 + y2) / 2))
            elif slope > 2:
                v_raw.append((x1, y1, x2, y2, (x1 + x2) / 2))

        h_clustered = self._cluster(h_raw)
        v_clustered = self._cluster(v_raw)

        if len(h_clustered) < 2 or len(v_clustered) < 2:
            return None

        # Top and bottom baselines (smallest and largest y)
        h_clustered.sort(key=lambda l: l[4])
        top = h_clustered[0][:4]
        bottom = h_clustered[-1][:4]

        # Singles sidelines: closest vertical line to center from each side.
        # In a doubles-court view this selects the inner (singles) sidelines;
        # in a singles-only view it selects the only sideline on each side.
        left_candidates = sorted([l for l in v_clustered if l[4] < center_x], key=lambda l: -l[4])
        right_candidates = sorted([l for l in v_clustered if l[4] >= center_x], key=lambda l: l[4])

        if left_candidates and right_candidates:
            left = left_candidates[0][:4]
            right = right_candidates[0][:4]
        else:
            # Fallback: use leftmost and rightmost detected vertical lines
            v_clustered.sort(key=lambda l: l[4])
            left = v_clustered[0][:4]
            right = v_clustered[-1][:4]

        # 4 corners as intersections of the two baselines with the two sidelines
        tl = self._intersect(top, left)
        tr = self._intersect(top, right)
        bl = self._intersect(bottom, left)
        br = self._intersect(bottom, right)

        if None in [tl, tr, bl, br]:
            return None

        # Discard corners that land far outside the frame (bad line extrapolation)
        margin = max(h, w) // 2
        for pt in [tl, tr, bl, br]:
            if not (-margin < pt[0] < w + margin and -margin < pt[1] < h + margin):
                return None

        return [tl, tr, bl, br]

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            raw_lines = self.get_court_lines(frame)
            corners = self.find_singles_corners(raw_lines, frame.shape)

            if corners:
                for pt in corners:
                    cv2.circle(frame, pt, 8, (0, 0, 255), -1)

            cv2.imshow("Singles Court Corner Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CourtDetector("data/input_video.mp4")
    detector.run()
