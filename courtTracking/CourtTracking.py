import cv2
import numpy as np

class CourtDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        # White color range in HSV
        # Lower: low saturation, high value | Upper: any hue, low saturation, max value
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 50, 255])

    def get_court_lines(self, frame):
        # 1. Pre-processing: Isolate white areas
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # 2. Edge Detection
        # Kernel to clean up noise and thicken lines slightly
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)

        # 3. Hough Line Transform (detects line segments)
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=20
        )
        
        return lines

    def filter_singles_court(self, lines, frame_width):
        """
        Heuristic to prioritize singles lines:
        In standard broadcast views, singles sidelines are closer to the center
        than doubles sidelines.
        """
        if lines is None:
            return []

        filtered_lines = []
        center_x = frame_width / 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope to distinguish horizontal (baselines) from vertical (sidelines)
            slope = abs((y2 - y1) / (x2 - x1)) if (x2 - x1) != 0 else 999
            
            # Keep horizontal lines (Baselines/Service lines)
            if slope < 0.5:
                filtered_lines.append((x1, y1, x2, y2))
            
            # Keep vertical lines (Sidelines) - Filter for inner ones
            elif slope > 2:
                # Basic heuristic: ignore lines too close to the very edges of the image
                if center_x * 0.2 < x1 < center_x * 1.8:
                    filtered_lines.append((x1, y1, x2, y2))

        return filtered_lines

    def draw_points_on_lines(self, frame, lines, num_points=10):
        """Draws small points along the detected line segments."""
        for x1, y1, x2, y2 in lines:
            for i in range(num_points + 1):
                # Linear interpolation to find points along the line
                curr_x = int(x1 + (x2 - x1) * (i / num_points))
                curr_y = int(y1 + (y2 - y1) * (i / num_points))
                
                # Draw small circle (point)
                cv2.circle(frame, (curr_x, curr_y), 3, (0, 0, 255), -1)

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            raw_lines = self.get_court_lines(frame)
            singles_lines = self.filter_singles_court(raw_lines, frame.shape[1])
            
            # Visualize
            self.draw_points_on_lines(frame, singles_lines)

            cv2.imshow("Singles Court Edge Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CourtDetector("data/input_video.mp4")
    detector.run()