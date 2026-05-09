import argparse
import cv2
import numpy as np

def build_playing_field_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Keep only the centered square zone
    h, w = mask.shape
    cx, cy = w // 2, h // 2
    half_w = w // 4
    half_h = int(h * 3 / 8)
    left, right = cx - half_w, cx + half_w
    top, bottom = cy - half_h, cy + half_h
    mask[:top, :] = 0
    mask[bottom:, :] = 0
    mask[:, :left] = 0
    mask[:, right:] = 0

    # PRE-PROCESSING:
    # We thicken the white pixels so that gaps in the service lines are closed
    # before we try to detect them as a single long line.
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Edge Detection
    edges = cv2.Canny(mask, 70, 120)

    clean_mask = np.zeros_like(mask)

    # We lower minLineLength to 80 to catch the shorter service lines
    # We increase maxLineGap to 50 to "jump" across players or shadows
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                            minLineLength=80, maxLineGap=30)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            dx = x2 - x1
            dy = y2 - y1
            angle = np.abs(np.degrees(np.arctan2(dy, dx)))
            avg_y = (y1 + y2) / 2

            # Ignore the net: horizontal lines near the frame center are likely the net
            if (angle < 20 and abs(avg_y - cy) < h * 0.12):
                continue
            if (angle < 20) or (70 < angle < 110):
                cv2.line(clean_mask, (x1, y1), (x2, y2), 255, 1)
    return clean_mask

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask_output = build_playing_field_mask(frame)
        cv2.imshow("Complete Court Mask", mask_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="data/input_video.mp4")
    args = parser.parse_args()
    main(args.video)