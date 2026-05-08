import cv2
import numpy as np

VIDEO = "input_video.mp4"

def fit(segs):
    pts = np.array([[s[0],s[1]] for s in segs] + [[s[2],s[3]] for s in segs], dtype=np.float32)
    return cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten().tolist()

def cross(l1, l2):
    vx1,vy1,cx1,cy1 = l1;  vx2,vy2,cx2,cy2 = l2
    d = vx1*vy2 - vy1*vx2
    if abs(d) < 1e-6: return None
    t = ((cx2-cx1)*vy2 - (cy2-cy1)*vx2) / d
    return int(cx1 + t*vx1), int(cy1 + t*vy1)

def get_corners(frame):
    h, w = frame.shape[:2]

    # Isolate white court lines in HSV, ignore crowd (top 22%) and score bar (bottom 8%)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 50, 255]))
    mask[:int(h * 0.22)] = 0
    mask[int(h * 0.92):] = 0

    edges = cv2.Canny(cv2.GaussianBlur(mask, (5, 5), 1), 50, 150)
    raw   = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=80, maxLineGap=15)
    if raw is None:
        return []

    h_segs, l_segs, r_segs = [], [], []
    for x1, y1, x2, y2 in raw[:, 0]:
        if np.hypot(x2-x1, y2-y1) < 70:
            continue
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
        mid_x = (x1+x2) / 2
        if angle < 15:                           # baseline / service line
            h_segs.append((x1, y1, x2, y2))
        elif angle > 45:                         # sideline (perspective angle)
            (l_segs if mid_x < w/2 else r_segs).append((x1, y1, x2, y2))

    if not h_segs or not l_segs or not r_segs:
        return []

    # Baselines: sort by y, take topmost cluster (far baseline) and bottommost (near baseline)
    h_segs.sort(key=lambda s: (s[1]+s[3])/2)
    n = max(1, len(h_segs) // 3)
    far_bl  = fit(h_segs[:n])
    near_bl = fit(h_segs[-n:])

    # Singles sidelines: take the innermost detected line on each side
    # left side  → rightmost cluster (highest x) = inner singles line
    # right side → leftmost cluster  (lowest x)  = inner singles line
    l_segs.sort(key=lambda s: (s[0]+s[2])/2, reverse=True)
    r_segs.sort(key=lambda s: (s[0]+s[2])/2)
    n_l = max(1, len(l_segs) // 2)
    n_r = max(1, len(r_segs) // 2)
    left_sl  = fit(l_segs[:n_l])
    right_sl = fit(r_segs[:n_r])

    corners = []
    for bl in [far_bl, near_bl]:
        for sl in [left_sl, right_sl]:
            pt = cross(bl, sl)
            if pt and -30 < pt[0] < w+30 and -30 < pt[1] < h+30:
                corners.append(pt)
    return corners

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open '{VIDEO}'")

print("Q = quit")
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    out = frame.copy()
    for pt in get_corners(frame):
        cv2.circle(out, pt, 8,  (0, 0, 255), -1,        cv2.LINE_AA)
        cv2.circle(out, pt, 13, (0, 255, 0),  2,         cv2.LINE_AA)

    cv2.imshow("Court Corners", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
