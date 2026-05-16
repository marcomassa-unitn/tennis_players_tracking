import csv
import os

import cv2
import numpy as np

VIDEO_PATH = "data/input_video.mp4"
_video_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
CSV_PATH    = os.path.join("outputs", "court_coordinates", f"{_video_stem}_court.csv")

# ── White HSV bounds ──────────────────────────────────────────────────────────
LOWER_WHITE = np.array([0,   0, 180], dtype=np.uint8)
UPPER_WHITE = np.array([180, 50, 255], dtype=np.uint8)

# ── Hough line detection ──────────────────────────────────────────────────────
HOUGH_RHO        = 1
HOUGH_THETA      = np.pi / 180
HOUGH_THRESHOLD  = 150
HOUGH_MIN_LENGTH = 80
HOUGH_MAX_GAP    = 30

# ── Intersection filtering ────────────────────────────────────────────────────
MIN_ANGLE_DEG  = 25
CLUSTER_RADIUS = 50
DOT_RADIUS     = 10

# ── Region-of-interest (ROI) ──────────────────────────────────────────────────
ROI_TOP_FRACTION    = 0.15
ROI_BOTTOM_FRACTION = 0.95
ROI_LEFT_FRACTION   = 0.02
ROI_RIGHT_FRACTION  = 0.98

# ── Corner selection ──────────────────────────────────────────────────────────
# 0 = absolute extreme corner (correct when only singles lines are detected).
# 1 = second-from-extreme (use when doubles-alley lines produce their own
#     corner dots that are more extreme than the singles corners).
SINGLES_CORNER_RANK = 0

# ── Singles-court proportional filter ────────────────────────────────────────
# ITF court dimensions (Rules of Tennis):
#   doubles width = 10.97 m,  singles width = 8.23 m
#   each alley    = (10.97 - 8.23) / 2 = 1.37 m
# The alley as a fraction of the total doubles width:
ALLEY_RATIO = 1.37 / 10.97   # ≈ 0.1249

# Two dots belong to the same horizontal court-line row when their y-values
# differ by less than this many pixels.
Y_GROUP_TOLERANCE = 50

# ── Net-zone filter ───────────────────────────────────────────────────────────
# Due to perspective, the net appears at roughly 35–45 % of the court's image
# height below the top baseline (not 50 %, which is real-world).
# Dots within ±NET_Y_BAND of that position are discarded from display.
NET_Y_FRACTION = 0.40   # approximate net position as fraction of court height
NET_Y_BAND     = 0.22   # half-width of the exclusion band (fraction of court height)


# ── Helpers ───────────────────────────────────────────────────────────────────

def line_intersection(seg1, seg2):
    """Return (x, y) intersection of two infinite lines, or None if parallel."""
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (x, y)


def angle_between(seg1, seg2):
    """Acute angle in degrees between two line segments."""
    dx1, dy1 = seg1[2] - seg1[0], seg1[3] - seg1[1]
    dx2, dy2 = seg2[2] - seg2[0], seg2[3] - seg2[1]
    cos_val = abs(dx1 * dx2 + dy1 * dy2) / (
        (np.hypot(dx1, dy1) + 1e-9) * (np.hypot(dx2, dy2) + 1e-9)
    )
    cos_val = np.clip(cos_val, 0, 1)
    return np.degrees(np.arccos(cos_val))


def cluster_points(points, radius):
    """Merge points within `radius` of each other; return cluster centroids."""
    clusters = []
    for pt in points:
        merged = False
        for c in clusters:
            if np.hypot(pt[0] - c[0], pt[1] - c[1]) < radius:
                c[0] = (c[0] + pt[0]) / 2
                c[1] = (c[1] + pt[1]) / 2
                merged = True
                break
        if not merged:
            clusters.append(list(pt))
    return [(int(round(c[0])), int(round(c[1]))) for c in clusters]


def filter_singles_by_proportion(dots, y_tol, alley_ratio):
    """
    Discard doubles-alley corner dots using ITF court proportions.

    For each horizontal row of dots (grouped by similar y):
      - Rows with <= 2 dots have no alley corners to strip — kept as-is.
      - Rows with >= 3 dots: the leftmost and rightmost are treated as the
        doubles-court corners.  The expected singles-corner x-positions are
        computed as:
            x_singles_left  = x_doubles_left  + span * alley_ratio
            x_singles_right = x_doubles_right - span * alley_ratio
        where span = x_doubles_right - x_doubles_left.
        From the remaining inner dots the one closest to each predicted
        singles position is kept; the rest are discarded.
    """
    if not dots:
        return dots

    sorted_dots = sorted(dots, key=lambda p: p[1])

    groups, current = [], [sorted_dots[0]]
    for pt in sorted_dots[1:]:
        if abs(pt[1] - current[0][1]) <= y_tol:
            current.append(pt)
        else:
            groups.append(current)
            current = [pt]
    groups.append(current)

    filtered = []
    for group in groups:
        by_x = sorted(group, key=lambda p: p[0])

        if len(by_x) <= 2:
            filtered.extend(by_x)
            continue

        x_dl, x_dr = by_x[0][0], by_x[-1][0]
        span = x_dr - x_dl
        x_sl_pred = x_dl + span * alley_ratio
        x_sr_pred = x_dr - span * alley_ratio

        inner = by_x[1:-1]   # everything between the two doubles corners

        sl = min(inner, key=lambda p: abs(p[0] - x_sl_pred))
        sr = min(inner, key=lambda p: abs(p[0] - x_sr_pred))

        if sl == sr:
            filtered.append(sl)
        else:
            filtered.extend([sl, sr])

    return filtered


def compute_service_corners(tl, tr, bl, br):
    """
    Project service-line corners from real-world ITF proportions into image
    space via a homography built from the 4 detected court corners.

    ITF dimensions used:
      court length = 78 ft,  singles width = 27 ft
      service line = 21 ft from net = 18 ft from each baseline
    """
    COURT_LEN     = 78.0
    COURT_WID     = 27.0
    SVC_FROM_BASE = 18.0

    src = np.array([               # real-world corners (ft, origin at BL)
        [0,          0         ],  # BL
        [COURT_WID,  0         ],  # BR
        [COURT_WID,  COURT_LEN ],  # TR
        [0,          COURT_LEN ],  # TL
    ], dtype=np.float32)

    dst = np.array([bl, br, tr, tl], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)

    def project(rx, ry):
        p = H @ np.array([rx, ry, 1.0])
        return (int(round(p[0] / p[2])), int(round(p[1] / p[2])))

    y_top = COURT_LEN - SVC_FROM_BASE   # 60 ft — top service line
    y_bot = SVC_FROM_BASE               # 18 ft — bottom service line
    return {
        "STL": project(0,          y_top),
        "STR": project(COURT_WID,  y_top),
        "SBL": project(0,          y_bot),
        "SBR": project(COURT_WID,  y_bot),
    }


def find_court_corners(dots, frame_shape, rank=0):
    """
    Select the 4 court boundary corners using normalised coordinate scoring.
      TL → min(norm_x + norm_y)   topmost  + leftmost
      TR → min(norm_y - norm_x)   topmost  + rightmost
      BL → min(norm_x - norm_y)   bottommost + leftmost
      BR → max(norm_x + norm_y)   bottommost + rightmost

    `rank` selects the n-th position in the sorted score; 0 = most extreme.
    """
    pts = np.array(dots, dtype=np.float32)
    if len(pts) < 4:
        raise ValueError(f"Need at least 4 dots; got {len(pts)}.")

    h, w = frame_shape[:2]
    nx = pts[:, 0] / w
    ny = pts[:, 1] / h

    def pick(scores, ascending):
        order = np.argsort(scores) if ascending else np.argsort(-scores)
        idx = min(rank, len(order) - 1)
        p = pts[order[idx]]
        return (int(round(p[0])), int(round(p[1])))

    tl = pick(nx + ny, ascending=True)
    tr = pick(ny - nx, ascending=True)
    bl = pick(nx - ny, ascending=True)
    br = pick(nx + ny, ascending=False)
    return tl, tr, bl, br


# ── Main ──────────────────────────────────────────────────────────────────────

os.makedirs(os.path.join("outputs", "court_coordinates"), exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read the first frame.")

h, w = frame.shape[:2]

# 1. White mask
hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

# 2. Apply ROI
roi_top    = int(h * ROI_TOP_FRACTION)
roi_bottom = int(h * ROI_BOTTOM_FRACTION)
roi_left   = int(w * ROI_LEFT_FRACTION)
roi_right  = int(w * ROI_RIGHT_FRACTION)
mask[:roi_top,    :] = 0
mask[roi_bottom:, :] = 0
mask[:, :roi_left]   = 0
mask[:, roi_right:]  = 0

# 3. Clean the mask
kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
edges      = cv2.Canny(mask_clean, 50, 150)

# 4. Detect line segments
raw_lines = cv2.HoughLinesP(
    edges,
    rho=HOUGH_RHO,
    theta=HOUGH_THETA,
    threshold=HOUGH_THRESHOLD,
    minLineLength=HOUGH_MIN_LENGTH,
    maxLineGap=HOUGH_MAX_GAP,
)

if raw_lines is None:
    print("No lines detected — try loosening HOUGH_THRESHOLD or HOUGH_MIN_LENGTH.")
    cv2.imshow("White Mask", cv2.bitwise_and(frame, frame, mask=mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise SystemExit

segments = raw_lines[:, 0, :]

# 5. Pairwise intersections
raw_intersections = []
for i in range(len(segments)):
    for j in range(i + 1, len(segments)):
        angle = angle_between(segments[i], segments[j])
        if angle < MIN_ANGLE_DEG or angle > (180 - MIN_ANGLE_DEG):
            continue
        pt = line_intersection(segments[i], segments[j])
        if pt is None:
            continue
        x, y = pt
        if roi_left <= x < roi_right and roi_top <= y < roi_bottom:
            raw_intersections.append((x, y))

# 6. Cluster nearby intersections
all_dots = cluster_points(raw_intersections, CLUSTER_RADIUS)
print(f"Detected {len(segments)} line segments → {len(all_dots)} intersection points (pre-filter).")

# 6b. Strip doubles-alley corners using ITF width proportions
dots = filter_singles_by_proportion(all_dots, Y_GROUP_TOLERANCE, ALLEY_RATIO)
print(f"After singles-court proportion filter: {len(dots)} dots.")
for i, d in enumerate(dots):
    print(f"  dot[{i:2d}] = {d}")

# 7. Select the 4 court corners
tl, tr, bl, br = find_court_corners(dots, frame.shape, rank=SINGLES_CORNER_RANK)
corners = {"TL": tl, "TR": tr, "BL": bl, "BR": br}
print(f"\nCourt corners  TL:{tl}  TR:{tr}  BL:{bl}  BR:{br}")

# 7b. Compute service-line corners from ITF proportions + homography
service = compute_service_corners(tl, tr, bl, br)
print(f"Service corners  STL:{service['STL']}  STR:{service['STR']}  SBL:{service['SBL']}  SBR:{service['SBR']}")

# 7c. Discard net-zone dots from display (corners are already saved above).
court_span = bl[1] - tl[1]
net_y      = tl[1] + court_span * NET_Y_FRACTION
dots = [d for d in dots if abs(d[1] - net_y) > court_span * NET_Y_BAND]

# 8. Save all 8 named corners to CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "x", "y"])
    for label in ("TL", "TR", "BL", "BR", "STL", "STR", "SBL", "SBR"):
        pt = {**corners, **service}[label]
        writer.writerow([label, pt[0], pt[1]])
print(f"Saved all corners to '{CSV_PATH}'.")

# 9. Play video — all dots in red, 4 corners highlighted with magenta ring
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30
delay = int(500 / fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
quad_pts = np.array([tl, tr, br, bl], dtype=np.int32)

while True:
    ret, current_frame = cap.read()
    if not ret:
        break

    # Semi-transparent green fill over the court area
    overlay = current_frame.copy()
    cv2.fillPoly(overlay, [quad_pts], (0, 180, 0))
    cv2.addWeighted(overlay, 0.20, current_frame, 0.80, 0, current_frame)

    # Court boundary outline
    cv2.polylines(current_frame, [quad_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # All detected dots — red filled circles
    for (x, y) in dots:
        cv2.circle(current_frame, (x, y), DOT_RADIUS, (0, 0, 255), -1)

    # Court corners — magenta ring + label
    for label, corner in corners.items():
        cv2.circle(current_frame, corner, DOT_RADIUS + 4, (255, 0, 255), 2)
        cv2.putText(current_frame, label, (corner[0] + 12, corner[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    # Service-line corners — cyan ring + label; STL gets a red fill too because
    # it is computed geometrically (not from Hough), so it has no red dot yet.
    for label, corner in service.items():
        if label == "STL":
            cv2.circle(current_frame, corner, DOT_RADIUS, (0, 0, 255), -1)
        cv2.circle(current_frame, corner, DOT_RADIUS + 4, (255, 255, 0), 2)
        cv2.putText(current_frame, label, (corner[0] + 12, corner[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    cv2.imshow("Tennis Court - Video Playback", current_frame)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
