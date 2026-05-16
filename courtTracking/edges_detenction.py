import cv2
import numpy as np

VIDEO_PATH = "data/input_video.mp4"

# ── White HSV bounds ──────────────────────────────────────────────────────────
LOWER_WHITE = np.array([0,   0, 180], dtype=np.uint8)
UPPER_WHITE = np.array([180, 50, 255], dtype=np.uint8)

# ── Hough line detection ──────────────────────────────────────────────────────
HOUGH_RHO        = 1       # distance resolution (pixels)
HOUGH_THETA      = np.pi / 180  # angle resolution (radians)
HOUGH_THRESHOLD  = 150      # minimum votes to accept a line
HOUGH_MIN_LENGTH = 150      # minimum segment length (pixels)
HOUGH_MAX_GAP    = 30      # maximum gap to join collinear segments (pixels)

# ── Intersection filtering ────────────────────────────────────────────────────
MIN_ANGLE_DEG   = 25       # ignore intersections between near-parallel lines
CLUSTER_RADIUS  = 50       # merge intersection dots closer than this (pixels)
DOT_RADIUS      = 10       # display radius of the red dot

# ── Region-of-interest (ROI) ──────────────────────────────────────────────────
# Broadcasts often have advertising banners at the top and scoreboards/overlays
# in the corners. Masking these out before Hough prevents spurious white blobs
# from generating false line segments that intersect outside the court.
ROI_TOP_FRACTION    = 0.20  # ignore top 20 % of frame (banner strip)
ROI_BOTTOM_FRACTION = 0.95  # ignore below 95 % of frame (score overlay)
ROI_LEFT_FRACTION   = 0.02  # ignore leftmost 2 % (frame edge artifacts)1
ROI_RIGHT_FRACTION  = 0.98  # ignore rightmost 2 %


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


# ── Main ──────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to read the first frame.")

h, w = frame.shape[:2]

# 1. White mask
hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

# 2. Apply ROI: zero out banner / overlay regions before any line detection
roi_top    = int(h * ROI_TOP_FRACTION)
roi_bottom = int(h * ROI_BOTTOM_FRACTION)
roi_left   = int(w * ROI_LEFT_FRACTION)
roi_right  = int(w * ROI_RIGHT_FRACTION)
mask[:roi_top,    :]    = 0   # top banner strip
mask[roi_bottom:, :]    = 0   # bottom overlay strip
mask[:, :roi_left]      = 0   # left edge
mask[:, roi_right:]     = 0   # right edge

# 3. Clean the mask: close small holes, then thin edges with Canny
kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask_clean   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
edges        = cv2.Canny(mask_clean, 50, 150)

# 4. Detect line segments via probabilistic Hough
raw_lines = cv2.HoughLinesP(
    edges,
    rho       = HOUGH_RHO,
    theta     = HOUGH_THETA,
    threshold = HOUGH_THRESHOLD,
    minLineLength = HOUGH_MIN_LENGTH,
    maxLineGap    = HOUGH_MAX_GAP,
)

if raw_lines is None:
    print("No lines detected — try loosening HOUGH_THRESHOLD or HOUGH_MIN_LENGTH.")
    cv2.imshow("White Mask — Step 1", cv2.bitwise_and(frame, frame, mask=mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise SystemExit

segments = raw_lines[:, 0, :]   # shape (N, 4)

# 5. Pairwise intersections: keep only those inside the ROI and non-parallel
raw_intersections = []
for i in range(len(segments)):
    for j in range(i + 1, len(segments)):
        angle = angle_between(segments[i], segments[j])
        # skip near-parallel (angle ≈ 0°) and near-antiparallel (angle ≈ 180°)
        if angle < MIN_ANGLE_DEG or angle > (180 - MIN_ANGLE_DEG):
            continue
        pt = line_intersection(segments[i], segments[j])
        if pt is None:
            continue
        x, y = pt
        if roi_left <= x < roi_right and roi_top <= y < roi_bottom:
            raw_intersections.append((x, y))

# 6. Cluster nearby intersections to get one dot per court corner/edge point
dots = cluster_points(raw_intersections, CLUSTER_RADIUS)

# 7. Draw on original frame
output = frame.copy()
for (x, y) in dots:
    cv2.circle(output, (x, y), DOT_RADIUS, (0, 0, 255), -1)   # filled red dot

print(f"Detected {len(segments)} line segments → {len(dots)} intersection points.")

cv2.imshow("White Mask — Step 1  |  intersection dots", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
