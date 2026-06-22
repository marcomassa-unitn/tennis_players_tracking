import argparse
import csv
import os

import cv2
import numpy as np

# Dual import: works both standalone (tracking/ on sys.path) and as a package
# (tracking.court_tracking, when run via pipeline.py).
try:
    from tracking._fps_utils import safe_fps
except ImportError:
    from _fps_utils import safe_fps

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
ROI_BOTTOM_FRACTION = 0.95
ROI_LEFT_FRACTION   = 0.02
ROI_RIGHT_FRACTION  = 0.98

# ── Corner selection ──────────────────────────────────────────────────────────
# Rank within the extreme-score ordering: 0 = absolute extreme; 1 = second-most
# extreme (use when doubles-alley dots sit further out than the singles corners).
SINGLES_CORNER_RANK = 0

# ── Singles-court proportional filter ────────────────────────────────────────
# ITF widths: doubles 10.97 m, singles 8.23 m, so each alley is 1.37 m.
# Alley as a fraction of the full doubles width:
ALLEY_RATIO = 1.37 / 10.97   # ≈ 0.1249

# Max y-gap (px) for two dots to count as the same horizontal court-line row.
Y_GROUP_TOLERANCE = 50

# ── Net-zone filter ───────────────────────────────────────────────────────────
# Perspective puts the net at ~35–45 % of court image height below the top
# baseline (not the real-world 50 %). Dots within ±NET_Y_BAND are dropped.
NET_Y_FRACTION = 0.40   # net position as fraction of court height
NET_Y_BAND     = 0.22   # half-width of the exclusion band (fraction of height)

# ── Playback ──────────────────────────────────────────────────────────────────
# Per-frame waitKey delay is base / fps (ms).
PLAYBACK_DELAY_BASE_MS = 500


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
    """Merge points within `radius`; return per-cluster centroids.

    Input is pre-sorted by (y, x) so the result is order-independent. Each
    cluster's center is the running mean of all its members.
    """
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))

    clusters = []   # each entry: [center_x, center_y, sum_x, sum_y, count]
    for pt in sorted_points:
        merged = False
        for c in clusters:
            if np.hypot(pt[0] - c[0], pt[1] - c[1]) < radius:
                c[2] += pt[0]
                c[3] += pt[1]
                c[4] += 1
                # Re-mean over all members so far.
                c[0] = c[2] / c[4]
                c[1] = c[3] / c[4]
                merged = True
                break
        if not merged:
            clusters.append([pt[0], pt[1], pt[0], pt[1], 1])
    return [(int(round(c[0])), int(round(c[1]))) for c in clusters]


def filter_singles_by_proportion(dots, y_tol, alley_ratio):
    """Drop doubles-alley corner dots using ITF court proportions.

    Per horizontal row (dots grouped by similar y):
      - <= 2 dots: no alley corners present, kept verbatim.
      - >= 3 dots: outermost two are the doubles corners; predict each singles
        corner x at `span * alley_ratio` inward and keep the inner dot nearest
        each prediction, discarding the rest.
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

        inner = by_x[1:-1]   # candidates between the doubles corners

        sl = min(inner, key=lambda p: abs(p[0] - x_sl_pred))
        sr = min(inner, key=lambda p: abs(p[0] - x_sr_pred))

        if sl == sr:
            filtered.append(sl)
        else:
            filtered.extend([sl, sr])

    return filtered


# ITF keypoint positions in feet, origin at BL (length 78 ft, singles width
# 27 ft, service lines 18 ft from each baseline).
REAL_FT = {
    "BL":  (0.0,  0.0),  "BR":  (27.0, 0.0),
    "TL":  (0.0, 78.0),  "TR":  (27.0, 78.0),
    "SBL": (0.0, 18.0),  "SBR": (27.0, 18.0),
    "STL": (0.0, 60.0),  "STR": (27.0, 60.0),
}


def project_all_corners(known):
    """Fit a real-world-to-image homography from >= 4 measured keypoints
    {label: (x, y) px} and project the remaining ITF keypoints into the image.

    Measured points are returned unchanged. Result has all 8 labels.

    Raises RuntimeError if the homography fit fails or is degenerate.
    """
    labels = list(known)
    src = np.array([REAL_FT[k] for k in labels], dtype=np.float32)
    dst = np.array([known[k] for k in labels], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC,
                              ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("Homography from detected corners failed.")

    out = {}
    for label, (rx, ry) in REAL_FT.items():
        p = H @ np.array([rx, ry, 1.0])
        # Near-zero w would produce inf/NaN pixels.
        if abs(p[2]) < 1e-9:
            raise RuntimeError(
                f"Degenerate homography: zero homogeneous divisor for {label}.")
        out[label] = (int(round(p[0] / p[2])), int(round(p[1] / p[2])))
    for label, (x, y) in known.items():
        out[label] = (int(round(x)), int(round(y)))
    return out


def fit_sideline(segments, corner, max_dist=12.0, min_len=100.0):
    """Least-squares fit x = a*y + b for the sideline through `corner`.

    Uses steep Hough segments (>= `min_len`) whose extended line passes within
    `max_dist` px of the corner. Returns (a, b), or None if < 2 points qualify.
    """
    cx, cy = corner
    pts = []
    for x1, y1, x2, y2 in segments:
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < min_len or abs(dy) <= abs(dx):
            continue
        dist = abs(dy * (cx - x1) - dx * (cy - y1)) / length
        if dist <= max_dist:
            pts += [(x1, y1), (x2, y2)]
    if len(pts) < 2:
        return None
    pts = np.array(pts, dtype=np.float64)
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    return float(a), float(b)


def find_ground_line_bands(hsv_img, xl_fit, xr_fit, y_min, y_max,
                           cov_thr=0.5, max_thickness=14):
    """Find horizontal ground lines via a per-row white-coverage profile.

    For each row, measures the fraction of white-ish pixels (relaxed mask,
    since far lines are dimmer) along the chord between the two fitted
    sidelines. Ground lines are THIN high-coverage runs; the elevated net is a
    thick run and is rejected. Returns the center y of each thin band.
    """
    relaxed = cv2.inRange(hsv_img,
                          np.array([0, 0, 195], dtype=np.uint8),
                          np.array([180, 95, 255], dtype=np.uint8))
    runs, run = [], None
    for y in range(int(y_min), int(y_max)):
        xl = int(xl_fit[0] * y + xl_fit[1]) + 6
        xr = int(xr_fit[0] * y + xr_fit[1]) - 6
        cov = relaxed[y, xl:xr].mean() / 255.0 if xr - xl >= 50 else 0.0
        if cov >= cov_thr:
            run = [y, y] if run is None else [run[0], y]
        elif run is not None:
            runs.append(run)
            run = None
    if run is not None:
        runs.append(run)
    return [(y0 + y1) / 2.0 for y0, y1 in runs
            if (y1 - y0 + 1) <= max_thickness]


def select_far_lines(bands, bl, br, xl_fit, xr_fit, tol=12.0):
    """Identify the far baseline and service-line bands by projective fit.

    For each candidate (baseline, service) pair, builds the homography from
    (TL, TR, BL, BR) and requires the projected far/near service lines to land
    on detected bands within `tol` px. Rejects banner/fence lines and the
    elevated net. Returns (y_baseline, y_service, residuals), or None.
    """
    best = None
    for yb in bands:
        for ys in bands:
            if ys <= yb:        # service line sits below the baseline in image y
                continue
            tl = (xl_fit[0] * yb + xl_fit[1], yb)
            tr = (xr_fit[0] * yb + xr_fit[1], yb)
            try:
                pts = project_all_corners(
                    {"TL": tl, "TR": tr, "BL": bl, "BR": br})
            except RuntimeError:
                continue
            y_far = (pts["STL"][1] + pts["STR"][1]) / 2.0
            y_near = (pts["SBL"][1] + pts["SBR"][1]) / 2.0
            r_far = abs(y_far - ys)
            r_near = min(abs(y_near - b) for b in bands)
            if r_far > tol or r_near > tol:
                continue
            score = r_far + r_near
            if best is None or score < best[0]:
                best = (score, yb, ys, r_far, r_near)
    if best is None:
        return None
    _, yb, ys, r_far, r_near = best
    return yb, ys, (r_far, r_near)


def find_court_corners(dots, frame_shape, rank=0):
    """Pick the 4 boundary corners by scoring normalised (x, y) coordinates.

      TL -> min(nx + ny)   top-left
      TR -> min(ny - nx)   top-right
      BL -> min(nx - ny)   bottom-left
      BR -> max(nx + ny)   bottom-right

    `rank` selects the n-th in each score order (0 = most extreme).
    Raises ValueError if fewer than 4 dots are given.
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


# ── Court tracker ─────────────────────────────────────────────────────────────

class CourtTracker:
    """Detect court keypoints: white mask + Hough lines + intersection
    clustering + ITF-proportion filtering."""

    def __init__(self, video_path, no_display=False, roi_top=0.15,
                 far_line="baseline", output_dir="outputs"):
        self.video_path = video_path
        self.no_display = no_display
        self.roi_top = roi_top
        self.far_line = far_line
        self.output_dir = output_dir

    def run(self):
        """Detect the 8 keypoints, write them to CSV, and (unless no_display)
        play back the annotated video. Returns the 8-keypoint dict."""
        video_stem = os.path.splitext(os.path.basename(self.video_path))[0]
        csv_path = os.path.join(self.output_dir, "court_coordinates",
                                f"{video_stem}_court.csv")

        os.makedirs(os.path.join(self.output_dir, "court_coordinates"),
                    exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        # finally releases the capture/window on every exit path.
        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read the first frame.")

            h, w = frame.shape[:2]

            # 1. White mask
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

            # 2. Apply ROI
            roi_top    = int(h * self.roi_top)
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
                if not self.no_display:
                    cv2.imshow("White Mask",
                               cv2.bitwise_and(frame, frame, mask=mask))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                raise RuntimeError(
                    "No court lines detected (try loosening HOUGH_THRESHOLD / "
                    "HOUGH_MIN_LENGTH).")

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
            print(f"Detected {len(segments)} line segments -> {len(all_dots)} intersection points (pre-filter).")

            # 6b. Strip doubles-alley corners using ITF width proportions
            dots = filter_singles_by_proportion(all_dots, Y_GROUP_TOLERANCE, ALLEY_RATIO)
            print(f"After singles-court proportion filter: {len(dots)} dots.")
            for i, d in enumerate(dots):
                print(f"  dot[{i:2d}] = {d}")

            # 7. Select the 4 extreme corners of the detected dots
            tl, tr, bl, br = find_court_corners(dots, frame.shape, rank=SINGLES_CORNER_RANK)

            # 7b. Refine the far side. Hough intersections are unreliable there (far
            # ground lines are thin/dim while banner edges and the elevated net are
            # bright), so instead scan a white-coverage profile between the fitted
            # sidelines and pick the (baseline, service) pair consistent with the ITF model.
            far_sel = None
            xl_fit = fit_sideline(segments, bl)
            xr_fit = fit_sideline(segments, br)
            if xl_fit is not None and xr_fit is not None:
                bands = find_ground_line_bands(hsv, xl_fit, xr_fit,
                                               y_min=roi_top,
                                               y_max=min(bl[1], br[1]) - 30)
                if bands:
                    far_sel = select_far_lines(bands, bl, br, xl_fit, xr_fit)

            if far_sel is not None:
                yb, ys, (r_far, r_near) = far_sel
                tl = (int(round(xl_fit[0] * yb + xl_fit[1])), int(round(yb)))
                tr = (int(round(xr_fit[0] * yb + xr_fit[1])), int(round(yb)))
                known = {"TL": tl, "TR": tr, "BL": bl, "BR": br}
                print(f"Far lines via coverage profile: baseline y={yb:.0f}, "
                      f"service y={ys:.0f} (residuals: far {r_far:.1f}px, "
                      f"near {r_near:.1f}px)")
            else:
                print("WARNING: coverage-profile far-line detection failed, falling "
                      "back to Hough corners" +
                      (" interpreted as the far service line (--far-line service)."
                       if self.far_line == "service" else "."))
                if self.far_line == "service":
                    known = {"STL": tl, "STR": tr, "BL": bl, "BR": br}
                else:
                    known = {"TL": tl, "TR": tr, "BL": bl, "BR": br}
            all_pts = project_all_corners(known)

            corners = {k: all_pts[k] for k in ("TL", "TR", "BL", "BR")}
            service = {k: all_pts[k] for k in ("STL", "STR", "SBL", "SBR")}
            tl, tr, bl, br = corners["TL"], corners["TR"], corners["BL"], corners["BR"]
            print(f"\nCourt corners  TL:{tl}  TR:{tr}  BL:{bl}  BR:{br}")
            print(f"Service corners  STL:{service['STL']}  STR:{service['STR']}  SBL:{service['SBL']}  SBR:{service['SBR']}")

            # 7c. Drop net-zone dots from the display only (corners already saved).
            court_span = bl[1] - tl[1]
            # A non-positive span (collapsed/inverted corners) makes the net band
            # meaningless, so skip the filter.
            if court_span > 0:
                net_y = tl[1] + court_span * NET_Y_FRACTION
                net_band = court_span * NET_Y_BAND
                dots = [d for d in dots
                        if abs(d[1] - net_y) > net_band]

            # 8. Save all 8 named corners to CSV
            keypoints = {**corners, **service}
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["label", "x", "y"])
                for label in ("TL", "TR", "BL", "BR", "STL", "STR", "SBL", "SBR"):
                    pt = keypoints[label]
                    writer.writerow([label, pt[0], pt[1]])
            print(f"Saved all corners to '{csv_path}'.")

            # 9. Play back: dots in red, the 4 corners ringed magenta
            if self.no_display:
                return keypoints

            self._play_back(cap, corners, service, dots, tl, tr, bl, br)
            return keypoints
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _play_back(self, cap, corners, service, dots, tl, tr, bl, br):
        """Rewind and draw, per frame, the court fill + outline, all dots, and
        the corner/service markers. Display only; writes nothing. Caller owns
        cap.release()/destroyAllWindows() and the no_display early-return."""
        fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
        delay = max(1, int(PLAYBACK_DELAY_BASE_MS / fps))

        # Seek to frame 0; on some codecs this lands on a nearby keyframe, not exactly 0.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        quad_pts = np.array([tl, tr, br, bl], dtype=np.int32)

        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            # Translucent green court fill
            overlay = current_frame.copy()
            cv2.fillPoly(overlay, [quad_pts], (0, 180, 0))
            cv2.addWeighted(overlay, 0.20, current_frame, 0.80, 0, current_frame)

            cv2.polylines(current_frame, [quad_pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Detected dots: filled red
            for (x, y) in dots:
                cv2.circle(current_frame, (x, y), DOT_RADIUS, (0, 0, 255), -1)

            # Court corners: magenta ring + label
            for label, corner in corners.items():
                cv2.circle(current_frame, corner, DOT_RADIUS + 4, (255, 0, 255), 2)
                cv2.putText(current_frame, label, (corner[0] + 12, corner[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

            # Service corners: cyan ring + label. STL also gets a red fill since it is
            # geometric (not from Hough) and thus has no red dot of its own.
            for label, corner in service.items():
                if label == "STL":
                    cv2.circle(current_frame, corner, DOT_RADIUS, (0, 0, 255), -1)
                cv2.circle(current_frame, corner, DOT_RADIUS + 4, (255, 255, 0), 2)
                cv2.putText(current_frame, label, (corner[0] + 12, corner[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            cv2.imshow("Tennis Court - Video Playback", current_frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Court keypoint detection: white mask + Hough lines + "
                    "intersection clustering + ITF-proportion filtering")
    parser.add_argument("--video", default="data/input_video2.mp4")
    parser.add_argument("--no-display", action="store_true",
                        help="skip the annotated video playback")
    parser.add_argument("--roi-top", type=float, default=0.15, dest="roi_top",
                        help="top ROI cut as fraction of frame height; raise it "
                             "when white banner/fence lines above the court get "
                             "mistaken for the far baseline (default: 0.15)")
    parser.add_argument("--far-line", choices=["baseline", "service"],
                        default="baseline", dest="far_line",
                        help="which court line the TOPMOST detected corner row "
                             "belongs to. Use 'service' when the far baseline is "
                             "too thin/washed-out for the Hough transform and "
                             "the top corners are on the far service line; the "
                             "real baseline corners are then projected from ITF "
                             "proportions (default: baseline)")
    parser.add_argument("--output", default="outputs", dest="output_dir",
                        help="output directory for generated files such as the "
                             "court-coordinates CSV (default: outputs)")
    args = parser.parse_args()

    CourtTracker(
        video_path=args.video,
        no_display=args.no_display,
        roi_top=args.roi_top,
        far_line=args.far_line,
        output_dir=args.output_dir,
    ).run()
