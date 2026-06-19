"""
evaluation/annotate.py

Interactive ground-truth annotation tool.

Two modes:

  boxes  — draw the two player bounding boxes on a subset of frames
           (every --step frames). Output CSV: frame,player_id,x,y,w,h
           P1 = NEAR player (bottom half), P2 = FAR player (top half),
           to stay consistent across frames regardless of tracker ids.

  court  — click the 8 court keypoints (TL, TR, BL, BR, STL, STR, SBL, SBR)
           on one frame. Output CSV: label,x,y  (same format produced by
           tracking/court_tracking.py, so the two are comparable).

Usage (from project root):
    python evaluation/annotate.py boxes --video data/Input_video2.mp4 \\
        --step 30 --output outputs/ground_truth/players_clip2_gt.csv

    python evaluation/annotate.py court --video data/Input_video2.mp4 \\
        --output outputs/ground_truth/input_video2_court_gt.csv

Keys (boxes mode, inside the OpenCV ROI selector):
    draw with the mouse, ENTER/SPACE = confirm box, c = cancel selection
    confirming an EMPTY selection (just press ENTER) skips that player
    after both players: any key moves on; ESC at any time = save & quit
"""

import argparse
import csv
import os

import cv2

DISPLAY_W = 1280  # annotation window width (coords are scaled back)

COURT_LABELS = ["TL", "TR", "BL", "BR", "STL", "STR", "SBL", "SBR"]
COURT_HINTS = {
    "TL":  "TOP-LEFT singles corner (far baseline)",
    "TR":  "TOP-RIGHT singles corner (far baseline)",
    "BL":  "BOTTOM-LEFT singles corner (near baseline)",
    "BR":  "BOTTOM-RIGHT singles corner (near baseline)",
    "STL": "far SERVICE line, LEFT end",
    "STR": "far SERVICE line, RIGHT end",
    "SBL": "near SERVICE line, LEFT end",
    "SBR": "near SERVICE line, RIGHT end",
}


def _scaled(frame):
    h, w = frame.shape[:2]
    # Cap the display scale at 1.0: never enlarge a frame narrower than
    # DISPLAY_W, otherwise we upscale past the source resolution and waste
    # precision (annotated coords are divided back by `s` anyway).
    s = min(1.0, DISPLAY_W / w)
    return cv2.resize(frame, (int(w * s), int(h * s))), s


# ── boxes mode ─────────────────────────────────────────────────────────────────

def annotate_boxes(video, step, start, output):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    rows = []
    frame_idx = start
    print(f"Annotating every {step} frames of {video} ({n_total} frames)")
    print("P1 = NEAR player (bottom), P2 = FAR player (top).")
    print("ENTER on an empty selection skips the player; ESC saves and quits.\n")

    quit_all = False
    while frame_idx < n_total and not quit_all:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        disp, s = _scaled(frame)

        for pid, who in ((1, "P1 (NEAR, bottom)"), (2, "P2 (FAR, top)")):
            view = disp.copy()
            cv2.putText(view, f"frame {frame_idx}  -  draw {who}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
            roi = cv2.selectROI("annotate", view, showCrosshair=True)
            # A zero-AREA roi (w==0 or h==0) is treated as "no box for this
            # player". selectROI returns the fully-empty (0,0,0,0) both for a
            # plain ENTER and for ESC, and a one-axis drag yields e.g.
            # (x,y,w,0); none of these is a usable box, so we route them all
            # through the skip/quit prompt rather than appending a degenerate
            # zero-area box. (selectROI gives no way to tell ESC from an empty
            # confirm apart, hence the explicit prompt below.)
            rx, ry, rw, rh = roi
            if rw == 0 or rh == 0:
                view2 = view.copy()
                cv2.putText(view2, "no box - q = save & quit, "
                                   "any other key = continue",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 200, 255), 2)
                cv2.imshow("annotate", view2)
                if (cv2.waitKey(0) & 0xFF) == ord("q"):
                    quit_all = True
                    break
                continue
            x, y, w, h = (int(round(c / s)) for c in roi)
            rows.append([frame_idx, pid, x, y, w, h])
            print(f"  frame {frame_idx}  P{pid}: x={x} y={y} w={w} h={h}")

        frame_idx += step

    cap.release()
    cv2.destroyAllWindows()

    with open(output, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "player_id", "x", "y", "w", "h"])
        wr.writerows(rows)
    print(f"\nSaved {len(rows)} boxes to {output}")


# ── court mode ─────────────────────────────────────────────────────────────────

def annotate_court(video, frame_no, output):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_no}")

    disp, s = _scaled(frame)
    points = []   # [(label, x_orig, y_orig)]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(COURT_LABELS):
            label = COURT_LABELS[len(points)]
            points.append((label, int(round(x / s)), int(round(y / s))))

    cv2.namedWindow("annotate court")
    cv2.setMouseCallback("annotate court", on_mouse)
    print("Click the 8 keypoints in order. u = undo, ENTER = save, q = abort.")

    while True:
        view = disp.copy()
        for label, xo, yo in points:
            p = (int(xo * s), int(yo * s))
            cv2.circle(view, p, 5, (0, 0, 255), -1)
            cv2.putText(view, label, (p[0] + 8, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) < len(COURT_LABELS):
            nxt = COURT_LABELS[len(points)]
            msg = f"click {nxt}: {COURT_HINTS[nxt]}"
        else:
            msg = "all 8 points set - press ENTER to save"
        cv2.putText(view, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 255), 2)
        cv2.imshow("annotate court", view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key in (13, 10) and len(points) == len(COURT_LABELS):
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("Aborted, nothing saved.")
            return

    cv2.destroyAllWindows()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "x", "y"])
        wr.writerows(points)
    print(f"Saved court keypoints to {output}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ground-truth annotation tool")
    sub = parser.add_subparsers(dest="mode", required=True)

    pb = sub.add_parser("boxes", help="annotate player bounding boxes")
    pb.add_argument("--video", default="data/Input_video2.mp4")
    pb.add_argument("--step",  type=int, default=30,
                    help="annotate every N-th frame (default: 30)")
    pb.add_argument("--start", type=int, default=30,
                    help="first frame to annotate (default: 30, after the "
                         "background warm-up)")
    pb.add_argument("--output",
                    default="outputs/ground_truth/players_clip2_gt.csv")

    pc = sub.add_parser("court", help="annotate court keypoints")
    pc.add_argument("--video", default="data/Input_video2.mp4")
    pc.add_argument("--frame", type=int, default=0,
                    help="frame to annotate (default: 0)")
    pc.add_argument("--output",
                    default="outputs/ground_truth/input_video2_court_gt.csv")

    args = parser.parse_args()
    if args.mode == "boxes":
        annotate_boxes(args.video, args.step, args.start, args.output)
    else:
        annotate_court(args.video, args.frame, args.output)


if __name__ == "__main__":
    main()
