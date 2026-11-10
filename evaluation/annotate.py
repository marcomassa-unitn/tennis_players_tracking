"""
evaluation/annotate.py

Interactive ground-truth annotation tool.

Three modes:

  boxes  — draw the two player bounding boxes on a subset of frames
           (every --step frames). Output CSV: frame,player_id,x,y,w,h
           P1 = NEAR player (bottom half), P2 = FAR player (top half),
           to stay consistent across frames regardless of tracker ids.

  court  — click the 8 court keypoints (TL, TR, BL, BR, STL, STR, SBL, SBR)
           on one frame. Output CSV: label,x,y  (same format produced by
           tracking/court_tracking.py, so the two are comparable).

  ball   — click the ball centre on every --step frames (n = not visible),
           with a magnifier around the cursor. Output CSV: frame,visible,x,y.
           Resumes an existing file, skipping frames already labelled.

  bounce — step through the clip and mark every frame where the ball touches the
           ground, clicking the ball, or as hidden if the contact cannot be seen.
           Output CSV: frame,visible,x,y. Resumes an existing file.

Usage (from project root):
    python evaluation/annotate.py boxes --video data/Input_video2.mp4 \\
        --step 30 --output ground_truth/players_Input_video2_gt.csv

    python evaluation/annotate.py court --video data/Input_video2.mp4 \\
        --output ground_truth/court_Input_video2_gt.csv

    python evaluation/annotate.py ball --video data/Input_video2.mp4 --step 10

    # training labels where the current tracker misses the ball:
    python evaluation/annotate.py ball --video data/input10.mp4 --step 2 \\
        --frames-from outputs/input10/ball_coordinates/ball_input10.csv

    python evaluation/annotate.py bounce --video data/Input_video2.mp4

Keys (boxes mode, inside the OpenCV ROI selector):
    draw with the mouse, ENTER/SPACE = confirm box, c = cancel selection
    confirming an EMPTY selection (just press ENTER) skips that player
    after both players: any key moves on; ESC at any time = save & quit
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import ground_truth_path

DISPLAY_W = 1280  # display width; clicks are divided back to source coords

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
    """Resize for display and return (resized_frame, scale). Never upscales."""
    h, w = frame.shape[:2]
    # Cap scale at 1.0 so a frame narrower than DISPLAY_W stays at native res.
    s = min(1.0, DISPLAY_W / w)
    return cv2.resize(frame, (int(w * s), int(h * s))), s


# ── boxes mode ─────────────────────────────────────────────────────────────────

def annotate_boxes(video, step, start, output):
    """Draw P1 (near) and P2 (far) boxes every `step` frames; write a GT CSV."""
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
            # Zero-area roi == "skip this player". selectROI returns (0,0,0,0)
            # for both plain ENTER and ESC (and a one-axis drag gives w or h==0),
            # with no way to tell them apart, so we fall through to the prompt.
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
            x, y, w, h = (int(round(c / s)) for c in roi)  # back to source coords
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
    """Click the 8 court keypoints on one frame; write label,x,y CSV."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_no}")

    disp, s = _scaled(frame)
    points = []   # collected as (label, x_source, y_source)

    def on_mouse(event, x, y, flags, param):
        # Record next keypoint at source-resolution coords; ignore extra clicks.
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
        elif key in (13, 10) and len(points) == len(COURT_LABELS):  # CR/LF = save
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


# ── ball mode ──────────────────────────────────────────────────────────────────

MAG_SRC = 40      # half-size (display px) of the magnified area around the cursor
MAG_ZOOM = 4      # magnification factor of the inset


def _draw_magnifier(view, disp, pos):
    """Draw a magnified inset of `disp` around the cursor `pos` on `view`."""
    if pos is None:
        return
    mx, my = pos
    h, w = disp.shape[:2]
    x0, y0 = max(0, mx - MAG_SRC), max(0, my - MAG_SRC)
    x1, y1 = min(w, mx + MAG_SRC), min(h, my + MAG_SRC)
    if x1 > x0 and y1 > y0:
        patch = cv2.resize(disp[y0:y1, x0:x1], None, fx=MAG_ZOOM, fy=MAG_ZOOM,
                           interpolation=cv2.INTER_NEAREST)
        cx, cy = (mx - x0) * MAG_ZOOM, (my - y0) * MAG_ZOOM
        cv2.line(patch, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 1)
        cv2.line(patch, (cx, cy - 12), (cx, cy + 12), (0, 255, 0), 1)
        ph, pw = patch.shape[:2]
        view[40:40 + ph, w - pw - 10:w - 10] = patch
        cv2.rectangle(view, (w - pw - 10, 40), (w - 10, 40 + ph), (0, 255, 0), 1)


def _load_ball_gt(path):
    """{frame: (visible, x, y)} from an existing ball GT CSV (to resume)."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vis = int(row["visible"])
            out[int(row["frame"])] = (vis, float(row["x"]) if vis else None,
                                      float(row["y"]) if vis else None)
    return out


def _save_ball_gt(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "visible", "x", "y"])
        for fr in sorted(labels):
            vis, x, y = labels[fr]
            wr.writerow([fr, vis, f"{x:.1f}" if vis else "", f"{y:.1f}" if vis else ""])


def annotate_ball(video, step, start, output, skip=()):
    """Click the ball centre on every `step`-th frame; `n` = ball not visible.

    skip: frames not to offer (e.g. those the tracker already found, see
    tracked_frames), so the labels go where the current detector fails.

    A magnified inset follows the cursor, since the ball is only ~10 px wide on
    screen. Annotations are saved on quit and on every 10 frames, and an
    existing output file is resumed (already labelled frames are skipped).
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = set(skip)
    frames = [f for f in range(start, n_total) if f not in skip][::step]
    labels = _load_ball_gt(output)
    todo = [f for f in frames if f not in labels]
    print(f"{len(frames)} frames to label ({len(frames) - len(todo)} already done).")
    print("Click the ball centre, then ENTER/SPACE to confirm (click again to adjust).")
    print("n = ball not visible   u = back to the previous frame   q = save and quit\n")

    win = "annotate ball"
    cv2.namedWindow(win)
    mouse = {"pos": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        mouse["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)

    cv2.setMouseCallback(win, on_mouse)
    history = []          # frames labelled in this session, for undo
    i = 0
    while i < len(todo):
        fr = todo[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, frame = cap.read()
        if not ok:
            break
        disp, s = _scaled(frame)
        mouse["click"] = None
        action = None
        while action is None:
            view = disp.copy()
            if mouse["click"] is not None:
                cv2.circle(view, mouse["click"], 7, (0, 0, 255), 1)
            done = len(frames) - len(todo) + i
            cv2.putText(view, f"frame {fr}  ({done + 1}/{len(frames)})  "
                              "click ball, ENTER = ok, n = not visible, u = back, q = quit",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            _draw_magnifier(view, disp, mouse["pos"])
            cv2.imshow(win, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32) and mouse["click"] is not None:
                cx, cy = mouse["click"]
                labels[fr] = (1, cx / s, cy / s)
                action = "next"
            elif key == ord("n"):
                labels[fr] = (0, None, None)
                action = "next"
            elif key == ord("u") and history:
                prev = history.pop()
                labels.pop(prev, None)
                i = todo.index(prev)
                action = "back"
            elif key == ord("q"):
                action = "quit"
        if action == "quit":
            break
        if action == "next":
            history.append(fr)
            i += 1
            if len(history) % 10 == 0:
                _save_ball_gt(output, labels)

    cap.release()
    cv2.destroyAllWindows()
    _save_ball_gt(output, labels)
    n_vis = sum(1 for v, _, _ in labels.values() if v)
    print(f"\nSaved {len(labels)} labelled frames ({n_vis} with the ball visible) to {output}")


def tracked_frames(ball_csv):
    """Frames with a real (not interpolated) detection in a ball tracker CSV."""
    with open(ball_csv, newline="") as f:
        return {int(r["frame"]) for r in csv.DictReader(f)
                if not int(r.get("interpolated") or 0) and r.get("cx")}


# ── bounce mode ────────────────────────────────────────────────────────────────

def annotate_bounce(video, start, output):
    """Mark every frame where the ball touches the ground.

    Move with a/d (one frame) and s/w (ten frames). Click the ball and press ENTER
    on the contact frame, or h if the contact is hidden (by a player, say) and only
    its frame can be estimated. x removes the mark on this frame, q saves and quits.
    An existing output file is resumed; the file is the same format as ball mode.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    labels = _load_ball_gt(output)
    print(f"{len(labels)} bounces already marked.")
    print("a/d = -1/+1 frame   s/w = -10/+10 frames")
    print("click the ball, then ENTER = bounce here   h = bounce here, but hidden")
    print("x = remove the mark on this frame   q = save and quit\n")

    win = "annotate bounce"
    cv2.namedWindow(win)
    mouse = {"pos": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        mouse["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)

    cv2.setMouseCallback(win, on_mouse)
    moves = {ord("a"): -1, ord("d"): 1, ord("s"): -10, ord("w"): 10}
    fr = min(start, n_total - 1)
    shown = None
    while True:
        if fr != shown:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, frame = cap.read()
            if not ok:
                break
            disp, s = _scaled(frame)
            mouse["click"] = None
            shown = fr
        view = disp.copy()
        if mouse["click"] is not None:
            cv2.circle(view, mouse["click"], 7, (0, 0, 255), 1)
        if fr in labels:
            vis, x, y = labels[fr]
            if vis:
                cv2.circle(view, (int(x * s), int(y * s)), 9, (0, 255, 0), 2)
        state = f"BOUNCE ({'hidden' if not labels[fr][0] else 'marked'})" if fr in labels else ""
        cv2.putText(view, f"frame {fr}/{n_total - 1}  marked {len(labels)}  {state}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        _draw_magnifier(view, disp, mouse["pos"])
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key in moves:
            fr = min(max(fr + moves[key], 0), n_total - 1)
        elif key in (13, 10, 32) and mouse["click"] is not None:
            cx, cy = mouse["click"]
            labels[fr] = (1, cx / s, cy / s)
            mouse["click"] = None
            _save_ball_gt(output, labels)
        elif key == ord("h"):
            labels[fr] = (0, None, None)
            _save_ball_gt(output, labels)
        elif key == ord("x"):
            labels.pop(fr, None)
            _save_ball_gt(output, labels)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    _save_ball_gt(output, labels)
    n_vis = sum(1 for v, _, _ in labels.values() if v)
    print(f"\nSaved {len(labels)} bounces ({n_vis} visible) to {output}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    """Dispatch to the boxes or court annotator per the CLI subcommand."""
    parser = argparse.ArgumentParser(description="Ground-truth annotation tool")
    sub = parser.add_subparsers(dest="mode", required=True)

    pb = sub.add_parser("boxes", help="annotate player bounding boxes")
    pb.add_argument("--video", default="data/Input_video2.mp4")
    pb.add_argument("--step",  type=int, default=30,
                    help="annotate every N-th frame (default: 30)")
    pb.add_argument("--start", type=int, default=30,
                    help="first frame to annotate (default: 30, after the "
                         "background warm-up)")
    pb.add_argument("--output", default=None,
                    help="output CSV (default: ground_truth/players_<video name>_gt.csv)")

    pc = sub.add_parser("court", help="annotate court keypoints")
    pc.add_argument("--video", default="data/Input_video2.mp4")
    pc.add_argument("--frame", type=int, default=0,
                    help="frame to annotate (default: 0)")
    pc.add_argument("--output", default=None,
                    help="output CSV (default: ground_truth/court_<video name>_gt.csv)")

    pl = sub.add_parser("ball", help="annotate the ball centre on sampled frames")
    pl.add_argument("--video", default="data/Input_video2.mp4")
    pl.add_argument("--step", type=int, default=10,
                    help="annotate every N-th frame (default: 10)")
    pl.add_argument("--start", type=int, default=0, help="first frame (default: 0)")
    pl.add_argument("--output", default=None,
                    help="output CSV (default: ground_truth/ball_<video name>_gt.csv)")
    pl.add_argument("--frames-from", default=None, dest="frames_from",
                    help="ball tracker CSV: offer only the frames it did not find the ball "
                         "on (--step then counts among those), to label for training")

    pn = sub.add_parser("bounce", help="mark the frames where the ball touches the ground")
    pn.add_argument("--video", default="data/Input_video2.mp4")
    pn.add_argument("--start", type=int, default=0, help="first frame (default: 0)")
    pn.add_argument("--output", default=None,
                    help="output CSV (default: ground_truth/bounce_<video name>_gt.csv)")

    args = parser.parse_args()
    kinds = {"boxes": "players", "court": "court", "ball": "ball", "bounce": "bounce"}
    output = args.output or ground_truth_path(kinds[args.mode], args.video)
    if args.mode == "boxes":
        annotate_boxes(args.video, args.step, args.start, output)
    elif args.mode == "ball":
        skip = tracked_frames(args.frames_from) if args.frames_from else ()
        annotate_ball(args.video, args.step, args.start, output, skip)
    elif args.mode == "bounce":
        annotate_bounce(args.video, args.start, output)
    else:
        annotate_court(args.video, args.frame, output)


if __name__ == "__main__":
    main()
