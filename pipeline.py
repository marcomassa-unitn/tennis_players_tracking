#!/usr/bin/env python3
"""Single-command driver for the full tennis-tracking chain.

Stages, in order:

    court detection -> player tracking -> player analysis ->
    motion estimation -> ball tracking -> shot analysis -> evaluation

All intermediate paths derive from --video and --output, so a typical run is
just `python pipeline.py --video data/Input_video2.mp4`.

Headless by default (no OpenCV windows). --display enables interactive
windows; --no-display is kept for backward compatibility and also forces
headless. Every step runs in its own try/except so one failure (missing
model, bad path) is reported and skipped, never aborting the run; the end
summary lists succeeded/skipped/failed steps.

Any step is skippable via --skip-* (reusing a CSV already on disk). Steps
call the underlying module code directly -- no subprocess. Class-based
modules (court/player/ball trackers) are instantiated; function-based modules
(player_analysis, optical_flow, block_matching, shot_analysis) run via their
unchanged main() under a temporarily overridden sys.argv, so each per-module
CLI behaves exactly as the README documents.
"""

import argparse
import contextlib
import os
import sys

# Put this file's directory on sys.path so tracking/, utils/, ... resolve as
# namespace packages regardless of the cwd.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "ball_tracker.pt")


@contextlib.contextmanager
def _override_argv(argv):
    """Swap sys.argv to `argv` for the duration of a module main() call."""
    saved = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = saved


def _ensure_parent(path):
    """Create the parent directory of `path` if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resolve_fps(video):
    """Frame rate of `video`, or 30.0 if unreadable/invalid.

    cv2/numpy are imported lazily so --help works without OpenCV installed.
    """
    fps = None
    try:
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(video)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if not fps or not np.isfinite(fps) or fps <= 0:
            fps = 30.0
    except Exception:
        fps = 30.0
    return float(fps)


# --------------------------------------------------------------------------- #
# Path derivation
# --------------------------------------------------------------------------- #
def derive_paths(video, output):
    """Map of every intermediate artifact path, keyed off the video stem."""
    output = output.rstrip("/\\") or "outputs"
    stem = os.path.splitext(os.path.basename(video))[0]          # e.g. Input_video2

    # Each artifact is named after the stem in its own per-modality subfolder,
    # matching each module's standalone default (e.g. player_coordinates/,
    # ball_coordinates/, court_coordinates/).
    return {
        "video": video,
        "output": output,
        "stem": stem,
        "court_csv": os.path.join(output, "court_coordinates", f"{stem}_court.csv"),
        "players_csv": os.path.join(output, "player_coordinates", f"players_{stem}.csv"),
        "ball_csv": os.path.join(output, "ball_coordinates", f"ball_{stem}.csv"),
        "analysis_dir": os.path.join(output, "player_analysis"),
        "motion_dir": os.path.join(output, "motion_estimation"),
        "shots_dir": os.path.join(output, "shot_analysis"),
        "eval_dir": os.path.join(output, "evaluation"),
    }


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
def step_court(p, args, produced):
    print("\n=== Step 1: Court Detection ===")
    if args.skip_court:
        if os.path.exists(p["court_csv"]):
            print(f"Skipped (reusing existing {p['court_csv']}).")
            produced["court_csv"] = p["court_csv"]
        else:
            print(f"Skipped, but no existing court CSV at {p['court_csv']}. "
                  "Downstream steps that need it may fail.")
        return

    from tracking.court_tracking import CourtTracker
    # Force headless: run()'s trailing playback loop is standalone-CLI viz only,
    # and would otherwise block on a keypress (and can OOM on long videos). The
    # CSV is already written before that loop.
    keypoints = CourtTracker(
        video_path=p["video"],
        no_display=True,
        roi_top=args.roi_top,
        far_line=args.far_line,
        output_dir=p["output"],
    ).run()
    print(f"Detected {len(keypoints)} court keypoints -> {p['court_csv']}")
    produced["court_csv"] = p["court_csv"]


def step_tracking(p, args, produced):
    print("\n=== Step 2: Player Tracking ===")
    if args.skip_tracking and os.path.exists(p["players_csv"]):
        print(f"Skipped (reusing existing {p['players_csv']}).")
        produced["players_csv"] = p["players_csv"]
        return
    if args.skip_tracking:
        print(f"Skipped, but no existing players CSV at {p['players_csv']}. "
              "Downstream steps that need it may fail.")
        return

    from tracking.playerTracking import PlayerTracker
    _ensure_parent(p["players_csv"])
    PlayerTracker(p["video"], p["players_csv"], display=not args.no_display).run()
    print(f"Player tracks -> {p['players_csv']}")
    produced["players_csv"] = p["players_csv"]


def step_analysis(p, args, produced):
    print("\n=== Step 3: Player Analysis ===")
    if args.skip_analysis:
        print("Skipped (--skip-analysis).")
        return
    if not os.path.exists(p["players_csv"]):
        print(f"Skipped: players CSV not found ({p['players_csv']}).")
        return

    from utils import player_analysis
    os.makedirs(p["analysis_dir"], exist_ok=True)
    argv = [
        "player_analysis",
        "--players", p["players_csv"],
        "--court", p["court_csv"],
        "--fps", str(args.fps),
        "--output", p["analysis_dir"],
        "--anchor", args.anchor,
    ]
    if args.no_display:
        # Minimap animation is slow; drop it in batch mode.
        argv.append("--no-animation")
    with _override_argv(argv):
        player_analysis.main()
    print(f"Player analysis -> {p['analysis_dir']}")
    produced["analysis_dir"] = p["analysis_dir"]


def step_motion(p, args, produced):
    print("\n=== Step 4: Motion Estimation ===")
    if args.skip_motion:
        print("Skipped (--skip-motion).")
        return
    if not os.path.exists(p["players_csv"]):
        print(f"Skipped: players CSV not found ({p['players_csv']}).")
        return

    os.makedirs(p["motion_dir"], exist_ok=True)

    # --- Optical flow ---
    # Dense flow is expensive; --flow-frames caps the frame count (0 = whole
    # video) so a long clip doesn't dominate the run.
    from motionEstimation import optical_flow
    of_argv = [
        "optical_flow",
        "--video", p["video"],
        "--players", p["players_csv"],
        "--court", p["court_csv"],
        "--fps", str(args.fps),
        "--frames", str(args.flow_frames),
        "--output", p["motion_dir"],
    ]
    with _override_argv(of_argv):
        optical_flow.main()
    print(f"Optical flow -> {p['motion_dir']}")

    # --- Block matching ---
    from motionEstimation import block_matching
    bm_argv = [
        "block_matching",
        "--video", p["video"],
        "--output", p["motion_dir"],
    ]
    with _override_argv(bm_argv):
        block_matching.main()
    print(f"Block matching -> {p['motion_dir']}")
    produced["motion_dir"] = p["motion_dir"]


def step_ball(p, args, produced):
    print("\n=== Step 5: Ball Tracking ===")
    if args.skip_ball:
        print("Skipped (--skip-ball).")
        return
    # Expensive: reuse an existing CSV rather than recompute.
    if os.path.exists(p["ball_csv"]):
        print(f"Already done; reusing existing {p['ball_csv']}.")
        produced["ball_csv"] = p["ball_csv"]
        return

    try:
        from tracking.BallTracking import BallTracker
    except ImportError as exc:
        print(f"Skipped: ball tracking dependencies unavailable ({exc}). "
              "Install ultralytics to enable this step.")
        return

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Skipped: model weights not found at {model_path}. "
              "Place the YOLO ball-tracker weights there (or pass --model) to "
              "enable this step.")
        return

    _ensure_parent(p["ball_csv"])
    tracker = BallTracker(model_path)
    tracker.run(
        p["video"],
        output_path=None,                 # no annotated video; CSV only here
        csv_path=p["ball_csv"],
        display=not args.no_display,
    )
    print(f"Ball tracks -> {p['ball_csv']}")
    produced["ball_csv"] = p["ball_csv"]


def step_shots(p, args, produced):
    print("\n=== Step 6: Shot Analysis ===")
    if args.skip_shots:
        print("Skipped (--skip-shots).")
        return
    if not os.path.exists(p["players_csv"]):
        print(f"Skipped: players CSV not found ({p['players_csv']}).")
        return
    # shot_analysis requires --ball and SystemExits without it; ball tracking
    # may have been skipped, so guard rather than pass a nonexistent path.
    if not os.path.exists(p["ball_csv"]):
        print("Skipped: ball CSV not found (run ball tracking first).")
        return

    from utils import shot_analysis
    os.makedirs(p["shots_dir"], exist_ok=True)
    argv = [
        "shot_analysis",
        "--ball", p["ball_csv"],
        "--players", p["players_csv"],
        "--court", p["court_csv"],
        "--video", p["video"],
        "--fps", str(args.fps),
        "--p1-hand", args.p1_hand,
        "--p2-hand", args.p2_hand,
        "--output", p["shots_dir"],
    ]
    if args.no_display:
        # Skip per-shot frame dumps in batch mode.
        argv.append("--no-frames")
    with _override_argv(argv):
        shot_analysis.main()
    print(f"Shot analysis -> {p['shots_dir']}")
    produced["shots_dir"] = p["shots_dir"]


def step_evaluation(p, args, produced):
    print("\n=== Step 7: Evaluation ===")
    if args.skip_evaluation:
        print("Skipped (--skip-evaluation).")
        return
    if not args.gt_csv:
        print("Skipped: no --gt-csv ground truth provided.")
        return
    if not os.path.exists(args.gt_csv):
        print(f"Skipped: ground truth CSV not found ({args.gt_csv}).")
        return

    from evaluation import evaluate_tracking
    os.makedirs(p["eval_dir"], exist_ok=True)
    argv = [
        "evaluate_tracking",
        "--gt", args.gt_csv,
        "--pred", p["players_csv"],
        "--court", p["court_csv"],
        "--court-pred", p["court_csv"],
        "--output", p["eval_dir"],
    ]
    if args.court_gt_csv:
        argv += ["--court-gt", args.court_gt_csv]
    with _override_argv(argv):
        evaluate_tracking.main()
    print(f"Evaluation -> {p['eval_dir']}")
    produced["eval_dir"] = p["eval_dir"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified tennis players tracking pipeline.")
    parser.add_argument("--video", default="data/Input_video2.mp4",
                        help="input video")
    parser.add_argument("--output", default="outputs/",
                        help="base output directory")
    parser.add_argument("--fps", type=float, default=None,
                        help="video frame rate (default: auto-read from the "
                             "input video, falling back to 30)")
    # Headless by default. --display turns windows on; --no-display forces
    # headless (kept for backward compatibility -- see main()).
    parser.add_argument("--display", action="store_true", dest="display",
                        help="enable interactive OpenCV windows (default: headless)")
    parser.add_argument("--no-display", action="store_true", dest="no_display",
                        help="force headless / batch mode (default; kept for "
                             "backward compatibility)")

    # Step skips.
    parser.add_argument("--skip-court", action="store_true", dest="skip_court",
                        help="skip court detection (reuse existing CSV if present)")
    parser.add_argument("--skip-tracking", action="store_true", dest="skip_tracking",
                        help="skip player tracking (reuse existing CSV)")
    parser.add_argument("--skip-analysis", action="store_true", dest="skip_analysis",
                        help="skip player analysis (heatmaps, zones, speeds)")
    parser.add_argument("--skip-motion", action="store_true", dest="skip_motion",
                        help="skip optical flow + block matching")
    parser.add_argument("--skip-ball", action="store_true", dest="skip_ball",
                        help="skip ball tracking (requires ultralytics + weights)")
    parser.add_argument("--skip-shots", action="store_true", dest="skip_shots",
                        help="skip shot analysis")
    parser.add_argument("--skip-evaluation", action="store_true", dest="skip_evaluation",
                        help="skip evaluation (requires ground truth CSVs)")

    # Court tuning (forwarded to CourtTracker).
    parser.add_argument("--roi-top", type=float, default=0.15, dest="roi_top",
                        help="top ROI fraction for court detection")
    parser.add_argument("--far-line", choices=["baseline", "service"],
                        default="baseline", dest="far_line",
                        help="far court line used for keypoints")

    # Motion estimation tuning.
    parser.add_argument("--flow-frames", type=int, default=200, dest="flow_frames",
                        help="max frames for dense optical flow (0 = whole video; "
                             "default 200 so long videos don't dominate runtime)")

    # Analysis / shots tuning.
    parser.add_argument("--anchor", choices=["feet", "centroid"], default="feet",
                        help="player anchor point for analysis (passed to player_analysis)")
    parser.add_argument("--p1-hand", default="right", dest="p1_hand",
                        help="player 1 dominant hand (passed to shot_analysis)")
    parser.add_argument("--p2-hand", default="right", dest="p2_hand",
                        help="player 2 dominant hand (passed to shot_analysis)")

    # Ball tracking weights.
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="YOLO ball-tracker weights (passed to BallTracker)")

    # Evaluation ground truth.
    parser.add_argument("--gt-csv", default=None, dest="gt_csv",
                        help="player ground-truth CSV; evaluation only runs when provided")
    parser.add_argument("--court-gt-csv", default=None, dest="court_gt_csv",
                        help="optional court ground-truth CSV for evaluation")
    return parser


# Execution order; main() iterates this to run and report on each step.
STEPS = [
    ("court", step_court),
    ("tracking", step_tracking),
    ("analysis", step_analysis),
    ("motion", step_motion),
    ("ball", step_ball),
    ("shots", step_shots),
    ("evaluation", step_evaluation),
]


def main():
    args = build_parser().parse_args()

    # No flags -> display False -> headless. --display opts in; --no-display
    # always forces headless.
    args.no_display = args.no_display or (not args.display)

    # Auto-detect fps unless explicitly given.
    if args.fps is None:
        args.fps = _resolve_fps(args.video)

    p = derive_paths(args.video, args.output)
    produced = {}

    print("Tennis players tracking pipeline")
    print(f"  video : {p['video']}")
    print(f"  output: {p['output']}")
    print(f"  fps   : {args.fps}")
    print(f"  mode  : {'headless' if args.no_display else 'display'}")

    # Each step runs in isolation: any failure (including SystemExit from a
    # sub-module's argparse) is recorded and we continue. A step that returns
    # without recording an artifact (--skip-* or missing-input early return)
    # counts as "skipped"; one that records/reuses an artifact "succeeded".
    succeeded, skipped, failed = [], [], []
    for idx, (name, func) in enumerate(STEPS, start=1):
        before = set(produced)
        try:
            func(p, args, produced)
        except (Exception, SystemExit) as exc:
            print(f"  [step {idx}] FAILED: {exc}")
            failed.append(name)
        else:
            if set(produced) - before:
                succeeded.append(name)
            else:
                skipped.append(name)

    print("\n=== Pipeline complete ===")

    # `produced` is the source of truth for real artifacts -- only populated
    # when a step actually ran, reused, or wrote one.
    print("Steps:")
    print(f"  succeeded: {', '.join(succeeded) if succeeded else '(none)'}")
    print(f"  skipped  : {', '.join(skipped) if skipped else '(none)'}")
    print(f"  failed   : {', '.join(failed) if failed else '(none)'}")

    if produced:
        print("Artifacts produced / reused:")
        for label, path in produced.items():
            print(f"  - {label:12s}: {path}")
    else:
        print("No artifacts were produced (all steps skipped or failed?).")


if __name__ == "__main__":
    main()
