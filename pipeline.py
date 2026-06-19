#!/usr/bin/env python3
"""Unified entry point for the tennis players tracking pipeline.

Runs the full chain end-to-end from a single command:

    court detection -> player tracking -> player analysis ->
    motion estimation -> ball tracking -> shot analysis -> evaluation

All intermediate paths are derived from --video and --output, so a typical
run is simply:

    python pipeline.py --video data/Input_video2.mp4

The pipeline is HEADLESS BY DEFAULT (no OpenCV windows / batch mode). Pass
--display to enable interactive windows; --no-display is still accepted for
backward compatibility and forces headless. Each step is wrapped in its own
try/except, so one failing step (a missing model, a bad path) is reported and
skipped rather than aborting the whole run; a summary at the end lists which
steps succeeded, were skipped, or failed.

Each step can be skipped with its --skip-* flag (reusing any CSV already on
disk). Steps are run by importing and calling the underlying module classes
and functions directly -- no subprocess. The class-based modules (court,
player, ball trackers) are instantiated; the function-based modules
(player_analysis, optical_flow, block_matching, shot_analysis) are driven by
temporarily overriding sys.argv and calling their unchanged main(), so every
per-module CLI keeps working exactly as documented in the README.
"""

import argparse
import contextlib
import os
import sys

# Project root = directory of this file. Ensure it is importable so that the
# package directories (tracking/, utils/, ...) resolve as
# namespace packages regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "ball_tracker.pt")


@contextlib.contextmanager
def _override_argv(argv):
    """Temporarily replace sys.argv with `argv` for a module main() call."""
    saved = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = saved


def _ensure_parent(path):
    """Make sure the parent directory of `path` exists."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resolve_fps(video):
    """Read the real frame rate from `video`, falling back to 30.0.

    Imports cv2/numpy lazily so the pipeline can still be parsed/--help'd in
    environments without OpenCV installed.
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
    """Derive every intermediate artifact path from the video and output dir."""
    output = output.rstrip("/\\") or "outputs"
    stem = os.path.splitext(os.path.basename(video))[0]          # e.g. Input_video2

    # Every artifact is named after the video stem and lives in its own
    # per-modality subfolder, matching each module's standalone default:
    #   playerTracking -> outputs/player_coordinates/players_<stem>.csv
    #   BallTracking   -> outputs/ball_coordinates/ball_<stem>.csv
    #   court_tracking -> outputs/court_coordinates/<stem>_court.csv
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
    # Court detection finishes (and writes the CSV) before the interactive
    # playback loop in run(); that loop is a standalone-CLI visualization only,
    # so we always run it headless here -- it would otherwise block the pipeline
    # waiting for a keypress (and can exhaust memory on long videos).
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
        # Skip the (slow) minimap animation when running headless/batch.
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
    # Dense optical flow is expensive; cap the number of frames so a long video
    # doesn't dominate the whole run. --flow-frames 0 means the whole video.
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
    # Ball tracking is expensive: if it has already been run, reuse the CSV
    # instead of recomputing it.
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
        output_path=None,                 # CSV only in pipeline mode
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
    # shot_analysis treats --ball as required and exits (SystemExit) if the file
    # is missing; ball tracking may have been skipped (no weights / no
    # ultralytics), so guard here instead of passing a nonexistent path.
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
        # Don't dump per-shot frames when running headless/batch.
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
    # Headless by default. --display turns windows on; --no-display is kept for
    # backward compatibility and forces headless (see main()).
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


# Ordered (name, function) list so main() can drive + report on every step.
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

    # Headless by default: with no flags args.display is False -> no_display
    # True. --display turns windows on; --no-display always forces headless.
    args.no_display = args.no_display or (not args.display)

    # Resolve fps from the source video unless the user gave an explicit value.
    if args.fps is None:
        args.fps = _resolve_fps(args.video)

    p = derive_paths(args.video, args.output)
    produced = {}

    print("Tennis players tracking pipeline")
    print(f"  video : {p['video']}")
    print(f"  output: {p['output']}")
    print(f"  fps   : {args.fps}")
    print(f"  mode  : {'headless' if args.no_display else 'display'}")

    # Run each step in isolation: a failure (including SystemExit raised by a
    # sub-module's argparse) is reported and recorded, then we move on. One bad
    # step must not abort the whole pipeline. A step that runs without error but
    # records no artifact (e.g. a --skip-* or a missing-input early return) is
    # classified as "skipped"; one that records/ reuses its artifact "succeeded".
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

    # Per-step status. The `produced` map below is the accurate record of the
    # real artifacts (only populated when a step actually ran / reused / wrote).
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
