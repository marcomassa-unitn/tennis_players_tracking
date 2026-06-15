#!/usr/bin/env python3
"""Unified entry point for the tennis players tracking pipeline.

Runs the full chain end-to-end from a single command:

    court detection -> player tracking -> player analysis ->
    motion estimation -> ball tracking -> shot analysis -> evaluation

All intermediate paths are derived from --video and --output, so a typical
run is simply:

    python pipeline.py --video data/Input_video2.mp4 --no-display

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
# package directories (courtTracking/, playerTracking/, utils/, ...) resolve as
# namespace packages regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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


# --------------------------------------------------------------------------- #
# Path derivation
# --------------------------------------------------------------------------- #
def derive_paths(video, output):
    """Derive every intermediate artifact path from the video and output dir."""
    output = output.rstrip("/\\") or "outputs"
    stem = os.path.splitext(os.path.basename(video))[0]          # e.g. Input_video2
    # The historical naming uses "clip2" for the Input_video2 sample; keep that
    # so the produced CSVs line up with each module's documented defaults.
    key = "clip2" if stem == "Input_video2" else stem.lower()

    return {
        "video": video,
        "output": output,
        "stem": stem,
        "key": key,
        "court_csv": os.path.join(output, "court_coordinates", f"{stem}_court.csv"),
        "players_csv": os.path.join(output, f"players_{key}.csv"),
        "ball_csv": os.path.join(output, f"ball_{key}.csv"),
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

    from courtTracking.court_tracking import CourtTracker
    # Court detection finishes (and writes the CSV) before the interactive
    # playback loop in run(); that loop is a standalone-CLI visualization only,
    # so we always run it headless here -- it would otherwise block the pipeline
    # waiting for a keypress (and can exhaust memory on long videos).
    keypoints = CourtTracker(
        video_path=p["video"],
        no_display=True,
        roi_top=args.roi_top,
        far_line=args.far_line,
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

    from playerTracking.playerTracking import PlayerTracker
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
    from motionEstimation import optical_flow
    of_argv = [
        "optical_flow",
        "--video", p["video"],
        "--players", p["players_csv"],
        "--court", p["court_csv"],
        "--fps", str(args.fps),
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
        from ballTracking.BallTracking import BallTracker
    except ImportError as exc:
        print(f"Skipped: ball tracking dependencies unavailable ({exc}). "
              "Install ultralytics to enable this step.")
        return

    model_path = os.path.join(PROJECT_ROOT, "ball_tracker.pt")
    if not os.path.exists(model_path):
        print(f"Skipped: model weights not found at {model_path}. "
              "Place the YOLO ball-tracker weights there to enable this step.")
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
    parser.add_argument("--fps", type=float, default=30.0,
                        help="video frame rate")
    parser.add_argument("--no-display", action="store_true", dest="no_display",
                        help="suppress all OpenCV windows / batch mode")

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

    # Analysis / shots tuning.
    parser.add_argument("--anchor", default="feet",
                        help="player anchor point for analysis (passed to player_analysis)")
    parser.add_argument("--p1-hand", default="right", dest="p1_hand",
                        help="player 1 dominant hand (passed to shot_analysis)")
    parser.add_argument("--p2-hand", default="right", dest="p2_hand",
                        help="player 2 dominant hand (passed to shot_analysis)")

    # Evaluation ground truth.
    parser.add_argument("--gt-csv", default=None, dest="gt_csv",
                        help="player ground-truth CSV; evaluation only runs when provided")
    parser.add_argument("--court-gt-csv", default=None, dest="court_gt_csv",
                        help="optional court ground-truth CSV for evaluation")
    return parser


def main():
    args = build_parser().parse_args()
    p = derive_paths(args.video, args.output)
    produced = {}

    print("Tennis players tracking pipeline")
    print(f"  video : {p['video']}")
    print(f"  output: {p['output']}")

    step_court(p, args, produced)
    step_tracking(p, args, produced)
    step_analysis(p, args, produced)
    step_motion(p, args, produced)
    step_ball(p, args, produced)
    step_shots(p, args, produced)
    step_evaluation(p, args, produced)

    print("\n=== Pipeline complete ===")
    if produced:
        print("Artifacts produced / reused:")
        for label, path in produced.items():
            print(f"  - {label:12s}: {path}")
    else:
        print("No artifacts were produced (all steps skipped?).")


if __name__ == "__main__":
    main()
