"""Run the whole analysis in real time: one causal pass over the frames.

The offline pipeline (pipeline.py) reads the clip many times and looks at the
whole of it; this one sees each frame once, in order, as a camera would give
them, and keeps up with the video when the hardware allows it. See
realtime/pipeline.py for the stages and README.md for the measured speeds.

Usage (from project root):
    python realtime_pipeline.py --video data/Input_video2.mp4 --model models/ball_motion.onnx
    python realtime_pipeline.py --video data/Input_video2.mp4 --calibrate     # from frame 0
    python realtime_pipeline.py --video data/Input_video2.mp4 --clock live --display
    python realtime_pipeline.py --video data/input3.mp4 --model diff        # no model
    # ball only, with the offline players and court:
    python realtime_pipeline.py --video data/Input_video2.mp4 --model models/ball_motion.onnx \\
        --court outputs/Input_video2/court_coordinates/Input_video2_court.csv \\
        --players outputs/Input_video2/player_coordinates/players_Input_video2.csv
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime.pipeline import RealtimePipeline

DEFAULT_MODEL = os.path.join("models", "ball_motion.onnx")


def build_parser():
    p = argparse.ArgumentParser(description="Causal, single-pass tennis analysis")
    p.add_argument("--video", default="data/Input_video2.mp4")
    p.add_argument("--output", default=os.path.join("outputs", "realtime"),
                   help="base output folder (default: outputs/realtime, one folder per clip)")
    p.add_argument("--model", default=None,
                   help="motion ball detector: .onnx, .pt or TensorRT .engine; 'diff' runs "
                        f"without a model (default: {DEFAULT_MODEL} if present, else diff)")
    p.add_argument("--fallback", default=None,
                   help="colour ball detector (e.g. models/ball_tracker.pt or its .onnx), run "
                        "only by the players where the motion model finds nothing; heavy "
                        "(YOLOv8x): meant for a GPU")
    p.add_argument("--device", default=None, help="'cpu' to keep the model off the GPU")
    p.add_argument("--court", default=None, help="known court CSV: skips the startup fit")
    p.add_argument("--background", default=None,
                   help="known background PNG (full resolution): skips the 2 s warm-up")
    p.add_argument("--calibrate", action="store_true",
                   help="calibrate the camera first, as before a match: court and median "
                        "background of the clip, saved in <output>/<clip>/calibration/ and "
                        "reused next time; the analysis then tracks from the first frame")
    p.add_argument("--recalibrate", action="store_true",
                   help="like --calibrate, recomputing the calibration")
    p.add_argument("--players", default=None,
                   help="offline players CSV: tracks the ball only (needs --court; no shots)")
    p.add_argument("--clock", choices=["all", "live"], default="all",
                   help="all: every frame, as fast as possible (throughput); live: at the "
                        "video's frame rate, dropping frames when behind")
    p.add_argument("--search", choices=["tiles", "blobs"], default=None,
                   help="ball search when lost: every court tile (GPU) or the moving blobs "
                        "only (CPU); default from the hardware")
    p.add_argument("--search-every", type=int, default=None, dest="search_every",
                   help="search on one frame in this many (default 1 on GPU, 2 on CPU)")
    p.add_argument("--display", action="store_true", help="show the live overlay")
    p.add_argument("--video-out", default=None, dest="video_out",
                   help="write the overlay to this video file")
    p.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    p.add_argument("--no-shots", action="store_true", dest="no_shots",
                   help="skip the online bounce and shot analysis")
    p.add_argument("--figures", action="store_true",
                   help="at the end, also draw the player analysis figures")
    p.add_argument("--log-every", type=int, default=150, dest="log_every",
                   help="print progress every this many frames (0 = never)")
    p.add_argument("--p1-hand", default="right", dest="p1_hand", choices=["right", "left"])
    p.add_argument("--p2-hand", default="right", dest="p2_hand", choices=["right", "left"])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")
    model = args.model
    if model is None:
        model = DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else "diff"
    if args.calibrate or args.recalibrate:
        from realtime.calibration import calibrate
        cal = calibrate(args.video, args.output, force=args.recalibrate)
        args.court = args.court or cal["court_csv"]
        args.background = args.background or cal["background"]
    if args.players and not args.court:
        raise SystemExit("--players needs --court")
    return RealtimePipeline(
        args.video, args.output, model, court_csv=args.court, players_csv=args.players,
        clock=args.clock, search=args.search, search_every=args.search_every,
        display=args.display, video_out=args.video_out, max_frames=args.max_frames,
        hands=(args.p1_hand, args.p2_hand), shots=not args.no_shots, figures=args.figures,
        device=args.device, log_every=args.log_every, background=args.background,
        fallback=args.fallback).run()


if __name__ == "__main__":
    main()
