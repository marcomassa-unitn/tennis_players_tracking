"""Where everything about a clip lives: one folder per video under outputs/.

    outputs/<video name>/
        court_coordinates/    <video name>_court.csv, <video name>_court.png
        player_coordinates/   players_<video name>.csv
        ball_coordinates/     ball_<video name>.csv, ball_candidates_..., ball_wholeframe_...
        player_analysis/      heatmaps, zones, speeds
        motion_estimation/    optical flow, block matching
        shot_analysis/        shots.csv, events.csv, bounces.csv, hitmap, bounce map
        evaluation/           metrics against the ground truth

Hand-made annotations are inputs rather than results: they live in ground_truth/,
under version control, and nothing regenerates them.
"""

import os

OUTPUT_ROOT = "outputs"
GROUND_TRUTH_DIR = "ground_truth"


def clip_stem(video: str) -> str:
    """Name of a clip: its file name without the extension."""
    return os.path.splitext(os.path.basename(video))[0]


def clip_paths(video: str, output: str = OUTPUT_ROOT) -> dict:
    """Every path of a clip, under the base directory `output`."""
    output = output.rstrip("/\\") or OUTPUT_ROOT
    stem = clip_stem(video)
    clip_dir = os.path.join(output, stem)
    ball_dir = os.path.join(clip_dir, "ball_coordinates")
    shots_dir = os.path.join(clip_dir, "shot_analysis")
    return {
        "video": video,
        "output": output,
        "stem": stem,
        "clip_dir": clip_dir,
        "court_csv": os.path.join(clip_dir, "court_coordinates", f"{stem}_court.csv"),
        "players_csv": os.path.join(clip_dir, "player_coordinates", f"players_{stem}.csv"),
        "ball_csv": os.path.join(ball_dir, f"ball_{stem}.csv"),
        "ball_candidates_csv": os.path.join(ball_dir, f"ball_candidates_{stem}.csv"),
        "ball_wholeframe_csv": os.path.join(ball_dir, f"ball_wholeframe_{stem}.csv"),
        "analysis_dir": os.path.join(clip_dir, "player_analysis"),
        "motion_dir": os.path.join(clip_dir, "motion_estimation"),
        "shots_dir": shots_dir,
        "shots_csv": os.path.join(shots_dir, "shots.csv"),
        "events_csv": os.path.join(shots_dir, "events.csv"),
        "bounces_csv": os.path.join(shots_dir, "bounces.csv"),
        "eval_dir": os.path.join(clip_dir, "evaluation"),
    }


HANDS_CSV = os.path.join(GROUND_TRUTH_DIR, "hands.csv")


def clip_hands(video: str, p1=None, p2=None) -> tuple:
    """Dominant hands (p1, p2) of a clip: given ones first, then HANDS_CSV, else right.

    HANDS_CSV (clip,p1_hand,p2_hand) lists only the clips with a left-hander.
    """
    listed = {}
    if os.path.exists(HANDS_CSV):
        import csv
        with open(HANDS_CSV, newline="") as f:
            listed = {r["clip"]: (r["p1_hand"], r["p2_hand"]) for r in csv.DictReader(f)}
    q1, q2 = listed.get(clip_stem(video), ("right", "right"))
    return p1 or q1, p2 or q2


def ground_truth_path(kind: str, video: str) -> str:
    """Annotation file of a clip; `kind` is players, court, ball, bounce or shots."""
    return os.path.join(GROUND_TRUTH_DIR, f"{kind}_{clip_stem(video)}_gt.csv")
