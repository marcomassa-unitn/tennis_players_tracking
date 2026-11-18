"""Shot types of a results folder against the hand labels.

Labels come from ground_truth/shots_<clip>_gt.csv (annotate.py shots: frame,
player_id,label, made on the shots of one run; a later run's shot of the same
player within LABEL_TOLERANCE frames is the same shot) and, for Input_video2,
from the 23 types in analysis.shot_analysis._TYPE_GROUND_TRUTH (offline contact
frames, matched within 10 frames). A shot's predicted label is its overhead when
it is a serve or a smash, else its shot_type. "unsure" labels are left out; a
"dropshot/slice" label accepts either.

Usage (from project root):
    python evaluation/evaluate_shot_types.py                       # outputs/realtime
    python evaluation/evaluate_shot_types.py --results outputs     # the offline shots
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.shot_analysis import _TYPE_GROUND_TRUTH
from utils.paths import GROUND_TRUTH_DIR, OUTPUT_ROOT, clip_paths

LEGACY_CLIP = "Input_video2"        # labelled before annotate.py had a shots mode
LEGACY_TOLERANCE = 10               # frames between its labels and the shots found
LABEL_TOLERANCE = 6                 # frames a contact moves between runs (same player)


def predicted(row):
    overhead = row["overhead"] if isinstance(row["overhead"], str) else ""
    return overhead if overhead in ("serve", "smash") else row["shot_type"]


def labelled_shots(results):
    """One row per labelled shot found: clip, frame, side, label, pred, flight features."""
    sets = {Path(p).name[len("shots_"):-len("_gt.csv")]: p
            for p in glob.glob(os.path.join(GROUND_TRUTH_DIR, "shots_*_gt.csv"))}
    if LEGACY_CLIP not in sets:
        sets[LEGACY_CLIP] = None
    rows = []
    for clip, path in sorted(sets.items()):
        shots_csv = clip_paths(clip + ".mp4", results)["shots_csv"]
        if not os.path.exists(shots_csv):
            continue
        shots = pd.read_csv(shots_csv)
        if path is None:
            labels = [(f, None, lab) for f, lab in _TYPE_GROUND_TRUTH.items()]
        else:
            g = pd.read_csv(path)
            labels = list(zip(g["frame"], g["player_id"], g["label"]))
        for frame, pid, label in labels:
            near = shots[(shots["frame"] - frame).abs()
                         <= (LEGACY_TOLERANCE if pid is None else LABEL_TOLERANCE)]
            if pid is not None:
                near = near[near["player_id"] == pid]
            if near.empty:
                continue
            r = near.iloc[(near["frame"] - frame).abs().argmin()]
            rows.append({"clip": clip, "frame": int(r["frame"]), "side": r["side"],
                         "label": label, "pred": predicted(r),
                         "flight_s": r.get("flight_s"), "depth_m": r.get("landing_depth_m"),
                         "apex_m": r.get("apex_m"), "speed_ms": r.get("flight_speed_ms")})
    t = pd.DataFrame(rows)
    if not t.empty:
        t = t[t["label"] != "unsure"]
        t["ok"] = [p in str(lab).split("/") for lab, p in zip(t["label"], t["pred"])]
    return t


def main():
    parser = argparse.ArgumentParser(description="Shot types against the hand labels")
    parser.add_argument("--results", default=os.path.join(OUTPUT_ROOT, "realtime"))
    args = parser.parse_args()
    t = labelled_shots(args.results)
    if t.empty:
        raise SystemExit("No labelled shot found in " + args.results)
    per_clip = t.groupby("clip")["ok"].agg(["sum", "count"])
    print(f"Shot types: {int(t['ok'].sum())}/{len(t)} right  "
          + "  ".join(f"{c} {int(r['sum'])}/{int(r['count'])}" for c, r in per_clip.iterrows()))
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nlabel (rows) vs prediction (columns):")
        print(pd.crosstab(t["label"], t["pred"]))
        print("\nwrong:")
        print(t[~t["ok"]].drop(columns="ok").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
