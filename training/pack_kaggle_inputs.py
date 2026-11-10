"""Zip what the Kaggle training notebook needs besides the videos.

The dataset is built on Kaggle from the videos plus the results of the offline
pipeline (they give the pseudo-labels, the players to mask and the court) and the
hand-made labels. This collects those small CSVs into one zip, keeping their
paths, to upload as (part of) a private Kaggle dataset next to the videos.

Usage (from project root):
    python training/pack_kaggle_inputs.py            # -> kaggle_inputs.zip
"""

import argparse
import glob
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import GROUND_TRUTH_DIR, OUTPUT_ROOT, clip_paths

KEYS = ("ball_csv", "ball_candidates_csv", "players_csv", "court_csv")


def main():
    parser = argparse.ArgumentParser(description="Pack the CSVs for the Kaggle notebook")
    parser.add_argument("--outputs", default=OUTPUT_ROOT)
    parser.add_argument("--zip", default="kaggle_inputs.zip")
    args = parser.parse_args()
    files = []
    for clip_dir in sorted(glob.glob(os.path.join(args.outputs, "*"))):
        p = clip_paths(os.path.basename(clip_dir) + ".mp4", args.outputs)
        found = [p[k] for k in KEYS if os.path.exists(p[k])]
        if len(found) == len(KEYS):
            files += found
        elif found:
            print(f"  {p['stem']}: incomplete, skipped ({len(found)}/{len(KEYS)} CSVs)")
    files += sorted(glob.glob(os.path.join(GROUND_TRUTH_DIR, "ball_*_gt.csv")))
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.relpath(f, ".").replace(os.sep, "/"))
    print(f"{len(files)} files -> {args.zip} ({os.path.getsize(args.zip) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
