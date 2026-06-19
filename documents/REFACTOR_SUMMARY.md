# Refactor Summary

This document records the refactor that connected the previously fragmented
scripts into a single, runnable pipeline.

## Goal

The project was functionally complete but spread across independent scripts,
some of which ran at import time, mixed Italian/English text, or could only be
invoked from the command line. The work below makes every module importable and
adds one unified entry point.

## Changes

### 1. `courtTracking/court_tracking.py` — wrapped into a class
- Previously the entire file ran at **module level** (argparse + OpenCV code at
  the top, no `main()` guard).
- Now wrapped into a `CourtTracker` class:
  ```python
  CourtTracker(video_path, no_display=False, roi_top=0.15, far_line="baseline")
  ```
  - `run() -> dict` returns the 8 court keypoints
    (`TL, TR, BL, BR, STL, STR, SBL, SBR`) **and** writes the CSV to
    `outputs/court_coordinates/{video_stem}_court.csv`.
  - The `os.makedirs` call, CSV-saving logic, and the playback loop now live
    inside `run()`.
- The 9 stateless helpers (`line_intersection`, `angle_between`,
  `cluster_points`, `filter_singles_by_proportion`, `project_all_corners`,
  `fit_sideline`, `find_ground_line_bands`, `select_far_lines`,
  `find_court_corners`) remain module-level functions.
- Added an `if __name__ == "__main__":` block with the **identical** CLI
  (`--video`, `--no-display`, `--roi-top`, `--far-line`).
- **Algorithm logic unchanged** — restructure only.

### 2. Italian → English translation
- `utils/player_analysis.py` — ~40 comments / docstrings / `print()` strings
  translated to English.
- `utils/shot_analysis.py` — ~35 items translated to English.
- `motionEstimation/optical_flow.py` — already English (verified).
- No variable names, function names, argparse argument names, column names, or
  data values were changed. Default paths were verified consistent with the
  `Input_video2` / `players_clip2` / `ball_clip2` family (no changes needed).

### 3. Path & import audit
- Audited all `.py` modules for default `--video` paths, `sys.path` hacks,
  cross-module imports, and `main()` / `__main__` guards. Findings drove the
  pipeline design (which modules are class-based vs `main()`-driven).

### 4. New file: `pipeline.py` (project root) — unified entry point
Runs the full chain end-to-end:

```
court detection -> player tracking -> player analysis ->
motion estimation -> ball tracking -> shot analysis -> evaluation
```

**How each step is invoked**
- Class-based modules are called directly:
  - `CourtTracker(...).run()`
  - `PlayerTracker(video, players_csv, display).run()`
  - `BallTracker(model_path).run(video, output_path=None, csv_path=..., display=...)`
- `main()`-driven modules (`player_analysis`, `optical_flow`, `block_matching`,
  `shot_analysis`, `evaluate_tracking`) are run via a scoped `sys.argv`
  override that calls their **unchanged** `main()`. This keeps every per-module
  CLI and all algorithm logic byte-for-byte intact.

**Derived paths** (from `--video` + `--output`, with the `Input_video2` →
`clip2` naming special case):

| Artifact      | Path                                              |
|---------------|---------------------------------------------------|
| court CSV     | `{output}/court_coordinates/{stem}_court.csv`     |
| players CSV   | `{output}/players_{key}.csv`                      |
| ball CSV      | `{output}/ball_{key}.csv`                         |
| analysis dir  | `{output}/player_analysis/`                       |
| motion dir    | `{output}/motion_estimation/`                     |
| shots dir     | `{output}/shot_analysis/`                         |
| eval dir      | `{output}/evaluation/`                            |

(`key` = `clip2` when the video stem is `Input_video2`, otherwise the lowercased stem.)

**CLI flags**
- `--video` (default `data/Input_video2.mp4`), `--output` (default `outputs/`),
  `--fps` (default 30.0), `--no-display`
- Step skips: `--skip-court`, `--skip-tracking`, `--skip-analysis`,
  `--skip-motion`, `--skip-ball`, `--skip-shots`, `--skip-evaluation`
  (skipped steps reuse existing CSVs when present)
- Forwarded tuning: `--roi-top`, `--far-line`, `--anchor`, `--p1-hand`,
  `--p2-hand`
- Evaluation: `--gt-csv`, `--court-gt-csv` (evaluation only runs when a ground
  truth CSV is provided)

**Robustness**
- Ball tracking wrapped in `try/except ImportError` (ultralytics optional) plus
  a weights-existence check for `ball_tracker.pt`.
- Each step prints `=== Step N: <Name> ===` and the run ends with an artifact
  summary.

## Usage

Run the whole pipeline:

```bash
python pipeline.py --video data/Input_video2.mp4 --no-display
```

Re-run only some steps (e.g. reuse court + tracking, redo analysis):

```bash
python pipeline.py --skip-court --skip-tracking
```

Every original per-module command still works exactly as before.

## Verification
- `python -m py_compile pipeline.py` — passes.
- All sub-module argparse names passed by the pipeline confirmed to exist
  (`--court-pred`, `--court-gt`, `--gt`, `--pred`, optical_flow / block_matching args).
- `CourtTracker` and `PlayerTracker` import successfully and their constructor
  signatures match the pipeline's calls.

## Hardening pass (2026-06-19)

After the initial integration, a multi-agent critical audit (`PROJECT_AUDIT.md`)
drove a project-wide hardening pass. Highlights:

- **Resource safety**: every `run()` (court / player / ball) now releases its
  `VideoCapture`/`VideoWriter` and closes CSVs via `try/finally`.
- **Pipeline robustness**: each step is isolated in `try/except (Exception,
  SystemExit)` (one failing step no longer aborts the run); `step_shots` skips
  gracefully when the ball CSV is absent; a final summary lists
  succeeded / skipped / failed steps.
- **Path consistency**: `CourtTracker` now takes `output_dir` (and a `--output`
  CLI arg), so the court CSV honors the pipeline `--output` — the previous
  hardcoded-`outputs/` caveat is **resolved**.
- **Headless by default**: `python pipeline.py` runs headless; `--display` opts
  into OpenCV windows (`--no-display` still accepted).
- **Correctness/robustness**: unified `isfinite` FPS guard across trackers,
  RANSAC homography, deterministic court clustering, contiguous-run gradients and
  fixed min-gap dedup in shot detection, IoU-gated evaluation metrics, real FPS
  read from the video, bounded optical flow (`--flow-frames`), and a new
  **player ID-swap correction** in `playerTracking.py`.

See `PROJECT_AUDIT.md` (status banner at top) for the full per-item list.

## Known caveat
- None outstanding from the original audit. The previously-noted
  `CourtTracker` hardcoded-output-path issue is resolved (see above).
