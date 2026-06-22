# Tennis Players Tracking

Computer-vision analysis of tennis **singles** matches for the *Signal,
Image and Video* course (University of Trento).

From a **single fixed-camera video** the pipeline detects the court and the
two players using **classical image-processing and motion-estimation
techniques** (no deep learning), and extracts kinematic and positional
statistics: distance covered, average / peak speed, time spent in court
zones, and trajectory heatmaps. Tracking accuracy is evaluated against
manual annotations.

## Pipeline overview

```
video ─┬─> tracking/court_tracking.py ──> outputs/court_coordinates/<video>_court.csv
       │       white mask + Hough lines + intersection clustering
       │       + ITF-proportion filtering -> 8 court keypoints
       │
       └─> tracking/playerTracking.py ──> outputs/player_coordinates/players_<video>.csv
               running-average background subtraction + connected
               components + nearest-centroid association -> 2 boxes/frame
                       │
                       v
           utils/player_analysis.py            (homography px -> metres)
               distance, speeds, zone occupancy, heatmaps, minimap GIF
                       │
                       v
           motionEstimation/                   (lab techniques)
               optical_flow.py     Farneback + Lucas-Kanade, speed from flow
               block_matching.py   full search + three-step search, PSNR
                       │
                       v
           evaluation/                         (quantitative accuracy)
               annotate.py            manual GT (boxes + court keypoints)
               evaluate_tracking.py   IoU, precision/recall, centre error,
                                      ID switches, keypoint error (px and m)

tracking/BallTracking.py (YOLO) ──> outputs/ball_coordinates/ball_<video>.csv
                       │
                       v
           utils/shot_analysis.py              (hit detection)
               vy reversals + acceleration peaks + player proximity
               -> hit frames -> forehand/backhand + flat/slice/dropshot/lob per shot

   ── all three CSVs ──> live_view.py   (front-end: replays the clip with boxes +
       minimap + live stats, auto-builds and shows the end-of-clip summary figures)
```

## Setup

```bash
pip install -r requirements.txt
```

Put the input clips in `data/` (broadcast footage of a singles match from a
fixed camera; the rally clip used in the examples is `data/Input_video2.mp4`).

## Quick start — launch sequence

Everything is keyed off the **video file name**, so every script below takes the
same `--video`; the CSVs are auto-named after the video stem and found
automatically by the next step. Run from the **project root**.

**You must run the three tracking scripts FIRST** — they produce the CSVs that
everything else (analysis, shot detection, the live viewer) reads. The order is
court → players → ball:

```bash
# 1. tracking — produces the CSVs (run these three first, in this order)
python tracking/court_tracking.py  --video data/Input_video2.mp4 --no-display
python tracking/playerTracking.py  --video data/Input_video2.mp4 --no-display
python tracking/BallTracking.py    --video data/Input_video2.mp4 --no-display --no-video

# 2. watch everything in one window (live overlays + end-of-clip stats)
python live_view.py --video data/Input_video2.mp4
```

That's the whole flow for a new clip. `live_view.py` is the front door: it
**auto-generates** the heatmaps, court-zone map and shot hitmap on launch (so you
don't run `player_analysis.py` / `shot_analysis.py` by hand) and shows them in a
summary window when the clip ends — see [§7](#7-live-viewer-live_viewpy).

**One-shot alternative.** `pipeline.py` chains all the tracking + analysis steps
itself (headless by default):

```bash
python pipeline.py --video data/Input_video2.mp4        # then: python live_view.py --video data/Input_video2.mp4
```

Swap in your own clip by changing `--video` everywhere (e.g.
`--video data/my_match.mp4`); no other paths need editing. The standalone
analysis/evaluation tools in §3–§6 below are for inspecting one stage in
isolation — the quick start above already covers the normal path.

## Usage (from the project root)

### 0. Full pipeline (one command)

`pipeline.py` runs the whole chain end-to-end (court → player tracking →
analysis → motion estimation → ball tracking → shot analysis → evaluation),
deriving every intermediate path from `--video`:

```bash
python pipeline.py --video data/Input_video2.mp4          # headless by default
python pipeline.py --video data/Input_video2.mp4 --display # show OpenCV windows
```

Each step is wrapped in its own try/except (one failing step is reported and
skipped, not fatal) and can be skipped with its `--skip-*` flag to reuse CSVs
already on disk. The individual steps below can also be run on their own.

### 1. Court keypoint detection

```bash
python tracking/court_tracking.py --video data/Input_video2.mp4 [--no-display]
```

Detects the white court lines on the first frame (HSV white mask, morphology,
Canny + probabilistic Hough), clusters the pairwise line intersections and
strips doubles-alley corners using the ITF width proportions (alley = 1.37 m
of 10.97 m).

The far half needs special care in broadcast footage: the far baseline and
service line are thin and dim (missed by the global Hough), while banner
edges and the **elevated** net band are bright and easily mistaken for them.
The far side is therefore refined with a **1-D white-coverage profile**: the
two sidelines are fitted from the Hough segments anchored at the (reliable)
near corners, and for every image row the fraction of white pixels along the
chord between them is computed. Thin full-coverage runs are horizontal ground
lines; the (baseline, service-line) pair is selected by **projective
consistency** with the ITF model (the homography built from a candidate pair
must predict the other detected lines within a few px). This automatically
rejects banners and the net; the residuals are printed for verification.
Saves 8 labelled keypoints (4 court corners + 4 service-line corners) to
`outputs/court_coordinates/<video>_court.csv`.

Always verify the overlay visually (run without `--no-display`). If the
profile method fails on a clip, `--roi-top` and `--far-line service` are
available as manual fallbacks.

### 2. Player tracking

```bash
python tracking/playerTracking.py --video data/Input_video2.mp4 [--no-display]
```

The output CSV defaults to `outputs/player_coordinates/players_<video name>.csv`
(e.g. `outputs/player_coordinates/players_Input_video2.csv`), so different inputs
no longer overwrite each other; override the path explicitly with `--csv <path>`
if needed.

Foreground extraction with a running-average background model
(`cv2.accumulateWeighted`), thresholded frame difference, morphological
cleaning, connected components; the two players are selected by area and
associated frame-to-frame by nearest centroid (with a max-displacement gate
and an overlap-suppression rule). Output: one bounding box + centroid per
player per frame.

### 3. Kinematic / positional statistics

```bash
python utils/player_analysis.py --video data/Input_video2.mp4 \
    [--no-animation] [--anchor feet|centroid]
```

`--players` and `--court` default to the CSVs derived from `--video`
(`outputs/player_coordinates/players_<video name>.csv` and
`outputs/court_coordinates/<video name>_court.csv`), matching the player- and
court-tracking outputs; pass them explicitly to override.

The **feet point** (bottom-centre of each box) is projected to court metres
through the homography built from the detected keypoints (`CourtConverter`).
All statistics cover the whole **walkable area**, not just the playing
rectangle: ±6.5 m behind each baseline and ±3.7 m beside each sideline
(typical tournament run-offs). Produces in `outputs/player_analysis/`:

- `speed_stats.csv` — per-frame position (m) and speed (km/h); displacements
  above a physiological `--max-speed` (45 km/h) are treated as tracking
  glitches and excluded;
- `zone_stats.csv` + `zones.png` — time spent in each zone: 6 depth bands
  (behind baseline / backcourt / service area, per half) × 4 columns
  (outside-left / left / right / outside-right);
- `heatmap_p1/p2/combined.png` — Gaussian-smoothed position density over the
  extended area;
- `minimap.gif` — top-down animation of both players (positions outside the
  court are shown too);
- terminal summary: total distance, mean / median / p95 / max speed,
  detection rate, top zones.

### 4. Motion estimation (lab techniques)

```bash
# Dense Farneback flow + pyramidal Lucas-Kanade demo; the mean flow inside
# each tracked player box is converted to km/h via the homography and
# compared with the positional speed:
python motionEstimation/optical_flow.py [--frames N] [--display]

# Block matching implemented from scratch (SAD criterion): exhaustive full
# search vs three-step search, motion-compensation PSNR report and
# motion-vector visualizations:
python motionEstimation/block_matching.py --method both [--display]
```

Outputs go to `outputs/motion_estimation/` (`flow_speeds.csv`, flow HSV /
arrow images, LK trails, block-matching vector fields).

### 5. Ball & shot analysis (hit detection, forehand/backhand, shot type)

Requires the ball CSV produced by the existing YOLO ball tracker
(`tracking/BallTracking.py`, needs `ultralytics` + the `ball_tracker.pt`
weights in the project root):

```bash
python tracking/BallTracking.py          # -> outputs/ball_coordinates/ball_Input_video2.csv
python utils/shot_analysis.py --video data/Input_video2.mp4 \
    --p1-hand right --p2-hand right          # handedness of each player

# validate the detection/classification logic without the YOLO model:
python utils/shot_analysis.py --self-test
# validate shot-type classification against the labelled ground truth
# (needs the real Input_video2 ball + players + court CSVs):
python utils/shot_analysis.py --type-self-test
```

Hit detection on the Savitzky-Golay-smoothed ball track: candidates are
persistent sign reversals of the vertical velocity `vy` **or** peaks of the
acceleration magnitude (sharp speed change); a candidate counts as a hit only
if the ball lies inside a player's expanded bounding box (bounces also flip
`vy`, but happen away from the players) and hits are at least `--min-gap`
seconds apart.

Stroke classification at the hit frame: the ball is compared with the
player's body axis. The near player is seen from behind (his right = image
right), the far player faces the camera (his right = image left); for a
right-hander the shot is a *forehand* when the ball is on the dominant-hand
side, a *backhand* otherwise, and the reasoning is inverted for left-handers
(`--p1-hand/--p2-hand left`).

Shot **type** (`flat` / `slice` / `dropshot` / `lob`) is read from the **shape
of the outgoing trajectory** in the frames right after contact. The fixed,
angled camera makes the near player's pixel speed ~3× the far player's for the
same physical shot, so a raw pixel speed is never compared across sides;
perspective is cancelled three ways: (a) a **scale-free pace** = `100 × (ball
pixels/frame) ÷ (ball bbox height)` (the ball's apparent size shrinks with
distance just as its apparent speed does); (b) a **court-metre speed** via the
homography (used only for the far flat/slice split); and (c) **dimensionless
shape** features — `diefrac` (how fast the ball stops travelling), `reach`, and
`bowback` (how much the ball arcs back). The rules: a far ball that dies quickly
is a `dropshot`; a near ball that arcs up and returns slowly is a `lob`; a
floated, decelerating ball is a `slice`; otherwise `flat`. **Caveat:** the
court homography is a ground plane while the ball is airborne at contact, so the
metre-based speeds are approximate — every threshold is a CLI flag
(`--far-peak-m`, `--drop-diefrac`, `--drop-tail-m`, `--lob-diefrac`,
`--lob-pace`, `--lob-reach`, `--nslice-bb`, `--nslice-diefrac`,
`--nslice-peak-m`, plus the windows `--k-window` / `--w30-window`) and may need
per-camera retuning; the scale-free features (pace / diefrac / reach / bowback)
do not. Defaults were fit against a 23-shot ground truth on `Input_video2`
(`--type-self-test` reproduces **23/23**, stable across `--k-window` 22–30); on
a different camera the type labels are indicative until recalibrated.

Output: `outputs/shot_analysis/shots.csv` (with `stroke`, `shot_type` and the
numeric `ball_pace` columns), an annotated PNG per shot and a terminal summary
that tallies forehands/backhands per player and the count of each shot type.

### 6. Ground truth & evaluation

```bash
# annotate player boxes every 30 frames (P1 = near player, P2 = far player):
python evaluation/annotate.py boxes --video data/Input_video2.mp4 --step 30

# annotate the 8 court keypoints on one frame:
python evaluation/annotate.py court --video data/Input_video2.mp4

# compare tracker + court detection against the annotations:
python evaluation/evaluate_tracking.py \
    --gt outputs/ground_truth/players_clip2_gt.csv \
    --pred outputs/player_coordinates/players_Input_video2.csv \
    --court outputs/court_coordinates/Input_video2_court.csv \
    --court-gt outputs/ground_truth/input_video2_court_gt.csv
```

Reported metrics: mean/median IoU, precision and recall @ IoU 0.5, centre
error (px), feet-point error (m), ID switches, per-keypoint court error in
pixels and metres. Per-frame details are saved in `outputs/evaluation/`.

### 7. Live viewer (`live_view.py`)

The single window that ties it all together: it replays the original clip with
the player boxes + ball box drawn on it, a synced top-down minimap, and a live
shot-tally stats panel, then shows a combined-statistics window at the end.

```bash
python live_view.py --video data/Input_video2.mp4        # press 'q' to quit
```

**Prerequisite:** the three tracking CSVs for that video must already exist —
run §1 court, §2 players and §3’s `BallTracking.py` first (or `pipeline.py`).
The viewer **aborts with a clear message** if a required CSV is missing.

On launch it regenerates the three summary figures in-process and headless
(combined heatmap + court zones + shot hitmap), so they always reflect the
current CSVs — you never have to run `player_analysis.py` / `shot_analysis.py`
by hand. The slow `minimap.gif` and per-shot PNGs are **not** regenerated here.

While the clip plays you see:

- the player boxes (`P1`/`P2`) and the ball box, drawn at display resolution so
  the labels stay crisp;
- a top-down **minimap** with the live player dots and a shot-type legend;
- a **stats panel** tallying each player’s shots per type, updated as shots occur.

When the clip ends it **freezes on the last frame** (full final tally) and opens
a **second, centred window** tiling the three whole-clip figures —
combined heatmap | court zones | shot hitmap — so the totals can be studied.
Press `q` to close.

Paths default to the same stem-based CSVs as everything else and can be
overridden (`--players`, `--ball`, `--court`, `--shots`, `--output`). Display
size is decoupled from render quality: the canvas is composed at
`--render-width` (1600 px, the text-sharpness knob) and shown at the smaller
`--window-width` (1100 px) — make the window any size without softening the text.
Other flags: `--anchor feet|centroid`, `--min-area`, `--max-shot-markers`,
`--fps`.

## Project structure

| Path | Purpose |
|---|---|
| `pipeline.py` | unified entry point — runs the whole chain from one command |
| `live_view.py` | live viewer — video + player/ball boxes + synced minimap |
| `tracking/court_tracking.py` | court keypoint detection (Hough + ITF proportions) |
| `tracking/playerTracking.py` | two-player tracking (background subtraction) |
| `utils/court_converter.py` | pixel → metre homography from the court CSV |
| `utils/player_analysis.py` | statistics, zones, heatmaps, minimap |
| `motionEstimation/optical_flow.py` | Farneback + Lucas-Kanade optical flow |
| `motionEstimation/block_matching.py` | full-search / three-step block matching |
| `evaluation/annotate.py` | manual ground-truth annotation tool |
| `evaluation/evaluate_tracking.py` | quantitative evaluation vs ground truth |
| `utils/shot_analysis.py` | hit detection + forehand/backhand + flat/slice/dropshot/lob shot type |
| `tracking/BallTracking.py` | YOLO ball tracking → ball CSV for shot analysis |
| `live_view.py` | live viewer: video + boxes + minimap + stats; auto-builds and shows the end-of-clip summary figures |
| `pipeline.py` | one-shot runner that chains all tracking + analysis steps (headless by default) |

## Design notes & limitations

- The court CSV used by the analysis **must come from the same video** as the
  player CSV (same camera framing). Since all three stages name their outputs
  after the video stem (`players_<video>.csv`, `<video>_court.csv`), passing the
  same `--video` to `player_analysis.py` pairs them automatically.
- Player positions are projected at the **feet**: projecting the body
  centroid through a ground-plane homography overestimates the distance from
  the camera (the centroid sits ~1 m above the ground).
- The tracker is appearance-free: identities are kept by proximity only, so
  long occlusions or players crossing sides can swap IDs (measured by the
  *ID switches* metric in the evaluation).
- Static camera is assumed: replays / camera cuts in broadcast footage should
  be trimmed beforehand.
