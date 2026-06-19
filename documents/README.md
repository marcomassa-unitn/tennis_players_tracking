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
               despike ball track + vy reversals + acceleration peaks + proximity
               -> hit frames -> forehand/backhand + drive/slice/dropshot per player
```

## Setup

```bash
pip install -r requirements.txt
```

Put the input clips in `data/` (broadcast footage of a singles match from a
fixed camera; the rally clip used in the examples is `data/Input_video2.mp4`).

## Usage (from the project root)

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
```

The raw ball track is first **despiked**: a position that jumps far from its
local neighbours for a single frame and then returns is a tracking outlier (a
confident YOLO mis-detection — net cord, line, shoe) and is removed before any
velocity is computed, so it cannot create a huge spurious speed. The gate keys
on the distance from a local rolling median and fires only when a point exceeds
*both* an absolute pixel floor (`--despike-px`, default 40) and an adaptive
multiple of the local MAD (`--despike-mad`, default 4), so a genuinely fast ball
is never clipped.

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

Shot type from the **outgoing ball pace** measured just after contact:
`drive` (≥ `--drive-thr`, default 15), `slice` (between the two thresholds) or
`dropshot` (< `--dropshot-thr`, default 4). Pace is a **scale-free** index —
`100 × (ball pixels/frame just after contact) ÷ (striker's on-screen box
height)`. We deliberately do **not** use a court-metre km/h: the ball is
airborne at contact and the ground-plane homography turns its arc into large
fake distances (worst at the far baseline, where the court spans only a few
pixels), so homography km/h scatters by which end of the court a player stands
on rather than by how hard the ball was hit. Dividing the pixel speed by the
striker's box height cancels perspective, making the near and far players
directly comparable. Pace alone cannot detect backspin, so `slice` means a
medium-pace control shot inferred from speed; the numeric `ball_pace` is always
written too. Defaults fit ~1080p/30fps footage — **calibrate `--drive-thr` /
`--dropshot-thr` against the saved PNGs**.

Output: `outputs/shot_analysis/shots.csv` (with `stroke`, `shot_type` and
`ball_pace` columns), an annotated PNG per shot and a terminal summary.

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

## Project structure

| Path | Purpose |
|---|---|
| `tracking/court_tracking.py` | court keypoint detection (Hough + ITF proportions) |
| `tracking/playerTracking.py` | two-player tracking (background subtraction) |
| `utils/court_converter.py` | pixel → metre homography from the court CSV |
| `utils/player_analysis.py` | statistics, zones, heatmaps, minimap |
| `motionEstimation/optical_flow.py` | Farneback + Lucas-Kanade optical flow |
| `motionEstimation/block_matching.py` | full-search / three-step block matching |
| `evaluation/annotate.py` | manual ground-truth annotation tool |
| `evaluation/evaluate_tracking.py` | quantitative evaluation vs ground truth |
<<<<<<< HEAD:README.md
| `utils/shot_analysis.py` | ball despike + hit detection + forehand/backhand + drive/slice/dropshot |
| `ballTracking/BallTracking.py` | YOLO ball tracking → ball CSV for shot analysis |
=======
| `utils/shot_analysis.py` | hit detection + forehand/backhand classification |
| `tracking/BallTracking.py` | YOLO ball tracking → ball CSV for shot analysis |
>>>>>>> f82e3117622e28f121a97efe79f29314e64bc9fa:documents/README.md

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
