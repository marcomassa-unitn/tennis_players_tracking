# Tennis Players Tracking

[![CI](https://github.com/marcomassa-unitn/tennis_players_tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/marcomassa-unitn/tennis_players_tracking/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Turn a single, uncalibrated broadcast tennis clip into structured match data:**
court geometry, metric player trajectories, movement statistics, the ball track
and a shot-by-shot breakdown (forehand/backhand, flat/slice/dropshot/lob,
serve/smash).

The core is **classical computer vision**. Court detection, player tracking, motion
estimation and shot reasoning are training-free and interpretable. The ball is
the one exception: it is small and fast, so it uses a YOLOv8 detector.

<p align="center">
  <img src="assets/demo.gif" alt="Live viewer: tracked players and ball, synced top-down minimap and live shot tally" width="900">
</p>

## Highlights

- **Court reconstruction without training.** The full ITF court model,
  including the doubles alleys, is fitted to the white line markings. Each
  candidate homography is scored by how much of the *whole* court it puts on
  line pixels, so banners, net and writing next to the court are rejected. The
  same settings work on five clips from three tournaments, and the fitted
  lines land within about 1 px of the painted ones.
- **One homography, metric everything.** A single RANSAC homography maps pixels
  to court metres, so positions, distances, speeds (km/h) and zone occupancy are
  all physical quantities.
- **Two-player tracking that does not depend on how the clip starts.** The
  background is the median of frames spread over the whole clip, so a player
  standing still to serve is still detected, and fade-ins from black do not
  matter. Identities come from the court half (through the homography), so a
  player who is lost comes back as soon as they are visible again. The camera
  tilting or panning a few pixels is compensated by phase correlation. Camera
  shake, cast shadows and on-screen TV graphics are filtered out. Both players
  are found on 95–100 % of the frames of all six test clips.
- **Ball tracking measured against hand-labelled positions.** A fine-tuned
  YOLOv8x runs on full-resolution tiles, and the ball is chosen as the best
  physically plausible path through the candidates. Against hand-labelled ball
  positions, recall rises from 0.64 to 0.92 with no false positives, and the
  centre error falls from 2.0 px to 1.2 px. On a second clip used only for
  testing, recall goes from 0.72 to 0.88.
- **Perspective-invariant shot classification.** Shot types are read from
  scale-free trajectory-shape features, so near-side and far-side shots need no
  separate calibration. It labels all 23 shots of a hand-labelled rally
  correctly (flat, slice, dropshot, lob).
  Every contact is also classified as forehand or backhand.
- **Classical motion estimation, implemented from scratch.** Full-search and
  three-step block matching (SAD), next to Farnebäck and Lucas–Kanade optical
  flow.
- **Engineered as a pipeline.** It runs with one command, headless, and each
  stage is isolated so one failure doesn't stop the rest. Stages pass data
  through a stable CSV contract. It also ships an annotation tool, IoU-based
  evaluation against manual ground truth, unit tests and CI.

## Results on the demo rally

The demo clip (`data/Input_video2.mp4`) is 1008 frames at 30 fps, 1080p. On it
the pipeline produces:

| Stage | Result |
|---|---|
| Court | 98 % of the court model on detected line pixels; fitted lines within ≈1 px of the painted ones |
| Players | both players detected on 100 % of the frames, identities fixed by court half |
| Ball | found on 90 % of the frames (original detector: 60 %); vs hand-labelled positions: precision 1.00, recall 0.92, centre error 1.2 px |
| Movement | near player 94.8 m covered (mean 10.6 km/h, p95 29.6 km/h); far player 104.2 m (mean 11.8 km/h, p95 27.4 km/h) |
| Shots | 25 contacts detected (13 near player, 12 far player), opening serve identified as a serve |
| Shot mix | 17 flat, 5 slice, 2 dropshot, 1 lob; 17 backhands, 6 forehands, 2 on the body axis (the serve included) |
| Shot type | **23/23** against a hand-labelled ground truth (`--type-self-test`)¹ |
| Self-tests | synthetic stroke test 4/4, synthetic serve/smash test 4/4 with zero false positives |

¹ The shot-type thresholds were tuned on this same rally, so 23/23 shows the
rules fit the labelled data. It is not an accuracy figure on unseen footage.
The report's 21/23 refers to an earlier version of the classifier.

| Combined position heatmap | Court-zone occupancy | Shot hitmap |
|:---:|:---:|:---:|
| <img src="assets/heatmap_combined.png" height="420"> | <img src="assets/zones.png" height="420"> | <img src="assets/shot_hitmap.png" height="420"> |

<p align="center">
  <img src="assets/court_keypoints.png" alt="Fitted court model and keypoints" width="49%">
  <img src="assets/flow_hsv.png" alt="Dense Farnebäck optical flow" width="49%">
  <br><em>Left: fitted court model (yellow) and the 8 keypoints (red). Right: dense optical flow (hue = direction, brightness = magnitude).</em>
</p>

## How it works

```mermaid
flowchart LR
    V[Video] --> C[Court detection<br/><sub>court-model fitting</sub>]
    V --> P[Player tracking<br/><sub>background subtraction</sub>]
    V --> B[Ball tracking<br/><sub>YOLOv8</sub>]
    C --> H((Homography<br/>px → m))
    H --> P
    H --> PA[Player analysis<br/><sub>distance, speed, zones, heatmaps</sub>]
    P --> PA
    P --> M[Motion estimation<br/><sub>optical flow, block matching</sub>]
    H --> M
    B --> S[Shot analysis<br/><sub>hits, stroke, shot type</sub>]
    P --> S
    H --> S
    P --> E[Evaluation<br/><sub>IoU vs ground truth</sub>]
    PA --> L[Live viewer]
    S --> L
```

1. **Court detection** ([`tracking/court_tracking.py`](tracking/court_tracking.py)).
   Line pixels are pixels brighter than their neighbours a few pixels away,
   which keeps thin, dim lines and drops large bright areas. A Hough transform
   gives candidate lines. Every pair of horizontal and pair of oblique
   candidates, matched to the court model's lines, defines a homography. It is
   scored by projecting the whole model, doubles alleys included, onto the line
   pixels. The best hypothesis is refined by refitting every line on its own
   pixels. Several frames are tried, which handles fade-ins and occlusions.
   The run fails, rather than saving wrong keypoints, if no frame matches the
   model well enough. An overlay image is saved next to the CSV to check the
   result.
2. **Player tracking** ([`tracking/player_tracking.py`](tracking/player_tracking.py)).
   The background is the per-pixel median of about 40 frames spread over the
   whole clip, leaving out black or fade-in frames, and aligned to each other.
   Each frame is first realigned to it, because a broadcast camera can tilt or
   pan a few pixels during a clip; the shift is measured by phase correlation.
   The frame is then gain-matched and compared with the background's local
   min/max over a few pixels, so the slight shake of broadcast cameras does not
   light up the court lines. While the camera is shifted, the court lines
   themselves are masked, and thin horizontal strips are removed from blobs.
   Cast
   shadows are removed, and so are on-screen graphics, which stay pixel-identical
   from frame to frame. Fragments of the same person are merged, and blobs are
   kept only if their feet stand in the walkable area of the court. P1 is the
   player in the near half, P2 the one in the far half. A player seen recently
   is followed by proximity; otherwise the largest plausible blob in that half
   is taken. Gaps up to 0.5 s are interpolated and flagged in an `interpolated`
   column.
3. **Player analysis** ([`analysis/player_analysis.py`](analysis/player_analysis.py)).
   The feet point of each box is projected to metres. From it the stage computes
   distance covered, mean, median, p95 and max speed (with a physiological 45 km/h
   glitch filter), occupancy of a 6×4 zone grid over the walkable area,
   Gaussian-smoothed heatmaps and an animated top-down minimap.
4. **Motion estimation** ([`motion_estimation/`](motion_estimation)).
   Dense Farnebäck flow inside each player box is converted to km/h and compared
   with positional speed. Lucas–Kanade tracks Shi–Tomasi corners. Block matching
   is written from scratch: full search and three-step search, with
   motion-compensated PSNR against a no-motion baseline.
5. **Ball tracking** ([`tracking/ball_tracking.py`](tracking/ball_tracking.py)).
   Two stages. **Detection:** a fine-tuned YOLOv8x runs on 640 px tiles of the
   frame at native resolution, plus the whole frame. On the whole frame alone
   the ~15 px ball would shrink to ~5 px. Every candidate is kept with its
   confidence, its source and how much it moved since the previous frame.
   **Tracking:** still candidates (balls held by ball kids) are discarded. The
   ball is then the best path through the remaining candidates over the whole
   clip, found by dynamic programming. Confident candidates earn points, while
   links cost in proportion to the implied speed and the frames skipped, and
   impossible speeds are forbidden. Knowing the frames after a racket hit, the
   path picks the right candidate where a frame-by-frame filter would not.
   One-frame spikes are removed, and gaps up to 0.33 s are filled with a
   shape-preserving cubic and flagged. Detection needs a GPU and runs in the
   [Colab notebook](notebooks/ball_tracking_colab.ipynb). Its candidates CSV is
   reused, so tracking runs on any laptop.

   | Ball vs hand-labelled positions | Precision | Recall | Centre error |
   |---|---|---|---|
   | Original detector (whole frame only), Input_video2 | 1.00 | 0.64 | 2.0 px |
   | **Current**, Input_video2 (101 labelled frames) | **1.00** | **0.92** | **1.2 px** |
   | Original detector, input_video | 0.97 | 0.72 | 2.1 px |
   | **Current**, input_video (43 frames, test only) | **0.95** | **0.88** | **1.3 px** |

   On the other test clips, the share of frames with a real detection rises from
   44–48 % to 82–87 %.
6. **Shot analysis** ([`analysis/shot_analysis.py`](analysis/shot_analysis.py)).
   Contacts are detected from persistent reversals of the vertical velocity and
   from acceleration peaks on the Savitzky–Golay-smoothed ball track, and are
   kept only when the ball is near a player. That rejects bounces. The stroke
   (forehand/backhand) comes from the ball's side relative to the body axis,
   taking handedness and court side into account. The shot type comes from
   perspective-invariant features of the outgoing trajectory. Serve and smash go
   in a separate, conservative overhead column.
7. **Evaluation** ([`evaluation/`](evaluation)). A small OpenCV tool annotates
   ground-truth boxes and court keypoints. The evaluator reports mean and median
   IoU, precision and recall at IoU 0.5, centre error (px), feet error (m), ID
   switches and per-keypoint court error.

The [**live viewer**](live_view.py) replays the clip with player and ball boxes,
a synced minimap and a running shot tally. At the end it shows the whole-clip
heatmap, zones and shot hitmap.

## Quick start

```bash
git clone https://github.com/marcomassa-unitn/tennis_players_tracking.git
cd tennis_players_tracking
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Optional: ball tracking.** Ball detection runs a large YOLO model, best on a
GPU. Download the trained weights from the
[latest release](https://github.com/marcomassa-unitn/tennis_players_tracking/releases/latest).
Then do one of the following:

- run [`notebooks/ball_tracking_colab.ipynb`](notebooks/ball_tracking_colab.ipynb)
  on a free Colab GPU and put the resulting `ball_candidates_<clip>.csv` in
  `outputs/ball_coordinates/`;
- or, with a GPU, put the weights in `models/ball_tracker.pt` and run
  `pip install ultralytics`.

Run the whole pipeline on the demo clip. It's headless and writes everything to
`outputs/`:

```bash
python pipeline.py --video data/Input_video2.mp4
```

Then watch the result:

```bash
python live_view.py --video data/Input_video2.mp4      # space: pause, n: next frame, q: quit
```

If the weights are missing, ball tracking and shot analysis are skipped cleanly
and the other stages still run. To use your own clip, put it in `data/` and
change `--video`. Every intermediate file is named after the video, so no other
paths need editing.

<details>
<summary><b>Running individual stages</b></summary>

```bash
# 1. Tracking: produces the CSVs everything else reads
python tracking/court_tracking.py  --video data/Input_video2.mp4 --no-display
python tracking/player_tracking.py --video data/Input_video2.mp4 --no-display
python display_player_tracking.py  --video data/Input_video2.mp4   # watch the tracker's stages (saves nothing)
python tracking/ball_tracking.py   --video data/Input_video2.mp4 --detect-only   # GPU: YOLO candidates
python tracking/ball_tracking.py   --video data/Input_video2.mp4 \
    --candidates outputs/ball_coordinates/ball_candidates_Input_video2.csv     # CPU: tracking

# Check the court keypoints: draws them and the reprojected court model on a frame
# (court_tracking.py also saves this image next to the court CSV)
python utils/court_overlay.py --video data/Input_video2.mp4

# 2. Analysis
python analysis/player_analysis.py --video data/Input_video2.mp4 [--no-animation]
python analysis/shot_analysis.py   --video data/Input_video2.mp4 --p1-hand right --p2-hand right

# 3. Motion estimation
python motion_estimation/optical_flow.py   --frames 200
python motion_estimation/block_matching.py --method both

# 4. Ground truth and evaluation
python evaluation/annotate.py boxes --video data/Input_video2.mp4 --step 30
python evaluation/annotate.py court --video data/Input_video2.mp4
python evaluation/annotate.py ball  --video data/Input_video2.mp4 --step 10   # click the ball, n = not visible
python evaluation/evaluate_tracking.py --gt <gt_boxes.csv> \
    --pred outputs/player_coordinates/players_Input_video2.csv \
    --court outputs/court_coordinates/Input_video2_court.csv \
    --court-gt <gt_court.csv>
python evaluation/evaluate_tracking.py --ball-gt outputs/ground_truth/ball_Input_video2_gt.csv \
    --ball-pred outputs/ball_coordinates/ball_Input_video2.csv
```

Useful `pipeline.py` flags:

- `--display` shows the OpenCV windows.
- `--skip-court`, `--skip-tracking`, … reuse CSVs that are already on disk.
- `--flow-frames N` caps the dense optical flow at N frames.
- `--gt-csv` / `--court-gt-csv` enable the evaluation stage.

Every script supports `--help`.

The shot-type thresholds are exposed as CLI flags (`--far-peak-m`,
`--drop-diefrac`, `--lob-pace`, …) and can be checked with the built-in
self-tests:

```bash
python analysis/shot_analysis.py --self-test           # synthetic strokes
python analysis/shot_analysis.py --overhead-self-test  # synthetic serve/smash
python analysis/shot_analysis.py --type-self-test      # labelled demo rally (needs the ball CSV)
```

</details>

## Project structure

```
├── pipeline.py              # one-command runner for all stages
├── live_view.py             # replay with overlays, minimap and live stats
├── display_player_tracking.py  # player tracker's stages side by side, for inspection
├── tracking/                # court keypoints, player tracker, YOLO ball tracker
├── analysis/                # player statistics and shot detection/classification
├── motion_estimation/       # Farnebäck / Lucas–Kanade, block matching from scratch
├── evaluation/              # annotation tool and tracking metrics
├── utils/                   # court geometry (ITF), pixel→metre homography, video helpers
├── tests/                   # pytest suite (run in CI)
├── notebooks/               # Colab notebook to run ball detection on a GPU
├── docs/                    # report (PDF + LaTeX) and presentation
├── data/                    # demo clip
└── models/                  # put ball_tracker.pt here
```

## Development

```bash
pip install -r requirements-dev.txt
pytest          # unit tests: homography, block matching, IoU, shot self-tests, …
ruff check .    # lint
```

## Limitations

- **Static camera assumed.** Replays and camera cuts must be trimmed beforehand.
- **Players stay in their half.** Identities come from the court half, which
  holds in singles play; a player crossing the net (rare) would swap IDs.
- **Distances include detection jitter.** They are summed frame by frame from
  the feet point, so the few centimetres of frame-to-frame jitter (more at the
  far end, where one pixel spans more court) inflate them.
- **Shot analysis still reads the original ball track.** Contact detection was
  designed and validated on the track of the original whole-frame detector.
  On the denser new track, it sometimes takes the bounce just before a shot
  for the contact. Shot analysis therefore reads `ball_wholeframe_<clip>.csv`,
  which is rebuilt from the same detection run with the original rules.
  Redesigning contact detection for the new track is future work.
- **Ball speeds are approximate.** The homography models the ground plane, while
  the ball is airborne at contact. The metre-based shot thresholds may need
  retuning for a different camera; the scale-free ones do not.
- **Thresholds are tuned on one camera setup.** The shot-type defaults were fit
  on the demo rally.

## Report and presentation

The [academic report](docs/report.pdf) covers the methodology, related work and
results in detail. It describes the original court detector (Hough
intersections and a coverage profile for the far baseline), which has since been
replaced by the court-model fitting described above. There is also an [HTML presentation](docs/presentation.html)
(download it and open it in a browser).

## Authors

Developed by **Marco Massa**, **Alfonso Antognozzi** and **Davide Cabitza** as
the project for the *Signal, Image and Video* course at the University of Trento.

Released under the [MIT License](LICENSE). The demo footage is included for
academic use only; see [`data/README.md`](data/README.md).
