# Tennis Players Tracking

[![CI](https://github.com/marcomassa-unitn/tennis_players_tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/marcomassa-unitn/tennis_players_tracking/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Turn a single, uncalibrated broadcast tennis clip into structured match data:**
court geometry, metric player trajectories, movement statistics, the ball track,
its bounces and a shot-by-shot breakdown (forehand/backhand, flat/slice/dropshot/lob/volley,
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
  positions, recall rises from 0.64 to 0.93 with no false positives, and the
  centre error falls from 2.0 px to 1.2 px. On a second clip, recall goes
  from 0.72 to 0.95.
- **Shot types from the ball's physics.** Bounces are found to within a fraction
  of a frame, and the flight to the bounce gives each shot's type: its duration,
  landing point and height work the same at both ends of the court. It agrees with
  all 23 shots of a hand-labelled rally (flat, slice, dropshot, lob), and volleys
  are told by the missing bounce.
  Every contact is also classified as forehand or backhand; in the hitmap the
  colour gives the type of shot and the letter (F, B) its side.
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
| Ball | found on 92 % of the frames (original detector: 60 %); vs hand-labelled positions: precision 1.00, recall 0.93, centre error 1.2 px |
| Movement | near player 94.8 m covered (mean 10.6 km/h, p95 29.6 km/h); far player 104.2 m (mean 11.8 km/h, p95 27.4 km/h) |
| Shots | 25 contacts detected (13 near player, 12 far player), opening serve identified as a serve |
| Bounces | vs 25 hand-labelled bounces: precision 1.00, recall 0.96, median time error 0.2 frames, median landing error 0.23 m; on a second clip (14 bounces) 1.00, 1.00, 0.2 frames, 0.12 m |
| Shot mix | 18 flat, 4 slice, 2 dropshot, 1 lob; 17 backhands, 7 forehands, 1 serve |
| Shot type | **23/23** against a hand-labelled ground truth (`--type-self-test`)¹ |
| Self-tests | synthetic stroke test 4/4, synthetic serve/smash test 5/5 with zero false positives |

¹ The three flight thresholds were chosen looking at this same rally, so 23/23
shows the rules fit the labelled data. It is not an accuracy figure on unseen
footage. On the other clips there are no labels: the 1.05 s split falls in a gap
(1.04–1.10 s) that held on the five clips measured, and the lobs found there were
checked by eye.

| Combined position heatmap | Court-zone occupancy | Shot hitmap |
|:---:|:---:|:---:|
| <img src="assets/heatmap_combined.png" height="420"> | <img src="assets/zones.png" height="420"> | <img src="assets/shot_hitmap.png" height="420"> |

<p align="center">
  <img src="assets/court_keypoints.png" alt="Fitted court model and keypoints" width="49%">
  <img src="assets/flow_hsv.png" alt="Dense Farnebäck optical flow" width="49%">
  <br><em>Left: fitted court model (yellow) and the 8 keypoints (red). Right: dense optical flow (hue = direction, brightness = magnitude).</em>
</p>

<p align="center">
  <img src="assets/impact_comparison.gif" alt="Ball path on the court: bending at every bounce when shots sit at the player's feet, straight when impacts sit on the flight line" width="365">
  <br><em>Seen from above, a bounce does not turn the ball. Left: with shots placed at the player's feet, the path bends at every bounce, by an arm and a racket. Right: with impacts placed on the flight line, it runs straight. Those impacts also tell forehand from backhand on the court.</em>
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
   **Motion:** ball-sized blobs in the difference from the clip's median background
   join the candidates. They do not depend on how the ball looks, so they fill in
   where the detector does not see it, as on a court unlike its usual ones. Blobs only
   count over the court and out of a player's reach, and alone they must form a series.
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
   | **Current**, Input_video2 (101 labelled frames) | **1.00** | **0.93** | **1.2 px** |
   | Original detector, input_video | 0.97 | 0.72 | 2.1 px |
   | **Current**, input_video (43 frames) | **0.95** | **0.95** | **1.5 px** |

   On the other test clips, the share of frames with a real detection rises from
   44–48 % to 88–91 %.
6. **Shot analysis** ([`analysis/shot_analysis.py`](analysis/shot_analysis.py)).
   Contacts are detected from persistent reversals of the vertical velocity and
   from acceleration peaks on the Savitzky–Golay-smoothed ball track, and are
   kept only when the ball is near a player. That rejects most bounces, and the
   ones left, right in front of the receiver, are moved to the hit that follows
   them using the events of the next step. A contact must then send the ball towards
   the opponent, and when one player seems to strike twice in a row the opponent's shot
   in between is added, from the events or from where the ball turned back. The stroke
   (forehand/backhand) comes from the ball's side relative to the body axis,
   taking handedness and court side into account. The shot type comes from the
   flight to the bounce of the next step: a flight under 1.05 s is flat; a slower
   ball is a dropshot if it lands within 4.5 m of the net, a lob if it flies above
   2 m, and a slice if it floats lower. The height is the ball's distance in the
   image from the point below it on the court. A shot played from inside the
   court with no bounce since the opponent's shot, with the ball coming and going
   back, is a volley. Shots with no bounce found fall back
   on perspective-invariant features of the outgoing trajectory. A serve is what opens a
   point: the first shot after a quiet spell, by a player standing still on their
   baseline. That holds for any player and needs no ball, often not seen at the far
   end. The peak of the box raised by arm and racket gives the contact, so the toss
   release is not mistaken for it. The serve is placed where the server stood
   before jumping, since feet in the air project to the wrong spot on the court.
   Smashes are often hit out of sight, the ball of a high lob having left the
   picture, so they are also found from the motion: the arm-up of a serve from inside
   the court, followed by the ball leaving the striker's side and crossing the net.
   Both go in a separate, conservative overhead column.
7. **Bounce detection** ([`analysis/bounce_detection.py`](analysis/bounce_detection.py)).
   Between two events the ball flies freely, so in the image it follows almost a
   parabola. The real detections are split into the best sequence of parabolas by
   dynamic programming, and the point where two of them meet gives the time, between
   frames if needed, and the landing point. Physics then filters the junctions: the
   ground only pushes the ball up the image (typically a jump of 9–13 m/s here, at
   both ends of the court once scaled to metres), and a bounce lies on the court,
   where the homography is exact. The order of a rally (hit, bounce on the other side,
   hit, …) picks the final labels, which is what separates a near bounce from the
   near hit that follows it. Pipeline step 6 runs this first, then the shots.
   Seen from above, a bounce does not turn the ball, so the previous impact, the bounce
   and the next impact lie on one line. The player's feet are about 0.85 m off it, an
   arm and a racket away, so the impact is taken on the line, unless the player is out
   of reach of it. The side of the impact then gives forehand or backhand on the court,
   and the two methods agree on 92 of the 97 shots where both apply; the five that
   differ were checked by eye, and the image-based rule was the wrong one in four.
8. **Evaluation** ([`evaluation/`](evaluation)). A small OpenCV tool annotates
   ground-truth boxes and court keypoints. The evaluator reports mean and median
   IoU, precision and recall at IoU 0.5, centre error (px), feet error (m), ID
   switches and per-keypoint court error.

The [**live viewer**](live_view.py) replays the clip with player and ball boxes,
a synced minimap, a running shot tally and a court following the ball, with its
bounces numbered as they happen. The ball moves there on its path over the ground,
from each contact to each bounce, since an airborne ball cannot be projected. At the end it shows the whole-clip
heatmap, zones, shot hitmap and bounces.

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
  on a free Colab GPU, or [`notebooks/ball_tracking_kaggle.ipynb`](notebooks/ball_tracking_kaggle.ipynb)
  on Kaggle, where a saved version keeps running with the computer off. Then put the
  resulting `ball_candidates_<clip>.csv` in `outputs/<clip>/ball_coordinates/`;
- or, with a GPU, put the weights in `models/ball_tracker.pt` and run
  `pip install ultralytics`.

Run the whole pipeline on the demo clip. It's headless and writes everything to
the clip's own folder, `outputs/Input_video2/`:

```bash
python pipeline.py --video data/Input_video2.mp4
```

It lists which steps already have results and asks which ones to redo; the rest
is reused. Redoing a step also offers to redo the steps that read its results.
Without a terminal, only the missing steps run. Every clip keeps its own results,
so the viewers open any processed clip without running anything again.

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
    --candidates outputs/Input_video2/ball_coordinates/ball_candidates_Input_video2.csv     # CPU: tracking

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
python evaluation/annotate.py bounce --video data/Input_video2.mp4   # a/d move, click + ENTER = bounce, h = hidden
python evaluation/evaluate_tracking.py --gt <gt_boxes.csv> \
    --pred outputs/Input_video2/player_coordinates/players_Input_video2.csv \
    --court outputs/Input_video2/court_coordinates/Input_video2_court.csv \
    --court-gt <gt_court.csv>
python evaluation/evaluate_tracking.py --ball-gt ground_truth/ball_Input_video2_gt.csv \
    --ball-pred outputs/Input_video2/ball_coordinates/ball_Input_video2.csv
python evaluation/evaluate_tracking.py --video data/Input_video2.mp4     --bounce-gt ground_truth/bounce_Input_video2_gt.csv     --court outputs/Input_video2/court_coordinates/Input_video2_court.csv
```

Useful `pipeline.py` flags:

- `--display` shows the OpenCV windows.
- `--redo 2,6`, `--redo shots`, `--redo all`, `--redo missing` choose the steps
  to redo without asking.
- `--skip-court`, `--skip-tracking`, … skip single steps, also without asking.
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

## Real time

`realtime_pipeline.py` runs the whole analysis (court, players, ball, bounces,
shots) in **one causal pass**: each frame is seen once, in order, as a camera
would give it. It writes the same CSVs as the offline pipeline, under
`outputs/realtime/<clip>/`, so `live_view.py --realtime` shows them and the
evaluation reads them unchanged.

```bash
pip install -r requirements-realtime.txt
python realtime_pipeline.py --video data/Input_video2.mp4 --model models/ball_motion.onnx
python realtime_pipeline.py --video data/Input_video2.mp4 --calibrate     # tracks from frame 0
python realtime_pipeline.py --video data/Input_video2.mp4 --calibrate --fallback models/ball_tracker.pt   # GPU
python realtime_pipeline.py --video data/Input_video2.mp4 --clock live --display   # like a camera
python realtime_pipeline.py --video data/Input_video2.mp4 --model diff            # no model
python live_view.py --video data/Input_video2.mp4 --realtime                       # watch it
```

**Calibration.** With a fixed camera, the court and the empty court can be
learnt before the match, during the warm-up for instance. `--calibrate` does it
once per clip and saves the result in `outputs/realtime/<clip>/calibration/`
(`court.csv`, `court.png`, `background.png`). The background is the median of
41 frames spread over the clip, at full resolution. The analysis then tracks
from the first frame, and the background keeps refreshing itself as below.
Without calibration, the court is fitted on the first frames and the background
learnt in the first 2 s, which go untracked.

**What changes from offline.**

- **Background from the past only.** It is the median of the first 2 s, then of
  a ring of samples, one per second, refreshed every 5 s in a worker thread
  (`utils/background.py`). The players are masked out of each sample, so a
  server standing still does not become part of the background.
- **A small ball detector that reads motion.** YOLO11n reads 320 × 320 crops at
  full resolution, with three channels: grey, difference from the background at
  *t*, the same at *t − 1* (`tracking/ball_online.py`). The difference shows the
  ball whatever the colour of the court; the grey image tells it from a racket.
  - While tracking, a Kalman filter predicts the ball and **one crop** is read.
  - When the ball is lost, the court tiles (GPU) or the moving blobs (CPU) are
    read instead.
  - The path search of the offline tracker runs with a **fixed lag of 16 frames
    (0.5 s)**: on the clips, 99.6 % of the positions match the offline path.
- **Bounces and shots on a sliding window.** The unchanged offline functions run
  every second on the last 10 s in a worker thread. An event is committed once
  it is 1 s old, a shot once it is 4 s old. Fed the offline tracks, this finds
  50 of the 51 events and all 25 shots of the demo rally, with the same strokes
  and types.
- **One loop.** Decoding has its own thread; ball and players of a frame run
  side by side.

**Speed-ups that keep the offline results identical.** These help the offline
pipeline too:

| Step | Before | After |
|---|---|---|
| Player tracking, Input_video2 | 147 s | 86 s |
| Bounce segmentation, 11 clips | 34 s | 5 s |

**First results.** YOLO11n trained on Kaggle with pseudo-labels only, the
second dataset (more contacts), best epoch 19.

| Ball vs hand labels, recall / precision | Input_video2 | input_video |
|---|---|---|
| Offline (YOLOv8x tiles, whole clip) | 0.93 / 1.00 | 0.95 / 0.95 |
| Real time, calibrated, with the colour fallback | 0.92 / 0.96 | 0.91 / 0.95 |
| Real time, calibrated | 0.87 / 0.96 | 0.91 / 0.97 |
| Real time, 2 s warm-up | 0.83 / 0.97 | 0.65 / 0.97 |
| Detector alone, crop around the true ball (recall) | 0.87 | 0.90 |

These real-time rows use the tile search of the GPU mode. The CPU blob search
loses a few points of recall: 0.80 and 0.77 after the warm-up.

| Laptop CPU only, onnxruntime, mains power | Input_video2 | input_video |
|---|---|---|
| Whole pipeline after the startup (video: 30 fps) | 28.9 fps | 32.3 fps |
| Players / ball, per frame (side by side) | 28 / 25 ms | 24 / 24 ms |

On battery the same laptop is about 3 times slower.

| Kaggle T4 (4 vCPU), steady fps | Input_video2 | input11 | input_video |
|---|---|---|---|
| Ball model on TensorRT FP16 | 27.9 | 24.8 | 34.0 |
| Ball model on PyTorch FP16 | 25.5 | 20.9 | 31.6 |
| PyTorch, with the colour fallback | 24.4 | 17.9 | 28.7 |

The frames take 23–27 ms at the median, players and ball side by side. What
still costs is the search over 21 court tiles while the ball is lost.
onnxruntime-gpu did not match the CUDA of the Kaggle image and ran on the CPU
(7–18 fps).

Most missed balls are next to a player: around a contact, or by the far player.
Next to a player the difference shows the player too, while the ball in flight
is found on 76 of 78 labelled frames. Two answers:

- **More contacts in the dataset.** The first dataset had few positives there
  and many negatives around the rackets. The builder now cuts three crops for
  each frame with the ball close to a player.
- **A colour fallback** (`--fallback models/ball_tracker.pt`). The offline
  YOLOv8x runs on one 320 px crop of the plain frame, only when the ball is
  expected by a player and the motion model found nothing there: about one frame
  in four or five.
  - Its detections always go to the path search.
  - They move the crop only when they move and lie where the Kalman filter
    expects the ball. A first version that let them steer freely followed
    yellow things by the players and lost the ball in flight: recall 0.78.
  - On Input_video2 recall goes from 0.87 to 0.92, against 0.93 offline. By
    the players it finds 43 of 51 balls instead of 39, and still 49 of 49 in
    flight.
  - YOLOv8x takes 0.4 s per crop on the laptop CPU, so the fallback is meant
    for a GPU.

**Training the detector** (`notebooks/ball_motion_training_kaggle.ipynb`). The
dataset comes from the clips: `training/build_ball_dataset.py` replays them with
the online background.
- Positives come from the reliable detections of the current tracker, and from
  hand labels where it misses the ball:
  `python evaluation/annotate.py ball --frames-from <ball CSV>`.
- Negatives come from frames where the ball is known to be elsewhere.
- Input_video2, input_video and input11 are held out for testing.
- `training/pack_kaggle_inputs.py` zips the CSVs to upload next to the videos.

`evaluation/benchmark_ball.py` scores every track against the ground truth, and
the detector alone on crops around the true ball.

**Limits.**

- **Startup.** Without `--calibrate`, fitting the court takes 5–9 s on a CPU and
  the background needs another 2 s.
- **Events arrive late.** Bounces are known 1 s after they happen, shots 4 s
  after.
- **The labels come from the current tracker.** The model first learns what that
  tracker already finds, so the hand labels on the green courts are what can
  make it better.

## Project structure

```
├── pipeline.py              # one-command runner for all stages
├── realtime_pipeline.py     # the same analysis in one causal pass, at video speed
├── live_view.py             # replay with overlays, minimap and live stats
├── display_player_tracking.py  # player tracker's stages side by side, for inspection
├── tracking/                # court keypoints, player tracker, YOLO ball tracker, motion ball detector
├── realtime/                # frame source, per-frame stages, overlay and timings of the live pipeline
├── training/                # dataset, training and export of the motion ball detector
├── analysis/                # player statistics and shot detection/classification
├── motion_estimation/       # Farnebäck / Lucas–Kanade, block matching from scratch
├── evaluation/              # annotation tool and tracking metrics
├── ground_truth/            # hand-made annotations used by the evaluation
├── utils/                   # court geometry (ITF), pixel→metre homography, paths, video helpers
├── tests/                   # pytest suite (run in CI)
├── notebooks/               # Colab / Kaggle notebooks: ball detection, motion detector training
├── data/                    # demo clip
├── models/                  # put ball_tracker.pt (and ball_motion.onnx) here
└── outputs/<clip>/          # all the results of a clip, one folder per kind (not in git)
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
- **Player positions are noisy.** They come from the bottom edge of the box, which
  moves with shadows, raised feet and jumps. At the far end one pixel spans about 8 cm of
  court, so the far player's dot jumps on the minimap, and distances, summed frame by
  frame, come out too long.
- **Ball speeds are approximate.** The homography models the ground plane, while
  the ball is airborne at contact. The metre-based shot thresholds may need
  retuning for a different camera; the scale-free ones do not.
- **The ball detector depends on the court.** On hard courts in blue or purple it
  offers the moving ball on 80–90 % of the frames; on a green indoor court only 50–60 %.
  The motion candidates bring the ball to about two thirds of the frames there, against
  80–95 % elsewhere, with a few false detections: fewer bounces and flights are measured,
  and more shots rely on the older rules. The motion candidates assume a fixed camera.
- **Shots need the ball.** When the ball is lost around a contact, the shot is placed
  where it comes back, sometimes 20 frames late, and its type suffers. A shot with no
  trace of the ball is missed, and a reversal near a player can pass for a shot.
- **Thresholds are tuned on one camera setup.** The shot-type defaults were fit
  on the demo rally.
- **Serves need the server in view.** A serve hit during a fade-in, or before
  the server's box is tracked, is not detected. A clip starting during a rally with a
  player standing on the baseline can open with a false serve. Smashes have only been
  checked on one real example.
- **Some bounces cannot be seen.** When the ball passes behind a player around
  the landing, or the track has a gap there, no bounce is reported: that is the
  bounce missed on the demo rally. The median landing error is 0.12–0.23 m but
  reaches about 1 m at the far end, so this is no line-calling tool.
- **Shot types are labelled on one rally only.** On the other clips they are
  not checked against labels. Shots with no bounce found use the older rules on
  the trajectory shape, which were tuned on the demo rally and are less reliable.
  A slow topspin ball can also fly above 2 m and be called a lob. Volleys were
  only checked by eye, and a false shot can pass as one.

## Future work

- **More hand labels for the motion detector.** Its first labels are the
  current tracker's own detections, so it starts from what that tracker already
  finds; the frames it misses (green courts, fast flights) need labels by hand,
  which `annotate.py ball --frames-from` offers one after the other. A temporal
  model on several consecutive frames, as TrackNet does, is the next step.
- **Smooth the player positions.** A median and a low-order filter over the clip remove
  almost all the impossible jumps with no delay, and would correct distances and speeds.
- **Detect the shots from the events alone.** Contacts still start from the reversals
  of the ball; with a more complete track the junctions of the flight could replace them.

## Authors

Developed by **Marco Massa**, **Alfonso Antognozzi** and **Davide Cabitza** as
the project for the *Signal, Image and Video* course at the University of Trento.

Released under the [MIT License](LICENSE). The demo footage is included for
academic use only; see [`data/README.md`](data/README.md).
