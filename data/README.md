# Data

`input2.mp4` is the demo clip used throughout the project: a single rally
from a singles match, filmed by a fixed broadcast camera (1008 frames, 30 fps).

It is excerpted from broadcast footage and included only for academic,
non-commercial use, so the pipeline can be reproduced. All rights to the
footage remain with their owners. The other 21 clips used for the evaluation
(`input1`, `input3` ... `input22`, 6.8-39.9 s each, several tournaments on hard, clay
and grass courts) are not distributed for the same reason; their hand labels
are in `ground_truth/`.

To analyse your own clip, drop it here and pass `--video data/<your_clip>.mp4`.
The pipeline assumes a fixed camera, so trim replays and camera cuts first. If a
player is left-handed, add a line to `ground_truth/hands.csv`.
