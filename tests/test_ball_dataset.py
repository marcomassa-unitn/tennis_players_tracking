import cv2
import numpy as np
import pandas as pd

from training.build_ball_dataset import (
    TEST_CLIPS,
    ClipWriter,
    pseudo_labels,
    refine_box_from_diff,
    split_of,
)


def test_test_clips_are_never_written_and_val_is_by_clip():
    for clip in TEST_CLIPS:
        assert split_of(clip, 10, 1000, 30.0) is None
    assert split_of("input4", 10, 1000, 30.0) == "val"
    assert split_of("input3", 10, 1000, 30.0) == "train"
    # input10: its tail is val, with a gap of untouched frames before it
    assert split_of("input10", 100, 1000, 30.0) == "train"
    assert split_of("input10", 740, 1000, 30.0) is None
    assert split_of("input10", 800, 1000, 30.0) == "val"


def test_pseudo_labels_keep_long_runs_only(tmp_path):
    rows = []
    for f in list(range(0, 8)) + [20, 21, 22] + list(range(40, 46)):
        rows.append([f, 100 + f, 200, 14, 14, 107.0 + f, 207.0, 196, 0])
    rows.append([8, 0, 0, 14, 14, 7.0, 7.0, 196, 1])                 # interpolated
    ball = tmp_path / "ball.csv"
    pd.DataFrame(rows, columns=["frame", "x", "y", "w", "h", "cx", "cy", "area",
                                "interpolated"]).to_csv(ball, index=False)
    # the detector found frames 0-7; 40-45 come from motion blobs only (6 < 8)
    cands = tmp_path / "cands.csv"
    pd.DataFrame([[f, 100 + f, 200, 114 + f, 214, 0.8, 0.5, "tile"] for f in range(8)],
                 columns=["frame", "x1", "y1", "x2", "y2", "conf", "motion", "source"]
                 ).to_csv(cands, index=False)
    labels = pseudo_labels(str(ball), str(cands), 1080)
    assert sorted(labels) == list(range(8))
    assert labels[3] == (110.0, 207.0, 14.0, 14.0)


def test_box_sides_are_clamped(tmp_path):
    rows = [[f, 100, 200, 60, 4, 130.0, 202.0, 240, 0] for f in range(6)]
    ball = tmp_path / "ball.csv"
    pd.DataFrame(rows, columns=["frame", "x", "y", "w", "h", "cx", "cy", "area",
                                "interpolated"]).to_csv(ball, index=False)
    cands = tmp_path / "c.csv"
    pd.DataFrame([[f, 100, 200, 160, 204, 0.8, 0.5, "tile"] for f in range(6)],
                 columns=["frame", "x1", "y1", "x2", "y2", "conf", "motion", "source"]
                 ).to_csv(cands, index=False)
    assert pseudo_labels(str(ball), str(cands), 1080)[0][2:] == (40.0, 10.0)


def test_the_box_of_a_click_comes_from_the_difference_blob():
    diff = np.zeros((100, 100), np.uint8)
    cv2.rectangle(diff, (40, 45), (51, 52), 200, -1)            # a 12 x 8 streak
    assert refine_box_from_diff(diff, 45, 48, (16, 16)) == (12.0, 8.0)
    assert refine_box_from_diff(diff, 10, 10, (16, 16)) == (16, 16)   # nothing there


def test_labels_are_normalised_yolo_and_images_lossless(tmp_path):
    w = ClipWriter(str(tmp_path), "clip", np.random.default_rng(0))
    crop = np.random.default_rng(1).integers(0, 256, (320, 320, 3), dtype=np.uint8)
    w.write("train", 7, 0, crop, (160.0, 80.0, 16.0, 32.0))
    w.write("train", 7, 1, crop, None)
    assert (tmp_path / "labels/train/clip_00007_0.txt").read_text().split() == \
        ["0", "0.500000", "0.250000", "0.050000", "0.100000"]
    assert (tmp_path / "labels/train/clip_00007_1.txt").read_text() == ""
    back = cv2.imread(str(tmp_path / "images/train/clip_00007_0.png"))
    assert np.array_equal(back, crop)
    assert w.count["train"] == [1, 1]
