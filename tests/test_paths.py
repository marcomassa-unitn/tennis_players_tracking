import os

from utils.paths import clip_paths, ground_truth_path


def test_every_result_of_a_clip_is_in_its_folder():
    p = clip_paths("data/my_match.mp4", "outputs/")
    clip = os.path.join("outputs", "my_match")
    assert p["stem"] == "my_match"
    assert p["output"] == "outputs"
    assert p["clip_dir"] == clip
    assert p["court_csv"] == os.path.join(clip, "court_coordinates", "my_match_court.csv")
    assert p["players_csv"] == os.path.join(clip, "player_coordinates", "players_my_match.csv")
    assert p["ball_csv"] == os.path.join(clip, "ball_coordinates", "ball_my_match.csv")
    assert p["shots_csv"] == os.path.join(clip, "shot_analysis", "shots.csv")
    paths = [v for k, v in p.items() if k.endswith(("_csv", "_dir"))]
    assert all(path.startswith(clip) for path in paths)


def test_two_clips_share_nothing():
    a, b = clip_paths("a.mp4"), clip_paths("b.mp4")
    keys = [k for k in a if k.endswith(("_csv", "_dir"))]
    assert not {a[k] for k in keys} & {b[k] for k in keys}


def test_custom_output_dir():
    p = clip_paths("clip.mp4", "results")
    assert p["shots_dir"] == os.path.join("results", "clip", "shot_analysis")
    assert p["eval_dir"] == os.path.join("results", "clip", "evaluation")


def test_ground_truth_stays_out_of_the_outputs():
    assert ground_truth_path("bounce", "data/my_match.mp4") == os.path.join(
        "ground_truth", "bounce_my_match_gt.csv")
