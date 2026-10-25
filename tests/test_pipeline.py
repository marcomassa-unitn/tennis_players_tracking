import os

from pipeline import derive_paths


def test_paths_derive_from_video_stem():
    p = derive_paths("data/my_match.mp4", "outputs/")
    assert p["stem"] == "my_match"
    assert p["output"] == "outputs"
    assert p["court_csv"] == os.path.join("outputs", "court_coordinates", "my_match_court.csv")
    assert p["players_csv"] == os.path.join("outputs", "player_coordinates", "players_my_match.csv")
    assert p["ball_csv"] == os.path.join("outputs", "ball_coordinates", "ball_my_match.csv")


def test_custom_output_dir():
    p = derive_paths("clip.mp4", "results")
    assert p["shots_dir"] == os.path.join("results", "shot_analysis")
    assert p["eval_dir"] == os.path.join("results", "evaluation")
