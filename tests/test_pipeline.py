import os
from argparse import Namespace

import pytest

from pipeline import REDO_STEPS, choose_steps, main, missing_steps, parse_steps, stale_steps
from utils.paths import clip_paths


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def _args(redo=None, **skips):
    return Namespace(redo=redo, **{f"skip_{name}": skips.get(name, False) for name in REDO_STEPS})


def _redone(args):
    return {name for name in REDO_STEPS if not getattr(args, f"skip_{name}")}


@pytest.fixture
def clip(tmp_path):
    """A clip with court and players done, and nothing else."""
    p = clip_paths("data/my_match.mp4", str(tmp_path))
    _touch(p["court_csv"])
    _touch(p["players_csv"])
    return p


def test_missing_steps_reads_the_clip_folder(clip):
    assert missing_steps(clip) == {"analysis", "motion", "ball", "shots"}


@pytest.mark.parametrize("text, expected", [
    ("", {"ball"}),
    ("missing", {"ball"}),
    ("none", set()),
    ("all", set(REDO_STEPS)),
    ("2,6", {"tracking", "shots"}),
    ("2 6", {"tracking", "shots"}),
    ("Tracking, shots", {"tracking", "shots"}),
])
def test_parse_steps(text, expected):
    assert parse_steps(text, {"ball"}) == expected


@pytest.mark.parametrize("text", ["7", "0", "players"])
def test_parse_steps_rejects_unknown_steps(text):
    with pytest.raises(ValueError):
        parse_steps(text, set())


def test_stale_steps_lists_what_reads_the_results():
    assert stale_steps({"tracking"}) == ["analysis", "motion", "shots"]
    assert stale_steps({"ball"}) == ["shots"]
    assert stale_steps({"court", "motion"}) == ["tracking", "analysis", "shots"]
    assert stale_steps({"shots"}) == []


def test_redo_flag_sets_exactly_those_steps(clip):
    args = _args(redo="tracking,shots")
    choose_steps(clip, args, interactive=True)
    assert _redone(args) == {"tracking", "shots"}


def test_without_a_terminal_only_the_missing_steps_run(clip):
    args = _args()
    choose_steps(clip, args, interactive=False)
    assert _redone(args) == {"analysis", "motion", "ball", "shots"}


def test_skip_flags_turn_the_question_off(clip):
    args = _args(court=True)
    choose_steps(clip, args, interactive=True)
    assert _redone(args) == set(REDO_STEPS) - {"court"}


def test_the_question_offers_the_steps_reading_the_results(clip, monkeypatch):
    answers = iter(["2", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    args = _args()
    choose_steps(clip, args, interactive=True)
    assert _redone(args) == {"tracking", "analysis", "motion", "shots"}


def test_declining_keeps_only_the_chosen_step(clip, monkeypatch):
    answers = iter(["oops", "2", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    args = _args()
    choose_steps(clip, args, interactive=True)
    assert _redone(args) == {"tracking"}


def test_a_missing_video_stops_the_pipeline_before_any_folder(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    monkeypatch.setattr("sys.argv", ["pipeline", "--video", "data/nope.mp4", "--output", str(out)])
    with pytest.raises(SystemExit, match="Video not found"):
        main()
    assert not out.exists()
