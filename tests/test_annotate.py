from evaluation.annotate import SHOT_KEYS, load_shot_labels, save_shot_labels


def test_shot_labels_round_trip_and_resume(tmp_path):
    path = tmp_path / "gt" / "shots_clip_gt.csv"
    assert load_shot_labels(str(path)) == {}                     # nothing yet: start over
    labels = {120: (1, "lob"), 45: (2, "flat"), 300: (2, "not_a_shot")}
    save_shot_labels(str(path), labels)
    assert load_shot_labels(str(path)) == labels
    assert path.read_text().splitlines()[:2] == ["frame,player_id,label", "45,2,flat"]


def test_every_shot_type_of_the_pipeline_has_a_key():
    labels = set(SHOT_KEYS.values())
    assert {"flat", "slice", "dropshot", "lob", "volley"} <= labels   # shot_type values
    assert {"smash", "serve"} <= labels                              # overhead values
    assert len(labels) == len(SHOT_KEYS)                             # one key per label
