"""Feature identity stays explicit even when nearby plays share clock ranges."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from reason_to_play.fmri import align_llm as align


def original(identity, ordinal, start=1_600_000_000.0):
    return {
        "_id": identity,
        "subj_id": 13,
        "run_id": 1,
        "game_name": "vgfmri4_bait",
        "level_id": 0,
        "_canonical": {
            "source_document_index": ordinal,
            "source_recording": "sub-13/bait_vgfmri4/elaborate.human.replay.json.gz",
        },
        "states": [{"ts": start + 2 * frame} for frame in range(5)],
    }


def feature(play, frames=(0, 2, 4), *, linked=True):
    references = [
        {
            "source_play_id": play["_id"],
            "source_recording": play["_canonical"]["source_recording"],
            "source_frame_index": frame,
            "source_document_index": play["_canonical"]["source_document_index"],
        }
        for frame in frames
    ]
    return {
        "timestamps": np.asarray([play["states"][i]["ts"] for i in frames]),
        "activations": {"layer_1": np.arange(len(frames)).reshape(-1, 1)},
        "metadata": {
            "play_id": "legacy composite display identity",
            "source_references": references if linked else None,
        },
    }


def test_exact_ids_disambiguate_overlapping_plays_and_missing_features():
    plays = {"a": original("a", 0), "idle": original("idle", 1), "b": original("b", 2)}
    features = {0: feature(plays["b"]), 1: feature(plays["a"])}
    matched = align.match_multiturn_plays(features, plays)
    assert list(matched) == ["b", "a"]
    assert matched["a"] is features[1]
    assert "idle" not in matched


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda f: f["metadata"]["source_references"][0].update(source_play_id="b"),
            "conflicting original",
        ),
        (
            lambda f: [
                r.update(source_play_id="missing")
                for r in f["metadata"]["source_references"]
            ],
            "absent",
        ),
        (
            lambda f: f["metadata"]["source_references"][0].update(
                source_recording="sub-12/bait_vgfmri4/elaborate.human.replay.json.gz"
            ),
            "recording/document",
        ),
        (
            lambda f: f["metadata"]["source_references"][0].update(
                source_document_index=1
            ),
            "recording/document",
        ),
        (
            lambda f: f["metadata"]["source_references"][0].update(
                source_frame_index=99
            ),
            "frame index",
        ),
        (
            lambda f: f["metadata"]["source_references"][0].update(
                source_frame_index=True
            ),
            "frame index",
        ),
        (
            lambda f: f["metadata"]["source_references"][0].pop("source_frame_index"),
            "complete source",
        ),
        (lambda f: f["metadata"]["source_references"].pop(), "complete source"),
        (
            lambda f: f["timestamps"].__setitem__(0, f["timestamps"][0] + 0.01),
            "timestamps differ",
        ),
        (
            lambda f: f["metadata"]["source_references"][0].update(
                source_frame_index=2
            ),
            "Duplicate original",
        ),
    ],
)
def test_linked_reference_errors_fail_before_alignment(change, match):
    plays = {"a": original("a", 0)}
    candidate = feature(plays["a"])
    change(candidate)
    with pytest.raises(ValueError, match=match):
        align.match_multiturn_plays({0: candidate}, plays)


def test_legacy_matching_is_unique_in_both_directions():
    a, b = original("a", 0), original("b", 1, start=1_600_000_030.0)
    candidate = feature(a, linked=False)
    matched = align.match_multiturn_plays({99: candidate}, {"b": b, "a": a})
    assert matched == {"a": candidate}
    with pytest.raises(ValueError, match="multiple original plays"):
        align.match_multiturn_plays({0: candidate}, {"a": a, "b": original("b", 1)})
    with pytest.raises(ValueError, match="Multiple feature groups"):
        align.match_multiturn_plays({0: candidate, 1: deepcopy(candidate)}, {"a": a})


def test_subsampling_retains_matching_frame_references():
    candidate = feature(original("a", 0), frames=(0, 1, 2, 3, 4))
    manager = align.LLMFeatureManager([], "sub-13")
    manager._apply_subsampling({0: candidate}, 2)
    assert [
        r["source_frame_index"] for r in candidate["metadata"]["source_references"]
    ] == [0, 2, 4]
    np.testing.assert_array_equal(
        candidate["activations"]["layer_1"].ravel(), [0, 2, 4]
    )
    assert (
        align.match_multiturn_plays({0: candidate}, {"a": original("a", 0)})["a"]
        is candidate
    )


def test_multiturn_file_to_aligned_rows_uses_original_identity(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    plays = {"a": original("a", 0), "b": original("b", 1)}
    monkeypatch.setattr(align, "load_canonical_plays", lambda *args: plays)
    monkeypatch.setattr(
        align,
        "_behavior_api",
        lambda: SimpleNamespace(
            play_states=lambda play: play["states"],
            load_runs=lambda root: {(13, 1): {"scan_start_ts": 1_600_000_000.0}},
        ),
    )
    np.savez(
        tmp_path / "base.npz",
        tr=2.0,
        ar1_corrected=False,
        n_volumes=10,
        game_names=["vgfmri4_bait"],
        play_ids=["a", "b"],
        play_game_idx=[0, 0],
        play_levels=[0, 0],
        play_n_volumes=[5, 5],
    )
    metadata, values = [], []
    # Reverse extraction order while BOLD order stays a, b; times overlap exactly.
    for trial, (identity, value) in enumerate([("b", 7), ("a", 5)]):
        row = feature(plays[identity], frames=range(5))
        for frame, references in enumerate(row["metadata"]["source_references"]):
            metadata.append(
                {
                    **references,
                    "play_id": f"sub-13_1_{plays[identity]['_canonical']['source_document_index']}",
                    "level_id": 0,
                    "trial_idx": trial,
                    "realworld_ts": row["timestamps"][frame],
                }
            )
            values.append([[value]])
    feature_dir = tmp_path / "features/sub-13"
    feature_dir.mkdir(parents=True)
    torch.save(
        {
            "session": {"num_layers": 1},
            "metadata": metadata,
            "features": torch.tensor(values, dtype=torch.float32),
        },
        feature_dir / "bait_vgfmri4.pt",
    )
    source = align.parse_llm_source_arg(
        f"name=fixture,dir={tmp_path / 'features'},stream=main"
    )
    result = align.process_subject(
        "sub-13",
        tmp_path / "base.npz",
        behavior_dir=tmp_path / "canonical",
        output_dir=tmp_path / "output",
        llm_sources=[source],
    )
    with np.load(result[0]) as output:
        np.testing.assert_array_equal(
            output["llm_fixture_layer_1_aligned"].ravel(), [5] * 5 + [7] * 5
        )


def test_per_game_file_references_keep_original_run_ordinals():
    path = "sub-13/bait_vgfmri4/elaborate.human.replay.json.gz"
    first = original("a", 4)
    second = original("b", 1)
    second["run_id"] = 5
    for play in (first, second):
        play["_canonical"]["source_recording"] = path
    features = {}
    for index, play in enumerate((second, first)):
        row = feature(play)
        for reference in row["metadata"]["source_references"]:
            reference["source_recording"] = path
        features[index] = row
    # Times overlap: only original IDs disambiguate plays from different runs.
    result = align.match_multiturn_plays(features, {"a": first, "b": second})
    assert list(result) == ["b", "a"]
    for condition in ("minimal", "oracle"):
        variant = deepcopy(features[0])
        for reference in variant["metadata"]["source_references"]:
            reference["source_recording"] = path.replace("elaborate.", condition + ".")
        assert align.match_multiturn_plays({0: variant}, {"b": second}) == {
            "b": variant
        }
    for invalid in (
        "plays/sub-13/run-01.json.gz",
        "plays/sub-13/run-05.json.gz",
        "sub-13/other/../bait_vgfmri4/elaborate.human.replay.json.gz",
        path.replace("sub-13", "sub-12"),
        path.replace("bait", "zelda"),
        path.replace("elaborate.", "unknown."),
    ):
        bad = deepcopy(features[0])
        for reference in bad["metadata"]["source_references"]:
            reference["source_recording"] = invalid
        with pytest.raises(ValueError, match="recording/document"):
            align.match_multiturn_plays({0: bad}, {"b": second})
    bad = deepcopy(features[1])
    bad["metadata"]["source_references"][0]["source_document_index"] = 0
    with pytest.raises(ValueError, match="recording/document"):
        align.match_multiturn_plays({0: bad}, {"a": first})


@pytest.mark.parametrize(
    "path",
    [
        None,
        "plays/sub-13/run-01.json.gz",
        "sub-12/bait_vgfmri4/elaborate.human.replay.json.gz",
        "sub-13/helper_vgfmri4/elaborate.human.replay.json.gz",
    ],
)
def test_referenced_features_require_actual_per_game_human_path(path):
    play = original("a", 0)
    candidate = feature(play)
    play["_canonical"]["source_recording"] = path
    with pytest.raises(ValueError, match="source_recording"):
        align.match_multiturn_plays({0: candidate}, {"a": play})
    # Features containing only timestamps retain their independently validated
    # unique-play matching rule; they do not invent a missing path.
    candidate["metadata"]["source_references"] = None
    assert align.match_multiturn_plays({0: candidate}, {"a": play}) == {"a": candidate}
