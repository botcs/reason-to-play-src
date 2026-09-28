"""Participant files must agree in bytes, sample identities and source coverage."""

import json

import numpy as np
import pytest

from data.neural import (
    BINDING_FIELDS,
    bind_to_samples,
    coverage_from_missing,
    file_sha256,
    load_inputs,
    load_samples,
    read_feature_binding,
    sample_order_sha256,
    write_feature_archive,
)
from analysis.neural.encoding import encoding_input_paths
from analysis.neural.prepare_inputs import (
    find_available_ez_plays,
    find_games_and_levels,
    load_model_features_for_level,
)


def base_values():
    return {
        "subject": np.array("sub-13"),
        "tr": 2.0,
        "ar1_corrected": True,
        "voxel_ts": np.ones((2, 4)),
        "mask": np.ones((2, 1, 1), dtype=bool),
        "n_games": 1,
        "game_names": np.array(["vgfmri4_bait"]),
        "play_ids": np.array(["000000000000000000000001", "000000000000000000000002"]),
        "play_boundaries": np.array([0, 2, 4]),
        "play_n_volumes": np.array([2, 2]),
        "play_game_idx": np.array([0, 0]),
        "play_levels": np.array([0, 1]),
        "tr_run_idx": np.array([1, 1, 1, 1]),
        "tr_game_idx": np.array([0, 0, 0, 0]),
        "tr_level_idx": np.array([0, 0, 1, 1]),
        "tr_play_idx": np.array([0, 0, 1, 1]),
    }


@pytest.fixture
def participant(tmp_path, participant_writer):
    values = base_values()
    path = participant_writer(tmp_path / "sub-13", values)
    features = path / "model-features/features.npz"
    write_feature_archive(
        features,
        {"new_aligned": np.ones((4, 3))},
        path,
        coverage_from_missing(values, []),
    )
    return path, features, values


def test_valid_explicit_inputs_need_no_pickle(participant, monkeypatch):
    path, features, values = participant
    original = np.load

    def checked(*args, **kwargs):
        assert kwargs.get("allow_pickle") is False
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", checked)
    loaded = load_inputs(path, features, "new")
    np.testing.assert_array_equal(loaded["voxel_ts"], values["voxel_ts"])
    assert json.loads(loaded["alignment_verification_json"])[-1]["status"] == "verified"


@pytest.mark.parametrize("archive", ["bold", "samples", "nuisance", "features"])
def test_swapping_equal_shape_bytes_is_rejected(participant, archive):
    path, features, _ = participant
    changed = features if archive == "features" else path / f"{archive}.npz"
    with np.load(changed, allow_pickle=False) as source:
        arrays = dict(source)
    if archive == "bold":
        arrays["voxel_ts"] *= 2
    elif archive == "samples":
        arrays["play_ids"] = arrays["play_ids"][::-1]
    elif archive == "nuisance":
        arrays["scores"] = np.zeros(4)
    else:
        arrays["new_aligned"] *= 2
    np.savez(changed, **arrays)
    with pytest.raises(ValueError, match="different"):
        load_inputs(path, features, "new")


def test_samples_sha_is_distinct_from_order_digest(participant):
    path, features, values = participant
    document = read_feature_binding(features, path)
    assert document["samples_sha256"] == file_sha256(path / "samples.npz")
    assert document["sample_order_sha256"] == sample_order_sha256(values)
    assert document["samples_sha256"] != document["sample_order_sha256"]
    with np.load(features, allow_pickle=False) as source:
        assert set(BINDING_FIELDS).issubset(source.files)
        assert "alignment_base_sha256" not in source.files


def test_reissued_bold_association_does_not_rebind_features(participant):
    path, features, values = participant
    samples_path = path / "samples.npz"
    with np.load(samples_path, allow_pickle=False) as source:
        samples = dict(source)
    samples["play_ids"] = samples["play_ids"][::-1]
    np.savez(samples_path, **samples)
    association = path / "bold.npz.alignment.json"
    document = json.loads(association.read_text())
    document.update(
        samples_sha256=file_sha256(samples_path),
        sample_order_sha256=sample_order_sha256(samples),
    )
    association.write_text(json.dumps(document))
    load_samples(path)
    with pytest.raises(ValueError, match="different samples_sha256"):
        load_inputs(path, features, "new")


@pytest.mark.parametrize(
    "archive", ["bold.npz", "nuisance.npz", "model-features/features.npz"]
)
def test_every_required_association_must_be_present(participant, archive):
    path, features, _ = participant
    (path / f"{archive}.alignment.json").unlink()
    with pytest.raises(ValueError, match="Download"):
        load_inputs(path, features, "new")


def test_embedded_binding_cannot_disagree_with_external(participant):
    path, features, values = participant
    binding = bind_to_samples(path)
    binding["alignment_sample_order_sha256"] = np.array("0" * 64)
    np.savez(features, new_aligned=np.ones((4, 3)), **binding)
    association = features.with_name(features.name + ".alignment.json")
    doc = json.loads(association.read_text())
    doc["feature_sha256"] = file_sha256(features)
    association.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="different sample_order_sha256"):
        load_inputs(path, features, "new")


def test_payload_can_have_external_binding_without_embedded_metadata(participant):
    path, features, _ = participant
    np.savez(features, new_aligned=np.ones((4, 3)))
    association = features.with_name(features.name + ".alignment.json")
    doc = json.loads(association.read_text())
    doc["feature_sha256"] = file_sha256(features)
    association.write_text(json.dumps(doc))
    read_feature_binding(features, path)
    load_inputs(path, features, "new")


def test_canonical_dataset_lookup(tmp_path, participant_writer):
    root = participant_writer(tmp_path / "neural/sub-13", base_values())
    features = root / "model-features/lrm/qwen3.5-27b/minimal--all--main.npz"
    features.parent.mkdir(parents=True)
    features.touch()
    assert encoding_input_paths(
        "sub-13", tmp_path, "llm_qwen35_27b_sugmin__all__main_layer_1"
    ) == [root / "bold.npz", root / "samples.npz", root / "nuisance.npz", features]


def test_baseline_canonical_paths_preserve_original_play_ids(tmp_path):
    ddqn = tmp_path / "ddqn/sub-13/avoidGeorge_vgfmri4/level-00.npz"
    ddqn.parent.mkdir(parents=True)
    np.savez(ddqn, num_plays=0)
    assert find_games_and_levels(tmp_path / "ddqn", "sub-13") == {
        "vgfmri4_avoidgeorge": [0]
    }
    assert (
        load_model_features_for_level(
            tmp_path / "ddqn", "sub-13", "vgfmri4_avoidgeorge", 0
        )
        == {}
    )
    play_id = "000000000000000000000001"
    ez = tmp_path / "ez/avoidGeorge_vgfmri4/sub-13/run-02" / f"play-{play_id}/traces.pt"
    ez.parent.mkdir(parents=True)
    ez.touch()
    available, by_id = find_available_ez_plays(tmp_path / "ez", "sub-13")
    assert by_id
    assert available == {play_id: (ez, "vgfmri4_avoidgeorge", 2)}


def test_original_mixed_case_game_resolves_model_tensor(tmp_path):
    from analysis.neural.align_llm import _multiturn_file_for

    feature = tmp_path / "sub-09/plaqueAttack_vgfmri3.pt"
    feature.parent.mkdir(parents=True)
    feature.touch()
    assert _multiturn_file_for(tmp_path, "sub-09", "vgfmri3_plaqueAttack") == feature
    assert _multiturn_file_for(tmp_path, "sub-09", "vgfmri3_plaqueattack") == feature


def test_loading_one_layer_does_not_materialize_other_layers(participant, monkeypatch):
    from numpy.lib.npyio import NpzFile

    path, features, values = participant
    write_feature_archive(
        features,
        {"new_aligned": np.ones((4, 3)), "other_aligned": np.ones((4, 2000))},
        path,
        coverage_from_missing(values, []),
    )
    original = NpzFile.__getitem__

    def checked(self, key):
        if key == "other_aligned":
            raise AssertionError("Unrequested activation matrix was loaded")
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", checked)
    loaded = load_inputs(path, features, "new")
    assert loaded["new_aligned"].shape == (4, 3)
    assert "other_aligned" not in loaded


@pytest.mark.parametrize(
    "name",
    [
        "bold.npz",
        "samples.npz",
        "nuisance.npz",
        "bold.npz.alignment.json",
        "nuisance.npz.alignment.json",
    ],
)
@pytest.mark.parametrize(
    "alias", ["same-path", "symlink", "hardlink", "association-symlink"]
)
def test_feature_writer_preserves_all_participant_inputs(
    participant, name, alias, tmp_path
):
    path, _, values = participant
    protected = path / name
    original = {
        str(item): item.read_bytes() for item in path.iterdir() if item.is_file()
    }
    destination = protected if alias == "same-path" else tmp_path / "new-features.npz"
    if alias == "symlink":
        destination.symlink_to(protected)
    elif alias == "hardlink":
        destination.hardlink_to(protected)
    elif alias == "association-symlink":
        destination.with_name(destination.name + ".alignment.json").symlink_to(
            protected
        )
    with pytest.raises(ValueError, match="must not overwrite participant inputs"):
        write_feature_archive(
            destination,
            {"new_aligned": np.ones((4, 3))},
            path,
            coverage_from_missing(values, []),
        )
    for filename, contents in original.items():
        from pathlib import Path

        assert Path(filename).read_bytes() == contents
