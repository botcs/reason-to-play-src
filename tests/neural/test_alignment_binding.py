"""Feature rows must belong to the supplied base, even when dimensions match."""

import json

import numpy as np
import pytest

from analysis.neural.alignment import (
    bind_to_base,
    file_sha256,
    sample_order_sha256,
)
from analysis.neural.encoding import (
    encoding_input_paths,
    load_aligned_data,
)
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


def test_same_size_wrong_base_is_rejected(tmp_path):
    values = base_values()
    base, other, sidecar = (
        tmp_path / name for name in ("base.npz", "other.npz", "features.npz")
    )
    np.savez(base, **values)
    np.savez(sidecar, new_aligned=np.ones((4, 3)), **bind_to_base(base, values))
    loaded = load_aligned_data([base, sidecar], "sub-13", "new", require_binding=True)
    assert '"status": "verified"' in loaded["alignment_verification_json"]
    values["voxel_ts"] = np.zeros((2, 4))
    np.savez(other, **values)
    assert base.stat().st_size == other.stat().st_size
    with pytest.raises(ValueError, match="different base archive"):
        load_aligned_data([other, sidecar], "sub-13", "new", require_binding=True)


def test_sample_order_tampering_is_rejected(tmp_path):
    values = base_values()
    base, sidecar = tmp_path / "base.npz", tmp_path / "features.npz"
    np.savez(base, **values)
    binding = bind_to_base(base, values)
    reordered = {**values, "play_ids": values["play_ids"][::-1]}
    binding["alignment_samples_sha256"] = sample_order_sha256(reordered)
    np.savez(sidecar, new_aligned=np.ones((4, 3)), **binding)
    with pytest.raises(ValueError, match="different scanner sample order"):
        load_aligned_data([base, sidecar], "sub-13", "new", require_binding=True)


def test_unbound_inputs_require_explicit_import_mode(tmp_path):
    values = base_values()
    base, sidecar = tmp_path / "base.npz", tmp_path / "features.npz"
    np.savez(base, **values)
    np.savez(sidecar, new_aligned=np.ones((4, 3)))
    with pytest.raises(ValueError, match="lacks a recorded"):
        load_aligned_data([base, sidecar], "sub-13", "new", require_binding=True)
    loaded = load_aligned_data([base, sidecar], "sub-13", "new", require_binding=False)
    assert '"status": "unverified"' in loaded["alignment_verification_json"]


def test_canonical_dataset_lookup(tmp_path):
    root = tmp_path / "analysis/neural/inputs"
    base = root / "sub-13/bold-ddqn-theory.npz"
    features = root / "model-features/lrm/qwen3.5-27b/minimal/all/main/sub-13.npz"
    for path in (base, features):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert encoding_input_paths(
        "sub-13", tmp_path, "llm_qwen35_27b_sugmin__all__main_layer_1"
    ) == [base, features]


def test_external_binding_pins_original_payload_bytes(tmp_path):
    values = base_values()
    base, sidecar = tmp_path / "base.npz", tmp_path / "features.npz"
    np.savez(base, **values)
    np.savez(sidecar, new_aligned=np.ones((4, 3)))
    document = {
        "schema": "reason-to-play/alignment-binding",
        "schema_version": 1,
        "feature_sha256": file_sha256(sidecar),
        "base_sha256": file_sha256(base),
        "sample_order_sha256": sample_order_sha256(values),
        "verification": {"status": "verified", "method": "numerical-realignment"},
    }
    association = sidecar.with_name(sidecar.name + ".alignment.json")
    association.write_text(json.dumps(document))
    assert (
        '"status": "verified"'
        in load_aligned_data([base, sidecar], "sub-13", "new")[
            "alignment_verification_json"
        ]
    )
    np.savez(sidecar, new_aligned=np.zeros((4, 3)))
    with pytest.raises(ValueError, match="different feature bytes"):
        load_aligned_data([base, sidecar], "sub-13", "new")


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
