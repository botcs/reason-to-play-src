"""Keep declared source gaps separate from alignment proof and the fitting mask."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from reason_to_play.analysis.neural.alignment import file_sha256, sample_order_sha256
from reason_to_play.analysis.neural.encoding import load_aligned_data


@pytest.fixture
def incomplete_features(tmp_path):
    rng = np.random.default_rng(31)
    lengths = np.full(6, 15)
    levels = np.array([0, 1, 3, 4, 6, 7])
    base = {
        "subject": np.array("sub-13"),
        "tr": 2.0,
        "ar1_corrected": True,
        "voxel_ts": rng.normal(size=(2, 90)).astype(np.float32),
        "n_volumes": 90,
        "n_games": 1,
        "game_names": np.array(["fixture"]),
        "play_ids": np.array([f"{index:024x}" for index in range(6)]),
        "play_boundaries": np.r_[0, np.cumsum(lengths)],
        "play_n_volumes": lengths,
        "play_levels": levels,
        "play_game_idx": np.zeros(6, dtype=int),
        "tr_run_idx": np.ones(90, dtype=int),
        "tr_game_idx": np.zeros(90, dtype=int),
        "tr_level_idx": np.repeat(levels, lengths),
        "tr_play_idx": np.repeat(np.arange(6), lengths),
        "mask": np.ones((2, 1, 1), dtype=bool),
        "mask_affine": np.eye(4),
    }
    base_path, features_path = tmp_path / "base.npz", tmp_path / "features.npz"
    np.savez(base_path, **base)
    features = rng.normal(size=(90, 3)).astype(np.float32)
    features[:15] = 0
    # The established numeric mask also excludes nonzero vectors summing to zero.
    # Do not silently replace that method with declared source coverage.
    features[15] = [1, -1, 0]
    np.savez(features_path, new_aligned=features)
    coverage = {
        "complete": False,
        "retained_sample_count": 90,
        "missing_feature_sample_count": 15,
        "missing_feature_play_ids": [str(base["play_ids"][0])],
        "missing_feature_sample_intervals": [
            {
                "play_id": str(base["play_ids"][0]),
                "sample_start": 0,
                "sample_stop": 15,
                "game": "fixture",
                "level": 0,
            }
        ],
        "missing_feature_policy": "zero-fill-original-aligned-samples",
        "absent_game_files": [],
    }
    document = {
        "schema": "reason-to-play/alignment-binding",
        "schema_version": 1,
        "feature_sha256": file_sha256(features_path),
        "base_sha256": file_sha256(base_path),
        "sample_order_sha256": sample_order_sha256(base),
        "verification": {"status": "verified"},
        "feature_coverage": coverage,
    }
    association_path = Path(str(features_path) + ".alignment.json")
    association_path.write_text(json.dumps(document))
    return base_path, features_path, association_path, document


def test_coverage_survives_real_fit_and_resume_checks_association(
    incomplete_features, tmp_path
):
    base, features, association, document = incomplete_features
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "reason_to_play.analysis.neural.encoding",
        "--subject",
        "sub-13",
        "--base-data",
        str(base),
        "--feature-file",
        str(features),
        "--layer",
        "new",
        "--output-dir",
        str(tmp_path / "fit"),
        "--max-level",
        "6",
        "--seed",
        "7",
        "--n-iter",
        "1",
        "--n-targets-batch",
        "2",
        "--n-alphas-batch",
        "4",
    ]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(command, cwd=tmp_path, env=env, check=True, capture_output=True)
    result_path = tmp_path / "fit/sub-13/encoding_results_new.npz"
    with np.load(result_path, allow_pickle=False) as result:
        evidence = json.loads(result["alignment_verification_json"].item())[0]
        assert evidence["status"] == "verified"
        assert evidence["feature_coverage"] == document["feature_coverage"]
        assert evidence["coverage_sample_order"] == "original-base-archive"
        assert result["n_volumes"].item() == 75
        assert result["n_valid_volumes"].item() == 59
        assert result["valid_sample_policy"].item() == (
            "feature-row-sum-nonzero-after-level-selection-and-shuffle"
        )
        assert {
            "role": "alignment-association",
            "sha256": file_sha256(association),
        } in json.loads(result["input_files_json"].item())
    subprocess.run(
        command + ["--resume"], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    # Byte identity includes coverage evidence, even when the arrays are unchanged.
    document["feature_coverage"]["evidence_note"] = "additional source audit"
    association.write_text(json.dumps(document))
    resumed = subprocess.run(
        command + ["--resume"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert resumed.returncode != 0
    assert "Cannot resume mismatched" in resumed.stderr


@pytest.mark.parametrize(
    "defect", ["count", "interval", "play", "duplicate", "complete", "boolean-count"]
)
def test_inconsistent_coverage_is_rejected_before_fitting(incomplete_features, defect):
    base, features, association, document = incomplete_features
    coverage = document["feature_coverage"]
    if defect == "count":
        coverage["retained_sample_count"] = 89
    elif defect == "interval":
        coverage["missing_feature_sample_intervals"][0]["sample_stop"] = 16
    elif defect == "play":
        coverage["missing_feature_play_ids"] = ["not-in-base"]
    elif defect == "duplicate":
        coverage["missing_feature_sample_intervals"] *= 2
    elif defect == "complete":
        coverage["complete"] = True
    else:
        coverage["missing_feature_sample_count"] = True
    association.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Feature coverage"):
        load_aligned_data([base, features], "sub-13", "new")


def test_absent_coverage_is_unknown_not_complete(incomplete_features):
    base, features, association, document = incomplete_features
    del document["feature_coverage"]
    association.write_text(json.dumps(document))
    result = load_aligned_data([base, features], "sub-13", "new")
    evidence = json.loads(result["alignment_verification_json"])[0]
    assert evidence["status"] == "verified"
    assert evidence["feature_coverage"] is None


def test_absent_game_without_retained_samples_is_incomplete(incomplete_features):
    base, features, association, document = incomplete_features
    coverage = document["feature_coverage"]
    coverage.update(
        missing_feature_play_ids=[],
        missing_feature_sample_intervals=[],
        missing_feature_sample_count=0,
        absent_game_files=["game-with-no-retained-samples"],
    )
    association.write_text(json.dumps(document))
    result = load_aligned_data([base, features], "sub-13", "new")
    recorded = json.loads(result["alignment_verification_json"])[0]["feature_coverage"]
    assert recorded["complete"] is False
    assert recorded["missing_feature_sample_count"] == 0
    coverage["complete"] = True
    association.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Feature coverage completeness"):
        load_aligned_data([base, features], "sub-13", "new")
    # Older declarations need not contain the optional game inventory.
    del coverage["absent_game_files"]
    association.write_text(json.dumps(document))
    load_aligned_data([base, features], "sub-13", "new")


def test_absent_games_must_be_a_list(incomplete_features):
    base, features, association, document = incomplete_features
    document["feature_coverage"]["absent_game_files"] = "game-name"
    association.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Feature coverage absent_game_files"):
        load_aligned_data([base, features], "sub-13", "new")
