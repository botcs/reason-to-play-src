"""Encoding results must preserve input identities, fits and partition labels."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT
        / "src/reason_to_play/analysis/neural"
        / {"encoding_model.py": "encoding.py", "load_and_parse.py": "roi.py"}[filename],
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


encoder = module("contract_encoder", "encoding_model.py")
roi = module("contract_roi", "load_and_parse.py")


@pytest.fixture
def aligned(tmp_path):
    rng = np.random.RandomState(12)
    lengths = np.full(6, 15)
    features = rng.normal(size=(90, 3)).astype(np.float32)
    data = {
        "subject": np.array("sub-13"),
        "fc1_aligned": features,
        "voxel_ts": rng.normal(size=(3, 90)).astype(np.float32),
        "n_voxels": 3,
        "n_volumes": 90,
        "n_games": 1,
        "game_names": np.array(["fixture"]),
        "play_boundaries": np.r_[0, np.cumsum(lengths)],
        "play_n_volumes": lengths,
        "play_levels": np.array([0, 1, 3, 4, 6, 7]),
        "play_game_idx": np.zeros(6, dtype=int),
        "tr_game_idx": np.zeros(90, dtype=int),
        "tr_level_idx": np.repeat([0, 1, 3, 4, 6, 7], lengths),
        "tr_play_idx": np.repeat(np.arange(6), lengths),
        "keystates": rng.randint(0, 2, size=(90, 2)).astype(np.float32),
        "time_in_play": np.tile(np.arange(15), 6),
        "time_in_experiment": np.arange(90),
        "mask": np.ones((3, 1, 1), dtype=bool),
        "mask_affine": np.eye(4),
    }
    path = tmp_path / "independent-base.npz"
    np.savez(path, **data)
    return data, path


@pytest.mark.parametrize(
    "defect",
    ["subject", "rows", "boundaries", "play_order", "level_identity", "target_count"],
)
def test_alignment_mismatch_fails_before_fitting(aligned, defect):
    original, _ = aligned
    data = dict(original)
    if defect == "subject":
        data["subject"] = "sub-14"
    elif defect == "rows":
        data["fc1_aligned"] = original["fc1_aligned"][:-1]
    elif defect == "boundaries":
        data["play_boundaries"] = np.array([0, 90])
    elif defect == "play_order":
        data["tr_play_idx"] = original["tr_play_idx"][::-1]
    elif defect == "level_identity":
        data["tr_level_idx"] = original["tr_level_idx"].copy()
        data["tr_level_idx"][0] = 6
    else:
        data["n_voxels"] = 2
    with pytest.raises(ValueError):
        encoder.validate_aligned_data(data, "fc1", "sub-13")


def test_sidecar_cannot_replace_alignment_identity(aligned, tmp_path):
    data, path = aligned
    sidecar = tmp_path / "chosen-features.npz"
    np.savez(sidecar, subject="sub-14", new_aligned=data["fc1_aligned"])
    with pytest.raises(ValueError, match="conflicts.*subject"):
        encoder.load_aligned_data([path, sidecar], "sub-13", "new")
    np.savez(sidecar, subject="sub-13", new_aligned=data["fc1_aligned"])
    loaded = encoder.load_aligned_data([path, sidecar], "sub-13", "new")
    np.testing.assert_array_equal(loaded["new_aligned"], data["fc1_aligned"])


def test_explicit_seed_repeats_actual_ridge_fit_and_records_inputs(aligned, tmp_path):
    _, path = aligned
    outputs = []
    for index in range(2):
        directory = tmp_path / f"run-{index}"
        encoder.run_encoding_model(
            "sub-13",
            None,
            directory,
            base_data=path,
            layer="fc1",
            include_nuisance_bands=True,
            seed=31,
            shuffle=True,
            shuffle_scope="games",
            n_iter=3,
            n_targets_batch=2,
            n_alphas_batch=4,
        )
        with np.load(
            directory / "sub-13/encoding_results_fc1_shuffled_games.npz"
        ) as result:
            outputs.append(result["performances_full"])
            assert result["seed"].item() == 31
            assert result["effective_shuffle_seed"].item() == 31
            assert result["n_alphas_batch"].item() == 4
            assert result["actual_backend"].item() == "numpy"
            recorded = json.loads(result["input_files_json"].item())
            assert recorded == [
                {
                    "role": "base",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ]
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("scope", ["games", "plays", "levels"])
def test_shuffle_seed_reproducible_within_declared_groups(scope):
    features = np.arange(24).reshape(12, 2)
    if scope == "games":
        function, args = encoder.shuffle_within_games, [np.repeat([0, 1], 6)]
    elif scope == "plays":
        function, args = encoder.shuffle_within_plays, [np.array([0, 6, 12])]
    else:
        function, args = (
            encoder.shuffle_within_levels,
            [np.repeat([0, 1], 6), np.zeros(12, dtype=int)],
        )
    left = function(features, *args, seed=7)
    right = function(features, *args, seed=7)
    np.testing.assert_array_equal(left, right)
    for start in (0, 6):
        assert set(map(tuple, left[start : start + 6])) == set(
            map(tuple, features[start : start + 6])
        )


def result_file(root, *, layer=1, subject="sub-13", **overrides):
    directory = root / subject
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"encoding_results_llm_qwen35_9b__all__main_layer_{layer}.npz"
    values = {
        "subject": subject,
        "layer": f"llm_qwen35_9b__all__main_layer_{layer}",
        "band_names": ["main"],
        "mask": np.ones((1, 1, 1), dtype=bool),
        "mask_affine": np.eye(4),
        "performances_main": np.array([[0.1, 0.2]]),
        "partition_ids": [4, 8],
    }
    values.update(overrides)
    np.savez(path, **values)
    return path


@pytest.mark.parametrize(
    "metadata",
    [
        {"subject": "sub-14"},
        {"layer": "llm_dsv32__compressed__main_layer_1"},
        {"shuffle": True},
        {"completion_status": "failed"},
    ],
)
def test_result_identity_mismatch_is_rejected(tmp_path, metadata):
    path = result_file(tmp_path)
    with np.load(path) as source:
        data = dict(source)
    data.update(metadata)
    np.savez(path, **data)
    with pytest.raises(ValueError):
        roi.find_npz_files(tmp_path)


def test_duplicate_fits_need_identical_bytes(tmp_path):
    first = result_file(tmp_path / "first")
    second = tmp_path / "second" / first.parent.name / first.name
    second.parent.mkdir(parents=True)
    shutil.copyfile(first, second)
    assert len(roi.find_npz_files([tmp_path / "first", tmp_path / "second"])) == 1
    result_file(tmp_path / "second", performances_main=np.array([[0.8, 0.9]]))
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        roi.find_npz_files([tmp_path / "first", tmp_path / "second"])


def test_roi_preserves_partition_ids_and_rejects_missing_bands(tmp_path):
    result_file(tmp_path)
    (item,) = roi.find_npz_files(tmp_path)
    roi._worker_init(
        {("test", "left"): np.array([True])}, {"sub-13": np.array([0])}, {"sub-13": 1}
    )
    rows = roi.process_one_file((item, ["main"]))
    assert [record["partition"] for record in rows] == [4, 8]
    assert [record["performance"] for record in rows] == [0.1, 0.2]
    with pytest.raises(ValueError, match="Requested band"):
        roi.process_one_file((item, ["time"]))
    result_file(tmp_path, performances_main=np.array([[0.1, 0.2], [0.3, 0.4]]))
    with pytest.raises(ValueError, match="shape"):
        roi.process_one_file((item, ["main"]))


def test_nifti_layer_option_restricts_selection(tmp_path, monkeypatch):
    results = tmp_path / "results"
    result_file(results, layer=1)
    result_file(results, layer=2)
    atlas = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1), dtype=np.int16), np.eye(4)), atlas)
    (tmp_path / "atlas.txt").write_text("label Calcarine_L 1\n")
    monkeypatch.setattr(
        roi, "build_lateralized_roi_codes", lambda _: {("test", "left"): [1]}
    )
    selected = []
    monkeypatch.setattr(
        roi,
        "write_best_layer_niftis",
        lambda items, *args, **kwargs: selected.extend(
            item["layer_idx"] for item in items
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "load_and_parse",
            "--results-dir",
            str(results),
            "--atlas",
            str(atlas),
            "--output",
            str(tmp_path / "out.csv"),
            "--bands",
            "main",
            "--workers",
            "1",
            "--nifti-dir",
            str(tmp_path / "niftis"),
            "--nifti-layers",
            "1",
        ],
    )
    roi.main()
    assert selected == [1]


def test_per_model_nifti_needs_equal_layer_coverage(tmp_path):
    result_file(tmp_path, subject="sub-13", layer=1)
    result_file(tmp_path, subject="sub-13", layer=2)
    result_file(tmp_path, subject="sub-14", layer=1)
    items = roi.find_npz_files(tmp_path)
    with pytest.raises(ValueError, match="Unequal subject coverage"):
        roi.write_best_layer_niftis(items, tmp_path / "output", "main", "per_model")


def test_library_import_does_not_change_thread_environment(tmp_path):
    script = ROOT / "src/reason_to_play/analysis/neural/encoding.py"
    code = """import importlib.util, os, sys
before = dict(os.environ)
spec = importlib.util.spec_from_file_location('imported_encoder', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
 assert before.get(name) == os.environ.get(name), name
"""
    subprocess.run([sys.executable, "-c", code, str(script)], cwd=tmp_path, check=True)


def test_explicit_input_cli_resume_checks_content_from_other_directory(
    aligned, tmp_path
):
    original, _ = aligned
    base = tmp_path / "bold-with-identities.npz"
    np.savez(
        base, **{key: value for key, value in original.items() if key != "fc1_aligned"}
    )
    features = tmp_path / "features-with-readable-name.npz"
    np.savez(features, llm_fixture_layer_1_aligned=original["fc1_aligned"])
    output = tmp_path / "encoded"
    command = [
        sys.executable,
        str(ROOT / "src/reason_to_play/analysis/neural/encoding.py"),
        "--subject",
        "sub-13",
        "--base-data",
        str(base),
        "--feature-file",
        str(features),
        "--layer",
        "llm_fixture_layer_1",
        "--output-dir",
        str(output),
        "--seed",
        "19",
        "--n-iter",
        "1",
    ]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    resumed = subprocess.run(
        command + ["--resume"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    assert "Verified completed output" in resumed.stderr
    np.savez(features, llm_fixture_layer_1_aligned=original["fc1_aligned"] + 1)
    changed = subprocess.run(
        command + ["--resume"], cwd=tmp_path, capture_output=True, text=True
    )
    assert changed.returncode != 0
    assert "Cannot resume mismatched" in changed.stderr
