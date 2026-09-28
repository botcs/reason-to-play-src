"""Bounded synthetic integration; no MRI downloads, API calls, or GPU required.

These tests exercise the real NIfTI/JSON/NPZ/PT boundaries and ridge solver.
They establish pipeline wiring, not reproduction of the paper's effect sizes.
"""

import importlib.util
import subprocess
import sys
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")
pytest.importorskip("himalaya")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preprocess = load_script("neural_preprocess", "src/reason_to_play/fmri/preprocess.py")
base = load_script("neural_align_base", "src/reason_to_play/fmri/align_baselines.py")
align = load_script("neural_align", "src/reason_to_play/fmri/align_llm.py")
encoder = load_script(
    "neural_encoder", "src/reason_to_play/analysis/neural/encoding.py"
)
extractor = load_script("neural_ddqn", "src/reason_to_play/features/ddqn.py")


def write_human_replay(path, plays, scanner):
    """Serialize explicit, small measured fixtures in the released JSON schema."""
    from reason_to_play.data.behavior import human_outcome

    states, records = [], []
    for ordinal, play in enumerate(plays):
        frames = play["states"]
        record = {
            key: value
            for key, value in play.items()
            if key not in {"states", "game_str", "level_str"}
        }
        record.update(
            state_start=len(states),
            state_count=len(frames),
            source_document_index=ordinal,
            block_size=1,
            grid_size=[1, 1],
            scanner=scanner,
            outcome=human_outcome(play, frames),
        )
        records.append(record)
        for frame in frames:
            stored = {
                key: value
                for key, value in frame.items()
                if key not in {"objects", "gt", "ts"}
            }
            stored.update(
                time=frame["gt"],
                realworld_ts=frame["ts"],
                source_play_id=play["_id"],
                sprites={},
            )
            states.append(stored)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump(
            {
                "schema": "reason-to-play/human-replay",
                "schema_version": 1,
                "source": "human",
                "subject": "sub-13",
                "game": "bait_vgfmri4",
                "meta": {"suggestion_level": "elaborate"},
                "game_description": plays[0]["game_str"],
                "total_frames": len(states),
                "plays": records,
                "states": states,
                "steps": [],
            },
            stream,
        )
    return path


@pytest.fixture(scope="module")
def synthetic_alignment(tmp_path_factory):
    root = tmp_path_factory.mktemp("neural-pipeline")
    subject = "sub-13"
    game = "vgfmri4_bait"
    rng = np.random.default_rng(23)
    func = root / "fmriprep" / subject / "func"
    func.mkdir(parents=True)
    stem = f"{subject}_task-gameplay_run-01"
    image = nib.Nifti1Image(
        rng.normal(size=(3, 3, 3, 120)).astype(np.float32),
        np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    image.header.set_zooms((2, 2, 2, 2))
    nib.save(
        image, func / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    )
    mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), image.affine)
    nib.save(
        mask, func / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
    )
    import pandas as pd

    pd.DataFrame(
        {
            name: rng.normal(size=120)
            for name in (
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "csf",
                "white_matter",
            )
        }
    ).to_csv(func / f"{stem}_desc-confounds_timeseries.tsv", sep="\t", index=False)
    result = preprocess.preprocess_voxelwise(
        subject, 1, root / "fmriprep", root / "preprocessed"
    )
    with np.load(result) as values:
        assert values["voxel_ts"].shape == (27, 119)
        assert bool(values["preproc_ar1_corrected"])
        assert bool(values["preproc_confounds_regressed"])
        np.testing.assert_allclose(
            values["voxel_ts_zscored"].mean(axis=1), 0, atol=1e-6
        )

    behavioral = root / "behavioral"
    behavioral.mkdir()
    scan_start = 1_600_000_000.0
    scanner = {"subj_id": 13, "run_id": 1, "scan_start_ts": scan_start}
    plays = []
    md, tensors = [], []
    # Includes level 9 so --max-level can be checked on LLM and BOLD together.
    for trial, level in enumerate((0, 3, 6, 9)):
        play_id = f"{trial + 1:024x}"
        timestamps = scan_start + 2 * np.arange(1 + trial * 25, 21 + trial * 25)
        states = [
            {
                "ts": float(t),
                "gt": i,
                "score": float(i),
                "objects": {},
                "keystate": [],
                "effectListByClass": [],
            }
            for i, t in enumerate(timestamps)
        ]
        plays.append(
            {
                "_id": play_id,
                "subj_id": 13,
                "run_id": 1,
                "game_name": game,
                "level_id": level,
                "play_id": trial,
                "game_str": "BasicGame\n    SpriteSet\n        avatar > MovingAvatar",
                "level_str": "A",
                "win": None,
                "states": states,
            }
        )
        features = torch.from_numpy(rng.normal(size=(20, 2)).astype(np.float32))
        extractor.save_model_features_for_level(
            [
                {
                    "activations": {"fc1": features},
                    "behavioral": {
                        "win": None,
                        "score": 0,
                        "timestamps": timestamps,
                        "num_states": 20,
                    },
                    "metadata": {
                        "play_id": play_id,
                        "subj_id": 13,
                        "run_id": 1,
                        "game_name": game,
                        "level_id": level,
                        "model_type": "synthetic",
                    },
                }
            ],
            root / "ddqn",
            "fixture",
            subject,
            game,
            level,
        )
        for timestamp in timestamps:
            md.append(
                {
                    "play_id": play_id,
                    "level_id": level,
                    "trial_idx": trial,
                    "realworld_ts": float(timestamp),
                }
            )
        tensors.append(features[:, None, :])
    recordings = write_human_replay(
        behavioral / "sub-13/bait_vgfmri4/elaborate.human.replay.json.gz",
        plays,
        scanner,
    )
    base_path = base.process_subject(
        subject,
        root / "preprocessed",
        root / "ddqn/model-fixture",
        output_dir=root / "aligned",
        behavior_dir=recordings,
    )
    with np.load(base_path, allow_pickle=True) as data:
        assert data["voxel_ts"].shape == (27, 80)
        np.testing.assert_array_equal(data["play_boundaries"], [0, 20, 40, 60, 80])
        np.testing.assert_array_equal(data["play_levels"], [0, 3, 6, 9])
    llm = root / "llm" / subject
    llm.mkdir(parents=True)
    torch.save(
        {"session": {"num_layers": 1}, "metadata": md, "features": torch.cat(tensors)},
        llm / "bait_vgfmri4.pt",
    )
    source = align.parse_llm_source_arg(f"name=fixture,dir={root / 'llm'},stream=main")
    paths = align.process_subject(
        subject,
        base_path,
        output_dir=root / "aligned",
        llm_sources=[source],
        behavior_dir=recordings,
    )
    assert len(paths) == 1
    with np.load(paths[0]) as features:
        assert features["llm_fixture_layer_1_aligned"].shape == (80, 2)
        assert int(features["llm_fixture_matched"]) == 4
    return root, subject


def test_json_records_drive_ddqn_and_theory_alignment(synthetic_alignment):
    from reason_to_play.data.behavior import iter_plays

    root, subject = synthetic_alignment
    recordings = root / "behavioral"
    plays, scanner_run = extractor.load_behavioral_data(subject, 1, recordings)
    assert len(plays) == 4
    assert scanner_run["scan_start_ts"] == 1600000000.0
    assert len(extractor.play_states(plays[0])) == 20
    regressors = []
    for play in iter_plays(recordings):
        regressors.append(
            {
                "play_key": play["_id"],
                "regressors": {
                    "theory_str": [
                        [play["game_str"], None, frame["ts"]]
                        for frame in play["states"]
                    ]
                },
            }
        )
    regressor_path = root / "empa.json.gz"
    with gzip.open(regressor_path, "wt") as stream:
        json.dump(
            {
                "schema": "reason-to-play/empa-regressors",
                "schema_version": 1,
                "regressors": regressors,
            },
            stream,
        )
    output = root / "hrr-aligned"
    base.process_subject(
        subject,
        root / "preprocessed",
        root / "ddqn/model-fixture",
        output_dir=output,
        behavior_dir=recordings,
        regressors_json_path=regressor_path,
    )
    with np.load(
        output / subject / "bold-ddqn-theory.npz", allow_pickle=True
    ) as result:
        np.testing.assert_array_equal(result["play_boundaries"], [0, 20, 40, 60, 80])
        np.testing.assert_array_equal(result["play_levels"], [0, 3, 6, 9])
        assert result["voxel_ts"].shape == (27, 80)
    with np.load(output / subject / "aligned_hrr_decomposed.npz") as result:
        assert any("sprite" in key for key in result.files)
        for key in result.files:
            if key.endswith("_aligned"):
                assert result[key].shape[0] == 80
                assert np.isfinite(result[key]).all()


def test_frame_timing_and_simultaneous_keypress_nuisance():
    timestamps = [1600000000.0, 1600000000.75, 1600000002.25, 1600000008.5]
    states = []
    for index, timestamp in enumerate(timestamps):
        keys = [0] * 300
        if index == 1:
            keys[273] = keys[275] = 1  # Preserve simultaneous up + right.
        states.append(
            {
                "ts": timestamp,
                "gt": index,
                "keystate": keys,
                "score": float(index),
                "objects": {},
            }
        )
    play = {"_id": "original", "states": states}
    run = {"scan_start_ts": timestamps[0]}
    for module in (base, align):
        timing = module.get_play_timing(play, run, 2.0, 3, ar1_corrected=False)
        np.testing.assert_array_equal(timing["valid_mask"], [True, True, True, False])
        np.testing.assert_array_equal(timing["volume_indices"], [0, 0, 1, 4])
    nuisance = base.extract_behavioral_features(play)
    np.testing.assert_array_equal(nuisance["keystates"][1], [1, 0, 0, 1, 0])
    np.testing.assert_array_equal(nuisance["scores"], [0, 1, 2, 3])
    np.testing.assert_array_equal(nuisance["score_deltas"], [0, 1, 1, 1])


@pytest.mark.parametrize(
    "layer,sidecar,nuisance",
    [
        ("llm_fixture_layer_1", None, False),
        ("llm_fixture_layer_1", None, True),
        ("ez_representation", "aligned_ez.npz", False),
        ("hrr_sprites", "aligned_hrr_decomposed.npz", False),
    ],
)
def test_preprocess_base_llm_and_ridge(synthetic_alignment, layer, sidecar, nuisance):
    root, subject = synthetic_alignment
    if sidecar:
        with np.load(root / "aligned" / subject / "aligned_llm_fixture.npz") as llm:
            np.savez_compressed(
                root / "aligned" / subject / sidecar,
                **{f"{layer}_aligned": llm["llm_fixture_layer_1_aligned"]},
                **{
                    key: llm[key]
                    for key in (
                        "alignment_binding_version",
                        "alignment_base_sha256",
                        "alignment_samples_sha256",
                    )
                },
            )
    encoder.run_encoding_model(
        subject,
        root / "aligned",
        root / "results",
        layer=layer,
        max_level=8,
        n_iter=1,
        n_targets_batch=32,
        include_nuisance_bands=nuisance,
    )
    suffix = "_with_nuisance" if nuisance else ""
    result = root / "results" / subject / f"encoding_results_{layer}{suffix}.npz"
    with np.load(result) as data:
        assert data["performances_full"].shape == (27, 3)
        assert np.isfinite(data["performances_full"]).all()
        assert data["n_volumes"] == 60
        assert data["completion_status"] == "complete"
        np.testing.assert_array_equal(data["partition_ids"], [0, 1, 2])
        expected_bands = (
            ["main", "button", "time", "identity"] if nuisance else ["main"]
        )
        np.testing.assert_array_equal(data["band_names"], expected_bands)


def test_fold_failure_cannot_be_saved_as_zero_scores(synthetic_alignment, monkeypatch):
    root, subject = synthetic_alignment

    def fail_fit(*args, **kwargs):
        raise ValueError("injected solver failure")

    monkeypatch.setattr(encoder.GroupRidgeCV, "fit", fail_fit)
    with pytest.raises(RuntimeError, match="no result file was written"):
        encoder.run_encoding_model(
            subject, root / "aligned", root / "failed", layer="fc1", n_iter=1
        )
    assert not list((root / "failed").glob("**/*.npz"))


def test_failed_layer_cli_returns_nonzero(synthetic_alignment):
    root, subject = synthetic_alignment
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/reason_to_play/analysis/neural/encoding.py"),
            "--subject",
            subject,
            "--data-dir",
            str(root / "aligned"),
            "--output-dir",
            str(root / "failed-cli"),
            "--layer",
            "absent",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Encoding failed" in result.stderr
    assert not list((root / "failed-cli").glob("**/*.npz"))


def test_lags_never_cross_play_boundaries():
    values = np.array([1, 2, 3, 101, 102, 103], dtype=np.float32)[:, None]
    actual = encoder.create_lagged_features(values, np.array([0, 3, 6]), lags=[1, 2])
    np.testing.assert_array_equal(
        actual, [[1, 1], [1, 1], [2, 1], [101, 101], [101, 101], [102, 101]]
    )


def test_checkpoint_map_works_without_private_service(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic")
    assert (
        extractor.download_checkpoint(
            "custom_game",
            4,
            tmp_path,
            checkpoint_map={"custom_game": {"4": str(checkpoint)}},
        )
        == checkpoint
    )
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        extractor.download_checkpoint("bait", 0, tmp_path)


def test_baseline_roi_aggregation_requires_explicit_layer_identity(synthetic_alignment):
    root, subject = synthetic_alignment
    roi = load_script("neural_roi", "src/reason_to_play/analysis/neural/roi.py")
    encoder.run_encoding_model(
        subject,
        root / "aligned",
        root / "baseline-results",
        layer="fc1",
        max_level=8,
        n_iter=1,
        n_targets_batch=32,
    )
    assert roi.find_npz_files(root / "baseline-results") == []
    mapping = roi.load_baseline_map(
        ROOT / "experiments/neurips2026/baseline-layer-map.example.json"
    )
    items = roi.find_npz_files(root / "baseline-results", baseline_map=mapping)
    assert len(items) == 1
    assert items[0]["model"] == "ddqn_fc1_rerun"
    masks = roi.collect_subject_masks(items)
    common_mask, affine, indexing = roi.build_common_mask_and_indexing(masks)
    atlas = root / "synthetic-atlas.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.int16), affine), atlas)
    selectors = roi.precompute_roi_voxel_masks(
        common_mask, affine, atlas, {("SyntheticROI", "left"): [1]}
    )
    roi._worker_init(selectors, indexing, {subject: 27})
    rows = roi.process_one_file((items[0], ["main", "full"]))
    assert len(rows) == 6
    assert all(row["n_voxels"] == 27 for row in rows)
    assert all(row["layer_idx"] == 0 and row["layer_depth"] == 0 for row in rows)
    assert all(row["fit_condition"] == "main-only" for row in rows)
    assert all(np.isfinite(row["performance"]) for row in rows)


def test_roi_requires_explicit_fit_condition_and_reads_shuffle_metadata(tmp_path):
    roi = load_script(
        "neural_roi_conditions", "src/reason_to_play/analysis/neural/roi.py"
    )
    subject = tmp_path / "sub-13"
    subject.mkdir()
    stem = "encoding_results_llm_qwen35_9b__all__main_layer_1"
    for suffix, nuisance in (("", False), ("_with_nuisance", True)):
        np.savez(
            subject / f"{stem}{suffix}.npz",
            include_nuisance_bands=nuisance,
            band_names=["main", "button", "time", "identity"] if nuisance else ["main"],
            mask=np.ones((1, 1, 1), dtype=bool),
            performances_main=np.array([[0.2 if nuisance else 0.1]]),
        )
    with pytest.raises(ValueError, match="Mixed main-only and nuisance fits"):
        roi.find_npz_files(tmp_path)
    for condition, suffix in (("main-only", ""), ("with-nuisance", "_with_nuisance")):
        items = roi.find_npz_files(tmp_path, fit_condition=condition)
        assert len(items) == 1
        assert items[0]["fit_condition"] == condition
        assert items[0]["filepath"].name == f"{stem}{suffix}.npz"
        assert items[0]["model"] == "qwen35_9b"
        roi._worker_init(
            {("SyntheticROI", "left"): np.array([True])},
            {"sub-13": np.array([0])},
            {"sub-13": 1},
        )
        rows = roi.process_one_file((items[0], ["main"]))
        assert rows[0]["fit_condition"] == condition
        assert rows[0]["performance"] == (0.1 if condition == "main-only" else 0.2)

    # The encoder's shuffle filename omits the nuisance flag; metadata wins.
    np.savez(
        subject / f"{stem}_shuffled_games.npz",
        band_names=["main", "button", "time", "identity"],
    )
    items = roi.find_npz_files(tmp_path, fit_condition="with-nuisance")
    assert {item["model"] for item in items} == {"qwen35_9b", "qwen35_9b_shuf_games"}
    assert all(item["fit_condition"] == "with-nuisance" for item in items)


@pytest.mark.parametrize("ar1_corrected", [True, False])
def test_llm_alignment_preserves_scan_truncated_play(tmp_path, ar1_corrected):
    subject, game = "sub-13", "vgfmri4_bait"
    play_id = "000000000000000000000099"
    scan_start = 1_600_000_000.0
    timestamps = scan_start + 2 * np.arange(1, 21)
    play = {
        "_id": play_id,
        "subj_id": 13,
        "run_id": 1,
        "game_name": game,
        "level_id": 0,
        "play_id": 0,
        "win": None,
        "game_str": "BasicGame\n    SpriteSet\n        avatar > MovingAvatar",
        "states": [
            {
                "ts": float(t),
                "gt": i,
                "objects": {},
                "keystate": [],
                "effectListByClass": [],
            }
            for i, t in enumerate(timestamps)
        ],
    }
    run = {"subj_id": 13, "run_id": 1, "scan_start_ts": scan_start}
    # Base alignment clips the final play at the actual preprocessed scan end.
    timing = base.get_play_timing(play, run, 2.0, 14, ar1_corrected)
    retained = int(timing["n_volumes"])
    np.savez(
        tmp_path / "base.npz",
        tr=2.0,
        ar1_corrected=ar1_corrected,
        n_volumes=retained,
        game_names=[game],
        play_ids=[str(play_id)],
        play_game_idx=[0],
        play_levels=[0],
        play_n_volumes=[retained],
        subject=subject,
        play_boundaries=[0, retained],
        tr_run_idx=np.ones(retained, dtype=int),
        tr_game_idx=np.zeros(retained, dtype=int),
        tr_level_idx=np.zeros(retained, dtype=int),
        tr_play_idx=np.zeros(retained, dtype=int),
    )
    recordings = write_human_replay(
        tmp_path / "recording.human.replay.json.gz", [play], run
    )
    features = torch.arange(1, 41, dtype=torch.float32).reshape(20, 1, 2)
    llm = tmp_path / "llm" / subject
    llm.mkdir(parents=True)
    torch.save(
        {
            "session": {"num_layers": 1},
            "metadata": [
                {
                    "play_id": str(play_id),
                    "level_id": 0,
                    "trial_idx": 0,
                    "realworld_ts": float(t),
                }
                for t in timestamps
            ],
            "features": features,
        },
        llm / "bait_vgfmri4.pt",
    )
    source = align.parse_llm_source_arg(
        f"name=fixture,dir={tmp_path / 'llm'},stream=main"
    )
    paths = align.process_subject(
        subject,
        tmp_path / "base.npz",
        output_dir=tmp_path / "aligned",
        llm_sources=[source],
        behavior_dir=recordings,
    )
    with np.load(paths[0]) as result:
        np.testing.assert_array_equal(
            result["llm_fixture_layer_1_aligned"], features[:retained, 0, :].numpy()
        )


def test_ar1_first_volume_offset_is_respected():
    states = [{"ts": 1_600_000_000.0 + 2 * i} for i in range(4)]
    play = {"_id": "fixture", "states": states}
    timing = base.get_play_timing(
        play, {"scan_start_ts": 1_600_000_000.0}, 2.0, 3, True
    )
    np.testing.assert_array_equal(timing["valid_mask"], [False, True, True, True])
    np.testing.assert_array_equal(timing["volume_indices"], [-1, 0, 1, 2])
    assert timing["n_volumes"] == 3


def test_missing_confounds_fails_before_an_alternate_pipeline_is_run(
    synthetic_alignment,
):
    root, subject = synthetic_alignment
    source = (
        root
        / "fmriprep"
        / subject
        / "func"
        / f"{subject}_task-gameplay_run-01_desc-confounds_timeseries.tsv"
    )
    saved = source.read_bytes()
    source.unlink()
    try:
        with pytest.raises(FileNotFoundError, match="Confounds are required"):
            preprocess.preprocess_voxelwise(
                subject, 1, root / "fmriprep", root / "missing-confounds"
            )
    finally:
        source.write_bytes(saved)


def test_vendored_baseline_provenance_rejects_changed_or_added_source(tmp_path):
    import hashlib
    import json

    source = tmp_path / "rl_models.py"
    original = b"# minimal source fixture\n"
    source.write_bytes(original)
    record = {
        "commit": extractor.RC_RL_EXTRACTION_REVISION,
        "files": [
            {"path": source.name, "sha256": hashlib.sha256(original).hexdigest()}
        ],
    }
    (tmp_path / "PROVENANCE.json").write_text(json.dumps(record))
    provenance = extractor.baseline_source_provenance(tmp_path)
    assert provenance["distribution"] == "curated-vendor"
    assert len(provenance["manifest_sha256"]) == 64
    source.write_bytes(b"# changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        extractor.baseline_source_provenance(tmp_path)
    source.write_bytes(original)
    (tmp_path / "unrecorded.py").write_text("# another module\n")
    with pytest.raises(ValueError, match="Unrecorded Python"):
        extractor.baseline_source_provenance(tmp_path)
