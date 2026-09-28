"""Invalid embedded scanner clocks must never silently discard usable samples."""

from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from reason_to_play.data.behavior import load_runs
from reason_to_play.fmri import align_baselines, align_llm


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))


@pytest.mark.parametrize("defect", ["missing", "nonfinite", "identity"])
def test_invalid_embedded_clock_rejects_partial_alignment(tmp_path, defect):
    human = tmp_path / "behavior/human"
    recording_path = human / "sub-13/bait_vgfmri4/elaborate.human.replay.json.gz"
    recording = {
        "schema": "reason-to-play/human-replay",
        "schema_version": 1,
        "subject": "sub-13",
        "game": "bait_vgfmri4",
        "source": "human",
        "meta": {"suggestion_level": "elaborate"},
        "game_description": "BasicGame\n    SpriteSet\n        avatar > MovingAvatar",
        "total_frames": 6,
        "steps": [],
        "plays": [],
        "states": [],
    }
    features = {"num_plays": 2}
    clock = 1_600_000_000.0
    for run in (1, 2):
        identity = f"original-play-{run}"
        times = clock + np.arange(3) * 2
        recording["plays"].append(
            {
                "_id": identity,
                "subj_id": 13,
                "run_id": run,
                "play_id": 0,
                "game_name": "vgfmri4_bait",
                "level_id": 0,
                "win": True,
                "outcome": "win",
                "source_document_index": 0,
                "state_start": len(recording["states"]),
                "state_count": 3,
                "block_size": 35,
                "grid_size": [1, 1],
                "scanner": {"subj_id": 13, "run_id": run, "scan_start_ts": clock},
            }
        )
        recording["states"].extend(
            {
                "time": index,
                "realworld_ts": timestamp,
                "source_play_id": identity,
                "score": index,
                "keystate": [],
                "keyPressType": None,
                "effectListByClass": [],
                "sprites": {},
            }
            for index, timestamp in enumerate(times)
        )
        processed = tmp_path / "processed/sub-13"
        processed.mkdir(parents=True, exist_ok=True)
        np.savez(
            processed / f"run-{run:02d}_voxelwise.npz",
            voxel_ts=np.arange(8, dtype=np.float32)[None, :],
            voxel_ts_zscored=np.arange(8, dtype=np.float32)[None, :],
            mask=np.ones((1, 1, 1), dtype=bool),
            mask_affine=np.eye(4),
            n_voxels=1,
            n_volumes=8,
            n_volumes_original=8,
            tr=2.0,
            preproc_ar1_corrected=False,
            preproc_ar1_rho=0.0,
        )
        prefix = f"play_{run - 1}_"
        features.update(
            {
                prefix + key: value
                for key, value in {
                    "metadata_play_id": identity,
                    "metadata_subj_id": 13,
                    "metadata_run_id": run,
                    "metadata_game_name": "vgfmri4_bait",
                    "metadata_level_id": 0,
                    "behavioral_num_states": 3,
                    "behavioral_timestamps": times,
                    "model_fc1": np.ones((3, 2), dtype=np.float32),
                }.items()
            }
        )
    model = tmp_path / "model/sub-13/vgfmri4_bait"
    model.mkdir(parents=True)
    np.savez(model / "level_00.npz", **features)
    damaged = deepcopy(recording)
    scanner = damaged["plays"][1]["scanner"]
    if defect == "missing":
        del damaged["plays"][1]["scanner"]
    elif defect == "nonfinite":
        scanner["scan_start_ts"] = float("nan")
    else:
        scanner["run_id"] = 1
    write_json(recording_path, damaged)
    arguments = {
        "subject": "sub-13",
        "preprocessed_dir": tmp_path / "processed",
        "model_features_dir": tmp_path / "model",
        "behavior_dir": human,
        "output_dir": tmp_path / "aligned",
    }
    with pytest.raises(ValueError, match="[Ss]canner"):
        align_baselines.process_subject(**arguments)
    assert not list((tmp_path / "aligned").rglob("*.npz"))

    # Correcting the inline measurement retains both runs and all six rows.
    write_json(recording_path, recording)
    result = align_baselines.process_subject(**arguments)
    with np.load(result) as base:
        np.testing.assert_array_equal(
            base["play_ids"], ["original-play-1", "original-play-2"]
        )
        assert int(base["n_volumes"]) == 6

    features = {}
    for index in range(2):
        features.update(
            {
                f"play_{index}_model_layer_1": np.full(
                    (3, 2), index + 1, dtype=np.float32
                ),
                f"play_{index}_behavioral_timestamps": np.arange(3),
                f"play_{index}_behavioral_num_states": 3,
            }
        )
    feature_path = tmp_path / "llm/sub-13/vgfmri4_bait"
    feature_path.mkdir(parents=True)
    np.savez(feature_path / "level_00.npz", **features)
    source = align_llm.parse_llm_source_arg(
        f"name=fixture,dir={tmp_path / 'llm'},layers=1"
    )
    llm_arguments = dict(
        subject="sub-13",
        aligned_data_path=result,
        behavior_dir=human,
        output_dir=tmp_path / "llm-aligned",
        llm_sources=[source],
    )
    write_json(recording_path, damaged)
    with pytest.raises(ValueError, match="[Ss]canner"):
        align_llm.process_subject(**llm_arguments)
    assert not list((tmp_path / "llm-aligned").rglob("*.npz"))
    write_json(recording_path, recording)
    outputs = align_llm.process_subject(**llm_arguments)
    with np.load(outputs[0]) as aligned:
        np.testing.assert_array_equal(
            aligned["llm_fixture_layer_1_aligned"][:, 0], [1, 1, 1, 2, 2, 2]
        )


def test_conflicting_inline_clocks_across_game_files_are_rejected(tmp_path):
    for game, timestamp in [("bait", 1000.0), ("helper", 1001.0)]:
        write_json(
            tmp_path / f"sub-13/{game}_vgfmri4/elaborate.human.replay.json.gz",
            {
                "schema": "reason-to-play/human-replay",
                "schema_version": 1,
                "source": "human",
                "subject": "sub-13",
                "game": f"{game}_vgfmri4",
                "meta": {"suggestion_level": "elaborate"},
                "game_description": "BasicGame\n    SpriteSet\n        avatar > MovingAvatar",
                "plays": [
                    {
                        "subj_id": 13,
                        "run_id": 1,
                        "scanner": {
                            "subj_id": 13,
                            "run_id": 1,
                            "scan_start_ts": timestamp,
                        },
                    }
                ],
            },
        )
    with pytest.raises(ValueError, match="Conflicting embedded scanner"):
        load_runs(tmp_path)
