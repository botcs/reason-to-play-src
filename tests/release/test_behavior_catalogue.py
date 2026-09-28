"""Catalogue play metadata and provenance using standalone JSON inputs."""

from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.release.build_behavior_catalogue import build_catalogue


def recording(game="bait", *, subject=13, condition="elaborate", plays=None):
    """Compact replay fixture: run, original ordinal, level, win, death."""
    plays = [(1, 4, 0, None, False)] if plays is None else plays
    cohort = "vgfmri3" if subject < 12 else "vgfmri4"
    record = {
        "schema": "reason-to-play/human-replay",
        "schema_version": 1,
        "subject": f"sub-{subject:02d}",
        "game": f"{game}_{cohort}",
        "source": "human",
        "game_description": "BasicGame\n    SpriteSet\n        avatar > MovingAvatar img=colors/DARKBLUE",
        "meta": {"suggestion_level": condition},
        "total_frames": 3 * len(plays),
        "delta_encoded": True,
        "plays": [],
        "states": [],
    }
    for run, ordinal, level, win, death in plays:
        identity = f"{subject * 10000 + run * 100 + ordinal:024x}"
        record["plays"].append(
            {
                "_id": identity,
                "subj_id": subject,
                "run_id": run,
                "play_id": ordinal + 50,
                "game_name": f"{cohort}_{game}",
                "level_id": level,
                "win": win,
                "source_document_index": ordinal,
                "state_start": len(record["states"]),
                "state_count": 3,
                "block_size": 35,
                "outcome": "win"
                if win is True
                else "avatar_died"
                if death
                else "loss"
                if win is False
                else "incomplete",
            }
        )
        for i in range(3):
            record["states"].append(
                {
                    "time": i,
                    "realworld_ts": 1000 + run * 60 + ordinal * 2 + i * 0.05,
                    "source_play_id": identity,
                    "keyPressType": "UP" if i == 1 else None,
                    "effectListByClass": [["killSprite", "avatar", "enemy"]]
                    if death and i == 2
                    else [],
                    "win": -1,
                }
            )
    record["states"][0]["sprites"] = {}
    return record


def write_record(root, record):
    path = (
        root
        / "behavior/human"
        / record["subject"]
        / record["game"]
        / f"{record['meta']['suggestion_level']}.human.replay.json.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(record).encode(), mtime=0))
    return path


@pytest.mark.parametrize("workers", [1, 2])
def test_nullable_death_practice_upper_levels_and_independent_clocks(tmp_path, workers):
    write_record(
        tmp_path, recording("sokoban", subject=1, plays=[(0, 2, 0, None, True)])
    )
    path = write_record(
        tmp_path,
        recording(subject=1, plays=[(1, 4, 10, None, False), (1, 9, 0, True, False)]),
    )
    report = build_catalogue(tmp_path, tmp_path / "catalogue", workers=workers)
    table = pq.read_table(tmp_path / "catalogue/human_plays.parquet")
    death, incomplete, win = table.to_pylist()
    assert table.schema.field("original_win") == pa.field(
        "original_win", pa.bool_(), nullable=True
    )
    assert (
        death["original_win"],
        death["outcome"],
        death["is_practice"],
        death["in_common_comparison"],
    ) == (None, "avatar_died", True, False)
    assert (
        incomplete["original_win"],
        incomplete["outcome"],
        incomplete["level"],
        incomplete["is_common_level"],
    ) == (None, "incomplete", 10, False)
    assert (win["outcome"], win["in_common_comparison"]) == ("win", True)
    assert all(
        row["engine_frames"] == 2 and row["keypress_frames"] == 1
        for row in (death, incomplete, win)
    )
    assert [row["source_document_index"] for row in (death, incomplete, win)] == [
        2,
        4,
        9,
    ]
    assert [row["play_id"] for row in (death, incomplete, win)] == [52, 54, 59]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert win["source_payload_sha256"] == digest
    assert win["source_artifact_id"] == "sha256:" + digest
    assert win["source_release_path"] == path.relative_to(tmp_path).as_posix()
    assert win["cohort"] == "vgfmri3"
    assert not win["is_primary_encoding_cohort"]
    assert report["row_count"] == report["unique_original_ids"] == 3
    assert report["practice_count"] == report["outside_common_levels_count"] == 1


def test_selects_one_condition_and_keeps_original_run_order(tmp_path, monkeypatch):
    import human.behavior as reader

    def no_sprite_expansion(*args):
        raise AssertionError("Catalogue must not expand sprite trajectories")

    monkeypatch.setattr(reader, "expand_delta_states", no_sprite_expansion)
    for condition in ("elaborate", "minimal", "oracle"):
        write_record(tmp_path, recording(condition=condition))
        write_record(
            tmp_path,
            recording("helper", condition=condition, plays=[(1, 1, 0, False, False)]),
        )
    for condition in ("elaborate", "minimal"):
        output = tmp_path / f"catalogue-{condition}"
        kwargs = {} if condition == "elaborate" else {"condition": condition}
        report = build_catalogue(tmp_path / "behavior/human", output, **kwargs)
        rows = pq.read_table(output / "human_plays.parquet").to_pylist()
        assert report["row_count"] == report["source_file_count"] == 2
        assert [row["source_document_index"] for row in rows] == [1, 4]
        assert [row["outcome"] for row in rows] == ["loss", "incomplete"]
        assert all(row["is_primary_encoding_cohort"] for row in rows)
        assert {row["prompt_condition"] for row in rows} == {condition}
        assert report["prompt_condition"] == condition


@pytest.mark.parametrize(
    "damage", ["bounds", "clock", "timestamp", "id", "outcome", "game"]
)
def test_rejects_corrupted_identity_or_measured_values(tmp_path, damage):
    record = recording()
    if damage == "bounds":
        record["plays"][0]["state_start"] = 1
    elif damage == "clock":
        record["states"][1]["time"] = 7
    elif damage == "timestamp":
        record["states"][1]["realworld_ts"] = None
    elif damage == "id":
        record["states"][1]["source_play_id"] = "a-different-play"
    elif damage == "outcome":
        record["plays"][0]["outcome"] = "win"
    elif damage == "game":
        record["plays"][0]["game_name"] = "vgfmri4_helper"
    path = write_record(tmp_path, record)
    with pytest.raises(ValueError):
        build_catalogue(path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("duplicate", ["id", "ordinal"])
def test_rejects_duplicate_observations_across_files(tmp_path, duplicate):
    record = recording()
    write_record(tmp_path, record)
    other = deepcopy(record)
    other["game"] = "helper_vgfmri4"
    other["plays"][0]["game_name"] = "vgfmri4_helper"
    if duplicate == "ordinal":
        other["plays"][0]["_id"] = "different-identity-same-run-ordinal"
        for state in other["states"]:
            state["source_play_id"] = other["plays"][0]["_id"]
    write_record(tmp_path, other)
    with pytest.raises(ValueError, match="Duplicate original"):
        build_catalogue(tmp_path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_standalone_cli_uses_own_condition_and_runs_outside_checkout(tmp_path):
    path = write_record(tmp_path, recording(condition="oracle"))
    standalone = tmp_path / "downloaded.json.gz"
    path.rename(standalone)
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "catalogue"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/release/build_behavior_catalogue.py"),
            "--input",
            str(standalone),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(root)),
        check=True,
        capture_output=True,
    )
    report = json.loads((output / "metadata.json").read_text())
    assert report["prompt_condition"] == "oracle"
    assert report["source_file_count"] == 1
    (row,) = pq.read_table(output / "human_plays.parquet").to_pylist()
    assert (
        row["source_release_path"]
        == "behavior/human/sub-13/bait_vgfmri4/oracle.human.replay.json.gz"
    )
    assert (
        row["source_payload_sha256"]
        == hashlib.sha256(standalone.read_bytes()).hexdigest()
    )
    original = (output / "human_plays.parquet").read_bytes()
    with pytest.raises(FileExistsError):
        build_catalogue(standalone, output)
    assert (output / "human_plays.parquet").read_bytes() == original


def test_rejects_relocated_directory_identity_and_missing_condition(tmp_path):
    path = write_record(tmp_path, recording())
    path.rename(path.with_name("minimal.human.replay.json.gz"))
    with pytest.raises(FileNotFoundError):
        build_catalogue(tmp_path, tmp_path / "missing")
    with pytest.raises(ValueError, match="Replay identity differs"):
        build_catalogue(tmp_path, tmp_path / "wrong", condition="minimal")
