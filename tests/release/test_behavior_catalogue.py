"""Guard nullable outcomes, independent clocks, source linkage and all-play scope."""

import hashlib
import json
import zlib

import bson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.release.build_behavior_catalogue import build_catalogue


def play(*, run=1, level=0, win=None, death=False, original_id=None):
    return {
        "_id": original_id or bson.ObjectId(),
        "subj_id": 1,
        "run_id": run,
        "play_id": 0,
        "level_id": level,
        "win": win,
        "game_name": "vgfmri3_sokoban" if run == 0 else "vgfmri3_bait",
        "game_str": "SpriteSet\n    avatar > MovingAvatar\n",
        "zstates": zlib.compress(
            bson.encode(
                {
                    "states": [
                        {"gt": 0, "keyPressType": None, "effectListByClass": []},
                        {"gt": 1, "keyPressType": "UP", "effectListByClass": []},
                        {
                            "gt": 2,
                            "keyPressType": None,
                            "effectListByClass": [["killSprite", "avatar", "enemy"]]
                            if death
                            else [],
                            "win": -1,
                        },
                    ]
                }
            )
        ),
    }


def staged(tmp_path, groups):
    sources = []
    for run, documents in groups.items():
        release_path = f"behavior/human/plays/sub-01/run-{run:02d}.bson"
        path = tmp_path / release_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(bson.encode(doc) for doc in documents)
        path.write_bytes(payload)
        sources.append(
            {
                "release_path": release_path,
                "artifact_id": "sha256:"
                + hashlib.sha256(release_path.encode()).hexdigest(),
                "source": {"verification": "head-verified-unversioned"},
                "payload": {
                    "validation": "bytes-verified",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in sources))
    return manifest, sources


@pytest.mark.parametrize("workers", [1, 2])
def test_nullable_death_practice_upper_levels_and_independent_clocks(tmp_path, workers):
    manifest, sources = staged(
        tmp_path,
        {
            0: [play(run=0, death=True)],
            1: [play(level=10), play(win=True)],
        },
    )
    report = build_catalogue(tmp_path, manifest, tmp_path / "catalog", workers=workers)
    table = pq.read_table(tmp_path / "catalog/human_plays.parquet")
    death, incomplete, win = table.to_pylist()
    assert table.schema.field("original_win") == pa.field(
        "original_win", pa.bool_(), nullable=True
    )
    assert death["original_win"] is None
    assert death["outcome"] == "avatar_died"
    assert death["is_practice"] is True
    assert death["in_common_comparison"] is False
    assert incomplete["original_win"] is None
    assert incomplete["outcome"] == "incomplete"
    assert incomplete["level"] == 10
    assert incomplete["is_common_level"] is False
    assert win["outcome"] == "win"
    assert win["in_common_comparison"] is True
    assert all(
        row["engine_frames"] == 2 and row["keypress_frames"] == 1
        for row in (death, incomplete, win)
    )
    assert death["source_artifact_id"] == sources[0]["artifact_id"]
    assert death["source_release_path"] == sources[0]["release_path"]
    assert report["row_count"] == report["unique_original_ids"] == 3
    assert report["practice_count"] == report["outside_common_levels_count"] == 1


def test_duplicate_original_id_is_rejected(tmp_path):
    doc = play()
    manifest, _ = staged(tmp_path, {1: [doc, doc]})
    with pytest.raises(ValueError, match="Duplicate original play ID"):
        build_catalogue(tmp_path, manifest, tmp_path / "catalog")
    assert not (tmp_path / "catalog/human_plays.parquet").exists()


def test_source_payload_must_match_verified_manifest(tmp_path):
    manifest, sources = staged(tmp_path, {1: [play()]})
    with (tmp_path / sources[0]["release_path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Staged bytes differ"):
        build_catalogue(tmp_path, manifest, tmp_path / "catalog")


def test_discontinuous_clock_cannot_be_labeled_engine_frames(tmp_path):
    doc = play()
    states = bson.decode(zlib.decompress(doc["zstates"]))
    states["states"][2]["gt"] = 4
    doc["zstates"] = zlib.compress(bson.encode(states))
    manifest, _ = staged(tmp_path, {1: [doc]})
    with pytest.raises(ValueError, match="Non-contiguous engine clock"):
        build_catalogue(tmp_path, manifest, tmp_path / "catalog")
