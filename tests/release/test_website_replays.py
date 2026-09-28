from copy import deepcopy
import gzip
import json

import pytest

from scripts.release.build_website_replays import (
    compact_replay,
    verify_replay,
    frames,
    build_one,
    sha256_file,
    safe_path,
)


def fixture():
    return {
        "game": "helper_vgfmri4",
        "source": "human",
        "game_description": "BasicGame\n  SpriteSet\n    avatar > MovingAvatar",
        "system_prompt": "Prompt with Unicode: →",
        "steps": [
            {
                "action": "down",
                "state_index": 1,
                "realworld_ts": 12.5,
                "response": {"rationale": "move"},
                "prompt": [{"role": "user", "content": "go"}],
            }
        ],
        "plays": [{"state_start": 0, "state_count": 4, "_id": "123", "win": None}],
        "states": [
            {
                "sprites": {
                    "avatar": [
                        {
                            "id": "a",
                            "key": "avatar",
                            "col": 0.125,
                            "row": 0.75,
                            "orientation": [0, -1],
                            "resources": {"red": 1},
                            "lastmove": i,
                        }
                    ],
                    "wall": [],
                },
                "time": i,
                "level": 0,
                "attempt": 0,
                "score": 1,
                "realworld_ts": 10 + i,
                "keystate": {"left": False},
                "kill_list_ID": ["x"],
                "effectList": [["killSprite"]],
                "action_log": "no-op",
            }
            for i in range(4)
        ],
    }


def test_retained_values_fractional_orientation_conversation_and_input_immutable():
    data = fixture()
    original = deepcopy(data)
    compact, evidence = compact_replay(data)
    assert data == original
    verify_replay(json.loads(json.dumps(compact)), evidence)
    assert compact["steps"] == original["steps"]
    assert compact["plays"] == original["plays"]
    assert compact["game_description"] == original["game_description"]
    assert "sprites" not in compact["states"][1]
    assert evidence["fractional_sprite_records"] == 4
    assert evidence["omitted_stored_fields"] == {
        "frame.keystate": 4,
        "frame.kill_list_ID": 4,
        "sprite.lastmove": 4,
    }
    assert list(frames(compact))[1][1]["avatar"][0]["orientation"] == [0, -1]
    damaged = deepcopy(compact)
    damaged["states"][0]["sprites"]["avatar"][0]["col"] = 0
    with pytest.raises(ValueError, match="retained frame"):
        verify_replay(damaged, evidence)
    damaged = deepcopy(compact)
    damaged["steps"][0]["prompt"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="conversation"):
        verify_replay(damaged, evidence)


def test_delta_removal_empty_type_and_level_reset():
    data = fixture()
    data["states"][1]["sprites"] = {
        "avatar": None,
        "enemy": [{"id": "e", "col": 2, "row": 3, "lastmove": 4}],
    }
    data["states"][2].pop("sprites")
    data["states"][3]["sprites"] = {
        "enemy": None,
        "avatar": [{"id": "b", "col": 0, "row": 0}],
    }
    data["states"][3]["time"] = 0
    data["states"][3]["attempt"] = 1
    data["delta_encoded"] = True
    compact, evidence = compact_replay(data)
    verify_replay(compact, evidence)
    expanded = list(frames(compact))
    assert "avatar" not in expanded[1][1]
    assert expanded[2][1]["enemy"] == [{"id": "e", "col": 2, "row": 3}]
    assert "enemy" not in expanded[3][1]
    assert expanded[3][1]["wall"] == []
    assert compact["states"][3]["attempt"] == 1


def test_build_one_pinned_bytes_deterministic_output_and_readonly_source(tmp_path):
    relative = "behavior/human/sub-01/helper_vgfmri3/elaborate.human.replay.json.gz"
    source = tmp_path / "source" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(gzip.compress(json.dumps(fixture()).encode(), mtime=0))
    row = {
        "release_path": relative,
        "artifact_id": "sha256:" + sha256_file(source),
        "payload": {"sha256": sha256_file(source), "size_bytes": source.stat().st_size},
    }
    first, evidence = build_one(
        (str(tmp_path / "source"), str(tmp_path / "output1"), row)
    )
    second, _ = build_one((str(tmp_path / "source"), str(tmp_path / "output2"), row))
    assert first["payload"] == second["payload"]
    assert sha256_file(source) == row["payload"]["sha256"]
    assert first["source"]["canonical_release_path"] == relative
    row["payload"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_one((str(tmp_path / "source"), str(tmp_path / "output3"), row))
    assert not (tmp_path / "output3").exists()


def test_unsafe_path_rejected(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../file")
    with pytest.raises(ValueError):
        safe_path(tmp_path, "/absolute")


def test_missing_first_delta_state_rejected():
    data = fixture()
    data["delta_encoded"] = True
    data["states"][0].pop("sprites")
    with pytest.raises(ValueError, match="First delta frame"):
        compact_replay(data)


@pytest.mark.parametrize("compact_index", [False, True])
def test_full_builder_uses_canonical_bytes_from_either_index(tmp_path, compact_index):
    from scripts.release.build_website_replays import build_website_assets

    source = tmp_path / "source"
    relative = "behavior/human/sub-01/helper_vgfmri3/elaborate.human.replay.json.gz"
    replay = source / relative
    replay.parent.mkdir(parents=True)
    replay.write_bytes(gzip.compress(json.dumps(fixture()).encode(), mtime=0))
    row = {
        "release_path": relative,
        "artifact_id": "sha256:" + sha256_file(replay),
        "payload": {"sha256": sha256_file(replay), "size_bytes": replay.stat().st_size},
    }
    (source / "manifest.jsonl.gz").write_bytes(
        gzip.compress((json.dumps(row) + "\n").encode(), mtime=0)
    )
    index = source / "website-assets/replays/manifest.json"
    index.parent.mkdir(parents=True)
    asset = "website-assets/replays/" + relative.removeprefix("behavior/")
    index.write_text(
        json.dumps(
            {
                "cohort3": {
                    "sub-01": {
                        "stats": {"wins": 1},
                        "replays": {
                            "helper_vgfmri3": asset if compact_index else relative
                        },
                    }
                }
            }
        )
    )
    before = sha256_file(index)
    report = build_website_assets(source, tmp_path / "built", workers=2)
    assert report["files"] == 1
    assert sha256_file(index) == before
    result = json.loads(
        (tmp_path / "built/dataset/website-assets/replays/manifest.json").read_text()
    )
    assert result["cohort3"]["sub-01"]["replays"]["helper_vgfmri3"] == asset
    assert result["cohort3"]["sub-01"]["stats"] == {"wins": 1}
    addition = json.loads((tmp_path / "built/additions.jsonl").read_text())
    assert addition["source"]["canonical_sha256"] == sha256_file(replay)
    assert addition["payload"]["sha256"] == sha256_file(
        tmp_path / "built/dataset" / asset
    )
    assert (
        json.loads((tmp_path / "built/index-row.json").read_text())["release_path"]
        == "website-assets/replays/manifest.json"
    )
    with pytest.raises(ValueError, match="outside the source"):
        build_website_assets(source, source, workers=2)
    with pytest.raises(ValueError, match="absent from the catalogue"):
        build_website_assets(
            source, tmp_path / "invalid", workers=2, select=["behavior/human/missing"]
        )
