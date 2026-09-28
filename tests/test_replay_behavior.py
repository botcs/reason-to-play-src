"""Released replays alone supply recorded frames and original identities."""

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import pytest

from data.values import encode_value
from human.behavior import iter_plays, load_runs
from data.replay_codec import save_replay
from human.behavior import HumanPlayLoader


def recording(game="bait", ordinal=4, condition="elaborate"):
    identity = f"{ordinal + 1:024x}"
    sprites = {
        "avatar": [
            {
                "id": 0,
                "key": "avatar",
                "col": 15.0,
                "row": 272 / 35,
                "x": 525,
                "y": 280,
                "color": [20, 40, 180],
                "color_name": "DARKBLUE",
                "_uuid": "010203",
                "source_key": "(525, 280)",
                "alive": True,
            }
        ],
        "quiet": [],
    }
    return {
        "schema": "reason-to-play/human-replay",
        "schema_version": 1,
        "subject": "sub-13",
        "game": f"{game}_vgfmri4",
        "source": "human",
        "game_description": "BasicGame\n    SpriteSet\n        avatar > MovingAvatar img=colors/DARKBLUE",
        "meta": {"suggestion_level": condition},
        "total_frames": 2,
        "steps": [],
        "plays": [
            {
                "_id": identity,
                "subj_id": 13,
                "run_id": 1,
                "play_id": ordinal,
                "game_name": f"vgfmri4_{game}",
                "level_id": 0,
                "win": None,
                "score": 0,
                "source_document_index": ordinal,
                "state_start": 0,
                "state_count": 2,
                "block_size": 35,
                "grid_size": [20, 15],
                "outcome": "incomplete",
                "start_time": encode_value(datetime(2021, 1, 1)),
                "scanner": {"subj_id": 13, "run_id": 1, "scan_start_ts": 999.0},
            }
        ],
        "states": [
            {
                "time": index,
                "realworld_ts": 1000 + index * 0.05,
                "source_play_id": identity,
                "score": 0,
                "ended": False,
                "win": -1,
                "keyPressType": 273 if index else None,
                "keystate": [False, True],
                "effectListByClass": [],
                "effectListByColor": [],
                "sprites": deepcopy(sprites),
            }
            for index in range(2)
        ],
    }


def write_record(root, record):
    path = (
        root
        / record["subject"]
        / record["game"]
        / f"{record['meta']['suggestion_level']}.human.replay.json.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_replay(record, path)
    return path


def test_standalone_replay_restores_measured_inputs_without_source_archive(tmp_path):
    path = write_record(tmp_path, recording())
    (play,) = iter_plays(path)
    assert play["start_time"] == datetime(2021, 1, 1)
    assert play["_canonical"]["source_document_index"] == 4
    assert (
        play["_canonical"]["source_recording"]
        == "sub-13/bait_vgfmri4/elaborate.human.replay.json.gz"
    )
    assert play["game_str"] == recording()["game_description"]
    assert "level_str" not in play
    obj = play["states"][0]["objects"]["avatar"]["(525, 280)"]
    assert obj["rect"] == {"pos": [525, 272], "size": [35, 35]}
    assert (obj["x"], obj["y"]) == (525, 280)
    assert obj["ID"] == bytes([1, 2, 3])
    assert "resources" not in obj
    assert play["states"][0]["objects"]["quiet"] == {}
    assert load_runs(path)[13, 1]["scan_start_ts"] == 999.0


def test_website_copy_cannot_supply_button_nuisance_regressors(tmp_path):
    from analysis.neural.prepare_inputs import extract_behavioral_features

    canonical = recording()
    for frame in canonical["states"]:
        frame["keystate"] = [False] * 277
    canonical["states"][1]["keystate"][273] = True
    path = write_record(tmp_path / "canonical", canonical)
    (play,) = iter_plays(path)
    assert extract_behavioral_features(play)["any_keypress"].tolist() == [0, 1]

    compact = deepcopy(canonical)
    for frame in compact["states"]:
        frame.pop("keystate")
    compact_path = write_record(tmp_path / "website-assets", compact)
    (display_play,) = iter_plays(compact_path)
    with pytest.raises(ValueError, match="complete behavior/human recording"):
        extract_behavioral_features(display_play)


def test_conditions_are_selected_once_and_original_run_order_is_preserved(tmp_path):
    for condition in ("elaborate", "minimal", "oracle"):
        write_record(tmp_path, recording("bait", 4, condition))
        write_record(tmp_path, recording("helper", 1, condition))
    plays = list(iter_plays(tmp_path))
    assert [p["_canonical"]["source_document_index"] for p in plays] == [1, 4]
    assert len(list(iter_plays(tmp_path, condition="minimal"))) == 2
    loader = HumanPlayLoader(str(tmp_path))
    assert loader.list_subjects() == ["sub-13"]
    assert loader.list_runs("sub-13") == [1]
    assert [p["play_idx"] for p in loader.list_plays("sub-13", 1)] == [1, 4]
    assert loader.load_play("sub-13", 1, 4)[0]["_id"] == f"{5:024x}"
    with pytest.raises(IndexError, match="Original play ordinal"):
        loader.load_play("sub-13", 1, 0)


@pytest.mark.parametrize("damage", ["bounds", "id", "clock", "definition"])
def test_reject_inconsistent_recorded_values(tmp_path, damage):
    record = recording()
    if damage == "bounds":
        record["plays"][0]["state_start"] = 1
    elif damage == "id":
        record["states"][0]["source_play_id"] = "different"
    elif damage == "clock":
        record["states"][1]["time"] = 3
    else:
        record["original_game_description"] = "a conflicting second definition"
    path = write_record(tmp_path, record)
    with pytest.raises(ValueError):
        list(iter_plays(path))


def test_conflicting_scanner_clocks_are_rejected(tmp_path):
    write_record(tmp_path, recording("bait", 4))
    second = recording("helper", 1)
    second["plays"][0]["scanner"]["scan_start_ts"] = 998.0
    write_record(tmp_path, second)
    with pytest.raises(ValueError, match="Conflicting embedded scanner"):
        load_runs(tmp_path)


def test_installed_package_reading_does_not_import_bson_or_gameplay(tmp_path):
    path = write_record(tmp_path, recording())
    code = """
import importlib.abc, sys
class DenyOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'bson', 'pymongo', 'agents', 'environments', 'pygame', 'gym'}:
            raise AssertionError('Unexpected runtime dependency: ' + name)
sys.meta_path.insert(0, DenyOptional())
from human.behavior import iter_plays, load_runs
assert len(list(iter_plays(sys.argv[1]))) == 1
assert load_runs(sys.argv[1])[13, 1]['scan_start_ts'] == 999.0
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run(
        [sys.executable, "-c", code, str(path)], cwd=tmp_path, env=env, check=True
    )


def test_replay_loader_keeps_all_idle_plays_and_rejects_non_json_inputs(tmp_path):
    root = tmp_path / "released"
    record = recording()
    for frame in record["states"]:
        frame["keyPressType"] = None
        frame["keystate"] = [False, False]
    path = write_record(root, record)
    loader = HumanPlayLoader(str(path))
    assert loader.list_subjects() == ["sub-13"]
    assert loader.list_runs("sub-13") == [1]
    assert loader.get_num_plays("sub-13", 1) == 1
    original, frames = loader.load_play("sub-13", 1, 4)
    assert original["_id"] == record["plays"][0]["_id"]
    assert len(frames) == 2 and all(frame["keyPressType"] is None for frame in frames)
    non_json = tmp_path / "non-json" / "plays" / "sub-13"
    non_json.mkdir(parents=True)
    (non_json / "run-01.bson").write_bytes(b"not a JSON recording")
    with pytest.raises(FileNotFoundError, match="No human JSON recordings"):
        HumanPlayLoader(str(non_json.parent.parent))


def test_play_listing_does_not_expand_frames_and_loading_caches_one_game(
    tmp_path, monkeypatch
):
    from human import behavior

    first = write_record(tmp_path, recording("bait", 4))
    second = write_record(tmp_path, recording("helper", 1))
    expected = {str(play["_id"]): play for play in iter_plays(tmp_path)}
    reads = []
    original_read = behavior.read_record

    def tracked_read(path, *, expand=True):
        reads.append((Path(path), expand))
        return original_read(path, expand=expand)

    monkeypatch.setattr(behavior, "read_record", tracked_read)
    loader = HumanPlayLoader(str(tmp_path))
    assert loader.list_runs("sub-13") == [1]
    assert [p["play_idx"] for p in loader.list_plays("sub-13", 1)] == [1, 4]
    assert loader.get_num_plays("sub-13", 1) == 2
    assert len(reads) == 2 and all(not expanded for _, expanded in reads)
    one, states = loader.load_play("sub-13", 1, 4)
    assert one == expected[one["_id"]] and states == one["states"]
    again, _ = loader.load_play("sub-13", 1, 4)
    assert again is one
    assert [path for path, expanded in reads if expanded] == [first]
    two, states = loader.load_play("sub-13", 1, 1)
    assert two == expected[two["_id"]] and states == two["states"]
    assert [path for path, expanded in reads if expanded] == [first, second]
    assert list(loader._cached_documents) == [two["_id"]]
