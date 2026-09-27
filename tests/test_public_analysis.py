"""Scientific data contracts for the offline public episode exporter."""

import gzip
import json
import zlib

import bson
import pytest

from scripts.analysis.build_episodes import human_outcome, human_rows, replay_rows


def human_doc(win=None, died=False):
    states = [
        {"gt": 0, "keyPressType": None, "effectListByClass": []},
        {
            "gt": 1,
            "keyPressType": 39,
            "effectListByClass": [["killSprite", "avatar", "enemy"]] if died else [],
        },
    ]
    return {
        "game_name": "vgfmri3_bait",
        "level_id": 0,
        "win": win,
        "game_str": "SpriteSet\n    avatar > MovingAvatar",
        "zstates": zlib.compress(bson.encode({"states": states})),
    }, states


def test_avatar_death_is_not_incomplete():
    doc, states = human_doc(died=True)
    assert human_outcome(doc, states) == "avatar_died"
    doc, states = human_doc()
    assert human_outcome(doc, states) == "incomplete"
    doc["win"] = False
    assert human_outcome(doc, states) == "loss"
    doc["win"] = True
    assert human_outcome(doc, states) == "win"
    doc["win"] = -1
    with pytest.raises(ValueError, match="three-valued"):
        human_outcome(doc, states)


def test_human_clock_and_cohort_filters(tmp_path):
    plays = tmp_path / "plays" / "sub-01"
    plays.mkdir(parents=True)
    doc, _ = human_doc(died=True)
    (plays / "run-00.bson").write_bytes(bson.encode(doc))
    beyond = dict(doc, level_id=9)
    (plays / "run-01.bson").write_bytes(bson.encode(doc) + bson.encode(beyond))
    rows = human_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["episode_steps"] == [1]
    assert rows[0]["episode_frames"] == [1]
    assert rows[0]["episode_outcomes"] == ["avatar_died"]
    assert rows[0]["cohort"] == "vgfmri3"


def test_replay_synthetic_markers_and_resumed_fragments(tmp_path):
    replay = {
        "source": "generative",
        "game": "bait_vgfmri4",
        "started_at": "2026-09-27",
        "meta": {
            "model": "test/model",
            "seed": 0,
            "rationale_mode": "action-only",
            "suggestion_level": "minimal",
        },
        "steps": [
            {"level": 0, "attempt": 0, "action": "right", "won": True},
            {"level": 0, "attempt": 0, "action": "_level_advance"},
            {"level": 0, "attempt": 1, "action": "left", "lose": True},
        ],
        "states": [
            {"level": 0, "attempt": 0, "time": 5},
            {"level": 0, "attempt": 1, "time": 3},
        ],
    }
    path = tmp_path / "test.generative.replay.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(replay, stream)
    (row,) = replay_rows(path)
    assert row["episode_steps"] == [1, 1]
    assert row["episode_frames"] == [5, 3]
    assert row["episode_outcomes"] == ["win", "loss"]
    replay["meta"]["resumed_at_level"] = 1
    with gzip.open(path, "wt") as stream:
        json.dump(replay, stream)
    with pytest.raises(ValueError, match="Merge resumed"):
        replay_rows(path)
