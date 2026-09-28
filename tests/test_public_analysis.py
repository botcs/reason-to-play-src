"""Analysis data contracts for the offline public episode exporter."""

import gzip
import json

import pytest

from reason_to_play.analysis.behavioral.episodes import (
    human_outcome,
    human_rows,
    replay_rows,
)
from test_replay_behavior import recording, write_record


def human_doc(win=None, died=False):
    states = [
        {"gt": 0, "ts": 10.0, "keyPressType": None, "effectListByClass": []},
        {
            "gt": 1,
            "ts": 10.05,
            "keyPressType": 39,
            "effectListByClass": [["killSprite", "avatar", "enemy"]] if died else [],
        },
    ]
    return {
        "_id": "000000000000000000000001",
        "subj_id": 1,
        "run_id": 1,
        "play_id": 1,
        "game_name": "vgfmri3_bait",
        "level_id": 0,
        "win": win,
        "game_str": "SpriteSet\n    avatar > MovingAvatar",
        "level_str": "A",
        "states": states,
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
    canonical = tmp_path / "canonical"
    record = recording()
    record.update(subject="sub-01", game="bait_vgfmri3", total_frames=6)
    from copy import deepcopy

    template = deepcopy(record)
    record["plays"], record["states"] = [], []
    for index, (run, level) in enumerate([(0, 0), (1, 0), (1, 9)]):
        identity = f"{index + 1:024x}"
        play = deepcopy(template["plays"][0])
        play.update(
            _id=identity,
            subj_id=1,
            run_id=run,
            level_id=level,
            game_name="vgfmri3_bait",
            source_document_index=index,
            state_start=2 * index,
            outcome="avatar_died",
        )
        play["scanner"].update(subj_id=1, run_id=run, scan_start_ts=9.0)
        record["plays"].append(play)
        for frame in deepcopy(template["states"]):
            frame["source_play_id"] = identity
            if frame["time"]:
                frame["effectListByClass"] = [["killSprite", "avatar", "enemy"]]
            record["states"].append(frame)
    write_record(canonical, record)
    rows = human_rows(canonical)
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


def test_human_rows_records_actual_per_game_source_path(monkeypatch):
    from reason_to_play.analysis.behavioral import episodes

    source = "sub-13/bait_vgfmri4/elaborate.human.replay.json.gz"
    plays = []
    for run, ordinal, keys in [(1, 4, [None, "right"]), (5, 1, [None, None])]:
        play, _ = human_doc()
        play.update(
            subj_id=13,
            game_name="vgfmri4_bait",
            run_id=run,
            states=[
                {"gt": index, "keyPressType": key, "effectListByClass": []}
                for index, key in enumerate(keys)
            ],
            _canonical={"source_recording": source, "source_document_index": ordinal},
        )
        plays.append(play)
    monkeypatch.setattr(episodes, "iter_plays", lambda root: iter(plays))
    (row,) = episodes.human_rows("unused")
    assert row["episode_steps"] == [1, 0]
    assert row["episode_frames"] == [1, 1]
    assert row["episode_outcomes"] == ["incomplete", "incomplete"]
    assert row["data_source"] == [source]


@pytest.mark.parametrize(
    "path",
    [
        None,
        "plays/sub-01/run-01.json.gz",
        "sub-12/bait_vgfmri3/elaborate.human.replay.json.gz",
        "sub-01/helper_vgfmri3/elaborate.human.replay.json.gz",
    ],
)
def test_human_rows_rejects_missing_or_fictitious_source_path(monkeypatch, path):
    from reason_to_play.analysis.behavioral import episodes

    play, _ = human_doc()
    play["_canonical"] = {"source_document_index": 0, "source_recording": path}
    monkeypatch.setattr(episodes, "iter_plays", lambda root: iter([play]))
    with pytest.raises(ValueError, match="source_recording"):
        episodes.human_rows("unused")
