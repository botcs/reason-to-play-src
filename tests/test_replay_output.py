"""New prompts retain complete measured records without external source files."""

from copy import deepcopy

import pytest

from human.behavior import iter_plays, load_runs
from data.replay_codec import save_replay
from agents.lrm.prompting.output import DISPLAY_FIELDS, merge_human_prompt_output
from agents.lrm.prompting.conversation import load_prompts
from test_replay_behavior import recording


def source_and_generated():
    source = recording()
    second = recording(ordinal=8)
    play = second["plays"][0]
    play.update(run_id=4, state_start=2)
    play["scanner"].update(run_id=4, scan_start_ts=1999.0)
    source["plays"].append(play)
    for state in second["states"]:
        state["realworld_ts"] += 1000
    source["states"].extend(second["states"])
    source["total_frames"] = 4
    for frame in source["states"]:
        frame.update(step=999, action_log="old display")
    generated = {
        "subject": source["subject"],
        "game": source["game"],
        "source": "human",
        "game_description": "must not replace authoritative input definition",
        "system_prompt": "new exact system prompt",
        "prompt_name": "action-only_minimal",
        "suggestion_level": "minimal",
        "model": "human (sub-13)",
        "start_level": 0,
        "started_at": "now",
        "finished_at": "later",
        "outcome": "completed",
        "total_steps": 1,
        "total_frames": 4,
        "meta": {
            "completed": True,
            "rationale_mode": "action-only",
            "suggestion_level": "minimal",
            "num_trials": 2,
            "game": source["game"],
            "model": "human (sub-13)",
            "subject": "sub-13",
            "pipeline": "unified",
        },
        "states": [
            {
                "time": i % 2,
                "level": 0,
                "attempt": i // 2,
                "won": False,
                "lose": False,
                "timeout": False,
                **({"action_log": "new display"} if i % 2 == 0 else {}),
            }
            for i in range(4)
        ],
        "plays": [
            {
                "source_play_id": p["_id"],
                "run": p["run_id"],
                "source_document_index": p["source_document_index"],
                "game_name": p["game_name"],
                "level_id": p["level_id"],
                "win": p["win"],
                "outcome": p["outcome"],
            }
            for p in source["plays"]
        ],
        "steps": [
            {
                "action": "up",
                "step": 0,
                "source_play_id": play["_id"],
                "source_recording": "sub-13/bait_vgfmri4/elaborate.human.replay.json.gz",
                "source_frame_index": 1,
                "source_document_index": 8,
                "play_idx": 8,
                "frame_idx": 1,
                "run": 4,
                "trial_idx": 1,
                "level": 0,
                "attempt": 1,
                "realworld_ts": 2000.05,
                "state_index": 3,
                "frame": 3,
                "play_id": "sub-13_4_8",
                "formatted_obs": "exact new observation\nwith spaces  ",
                "response": {"action": "up", "rationale": ""},
                "won": False,
                "lose": False,
                "timeout": False,
                "reward": 0,
            }
        ],
    }
    return source, generated


def test_preserve_science_and_zero_prompt_plays_with_new_condition(tmp_path):
    source, generated = source_and_generated()
    before_source, before_generated = deepcopy(source), deepcopy(generated)
    output = merge_human_prompt_output(source, generated)
    assert source == before_source and generated == before_generated
    assert output["plays"] == source["plays"]
    assert output["game_description"] == source["game_description"]
    assert output["steps"][0]["formatted_obs"] == generated["steps"][0]["formatted_obs"]
    assert output["steps"] == [
        {
            **generated["steps"][0],
            "source_recording": "sub-13/bait_vgfmri4/minimal.human.replay.json.gz",
        }
    ]
    assert (
        output["steps"][0]["source_recording"]
        == "sub-13/bait_vgfmri4/minimal.human.replay.json.gz"
    )
    for original, current in zip(source["states"], output["states"], strict=True):
        assert {k: v for k, v in original.items() if k not in DISPLAY_FIELDS} == {
            k: v for k, v in current.items() if k not in DISPLAY_FIELDS
        }
        assert "step" not in current
    assert output["states"][0]["action_log"] == "new display"
    assert "action_log" not in output["states"][1]
    path = tmp_path / "minimal.human.replay.json.gz"
    save_replay(output, path)
    assert len(list(iter_plays(path))) == 2
    assert set(load_runs(path)) == {(13, 1), (13, 4)}
    prompts, meta = load_prompts(path)
    assert len(prompts) == 1 and meta["num_trials"] == 2
    assert prompts[0]["messages"][-2]["content"].endswith(
        "exact new observation\nwith spaces  "
    )


@pytest.mark.parametrize(
    "damage",
    [
        "subset",
        "order",
        "ordinal",
        "time",
        "frame",
        "duplicate",
        "synthetic",
        "partial",
        "clock",
    ],
)
def test_reject_incomplete_or_inconsistent_conversation(damage):
    source, generated = source_and_generated()
    if damage == "subset":
        generated["plays"].pop()
    elif damage == "order":
        generated["plays"].reverse()
    elif damage == "ordinal":
        generated["steps"][0]["source_document_index"] = 0
    elif damage == "time":
        generated["steps"][0]["realworld_ts"] += 0.01
    elif damage == "frame":
        generated["steps"][0]["state_index"] = 1
    elif damage == "duplicate":
        generated["steps"].append(deepcopy(generated["steps"][0]))
        generated["total_steps"] = 2
    elif damage == "synthetic":
        generated["steps"][0]["action"] = "_level_advance"
    elif damage == "partial":
        generated["meta"]["completed"] = False
    else:
        generated["states"][1]["time"] = 0
    with pytest.raises(ValueError):
        merge_human_prompt_output(source, generated)


def test_reject_reordered_step_sequence():
    source, generated = source_and_generated()
    first = deepcopy(generated["steps"][0])
    play = source["plays"][0]
    first.update(
        source_play_id=play["_id"],
        source_frame_index=1,
        source_document_index=4,
        play_idx=4,
        run=1,
        trial_idx=0,
        attempt=0,
        realworld_ts=1000.05,
        state_index=1,
        frame=1,
        play_id="sub-13_1_4",
    )
    generated["steps"].append(first)
    generated["total_steps"] = 2
    with pytest.raises(ValueError, match="reordered"):
        merge_human_prompt_output(source, generated)


def test_reject_conflicting_prompt_condition():
    source, generated = source_and_generated()
    generated["meta"]["suggestion_level"] = "oracle"
    with pytest.raises(ValueError, match="metadata"):
        merge_human_prompt_output(source, generated)
