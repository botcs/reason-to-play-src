"""Replay visuals preserve rendering rectangles independently of model inputs."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.llm_eval.generative_gameplay.export_replay import capture_state
from src.llm_eval.human_replay.zstate_adapter import (
    convert_zstate_to_viewer,
    realign_zstate_positions,
)
from src.llm_eval.shared.replay_codec import load_replay, save_replay


def frame(tick=0, *, x=35, y=70, rect_pos=(43.75, 61.25)):
    return {
        "objects": {
            "avatar": {
                "source-position": {
                    "ID": b"avatar",
                    "colorName": "YELLOW",
                    "x": x,
                    "y": y,
                    "rect": {"pos": list(rect_pos), "size": [35, 35]},
                    "resources": {},
                }
            }
        },
        "gt": tick,
        "ts": 1000 + 0.05 * tick,
        "score": tick,
        "ended": False,
        "win": None,
        "keyPressType": "up" if tick else None,
        "keystate": [int(tick > 0 and key == 273) for key in range(300)],
        "effectListByColor": [],
    }


@pytest.mark.parametrize("position", [(43.75, 61.25), (-8.75, 87.5)])
def test_human_visual_uses_rendered_rectangle_without_flooring(position):
    source = frame(rect_pos=position)
    unchanged = deepcopy(source)
    visual = convert_zstate_to_viewer(source, 35)["sprites"]["avatar"][0]
    assert (visual["col"], visual["row"]) == tuple(v / 35 for v in position)
    assert source == unchanged


@pytest.mark.parametrize("rectangle", [None, {}, {"size": [35, 35]}])
def test_object_without_rectangle_position_has_explicit_xy_fallback(rectangle):
    source = frame(x=-8.75, y=87.5)
    source["objects"]["avatar"]["source-position"]["rect"] = rectangle
    visual = convert_zstate_to_viewer(source, 35)["sprites"]["avatar"][0]
    assert (visual["col"], visual["row"]) == (-0.25, 2.5)


def test_generative_visual_preserves_fractional_rectangle_and_codec(tmp_path):
    sprite = SimpleNamespace(
        id=1,
        key="avatar",
        rect=SimpleNamespace(x=43.75, y=-8.75),
        alive=True,
        resources={},
        speed=0.25,
    )
    game = SimpleNamespace(
        block_size=35,
        sprite_registry=SimpleNamespace(
            sprite_keys=["avatar"],
            _live_sprites_by_key={"avatar": [sprite]},
            _dead_sprites_by_key={},
        ),
        won=False,
        ended=False,
        score=0,
        time=0,
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(game=game))
    first = capture_state(env)
    assert (
        first["sprites"]["avatar"][0]["col"],
        first["sprites"]["avatar"][0]["row"],
    ) == (1.25, -0.25)
    assert (sprite.rect.x, sprite.rect.y) == (43.75, -8.75)
    sprite.rect.x += 8.75
    game.time += 1
    second = capture_state(env)
    replay = {"states": [first, deepcopy(first), second]}
    path = tmp_path / "fractional.replay.json.gz"
    save_replay(replay, path)
    assert load_replay(path) == replay


@pytest.mark.parametrize(
    "source_recording",
    [None, "sub-13/avoidGeorge_vgfmri4/elaborate.human.replay.json.gz"],
)
def test_process_game_embeds_raw_visuals_and_keeps_prompt_alignment(
    tmp_path, monkeypatch, source_recording
):
    from src.llm_eval.human_replay import run_replay
    from src.llm_eval.shared.config import HarnessConfig, LLMConfig, ReplayConfig

    raw = [
        frame(tick, x=35 * (tick + 1), rect_pos=(43.75 + 35 * tick, 61.25))
        for tick in range(3)
    ]
    raw[-1].update(ended=True, win=-1)
    unchanged = deepcopy(raw)
    expected_aligned = realign_zstate_positions(raw, True)
    captured = {}
    original_run = run_replay.ReplayAgent.run_replay

    def capture_prompt_inputs(self, states, **kwargs):
        assert states == expected_aligned
        captured["records"] = original_run(self, states, **kwargs)
        return captured["records"]

    monkeypatch.setattr(run_replay.ReplayAgent, "run_replay", capture_prompt_inputs)
    doc = {"_id": "original-play", "win": True}
    loader = SimpleNamespace(load_play=lambda *args: (doc, raw), per_game=False)
    play = {
        "run": 1,
        "play_idx": 0,
        "game_name": "vgfmri4_avoidgeorge",
        "level_id": 0,
        "win": True,
        "source_play_id": "original-play",
        "source_recording": source_recording,
    }
    result = run_replay.process_game(
        play["game_name"],
        [play],
        "sub-13",
        HarnessConfig(rationale_mode="action-only", suggestion_level="minimal"),
        ReplayConfig(data_dir=str(tmp_path / "absent-source")),
        LLMConfig(backend="mock"),
        tmp_path,
        "geometry",
        loader,
    )
    replay = load_replay(result["output_path"])
    assert len(replay["states"]) == len(raw)
    assert len(replay["steps"]) == 2
    assert replay["system_prompt"]
    assert replay["game_description"]
    assert replay["plays"][0]["source_play_id"] == "original-play"
    for original, visual in zip(raw, replay["states"], strict=True):
        position = original["objects"]["avatar"]["source-position"]["rect"]["pos"]
        sprite = visual["sprites"]["avatar"][0]
        assert [sprite["col"], sprite["row"]] == [v / 35 for v in position]
        assert visual["time"] == original["gt"]
    assert replay["states"][-1]["won"] is True
    for record, step in zip(captured["records"], replay["steps"], strict=True):
        assert step["formatted_obs"] == record["formatted_obs"]
        assert step["realworld_ts"] == record["realworld_ts"]
        assert step["source_play_id"] == "original-play"
        assert step["source_recording"] == source_recording
    assert raw == unchanged
