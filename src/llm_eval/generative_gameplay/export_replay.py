# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
Replay file utilities: state capture and .generative.replay.json.gz writing.

The agent produces .generative.replay.json.gz files inline during runs.
This module provides the two functions used by the agent:
  - capture_state(): snapshot the current game state
  - write_replay_file(): normalize and write the replay file
"""

import os

import gym

from src.llm_eval.shared.replay_codec import save_replay


def capture_state(env: gym.Env) -> dict:
    """Capture the current game state as a JSON-serializable dict.

    All positions are grid coordinates (col, row), not pixels.
    """
    game = env.unwrapped.game
    block_size = game.block_size

    sprites = {}
    for key in game.sprite_registry.sprite_keys:
        live = game.sprite_registry._live_sprites_by_key.get(key, [])
        dead = game.sprite_registry._dead_sprites_by_key.get(key, [])
        all_sprites = live + dead

        if not all_sprites:
            continue

        sprite_list = []
        for sprite in all_sprites:
            col = sprite.rect.x // block_size
            row = sprite.rect.y // block_size

            # Convert resources from defaultdict to plain dict
            resources = dict(sprite.resources) if sprite.resources else {}

            # Convert orientation to plain dict if present
            orientation = None
            if hasattr(sprite, "orientation"):
                ori = sprite.orientation
                if hasattr(ori, "x") and hasattr(ori, "y"):
                    orientation = {"x": float(ori.x), "y": float(ori.y)}

            sprite_data = {
                "id": sprite.id,
                "key": sprite.key,
                "col": col,
                "row": row,
                "alive": sprite.alive,
                "resources": resources,
            }

            # Optional attributes - only include if meaningful
            if sprite.speed is not None:
                sprite_data["speed"] = sprite.speed
            if hasattr(sprite, "cooldown") and sprite.cooldown:
                sprite_data["cooldown"] = sprite.cooldown
            if orientation is not None:
                sprite_data["orientation"] = orientation
            if hasattr(sprite, "_age"):
                sprite_data["_age"] = sprite._age
            if hasattr(sprite, "lastmove"):
                sprite_data["lastmove"] = sprite.lastmove

            sprite_list.append(sprite_data)

        sprites[key] = sprite_list

    # Generative mode plays through the VGDL engine directly, which
    # resolves every terminal state as either won or lost -- there is
    # no "level time budget" abstraction above the engine.  We still
    # emit `lose` and `timeout` explicitly so the viewer can use the
    # same schema contract for both replay sources.
    won = game.won
    lose = bool(game.ended) and not won
    return {
        "score": game.score,
        "time": game.time,
        "ended": game.ended,
        "won": won,
        "lose": lose,
        "timeout": False,
        "sprites": sprites,
    }


def write_replay_file(log_data: dict, states: list[dict], output_path: str) -> str:
    """Normalize log data and write a .replay.json.gz file.

    Args:
        log_data: The raw log_data dict from the agent run.
        states: List of game state dicts captured at each step.
        output_path: Path for the output .replay.json.gz file.

    Returns:
        The output file path.
    """
    steps = log_data["steps"]

    # Keys to strip from each step -- large or redundant for the viewer
    strip_keys = {
        "system_prompt",  # hoisted to top level
        "raw_response",  # large, redundant
        "messages",  # not stored inline; reconstructed at load time
    }

    system_prompt = log_data["system_prompt"]

    stripped_steps = []
    for step in steps:
        stripped = {k: v for k, v in step.items() if k not in strip_keys}

        existing_response = stripped.get("response", {})
        action = existing_response.get("action", stripped.get("action", ""))

        stripped["response"] = {
            "rationale": stripped.get("rationale", "")
            or existing_response.get("rationale", ""),
            "action": action,
        }

        stripped_steps.append(stripped)

    output = {k: v for k, v in log_data.items() if k != "steps"}
    output["source"] = "generative"
    output["system_prompt"] = system_prompt
    output["steps"] = stripped_steps
    output["states"] = states

    # Ensure path ends with .gz
    if not output_path.endswith(".gz"):
        output_path += ".gz"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_replay(output, output_path)

    return output_path
