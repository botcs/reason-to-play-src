# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
#!/usr/bin/env python
"""
Entry point for human replay prompt generation using the unified Harness.

Uses Hydra for config composition and CLI overrides, mirroring
``src.llm_eval.generative_gameplay.run``. Processes behavioral data from
Tomov et al. 2023 through the same Harness pipeline used for generative
gameplay, producing .replay.json.gz files with extraction-ready messages
and viewer-compatible state snapshots.

Replay supports two rationale modes:
- 'action-only'      : no LLM, human action injected directly
                       -> .human.replay.json.gz
- 'copied-reasoning' : LLM runs with native reasoning; the hidden trace is
                       copied into context as the player's stated rationale
                       -> .imputed.replay.json.gz + .narration.replay.json.gz

'prompted-rationale' is not supported in replay.

Usage:
    # action-only (no LLM)
    python -m src.llm_eval.human_replay.run_replay \\
        replay.subject=sub-01 replay.game_filter=vgfmri3_bait \\
        harness.rationale_mode=action-only

    # copied-reasoning (requires a model with native reasoning)
    python -m src.llm_eval.human_replay.run_replay \\
        replay.subject=sub-01 replay.game_filter=vgfmri3_bait \\
        harness.rationale_mode=copied-reasoning \\
        llm.model=deepseek/deepseek-v3.2
"""

import hashlib
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

# Ensure repo root is on the path for src.* imports
repo_path = "/".join(os.path.abspath(__file__).split("/")[:-3]) + "/"
sys.path.insert(0, repo_path)

from src.llm_eval.shared.config import (  # noqa: E402
    Config,
    HarnessConfig,
    LLMConfig,
    ReplayConfig,
    rationale_mode_to_effort,
    validate_config,
)
from src.llm_eval.shared.prompt_loader import PromptLoader  # noqa: E402
from src.llm_eval.shared.replay_codec import save_replay  # noqa: E402
from src.llm_eval.shared.response_parser import ResponseParser  # noqa: E402
from src.llm_eval.human_replay.data_loader import HumanPlayLoader  # noqa: E402
from src.llm_eval.human_replay.replay_agent import ReplayAgent  # noqa: E402
from src.llm_eval.generative_gameplay.agent import (  # noqa: E402
    InteractionTracker,
    build_interactions_from_game,
)
from src.llm_eval.human_replay.zstate_adapter import (  # noqa: E402
    ZstateAdapter,
    convert_zstate_to_viewer,
    realign_zstate_positions,
)


def get_all_games_for_subject(
    loader: HumanPlayLoader, subject: str
) -> dict[str, list[dict]]:
    """Get all plays for a subject, organized by game and sorted chronologically.

    Returns:
        Dict mapping game_name to list of play info dicts, sorted by (run, play_idx).
        Each play_info has: game_name, level_id, win, score, play_idx, run.
    """
    games: dict[str, list[dict]] = defaultdict(list)

    runs = loader.list_runs(subject)
    for run in runs:
        plays = loader.list_plays(subject, run)
        for play_info in plays:
            play_info["run"] = run
            games[play_info["game_name"]].append(play_info)

    for game_name in games:
        games[game_name].sort(key=lambda p: (p["run"], p["play_idx"]))

    return dict(games)


def _harness_mode_str(cfg: HarnessConfig) -> str:
    """Return the harness rationale_mode string (used for logging)."""
    return cfg.rationale_mode


def _needs_llm(cfg: HarnessConfig) -> bool:
    """Whether the replay phase needs an LLM call per step."""
    return cfg.rationale_mode == "copied-reasoning"


_CANONICAL_VGDL_NAMES: dict[str, str] | None = None


def _canonical_vgdl_names() -> dict[str, str]:
    """Build a lowercase-to-canonical map of VGDL game names from games/.

    The behavioral data happens to lowercase some game tokens
    (e.g. `vgfmri4_avoidgeorge` where the canonical VGDL name is
    `avoidGeorge_vgfmri4`).  The naive word-swap produces a lowercase
    string that does not match the canonical directory name or any DB
    slot_id, so we look up the canonical casing in ``games/``.
    """
    global _CANONICAL_VGDL_NAMES
    if _CANONICAL_VGDL_NAMES is not None:
        return _CANONICAL_VGDL_NAMES
    games_dir = Path(__file__).resolve().parents[3] / "games"
    if not games_dir.is_dir():
        raise RuntimeError(f"games/ directory not found at {games_dir}")
    out: dict[str, str] = {}
    for d in games_dir.iterdir():
        if d.is_dir() and d.name.endswith("_v0"):
            name = d.name.removesuffix("_v0")
            out[name.lower()] = name
    if not out:
        raise RuntimeError(f"No *_v0 game directories found in {games_dir}")
    _CANONICAL_VGDL_NAMES = out
    return out


def _behavioral_to_vgdl_game_name(name: str) -> str:
    """Convert behavioral data game name to canonical VGDL registry name.

    Behavioral data uses '{version}_{game}' (e.g. 'vgfmri3_bait'),
    VGDL registry uses '{game}_{version}' (e.g. 'bait_vgfmri3').  Casing
    in behavioral data is inconsistent (`avoidgeorge` vs the canonical
    `avoidGeorge`), so the swapped lowercase is looked up against the
    games/ directory to recover the canonical casing.
    """
    parts = name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected game name format: {name!r}")
    swapped = f"{parts[1]}_{parts[0]}"
    canonical_map = _canonical_vgdl_names()
    canonical = canonical_map.get(swapped.lower())
    if canonical is None:
        raise ValueError(
            f"No canonical VGDL name for behavioral={name!r} "
            f"(attempted {swapped!r}); known: {sorted(canonical_map.values())}"
        )
    return canonical


def _build_known_interactions(
    game_name: str, color_mapping: dict[str, str]
) -> set[tuple[str, str, str]]:
    """Build known interactions for a game, remapped to randomized colors.

    Creates a temporary gym env from canonical game files, extracts interactions
    via build_interactions_from_game, then remaps canonical colors to the
    subject-specific randomized colors from color_mapping.
    """
    import gym
    from src.utils import register_game

    vgdl_name = _behavioral_to_vgdl_game_name(game_name)
    register_game(vgdl_name, level=0)
    temp_env = gym.make(f"{vgdl_name}-v0")

    canonical_known = build_interactions_from_game(temp_env.game)

    # Build canonical_color -> type_name mapping from sprite registry
    canonical_to_type: dict[str, str] = {}
    registry = temp_env.game.sprite_registry
    for key in registry.sprite_keys:
        img = registry.class_args[key].get("img")
        if img and "/" in img:
            canonical_color = img.split("/")[1].upper()
            canonical_to_type[canonical_color] = key

    # Build canonical_color -> randomized_color mapping
    color_remap: dict[str, str] = {}
    for type_name, randomized_color in color_mapping.items():
        for canon_color, t_name in canonical_to_type.items():
            if t_name == type_name:
                color_remap[canon_color] = randomized_color.upper()
                break

    # Remap known interactions to randomized colors
    remapped: set[tuple[str, str, str]] = set()
    for c1, eff, c2 in canonical_known:
        rc1 = color_remap.get(c1, c1)
        rc2 = color_remap.get(c2, c2)
        remapped.add((rc1, eff, rc2))

    return remapped


def _parse_colored_sprite_types(game_name: str) -> set[str]:
    """Parse sprite type names with img=colors/ from the canonical game description."""
    vgdl_name = _behavioral_to_vgdl_game_name(game_name)
    game_dir = Path(__file__).resolve().parents[3] / "games" / f"{vgdl_name}_v0"
    game_file = game_dir / f"{vgdl_name}.txt"
    if not game_file.exists():
        raise FileNotFoundError(f"Game description not found: {game_file}")
    types = set()
    for line in game_file.read_text().split("\n"):
        stripped = line.strip()
        if ">" in stripped and "img=colors/" in stripped:
            sprite_type = stripped.split(">")[0].strip().split()[-1]
            types.add(sprite_type)
    return types


def _rewrite_game_description(game_name: str, color_mapping: dict[str, str]) -> str:
    """Rewrite the canonical VGDL game description with actual randomized colors.

    Tomov et al. 2023 randomized color-to-sprite mappings per subject+game.
    The canonical game files in games/ already have the correct VGDL structure
    (hierarchy, EOS interactions, floor mappings, etc.) from convert_tomov23_games.py.
    We only need to substitute img=colors/ values for sprites whose color was
    randomized for this subject.

    The full game description is kept intact -- no lines are removed -- to
    preserve the sprite hierarchy that the JS replay viewer needs.
    """
    vgdl_name = _behavioral_to_vgdl_game_name(game_name)
    game_dir = Path(__file__).resolve().parents[3] / "games" / f"{vgdl_name}_v0"
    game_file = game_dir / f"{vgdl_name}.txt"
    if not game_file.exists():
        raise FileNotFoundError(f"Game description not found: {game_file}")

    desc = game_file.read_text()
    lines = desc.split("\n")
    result = []

    for line in lines:
        stripped = line.strip()

        if ">" in stripped:
            sprite_type = stripped.split(">")[0].strip().split()[-1]
            if sprite_type in color_mapping:
                line = re.sub(
                    r"img=colors/\w+",
                    f"img=colors/{color_mapping[sprite_type]}",
                    line,
                )
                line = re.sub(
                    r"(?<!\w)color=\w+",
                    f"color={color_mapping[sprite_type]}",
                    line,
                )

        result.append(line)

    return "\n".join(result)


def _build_output_snapshot(
    game_name: str,
    subject: str,
    agent: ReplayAgent,
    harness_cfg: HarnessConfig,
    harness_mode: str,
    replay_cfg: ReplayConfig,
    llm_config: LLMConfig,
    game_plays: list[dict],
    all_steps: list[dict],
    all_viewer_states: list[dict],
    total_prompts: int,
    total_frames: int,
    color_mapping: dict[str, str],
    game_description: str,
    source: str,
    system_prompt: str,
    completed: bool = False,
) -> dict:
    """Build the .replay.json.gz output dict from current accumulated state."""
    vgdl_game_name = _behavioral_to_vgdl_game_name(game_name)
    return {
        "game": vgdl_game_name,
        "source": source,
        "model": f"human ({subject})",
        "subject": subject,
        "system_prompt": system_prompt,
        "prompt_name": agent.harness.prompt_name,
        "suggestion_level": harness_cfg.suggestion_level,
        "start_level": game_plays[0]["level_id"] if game_plays else 0,
        "started_at": datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
        "outcome": "completed" if completed else "running",
        "total_steps": total_prompts,
        "total_frames": total_frames,
        "meta": {
            "game": vgdl_game_name,
            "model": f"human ({subject})",
            "subject": subject,
            "rationale_mode": harness_cfg.rationale_mode,
            "suggestion_level": harness_cfg.suggestion_level,
            "harness_mode": harness_mode,
            "action_frames_only": replay_cfg.action_frames_only,
            "stride": replay_cfg.stride,
            "num_trials": len(game_plays),
            "llm_model": llm_config.model if llm_config.model else None,
            "completed": completed,
            "pipeline": "unified",
            "created_at": datetime.now().isoformat(),
        },
        "color_mapping": color_mapping,
        "game_description": game_description,
        "steps": all_steps,
        "states": all_viewer_states,
    }


def process_game(
    game_name: str,
    game_plays: list[dict],
    subject: str,
    harness_cfg: HarnessConfig,
    replay_cfg: ReplayConfig,
    llm_config: LLMConfig,
    output_dir: Path,
    args_hash: str,
    loader: HumanPlayLoader,
    wandb_run=None,
    seed: int = 0,
    s3_sync: tuple | None = None,
) -> dict:
    """Process a single game's plays through the unified Harness pipeline.

    Produces a .replay.json.gz file with extraction-ready messages and
    viewer-compatible state snapshots. Logs metrics to wandb if wandb_run
    is provided.

    Returns dict with: game_name, total_prompts, num_trials, output_path, skipped.
    """
    harness_mode = _harness_mode_str(harness_cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # "imputed" when the LLM imputed reasoning for the human actions
    needs_llm = _needs_llm(harness_cfg)
    gameplay_tag = "imputed" if needs_llm else "human"
    output_file = (
        output_dir / f"{timestamp}_{args_hash}.{gameplay_tag}.replay.json.gz"
    ).resolve()
    narration_file = (
        (output_dir / f"{timestamp}_{args_hash}.narration.replay.json.gz").resolve()
        if needs_llm
        else None
    )

    # Instantiate LLM for copied-reasoning imputation
    llm = None
    if needs_llm and llm_config.model:
        from src.llm_eval.shared.openrouter_wrapper import OpenRouterWrapper

        effort = rationale_mode_to_effort(harness_cfg.rationale_mode)
        llm = OpenRouterWrapper(
            model=llm_config.model,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            reasoning_effort=effort,
            seed=seed,
        )
        llm.require_reasoning_support()

    prompt_loader = PromptLoader()

    # Detect block_size from first play
    first_play_doc, first_states = loader.load_play(
        subject, game_plays[0]["run"], game_plays[0]["play_idx"]
    )
    block_size = ZstateAdapter(block_size=1).detect_block_size(first_states[0])

    # Single ReplayAgent for the entire game (Harness state persists across trials)
    vgdl_game_name = _behavioral_to_vgdl_game_name(game_name)
    agent = ReplayAgent(
        harness_cfg,
        prompt_loader,
        block_size=block_size,
        llm=llm,
        max_output_tokens=llm_config.max_tokens,
        game_name=vgdl_game_name,
    )

    all_steps = []
    all_imputation_steps: list[dict] = []
    all_viewer_states = []
    total_prompts = 0
    total_frames = 0
    cumulative_score = 0
    cumulative_wins = 0
    cumulative_losses = 0
    interaction_tracker = InteractionTracker()
    _wandb = wandb_run is not None
    _response_parser = ResponseParser()

    _game_start_time = time.monotonic()

    if _wandb:
        interaction_table = wandb.Table(
            columns=[
                "step",
                "level",
                "attempt",
                "color1",
                "effect",
                "color2",
                "obj_id1",
                "obj_id2",
                "first_occurrence",
            ]
        )
    else:
        interaction_table = None

    level_attempts: dict[int, int] = defaultdict(int)

    # Extract actual color mapping from zstates.
    # Tomov et al. 2023 randomized colors per subject+game. Scan frames until
    # every sprite type with img=colors/ in the game description has been seen
    # with at least one live object. Types created mid-game by transformTo
    # (e.g. carcass in chase) have empty positions dicts until instantiated,
    # so we must scan past the first frames.
    expected_types = _parse_colored_sprite_types(game_name)
    color_mapping: dict[str, str] = {}
    for play_info in game_plays:
        if set(color_mapping) >= expected_types:
            break
        _, zs = loader.load_play(subject, play_info["run"], play_info["play_idx"])
        if not zs:
            continue
        for z in zs:
            for sprite_type, positions in z.get("objects", {}).items():
                if sprite_type in color_mapping or sprite_type not in expected_types:
                    continue
                for _pos_key, obj_data in positions.items():
                    color_name = obj_data.get("colorName")
                    if color_name is not None:
                        color_mapping[sprite_type] = color_name
                    break
            if set(color_mapping) >= expected_types:
                break
    game_description = _rewrite_game_description(game_name, color_mapping)

    # Populate known interactions (remapped to randomized colors)
    interaction_tracker.known = _build_known_interactions(game_name, color_mapping)

    # Substitute {sprite_type} placeholders in the gameplay system prompt
    # with this subject's randomized colors from color_mapping.
    for sprite_type, color_name in color_mapping.items():
        agent._gameplay_system_prompt = agent._gameplay_system_prompt.replace(
            "{" + sprite_type + "}", color_name.upper()
        )

    # Reset harness state for this game (clears action history from any prior game)
    agent.harness.reset()

    for trial_idx, play_info in enumerate(game_plays):
        run = play_info["run"]
        play_idx = play_info["play_idx"]
        level_id = play_info["level_id"]
        play_id = f"{subject}_{run}_{play_idx}"

        play_doc, raw_zstates = loader.load_play(subject, run, play_idx)
        if not raw_zstates:
            raise ValueError(
                f"Empty states for {play_id} (run={run}, play_idx={play_idx})"
            )

        # Fix split-brain timing AND apply the play-level outcome
        # override in one pass.  realign_zstate_positions:
        #   (a) patches sprite positions / ended / win / score from the
        #       next frame so every post-action field describes the same
        #       tick (Tomov's consecutive-frame lag), and
        #   (b) overrides the terminal frame's `win` with `play_doc.win`
        #       so TIMEOUT plays (~42% of the dataset) are not
        #       mis-classified as LOSS.  See
        #       src/llm_eval/human_replay/zstate_adapter.py and
        #       VGFMRI_DB_README.md:174-176.
        zstates = realign_zstate_positions(raw_zstates, play_doc.get("win"))

        final_win_value = zstates[-1].get("win")
        trial_won = final_win_value is True or final_win_value == 1

        # Count expected steps before running
        expected_steps = sum(
            1
            for i, z in enumerate(zstates)
            if agent._should_process_frame(
                i, z, replay_cfg.action_frames_only, replay_cfg.stride
            )
        )

        print(
            f"    Level {level_id} | "
            f"attempt {level_attempts[level_id]} for this level | "
            f"trial {trial_idx + 1}/{len(game_plays)} overall | "
            f"{expected_steps} moves",
            flush=True,
        )

        play_level = level_id
        play_attempt = level_attempts[level_id]

        records = agent.run_replay(
            zstates,
            level=level_id,
            attempt=play_attempt,
            action_frames_only=replay_cfg.action_frames_only,
            stride=replay_cfg.stride,
        )
        level_attempts[level_id] += 1

        # Convert zstates to viewer format + compute per-frame logs.
        # Under the clean causal model, viewer_state[fi] represents what
        # the user saw at frame fi.  The action_log attached to
        # viewer_state[fi] describes the transition OUT of fi
        # (`states[fi] -> states[fi+1]`) caused by the key captured in the
        # keystate of zstate[fi+1].  It contains only real VGDL engine
        # events from `effectListByColor` plus score/win/lose markers --
        # the same format the LLM sees in its prompt history.
        state_offset = len(all_viewer_states)
        frame_offset = total_frames
        frame_adapter = ZstateAdapter(block_size=block_size)
        frame_adapter.register_objects(zstates[0])
        viewer_states = [convert_zstate_to_viewer(z, block_size) for z in zstates]
        # Stamp level/attempt onto every viewer state so the JS viewer
        # can detect play boundaries even for all-idle plays (no
        # keypresses), where the step-record-driven separator would
        # otherwise never fire.
        for vs in viewer_states:
            vs["level"] = play_level
            vs["attempt"] = play_attempt
        for fi in range(len(zstates) - 1):
            # Skip transitions that START from an already-terminal frame.
            # `realign_zstate_positions` copies `ended`/`win` backward from
            # frame i+1 into frame i, so BOTH the winning frame and the
            # post-terminal snapshot report `ended=True`.  Emitting a log
            # for the post-terminal transition would re-tack a sticky
            # `[WIN]`/`[LOSE]` marker onto what is really a no-op tick.
            if zstates[fi].get("ended"):
                continue
            frame_adapted = frame_adapter.adapt_frame(zstates[fi], zstates[fi + 1])
            frame_log = agent.event_logger.log_action(
                step_num=fi,
                action=frame_adapted["action_idx"],
                prev_obs=frame_adapted["prev_observation"],
                curr_obs=frame_adapted["observation"],
                events=frame_adapted["events"],
                reward=frame_adapted["reward"],
                won=frame_adapted["won"],
                lose=frame_adapted["lose"],
                prev_score=frame_adapted["prev_score"],
                curr_score=frame_adapted["score"],
                level=play_level,
                attempt=play_attempt,
                timeout=frame_adapted["timeout"],
            )
            viewer_states[fi]["action_log"] = frame_log
        all_viewer_states.extend(viewer_states)
        total_frames += len(zstates)

        trial_steps = 0
        for record in records:
            total_prompts += 1
            trial_steps += 1
            cumulative_score += int(record["reward"])

            # Track interactions
            events = record["events"]
            prev_obs = record["prev_obs"]
            if events and prev_obs is not None:
                before = len(interaction_tracker.all_instances)
                interaction_tracker.record_events(events, prev_obs, total_prompts)
                # Log new interactions to table
                if interaction_table is not None:
                    for inst_step, interaction, (
                        oid1,
                        oid2,
                    ) in interaction_tracker.all_instances[before:]:
                        interaction_table.add_data(
                            inst_step,
                            level_id,
                            trial_idx,
                            interaction[0],
                            interaction[1],
                            interaction[2],
                            oid1,
                            oid2,
                            inst_step == interaction_tracker.discovery_order[-1][0]
                            if interaction_tracker.discovery_order
                            else False,
                        )

            # Parse generation response for rationale
            gen_resp = record["generation_response"]
            rationale = ""
            if gen_resp and isinstance(gen_resp, str) and gen_resp.strip():
                parsed = _response_parser.parse_replay(gen_resp)
                rationale = parsed.get("rationale", "")

            # Common fields shared between gameplay and imputation steps
            base_step = {
                "step": record["step_num"],
                "frame": frame_offset + record["frame_idx"],
                "level": level_id,
                "attempt": play_attempt,
                "action": record["action_taken"].lower(),
                "action_log": record["action_log"],
                "reward": record["reward"],
                "won": record["won"],
                "lose": record["lose"],
                "timeout": record["timeout"],
                "state_index": state_offset + record["frame_idx"],
                # Wall-clock Unix timestamp of the observation the participant
                # saw before this action (sourced from zstate["ts"] by
                # replay_agent.py).  Flows through _load_replay_gz into the
                # prompt record so extract_features.py can tag extracted
                # features with the original fMRI experiment's real time.
                "realworld_ts": record["realworld_ts"],
                "response": {
                    "rationale": rationale,
                    "action": record["action_taken"].lower(),
                },
                # Replay-specific metadata
                "frame_idx": record["frame_idx"],
                "play_id": play_id,
                "trial_idx": trial_idx,
                "play_idx": play_idx,
                "run": run,
            }

            # Gameplay step: extraction messages, hidden reasoning erased
            # (the gameplay system prompt never asked for CoT, so including
            # it would be unfaithful to what the prompt requested)
            gameplay_step = {
                **base_step,
                "formatted_obs": record["formatted_obs"],
                "raw_response": record["generation_response"],
                "hidden_reasoning": None,
            }
            all_steps.append(gameplay_step)

            # Imputation step: original imputation messages with hidden
            # reasoning preserved (for debugging the generation phase)
            if record.get("imputation_messages") is not None:
                imputation_step = {
                    **base_step,
                    "formatted_obs": record["formatted_obs"],
                    "raw_response": record["generation_response"],
                    "hidden_reasoning": (
                        record["generation_context"].get("reasoning")
                        if record.get("generation_context")
                        else None
                    ),
                }
                all_imputation_steps.append(imputation_step)

            # Per-step wandb logging
            if _wandb:
                wandb_log = {
                    "step_num": total_prompts,
                    "frame_num": frame_offset + record["frame_idx"],
                    "level": level_id,
                    "attempt": play_attempt,
                    "reward": record["reward"],
                    "cumulative_score": cumulative_score,
                    "cumulative_wins": cumulative_wins,
                    "cumulative_losses": cumulative_losses,
                    "cumulative_success": cumulative_wins - cumulative_losses,
                    "won": record["won"],
                    "lose": record["lose"],
                    "timeout": record["timeout"],
                    "interactions_discovered": interaction_tracker.interactions_discovered,
                    "interaction_coverage": interaction_tracker.coverage,
                }
                ctx = record["generation_context"]
                if ctx is not None:
                    wandb_log["input_tokens"] = ctx["input_tokens"]
                    wandb_log["output_tokens"] = ctx["output_tokens"]
                    wandb_log["reasoning_tokens"] = ctx["reasoning_tokens"]
                    wandb_log["input_chars"] = ctx["input_chars"]
                    wandb_log["output_chars"] = ctx["output_chars"]
                    wandb_log["chars_per_token"] = ctx["chars_per_token"]
                    wandb_log["pct_capacity"] = ctx["pct_capacity"]
                    reasoning = ctx.get("reasoning")
                    wandb_log["reasoning_chars"] = len(reasoning) if reasoning else 0
                    if ctx["retries"] > 0:
                        wandb_log["provider_retries"] = ctx["retries"]
                        wandb_log["provider_retry_wait_s"] = ctx["retry_total_wait"]
                wandb_log["num_conversation_messages"] = len(record["messages"])
                wandb.log(wandb_log)

        # Save after each trial
        _snap = _build_output_snapshot(
            game_name,
            subject,
            agent,
            harness_cfg,
            harness_mode,
            replay_cfg,
            llm_config,
            game_plays,
            all_steps,
            all_viewer_states,
            total_prompts,
            total_frames,
            color_mapping,
            game_description,
            source=gameplay_tag,
            system_prompt=agent._gameplay_system_prompt,
        )
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        save_replay(_snap, output_file)
        print(f"    Saved {output_file}")

        # Incremental S3 sync
        if s3_sync is not None:
            _s3c, _s3b, _s3k = s3_sync
            _s3c.upload_file(str(output_file), _s3b, _s3k)

        # Save narration file alongside (persistent modes only)
        if narration_file is not None and all_imputation_steps:
            _nar_snap = _build_output_snapshot(
                game_name,
                subject,
                agent,
                harness_cfg,
                harness_mode,
                replay_cfg,
                llm_config,
                game_plays,
                all_imputation_steps,
                all_viewer_states,
                total_prompts,
                total_frames,
                color_mapping,
                game_description,
                source="narration",
                system_prompt=agent.harness.system_prompt,
            )
            save_replay(_nar_snap, narration_file)
            print(f"    Saved {narration_file}")

        if _wandb:
            wandb.save(str(output_file), policy="live")
            if narration_file is not None and all_imputation_steps:
                wandb.save(str(narration_file), policy="live")

        # Per-trial metrics
        if trial_won:
            cumulative_wins += 1
        else:
            cumulative_losses += 1

        if _wandb:
            wandb.log(
                {
                    "trial_completed": trial_idx,
                    "trial_level": level_id,
                    "trial_steps": trial_steps,
                    "trial_outcome": "won" if trial_won else "lost",
                }
            )

        # Progress with ETA
        elapsed = time.monotonic() - _game_start_time
        done = trial_idx + 1
        total_trials = len(game_plays)
        eta = elapsed / done * (total_trials - done) if done > 0 else 0
        print(
            f"    Trial {done}/{total_trials} | level {level_id} | "
            f"{total_prompts} steps | ETA {eta:.0f}s"
        )

    # Final save
    output = _build_output_snapshot(
        game_name,
        subject,
        agent,
        harness_cfg,
        harness_mode,
        replay_cfg,
        llm_config,
        game_plays,
        all_steps,
        all_viewer_states,
        total_prompts,
        total_frames,
        color_mapping,
        game_description,
        source=gameplay_tag,
        system_prompt=agent._gameplay_system_prompt,
        completed=True,
    )
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    save_replay(output, output_file)
    print(f"    Saved {output_file} (completed)")

    # Final S3 sync
    if s3_sync is not None:
        _s3c, _s3b, _s3k = s3_sync
        _s3c.upload_file(str(output_file), _s3b, _s3k)

    # Final narration save (persistent modes only)
    if narration_file is not None and all_imputation_steps:
        nar_output = _build_output_snapshot(
            game_name,
            subject,
            agent,
            harness_cfg,
            harness_mode,
            replay_cfg,
            llm_config,
            game_plays,
            all_imputation_steps,
            all_viewer_states,
            total_prompts,
            total_frames,
            color_mapping,
            game_description,
            source="narration",
            system_prompt=agent.harness.system_prompt,
            completed=True,
        )
        save_replay(nar_output, narration_file)
        print(f"    Saved {narration_file} (completed)")

    # Final wandb summary for this game
    if _wandb:
        wandb.save(str(output_file), policy="end")
        if narration_file is not None and all_imputation_steps:
            wandb.save(str(narration_file), policy="end")
        wandb.log(
            {
                "final_outcome": "completed",
                "total_steps": total_prompts,
                "final_cumulative_wins": cumulative_wins,
                "final_cumulative_losses": cumulative_losses,
                "final_cumulative_success": cumulative_wins - cumulative_losses,
                "final_interactions_discovered": interaction_tracker.interactions_discovered,
                "final_interaction_coverage": interaction_tracker.coverage,
            }
        )
        if interaction_table is not None:
            wandb.log({"interaction_sequence": interaction_table})

    return {
        "game_name": game_name,
        "total_prompts": total_prompts,
        "num_trials": len(game_plays),
        "output_path": str(output_file),
        "skipped": False,
    }


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    schema = OmegaConf.structured(Config)
    merged = OmegaConf.merge(schema, cfg)
    typed_cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]
    validate_config(typed_cfg)

    rcfg = typed_cfg.replay
    hcfg = typed_cfg.harness
    lcfg = typed_cfg.llm
    harness_mode = _harness_mode_str(hcfg)

    # Print config summary
    print(f"\n{'=' * 60}")
    print("Human Replay Prompt Generation")
    print(f"{'=' * 60}")
    print(f"Subject: {rcfg.subject or '(all)'}")
    print(f"Game filter: {rcfg.game_filter or '(all)'}")
    print(f"Rationale mode: {harness_mode}")
    print(f"Suggestions: {hcfg.suggestion_level}")
    if _needs_llm(hcfg):
        print(f"LLM model: {lcfg.model or '(none -- dry run)'}")
    print(f"Data dir: {rcfg.data_dir}")
    print(f"Output dir: {typed_cfg.logging.output_dir}")
    if rcfg.action_frames_only:
        print("Sampling: action frames only")
    elif rcfg.stride > 1:
        print(f"Sampling: every {rcfg.stride} frames")
    print(f"{'=' * 60}\n")

    if _needs_llm(hcfg) and not lcfg.model:
        raise ValueError(
            "rationale_mode='copied-reasoning' requires llm.model to be set "
            "(hidden reasoning traces come from a real LLM). Either set "
            "llm.model=<openrouter-model-id> or switch to "
            "harness.rationale_mode=action-only."
        )

    # Initialize wandb if configured
    wandb_run = None
    if typed_cfg.logging.wandb_project:
        wandb_run = wandb.init(
            project=typed_cfg.logging.wandb_project,
            config={
                "pipeline": "human_replay",
                "subject": rcfg.subject,
                "game_filter": rcfg.game_filter,
                "harness_mode": harness_mode,
                "rationale_mode": hcfg.rationale_mode,
                "suggestion_level": hcfg.suggestion_level,
                "llm_model": lcfg.model,
                "action_frames_only": rcfg.action_frames_only,
                "stride": rcfg.stride,
            },
        )

    # Experiment DB tracking
    exp_db_client = None
    _exp_db_worker = ""
    if typed_cfg.exp_db.enabled:
        import socket

        from src.llm_eval.neurips_exp_db import ExpDB

        if not typed_cfg.logging.wandb_project:
            raise ValueError(
                "exp_db.enabled=true requires logging.wandb_project to be set "
                "(experiment tracking needs wandb run URLs)"
            )
        exp_db_client = ExpDB(typed_cfg.exp_db.region)
        _exp_db_worker = socket.gethostname()
        print(f"ExpDB: tracking enabled (region={typed_cfg.exp_db.region})")

    loader = HumanPlayLoader(rcfg.data_dir)
    output_dir = Path(typed_cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args_hash = hashlib.sha256(" ".join(sys.argv).encode()).hexdigest()[:8]

    # Determine subjects
    if rcfg.subject:
        subjects = [rcfg.subject]
    else:
        subjects = loader.list_subjects()

    grand_total = 0
    for subject in subjects:
        print(f"\n--- {subject} ---")
        all_games = get_all_games_for_subject(loader, subject)

        # Filter games
        if rcfg.game_filter:
            filtered = {k: v for k, v in all_games.items() if rcfg.game_filter in k}
            if not filtered:
                raise ValueError(
                    f"No games matching filter '{rcfg.game_filter}' for {subject}. "
                    f"Available: {list(all_games.keys())}"
                )
            all_games = filtered

        for game_name, game_plays in sorted(all_games.items()):
            print(f"  {game_name}: {len(game_plays)} plays")

            # Claim exp_db slot for this game
            # Normalize game name: behavioral data uses 'vgfmri3_bait' but
            # the DB was populated with VGDL-style 'bait_vgfmri3'.
            _exp_db_game = _behavioral_to_vgdl_game_name(game_name)
            _exp_db_sid = None
            _s3_sync = None
            _s3_key = None
            if exp_db_client is not None:
                from src.llm_eval.neurips_exp_db import slot_id_replay
                from src.llm_eval.neurips_exp_db.client import TABLE_REPLAY
                from src.llm_eval.neurips_exp_db.slot_ids import NO_MODEL_SENTINEL

                _exp_db_sid = slot_id_replay(
                    rationale_mode=hcfg.rationale_mode,
                    model=lcfg.model if _needs_llm(hcfg) else NO_MODEL_SENTINEL,
                    suggestion_level=hcfg.suggestion_level,
                    subject=subject,
                    game=_exp_db_game,
                )

                # Compute S3 key for replay upload
                gameplay_tag = "imputed" if _needs_llm(hcfg) else "human"
                _s3_key = (
                    f"{typed_cfg.exp_db.replay_s3_prefix}"
                    f"{_exp_db_sid}/{_exp_db_sid}.{gameplay_tag}.replay.json.gz"
                )

                claimed = exp_db_client.claim_slot(
                    TABLE_REPLAY,
                    _exp_db_sid,
                    worker=_exp_db_worker,
                    wandb_run_id=wandb_run.id,
                    wandb_run_url=wandb_run.url,
                    s3_replay_key=_s3_key,
                )
                if not claimed:
                    print(f"    -> skipped (slot not claimable: {_exp_db_sid})")
                    continue

                _s3_sync = (
                    exp_db_client._s3,
                    typed_cfg.exp_db.s3_bucket,
                    _s3_key,
                )

            try:
                result = process_game(
                    game_name=game_name,
                    game_plays=game_plays,
                    subject=subject,
                    harness_cfg=hcfg,
                    replay_cfg=rcfg,
                    llm_config=lcfg,
                    output_dir=output_dir,
                    args_hash=args_hash,
                    loader=loader,
                    wandb_run=wandb_run,
                    seed=typed_cfg.seed,
                    s3_sync=_s3_sync,
                )
            except BaseException as e:
                if exp_db_client is not None and _exp_db_sid is not None:
                    exp_db_client.fail_slot(TABLE_REPLAY, _exp_db_sid, error_msg=str(e))
                raise
            else:
                if exp_db_client is not None and _exp_db_sid is not None:
                    exp_db_client.complete_slot(
                        TABLE_REPLAY, _exp_db_sid, s3_replay_key=_s3_key
                    )
            grand_total += result["total_prompts"]
            status = (
                "skipped"
                if result.get("skipped")
                else f"{result['total_prompts']} prompts"
            )
            print(f"    -> {status}")

    print(f"\n{'=' * 60}")
    print(f"Done. Total prompts: {grand_total}")
    print(f"{'=' * 60}")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
