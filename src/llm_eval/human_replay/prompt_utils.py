# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Utility functions for loading prompt files (.replay.json.gz or legacy .jsonl)."""

import json
from pathlib import Path

from src.llm_eval.shared.replay_codec import load_replay


def load_prompts(prompts_file: Path) -> tuple[list[dict], dict]:
    """Load prompts from a .replay.json.gz or legacy .jsonl file.

    For .replay.json.gz: extracts steps from the unified format and maps
    fields to the prompt record schema expected by extract_features.py.

    For .jsonl (legacy): reads line-delimited JSON with optional .meta.json.

    Returns:
        Tuple of (list of prompt records, metadata dict).
        Each prompt record has at least: messages, game_name, level_id,
        step_num, action_taken, trial_idx, play_idx.
    """
    path = Path(prompts_file)

    if path.name.endswith(".replay.json.gz"):
        return _load_replay_gz(path)
    return _load_jsonl(path)


def reconstruct_messages(
    system_prompt: str,
    steps: list[dict],
    up_to_index: int,
    *,
    rationale_mode: str,
) -> list[dict]:
    """Reconstruct the multi-turn LLM message list at a given step.

    Per-step messages are not stored inline (that would duplicate the system
    prompt and be O(n^2) in file size).  Instead the conversation at step N
    is rebuilt from system_prompt + the sequence of per-step
    (formatted_obs, raw_response/action) fields.

    Args:
        system_prompt: The system prompt for the run.
        steps: The full list of step dicts from the replay file.
        up_to_index: Reconstruct conversation up to and including this step.
        rationale_mode: Rationale mode used during generation -- controls
            whether the assistant message contains just the action or
            a full ``{"rationale", "action"}`` pair.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    if rationale_mode not in ("action-only", "prompted-rationale", "copied-reasoning"):
        raise ValueError(
            f"reconstruct_messages: invalid rationale_mode={rationale_mode!r}. "
            f"Must be one of 'action-only', 'prompted-rationale', 'copied-reasoning'."
        )

    messages = [{"role": "system", "content": system_prompt}]
    for i in range(up_to_index + 1):
        step = steps[i]
        # Synthetic markers (e.g. _level_advance) are bookkeeping rows
        # with no formatted_obs -- skip, they do not correspond to a
        # user/assistant turn.
        if step["action"].startswith("_"):
            continue
        user_content = _mt_user_content(steps, i)
        messages.append({"role": "user", "content": user_content})
        response = step["response"]
        if rationale_mode == "action-only":
            assistant_content = json.dumps({"action": response["action"]})
        else:
            # prompted-rationale or copied-reasoning: full response with rationale
            assistant_content = json.dumps(response)
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


# ---------------------------------------------------------------------------
# Helpers: per-step user message reconstruction
# ---------------------------------------------------------------------------


def _mt_user_content(steps: list[dict], idx: int) -> str:
    """Return the multi-turn user message for *steps[idx]*.

    Emits ``Harness.build_messages``'s format: optional TRIAL ENDED /
    NEW TRIAL markers when this step starts a new trial, followed by
    ``# Step N ...`` header + ``formatted_obs``.  Legacy replay files
    with ``user_prompt`` stored inline are returned verbatim.
    """
    step = steps[idx]
    if "user_prompt" in step:
        return step["user_prompt"]

    step_num = step["step"]
    level = step["level"]
    attempt = step["attempt"]

    parts: list[str] = []
    _append_trial_transition(parts, steps, idx, level, attempt)
    parts.append(f"# Step {step_num} (Level {level}, Attempt {attempt})")
    parts.append("")
    parts.append(step["formatted_obs"])
    return "\n".join(parts)


def _append_trial_transition(
    parts: list[str],
    steps: list[dict],
    idx: int,
    level: int,
    attempt: int,
) -> None:
    """Emit TRIAL ENDED / NEW TRIAL block if step idx starts a new trial.

    Mirrors ``Harness.build_messages``'s transition block.  The boundary
    is detected against the most recent *real* predecessor because
    synthetic ``_level_advance`` rows are stamped with the NEW trial's
    (level, attempt) and would mask the transition.
    """
    prev_real = None
    for j in range(idx - 1, -1, -1):
        s = steps[j]
        if not s["action"].startswith("_"):
            prev_real = s
            prev_real_idx = j
            break

    if prev_real is None:
        return
    if prev_real["level"] == level and prev_real["attempt"] == attempt:
        return

    if prev_real["won"]:
        prev_outcome = "won"
    elif prev_real["lose"]:
        prev_outcome = "died"
    elif prev_real["timeout"]:
        prev_outcome = "timeout"
    else:
        prev_outcome = ""

    prev_score = 0
    for j in range(prev_real_idx, -1, -1):
        s = steps[j]
        if s["action"].startswith("_"):
            continue
        if s["level"] == prev_real["level"] and s["attempt"] == prev_real["attempt"]:
            prev_score += s["reward"]
        else:
            break

    if prev_outcome:
        parts.append(
            f"--- TRIAL ENDED outcome: {prev_outcome}, score: {prev_score} ---"
        )
        parts.append("")
    parts.append(f"--- NEW TRIAL (Level {level}, Attempt {attempt}) ---")
    parts.append("")


_GENERATIVE_SOURCE = "generative"
_REPLAY_SOURCES = ("human", "imputed", "narration")
_VALID_SOURCES = (_GENERATIVE_SOURCE,) + _REPLAY_SOURCES

# Fields that are part of the replay-only schema -- they only make sense for
# files produced by the human-replay pipeline.  Generative files leave them as
# None so downstream code that uses them on the wrong file type fails with a
# TypeError rather than silently processing a zero/empty-string placeholder.
_REPLAY_ONLY_STEP_FIELDS = (
    "frame_idx",
    "trial_idx",
    "play_idx",
    "play_id",
    "run",
)
_REPLAY_ONLY_META_FIELDS = ("subject", "num_trials", "pipeline", "completed")


def _require(d: dict, key: str, where: str):
    """Return ``d[key]`` or raise a clear error if the key is missing."""
    if key not in d:
        raise KeyError(
            f"{where} is missing required field {key!r}. Got keys: {sorted(d.keys())}"
        )
    return d[key]


def _load_replay_gz(path: Path) -> tuple[list[dict], dict]:
    """Load prompts from the unified .replay.json.gz format.

    Enforces the file's data contract strictly: missing required top-level,
    meta, or per-step fields raise ``KeyError`` rather than silently
    defaulting.  Fields that only apply to replay-sourced files (frame_idx,
    trial_idx, play_idx, play_id, run, subject, num_trials, pipeline,
    completed) are returned as ``None`` for files with ``source='generative'``
    so downstream misuse surfaces as a ``TypeError`` at the boundary.
    """
    data = load_replay(path)

    where = f"Replay file {path}"
    game_name = _require(data, "game", where)
    source = _require(data, "source", where)
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"{where} has unknown source={source!r}. Must be one of {_VALID_SOURCES}."
        )
    meta = _require(data, "meta", where)
    system_prompt = _require(data, "system_prompt", where)
    all_steps = _require(data, "steps", where)

    meta_where = f"{where} meta"
    rationale_mode = _require(meta, "rationale_mode", meta_where)
    suggestion_level = _require(meta, "suggestion_level", meta_where)

    is_replay_source = source in _REPLAY_SOURCES

    # Map steps to prompt records
    prompts = []
    for step_idx, step in enumerate(all_steps):
        step_where = f"{where} step[{step_idx}]"
        action_taken = _require(step, "action", step_where)
        is_synthetic = action_taken.startswith("_")

        # Messages are never stored inline -- always reconstruct from
        # per-step formatted_obs + the per-step response.
        step_messages = reconstruct_messages(
            system_prompt,
            all_steps,
            step_idx,
            rationale_mode=rationale_mode,
        )

        record = {
            "game_name": game_name,
            "level_id": _require(step, "level", step_where),
            "step_num": _require(step, "step", step_where),
            "action_taken": action_taken,
            "messages": step_messages,
            "rationale_mode": rationale_mode,
            "suggestion_level": suggestion_level,
        }

        if is_synthetic:
            # Synthetic marker steps (e.g. _level_advance) are bookkeeping
            # rows that downstream consumers skip via action_taken.startswith(_).
            # Fill non-core fields with None so any leak through the filter
            # fails loudly rather than silently.
            record["win"] = None
            record["score"] = None
            record["realworld_ts"] = None
            for f in _REPLAY_ONLY_STEP_FIELDS:
                record[f] = None
        else:
            record["win"] = _require(step, "won", step_where)
            record["score"] = _require(step, "reward", step_where)
            record["realworld_ts"] = _require(step, "realworld_ts", step_where)
            if is_replay_source:
                for f in _REPLAY_ONLY_STEP_FIELDS:
                    record[f] = _require(step, f, step_where)
            else:
                for f in _REPLAY_ONLY_STEP_FIELDS:
                    record[f] = None

        prompts.append(record)

    # The harness prompt_name is always written at the top level by both
    # writers; if it ever isn't, the file is malformed and we fail loud
    # rather than fabricating one.
    prompt_name = _require(data, "prompt_name", where)

    metadata = {
        "game_name": game_name,
        "source": source,
        "rationale_mode": rationale_mode,
        "suggestion_level": suggestion_level,
        # prompt_variant is the same string as the harness prompt_name
        # (rationale_mode + suggestion_level) -- consumers (extract_features)
        # use it as a directory segment.
        "prompt_variant": prompt_name,
        "total_prompts": len(prompts),
    }
    if is_replay_source:
        for f in _REPLAY_ONLY_META_FIELDS:
            metadata[f] = _require(meta, f, meta_where)
    else:
        for f in _REPLAY_ONLY_META_FIELDS:
            metadata[f] = None

    return prompts, metadata


def _load_jsonl(path: Path) -> tuple[list[dict], dict]:
    """Load prompts from legacy .jsonl format."""
    prompts = []
    with open(path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    metadata_file = path.with_suffix(".meta.json")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return prompts, metadata
