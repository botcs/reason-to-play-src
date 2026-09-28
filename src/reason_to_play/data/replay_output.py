"""Retain measured human trajectories when generating a new conversation."""

from collections import defaultdict
import math

from .replay_behavior import SCHEMA, SCHEMA_VERSION, recording_path

# These annotate the displayed conversation, not the recorded measurements.
DISPLAY_FIELDS = ("step", "action_log", "level", "attempt", "won", "lose", "timeout")
CONVERSATION_FIELDS = (
    "system_prompt",
    "prompt_name",
    "suggestion_level",
    "model",
    "start_level",
    "started_at",
    "finished_at",
    "outcome",
    "total_steps",
)


def merge_human_prompt_output(source: dict, generated: dict) -> dict:
    """Combine a complete new action-only conversation with its measured input.

    The source must be an expanded, self-contained participant/game replay. Only
    a complete play inventory is supported: prompt sampling may change, but
    measurements cannot be dropped or reordered. Neither input is mutated.
    """
    if (source.get("schema"), source.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        raise ValueError("Expected a self-contained human replay source")
    if source.get("delta_encoded") or generated.get("delta_encoded"):
        raise ValueError("Expand replay states before merging conversations")
    if "original_game_description" in source:
        raise ValueError("Human replay must have one translated game description")
    for field in ("subject", "game", "source"):
        if source.get(field) != generated.get(field):
            raise ValueError(f"Generated human replay differs in {field}")
    meta = generated["meta"]
    if any(
        meta.get(field) != generated[field]
        for field in ("subject", "game", "suggestion_level")
    ):
        raise ValueError(
            "Generated conversation metadata disagrees with its identity/condition"
        )
    if (
        source["source"] != "human"
        or meta.get("completed") is not True
        or meta.get("rationale_mode") != "action-only"
    ):
        raise ValueError("Only completed action-only human conversations can be merged")
    plays, inventory = source["plays"], generated["plays"]
    if not plays or len(plays) != len(inventory) or meta["num_trials"] != len(plays):
        raise ValueError(
            "Generated conversation must retain the complete play inventory"
        )
    if (
        source["total_frames"] != len(source["states"])
        or generated["total_frames"] != source["total_frames"]
        or len(generated["states"]) != source["total_frames"]
    ):
        raise ValueError("Generated conversation must retain every original frame")
    if generated["total_steps"] != len(generated["steps"]):
        raise ValueError("Generated conversation step count disagrees")

    by_id = {}
    attempts = defaultdict(int)
    expected_start = 0
    for trial, (play, item) in enumerate(zip(plays, inventory, strict=True)):
        checks = {
            "source_play_id": play["_id"],
            "run": play["run_id"],
            "source_document_index": play["source_document_index"],
            "game_name": play["game_name"],
            "level_id": play["level_id"],
            "win": play["win"],
            "outcome": play["outcome"],
        }
        if any(item.get(key) != value for key, value in checks.items()):
            raise ValueError(
                "Generated conversation changed original play identity/order"
            )
        start, count = play["state_start"], play["state_count"]
        if (
            any(isinstance(n, bool) or not isinstance(n, int) for n in (start, count))
            or start != expected_start
            or count < 1
        ):
            raise ValueError("Invalid original play frame bounds")
        expected_start += count
        if expected_start > len(source["states"]) or play["_id"] in by_id:
            raise ValueError("Repeated play or out-of-range original frames")
        attempt = attempts[play["level_id"]]
        attempts[play["level_id"]] += 1
        by_id[play["_id"]] = (play, trial, attempt)
        for index in range(start, expected_start):
            measured, display = source["states"][index], generated["states"][index]
            if (
                measured["source_play_id"] != play["_id"]
                or measured["time"] != index - start
                or display["time"] != measured["time"]
                or display["level"] != play["level_id"]
                or display["attempt"] != attempt
                or isinstance(measured["realworld_ts"], bool)
                or not isinstance(measured["realworld_ts"], (int, float))
                or not math.isfinite(measured["realworld_ts"])
            ):
                raise ValueError(
                    "Generated display frame differs from its original play/frame"
                )
    if expected_start != len(source["states"]):
        raise ValueError("Original play inventory leaves unassigned frames")

    result = dict(source)
    result["meta"] = {**source["meta"], **meta}
    self_path = recording_path(result)
    steps = []
    seen = set()
    previous_offset = -1
    for step in generated["steps"]:
        if step["action"].startswith("_"):
            raise ValueError("Synthetic human steps require an explicit frame mapping")
        identity = step["source_play_id"]
        if identity not in by_id:
            raise ValueError("Generated step references an unknown original play")
        play, trial, attempt = by_id[identity]
        frame = step["source_frame_index"]
        if (
            isinstance(frame, bool)
            or not isinstance(frame, int)
            or not 0 <= frame < play["state_count"]
            or (identity, frame) in seen
        ):
            raise ValueError("Invalid or repeated original step frame")
        seen.add((identity, frame))
        offset = play["state_start"] + frame
        if offset <= previous_offset:
            raise ValueError("Generated steps reordered original frames")
        previous_offset = offset
        expected = {
            "frame_idx": frame,
            "run": play["run_id"],
            "source_document_index": play["source_document_index"],
            "play_idx": play["source_document_index"],
            "trial_idx": trial,
            "level": play["level_id"],
            "attempt": attempt,
            "realworld_ts": source["states"][offset]["realworld_ts"],
            "frame": offset,
            "state_index": offset,
            "play_id": f"{source['subject']}_{play['run_id']}_{play['source_document_index']}",
        }
        if any(step.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Generated step identity, clock or display index disagrees"
            )
        steps.append({**step, "source_recording": self_path})
    result["steps"] = steps
    result["states"] = []
    for measured, display in zip(source["states"], generated["states"], strict=True):
        frame = {
            key: value for key, value in measured.items() if key not in DISPLAY_FIELDS
        }
        frame.update({key: display[key] for key in DISPLAY_FIELDS if key in display})
        result["states"].append(frame)
    result.update({key: generated[key] for key in CONVERSATION_FIELDS})
    return result
