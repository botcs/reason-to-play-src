"""Behavior readers for self-contained participant/game human replay files."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import gzip
import json
import math
from pathlib import Path

from .replay_codec import expand_delta_states

SCHEMA = "reason-to-play/human-replay"
SCHEMA_VERSION = 1
CONDITIONS = ("elaborate", "minimal", "oracle")
FRAME_FIELDS = (
    "score",
    "ended",
    "win",
    "keystate",
    "keyPressType",
    "effectList",
    "effectListByClass",
    "effectListByColor",
    "kill_list_ID",
    "new_sprites_ID",
    "dt",
    "datetime",
)


def recording_path(record: dict) -> str:
    """Canonical dataset-relative identity; independent of local download location."""
    from .behavior import _subject_name, game_identity

    subject = _subject_name(record["subject"])
    game = record["game"]
    game_identity(game)
    condition = record["meta"]["suggestion_level"]
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown human prompt condition: {condition!r}")
    return f"{subject}/{game}/{condition}.human.replay.json.gz"


def play_recording_path(play: dict) -> str:
    """Require the participant/game JSON path associated with a measured play."""
    from .behavior import _subject_name, game_identity

    path = play.get("_canonical", {}).get("source_recording")
    parts = path.split("/") if isinstance(path, str) else []
    if (
        len(parts) != 3
        or parts[0] != _subject_name(play["subj_id"])
        or parts[2]
        not in {f"{condition}.human.replay.json.gz" for condition in CONDITIONS}
    ):
        raise ValueError("Human play requires a participant/game source_recording path")
    if game_identity(parts[1]) != game_identity(play["game_name"]):
        raise ValueError("Human source_recording path names a different game")
    return path


def replay_paths(root, subject=None, condition="elaborate") -> list[Path]:
    """Select one prompt condition to avoid counting a trajectory three times."""
    from .behavior import _subject_name, behavior_root

    root = behavior_root(root)
    if root.is_file():
        return [root]
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown human prompt condition: {condition!r}")
    sub = "sub-*" if subject is None else _subject_name(subject)
    return sorted(root.glob(f"{sub}/*/{condition}.human.replay.json.gz"))


def read_record(path, *, expand=True) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    if (record.get("schema"), record.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        raise ValueError(f"Expected a self-contained human replay in {path}")
    if record.get("source") != "human":
        raise ValueError(f"Expected recorded human behaviour in {path}")
    recording_path(record)
    if not isinstance(record.get("plays"), list) or not record["plays"]:
        raise ValueError(f"Missing original play metadata in {path}")
    if (
        not isinstance(record.get("game_description"), str)
        or not record["game_description"]
    ):
        raise ValueError(f"Missing translated game definition in {path}")
    if "original_game_description" in record:
        raise ValueError(
            f"A replay must contain only its translated game definition: {path}"
        )
    if expand:
        expand_delta_states(record)
    return record


def _pixel(value, block_size):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Invalid drawing coordinate: {value!r}")
    scaled = value * block_size
    pixel = round(scaled)
    if not math.isclose(scaled, pixel, rel_tol=0, abs_tol=1e-7):
        raise ValueError(
            "Drawing coordinate does not represent a recorded integer pixel"
        )
    return pixel


def restore_frame(frame, block_size):
    """Return only measured fields needed by analysis and baseline inputs."""
    from .behavior import BinaryValue, decode_value

    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("Original cell size must be a positive integer")
    state = {key: decode_value(frame[key]) for key in FRAME_FIELDS if key in frame}
    state["gt"] = frame["time"]
    state["ts"] = frame["realworld_ts"]
    objects = {}
    for kind, sprites in frame["sprites"].items():
        objects[kind] = {}
        for sprite in sprites:
            source_key = sprite["source_key"]
            if source_key in objects[kind]:
                raise ValueError(f"Repeated original sprite key: {kind}/{source_key}")
            obj = {
                "ID": BinaryValue(
                    bytes.fromhex(sprite["_uuid"]), sprite.get("_uuid_subtype", 0)
                ),
                "x": sprite["x"],
                "y": sprite["y"],
                "color": sprite["color"],
                "rect": {
                    "pos": [
                        _pixel(sprite["col"], block_size),
                        _pixel(sprite["row"], block_size),
                    ],
                    "size": [block_size, block_size],
                },
            }
            if "color_name" in sprite:
                obj["colorName"] = sprite["color_name"]
            if "resources" in sprite:
                obj["resources"] = decode_value(sprite["resources"])
            objects[kind][source_key] = obj
    state["objects"] = objects
    return state


def record_plays(record: dict):
    from .behavior import (
        _subject_name,
        decode_value,
        game_identity,
        human_outcome,
        validate_frame_clocks,
    )

    path = recording_path(record)
    states = record["states"]
    if record["total_frames"] != len(states):
        raise ValueError(f"Frame count differs in {path}")
    previous_end = 0
    seen = set()
    for stored_play in record["plays"]:
        play = decode_value(stored_play)
        start, count = play.pop("state_start"), play.pop("state_count")
        if (
            any(isinstance(n, bool) or not isinstance(n, int) for n in (start, count))
            or count < 1
        ):
            raise ValueError(f"Invalid play frame bounds in {path}")
        if start != previous_end or start + count > len(states):
            raise ValueError(
                f"Overlapping, missing or out-of-range play frames in {path}"
            )
        previous_end = start + count
        identity = play["_id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"Missing or repeated original play ID in {path}")
        seen.add(identity)
        if _subject_name(play["subj_id"]) != _subject_name(record["subject"]):
            raise ValueError(f"Play participant differs from replay in {path}")
        if game_identity(play["game_name"]) != game_identity(record["game"]):
            raise ValueError(f"Replay contains another game in {path}")
        ordinal = play.pop("source_document_index")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError(f"Invalid original run document ordinal in {path}")
        frames = states[start : start + count]
        if any(frame["source_play_id"] != identity for frame in frames):
            raise ValueError(f"Frame/play identity differs in {path}")
        restored = [restore_frame(frame, play["block_size"]) for frame in frames]
        validate_frame_clocks(restored, identity)
        play["game_str"] = record["game_description"]
        play["states"] = restored
        outcome = play.pop("outcome")
        if human_outcome(play, restored) != outcome:
            raise ValueError(f"Original events/outcome disagree in {path}: {identity}")
        play["_canonical"] = {
            "source_document_index": ordinal,
            "source_recording": path,
            "outcome": outcome,
        }
        yield play
    if previous_end != len(states):
        raise ValueError(f"Unassigned frames in {path}")


def iter_replay_plays(root, subject=None, run=None, *, condition="elaborate"):
    """Read each participant chronologically, retaining original run ordinals."""
    from .behavior import _subject_name

    paths = replay_paths(root, subject, condition)
    if not paths:
        raise FileNotFoundError(f"No {condition} human replay files in {root}")
    grouped = defaultdict(list)
    for path in paths:
        grouped[path.parent.parent.name].append(path)
    for group in sorted(grouped):
        plays = []
        for path in grouped[group]:
            record = read_record(path)
            if subject is not None and _subject_name(
                record["subject"]
            ) != _subject_name(subject):
                continue
            for play in record_plays(record):
                if run is None or int(play["run_id"]) == run:
                    plays.append(play)
            del record
        plays.sort(
            key=lambda p: (
                int(p["subj_id"]),
                int(p["run_id"]),
                p["_canonical"]["source_document_index"],
            )
        )
        seen_ids, seen_ordinals = set(), set()
        for play in plays:
            key = (
                int(play["subj_id"]),
                int(play["run_id"]),
                play["_canonical"]["source_document_index"],
            )
            if play["_id"] in seen_ids or key in seen_ordinals:
                raise ValueError(f"Duplicate human observation across files: {key}")
            seen_ids.add(play["_id"])
            seen_ordinals.add(key)
            yield play


def replay_runs(root, *, condition="elaborate"):
    from .behavior import decode_value

    runs = {}
    paths = replay_paths(root, condition=condition)
    if not paths:
        raise FileNotFoundError(f"No {condition} human replay files in {root}")
    for path in paths:
        record = read_record(path, expand=False)
        for play in record["plays"]:
            context = f"{path}, play {play.get('_id')}, run {play.get('run_id')}"
            scanner = decode_value(play.get("scanner"))
            if (
                not isinstance(scanner, dict)
                or not {"subj_id", "run_id", "scan_start_ts"} <= scanner.keys()
            ):
                raise ValueError(f"Missing original scanner metadata in {context}")
            key = int(play["subj_id"]), int(play["run_id"])
            if (int(scanner["subj_id"]), int(scanner["run_id"])) != key:
                raise ValueError(f"Scanner/play identity differs in {context}")
            timestamp = scanner["scan_start_ts"]
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
            ):
                raise ValueError(f"Missing original scanner clock in {context}")
            if "clock_only" in scanner:
                provenance = scanner.get("provenance")
                if (
                    scanner["clock_only"] is not True
                    or not isinstance(provenance, dict)
                    or not provenance.get("method")
                    or not isinstance(scanner.get("scan_start_dt"), datetime)
                ):
                    raise ValueError(
                        f"Invalid derived scanner clock or missing provenance in {context}"
                    )
            if key in runs and runs[key] != scanner:
                raise ValueError(f"Conflicting embedded scanner records: {key}")
            runs[key] = scanner
    return runs
