"""Read self-contained human gameplay, recorded frames and scanner clocks."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import re

from data.game_ids import game_identity
from data.values import BinaryValue, decode_value
from data.replay_codec import expand_delta_states

_AVATAR_LINE = re.compile(r"^\s*(\w+)\s*>\s*(\S*Avatar\S*)", re.M)


def human_outcome(doc: dict, states: list[dict] | None = None) -> str:
    """Classify original play outcome; avatar deaths precede null/incomplete."""
    win = doc["win"]
    if win is not True and win is not False and win is not None:
        raise ValueError(f"Expected three-valued play win, got {win!r}")
    if win is True:
        return "win"
    avatars = {match[1] for match in _AVATAR_LINE.finditer(doc["game_str"])}
    if not avatars:
        raise ValueError("No avatar class in human game_str")
    states = play_states(doc) if states is None else states
    died = any(
        len(event) >= 2 and event[0] == "killSprite" and event[1] in avatars
        for state in states
        for event in state["effectListByClass"]
    )
    if died:
        return "avatar_died"
    return "loss" if win is False else "incomplete"


def play_states(play: dict) -> list[dict]:
    """Return original decoded frames, never lag-corrected viewer snapshots."""
    states = play.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError(f"Play {play.get('_id')} has no canonical states")
    return states


def validate_frame_clocks(states: list[dict], original_id: str) -> None:
    if any(state["gt"] != i for i, state in enumerate(states)):
        raise ValueError(f"Non-contiguous engine clock in play {original_id}")
    if any(
        isinstance(state.get("ts"), bool)
        or not isinstance(state.get("ts"), (int, float))
        or not math.isfinite(state["ts"])
        for state in states
    ):
        raise ValueError(
            f"Missing or nonfinite original timestamps in play {original_id}"
        )


def behavior_root(root: Path | str) -> Path:
    """Accept a human-behavior directory or the containing release directory."""
    root = Path(root)
    nested = root / "behavior" / "human"
    return nested if nested.is_dir() else root


def _subject_name(subject: str | int) -> str:
    try:
        number = int(str(subject).removeprefix("sub-"))
    except ValueError as error:
        raise ValueError(f"Invalid subject: {subject}") from error
    if number < 1:
        raise ValueError(f"Invalid subject: {subject}")
    return f"sub-{number:02d}"


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

    subject = _subject_name(record["subject"])
    game = record["game"]
    game_identity(game)
    condition = record["meta"]["suggestion_level"]
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown human prompt condition: {condition!r}")
    return f"{subject}/{game}/{condition}.human.replay.json.gz"


def play_recording_path(play: dict) -> str:
    """Require the participant/game JSON path associated with a measured play."""

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


def iter_plays(root, subject=None, run=None, *, condition="elaborate"):
    """Read each participant chronologically, retaining original run ordinals."""

    if run is not None and (not isinstance(run, int) or run < 0):
        raise ValueError(f"Invalid run: {run}")
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


def load_runs(root, *, condition="elaborate"):

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


def load_behavioral_data(subject: str, run: int, data_path: str) -> tuple:
    """Load a participant/run from self-contained human JSON files."""
    subj_str = subject if subject.startswith("sub-") else f"sub-{int(subject):02d}"
    plays = list(iter_plays(data_path, subject=subj_str, run=run))
    if not plays:
        raise ValueError(f"No human plays found for {subj_str} run {run}")
    run_doc = load_runs(data_path).get((int(subj_str[4:]), run))
    if run_doc is None:
        raise ValueError(f"Scanner run not found: {subj_str} run {run}")
    return plays, run_doc


class HumanPlayLoader:
    """List recorded plays cheaply and decode one participant/game file at a time."""

    def __init__(self, data_dir: str):
        self.data_dir = behavior_root(data_dir)
        if not replay_paths(self.data_dir):
            raise FileNotFoundError(
                f"No human JSON recordings found in {self.data_dir}"
            )
        self.per_game = True
        self.canonical = True
        self._indexed_subject = None
        self._plays_by_run = {}
        self._play_files = {}
        self._cached_file = None
        self._cached_documents = {}

    def _index_subject(self, subject):
        subject = _subject_name(subject)
        if self._indexed_subject == subject:
            return
        by_run, play_files, seen_ids = defaultdict(list), {}, set()
        for path in replay_paths(self.data_dir, subject):
            record = read_record(path, expand=False)
            if _subject_name(record["subject"]) != subject:
                continue
            source_recording = recording_path(record)
            for stored in record["plays"]:
                ordinal = stored["source_document_index"]
                identity = stored["_id"]
                run = int(stored["run_id"])
                if (
                    type(ordinal) is not int
                    or ordinal < 0
                    or not isinstance(identity, str)
                    or not identity
                    or (run, ordinal) in play_files
                    or identity in seen_ids
                ):
                    raise ValueError(
                        f"Duplicate or invalid human play identity: {path}"
                    )
                if _subject_name(stored["subj_id"]) != subject or game_identity(
                    stored["game_name"]
                ) != game_identity(record["game"]):
                    raise ValueError(f"Human play/recording identity differs: {path}")
                seen_ids.add(identity)
                play_files[run, ordinal] = path, identity
                by_run[run].append(
                    {
                        "play_idx": ordinal,
                        "game_name": stored["game_name"],
                        "level_id": stored["level_id"],
                        "win": decode_value(stored.get("win")),
                        "outcome": stored.get("outcome"),
                        "score": decode_value(stored.get("score")),
                        "source_play_id": identity,
                        "source_recording": source_recording,
                    }
                )
        for plays in by_run.values():
            plays.sort(key=lambda item: item["play_idx"])
        self._indexed_subject = subject
        self._plays_by_run = dict(by_run)
        self._play_files = play_files
        self._cached_file = None
        self._cached_documents = {}

    def list_subjects(self) -> list[str]:
        """List participant identifiers in the available human files."""
        return sorted(
            {
                read_record(path, expand=False)["subject"]
                for path in replay_paths(self.data_dir)
            }
        )

    def list_runs(self, subject: str) -> list[int]:
        """List scanner runs from metadata without expanding recorded frames."""
        self._index_subject(subject)
        return sorted(self._plays_by_run)

    def list_plays(self, subject: str, run: int) -> list[dict]:
        """Return recorded play metadata, ordered by original run document index."""
        self._index_subject(subject)
        return [dict(play) for play in self._plays_by_run.get(run, [])]

    def load_play(
        self, subject: str, run: int, play_idx: int
    ) -> tuple[dict, list[dict]]:
        """Return the exact measured play/frames, caching only its game file."""
        self._index_subject(subject)
        location = self._play_files.get((run, play_idx))
        if location is None:
            raise IndexError(
                f"Original play ordinal {play_idx} not found uniquely in {subject} run {run}"
            )
        path, identity = location
        if self._cached_file != path:
            # Release the preceding game before expanding the next one.
            self._cached_documents = {}
            self._cached_file = None
            record = read_record(path)
            self._cached_documents = {
                play["_id"]: play for play in record_plays(record)
            }
            self._cached_file = path
        document = self._cached_documents[identity]
        return document, play_states(document)

    def get_num_plays(self, subject: str, run: int) -> int:
        """Count recorded play metadata without decoding trajectories."""
        self._index_subject(subject)
        return len(self._plays_by_run.get(run, []))
