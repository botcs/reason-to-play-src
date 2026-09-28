"""Read measured human behaviour from self-contained participant/game JSON.

Replay files retain trajectories, conversations, play identities and scanner
clocks together. Typed binary values, datetimes and mapping keys use explicit
JSON tags decoded with the standard library.
"""

from __future__ import annotations

import base64
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import re

SCHEMA_VERSION = 1
REGRESSORS_SCHEMA = "reason-to-play/empa-regressors"
_TAG = "$rtp"
_AVATAR_LINE = re.compile(r"^\s*(\w+)\s*>\s*(\S*Avatar\S*)", re.M)
GAMES = {"bait", "chase", "helper", "lemmings", "zelda", "plaqueattack", "avoidgeorge"}


class BinaryValue(bytes):
    """Bytes carrying the original binary subtype."""

    def __new__(cls, value: bytes, subtype: int = 0):
        obj = super().__new__(cls, value)
        obj.subtype = subtype
        return obj


def encode_value(value):
    """Convert source values to unambiguous JSON-compatible values."""
    if isinstance(value, bytes):
        return {
            _TAG: "binary",
            "base64": base64.b64encode(value).decode("ascii"),
            "subtype": getattr(value, "subtype", 0),
        }
    if isinstance(value, datetime):
        return {_TAG: "datetime", "iso": value.isoformat()}
    if isinstance(value, dict):
        if _TAG in value or any(not isinstance(key, str) for key in value):
            return {
                _TAG: "mapping",
                "items": [[encode_value(k), encode_value(v)] for k, v in value.items()],
            }
        return {key: encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {_TAG: "float", "value": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported source value type: {type(value).__name__}")


def decode_value(value):
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get(_TAG)
    if kind == "binary":
        return BinaryValue(
            base64.b64decode(value["base64"], validate=True), value["subtype"]
        )
    if kind == "datetime":
        return datetime.fromisoformat(value["iso"])
    if kind == "mapping":
        return {decode_value(k): decode_value(v) for k, v in value["items"]}
    if kind == "float":
        if value["value"] not in {"nan", "inf", "-inf"}:
            raise ValueError("Invalid tagged float")
        return float(value["value"])
    if kind is not None:
        raise ValueError(f"Unknown canonical value type: {kind}")
    return {key: decode_value(item) for key, item in value.items()}


def game_identity(name: str) -> tuple[str, str]:
    match = re.fullmatch(r"(vgfmri[34])_(.+)|(.+)_(vgfmri[34])", name, re.I)
    if match is None:
        raise ValueError(f"Unrecognized game variant: {name!r}")
    cohort, game = (match[1], match[2]) if match[1] else (match[4], match[3])
    return game.lower(), cohort.lower()


def canonical_game_id(name: str) -> str:
    """Use the game identifiers shared by released human and agent data."""
    game, cohort = game_identity(name)
    if game not in GAMES:
        raise ValueError(f"Unrecognized game: {name!r}")
    spelling = {"avoidgeorge": "avoidGeorge", "plaqueattack": "plaqueAttack"}
    return f"{spelling.get(game, game)}_{cohort}"


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


def _read_envelope(path: Path, schema: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("schema") != schema or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported canonical schema in {path}")
    return decode_value(record)


def _subject_name(subject: str | int) -> str:
    try:
        number = int(str(subject).removeprefix("sub-"))
    except ValueError as error:
        raise ValueError(f"Invalid subject: {subject}") from error
    if number < 1:
        raise ValueError(f"Invalid subject: {subject}")
    return f"sub-{number:02d}"


def iter_plays(
    root: Path | str, subject=None, run: int | None = None, *, condition="elaborate"
):
    """Yield measured plays from one prompt condition in original run order."""
    if run is not None and (not isinstance(run, int) or run < 0):
        raise ValueError(f"Invalid run: {run}")
    from .replay_behavior import iter_replay_plays

    yield from iter_replay_plays(behavior_root(root), subject, run, condition=condition)


def load_runs(
    root: Path | str, *, condition="elaborate"
) -> dict[tuple[int, int], dict]:
    """Read scanner identities and clocks embedded in the human recordings."""
    from .replay_behavior import replay_runs

    return replay_runs(behavior_root(root), condition=condition)


def iter_regressors(root_or_file: Path | str):
    """Read EMPA theory records from a dataset root or an explicit JSON file."""
    path = Path(root_or_file)
    if path.is_dir():
        path = path / "features" / "theory" / "empa" / "source-regressors.json.gz"
    record = _read_envelope(path, REGRESSORS_SCHEMA)
    yield from record["regressors"]
