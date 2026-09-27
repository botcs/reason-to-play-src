"""Build the figure input CSV from local human BSON and generative replays.

No experiment database is required. Human steps count non-idle keypress frames;
model steps count decisions. Engine frames are stored separately. Select one
complete generative replay per run; resumed fragments are rejected to avoid
silently double-counting or discarding earlier levels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

AVATAR_LINE = re.compile(r"^\s*(\w+)\s*>\s*(\S*Avatar\S*)", re.M)
GAMES = {"bait", "chase", "helper", "lemmings", "zelda", "plaqueattack", "avoidgeorge"}
FIELDS = [
    "agent_type",
    "model",
    "rationale_mode",
    "suggestion_level",
    "instance_id",
    "seed",
    "game",
    "cohort",
    "level",
    "owner",
    "episode_steps",
    "episode_frames",
    "episode_outcomes",
    "step_unit",
    "data_source",
]


def game_identity(name: str) -> tuple[str, str]:
    match = re.fullmatch(r"(vgfmri[34])_(.+)|(.+)_(vgfmri[34])", name, re.I)
    if match is None:
        raise ValueError(f"Unrecognized game variant: {name!r}")
    cohort, game = (match[1], match[2]) if match[1] else (match[4], match[3])
    return game.lower(), cohort.lower()


def human_outcome(doc: dict, states: list[dict]) -> str:
    win = doc["win"]  # Never substitute terminal zstate win (-1 means incomplete).
    if win is not True and win is not False and win is not None:
        raise ValueError(f"Expected three-valued play win, got {win!r}")
    if win is True:
        return "win"
    avatars = {match[1] for match in AVATAR_LINE.finditer(doc["game_str"])}
    if not avatars:
        raise ValueError("No avatar class in human game_str")
    died = any(
        len(event) >= 2 and event[0] == "killSprite" and event[1] in avatars
        for state in states
        for event in state["effectListByClass"]
    )
    if died:
        return "avatar_died"
    return "loss" if win is False else "incomplete"


def human_rows(data_dir: Path) -> list[dict]:
    from src.llm_eval.human_replay.data_loader import (
        decompress_zstates,
        read_bson_file,
    )

    files = sorted((data_dir / "plays").glob("sub-*/run-*.bson"))
    if not files:
        raise FileNotFoundError(f"No plays/sub-*/run-*.bson in {data_dir}")
    grouped: dict[tuple, dict] = {}
    for path in files:
        if path.stem == "run-00":
            continue
        for doc in read_bson_file(path):
            game, cohort = game_identity(doc["game_name"])
            level = int(doc["level_id"])
            if game not in GAMES or not 0 <= level <= 8:
                continue
            states = decompress_zstates(doc["zstates"])
            if not states:
                raise ValueError(f"Empty zstates in {path}")
            if any(state["gt"] != i for i, state in enumerate(states)):
                raise ValueError(f"Non-contiguous engine clock in {path}")
            key = path.parent.name, game, cohort, level
            row = grouped.setdefault(
                key,
                {
                    "agent_type": "human",
                    "model": "Human",
                    "rationale_mode": "none",
                    "suggestion_level": "none",
                    "instance_id": path.parent.name,
                    "seed": "none",
                    "game": game,
                    "cohort": cohort,
                    "level": level,
                    "owner": "human",
                    "episode_steps": [],
                    "episode_frames": [],
                    "episode_outcomes": [],
                    "step_unit": "non_idle_keypress_frame",
                    "data_source": [],
                },
            )
            row["episode_steps"].append(
                sum(s.get("keyPressType") is not None for s in states)
            )
            row["episode_frames"].append(len(states) - 1)
            row["episode_outcomes"].append(human_outcome(doc, states))
            source = path.relative_to(data_dir).as_posix()
            if source not in row["data_source"]:
                row["data_source"].append(source)
    return list(grouped.values())


def replay_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        replay = json.load(stream)
    if replay.get("source") != "generative":
        raise ValueError(f"Expected generative replay: {path}")
    meta = replay["meta"]
    if "resumed_at_level" in meta or "resumed_from_steps" in meta:
        raise ValueError(f"Merge resumed replay fragments before exporting: {path}")
    game, cohort = game_identity(replay["game"])
    if game not in GAMES:
        return []
    # Identity uses the recorded run, never the filename or the model alone.
    identity = [
        meta["model"],
        replay["game"],
        meta["seed"],
        replay["started_at"],
        meta["rationale_mode"],
        meta["suggestion_level"],
        meta.get("wandb_run_url"),
    ]
    instance = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for step in replay["steps"]:
        if not step["action"].startswith("_") and 0 <= int(step["level"]) <= 8:
            groups[int(step["level"]), int(step["attempt"])].append(step)
    frames: dict[tuple, int] = {}
    for state in replay.get("states", []):
        key = int(state["level"]), int(state["attempt"])
        frames[key] = max(frames.get(key, 0), state["time"])
    rows: dict[int, dict] = {}
    for (level, attempt), steps in groups.items():
        row = rows.setdefault(
            level,
            {
                "agent_type": "llm",
                "model": meta["model"],
                "rationale_mode": meta["rationale_mode"],
                "suggestion_level": meta["suggestion_level"],
                "instance_id": instance,
                "seed": str(meta["seed"]),
                "game": game,
                "owner": "model",
                "cohort": cohort,
                "level": level,
                "episode_steps": [],
                "episode_frames": [],
                "episode_outcomes": [],
                "step_unit": "llm_decision",
                "data_source": [path.name],
            },
        )
        row["episode_steps"].append(len(steps))
        row["episode_frames"].append(frames.get((level, attempt)))
        row["episode_outcomes"].append(
            "win"
            if any(s.get("won") is True for s in steps)
            else "loss"
            if any(s.get("lose") is True for s in steps)
            else "incomplete"
        )
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-data", type=Path)
    parser.add_argument("--replays", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--efficientzero",
        type=Path,
        help="Directory containing GAME/self_play_episodes.csv",
    )
    parser.add_argument(
        "--ddqn", type=Path, help="Archived or unsampled-export DDQN JSON"
    )
    parser.add_argument(
        "--empa",
        type=Path,
        help="Directory containing trusted, converted EMPA1 episode-summary JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not any(
        (args.human_data, args.replays, args.efficientzero, args.ddqn, args.empa)
    ):
        parser.error("provide at least one input source")
    rows = human_rows(args.human_data) if args.human_data else []
    seen = set()
    for path in args.replays:
        for row in replay_rows(path):
            identity = row["instance_id"], row["game"], row["cohort"], row["level"]
            if identity in seen:
                raise ValueError(f"Duplicate run/level input: {path}")
            seen.add(identity)
            rows.append(row)
    from scripts.analysis.baseline_episodes import (
        ddqn_rows,
        efficientzero_rows,
        empa_rows,
    )

    for source, builder in (
        (args.efficientzero, efficientzero_rows),
        (args.ddqn, ddqn_rows),
        (args.empa, empa_rows),
    ):
        if source is not None:
            rows.extend(builder(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )
    print(f"Wrote {len(rows)} agent/game/level rows to {args.output}")


if __name__ == "__main__":
    main()
