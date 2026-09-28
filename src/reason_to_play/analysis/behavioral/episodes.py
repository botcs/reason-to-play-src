"""Build the figure input CSV from canonical human recordings and agent replays.

No experiment database is required. Human steps count non-idle keypress frames;
model steps count decisions. Engine frames are stored separately. Select one
complete generative replay per run, including the preceding steps when a run
continues from a checkpoint. Episode boundaries follow the recorded frame clocks.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from reason_to_play.data.behavior import (
    GAMES,
    game_identity,
    human_outcome,
    iter_plays,
    play_states,
)

from reason_to_play.data.replay_behavior import play_recording_path, replay_paths

__all__ = ["GAMES", "game_identity", "human_outcome", "human_rows", "replay_rows"]
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


def human_rows(data_dir: Path) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for doc in iter_plays(data_dir):
        if int(doc["run_id"]) == 0:
            continue
        game, cohort = game_identity(doc["game_name"])
        level = int(doc["level_id"])
        if game not in GAMES or not 0 <= level <= 8:
            continue
        states = play_states(doc)
        subject = f"sub-{int(doc['subj_id']):02d}"
        key = subject, game, cohort, level
        row = grouped.setdefault(
            key,
            {
                "agent_type": "human",
                "model": "Human",
                "rationale_mode": "none",
                "suggestion_level": "none",
                "instance_id": subject,
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
        source = play_recording_path(doc)
        if source not in row["data_source"]:
            row["data_source"].append(source)
    return list(grouped.values())


def replay_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        replay = json.load(stream)
    if replay.get("source") != "generative":
        raise ValueError(f"Expected generative replay: {path}")
    meta = replay["meta"]
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
    actions = [s for s in replay["steps"] if not s["action"].startswith("_")]
    if [s.get("step") for s in actions] != list(range(len(actions))):
        raise ValueError(
            f"Expected complete, ordered action steps starting at zero: {path}"
        )
    states = replay.get("states", [])
    if not states:
        raise ValueError(f"Replay has no recorded frames: {path}")
    if "resumed_at_level" in meta or "resumed_from_steps" in meta:
        prefix = meta.get("resumed_from_steps")
        first = states[0]
        if (
            not isinstance(prefix, int)
            or not 0 <= prefix <= len(actions)
            or replay.get("total_steps") != len(actions)
            or int(first["attempt"]) != 0
            or first["time"] != 0
        ):
            raise ValueError(
                f"Continuation must include its complete preceding history: {path}"
            )

    # Attempt labels can restart after continuation. The frame timeline is the
    # identity of an episode; a repeated (level, attempt) is not the same play.
    frame_episodes = []
    frames: dict[int, int] = {}
    episode = -1
    previous = None
    for state in states:
        identity = int(state["level"]), int(state["attempt"])
        if previous is None or identity != previous[:2] or state["time"] < previous[2]:
            episode += 1
        frame_episodes.append(episode)
        frames[episode] = max(frames.get(episode, 0), state["time"])
        previous = (*identity, state["time"])
    groups: dict[tuple, list[dict]] = defaultdict(list)
    previous_index = -1
    for step in actions:
        index = step.get("state_index")
        if not isinstance(index, int) or not previous_index <= index < len(states):
            raise ValueError(f"Invalid or unordered action frame reference: {path}")
        state = states[index]
        level, attempt = int(step["level"]), int(step["attempt"])
        if (level, attempt) != (int(state["level"]), int(state["attempt"])):
            raise ValueError(
                f"Action and referenced frame disagree on level/attempt: {path}"
            )
        previous_index = index
        if 0 <= level <= 8:
            groups[level, frame_episodes[index]].append(step)
    rows: dict[int, dict] = {}
    for (level, episode), steps in groups.items():
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
        row["episode_frames"].append(frames[episode])
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
    parser.add_argument(
        "--human-data",
        type=Path,
        help="Canonical behavior/human directory, or downloaded release root",
    )
    parser.add_argument(
        "--replays",
        type=Path,
        nargs="*",
        default=[],
        help="Generative replay files or directories searched recursively",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--efficientzero",
        type=Path,
        help="Directory containing GAME/episodes.csv",
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
    if args.workers < 1:
        parser.error("--workers must be positive")
    human_files = replay_paths(args.human_data) if args.human_data else []
    if args.human_data and not human_files:
        parser.error(f"No human files below {args.human_data}")
    model_files = []
    for path in args.replays:
        if path.is_dir():
            files = sorted(path.rglob("*.generative.replay.json.gz"))
            if not files:
                parser.error(f"No generative replay files below {path}")
            model_files.extend(files)
        else:
            model_files.append(path)
    rows = []
    seen = set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for group in pool.map(human_rows, human_files):
            rows.extend(group)
        for path, group in zip(model_files, pool.map(replay_rows, model_files)):
            for row in group:
                identity = row["instance_id"], row["game"], row["cohort"], row["level"]
                if identity in seen:
                    raise ValueError(f"Duplicate run/level input: {path}")
                seen.add(identity)
                rows.append(row)
    from reason_to_play.analysis.behavioral.baselines import (
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
