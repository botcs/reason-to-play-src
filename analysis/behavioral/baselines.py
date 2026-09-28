"""Offline importers for the preserved baseline behavior exports.

EfficientZero CSV and DDQN JSON preserve recorded step counts; frame counts are
unknown here and are never invented. EMPA input is JSON with episode_summaries;
convert legacy trusted pickles separately before distribution.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from analysis.behavioral.episodes import GAMES, game_identity


def _row(
    agent: str, model: str, game_raw: str, instance: str, level: int, source: str
) -> dict:
    game, cohort = game_identity(game_raw)
    if game not in GAMES:
        raise ValueError(f"Unrecognized game: {game_raw}")
    return {
        "agent_type": agent,
        "model": model,
        "rationale_mode": "none",
        "suggestion_level": "none",
        "instance_id": instance,
        "seed": "none",
        "game": game,
        "cohort": cohort,
        "level": level,
        "owner": "baseline",
        "episode_steps": [],
        "episode_frames": [],
        "episode_outcomes": [],
        "step_unit": "baseline_recorded_step",
        "data_source": [source],
    }


def _append(row: dict, steps: int, outcome: str) -> None:
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError(f"Invalid episode step count: {steps!r}")
    if not isinstance(outcome, str) or not outcome.strip():
        raise ValueError(f"Unknown baseline outcome: {outcome!r}")
    row["episode_steps"].append(int(steps))
    row["episode_frames"].append(None)
    row["episode_outcomes"].append(outcome)


def _win(value: int) -> str:
    if value not in (0, 1):
        raise ValueError(f"Expected binary baseline win, got {value!r}")
    return "win" if value == 1 else "loss"


def efficientzero_rows(root: Path) -> list[dict]:
    """Read GAME/episodes.csv, ordered by preserved episode_index."""
    files = sorted(root.glob("*/episodes.csv"))
    files += sorted(root.glob("*/self_play_episodes.csv"))
    if not files:
        raise FileNotFoundError(f"No GAME/episodes.csv below {root}")
    output = []
    seen_games = set()
    for path in files:
        game_key = game_identity(path.parent.name)
        if game_key in seen_games:
            raise ValueError(f"Duplicate EfficientZero game {game_key} below {root}")
        seen_games.add(game_key)
        with path.open(newline="") as stream:
            episodes = sorted(
                csv.DictReader(stream), key=lambda row: int(row["episode_index"])
            )
        grouped = {}
        seen = set()
        for episode in episodes:
            index = int(episode["episode_index"])
            if index in seen:
                raise ValueError(f"Duplicate episode_index {index} in {path}")
            seen.add(index)
            level = int(episode["resolved_level"])
            if not 0 <= level <= 8:
                continue  # Warmup levels are not human experimental levels.
            row = grouped.setdefault(
                level,
                _row(
                    "ez",
                    "EfficientZero",
                    path.parent.name,
                    "run0",
                    level,
                    path.relative_to(root).as_posix(),
                ),
            )
            _append(row, int(episode["episode_len"]), _win(int(episode["win"])))
        output.extend(grouped.values())
    return output


def ddqn_rows(path: Path) -> list[dict]:
    """Read archived records containing run_id, game_raw, and episode lists."""
    records = json.loads(path.read_text())
    if isinstance(records, dict):
        records = records["runs"]
    output = []
    seen = set()
    for record in records:
        identity = record["run_id"], record["game_raw"]
        if identity in seen:
            raise ValueError(f"Duplicate DDQN run {identity}")
        seen.add(identity)
        grouped = {}
        for episode in record["episodes"]:
            level = int(episode["level"])
            if not 0 <= level <= 8:
                continue
            row = grouped.setdefault(
                level,
                _row(
                    "ddqn",
                    "DDQN",
                    record["game_raw"],
                    record["run_id"],
                    level,
                    path.name,
                ),
            )
            row["seed"] = str(record.get("seed", "none"))
            _append(row, episode["steps"], _win(episode["win"]))
        output.extend(grouped.values())
    return output


def empa_rows(root: Path) -> list[dict]:
    """Read GAME/trial-NN/level-NN.json with zero-based level IDs."""
    files = sorted(root.glob("*/trial-*/level-*.json"))
    files += sorted(root.glob("*/*/dumps/interaction_data/lvl_*.json"))
    if not files:
        raise FileNotFoundError(f"No EMPA episode-summary JSON below {root}")
    output = []
    seen = set()
    for path in files:
        current = path.name.startswith("level-")
        match = re.fullmatch(r"level-(\d+)" if current else r"lvl_(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Invalid EMPA level filename: {path}")
        level = int(match[1]) - (0 if current else 1)
        if not 0 <= level <= 8:
            continue
        relative = path.relative_to(root)
        game, trial = relative.parts[:2]
        if current:
            trial_match = re.fullmatch(r"trial-(\d+)", trial)
            if trial_match is None:
                raise ValueError(f"Invalid EMPA trial directory: {path}")
            trial = str(int(trial_match[1]))
        identity = (*game_identity(game), trial, level)
        if identity in seen:
            raise ValueError(f"Duplicate EMPA game/trial/level {identity}")
        seen.add(identity)
        row = _row("empa1", "EMPA", game, f"trial{trial}", level, relative.as_posix())
        document = json.loads(path.read_text())
        if current:
            if (
                game_identity(document["game"]) != game_identity(game)
                or document["trial"] != int(trial)
                or document["level"] != level
            ):
                raise ValueError(f"EMPA metadata disagrees with path: {path}")
        for episode in document["episode_summaries"]:
            _append(row, episode["steps"], episode["outcome"])
        output.append(row)
    return output
