#!/usr/bin/env python3
"""Convert trusted legacy EMPA pickles to portable episode-summary JSON.

Pickle can execute code during loading. This opt-in converter is for locally
verified research exports; the release analysis consumes JSON only.
"""

import argparse
import json
import pickle
from pathlib import Path

from reason_to_play.data.behavior import canonical_game_id


def convert(source: Path, output: Path) -> int:
    paths = sorted(source.glob("*/*/dumps/interaction_data/lvl_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No EMPA interaction-data pickles below {source}")
    destinations = set()
    for path in paths:
        with path.open("rb") as stream:
            data = pickle.load(stream)
        episodes = []
        for episode in data["episode_summaries"]:
            steps, outcome = episode["steps"], episode["outcome"]
            if isinstance(steps, bool) or int(steps) != steps or steps < 0:
                raise ValueError(f"Invalid episode steps in {path}")
            if not isinstance(outcome, str) or not outcome.strip():
                raise ValueError(f"Unknown episode outcome {outcome!r} in {path}")
            episodes.append({"steps": int(steps), "outcome": outcome})
        relative = path.relative_to(source)
        game = canonical_game_id(relative.parts[0])
        trial = int(relative.parts[1])
        level = int(path.stem.removeprefix("lvl_")) - 1
        if level < 0:
            raise ValueError(f"Invalid one-based EMPA level in {path}")
        destination = output / game / f"trial-{trial:02d}" / f"level-{level:02d}.json"
        if destination in destinations:
            raise ValueError(f"Multiple EMPA source files map to {destination}")
        destinations.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "game": game,
                    "source_game": relative.parts[0],
                    "trial": trial,
                    "level": level,
                    "episode_summaries": episodes,
                },
                indent=2,
            )
            + "\n"
        )
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-pickle", action="store_true")
    args = parser.parse_args()
    if not args.trusted_pickle:
        parser.error(
            "--trusted-pickle is required to load trusted legacy research files"
        )
    print(f"Converted {convert(args.input, args.output)} files")


if __name__ == "__main__":
    main()
