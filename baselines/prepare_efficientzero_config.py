#!/usr/bin/env python3
"""Create a portable copy of the pinned EfficientZero experiment configuration.

Only the game-directory location changes. The source training CLI merges its
experiment file after some command-line overrides, so editing a copied config
is more reliable than passing an absolute env.game_folder override alone.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "checkouts/efficientzero/ez/config/exp/vgdl.yaml",
    )
    parser.add_argument(
        "--games", type=Path, default=ROOT / "vendor/rc_rl/ez/all_games_recovered"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.games.is_dir():
        parser.error(f"missing game directory: {args.games}")
    if args.output.resolve() == args.experiment.resolve():
        parser.error(
            "write a separate portable config; do not overwrite the pinned source"
        )
    config = OmegaConf.load(args.experiment)
    config.env.game_folder = str(args.games.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output)
    print(f"Wrote portable experiment configuration: {args.output.resolve()}")


if __name__ == "__main__":
    main()
