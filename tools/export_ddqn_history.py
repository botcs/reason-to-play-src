#!/usr/bin/env python3
"""Export complete DDQN W&B episode histories to portable local JSON.

Uses scan_history rather than the sampled history endpoint. Existing downloaded
histories may have used sampling; their completeness cannot be inferred from
successful parsing. No run is modified by this exporter.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SWEEPS = ["h5r1rvvd", "61zc6ya5", "62owf7ea", "gezu9wj8", "s3xrjdkr"]


def export(api, project: str, sweeps: list[str], workers: int = 32) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be positive")
    seen = set()
    selected = []
    for sweep_id in sweeps:
        for run in api.sweep(f"{project}/{sweep_id}").runs:
            if run.id in seen or run.state not in {"finished", "crashed"}:
                continue
            seen.add(run.id)
            game = run.config.get("game_name")
            if not game:
                continue
            selected.append((sweep_id, run, game))

    def read_run(selection):
        sweep_id, run, game = selection
        episodes = []
        keys = ["episode_win", "episode_length", "current_level"]
        for row in run.scan_history(keys=keys, page_size=1000):
            if any(row.get(key) is None for key in keys):
                raise ValueError(f"Incomplete episode record in {run.id}")
            episode = {
                "level": int(row["current_level"]),
                "steps": int(row["episode_length"]),
                "win": int(row["episode_win"]),
            }
            if "_step" in row:
                episode["history_step"] = int(row["_step"])
            episodes.append(episode)
        return {
            "run_id": run.id,
            "sweep_id": sweep_id,
            "game_raw": game,
            "seed": run.config.get("random_seed", "none"),
            "episodes": episodes,
        }

    # map preserves sweep/run order regardless of download completion order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(read_run, selected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="dpag-rl/ddqn-vgdl")
    parser.add_argument("--sweeps", nargs="+", default=DEFAULT_SWEEPS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    import wandb

    records = export(wandb.Api(), args.project, args.sweeps, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "source": f"wandb:{args.project}",
                "method": "scan_history (unsampled)",
                "runs": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(records)} runs to {args.output}")


if __name__ == "__main__":
    main()
