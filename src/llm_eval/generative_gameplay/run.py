# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
#!/usr/bin/env python
"""
Entry point for VGDL LLM evaluation.

Uses Hydra for config composition and CLI overrides. All arguments are
key=value pairs -- no double-hyphen flags. Harness presets are Hydra
config groups (conf/harness/*.yaml), selected with harness=<name>.

This design makes W&B sweep integration trivial:
    command:
      - ${env}
      - python
      - -m
      - src.llm_eval.generative_gameplay.run
      - ${args_no_hyphens}

Usage:
    python -m src.llm_eval.generative_gameplay.run llm.backend=mock game.max_levels=2
    python -m src.llm_eval.generative_gameplay.run harness.rationale_mode=copied-reasoning llm.model=deepseek/deepseek-v3.2
    python -m src.llm_eval.generative_gameplay.run harness.rationale_mode=action-only game.game=zelda_vgfmri4
"""

import os
import sys
import traceback

# Ensure repo root is on the path for src.* imports
repo_path = "/".join(os.path.abspath(__file__).split("/")[:-4]) + "/"
sys.path.insert(0, repo_path)

import hydra
from omegaconf import DictConfig, OmegaConf

from src.llm_eval.shared.config import (
    Config,
    FixedBudgetAdvancement,
    validate_config,
)
from src.llm_eval.generative_gameplay.agent import GameplayAgent
from src.llm_eval.shared.replay_codec import load_replay


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Merge with structured schema so to_object() returns a Config dataclass
    schema = OmegaConf.structured(Config)
    merged = OmegaConf.merge(schema, cfg)
    typed_cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]
    validate_config(typed_cfg)

    # Print config summary
    print(f"\n{'=' * 60}")
    print("VGDL LLM Evaluation")
    print(f"{'=' * 60}")
    print(f"Game: {typed_cfg.game.game}")
    print(f"Backend: {typed_cfg.llm.backend}")
    print(f"Model: {typed_cfg.llm.model or '(mock)'}")
    print(f"Rationale mode: {typed_cfg.harness.rationale_mode}")
    print(f"Suggestions: {typed_cfg.harness.suggestion_level}")
    print(f"Max levels: {typed_cfg.game.max_levels}")
    adv = typed_cfg.game.advancement
    if isinstance(adv, FixedBudgetAdvancement):
        print(f"Advancement: fixed_budget ({adv.level_frame_budget} frames/level)")
        budget = adv.level_frame_budget
    else:
        print(
            f"Advancement: blocked_curricula "
            f"({adv.total_frame_budget} total frames, "
            f"{adv.consecutive_wins_required} consecutive wins to advance)"
        )
        budget = adv.total_frame_budget
    if typed_cfg.game.idle_frames > 0 and budget > 0:
        effective = budget // (1 + typed_cfg.game.idle_frames)
        print(
            f"Idle frames: {typed_cfg.game.idle_frames} per action (~{effective} decisions/budget)"
        )
    print(f"Seed: {typed_cfg.seed}")
    if typed_cfg.logging.wandb_project:
        print(f"W&B: {typed_cfg.logging.wandb_project}")

    # Set random seeds
    import random
    import numpy as np

    random.seed(typed_cfg.seed)
    np.random.seed(typed_cfg.seed)

    # Run
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent = GameplayAgent(typed_cfg, timestamp=timestamp)
    print(f"Output: {agent._log_filepath}")

    # Experiment DB tracking -- claim after agent init (wandb run exists)
    exp_db_client = None
    exp_db_slot_id = None
    resume_level = None
    resume_state = None
    if typed_cfg.exp_db.enabled:
        import socket

        from src.llm_eval.neurips_exp_db import ExpDB, slot_id_gameplay
        from src.llm_eval.neurips_exp_db.client import TABLE_GAMEPLAY

        if not typed_cfg.logging.wandb_project:
            raise ValueError(
                "exp_db.enabled=true requires logging.wandb_project to be set "
                "(experiment tracking needs wandb run URLs)"
            )
        exp_db_client = ExpDB(typed_cfg.exp_db.region)
        exp_db_slot_id = slot_id_gameplay(typed_cfg)
        wb = agent._wandb_run

        # Compute S3 key upfront so it's written to DB at claim time
        s3_key = None
        if agent._log_filepath:
            filename = os.path.basename(agent._log_filepath)
            s3_key = f"{typed_cfg.exp_db.s3_prefix}{exp_db_slot_id}/{filename}"

        if typed_cfg.exp_db.resume:
            from pathlib import Path

            slot = exp_db_client.get_slot(TABLE_GAMEPLAY, exp_db_slot_id)
            if slot is None:
                raise RuntimeError(
                    f"Resume: slot {exp_db_slot_id} not found in {TABLE_GAMEPLAY}"
                )
            if slot["status"] != "failed":
                raise RuntimeError(
                    f"Resume: slot {exp_db_slot_id} status is '{slot['status']}', "
                    f"expected 'failed'"
                )
            prior_s3_key = slot.get("s3_replay_key")
            if not prior_s3_key:
                raise RuntimeError(
                    f"Resume: slot {exp_db_slot_id} has no s3_replay_key"
                )
            safe_name = exp_db_slot_id.replace("/", "_").replace("|", "_")
            local_replay = Path(f"/tmp/resume_{safe_name}.json.gz")
            print(f"Resume: downloading {prior_s3_key} ...")
            exp_db_client.download_replay(
                typed_cfg.exp_db.s3_bucket, prior_s3_key, local_replay
            )
            replay_data = load_replay(local_replay)
            resume_level, resume_state = agent.restore_from_replay(replay_data)

            reclaimed = exp_db_client.reclaim_slot(
                TABLE_GAMEPLAY,
                exp_db_slot_id,
                worker=socket.gethostname(),
                wandb_run_id=wb.id,
                wandb_run_url=wb.url,
                s3_replay_key=s3_key,
            )
            if not reclaimed:
                raise RuntimeError(
                    f"Resume: slot {exp_db_slot_id} not reclaimable "
                    f"(status != failed -- race condition?)"
                )
            print(f"ExpDB: reclaimed slot {exp_db_slot_id}")
        else:
            claimed = exp_db_client.claim_slot(
                TABLE_GAMEPLAY,
                exp_db_slot_id,
                worker=socket.gethostname(),
                wandb_run_id=wb.id,
                wandb_run_url=wb.url,
                s3_replay_key=s3_key,
            )
            if not claimed:
                raise RuntimeError(
                    f"Slot {exp_db_slot_id} not claimable (status != pending)"
                )
            print(f"ExpDB: claimed slot {exp_db_slot_id}")

        # Enable incremental S3 sync (every 10 steps, same cadence as local file)
        if s3_key:
            agent.set_s3_sync(exp_db_client._s3, typed_cfg.exp_db.s3_bucket, s3_key)

    print(f"{'=' * 60}\n")

    start = resume_level if resume_level is not None else typed_cfg.game.start_level
    run_failed = False
    results = None
    try:
        results = agent.run(start_level=start, resume_state=resume_state)
    except BaseException as e:
        run_failed = True
        if exp_db_client is not None:
            tb = traceback.format_exc()
            error_msg = f"{e}\n\nstep={getattr(agent, 'step_num', '?')}\n\n{tb}"
            exp_db_client.fail_slot(TABLE_GAMEPLAY, exp_db_slot_id, error_msg=error_msg)
        raise

    if not run_failed and exp_db_client is not None:
        exp_db_client.complete_slot(
            TABLE_GAMEPLAY,
            exp_db_slot_id,
            s3_replay_key=agent._s3_key,
        )

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Outcome: {results['outcome']}")
    print(f"Final Level: {results['final_level']}")
    print(f"Total Steps: {results['total_steps']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
