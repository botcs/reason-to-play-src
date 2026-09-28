# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
#!/usr/bin/env python
"""
Entry point for VGDL LLM evaluation.

Uses Hydra for config composition and CLI overrides. All arguments are
key=value pairs -- no double-hyphen flags. Harness presets are Hydra
config groups (agents/lrm/configs/harness/*.yaml), selected with harness=<name>.

This design makes W&B sweep integration trivial:
    command:
      - ${env}
      - python
      - -m
      - agents.lrm.play
      - ${args_no_hyphens}

Usage:
    python -m agents.lrm.play llm.backend=mock game.max_levels=2
    python -m agents.lrm.play harness.rationale_mode=copied-reasoning llm.model=deepseek/deepseek-v3.2
    python -m agents.lrm.play harness.rationale_mode=action-only game.game=zelda_vgfmri4
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from agents.lrm.config import (
    Config,
    FixedBudgetAdvancement,
    validate_config,
)
from agents.lrm.gameplay.agent import GameplayAgent
from data.replay_codec import load_replay


@hydra.main(config_path="configs", config_name="config", version_base=None)
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

    resume_level = None
    resume_state = None
    if typed_cfg.game.resume_from:
        replay_data = load_replay(typed_cfg.game.resume_from)
        resume_level, resume_state = agent.restore_from_replay(replay_data)
    start = resume_level if resume_level is not None else typed_cfg.game.start_level
    results = agent.run(start_level=start, resume_state=resume_state)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Outcome: {results['outcome']}")
    print(f"Final Level: {results['final_level']}")
    print(f"Total Steps: {results['total_steps']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
