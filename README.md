# Source Code

Anonymized source code for the LLM game-playing evaluation framework.

## Structure

```
source-code/
  src/
    llm_eval/              -- LLM evaluation pipeline
      shared/              -- config, harness, prompt loading, event logging,
                              observation formatting, response parsing,
                              LLM wrappers (OpenRouter, Transformers, DeepSpeed)
      generative_gameplay/ -- LLM plays games (agent loop, Hydra entry point)
      human_replay/        -- replay pipeline (imputation + feature extraction)
    vgdl/                  -- VGDL game engine (interpreter, sprites, renderer)
  prompts/
    gameplay/              -- system prompts by rationale_mode x suggestion_level
    replay/                -- imputation prompts for human replay pipeline
    game_rules/            -- per-game oracle rule descriptions
  games/                   -- VGDL game description + level layout files
  sweeps/                  -- W&B sweep configs for experiment grid
  conf/
    config.yaml            -- Hydra default configuration
```

## Running

Generative gameplay (LLM plays a game):

```bash
python -m src.llm_eval.generative_gameplay.run \
    game.game=bait_vgfmri4 \
    llm.backend=openrouter llm.model=<MODEL> \
    harness.rationale_mode=copied-reasoning
```

Human replay imputation:

```bash
python -m src.llm_eval.human_replay.run_replay \
    replay.subject=sub-01 \
    harness.rationale_mode=action-only
```

## Anonymization

Infrastructure identifiers (S3 bucket names, AWS regions, W&B project
names, local filesystem paths, internal run IDs) have been replaced
with placeholder tokens (e.g. `<ANONYMIZED-BUCKET>`, `<REGION>`).
These do not affect code logic.
