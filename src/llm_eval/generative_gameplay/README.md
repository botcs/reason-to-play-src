# Generative Gameplay

LLM agents play VGDL games in a closed loop, inferring rules by trial and error.
The agents are deliberately NOT told the game rules -- they must learn through
observation of scores, events, and win/lose outcomes.

## Architecture

```
GameplayAgent.run()
  |
  +-- for each level (until max_levels or all levels done):
  |     |
  |     +-- _run_level()
  |     |     |
  |     |     +-- env.reset()
  |     |     +-- loop:
  |     |     |     Harness.build_messages(state, history)
  |     |     |     LLM.generate(messages) -> action
  |     |     |     env.step(action)
  |     |     |     inject idle_frames NO_OPs
  |     |     |     EventLogger.log_action(...)
  |     |     |     check win/lose/timeout
  |     |     |
  |     |     +-- return (won, died)
  |     |
  |     +-- update cumulative_wins / cumulative_losses / cumulative_success
  |     +-- on win: replay same level (Tomov 2023 protocol)
  |     +-- on death: restart same level
  |     +-- on timeout: advance to next level
  |
  +-- log final results to W&B and .replay.json.gz
```

## Key files

- `agent.py` -- GameplayAgent (main loop), InteractionTracker, build_interactions_from_game()
- `run.py` -- Hydra entry point
- `export_replay.py` -- capture_state() and write_replay_file() for crash recovery

All shared infrastructure (config, harness, formatters, wrappers) lives in
`src/llm_eval/shared/`.

## Usage

```bash
# Mock mode (testing)
python -m src.llm_eval.generative_gameplay.run game.game=bait_vgfmri4 llm.backend=mock game.advancement.level_frame_budget=50

# OpenRouter
python -m src.llm_eval.generative_gameplay.run llm.backend=openrouter llm.model=deepseek/deepseek-v3.2

# Full sweep grid
python -m src.llm_eval.generative_gameplay.run harness.rationale_mode=copied-reasoning harness.suggestion_level=elaborate
```
