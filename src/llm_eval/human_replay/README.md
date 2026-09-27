# Human Replay

Replays human behavioral data from Tomov et al. 2023 through the same harness
used by generative gameplay, to extract LLM representations for fMRI decoding.

See `src/llm_eval/README.md` for the full pipeline architecture and flowcharts.

## Two-phase pipeline

### Phase 1: Imputation (`rationale_mode=copied-reasoning` only)

A reasoning-capable LLM is hinted with the human's action ("Your action: X")
and asked to think through why the player would have chosen it. The hidden
reasoning trace is then copied into the conversation history as the
player's stated rationale. Uses prompts from `prompts/replay/`.

`rationale_mode=action-only` skips Phase 1 -- the human action is injected
directly, no LLM call.

`rationale_mode=prompted-rationale` is not supported in replay.

### Phase 2: Feature extraction

The saved traces are reconstructed with the **gameplay** system prompt
(from `prompts/gameplay/`) so the trace is structurally identical to what the
model would produce during generative gameplay. This enables direct comparison
of hidden-state features between gameplay and replay.

Reconstruction steps:
1. Swap system prompt: replay (imputation) --> gameplay
2. Strip "Your action:" hints from user messages
3. Actions in assistant role (not user role)

## Key files

**Generation (Phase 1):**
- `run_replay.py` -- Hydra entry point
- `replay_agent.py` -- Processes zstates through Harness, generates reasoning
- `zstate_adapter.py` -- Converts Tomov 2023 BSON zstates to engine format
- `data_loader.py` -- Loads human play data from BSON files
- `sampling.py` -- Step selection strategies

**Extraction (Phase 2):**
- `extract_features.py` -- Main sliding-window entry point; saves session `.pt` tensors and `_prompts.jsonl.gz`
- `chunked_feature_saver.py` -- Legacy per-level layout, not the current extractor's output
- `prompt_utils.py` -- Loads saved prompts from Phase 1

All shared infrastructure (config, harness, formatters, wrappers) lives in
`src/llm_eval/shared/`.

## Usage

```bash
# Phase 1: run replay for a single subject (action-only, no LLM)
python -m src.llm_eval.human_replay.run_replay replay.subject=sub-01 harness.rationale_mode=action-only

# copied-reasoning imputation with a reasoning model
python -m src.llm_eval.human_replay.run_replay replay.subject=sub-01 \
    harness.rationale_mode=copied-reasoning llm.model=deepseek/deepseek-v3.2
```

## Sliding-window extraction

```bash
python -m src.llm_eval.human_replay.extract_features \
    'prompts=out/replays/*.human.replay.json.gz' \
    model=Qwen/Qwen3.5-9B \
    output_dir=out/features/action-only/minimal/wf-0.3-overlap-0.5
```

Use a separate output root per condition. Files below that root are named
`model-{sanitized_model_id}/{all|compressed}/{subject}/{game}.pt`; the path
does not encode rationale mode, suggestion level, or window settings. Each
file stores these settings and the exact model ID in its `session` dictionary.
Compressed actions produce irregular time intervals: align using target
metadata timestamps, never by assuming uniform spacing between tensor rows.

See [the release guide](../../../docs/reproducibility.md) for GPU prerequisites
and the distinction between raw activations and scanner-aligned features.
