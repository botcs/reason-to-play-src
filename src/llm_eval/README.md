# LLM Evaluation Pipeline

Two parallel pipelines that produce feature-comparable traces:

1. **Generative gameplay** -- LLM plays VGDL games, choosing actions itself
2. **Human replay** -- LLM processes human behavioral data (Tomov et al. 2023)

Both pipelines produce traces with identical message structure and the same
gameplay system prompt, enabling direct comparison of hidden-state features.

## Pipeline Overview

```
                        prompts/gameplay/                prompts/replay/
                    (gameplay system prompts)         (imputation prompts)
                              |                              |
           +------------------+------------------+           |
           |                                     |           |
    Generative Gameplay                   Replay Phase 2     |
    (LLM plays games)                  (feature extraction)  |
           |                                     |           |
           |    +--------------------------------+           |
           |    |  SAME system prompt                        |
           |    |  SAME message structure                    |
           |    |  (user=state, assistant=action+reasoning)  |
           |    |                                            |
           v    v                                            |
     Feature Extraction                              Replay Phase 1
     (hidden states)                                (imputation, persistent only)
                                                     LLM generates reasoning
                                                     for human actions
```

### Generative Gameplay

```
System prompt: prompts/gameplay/{mode}_{suggestion}.txt

+-- For each step: ----------------------------------------+
|                                                           |
|   User: game state + action log                           |
|     |                                                     |
|     v                                                     |
|   LLM: {"rationale": "...", "action": "right"}            |
|     |                                                     |
|     v                                                     |
|   Game engine executes action, returns next state         |
|                                                           |
+-----------------------------------------------------------+
     |
     v
  .replay.json.gz  -->  feature extraction
```

### Human Replay -- Phase 1 (Imputation)

Persistent modes only. Ephemeral modes skip this phase.

The imputation LLM generates reasoning for the human's action:
- System: imputation prompt (`prompts/replay/persistent_*.txt`)
- User: participant's observation + participant's action
- Assistant: `{"rationale": "<imputed reasoning>", "action": "down"}`
- User: participant's observation + participant's action
- Assistant: `{"rationale": "<imputed reasoning>", "action": "right"}`
- ...

The human's actual action is revealed in each user message.
The LLM must produce reasoning that concludes with that action.

### Human Replay -- Phase 2 (Feature Extraction)

The extraction LLM encodes the trace into hidden-state features:
- System: gameplay prompt (`prompts/gameplay/persistent_*.txt`)
- User: participant's observation
- Assistant: `{"rationale": "<copy from phase 1>", "action": "down"}`
- User: participant's observation
- Assistant: `{"rationale": "<copy from phase 1>", "action": "right"}`
- ...

Key differences: the system prompt is swapped to gameplay, participant's actions
are stripped from user messages, and hidden reasoning (CoT) is never included.

## Prompt Architecture

```
prompts/
  gameplay/           <-- used by generative gameplay AND replay Phase 2
    ephemeral_elaborate.txt
    persistent_elaborate.txt
    ephemeral_minimal.txt
    persistent_minimal.txt
  replay/             <-- used by replay Phase 1 (imputation) only
    persistent_elaborate.txt  (imputation: participant's action is revealed)
    persistent_minimal.txt
```

Ephemeral replay modes skip Phase 1 entirely and use gameplay prompts
directly in Phase 2. No ephemeral imputation prompts exist.

## Suggestion Levels

The `suggestion_level` controls how much the system prompt scaffolds the
agent's reasoning:

| Level | System prompt |
|-------|----------------|
| `elaborate` | Detailed game-agent instructions, strategy hints, structured rationale format |
| `minimal` | Bare-bones JSON response schema, no scaffolding |

## Subpackages

- `generative_gameplay/` -- LLM plays games (see its README)
- `human_replay/` -- Two-phase replay pipeline (see its README)
- `shared/` -- Config, harness, formatters, LLM wrappers
