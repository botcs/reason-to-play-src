# LLM evaluation pipeline

All runs are multi-turn: conversation history persists across decisions.
`harness.rationale_mode` selects one of three mechanisms:

| Mode | Assistant context | Human replay |
| --- | --- | --- |
| `action-only` | `{"action": "right"}` | Inject the recorded human action; no LLM call |
| `prompted-rationale` | Explicit rationale plus action | Not supported |
| `copied-reasoning` | Native reasoning copied into a rationale field plus action | Impute reasoning conditioned on the recorded action |

Generative gameplay uses `prompts/gameplay/{mode}_{suggestion}.txt`.
Copied-reasoning imputation uses `prompts/replay/copied-reasoning_{suggestion}.txt`:
the action is disclosed to the imputation model. Before feature extraction the
trace uses the gameplay prompt, removes the action hint from user turns, and
places the copied rationale and recorded action in assistant turns. The
imputation model and extraction model are independently selected.

`harness.suggestion_level` is `minimal`, `elaborate`, or `oracle` for gameplay.
Oracle provides the game rules and must be reported as an ablation. Do not
mix it with the rule-discovery conditions.

## Entry points

- [`generative_gameplay/run.py`](generative_gameplay/run.py): LLM decisions and
  replay generation, controlled by [`../../conf/config.yaml`](../../conf/config.yaml).
- [`human_replay/run_replay.py`](human_replay/run_replay.py): BSON human plays to
  action-only or imputed sessions.
- [`human_replay/extract_features.py`](human_replay/extract_features.py):
  overlapping conversation windows to per-layer hidden, attention, and MLP
  activations; configured by
  [`../../conf/extract_features/default.yaml`](../../conf/extract_features/default.yaml).

See [the reproducibility guide](../../docs/reproducibility.md) for commands,
input/output paths, fMRI alignment, analysis, and known release gaps.
