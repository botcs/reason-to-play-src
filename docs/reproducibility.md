# Reproducing Reason to Play

[NeurIPS 2026 paper](https://openreview.net/forum?id=Y1oX1yuaWM) ·
[Citation](../CITATION.cff) ·
[Interactive results](https://botcs.github.io/reason-to-play/)

Start with the [dataset analysis guide](guides/dataset-analysis.md) to analyze
canonical human recordings, processed BOLD and model features from a local
copy of the derivative dataset. Ordinary analysis needs neither OpenNeuro BSON
nor raw MRI, author cloud credentials or an online atlas download. The release
is being prepared; pin the verified dataset and code revisions when published.

For new gameplay or feature extraction, follow the examples below. Gameplay
and extraction modules currently run from the checkout root; installed
`reason_to_play` analysis modules can run from any working directory. Data and
model weights are separate from this code checkout.

## Pipeline and inputs

| Stage | Implementation | Input | Output |
| --- | --- | --- | --- |
| Human replay | `src.llm_eval.human_replay.run_replay` | Released per-game human files | New replay conversations with embedded trajectories |
| Model gameplay | `src.llm_eval.generative_gameplay.run` | Game definitions, prompts and model | Generative replays |
| LLM feature extraction | `src.llm_eval.human_replay.extract_features` | Prompt records and model weights | Activations, target metadata and exact prompts |
| fMRI reconstruction | `reason_to_play.fmri.fmriprep`, `.preprocess` | Raw BIDS, then fMRIPrep output | Preprocessed BOLD |
| Neural alignment | `reason_to_play.fmri.align_baselines`, `.align_llm` | Canonical behavior, BOLD and features | Features and nuisance variables sampled on the BOLD timeline |
| Behavioral analysis | `reason_to_play.analysis.behavioral` | Canonical behavior and recorded model runs | Episode tables and figures |
| Neural analysis | `reason_to_play.analysis.neural` | Processed BOLD/features, folds, masks and atlas | Encoding results, ROI tables and figures |

Raw model activations and features sampled at scanner times are different
products. Join them using recorded timestamps and scanner timing. In particular,
compressed-action tensor row numbers are not scanner-volume indices.

## fMRI reconstruction and shared data paths

The [optional raw preprocessing guide](guides/fmri-preprocessing.md) starts from
OpenNeuro ds004323 v1.0.0 and the recorded fMRIPrep configurations. It is useful
when studying preprocessing choices or reconstructing upstream derivatives.
For fitting the supplied analysis inputs or aligning a new model to the supplied
BOLD, use the [dataset guide](guides/dataset-analysis.md).

The original launcher, recorded configurations and analysis methods carry
source attribution. Available source code does not prove
the producing invocation, checkpoint or random seed of every archived result;
the [reproduction limits](reproduction-limits.md) distinguishes these.

## Installation

For API gameplay and action-only human replay, run from the checkout using
Python 3.12:

```bash
python -m pip install -r requirements.txt
```

For local model inference and feature extraction, the pinned environment uses
CUDA 12.8 on an HGX B200 node with eight GPUs. The matching install is:

```bash
python -m pip install -r requirements-full.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

These pins describe that hardware environment. For another CUDA/GPU stack,
select compatible Torch, Triton and kernel builds; the Blackwell-oriented extras
include `tilelang`, `flash-linear-attention`, `causal-conv1d` and
`torch-c-dlpack-ext`. Do not treat the complete pinned environment as a portable
CPU install. The [DeepSeek runtimes](#optional-deepseek-runtimes) have additional
kernel and model-conversion requirements. Full-model execution on other hardware
is outside the bounded validation reported here.

Dataset analysis has its own smaller [installation](guides/dataset-analysis.md#install),
also available through `requirements-analysis.txt`. DDQN and EfficientZero use
the [baseline environment](guides/baselines.md), including its separately
pinned engines and hardware-appropriate Torch build. EfficientZero feature
extraction runs directly as `python -m agents.efficientzero.extract_features`
or `python -m agents.efficientzero.extract_traces`. Its optional pinned
[training submodule](guides/baselines.md#efficientzero-training) has a separate
upstream environment and is not needed for analysis or feature extraction.
Raw-MRI reconstruction
requires the container and external inputs in the
[fMRI guide](guides/fmri-preprocessing.md).

The source tree can run without the private DynamoDB experiment tables:
`exp_db.enabled=false` is the default. `logging.wandb_project=''` disables
remote logging. API gameplay additionally needs `OPENROUTER_API_KEY`.

Optional [Text-observation experiments](guides/text-observations.md) use a
separate vLLM environment and JSONL interface. They are separate from the
paper's human replay and feature workflow below.

## Replay and model gameplay

Open saved replay files in the [web viewer](https://botcs.github.io/reason-to-play/replay.html).
The same webapp provides [interactive play](https://botcs.github.io/reason-to-play/interactive-gameplay.html)
and the [replay catalogue](https://botcs.github.io/reason-to-play/catalogue.html).
Python environments run headlessly. Their `rgb_array` output supplies image
arrays to computation; it does not open a player window.

The Python `VGDLEnv` interface returns `(observation, info)` from `reset()`
and `(observation, reward, terminated, truncated, info)` from `step()`.
Call `reset()` before stepping and `close()` when finished.

Conversation history persists across decisions. `harness.suggestion_level`
selects `minimal`, `elaborate`, or `oracle`; oracle supplies the game rules.
Gameplay supports `action-only`, `prompted-rationale`, and `copied-reasoning`.
The `game.advancement` strategy selects `blocked_curricula` or `fixed_budget`;
frame budgets count engine updates, including idle frames, separately from
model decisions. Their configuration is defined in
[`shared/config.py`](../src/llm_eval/shared/config.py).

The human data root contains
`sub-XX/GAME/CONDITION.human.replay.json.gz`, with the trajectory, conversation
and scanner metadata embedded in each file. Point to that directory:

```bash
python -m src.llm_eval.human_replay.run_replay \
    replay.data_dir=/absolute/path/to/behavior/human \
    replay.subject=sub-13 \
    harness.rationale_mode=action-only harness.suggestion_level=minimal \
    logging.output_dir=out/replays/action-only/minimal
```

Action-only replay makes no model API call. It creates new conversations from
the recorded observations and embeds the trajectory for browser inspection.
Each selected step retains its original play ID and frame index. For paper
feature reproduction, use the already saved conversations in the release;
regenerating prompts with a different harness revision can change model inputs.
Completed action-only output retains the complete selected participant/game
inventory; partial-play subsets are rejected. Imputed and narration artifacts
are not part of the released human dataset. For a new imputation experiment, set
`harness.rationale_mode=copied-reasoning` and a reasoning-capable
`llm.model`. The model is given the recorded action, then its reasoning is
copied into the saved trace; this is not evidence of the participant's actual
thought process. Replay does not support `prompted-rationale`.

```bash
python -m src.llm_eval.generative_gameplay.run \
    game.game=bait_vgfmri4 \
    llm.backend=openrouter llm.model=deepseek/deepseek-v3.2 \
    harness.rationale_mode=copied-reasoning \
    logging.output_dir=out/gameplay
```

A credential-free engine check is:

```bash
python -m src.llm_eval.generative_gameplay.run \
    llm.backend=mock game.game=bait_vgfmri4 \
    game.advancement.total_frame_budget=20 logging.output_dir=out/smoke
```

## Latent activation extraction

The generic Hugging Face extractor uses
[`conf/extract_features/default.yaml`](../conf/extract_features/default.yaml).
It builds overlapping windows, preserves complete messages, and assigns every
assistant-turn target to exactly one window. It stores post-residual hidden
states plus pre-residual attention and MLP states.

```bash
python -m src.llm_eval.human_replay.extract_features \
    'prompts=out/replays/action-only/minimal/*.human.replay.json.gz' \
    model=Qwen/Qwen3.5-9B \
    model_revision="${HF_MODEL_COMMIT:?Set HF_MODEL_COMMIT to an immutable model commit}" \
    window_fraction=0.3 overlap=0.5 action_compression=false \
    output_dir=out/features/action-only/minimal/wf-0.3-overlap-0.5
```

The directory layout below `output_dir` is:

```text
model-{sanitized_model_id}/{all|compressed}/{subject}/{game}.pt
model-{sanitized_model_id}/{all|compressed}/{subject}/{game}_prompts.jsonl.gz
```

Set `HF_MODEL_COMMIT` to the model repository’s immutable commit before running.
The generic loader resolves the configuration once, uses that same snapshot for
the tokenizer and model weights, and records both `requested_model_revision` and
the resolved `model_revision` in the session and provenance. Historical files
without that metadata remain unpinned; a present model ID alone is not a
checkpoint hash. Pinned extraction requires `exp_db.enabled=false`: the
experiment-table slot IDs lack a revision dimension, so enabling them
would risk reusing a different checkpoint.

Use a distinct `output_dir` for each revision/rationale/suggestion/window condition;
those dimensions are recorded in the payload but absent from its
filename. The `session.model` and `provenance.model` fields preserve the exact
Hugging Face ID; filesystem sanitization is not a reversible model identifier.
Keep all three tensor streams, target `metadata`, `windows`, `session`,
`provenance`, and companion prompts together. Do not discard `realworld_ts`,
run/play identifiers, or token offsets during release normalization.

Message-boundary detection supports ChatML and DeepSeek chat templates; other
templates fail explicitly. Extraction stops on the first failed session by
default. `exp_db.continue_on_session_error=true` processes the remaining sessions
and retains successful outputs, but all three extraction entrypoints still exit
nonzero if any selected session failed. Completed, skipped/non-claimable and
failed sessions are counted separately.

### Optional DeepSeek runtimes

`deepseek_inference/` contains the V3.2 custom runtime and checkpoint converter.
Its upstream MIT notice and [sources](sources/README.md#sources-and-implementations) identify
the implementation.
Convert model weights for the same tensor-parallel size used during extraction:

```bash
python deepseek_inference/convert.py \
    --hf-ckpt-path /absolute/path/to/DeepSeek-V3.2 \
    --save-path /absolute/path/to/converted-v32 \
    --n-experts 256 --model-parallel 8

torchrun --nproc-per-node 8 --standalone \
    -m src.llm_eval.human_replay.extract_features_v32 \
    'prompts=out/replays/action-only/minimal/*.human.replay.json.gz' \
    model=deepseek-ai/DeepSeek-V3.2 \
    +ckpt_path=/absolute/path/to/converted-v32 \
    +ds_config=deepseek_inference/vgdl_DS32.json \
    output_dir=out/features-v32/action-only/minimal
```

The leading `+` on the two DeepSeek-specific overrides is required because
they are not members of the shared Hydra extraction config. V3.2 also needs
its runtime kernel dependencies, including `fast_hadamard_transform`; see
`deepseek_inference/requirements.txt` and the full environment requirements.
Full-scale GPU extraction is outside the bounded checks reported here.

The V4 runtime is also included in `deepseek_v4_inference/`, with the MIT
notice and exact upstream Hugging Face revision recorded in its provenance.
Its converter and configs distinguish Pro, Flash, and Base checkpoints. Use
`extract_features_v4` with the matching converted checkpoint and config;
conversion parallelism must match `torchrun` parallelism:

```bash
torchrun --nproc-per-node 8 --standalone \
    -m src.llm_eval.human_replay.extract_features_v4 \
    'prompts=out/replays/action-only/minimal/*.human.replay.json.gz' \
    model=deepseek-ai/DeepSeek-V4-Flash \
    +ckpt_path=/absolute/path/to/converted-v4-flash \
    +ds_config=deepseek_v4_inference/vgdl_DS4_Flash.json \
    output_dir=out/features-v4-flash/action-only/minimal
```

Read each runtime's `convert.py --help` and `requirements.txt` before converting
weights; V4 uses different expert counts and quantization from V3.2. Run V3.2
and V4 in separate processes because their runtime module names overlap.
Match the V4 model variant, expert count and runtime configuration; Pro, Flash
and corresponding Base configurations are included. Source completeness
and configuration checks do not establish a successful full-model GPU run.

## Behavioural analysis without cloud credentials

The [behavioral workflow](guides/dataset-analysis.md#behavioral-analysis) reads
complete human recordings and model gameplay. It preserves original nullable
outcomes, identifies avatar deaths from events, excludes practice and uses an
explicit cohort/level selection. Human keypress frames, model decisions and
engine frames remain separate measurements.

The [baseline guide](guides/baselines.md) covers offline DDQN, EfficientZero
and EMPA inputs. The plot's retrospective blocked-curriculum rule does not make
the original human and model advancement protocols identical. Report whether
solve rates divide by available levels or reached levels.

## Neural figures

The [neural workflow](guides/dataset-analysis.md#neural-encoding-from-processed-inputs)
documents explicit processed inputs, seeded fresh fits, the shared spatial
mask, ROI aggregation and model/cohort selection. Use the supplied atlas and
pinned mask when rebuilding the study's ROI table. Fresh baseline result labels
require an explicit [layer map](../experiments/neurips2026/baseline-layer-map.example.json).
The unresolved historical EfficientZero layer mapping is not inferred.

## Validation and release boundaries

```bash
python -m pip install -r requirements-dev.txt
ruff check src/ tests/ tools/
ruff format --check src/ tests/ tools/
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/ -x -q
```

Set `REASON_TO_PLAY_HUMAN_DATA` to a downloaded human JSON file or human-data
directory to run the optional recorded-trajectory checks. These checks use a
Bait play; choose a file containing that game when providing a single file.

Tests cover canonical/source equivalence, timing and nuisance alignment,
synthetic NIfTI through ridge fitting, result identities, ROI masks, replay
clocks and behavioral exports. Optional tokenizer tests skip without cached
model files. Read the [dataset validation limits](guides/dataset-analysis.md#verification-boundary)
before equating these checks with a complete historical paper rerun.
