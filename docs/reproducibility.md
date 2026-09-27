# Reproducing Reason to Play

[Paper](https://arxiv.org/abs/2605.08019) ·
[Interactive results](https://botcs.github.io/reason-to-play/) ·
[Raw OpenNeuro dataset, version 1.0.0](https://openneuro.org/datasets/ds004323/versions/1.0.0)

Run commands from this repository's root. The data and model weights are not
part of the source checkout. The [inventory summary](release/derivative-inventory.md) and
[Hugging Face plan](release/huggingface-plan.md)
distinguish artifacts selected for publication from historical experiments.
Keep the raw OpenNeuro version and original S3 keys in provenance; a release
path is a mapping to an existing artifact, not evidence that it was regenerated.
Detailed manifests, participant catalogues and historical tables are prepared
separately and are not included in this Git release. Acquire the raw snapshot
through OpenNeuro and run the preprocessing guide, or use a separately
published, verified derivative release when available. The
[manifest workflow](release/manifest-guide.md) explains how to prepare such a
release with an explicit selection policy.

## Pipeline and inputs

| Stage | Code | Input | Output |
| --- | --- | --- | --- |
| fMRI preprocessing | `analysis/tomov23/` (integrated original launcher/configuration records) | ds004323 1.0.0 BIDS plus recorded fMRIPrep configuration | fMRIPrep derivatives and postprocessed BOLD |
| Behavioural preparation and scanner alignment | `analysis/tomov23/` | original behavioural records, scanner timing, baseline features | `plays/sub-*/run-*.bson`, run metadata, aligned neural targets |
| Human replay generation | `src/llm_eval/human_replay/run_replay.py` | prepared behavioural BSON | `.human.replay.json.gz` or `.imputed.replay.json.gz` plus `.narration.replay.json.gz` |
| Model gameplay generation | `src/llm_eval/generative_gameplay/run.py` | VGDL descriptions, layouts, prompts, model ID | `.generative.replay.json.gz` |
| Latent activation extraction | `src/llm_eval/human_replay/extract_features.py` | replay sessions plus extraction model weights | `.pt` session tensors and `_prompts.jsonl.gz` |
| Neural alignment and encoding | `analysis/tomov23/` | raw activations, their timestamps, scanner-aligned BOLD and nuisance regressors | TR-aligned features and voxelwise encoding results |
| Behavioural analysis | `scripts/analysis/build_episodes.py`, `plot_behavioural.py` | human BSON, complete generative replays, recorded DDQN JSON/EfficientZero CSV/EMPA JSON | episode CSV and behavioural figures |
| Neural figures | `scripts/analysis/plot_encoding.py` | aggregated encoding result CSV | encoding figures |

Raw activations and TR-aligned features are different products. The neural
alignment stage must join activations using their recorded timestamps and
scanner timing. In particular, `action_compression=true` produces irregular
intervals between targets: tensor row number is not a scanner-volume index.

## fMRI reconstruction and shared data paths

The [curated preprocessing/encoding guide](../analysis/tomov23/README.md)
contains the ordered commands from raw OpenNeuro acquisition through result
aggregation. Its commands run from `analysis/tomov23/`; its `workdir/` is a
single explicit local data root. The original `RC_RL` fMRIPrep launcher from
16 October 2025 is retained with checksums, alongside all 38 recorded 24.1.0
TOMLs. The portable runner selects the latest recorded configuration for each
subject, replays its scientific flags explicitly, and records the supplied
container image hash. The original launcher invokes Docker while the TOMLs
report Singularity; this does not establish the exact producing invocation for
every archived derivative.

The preprocessing, base alignment, LLM alignment, encoding, and ROI scripts are
included in this checkout. [Exact source pins](../analysis/tomov23/docs/source-map.json)
and [scientific conventions](../analysis/tomov23/docs/scientific-conventions.md)
distinguish the integrated code from historical sibling branches. Historical
tags refer to the research repository, not this public checkout. Optional
DDQN regeneration uses the [bundled baseline sources](../baselines/README.md)
and explicit local checkpoints, so its default engine path needs no private
GitHub access. The extractor verifies the curated file manifest before import;
existing indexed DDQN features can be used without regeneration.

Follow stages 1–4 there first. Point the replay command below at the resulting
`analysis/tomov23/workdir/prepare_behavioral_data` directory (or its absolute
location). The worked neural path uses **sub-13, vgfmri4**, which is compatible
with the submitted LLM alignment script's game mapping. The sub-01/bait
example in the quickstart is a separate replay demonstration; it is not an
input for the vgfmri4 neural example.

After extracting Qwen features below, the exact directory supplied to the
alignment `--llm-source` argument is:

```text
out/features/action-only/minimal/wf-0.3-overlap-0.5/model-Qwen_Qwen3.5-9B/all
```

From this repository root, for the default local preprocessing root:

```bash
python analysis/tomov23/encoding_model_code/align.py --subject sub-13 \
    --aligned-data analysis/tomov23/workdir/aligned_data/sub-13/aligned_data.npz \
    --plays-bson analysis/tomov23/workdir/raw_behavior/dump/heroku_7lzprs54/plays.bson \
    --runs-bson analysis/tomov23/workdir/prepare_behavioral_data/runs.bson \
    --llm-source 'name=qwen35_9b_sugmin__all__main,dir=out/features/action-only/minimal/wf-0.3-overlap-0.5/model-Qwen_Qwen3.5-9B/all,stream=main' \
    --output-dir analysis/tomov23/workdir/aligned_data
python analysis/tomov23/encoding_model_code/encoding_model.py --subject sub-13 \
    --data-dir analysis/tomov23/workdir/aligned_data \
    --output-dir analysis/tomov23/workdir/encoding_results \
    --layer llm_qwen35_9b_sugmin__all__main_layer_1 --max-level 8 --include-nuisance-bands
```

These commands illustrate an action-only/minimal condition, not an assertion
that it is the headline paper condition. Match the archived model, rationale,
suggestion, window and compression settings when reproducing a reported result.
The base-alignment dependency is explicit: use the indexed DDQN-curriculum
features or existing aligned BOLD. DDQN extraction accepts `--checkpoint-map`
and `--rc-rl-dir`; newly generated features record checkpoint SHA-256 and source
revision. The old trial1-sequential W&B mapping does not prove the identity of
the later indexed curriculum checkpoints.

EfficientZero traces from `baselines/run_efficientzero.py` enter base alignment
through `--ez-features-dir`; EMPA theory records enter through
`--regressors-bson`. The base builder writes separate EZ and decomposed HRR
sidecars, which the encoder now loads alongside base BOLD. The archived aligned
objects inspected so far do not contain the EZ inputs that generated the
published baseline CSV; keep that provenance gap explicit.

The submitted encoder defaults to main features only. The explicit nuisance
flag adds game/level identity, button presses, and time; `--max-level 8` keeps
cross-cohort comparisons on levels 0–8. Past lags 2–5 and within-fold PCA are
preserved. A failed fold produces no result, a failed layer makes the command
exit nonzero, and `--resume` checks completed-result metadata. Use a fresh output
directory when inputs change. The preserved unseeded ridge search and historical
shuffle behavior prevent a claim of identical rerun scores; see the conventions
record before changing scientific settings.

## Installation

Use Python 3.12 and `pip install -r requirements.txt` for API gameplay and
human action-only replay. DDQN and EfficientZero inference additionally use
`baselines/requirements-inference.txt` plus a hardware-appropriate Torch build;
the essential baseline engines, model classes, and extraction scripts are
included without private repository access. `requirements-analysis.txt` adds the dependencies
for CSV generation and figures without installing GPU inference kernels.
`requirements-full.txt` records the original CUDA 12.8/B200 environment for
local feature extraction; adapt hardware-specific kernel packages as described
in the top-level README. fMRI processing uses the separate requirements and
container recipe in `analysis/tomov23/`.

The source tree can run without the private DynamoDB experiment tables:
`exp_db.enabled=false` is the default. `logging.wandb_project=''` disables
remote logging. API gameplay additionally needs `OPENROUTER_API_KEY`.

The optional [legacy text-observation generator](legacy-observation-generation.md)
has a recovered vLLM batch wrapper and uses a separate GPU environment. Its
archived JSONL format is outside the replay/feature workflow below.

## Replay and model gameplay

The prepared behavioural input root must contain `plays/sub-01/run-01.bson`
and the other subject/run files. Point to the directory itself:

```bash
python -m src.llm_eval.human_replay.run_replay \
    replay.data_dir=/absolute/path/to/behavior/human \
    replay.subject=sub-13 \
    harness.rationale_mode=action-only harness.suggestion_level=minimal \
    logging.output_dir=out/replays/action-only/minimal
```

Action-only replay makes no model API call. For imputed reasoning, set
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
legacy experiment-table slot IDs lack a revision dimension, so enabling them
would risk reusing a different checkpoint.

Use a distinct `output_dir` for each revision/rationale/suggestion/window condition;
those dimensions are recorded in the payload but absent from its legacy
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

`deepseek_inference/` restores the V3.2 custom runtime and checkpoint converter
from the development snapshot. Its upstream MIT notice is included. Preserve
[the original source provenance](source-provenance.md) when updating it.
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
No GPU-scale extraction was rerun as part of the release audit.

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

Read each runtime's README and converter help before converting weights; V4
uses different expert counts and quantization from V3.2. Source completeness
and configuration checks do not establish a successful full-model GPU run.

## Behavioural analysis without cloud credentials

Build one row per subject/model run, game variant, and level:

```bash
python scripts/analysis/build_episodes.py \
    --human-data /absolute/path/to/behavior/human \
    --replays /absolute/path/to/selected-replays/*.generative.replay.json.gz \
    --output out/behavioural_cache/episodes.csv

python scripts/analysis/plot_behavioural.py discovery_curriculum_combined \
    --csv out/behavioural_cache/episodes.csv --output-dir out/figures
```

The exporter includes levels 0–8 and excludes practice `run-00`. It reads the
play document's three-valued `win` field, checks avatar `killSprite` events
before classifying a `None` outcome as incomplete, and retains `avatar_died`,
`loss`, and `incomplete` separately. Terminal zstate `win=-1` is never used as
an outcome. It rejects resumed replay fragments and duplicate run/level
inputs; select or reconstruct a complete run before export. Baseline DDQN,
EfficientZero, and EMPA rows come from their own recorded runs. Add
`--ddqn /absolute/data/ddqn.json`, `--efficientzero /absolute/data/EZV2-behaviour`,
and `--empa /absolute/data/empa-json` to include them. The
[baseline guide](../baselines/README.md) contains the unsampled W&B exporter,
trusted-pickle-to-JSON conversion, and offline input schemas. EMPA labels such
as `fmri_timeout` remain intact; missing baseline frame counts remain null.

`episode_steps` follows the historical figure convention: human non-idle
keypress frames and model decisions. These are different clocks. The exporter
also retains `episode_frames` separately, with human frames equal to
`n_states - 1` and model frames taken from the per-attempt engine clock. A
missing model clock is JSON `null`, never an inferred duration. Zero-action
human episodes remain zero, rather than being silently changed to one.

The plotting code retains the historical figure computations, with explicit
CSV/output paths and optional `--tex` for an installed LaTeX distribution.
Its default retrospective curriculum rule requires two consecutive wins.
Humans actually advanced on a fixed schedule; model experiments used blocked
advancement. Do not describe their raw advancement conditions as identical.
The denominator is also material: available levels versus reached levels
produce different solve rates. EMPA means `empa1`; `empa2` remains excluded
by default. Original-vs-converted game Timeout rules differ on some vgfmri4
variants, so elapsed-time comparisons require checking the source rules.

## Neural figures

The historical figure script consumes a table with columns including
`model`, `variant`, `subject`, `layer_idx`, `ROI`, `band`, and `performance`.

Generate LLM ROI rows from completed fits with an explicitly supplied AAL atlas:

```bash
python analysis/tomov23/encoding_model_code/load_and_parse.py \
    --results-dir analysis/tomov23/workdir/encoding_results \
    --fit-condition with-nuisance \
    --atlas /absolute/path/to/ROI_MNI_V4.nii \
    --atlas-labels /absolute/path/to/ROI_MNI_V4.txt \
    --output out/analysis/encoding_roi_streams.csv
```

For fresh baseline fits, add `--baseline-map /absolute/path/to/layer-map.json`.
Each map entry declares a semantic feature name, model, layer index/count,
and normalized depth; [the example](../analysis/tomov23/docs/baseline-layer-map.example.json)
labels new FC1/HRR runs. The recovered EZ extractor has 15 named hooks, while
the published baseline CSV has 11 numeric layer labels with no recovered
mapping. This connector does not invent that mapping. The archived CSV remains
the source for those reported results until its producing inputs are identified.
ROI aggregation fails on unreadable files or inconsistent masks instead of
silently exporting partial results.
Its `fit_condition` column distinguishes main-only and nuisance fits from the
emitted performance `band`. Mixed fit conditions require explicit selection
with `--fit-condition main-only` or `with-nuisance` into separate CSVs.

```bash
python scripts/analysis/plot_encoding.py groups \
    --csv /absolute/path/to/master_encoding_data.csv --outdir out/figures
```

To plot just the fresh Qwen condition from the worked example:

```bash
python scripts/analysis/plot_encoding.py groups \
    --csv out/analysis/encoding_roi_streams.csv \
    --models qwen35_9b_sugmin --variant all --fit-condition with-nuisance \
    --outdir out/figures/fresh_qwen
```

This writes `encoding_groups_selected.pdf`. `--models` accepts exact CSV model
IDs in display order, including new baseline names such as `ddqn_fc1_rerun`
and `hrr_rerun`. Unknown names retain their full label and receive a stable
color. Partial tables support `aggregate`, `rois`, and `groups`; absent ROIs
remain empty. A single subject has no estimable across-subject SEM. Without
`--models`, a partial table uses its available models, while a complete
historical table retains the original figure sets and styles. Tables combining
main-only and nuisance-adjusted fits require an explicit `--fit-condition`.

This figure script summarizes encoding fits; it does not fit the model or
align fMRI. Follow the curated analysis pipeline for those stages and retain
its train/test folds, nuisance regressors, feature variant, layer, and subject
metadata when producing the table. Best-layer selection in the figure script
is preserved from the historical code and is not an independent validation
step.

A bounded CPU integration test exercises synthetic NIfTI preprocessing,
BSON/DDQN base alignment, multi-turn PT alignment, main/nuisance ridge fitting,
and ROI aggregation:

```bash
python -m pytest analysis/tomov23/tests -q
```

The [verification record](../analysis/tomov23/docs/verification-2026-09-27.json)
captures the test environment. A separate
[real EfficientZero trace check](../analysis/tomov23/docs/efficientzero-contract-check.json)
checks the recovered extractor's output against the base aligner. These checks
do not constitute a full human fMRIPrep or paper-statistics rerun.

## Validation and release boundaries

```bash
pip install -r requirements-dev.txt
ruff check src/ tests/ scripts/analysis/
ruff format --check src/ tests/ scripts/analysis/
python -m pytest tests/ -x -q
```

The extraction unit tests skip when optional PyTorch is absent; install it
to exercise those contracts. Tokenizer integration checks skip when the model
tokenizer is unavailable. CPU tests validate
conversation/window contracts, replay codecs, advancement, BSON outcomes,
and episode export. They do not establish that the full paper's fMRI fits or
large-model activations have been recomputed. The release inventory and
publication plan record the remaining data, licensing, and provenance gates.
The [dated validation summary](release/validation-2026-09-27.json) records the
completed source-tree checks and representative CPU runs without embedding
participant tables, scientific payloads or detailed storage metadata.
