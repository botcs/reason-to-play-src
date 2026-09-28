# Agent guide

Reason to Play supports model gameplay and human-to-model neural comparisons.
Keep reusable implementations configurable; put study-specific choices in
explicit experiment configuration. Read this guide before changing code.

## Start here

- [README.md](README.md): installation, quickstart and project overview.
- [Reproduction guide](docs/reproducibility.md): workflow order, commands,
  inputs, outputs and verification limits.
- [Analysis methods](docs/analysis-methods.md):
  retained preprocessing, alignment and encoding behavior.
- [Data format](docs/data-format.md): human-file schema, frame clocks and
  measurement identities.
- [Historical behavioural source reference](docs/sources/tomov23-behavior-notes.md):
  pinned upstream BSON documentation, lineage and corrections.
- [Sources](docs/sources/README.md): preceding research code, game translation
  and original source references.
- [Reproduction limits](docs/reproduction-limits.md): known missing
  artifacts and unresolved checkpoint/layer attribution.
- [THIRD_PARTY.md](THIRD_PARTY.md): upstream licenses and source boundaries.
- [CITATION.cff](CITATION.cff): canonical NeurIPS 2026 paper citation.

## Find the implementation

| Task | Location |
| --- | --- |
| Model gameplay | `agents/lrm/gameplay/` |
| Human replay and rationale imputation | `agents/lrm/prepare_prompts.py`, `agents/lrm/prompting/replay_agent.py` |
| LLM activation extraction | `agents/lrm/features/extract*.py` |
| Harness and observations | `agents/lrm/prompting/` |
| Model adapters and inference runtimes | `agents/lrm/backends/` |
| Game engine, game definitions and prompts | `environments/vgdl/`, `environments/definitions/`, `agents/lrm/prompts/` |
| Replay inspection and interactive play | [Webapp](https://botcs.github.io/reason-to-play/) |
| Raw-data acquisition and fMRIPrep | `reconstruction/tomov23/download_openneuro.sh`, `reconstruction/tomov23/fmriprep.py` |
| Human recordings, outcomes and clocks | `human/behavior.py` |
| Replay format, tagged JSON values and neural input associations | `data/` |
| BOLD processing | `human/neural/process_bold.py` |
| Features sampled at scanner times | `analysis/neural/prepare_inputs.py`, `align_llm.py`, `align_efficientzero.py` |
| Encoding fits, ROI aggregation and neural figures | `analysis/neural/` |
| Behavioral episode tables and figures | `analysis/behavioral/` |
| DDQN activation extraction | `agents/ddqn/extract_features.py` |
| EfficientZero activation and trace extraction | `agents/efficientzero/extract_features.py`, `extract_traces.py`; model inputs in `observations.py` |
| EfficientZero alignment to existing BOLD samples | `analysis/neural/align_efficientzero.py` |
| EfficientZero runtime and environment | `agents/efficientzero/inference/`, `environment/` |
| Optional EfficientZero training | `agents/efficientzero/training/` submodule; `prepare_training_config.py` creates its local experiment config |
| Study settings | `experiments/neurips2026/` |
| Agent implementations and source pins | `agents/ddqn/`, `agents/efficientzero/`, `agents/empa/` |
| Dataset inventory, selection and catalogue preparation | `scripts/release/` |

Read the relevant module and its tests before editing. Update this map when
moving an implementation; do not leave a second implementation at the old path.

## Preserve data and analysis conventions

- Keep structural changes separate from changes to analysis methods or study
  settings. Tests passing does not establish reproduction of the paper results.
- Ordinary analysis reads the derivative dataset alone. Keep raw-MRI
  reconstruction optional. The public pipeline reads JSON behaviour; do not
  add BSON/PyMongo dependencies. Each human file is one self-contained
  participant/game JSON spanning levels, attempts and scanner runs, with its
  trajectory and exact prompts together. Preserve original play/run identities,
  frame timestamps and the input/event fields needed by analyses. Do not require
  external prompt-to-trajectory joins or dynamic reconstruction merely to avoid
  repetition. Human-data readers select one prompt condition by default to
  avoid counting repeated behaviour three times.
- The study ROI mask is the intersection across all 32 source participants,
  including when fitting/plotting a smaller cohort. Pass the released mask
  explicitly; recomputing the intersection from available result files can change
  ROI scores.
- Human play outcomes are three-valued. Check avatar-death events before
  classifying a null `win` as incomplete; terminal zstate `win=-1` is not an
  outcome. Preserve practice/cohort flags and comparison denominators.
- Engine frames, model decisions and scanner TRs are distinct units. Preserve
  timestamps, play/run identifiers, lag boundaries and alignment metadata.
  Each participant has separate BOLD, samples, nuisance and model-feature files.
  Model features and nuisance variables must bind to both exact BOLD and samples
  bytes, plus the ordered-sample digest, through adjacent `.npz.alignment.json`
  records. BOLD also binds to its samples file. Row counts alone do not establish
  correspondence.
  Keep missing-feature coverage separate from a verified alignment.
- Continued model runs may embed their preceding gameplay and restart attempt
  numbers. Use chronological frame boundaries and action `state_index` references
  to distinguish attempts; `(level, attempt)` alone is not a unique identity.
- Preserve fractional replay positions from the original rendering rectangles.
  Human replay visuals use the recorded frame's rectangle coordinates; prompt
  alignment and model input quantization are separate operations.
  Do not replace recorded visuals with rounded observation grids or simulated
  positions. Validate these changes independently of prompt/feature contents.
- Each replay has one authoritative `game_description`: the translated VGDL
  definition shared by the Colas-based Python engine and browser interpreter.
  Preserve that translation. Do not add a second source-game definition, rule
  patches, or an alternative game-description format to the released replay.
  Historical source rules belong only to explicit source-conversion checks.
- Imputed model reasoning is not a measurement of a participant's thoughts.
  Keep imputation-model and feature-extraction-model provenance separate.
- Preserve preprocessing flags, folds, within-fold transforms, nuisance bands,
  feature streams and temporal-window assignments during refactoring. Consult
  the analysis methods before modifying those contracts.
- Distinct baseline engine revisions and inference runtimes are intentional.
  Preserve their isolation, licenses and provenance manifests; do not deduplicate
  snapshots based on matching filenames or merge their import paths. EfficientZero
  training is an optional upstream submodule pinned to
  `29157d4892afd9467b1bd0994de1355086145490`; analysis and feature extraction must
  work without initializing it. Use direct Python modules for extraction and
  upstream `python -m ez.train` for training; see the [baseline guide](docs/guides/baselines.md).

## Implementation and documentation

- Keep input/output roots explicit and resource resolution independent of the
  caller's working directory. Do not introduce machine-specific absolute paths.
- Keep optional engine, GPU, fMRI and cloud dependencies out of unrelated import
  paths. Imports should not launch jobs or change process-wide settings.
- The webapp is the interactive viewer and player. Keep Python environments
  headless; numerical observations and offscreen image generation support model
  inputs and must not open desktop windows. Do not add a second viewer.
- Expose ordinary Python functions and directly executable Python modules or
  scripts. Do not add a custom command facade or a central command dispatcher.
  Add abstractions only when they solve a concrete implementation problem.
  Avoid adding dependencies from library code onto standalone command scripts.
- Treat code paths and dataset artifact paths as separate contracts. Moving
  source files does not change source-object identities or historical hashes.
- Maintain commands in the workflow guides. README, website and dataset-card
  references should lead to those guides; use `CITATION.cff` for citation data.
- Use concrete names: recorded frames, play IDs, scanner times, model inputs
  and analysis results. Public documentation describes the data and supported
  workflows. Keep release-preparation notes in the development repository.

## Verification

Run from the repository root in an environment with the relevant dependencies.
The Ruff commands below cover the existing configured lint scope; the pytest
command includes both the main and neural suites:

```sh
ruff check agents/ human/ data/ analysis/ environments/ reconstruction/ tests/ scripts/
ruff format --check agents/ human/ data/ analysis/ environments/ reconstruction/ tests/ scripts/
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/ -x -q
```

Set `REASON_TO_PLAY_HUMAN_DATA` to a downloaded human JSON file or human-data
directory to include the recorded-trajectory integration tests. Without that
input, those tests skip explicitly.

The neural suite is included under `tests/neural/`. For a localized change, first run the relevant tests. For command or
resource changes, also exercise the actual entry point and resolve its inputs
from outside the checkout. Check relative documentation links after file moves.
Documentation-only edits need link/content checks rather than new unit tests.

Report what ran, what was skipped and why. Bounded CPU fixtures do not validate
full fMRIPrep, distributed model extraction or complete paper statistics.
