# Analysis methods

The implementation follows the July 2026 voxelwise preprocessing, LLM alignment
and encoding methods, with base alignment from the March 2026 implementation.
[Source pins](sources/code-origins.json) identify that code; they do not identify the
producing invocation of every archived result. Use the
[dataset analysis guide](guides/dataset-analysis.md) for commands and the
[reproduction limits](reproduction-limits.md) for
outstanding artifact associations.

| Step | Selected implementation | Explicit convention |
| --- | --- | --- |
| Raw fMRI | OpenNeuro ds004323 1.0.0 | BIDS snapshot `17d00770c7b7885ed51dd42d9dee44909bd26c6c` |
| fMRIPrep | Original RC_RL launcher plus retained 24.1.0 TOMLs | Latest recorded config per subject; MNI152NLin2009cAsym 2 mm and anatomical space, BBR, DOF 9, dummy scans 0, master seed 23; this selection is not a producing-run attribution |
| Postprocessing | `reason_to_play.fmri.preprocess` | 8 mm smoothing, 1/128 Hz high-pass filter, linear detrending, available motion parameters/derivatives plus CSF and white matter, global AR(1) correction dropping volume 0, voxel z-score |
| Base alignment | `reason_to_play.fmri.align_baselines` | Intersect run masks; behavioral timestamps relative to scanner start, rounded to nearest TR; subtract one TR for AR(1), then concatenate by game/level/play |
| DDQN | `reason_to_play.features.ddqn` | 8-frame stack, original `GridDQN` architecture; SHA-256 checkpoint and exact source revision recorded for newly generated features |
| EMPA/HRR | Base aligner | Reads theory-regressor JSON and embeds symbolic theory sequences with the historical HRR settings; no new EMPA fitting is performed here |
| EfficientZero | `reason_to_play.fmri.align_efficientzero` | Original play/frame identities; four representation and seven initial value/policy hooks, averaged in the existing BOLD sample bins; optional timestep-averaged dynamics/reward hooks |
| LLM alignment | `reason_to_play.fmri.align_llm` | Original play/frame identities and wall-clock timestamps, average activations per TR, forward fill only inside a play; archived frame-indexed inputs retain their timing convention |
| Encoder | `reason_to_play.analysis.neural.encoding` | Past lags 2–5 TR, padding with the first row within each play, training-fold scaling/PCA before lags; outer and inner level-partition CV |
| Bands | Explicit CLI setting | Default **main only**; `--include-nuisance-bands` adds button, time, and game/level identity bands |
| Ridge | Submitted settings | Alpha grid `logspace(-5, 10, 20)`, random search, local alpha, alpha jitter, 100 iterations by default |
| Cohort comparison | Explicit `--max-level 8` | Restrict levels to 0–8 for comparisons; vgfmri3 participants also have levels 9–11 |

## Historical revisions are not interchangeable

The December 17, 2025 import was parcelwise and its lag function selected future
feature rows. January 28 introduced voxelwise AR(1) processing. The March clean
import and July submission are sibling Git branches: their preprocessing is
byte-identical upstream, but their encoders, default bands, and alignment
entrypoints differ. March's full aligner creates `aligned_data.npz`; July's
LLM-only aligner requires it. The December alpha grid also differs. These facts
do not establish which historical result files used which implementation.

The original RC_RL Docker launcher is included verbatim.
The corresponding S3 TOMLs report Singularity. Their processing parameters
agree for the 32 selected configurations; this does not prove the exact runtime
or container image digest that generated each derivative. The earliest sub-02
TOML used DOF 6; its later retained TOMLs use DOF 9. The old shell's `all` subject
list excluded 14/17/22 and a later edit ran only subject 22. Those launch-time
lists are **not** evidence of final analysis exclusions: all 32 subjects have
retained configurations.

## Input validation and execution

The pipeline uses the settings above with these input and execution contracts:

- Paths are explicit, bundled baseline sources are checksum-verified, and
  optional external checkouts are supported.
- The encoder loads the base aligner's separate EZ/HRR files and filters all
  aligned feature families together when restricting levels.
- LLM alignment derives run bounds from the base file's retained play lengths.
  Behavioral timestamps after the final acquired scan volume stay clipped, as
  they were in base alignment; unclipped plays keep the same timing and values.
- ROI exports record `fit_condition` from result metadata, including shuffled
  fits whose filenames omit the nuisance flag. Mixed main-only and nuisance
  inputs require explicit `--fit-condition` selection into separate CSVs;
  different statistical fits retain separate identities. The emitted `band=main`
  alone does not identify the fitted condition.
- Missing confounds and inconsistent voxel grids fail before changing the intended
  preprocessing; ROI aggregation rejects unreadable results and inconsistent masks.
- Missing input/feature/fold failures cannot be saved as zero-valued successful
  results. Bundles finish independent layers but return nonzero for failures.
- Result NPZs record completion, source hash, package versions, alpha grid,
  iteration count, and requested backend. `--resume` requires matching recorded
  settings and a completed public-run result. Use a new output directory when
  inputs change; resume verifies input file content hashes as well as settings.
- fMRIPrep command/config/image hashes and failure/completion status are saved.
- Missing DDQN checkpoints fail clearly; local checkpoint maps work without
  W&B. The historical W&B download mapping requires an explicit flag.

## Unresolved producing-run provenance

The historical encoder used unseeded ridge search and game/play shuffling; only
within-level shuffling used its supplied seed. Omitting `--seed` preserves that
behavior. For fresh runs, `--seed` controls both ridge search and all shuffle
scopes and is recorded with input/source hashes and dependency versions. This
defines a repeatable new fit, not the unknown historical random realization.
Identical historical scores cannot be promised. The submitted validity test is `feature_row.sum() != 0`, which can
also exclude nonzero vectors whose entries cancel. Changing that validity rule
or the fitting method defines a different analysis.

The exact DDQN checkpoint family for the indexed `model-ddqn-curriculum` features
is not established by the earlier `trial1-sequential` W&B mapping. New extraction
outputs must have a distinct model label until checkpoint identity is proven.
A synthetic CPU integration test checks BOLD/JSON/DDQN/PT alignment, nuisance
bands, sidecar loading, level filtering, and actual ridge fitting. It validates
software connections, not reproduction of human MRI processing or paper scores.

The ROI parser's `--baseline-map` assigns explicit metadata to named DDQN/EZ/HRR
features, using the same atlas and mask-intersection calculation. The example
map labels new FC1 and HRR runs; it does not identify the archived baseline CSV's
layers. The archived job configuration selects four representation and seven
value/policy hooks from the 15 extracted hooks, explaining the eleven-layer
selection. It does not establish the mapping to the CSV's numeric layer indices.
EfficientZero alignment from the released traces uses explicit hook names and
the existing BOLD sample order. It supports new fits without assigning those
arrays to unverified historical result labels.

## Dataset-only analysis contract

Human files preserve original play identities/outcomes, every recorded engine
state and timestamp, scanner timing, button states and structured events. Each
participant/game file embeds its conversation and trajectory across scanner
runs. The public pipeline reads JSON and has no BSON/PyMongo dependency.

The fixed study ROI mask is the intersection across all 32 source-subject masks,
not just the 21 participants in the headline encoding comparison. The
[ROI guide](guides/dataset-analysis.md#roi-aggregation-and-figures)
documents mask selection, validation and the archived-table comparison. Using
the 21-participant intersection changes some ROI values.

Preprocessing for hyperparameter selection is not fully nested: scaling/PCA are
fitted on the outer training set and reused across its inner alpha-selection
folds. The outer held-out partition is excluded from those fits. This retained
method should be distinguished from independent preprocessing in every inner
fold; changing it defines a new analysis.
