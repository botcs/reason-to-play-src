# Scientific conventions and revision boundaries

This public integration uses the submitted July 2026 voxelwise preprocessing,
LLM alignment, and encoding implementation, plus the missing base alignment
from the working branch. It does **not** establish which revision produced each
published result file. That requires the result manifest's code, configuration,
and artifact links. Exact source pins are in [source-map.json](source-map.json).

| Step | Selected implementation | Explicit convention |
| --- | --- | --- |
| Raw fMRI | OpenNeuro ds004323 1.0.0 | BIDS snapshot `17d00770c7b7885ed51dd42d9dee44909bd26c6c` |
| fMRIPrep | Original RC_RL launcher plus retained 24.1.0 TOMLs | Latest recorded config per subject; MNI152NLin2009cAsym 2 mm and anatomical space, BBR, DOF 9, dummy scans 0, master seed 23; this selection is not a producing-run attribution |
| Postprocessing | Submitted `preprocess.py` | 8 mm smoothing, 1/128 Hz high-pass filter, linear detrending, available motion parameters/derivatives plus CSF and white matter, global AR(1) correction dropping volume 0, voxel z-score |
| Base alignment | Recovered `align_base.py` | Intersect run masks; behavioral timestamps relative to scanner start, rounded to nearest TR; subtract one TR for AR(1), then concatenate by game/level/play |
| DDQN | Recovered per-state extractor | 8-frame stack, original `GridDQN` architecture; SHA-256 checkpoint and exact source revision recorded for newly generated features |
| EMPA/HRR | Recovered base aligner | Reads `regressors.bson` and embeds symbolic theory sequences with the historical HRR settings; no new EMPA fitting is performed here |
| EfficientZero | Recovered base aligner | Representation entries, initial-phase value/policy entries, timestep-averaged dynamics/reward entries; separate `aligned_ez.npz` |
| LLM alignment | Submitted `align.py` | Multi-turn wall-clock timestamps, average activations per TR, forward fill only inside a play; legacy format uses state indices |
| Encoder | Submitted `encoding_model.py` | Past lags 2–5 TR, padding with the first row within each play, training-fold scaling/PCA before lags; outer and inner level-partition CV |
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

The original RC_RL Docker launcher has been recovered and retained verbatim.
The corresponding S3 TOMLs report Singularity. Their scientific parameters
agree for the 32 selected configurations; this does not prove the exact runtime
or container image digest that generated each derivative. The earliest sub-02
TOML used DOF 6; its later retained TOMLs use DOF 9. The old shell's `all` subject
list excluded 14/17/22 and a later edit ran only subject 22. Those launch-time
lists are **not** evidence of final analysis exclusions: all 32 subjects have
retained configurations.

## Integration changes made in the public repository

Successful numerical calculations retain the submitted settings above. The
public integration repairs execution and data boundaries:

- Portable paths and checksum-verified bundled baseline sources replace private
  checkout paths; optional external checkouts remain supported.
- The encoder loads the base aligner's separate EZ/HRR files and filters all
  aligned feature families together when restricting levels.
- LLM alignment derives run bounds from the base file's retained play lengths.
  Behavioral timestamps after the final acquired scan volume stay clipped, as
  they were in base alignment; unclipped plays keep the same timing and values.
- ROI exports record `fit_condition` from result metadata, including shuffled
  fits whose filenames omit the nuisance flag. Mixed main-only and nuisance
  inputs require explicit `--fit-condition` selection into separate CSVs;
  different statistical fits can no longer be silently deduplicated. Model
  identities, ROI averaging, band scores, and within-fit layer selection stay
  unchanged. The emitted `band=main` alone does not identify the fitted condition.
- Missing confounds and inconsistent voxel grids fail before changing the intended
  preprocessing; ROI aggregation rejects unreadable results and inconsistent masks.
- Missing input/feature/fold failures cannot be saved as zero-valued successful
  results. Bundles finish independent layers but return nonzero for failures.
- New result NPZs record completion, source hash, package versions, alpha grid,
  iteration count, and requested backend. `--resume` requires matching recorded
  settings and a completed public-run result. Use a new output directory when
  inputs change; the resume check does not hash large input arrays.
- fMRIPrep command/config/image hashes and failure/completion status are saved.
- Missing DDQN checkpoints fail clearly; local checkpoint maps work without
  W&B. The historical W&B download mapping requires an explicit flag.

## Unresolved producing-run provenance

The preserved encoder's ridge search is unseeded. Its game/play shuffle functions
ignore their `seed` argument, while within-level shuffling uses it. The public
integration documents these behaviors instead of silently selecting a new
control analysis. Identical scores or historical shuffle realizations cannot be
promised. The submitted validity test is `feature_row.sum() != 0`, which can
also exclude nonzero vectors whose entries cancel; it is preserved here for
revision fidelity. These are review items for future scientific changes.

The exact DDQN checkpoint family for the indexed `model-ddqn-curriculum` features
is not established by the earlier `trial1-sequential` W&B mapping. New extraction
outputs must have a distinct model label until checkpoint identity is proven.
A synthetic CPU integration test checks BOLD/BSON/DDQN/PT alignment, nuisance
bands, sidecar loading, level filtering, and actual ridge fitting. It validates
software connections, not reproduction of human MRI processing or paper scores.

The submitted ROI parser supported only LLM filenames. Public `--baseline-map`
adds explicit metadata for named DDQN/EZ/HRR features, using the same atlas and
mask-intersection calculation. The example map labels new FC1 and HRR runs; it
is not a recovered map for the archived baseline CSV. That CSV has 11 numeric
EZ layer indices, but the recovered traces have 15 named hooks and neither its
original baseline parser nor its aligned EZ inputs have been found. An inspected
`sub-12/aligned_data.npz` has `has_ez_data=False`; the current aligned prefix
contains no `aligned_ez.npz`. These are unresolved source links, not permission
to infer a historical layer mapping.
