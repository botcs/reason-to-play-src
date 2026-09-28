# Analysis from the released dataset

The analysis input is a local copy of the derivative dataset. Behavioral
analysis, encoding fits and ROI aggregation use its canonical human recordings,
processed BOLD, features and atlas. They do not require the OpenNeuro BSON dump,
raw MRI, AWS credentials or a model API. The dataset is still being prepared;
use a verified dataset commit when it is published. See the
[workflow overview](../reproducibility.md)
for generating new gameplay or features.

## Install

```bash
python -m pip install -e '.[analysis]'
```

Install a hardware-appropriate PyTorch build only when reading `.pt` features
or using a Torch fitting backend. Canonical human recordings and NPZ encoding
inputs use the public JSON reader and have no BSON/PyMongo dependency.
The Python modules below can be called from any working directory after install.
Relative paths to `experiments/` in the plotting examples are relative to the
code checkout; supply absolute configuration paths from another directory.

Set `DATASET` to your downloaded dataset root. The examples use the paths
listed below; other models and conditions are indexed in `manifest.jsonl.gz`.
Output paths are local directories you choose.

```bash
DATASET=/data/reason-to-play
```

## Human behaviour files

`behavior/human/sub-XX/GAME/CONDITION.human.replay.json.gz` contains one
participant's plays of one game across levels, attempts and scanner runs. Each
file embeds its trajectory, exact prompts, original play identities and scanner
clocks. The three prompt conditions are `elaborate`, `minimal` and `oracle`;
readers select `elaborate` by default to count each measured trajectory once.
One file is sufficient for that participant/game: no external human records or
definitions are required. The reader returns the recorded fields as
ordinary Python dictionaries:

```python
from human.behavior import iter_plays, load_runs, play_states

runs = load_runs("/data/reason-to-play/behavior/human")
for play in iter_plays("/data/reason-to-play/behavior/human", subject="sub-13"):
    frames = play_states(play)
    scanner_start = runs[(int(play["subj_id"]), int(play["run_id"]))]["scan_start_ts"]
    observation_times = [frame["ts"] - scanner_start for frame in frames]
```

A frame is one recorded engine state from the source `zstates` sequence, including
the initial state and states with no action. Its engine tick resets for each play.
Original wall-clock timestamps are retained without assuming a fixed frame rate;
no states are synthesized during gaps between plays.

Original IDs, nullable outcomes, per-frame timestamps, engine ticks, structured
keys/events and plays with no selected actions remain available. Recorded
rendering rectangles retain fractional grid positions; original logical pixel
positions are also preserved for baseline model inputs. Do not infer participant
outcomes from the terminal flags of a display replay.

Participant 11/run 05 has no separate source scanner record. Its start time
is derived from the nine play records, which all agree on that time. The
embedded scanner entry identifies how the time was derived. No additional
scanner file is needed by the reader. Missing or conflicting clocks are errors.
See the [data format guide](../data-format.md) for field meanings and the
[higher-detail source reference](../sources/tomov23-behavior-notes.md)
for the historical archive. That reference is not the public replay schema.

## Behavioral analysis

```bash
python -m analysis.behavioral.episodes \
  --human-data "$DATASET/behavior/human" \
  --replays "$DATASET/behavior/lrm" --workers 32 \
  --ddqn "$DATASET/behavior/ddqn/episode-history.json" \
  --efficientzero "$DATASET/behavior/efficientzero" \
  --empa "$DATASET/behavior/empa" \
  --output /results/episodes.csv
python -m analysis.behavioral.plots discovery_curriculum_combined \
  --csv /results/episodes.csv --output-dir /results/figures
```

The replay argument accepts a directory or explicit files; directories are
searched recursively. Baseline input formats are described in the [baseline guide](baselines.md). Human keypress
frames, model decisions and engine frames are separate columns. Practice runs
and levels outside 0–8 are excluded by the comparison exporter. Participant
advancement used a fixed scanner schedule; the model curriculum required
consecutive wins. The plotting rule does not make those collection protocols
identical.

The dataset also includes `analysis/behavioral/retained-episode-table.csv`,
the input table for the retained behavioral summaries. Its DDQN rows use sampled
W&B histories: 32,818 episodes from 88 runs. The separate DDQN history JSON
contains all 170,546 recorded episodes from those same runs. Use the table to
render its summaries and the full histories for analyses requiring every episode;
the two are different analysis inputs.

```bash
python -m analysis.behavioral.plots discovery_curriculum_combined \
  --csv "$DATASET/analysis/behavioral/retained-episode-table.csv" \
  --output-dir /results/retained-behavior-figures
```

## Neural encoding from processed inputs

An encoding fit needs processed BOLD, its mask/affine, sample/play boundaries,
CV partitions, nuisance variables, and the selected model features in exactly
that sample order. Existing base archives also contain DDQN and theory
features; they are shared analysis inputs, not fMRIPrep images. Feature files
can be supplied explicitly. `all` selects all prompt action steps; it does not
describe file compression. The following paths select participant 13 and the
Qwen3.5-9B elaborate/all/main condition. The layer key is stored inside that
feature archive.

```bash
BASE_DATA="$DATASET/analysis/neural/inputs/sub-13/bold-ddqn-theory.npz"
MODEL_FEATURES="$DATASET/analysis/neural/inputs/model-features/lrm/qwen3.5-9b/elaborate/all/main/sub-13.npz"
```

```bash
python -m analysis.neural.encoding \
  --subject sub-13 \
  --base-data "${BASE_DATA:?Set BASE_DATA to the downloaded sub-13 base NPZ}" \
  --feature-file "${MODEL_FEATURES:?Set MODEL_FEATURES to the matching feature NPZ}" \
  --layer llm_qwen35_9b__all__main_layer_1 \
  --max-level 8 --include-nuisance-bands --seed 23 \
  --output-dir /results/encoding
```

Download the adjacent `.npz.alignment.json` file with each separate model-feature
archive. It binds the feature bytes to the BOLD archive and its sample order;
the reader checks those identities before fitting. New alignment outputs embed
the same identities. An archived input without this evidence requires the
explicit `--allow-unverified-alignment` option and is recorded as unverified.
Verified alignment does not imply that features exist for every play: missing
source inputs and their zero-filled rows remain reported separately.
The result's `alignment_verification_json` retains each association's
`feature_coverage`, including missing play IDs and sample intervals; null means
coverage was not supplied. Intervals use the original base archive's sample
order, before level selection or shuffling. The established fit excludes rows
whose feature-vector sum is zero after those operations; it does not use the
coverage declaration to change that mask. `valid_sample_policy` and
`n_valid_volumes` record this numerical rule and its resulting sample count.
An all-zero row is not evidence by itself that its source observation is missing.

The example's seed defines a fresh repeatable fit. The historical fitting seed
was not recorded, so it does not claim to recover the original random search.
The result records input content hashes, source hash, dependency versions and
fitting settings. `--resume` checks those identities. Use a separate output
root for a different experimental condition.

For EfficientZero, select the participant's archive and a named hook:

```bash
python -m analysis.neural.encoding \
  --subject sub-13 --base-data "${BASE_DATA:?Set BASE_DATA}" \
  --feature-file "$DATASET/analysis/neural/inputs/model-features/efficientzero/sub-13.npz" \
  --layer ez_value_policy_fc_policy_0 \
  --max-level 8 --include-nuisance-bands --seed 23 \
  --output-dir /results/encoding-ez
```

The archive uses full hook names and records coverage separately for each
layer. See the [baseline guide](baselines.md#efficientzero-hidden-features) to
align traces for a new dataset.

The model/variant, layer and participant selection for the headline comparison
is [experiments/neurips2026/analysis/encoding.json](../../experiments/neurips2026/analysis/encoding.json).
Qwen uses `all`; DeepSeek uses `compressed`. Feature stream, performance band
and nuisance-fit condition are separate dimensions. Missing data are not zero
performance, and a partial model/cohort must not silently become a complete
comparison.

## ROI aggregation and figures

Use the included AAL-SPM12 image, label table and the intersection mask from
all 32 participants:

```bash
ATLAS="$DATASET/analysis/neural/inputs/atlas/aal-spm12/ROI_MNI_V4.nii"
ATLAS_LABELS="$DATASET/analysis/neural/inputs/atlas/aal-spm12/ROI_MNI_V4.txt"
COMMON_MASK="$DATASET/analysis/neural/inputs/atlas/common-mask.npz"
```

Verify their manifest hashes rather than downloading another atlas version.

```bash
python -m analysis.neural.roi \
  --results-dir /results/encoding --workers 32 \
  --fit-condition with-nuisance \
  --atlas "${ATLAS:?Set ATLAS to the released AAL-SPM12 image}" \
  --atlas-labels "${ATLAS_LABELS:?Set ATLAS_LABELS to its label table}" \
  --common-mask "${COMMON_MASK:?Set COMMON_MASK to the released study mask}" \
  --output /results/encoding_roi.csv
```

The atlas image and label table are release inputs, with upstream terms and
hashes. ROI voxel selection also depends on the common brain mask: the mask
cohort must be stated separately from the cohort selected for the final plot.
The supplied mask is derived from subjects 01–32. Selecting encoding results
for subjects 12–32, or a smaller demonstration, retains that fixed voxel set.
The parser checks that it is contained in every selected subject's mask. A
representative archived-result comparison reproduces 6,426 ROI rows within
absolute error `1e-8` with this mask; the original aggregation command remains
unrecorded.

For a new experiment, omit `--common-mask` to intersect the selected result
subjects and save the inferred mask beside the CSV. That changes the analysis
support. Each run writes `encoding_roi.csv.reproducibility.json` with the
atlas, labels and common-mask hashes, the mask cohort, the result cohort and
fit/feature selections.

For a completed whole-cohort, whole-layer run:

```bash
python -m analysis.neural.plots groups \
  --csv /results/encoding_roi.csv \
  --selection-config experiments/neurips2026/analysis/encoding.json \
  --outdir /results/figures
```

For a new experiment, provide your own model/variant/subject/layer selection,
or explicit `--models`, `--variant`, `--stream`, `--subjects` and
`--fit-condition` arguments. The plotter rejects unintended mixed conditions
and incomplete declared coverage. Historical tables without fit metadata must
retain that uncertainty unless an evidence-backed condition declaration is
available.

To render the archived summary table, set `ARCHIVED_TABLE` to the downloaded
`analysis/neural/results/master_encoding_data.csv` and use its separate selection:

```bash
python -m analysis.neural.plots groups \
  --csv "${ARCHIVED_TABLE:?Set ARCHIVED_TABLE to master_encoding_data.csv}" \
  --selection-config experiments/neurips2026/analysis/archived-encoding-table.json \
  --outdir /results/archived-figures
```

This selection checks the table's SHA-256, keeps its missing fit metadata
unknown, and selects the unlabeled baseline streams without renaming them.
It acknowledges exactly two absent participant/layer cells and requires the
remaining declared coverage. It reproduces summaries of the supplied table;
it does not establish the missing upstream fit settings or baseline hook map.

## Realigning a new model

Realignment uses the human files and the processed base archive selected above.
Set `LLM_FEATURES` to the extraction directory containing
`sub-XX/GAME.pt`; for the generic extractor this is its
`model-MODEL/{all|compressed}/` directory. This is a different input from the
already sampled feature NPZ used in the encoding example.

```bash
python -m analysis.neural.align_llm \
  --subject sub-13 \
  --behavior-dir "$DATASET/behavior/human" \
  --aligned-data "${BASE_DATA:?Set BASE_DATA to the downloaded sub-13 base NPZ}" \
  --llm-source "name=my_model__all__main,dir=${LLM_FEATURES:?Set LLM_FEATURES to the per-game PT directory},stream=main" \
  --output-dir /results/aligned
```

Each feature must retain its original observation timestamp and play identity.
Alignment uses all frame timestamps plus scanner start time, clips to retained
scan volumes, and applies the recorded AR(1) offset. A feature row number is
not a scanner-volume index. Recorded theory-regressor sequences are separate model
data; [base alignment](fmri-preprocessing.md#process-bold-and-align-baseline-features)
accepts their explicit JSON file. Output NPZs are written under
`/results/aligned/sub-13/`; pass the required file with `--feature-file` when fitting.

## Verification boundary

The test suite exercises timing and nuisance alignment, ridge fitting, ROI
aggregation, input identity and coverage checks. Independent source comparisons
cover measured frames and representative model inputs. For the real sub-13
helper file, 20 plays and 12,775 frames matched DDQN grids, EfficientZero input
images and fMRI timing/nuisance inputs. The 6,426-row ROI check above starts from
saved encoding results; it does not refit their models.

These checks do not establish exact historical weight revisions, recover
unrecorded random seeds, or supply absent EfficientZero aligned inputs and the
unknown historical layer mapping. The [reproduction limits](../reproduction-limits.md)
identifies those remaining limits. Original raw-data preprocessing is an
optional reconstruction workflow in the [fMRI guide](fmri-preprocessing.md).
