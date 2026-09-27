# Neural Encoding Pipeline

Maps LLM hidden-state features (and other feature sources) to voxelwise
fMRI BOLD responses using banded ridge regression with leave-one-partition-out
cross-validation.

## Pipeline stages

The scripts are meant to be run in order:

Start with the [full reproduction guide](../README.md) for the pinned raw
OpenNeuro acquisition, recovered fMRIPrep configurations, behavioral BSON
preparation, and the restored `align_base.py` dependency.

### 1. preprocess.py -- fMRI preprocessing

Applies SPM12-style preprocessing to fMRIPrep output:
spatial smoothing (8 mm FWHM), high-pass filtering (1/128 Hz),
confound regression (motion + CSF + white matter), AR(1) pre-whitening,
and z-scoring.  Produces per-subject per-run NPZ files with masked
voxel timeseries.

### 2. align.py -- Feature-to-BOLD alignment

Requires `aligned_data.npz` from `align_base.py` (or the corresponding indexed
derivative). This script does not create the base BOLD/behavioral alignment.
Aligns LLM hidden-state features to the fMRI timecourse. Handles the
mapping from game events (variable-length plays across levels) to TR
indices, respecting the exact stimulus timing from the behavioral logs.
Outputs per-subject NPZ files with feature matrices aligned to BOLD
volumes.

### 3. encoding_model.py -- Banded ridge regression

Fits a voxelwise encoding model with separate regularization bands for:
- Band 1: main features (LLM layer activations, DDQN, or HRR)
- Optional bands, all enabled by `--include-nuisance-bands`: game + level
  identity, button presses, and time regressors. Default: main features only.

Features are lagged 2--5 TRs into the past to capture HRF delay.
PCA is applied before lagging when the raw feature dimension is large
relative to training samples.  Evaluation uses leave-one-partition-out
CV following the protocol of Tomov et al. (2023).

### 4. load_and_parse.py -- Result aggregation

Parses per-subject encoding results into a long-format CSV with columns
for model, variant, stream, layer depth, ROI-level performance, and
subject.  ROIs are defined by an explicitly supplied AAL atlas and label file.
The `fit_condition` column records `main-only` or `with-nuisance` from the NPZ
metadata, independently of the emitted `band`. When both fit conditions are
present, select `--fit-condition main-only` or `--fit-condition with-nuisance`
and produce separate CSVs; the parser refuses to silently deduplicate them.

## Data dependencies

- `aal/ROI_MNI_V4.nii` + `ROI_MNI_V4.txt` -- AAL atlas for ROI-level
  aggregation of voxelwise encoding performance
- fMRIPrep output (input to `preprocess.py`)
- Behavioral BSON files from Tomov et al. (2023) (input to `align.py`)
- LLM hidden-state `.pt` files from the feature extraction pipeline
  (input to `align.py`)

## Verification and revision limits

Run `python -m pytest tests/` from the analysis root (the parent of
`encoding_model_code/`) with the analysis root's
requirements installed. The synthetic test runs the real NIfTI-to-ridge boundary
without downloading a dataset or calling an API. Historical settings, source
pins, and known producing-run gaps are documented in
[scientific-conventions.md](../docs/scientific-conventions.md).
