# Reconstructing fMRI derivatives from raw data

This optional workflow starts with OpenNeuro ds004323 v1.0.0. To analyze the
derivative dataset, use the [dataset analysis guide](dataset-analysis.md);
its processed inputs are sufficient for the supported analysis workflow.

Run the acquisition script from the checkout root. Install `.[analysis]`,
DataLad and git-annex, and provide a fMRIPrep 24.1.0 SIF, FreeSurfer license and
TemplateFlow cache. Use an explicit output root for each new reconstruction.

## Acquire the pinned source

```bash
bash reconstruction/tomov23/download_openneuro.sh /data/raw/ds004323-1.0.0 sub-13
```

The script pins Git snapshot `17d00770c7b7885ed51dd42d9dee44909bd26c6c` and
fetches the selected participant's MRI files. For behavioural alignment use the
released self-contained human JSON files, including their scanner clocks, and
the separate released EMPA regressor JSON. The public workflow has no BSON
reader or conversion dependency. The
[historical source reference](../sources/tomov23-behavior-notes.md)
documents the original archive for independent data archaeology.

## Run fMRIPrep

```bash
python -m reconstruction.tomov23.fmriprep --subject sub-13 \
  --bids-dir /data/raw/ds004323-1.0.0 \
  --output-dir /data/reconstruction/fmri/fmriprep \
  --work-dir /data/work/fmriprep/sub-13 \
  --templateflow-dir /data/templateflow \
  --fs-license /data/licenses/freesurfer.txt \
  --image /data/containers/fmriprep-24.1.0.sif --dry-run
```

Inspect the command, then remove `--dry-run` to execute it. Apptainer is the
default; `--runtime singularity` selects Singularity. The runner checks the
container version, records its hash and replays selected subject settings
explicitly. Retain HTML reports, logs, confounds and masks for quality review.

The recorded settings are fMRIPrep 24.1.0, MNI152NLin2009cAsym at 2 mm plus
anatomical space, BOLD-to-anatomical DOF 9, forced BBR, no discarded dummy scans,
no FreeSurfer reconstruction, all confound components, slice reference 0.5,
fixed skull-stripping seed and master seed 23. An earlier sub-02 configuration
used DOF 6; the runner selects the latest recorded configuration for each subject.

The [original launcher and 38 configuration records](../../reconstruction/tomov23/configs)
cover all 32 participants. The original launcher used Docker while the TOMLs
report Singularity. The original container digest and full cache are unknown;
this reconstruction does not promise byte-identical historical derivatives.
See [source pins](../sources/code-origins.json) and
the [research lineage](../sources/README.md).

The latest-per-participant configuration selection is a reconstruction policy;
it does not establish which invocation produced every archived derivative.
Historical launch-time subject lists also do not define the analysis cohort:
configuration records cover all 32 participants. The workflow uses directly
generated MNI-space BOLD and masks. Do not apply an additional T1w-to-MNI
resampling stage to those derivatives.

## Process BOLD and align baseline features

Set `DATASET` to the downloaded derivative dataset root, `DDQN_FEATURES` to the
selected per-frame DDQN NPZ directory and `EMPA_REGRESSORS` to the selected
`analysis/neural/inputs/theory-regressors.json.gz`. Choose those input paths from the release manifest.
The DDQN directory must contain `sub-XX/GAME/level-YY.npz`; the extractor writes
that structure beneath its `model-MODEL_ID/` directory.

```bash
python -m human.neural.process_bold --subject sub-13 \
  --fmriprep-dir /data/reconstruction/fmri/fmriprep \
  --output-dir /data/reconstruction/fmri/preprocessed
python -m analysis.neural.prepare_inputs --subject sub-13 \
  --preprocessed-dir /data/reconstruction/fmri/preprocessed \
  --model-features-dir "${DDQN_FEATURES:?Set DDQN_FEATURES to the selected per-frame model directory}" \
  --behavior-dir "${DATASET:?Set DATASET to the downloaded dataset root}/behavior/human" \
  --regressors-json "${EMPA_REGRESSORS:?Set EMPA_REGRESSORS to the theory-regressor JSON}" \
  --max-level 8 --output-dir /data/reconstruction/analysis-inputs
```

Postprocessing uses 8 mm spatial smoothing, 1/128 Hz high-pass filtering,
motion/CSF/white-matter regression, AR(1) pre-whitening with the first volume
dropped, and voxel z-scoring. Available runs are discovered; sub-09 has five
scanner runs. The worked LLM comparison uses the vgfmri4 cohort and levels 0–8.

The base archive contains BOLD, DDQN/theory features, nuisance variables and
sample boundaries. The aligner currently requires DDQN inputs even for later
LLM fits. Select the matching per-frame DDQN inputs explicitly. The command
writes `/data/reconstruction/analysis-inputs/sub-13/bold-ddqn-theory.npz` and any
configured feature sidecars; that base file can be passed to the encoder as
`--base-data`. Existing processed base archives avoid this reconstruction step.

To regenerate DDQN features with an explicit local checkpoint map:

```bash
python -m agents.ddqn.extract_features --subject sub-13 --run 1 \
  --behavior-dir "$DATASET/behavior/human" \
  --output-dir /data/reconstruction/features/ddqn \
  --checkpoint-map /data/checkpoints/ddqn.json --model-id ddqn-local-rerun
```

The checkpoint map is `{ "bait": { "0": "/absolute/path/to/weights.pt" } }`;
provide entries for all requested games/levels. This one-run example does not
produce a complete participant's features.
Use the [baseline guide](baselines.md) for the separately pinned
inference dependencies and EfficientZero traces. New DDQN features record
source and checkpoint hashes; the historical mapping alone does not
prove which checkpoint produced an archived feature family.

## Align LLM features and fit results

Continue with [new-model alignment](dataset-analysis.md#realigning-a-new-model)
and the encoding/ROI examples in the same guide. The imputation model and
extraction model are distinct. Preserve timestamps, original play identity,
feature stream and prompt metadata throughout.

Analysis methods, producing-run uncertainties and bounded validation
are documented in [reproduction limits](../reproduction-limits.md).
Validation does not include full fMRIPrep execution or full-scale GPU extraction.
