# Historical preprocessing archaeology

The first search, limited to **tomov23-analysis**, covered all 83 commits reachable from the four original branches
and submission tag, every historical code/document blob, deleted paths, and
all advertised remote refs (no additional pull-request refs were advertised).
The machine-readable search evidence is [historical-launcher-search.json](historical-launcher-search.json).
Unadvertised deleted refs or local-only commits are outside this search.

The broader project search then recovered the actual raw preprocessing launcher:
[`RC_RL/fmriprep_pipeline/run_fmriprep.sh`](https://github.com/botcs/RC_RL/blob/969141d09369ecd783965fadbbe2a1920065cf24/fmriprep_pipeline/run_fmriprep.sh),
committed **16 October 2025**. It and its README are
[preserved with checksums](../provenance/fmriprep/original-launcher/source.json).
The portable public entrypoint is `scripts/run_fmriprep.py`; the original shell
is historical evidence and retains old local paths and launch-time subject lists.

Within tomov23-analysis, the older files were:

| Historical path | Added | Deleted | What it did |
| --- | --- | --- | --- |
| `docker/register_bold_to_mni/Dockerfile` | `b7f640af7de728a38056649b857c2e79aaade3f9` | `3bc48eeea10afdcc78aa312b8d1c5631f1f43ee1` | Built from `nipreps/fmriprep:24.1.0` to supply ANTs and templates |
| `scripts/register_bold_to_mni.sh` | `b7f640af7de728a38056649b857c2e79aaade3f9` | `3bc48eeea10afdcc78aa312b8d1c5631f1f43ee1` | Applied `antsApplyTransforms -d 3 -e 3` to existing T1w-space preprocessed BOLD using the existing T1w-to-MNI transform |
| `scripts/upload_fmriprep_to_s3.py` | `5c53c931fce464c5f708e3f416db4330be434fa8` | `578d7e6` | Uploaded already-produced local fMRIPrep outputs |
| `encoding_model_slurm/stages/preprocess.py` | `26fa74c` | retained in archived working branch | Post-fMRIPrep smoothing, confounds, AR(1), and z-scoring |

The old registration stage's inputs were
`fmriprep/sub-NN/func/*_space-T1w_desc-preproc_bold.nii.gz` and
`fmriprep/sub-NN/anat/*_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5`.
It did not run fMRIPrep on raw BIDS data. Its output was a separate
`register_bold_to_mni/` prefix. It is superseded by the directly generated
MNI BOLD and masks in the current `fmriprep/` derivative prefix, so adding it
to the active pipeline would introduce an unnecessary second resampling path.
The archived branch tags preserve the code and history without making it an
active stage.

All original branch histories share root `5c53c93`, whose commit message is
`Local fmriprep upload (skip stage 1)`. The initial README described a proposed
raw fMRIPrep stage; the initial tracked tree contained an upload script rather
than that proposed launcher. No raw-to-fMRIPrep invocation was found in that repository's reachable
committed code; the raw launcher lives in **RC_RL**. The recovered S3 TOMLs
independently record the scientific settings, and the public runner replays them
with explicit CLI overrides for fMRIPrep 24.1.0 defaults.

The original launcher is Docker-based, whereas retained configurations identify
Singularity. The selected TOMLs agree with the launcher's scientific flags, but
no original image digest or complete producing-run attribution has been recovered.
The TOMLs' latest-per-subject selection is an explicit replay policy, not a claim
that the latest invocation produced every object in the derivative bucket.
