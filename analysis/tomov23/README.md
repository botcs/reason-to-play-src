# Reason to Play: fMRI preprocessing and neural encoding

> Integrated sources from `botcs/tomov23-analysis` and the original
> `botcs/RC_RL` preprocessing launcher. Exact upstream revisions are in
> [source-map.json](docs/source-map.json); local changes and selected scientific
> settings are in [scientific-conventions.md](docs/scientific-conventions.md). See the public
> [integration guide](../../docs/reproducibility.md) for the exact LLM-feature
> paths connecting this code to the rest of this checkout.

This repository restores the preprocessing dependencies of the submitted
`encoding_model_code/` release. The original submission remains available at
tag `encoding-suppmat-v1.0`. See [branch provenance](docs/branch-archive.json)
for the exact sources recovered from the working branches.

The public release integrates these scripts with LLM gameplay, human replay,
and latent extraction in [reason-to-play-src](https://github.com/botcs/reason-to-play-src).

## Verified provenance and limits

The raw input is [OpenNeuro ds004323 version 1.0.0](https://openneuro.org/datasets/ds004323/versions/1.0.0),
Git snapshot `17d00770c7b7885ed51dd42d9dee44909bd26c6c`, licensed CC0.
The original raw-data launcher was recovered in `RC_RL/fmriprep_pipeline`
(commit `969141d09369ecd783965fadbbe2a1920065cf24`, 16 October 2025) and is
[preserved verbatim](provenance/fmriprep/original-launcher). The deleted
registration stage in tomov23-analysis operated on existing fMRIPrep outputs;
see [the archaeology record](docs/historical-preprocessing.md). The retained
derivative-bucket TOMLs independently record the preprocessing settings:

- fMRIPrep **24.1.0**, MNI152NLin2009cAsym at 2 mm plus anatomical space.
  The original launcher uses Docker; the retained TOMLs report Singularity.
  Their scientific flags agree, but exact producing invocations remain unresolved.
- 38 original TOMLs across 32 subjects, alongside the original derivative
  description and citation boilerplate, are in [provenance/fmriprep](provenance/fmriprep).
- Every latest recorded subject configuration uses BOLD-to-anatomical DOF 9,
  forced BBR, no discarded dummy scans, no FreeSurfer surface reconstruction,
  all confound components, slice reference 0.5, fixed skull stripping seed,
  and master seed 23. One earlier sub-02 invocation used DOF 6. All 38 files
  remain available; the manifest states which configuration the runner selects.
- The original container digest and full TemplateFlow cache are not recovered.
  A new run records the provided SIF hash and must report version 24.1.0.
  This makes the command auditable, without claiming bitwise reproduction.
- sub-09 has five recorded scan runs; discover available runs rather than
  requiring six for every participant. Cross-agent comparisons use levels 0-8;
  the vgfmri3 cohort also has later levels. The LLM example below uses sub-13
  from the vgfmri4 cohort.

No human MRI or full paper encoding rerun was performed during release
consolidation. A bounded synthetic integration test runs post-fMRIPrep NIfTI
processing, behavioral/base/LLM alignment, and the actual ridge solver; tests
also check original-launcher/TOML agreement and failed-run detection. Scientific reproducibility still requires a data-backed subject run,
comparison against archived derivatives/results, and a dependency lockfile.

## Paths

Run commands from this directory. `workdir` is one local data root. Stage
directories are explicit CLI arguments; no absolute laboratory paths are needed.
Keep source objects at their existing S3 keys, and use the release inventory to
map those keys to local paths. Do not infer an S3 key by adding an extra
`derivatives/` prefix: the current fMRIPrep source is
`s3://ds004323-derivatives/fmriprep/`.

| Local path | Contents |
| --- | --- |
| `workdir/raw/ds004323-1.0.0/` | Pinned raw BIDS checkout |
| `workdir/raw_behavior/dump/heroku_7lzprs54/` | Original plays/runs/regressors BSON |
| `workdir/fmriprep/` | fMRIPrep derivatives; task is always `gameplay` |
| `workdir/prepare_behavioral_data/` | Split `plays/sub-NN/run-NN.bson` and `runs.bson` |
| `workdir/preprocessed/` | `sub-NN/run-NN_voxelwise.npz` |
| `workdir/extract_model_features_to_npz/model-ddqn-curriculum/` | Existing indexed DDQN state features; mirror the exact S3 prefix |
| `workdir/llm_features/<source>/` | Multi-turn `sub-NN/<game>_vgfmri4.pt` |
| `workdir/aligned_data/` | `sub-NN/aligned_data.npz` plus `aligned_llm_<source>.npz` |
| `workdir/encoding_results/` | Per-subject, per-layer encoding results |
| `workdir/analysis/` | ROI CSV and downstream figures |

## Reproduction order

1. **Acquire the raw snapshot.** Install DataLad and git-annex, then fetch one
   subject plus the behavioral archive. The 5.26 GB behavioral archive is also
   supplied upstream in three parts; the full archive is fetched here once.

   ```bash
   bash scripts/download_openneuro.sh workdir/raw/ds004323-1.0.0 sub-13
   mkdir -p workdir/raw_behavior
   tar -xzf workdir/raw/ds004323-1.0.0/behavior/dump.tar.gz -C workdir/raw_behavior
   ```

   The archive contains `dump/heroku_7lzprs54/{plays,runs,regressors}.bson`.
   MongoDB does not need to run: the scripts decode BSON directly with PyMongo.

2. **Run fMRIPrep using the recovered subject configuration.** Acquire a
   fMRIPrep 24.1.0 SIF and a FreeSurfer license; provide each path explicitly.
   First inspect the command with `--dry-run`, then remove that flag to run.
   A new output directory avoids mixing invocation provenance.

   ```bash
   python scripts/run_fmriprep.py --subject sub-13 \
     --bids-dir workdir/raw/ds004323-1.0.0 \
     --output-dir workdir/fmriprep --work-dir workdir/fmriprep_work/sub-13 \
     --templateflow-dir workdir/templateflow \
     --fs-license /absolute/path/to/license.txt \
     --image /absolute/path/to/fmriprep-24.1.0.sif --dry-run
   ```

   The runner supports Apptainer or `--runtime singularity`, mounts raw inputs
   read-only, checks the fMRIPrep version, and records image/config checksums.
   It passes recovered scientific settings explicitly, because fMRIPrep 24.1.0
   CLI defaults override values loaded from `--config-file`. Its generated
   replay config drops stale BIDS database and invocation bookkeeping; all
   original TOMLs remain unchanged.
   Keep the generated HTML reports, logs, brain masks, and confounds for QC.

3. **Prepare behavioral data and voxel timeseries.** Use Python 3.11+ and
   `pip install -r requirements.txt`. The requirements list is not a recovered
   historical lockfile.

   ```bash
   python stages/prepare_behavioral_data.py \
     --input-dir workdir/raw_behavior/dump/heroku_7lzprs54 \
     --output-dir workdir/prepare_behavioral_data
   python encoding_model_code/preprocess.py --subject sub-13 \
     --fmriprep-dir workdir/fmriprep --output-dir workdir/preprocessed
   ```

   Post-fMRIPrep processing uses the fMRIPrep brain mask, 8 mm spatial
   smoothing, 1/128 Hz high-pass filtering, motion/CSF/white-matter regression,
   AR(1) pre-whitening (dropping the first volume), and voxel z-scoring.

4. **Recover base alignment.** The submitted `align.py` is LLM-only and cannot
   create `aligned_data.npz`. The restored `align_base.py` builds that missing
   input. Its historical implementation requires per-state DDQN activations
   even if the subsequent encoder is an LLM. Supply the corresponding indexed
   derivative features under the actual indexed
   `extract_model_features_to_npz/model-ddqn-curriculum/` prefix.
   A [representative NPZ schema check](docs/ddqn-schema-check.json) confirmed
   the restored loader's per-play activations, timestamps, and metadata keys.
   Alternatively, download the existing
   `s3://ds004323-derivatives/vgfmri_preprocessed_aligned/sub-NN/aligned_data.npz`
   into `workdir/aligned_data/sub-NN/aligned_data.npz` and proceed directly to
   LLM alignment. The separate
   `extract_model_features/ddqn/` prefix is another asset family; do not alias
   it without checking the schema.

   ```bash
   python encoding_model_code/align_base.py --subject sub-13 \
     --preprocessed-dir workdir/preprocessed \
     --model-features-dir workdir/extract_model_features_to_npz/model-ddqn-curriculum \
     --behavioral-dir workdir/prepare_behavioral_data \
     --plays-bson workdir/raw_behavior/dump/heroku_7lzprs54/plays.bson \
     --regressors-bson workdir/raw_behavior/dump/heroku_7lzprs54/regressors.bson \
     --max-level 8 --output-dir workdir/aligned_data
   ```

   `stages/extract_model_features_to_npz.py` preserves the historical DDQN
   regeneration implementation. Its W&B mapping names `trial1-sequential`
   checkpoints; that is not evidence that they generated the later
   `model-ddqn-curriculum` prefix. Verify the selected checkpoint hashes against
   derivative provenance before claiming identical regeneration, and give any
   newly generated features a distinct output/model label. The public
   [bundled baseline sources](../../baselines/README.md) include the archived
   `RC_RL` extraction revision `8f5f8a3c6facba2962319ea4892396057061370c` at
   `../../baselines/vendor/rc_rl/extraction`; `--rc-rl-dir` can override this path.
   The extractor verifies the bundled source manifest and every recorded file
   before import, and records that manifest's hash in generated features.
   No private GitHub checkout is needed for the default path.
   Install the public checkout's baseline inference dependencies following its
   [baseline guide](../../baselines/README.md) before regenerating activations.
   Provide a local JSON checkpoint map `{ "bait": { "0": "/path/to/weights.pt" } }`.
   The old W&B run mapping is available only with `--allow-checkpoint-download`.
   Every new feature file records checkpoint hashes and the actual source
   revision, so these outputs can be distinguished from the indexed derivatives.

   ```bash
   python stages/extract_model_features_to_npz.py --subject sub-13 --run 1 \
     --local-dir workdir --output-dir workdir/extract_model_features_to_npz \
     --checkpoint-map /absolute/path/to/checkpoints.json --model-id ddqn-local-rerun
   ```

   EfficientZero's recovered extractor is exposed by
   `../../baselines/run_efficientzero.py`. Pass its trace root to base alignment
   with `--ez-features-dir`; it reads
   `<game>/subj13/run1/play0_key<MongoDB-ID>/traces.pt` and writes `aligned_ez.npz`.
   EMPA symbolic regressors supplied by OpenNeuro are encoded as HRR directly
   when `--regressors-bson` is provided.

5. **Generate LLM traces and features** using the public release's
   `src.llm_eval.human_replay.run_replay` and
   `src.llm_eval.human_replay.extract_features`. The imputation model and
   extraction model are separate. Use only `action-only` or `copied-reasoning`
   for replay; retain timestamps, play IDs, and all extracted stream metadata.
   The multi-turn layout required below is
   `workdir/llm_features/<source>/sub-NN/<game>_vgfmri4.pt`.

6. **Align and encode LLM features.** Source names are stable labels, not
   directory guesses. Use a source name understood by the ROI parser's
   `MODEL_INFO`; unfamiliar models require explicit metadata there.

   ```bash
   python encoding_model_code/align.py --subject sub-13 \
     --aligned-data workdir/aligned_data/sub-13/aligned_data.npz \
     --plays-bson workdir/raw_behavior/dump/heroku_7lzprs54/plays.bson \
     --runs-bson workdir/prepare_behavioral_data/runs.bson \
     --llm-source 'name=qwen35_9b__all__main,dir=workdir/llm_features/qwen35_9b,stream=main' \
     --output-dir workdir/aligned_data
   python encoding_model_code/encoding_model.py --subject sub-13 \
     --data-dir workdir/aligned_data --output-dir workdir/encoding_results \
     --layer llm_qwen35_9b__all__main_layer_1 --max-level 8 --include-nuisance-bands
   ```

   The submitted encoder uses banded ridge, within-fold PCA before lags,
   lags 2-5 TR, and leave-one-level-partition-out validation (0-2, 3-5, 6-8).
   Default bands are main only; the explicit flag above adds game/level, button,
   and time bands. New runs fail on fold errors and exit nonzero when any layer
   fails. Existing results require explicit `--resume` with matching settings.
   See [scientific conventions](docs/scientific-conventions.md) for remaining
   historical randomness and producing-run limitations.

7. **Aggregate neural results**, then use the public release's neural and
   behavioral analysis scripts. Obtain the AAL atlas and its labels separately;
   this repo does not redistribute an atlas with unknown provenance.

   ```bash
   mkdir -p workdir/analysis
   python encoding_model_code/load_and_parse.py \
     --results-dir workdir/encoding_results \
     --fit-condition with-nuisance \
     --atlas /absolute/path/to/ROI_MNI_V4.nii \
     --atlas-labels /absolute/path/to/ROI_MNI_V4.txt \
     --output workdir/analysis/encoding_roi_streams.csv
   ```

   `fit_condition` records whether nuisance bands were fitted. It is distinct
   from the emitted `band` column: a main-band score from a nuisance fit is not
   a main-only fit. The parser requires explicit `--fit-condition main-only`
   or `with-nuisance` when inputs contain both; write separate CSVs for them.

   The historical parser recognized LLM filenames only. The public parser also
   accepts `--baseline-map /path/to/layer-map.json` for DDQN/EZ/HRR outputs.
   [This example](docs/baseline-layer-map.example.json) labels fresh FC1/HRR runs
   explicitly; customize the semantic feature name, model, layer index/count,
   and normalized depth for every requested baseline layer. The historical
   baseline CSV's 11 numeric EZ labels have not been mapped to the recovered
   15 named trace hooks, so this example intentionally makes no claim to
   reproduce that mapping. Use the archived baseline CSV for those reported
   results while their exact producing alignment remains unresolved.

   ROI aggregation now fails for unreadable results, inconsistent masks, worker
   failures, and empty output instead of silently emitting a partial CSV.

## Bounded verification

```bash
python -m pytest tests/ -q
```

The integration test constructs a small synthetic 4D NIfTI, mask, confounds,
BSON plays/runs, DDQN state features, and multi-turn LLM features. It runs the
actual postprocessing, base and LLM alignment, main/nuisance ridge fits, and ROI
aggregation. It covers EZ/HRR sidecar loading, AR(1) volume offsets, per-play
lag boundaries, scan-truncated final plays, nuisance-fit selection, level restriction,
local checkpoint lookup, and failed-run
reporting. It does not invoke fMRIPrep, download MRI/model weights, or reproduce
paper statistics. The dependency-free provenance tests check the container
command against the pinned fMRIPrep parser and all retained configurations.

Additional bounded checks exercise real baseline inputs:
[EfficientZero extractor-to-aligner](docs/efficientzero-contract-check.json)
loads recorded-checkpoint traces for 64-state and 12-state human plays;
[DDQN engine-to-extractor](docs/ddqn-vendor-contract-check.json) uses the same
64-state human play with a synthetic untrained checkpoint. The latter verifies
the bundled source boundary and output schema, not the paper's DDQN weights.

## Branch archival

[docs/branch-archive.json](docs/branch-archive.json) records every original
branch tip and the SHA-256 of a verified full Git bundle. The
`archive/2026-09-27/<branch>` tags preserve all three original non-main tips;
their remote targets were verified against the recorded branch SHAs. After
the integration PR merges, original active branch refs are removed only if
they still match the preserved tips. The archive tags remain the recovery refs.
The curated release restores the needed code without importing Terraform
state, personal job scripts, legacy RSA experiments, or post-paper geometry
experiments into main. Those histories remain recoverable through the tags.

Sources: [OpenNeuro snapshot](https://github.com/OpenNeuroDatasets/ds004323/tree/1.0.0),
[fMRIPrep command documentation](https://fmriprep.org/en/24.1.0/usage.html).
