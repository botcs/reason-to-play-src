# tomov23-analysis historical reconstruction

> Historical archaeology report captured before the final source recovery and integration.
> See the [current reproduction guide](../reproducibility.md) and [artifact audit](../release/scientific-provenance.md) for subsequent findings and release state.

Read-only historical audit, 2026-09-27. Repository contents and refs were not
changed during this audit. The earlier release PR remains separate and unmerged.

## Main findings

1. The earliest reachable commit in **this repository** is **2025-10-28**, not
   September. The recovered fMRIPrep execution records predate it (October 13-28).
   The absence of earlier commits here does not establish that preprocessing
   code was absent from the broader project: the user located it in `botcs/RC_RL`,
   which is being audited separately.
2. The AWS/staged RSA pipeline was developed by `botcs` during October-November
   2025 and continued through December 11. The first Sreejan Kumar-authored
   encoding import in the reachable history is **December 17, 2025**, not
   November. This establishes a commit date, not the date work began offline.
3. The original `encoding_model` branch, the later clean SLURM implementation,
   and the submitted `main` are **different branches/imports**, not one continuous
   sequence of merges. Their scientific behavior must be compared explicitly.
4. The 2024 Nature Communications paper points to `tsumers/bert-brains` as its
   analysis code. Shared author and methodological similarities are established;
   a direct code import from that repository is not established by the current
   tomov23-analysis commit messages or source references.

## Scope and evidence

The original four branches plus the submission tag contain 83 reachable commits.
I inspected author/committer timestamps, parent commits, changed-file lists,
representative scripts at each transition, every historical code/document blob
for fMRIPrep invocation patterns, and historical references to the Nature DOI,
`bert-brains`, Sumers, Nastase, and functional specialization. All original
histories share root `5c53c931fce464c5f708e3f416db4330be434fa8`.

This audit does not cover unpublished local commits or deleted refs no longer
advertised by GitHub. Git authorship records attribution as committed; it does
not establish who wrote every line or when an imported file was first created.

## Timeline

| Date (author timestamp) | Author | Commit | Recorded transition and evidence |
| --- | --- | --- | --- |
| 2025-10-28 | botcs | [5c53c93](https://github.com/botcs/tomov23-analysis/commit/5c53c931fce464c5f708e3f416db4330be434fa8) | Root commit, “Local fmriprep upload (skip stage 1)”. Tracked tree contains README, params/subjects, an fMRIPrep upload script, and upload manifest. README sketches raw fMRIPrep/native RSA/warp stages, but the raw launcher is not in this tree. |
| 2025-10-28 | botcs | [6f69f9c](https://github.com/botcs/tomov23-analysis/commit/6f69f9c) | Adds RC_RL submodule for DDQN extraction, explicitly connecting this repository to the older project. |
| 2025-10-28–29 | botcs | [58ef349](https://github.com/botcs/tomov23-analysis/commit/58ef349), [cb6be88](https://github.com/botcs/tomov23-analysis/commit/cb6be88), [91f90e8](https://github.com/botcs/tomov23-analysis/commit/91f90e8), [c777a72](https://github.com/botcs/tomov23-analysis/commit/c777a72), [9e821d6](https://github.com/botcs/tomov23-analysis/commit/9e821d6) | DDQN/state extraction, human BOLD extraction, HRF convolution, parallel RDM computation. October 29 commit explicitly records updating Stage 3A with Momchil's suggestions. This is an RSA pipeline, not yet the later fitted voxelwise encoding model. |
| 2025-11-03–04 | botcs | [578d7e6](https://github.com/botcs/tomov23-analysis/commit/578d7e6), [b7f640a](https://github.com/botcs/tomov23-analysis/commit/b7f640af7de728a38056649b857c2e79aaade3f9) | Refactors stage organization; adds working MNI-to-NPZ extraction. Includes historical `docker/register_bold_to_mni/Dockerfile` based on fMRIPrep 24.1.0 and `scripts/register_bold_to_mni.sh`. That script resamples existing T1w-space preprocessed BOLD with an existing anatomical transform, not raw BIDS preprocessing. |
| 2025-11-05–06 | botcs | [f88c70c](https://github.com/botcs/tomov23-analysis/commit/f88c70c), [b0d53e5](https://github.com/botcs/tomov23-analysis/commit/b0d53e5), [3bc48ee](https://github.com/botcs/tomov23-analysis/commit/3bc48eeea10afdcc78aa312b8d1c5631f1f43ee1), [ca5c36e](https://github.com/botcs/tomov23-analysis/commit/ca5c36e) | Adds MNI/native searchlight RDM and between-subject similarity stages, removes extra T1w-to-MNI registration stage, and consolidates Docker images. |
| 2025-11-06–09 | botcs | [2bafeec](https://github.com/botcs/tomov23-analysis/commit/2bafeec), [6e4f412](https://github.com/botcs/tomov23-analysis/commit/6e4f412), [40c35ce](https://github.com/botcs/tomov23-analysis/commit/40c35ce3ea250dd9ceb9b29d16e7fdaa186c4458), [4fc2ef3](https://github.com/botcs/tomov23-analysis/commit/4fc2ef3) | AWS Batch Terraform and config-driven orchestration; stages switch to fMRIPrep-provided brain masks. Full fMRIPrep upload uses bucket-root `fmriprep/`, superseding initial `derivatives/fmriprep/` conventions. |
| 2025-11-09–14 | botcs | [08a04fa](https://github.com/botcs/tomov23-analysis/commit/08a04fa), [fe8f261](https://github.com/botcs/tomov23-analysis/commit/fe8f261), [2eefdd2](https://github.com/botcs/tomov23-analysis/commit/2eefdd2) | Pairwise similarity, aggregation, downsampling/QC, model RDM, HRF convolution, and individual RSA evaluation. The intended abstraction is named stage inputs/outputs mirrored locally and in S3. |
| 2025-11-14 | botcs | [f092d8c](https://github.com/botcs/tomov23-analysis/commit/f092d8c), [dafb69b](https://github.com/botcs/tomov23-analysis/commit/dafb69b2430d18d51c6ce3c386b5376e73a2e47a) | Starts LLM support in scratch inference code; the `llm` branch experiments with text/grid prompts and char sets. This branch is separate from the later encoding import. |
| 2025-11-20–12-11 | botcs | [c19bb6a](https://github.com/botcs/tomov23-analysis/commit/c19bb6a), [1ba4a45](https://github.com/botcs/tomov23-analysis/commit/1ba4a45), [2d9cea6](https://github.com/botcs/tomov23-analysis/commit/2d9cea6), [e7549c4](https://github.com/botcs/tomov23-analysis/commit/e7549c4), [d8eaa5a](https://github.com/botcs/tomov23-analysis/commit/d8eaa5a) | Affine-from-data fix, metadata/run-argument fixes, rsatoolbox reevaluation, AWS model extraction coverage, and extraction of incomplete cohort runs. These are empirical maintenance points, not evidence of overall code quality. |
| **2025-12-17** | **Sreejan Kumar** | [3199db9](https://github.com/botcs/tomov23-analysis/commit/3199db9) | First committed encoding import: 10 added files, 3,963 lines. Adds Schaefer 1000-parcel BOLD extraction, feature alignment, HRR embedding code, `GroupRidgeCV` encoding, AAL helper, and SLURM job submitters. Target is parcelwise BOLD; model features are fitted directly to BOLD, with game/level, button, and time regressors. |
| 2025-12-18 | Sreejan Kumar | [718e09c](https://github.com/botcs/tomov23-analysis/commit/718e09c) | Adds `preprocess_and_parcellate.py` and substantially revises aligner/encoder. Explicit post-fMRIPrep smoothing at 8 mm, 1/128 Hz high-pass filtering, and Schaefer 1000 parcels; confound TSV input is supported. |
| **2026-01-28** | **Sreejan Kumar** | [3e0f53a](https://github.com/botcs/tomov23-analysis/commit/3e0f53a750c6544fa9a6261adeb04c6441f08803) | `encoding_model` branch tip: adds `preprocess_voxelwise.py`, `align_model_features_ez_v6.py`, and `encoding_model_v4.py` (3,375 lines). Moves from parcels to voxels, with fMRIPrep masks, confound regression, AR(1) whitening, first-volume removal, and per-voxel z-scoring; includes EfficientZero support. Older scripts remain alongside new ones. |
| **2026-03-23** | **Sreejan Kumar** | [26fa74c](https://github.com/botcs/tomov23-analysis/commit/26fa74c) | Separate clean import, 9 files / 7,264 lines under `encoding_model_slurm/`: preprocess, multi-source align, encode, decode, and four SLURM submitters plus README. Adds/organizes DDQN, EZ, HRR, LLM sources; documents PCA before lag expansion, 2–5 past TR lags, and three level partitions. |
| 2026-07-30–31 | botcs | [447a56a](https://github.com/botcs/tomov23-analysis/commit/447a56a), [d4e0295](https://github.com/botcs/tomov23-analysis/commit/d4e0295803c347d41a68ee2c9ff637ef4125d9fd), [dbc779b](https://github.com/botcs/tomov23-analysis/commit/dbc779b) | Reproduction branch receives explicit alpha-grid, seeding/provenance, failure-handling, kernel/feature solver equivalence, subsampling/TF32, second-band, and PCA diagnostic changes. These edits target the SLURM encoder. |
| **2026-07-31** | **botcs** | [62c2021](https://github.com/botcs/tomov23-analysis/commit/62c202130d057e585b6bf7978f621fcd0255919c) | Creates the submitted `main` tree with only five `encoding_model_code/` files, deleting earlier pipeline/infrastructure from this branch. Tag `encoding-suppmat-v1.0` preserves this artifact. The commit is an import/reduction from the older AWS lineage, not a merge of the clean/reproduction branch. |
| 2026-07-31–08-20 | botcs | [f95147c](https://github.com/botcs/tomov23-analysis/commit/f95147c), [899edf6](https://github.com/botcs/tomov23-analysis/commit/899edf6), [d49655c](https://github.com/botcs/tomov23-analysis/commit/d49655c8cb179e6d7ecd601ccb766eb34331404c) | Working branch documents `main` as frozen supplement and its own stage encoder as superseded. Adds 146 geometry/encoding-reproduction experiment files and a beta viewer. This later research branch is broader than the accepted paper's reproduction pipeline. |

## Branch topology: why a wholesale merge is scientifically unsafe

The following parent relationships were read directly from Git:

```text
shared AWS/RSA history
  ... d0fc2a0 (2025-12-03)
      |-- 3199db9 (2025-12-17, Sreejan)
      |     -> 718e09c -> 3e0f53a [encoding_model]
      |
      `-- ... d8eaa5a (2025-12-11, botcs)
              |-- 26fa74c (2026-03-23, Sreejan clean import)
              |     -> July reproduction edits
              |     -> f95147c -> geometry additions -> d49655c
              |        [encoding_model-reproduction_temporal]
              |
              `-- 62c2021 (2026-07-31, submission import)
                  [main; encoding-suppmat-v1.0]
```

The earlier `llm` scratch branch diverges in November and ends at `dafb69b`.
No merge commits connect these three encoding implementations. In particular,
`3199db9` has parent `d0fc2a0`; `26fa74c` and `62c2021` both have parent `d8eaa5a`.
The working branch's CLAUDE.md records that an `encoding_model_clean` branch was
later deleted, with tip `26fa74c` retained in history.

## Concrete scientific differences requiring reconciliation

- **Parcel to voxel transition:** December scripts use Schaefer parcellation;
  January voxelwise preprocessing adds AR(1) whitening and explicitly changes
  time length by one volume. Mixing their alignment products would be invalid.
- **Lag direction changed:** the initial December 17 encoder's
  `create_lagged_features` uses `src_idx = start + t + lag` (future feature rows
  relative to its current target indexing). The March clean and July submitted
  encoder explicitly use past feature rows (`src_t = t - lag`). The effect on
  historical results requires a data-backed audit; do not assume output files
  from these revisions are equivalent.
- **Regularization differs:** the December import includes
  `np.logspace(-6, 6, 25)`; the submitted main uses `np.logspace(-5, 10, 20)`.
  July reproduction commits also change grid/seeding/failure behavior on the
  separate working encoder.
- **Band defaults differ:** March clean code has
  `DEFAULT_BANDS = ['main', 'identity']` with button/time optional. Submitted
  main has `DEFAULT_BANDS = ['main']`, with identity/button/time all optional
  under nuisance-band inclusion. Examples that pass `--include-nuisance-bands`
  explicitly avoid ambiguity; descriptions saying identity is always present
  are inaccurate for submitted main.
- **Base alignment disappeared from the submitted tree:** March's full aligner
  creates `aligned_data.npz` from BOLD/DDQN/behavior/HRR. July's `align.py` is an
  LLM-only addition to already-existing `aligned_data.npz`. It does not replace
  the missing base builder.
- **Only preprocessing is byte-identical:** July main `preprocess.py` equals
  March clean `preprocess.py` byte-for-byte (535 lines each). July `align.py`
  (1,503 lines) and `encoding_model.py` (1,118) differ from March `align.py`
  (2,728) and `encode.py` (948). July `load_and_parse.py` is additional.

These differences explain why all original lineages should remain accessible
while constructing a validated canonical pipeline. They are evidence about
code behavior, not judgments about an individual contributor.

## Relation to the Nature Communications paper

The referenced article is **Shared functional specialization in transformer-based
language models and the human brain**, published **2024-06-29**, with Sreejan
Kumar and Theodore Sumers as co-first authors. Its Code availability section
points to [tsumers/bert-brains](https://github.com/tsumers/bert-brains) and
[Zenodo 10863840](https://doi.org/10.5281/zenodo.10863840).
[Paper and code availability](https://www.nature.com/articles/s41467-024-49173-5).

The paper uses Narratives fMRI, Schaefer 1000 parcels, and banded ridge
encoding. The December VGDL import has corresponding methodological motifs
but uses VGDL timing/levels and computational-model features. This supports a
plausible methodological connection, not a demonstrated source-code import.
The upstream README names `analysis/banded_ridge_regression.py`, headwise
encoding variants, noise-ceiling code, and SLURM job generation. It declares
GPL-3.0. [Upstream repository](https://github.com/tsumers/bert-brains).

No explicit reference to this DOI, `bert-brains`, Sumers, or Nastase was found
in the reachable tomov23-analysis historical source/docs searched. The March
clean README instead cites Tomov 2023 and the banded-ridge method. To establish
exact source provenance, compare the December imported functions against the
paper's archived code, or recover the contributor's original working history.
Do not infer provenance solely from a shared library import or design pattern.

## Historical documentation caveats

The archived `MONGODB_README.md` calls `plays.win` boolean and labels terminal
zstate `-1` as loss, while the verified dataset contract is three-valued and
requires avatar-death detection before treating remaining `None` as incomplete.
It also contains older 92.9% win-rate statistics. These are historical text,
not reliable evidence about the current full dataset or results.

The first README's unsmoothed/native-RSA plan also differs from the actual
later MNI/smoothed/voxelwise encoding pipeline. Treat README claims as intended
architecture for that revision, and scripts/manifests/data as evidence of
what was actually executed.

## State at the user hold

No original branch was deleted and `main` was not merged by this agent.
The three original non-main tips have archive tags created before the hold.
A release proposal remains at PR 1, latest proposed head
`21589498972d4c5a885168f6ee52c9bc9a2493cd`; its last documentation correction
completed immediately before the hold was delivered. No further publication,
repository edits, or branch operations occurred during this timeline audit.
