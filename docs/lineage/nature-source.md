# Nature article and Sreejan's neural-pipeline provenance

> Historical archaeology report captured before the final source recovery and integration.
> See the [current reproduction guide](../reproducibility.md) and [artifact audit](../release/scientific-provenance.md) for subsequent findings and release state.

Read-only investigation, 2026-09-27. No release files, branches, or remote repositories were changed for this investigation.

## Article identity

**Shared functional specialization in transformer-based language models and the human brain**, *Nature Communications* **15**, 5523 (2024), DOI [10.1038/s41467-024-49173-5](https://www.nature.com/articles/s41467-024-49173-5).

Authors: Sreejan Kumar, Theodore R. Sumers, Takateru Yamakoshi, Ariel Goldstein, Uri Hasson, Kenneth A. Norman, Thomas L. Griffiths, Robert D. Hawkins, and Samuel A. Nastase. Kumar and Sumers are joint first authors. Received **21 July 2023**, accepted **24 May 2024**, published **29 June 2024**.

The article's code-availability statement points to [tsumers/bert-brains](https://github.com/tsumers/bert-brains) and [Zenodo 10863840](https://doi.org/10.5281/zenodo.10863840). Its human data are the Narratives collection, **OpenNeuro ds002345 v1.1.4**, rather than Tomov's ds004323. Its methods use existing Narratives derivatives, fMRIPrep **20.0.5**, AFNI confound regression, and unsmoothed images. Thus this citation establishes an earlier encoding-method source, not the raw preprocessing provenance of the later VGDL dataset. [Article methods and availability](https://www.nature.com/articles/s41467-024-49173-5#Sec8).

## Official source repository chronology

Verified through GitHub API commit/tree responses and the Zenodo record API:

| Date | Evidence |
| --- | --- |
| 2020-08-27 | GitHub repository `tsumers/bert-brains` created. This is repository creation, not paper publication. |
| 2022-04-20 | User `sreejank` adds the main analysis scripts, including root-level `banded_ridge_regression.py`, commit [`0574549cfdda381277d20b5c82e09e6da83042ca`](https://github.com/tsumers/bert-brains/commit/0574549cfdda381277d20b5c82e09e6da83042ca). |
| 2022-05-26 | Ted reorganizes files; `banded_ridge_regression.py` is explicitly renamed to `analysis/banded_ridge_regression.py`, commit [`645fc70089b87b34f7bfc27738961844f8395114`](https://github.com/tsumers/bert-brains/commit/645fc70089b87b34f7bfc27738961844f8395114). |
| 2022-06-13 | Commit [`26d4da6c1419fc8e635351ddb692897e2d0a1cc5`](https://github.com/tsumers/bert-brains/commit/26d4da6c1419fc8e635351ddb692897e2d0a1cc5), authored by Samuel Nastase, is the target of the `camera-ready` tag. The commit's timestamp is distinct from the later archive publication date. |
| 2024-03-23 | Zenodo archive [10.5281/zenodo.10863840](https://doi.org/10.5281/zenodo.10863840) published, version `camera-ready`, creators Theodore Sumers, Sreejan Kumar and Sam Nastase. The record links that GitHub tag. |
| 2024-06-06 | Sreejan Kumar updates the publication title in README, current repository head [`225b8b3bcee0dd609d313fb72b01ccd08b330e6d`](https://github.com/tsumers/bert-brains/commit/225b8b3bcee0dd609d313fb72b01ccd08b330e6d). |

Key source files:

- [`analysis/banded_ridge_regression.py`](https://github.com/tsumers/bert-brains/blob/26d4da6c1419fc8e635351ddb692897e2d0a1cc5/analysis/banded_ridge_regression.py): the original banded-ridge encoding analysis.
- [`analysis/headwise_banded_ridge_regression.py`](https://github.com/tsumers/bert-brains/blob/26d4da6c1419fc8e635351ddb692897e2d0a1cc5/analysis/headwise_banded_ridge_regression.py): head-specific evaluation.
- [`data_handling/create_fmri_dataset.py`](https://github.com/tsumers/bert-brains/blob/26d4da6c1419fc8e635351ddb692897e2d0a1cc5/data_handling/create_fmri_dataset.py): handles Narratives data, not ds004323 raw MRI acquisition/preprocessing.
- [`text_processing_and_transformers/transformer_utils.py`](https://github.com/tsumers/bert-brains/blob/26d4da6c1419fc8e635351ddb692897e2d0a1cc5/text_processing_and_transformers/transformer_utils.py): transformer feature utilities.

The GitHub repository has a GPL-3.0 license file; Zenodo metadata labels the archived record CC-BY-4.0. Record this discrepancy if code reuse is contemplated; do not infer that the archive metadata removes the source license notice. No source was copied into the release during this investigation.

## Direct evidence in tomov23-analysis

Git history in [`botcs/tomov23-analysis`](https://github.com/botcs/tomov23-analysis) records Sreejan's later VGDL-specific pipeline work. These are author and commit dates, not claims about when computations were run:

| Date | Commit and actual files introduced |
| --- | --- |
| 2025-12-17 | [`3199db9946ada6328fab419d2c5f9fb1bb9c5ca3`](https://github.com/botcs/tomov23-analysis/commit/3199db9946ada6328fab419d2c5f9fb1bb9c5ca3), Sreejan Kumar, `Current encoding model pipeline`: adds `stages/encoding_model.py`, `stages/align_model_features.py`, `stages/parcellate_bold.py`, HRR/theory helpers and job submission scripts. |
| 2025-12-18 | [`718e09c`](https://github.com/botcs/tomov23-analysis/commit/718e09c), Sreejan Kumar, `Current approach`. |
| 2026-01-28 | [`3e0f53a750c6544fa9a6261adeb04c6441f08803`](https://github.com/botcs/tomov23-analysis/commit/3e0f53a750c6544fa9a6261adeb04c6441f08803), Sreejan Kumar, `Current voxelwise pipeline`: adds `stages/preprocess_voxelwise.py`, `stages/align_model_features_ez_v6.py`, and `stages/encoding_model_v4.py`. |
| 2026-03-23 | [`26fa74ce715781bfdd7a6f6a4bc2a6685e5a8678`](https://github.com/botcs/tomov23-analysis/commit/26fa74ce715781bfdd7a6f6a4bc2a6685e5a8678), Sreejan Kumar, `Push clean encoding model code`: adds `encoding_model_slurm/stages/{preprocess,align,encode,decode}.py`, README and submission scripts. |
| 2026-07-31 | [`62c2021`](https://github.com/botcs/tomov23-analysis/commit/62c2021), botcs, reduces main to `encoding_model_code/` supplementary submission. |

One exact content linkage is established: March's `encoding_model_slurm/stages/preprocess.py` and July's `encoding_model_code/preprocess.py` have the **same Git blob** `3f78dbfa617cca456b0e9301a879e93820a7ea77`. Therefore the supplementary preprocessing script is demonstrably Sreejan's earlier committed file. March's `encode.py` and July's `encoding_model.py` differ (blobs `0dd6e51f5c263e06795d7ecb0187be445cbfa3a1` versus `b3380bc3f03b56ff7879defd20203ade695f8dd4`); later encoding revisions must be tracked separately.

## Relationship supported by the code

The earlier official script and later VGDL scripts share concrete methodological choices: `himalaya.ridge.GroupRidgeCV`, Pearson-correlation scoring, delayed feature matrices, and nuisance-feature handling. The official script at the archived commit uses `delays=[2,3,4,5]` (lines 84 and 193), nested non-shuffled three-fold `KFold` (310–311), and `n_iter=100` (341). The current VGDL encoder uses `LAGS=[2,3,4,5]`, game/level-aware partitions, optional button/time nuisance bands, and training-fold PCA before lagging. The initial 2025 VGDL encoder uses a single-group `n_iter=1` path; later code uses richer banded modeling. Those distinctions matter when identifying which version produced an output.

This supports **shared author and methodological continuity**. It does **not** establish that the current VGDL pipeline is a direct file copy of `bert-brains`: no explicit `bert-brains` URL or article identifier was found in the examined tomov Git text history, and a whitespace-normalized comparison of the 2022 ridge script with the first 2025 VGDL encoder shares only generic imports/control syntax. A rewritten/adapted implementation remains possible; absence of exact copied lines cannot establish independent authorship.

The raw ds004323 fMRIPrep launcher should therefore be traced through the Tomov/RC_RL acquisition and preprocessing history, independently of this 2024 article's Narratives preprocessing description. Sreejan's post-fMRIPrep preprocessing, alignment and neural encoding code is already directly evidenced by the tomov commits above.
