# Baseline lineage audit (read-only, 2026-09-27)

> Historical archaeology report captured before the final source recovery and integration.
> See the [current reproduction guide](../reproducibility.md) and [artifact audit](../release/scientific-provenance.md) for subsequent findings and release state.

Release changes remain on hold. This note records history; no repositories, branches, datasets, public pages, or releases were changed for this audit. Git dates establish committed code availability, not dates of scientific discovery or proof that a particular published result used that revision. S3 `LastModified` dates are upload dates, not creation dates.

## Main finding

RC_RL is part of the actual baseline pipeline: it contains the DDQN implementation and the game/environment changes consumed by EfficientZero. Its current warning that it is the wrong starting point concerns the origin of the **original human experiment / EMPA analysis**; reading that warning as excluding RC_RL from the later study loses real work.

The separate EfficientZero implementation is **[A-Andrews/EfficientZeroV2](https://github.com/A-Andrews/EfficientZeroV2)**, a fork of **[Shengjiewang-Jason/EfficientZeroV2](https://github.com/Shengjiewang-Jason/EfficientZeroV2)**. The research implementation lives on `development` and then `integrated-development`. Its default `main` branch does not contain the substantive VGDL integration. `botcs/EfficientZeroV2` returned HTTP 404 to the available credentials; that was an unsuccessful guessed location, not evidence that the implementation did not exist.

The S3 backup independently confirms this origin. Reading only the small Git metadata at `s3://ds004323-derivatives/brain-wide_strategies/EfficientZeroV2/.git/config` showed `origin=https://github.com/A-Andrews/EfficientZeroV2.git`, `upstream=https://github.com/Shengjiewang-Jason/EfficientZeroV2.git`; `.git/HEAD` points to `refs/heads/integrated-development`. Any URL credentials were removed before displaying metadata.

## Distinct origins

- **Human task, EMPA, original fMRI analysis:** [tsividis/vgdl, refactor_fMRI_cannon](https://github.com/tsividis/vgdl/tree/refactor_fMRI_cannon). RC_RL pins its `py-vgdl-reference` submodule to [f41409ae242412e6722a2c7518547ac7d1eb36d2](https://github.com/tsividis/vgdl/commit/f41409ae242412e6722a2c7518547ac7d1eb36d2), authored **2023-01-02** (“rerun ablations”). This is an origin/reference dependency, not a substitute for later RC_RL baseline or fMRIPrep work.
- **DDQN and shared VGDL environment:** [botcs/RC_RL](https://github.com/botcs/RC_RL), retaining older RC_RL/DDQN history across multiple branches and the current team's Python 3, input representation, protocol, tracing, and curriculum changes. Current `master` descends from the September 2019 commit `8df431a1b5311ad8ff9d1a55e0a3a04076128ab3`; the later 2021–2022 upstream fMRI/DDQN work is retained on the separate `fmri` branch, not in current `master` ancestry. Its older README also links the [ACampero/dopamine](https://github.com/ACampero/dopamine) submodule separately from the PyTorch `runDDQN.py` implementation; do not conflate those implementations.
- **EfficientZero V2 algorithm:** [official ICML 2024 implementation](https://github.com/Shengjiewang-Jason/EfficientZeroV2) and [Wang et al., ICML 2024](https://proceedings.mlr.press/v235/wang24at.html), itself building on the earlier EfficientZero project. The official README announces release in May 2024; the earliest commit retained in this fork is [1367bca](https://github.com/Shengjiewang-Jason/EfficientZeroV2/commit/1367bca20374a334b3ee9b0d0ad19ce6cc7c2e79), **2024-06-07**. These are different kinds of dates.
- **This study's EfficientZero-to-VGDL integration:** Austin Andrews's fork, especially [integrated-development at 29157d4](https://github.com/A-Andrews/EfficientZeroV2/tree/29157d4892afd9467b1bd0994de1355086145490), together with [RC_RL EZ-development-changes at b1e3376](https://github.com/botcs/RC_RL/tree/b1e33768b9f1d799d7780a7e6556d328be5174ab).

## Timeline with immutable evidence

| Date | Event and evidence | What it establishes |
|---|---|---|
| 2021-12-12 to 2022-05-09 | The separately retained RC_RL `fmri` branch includes [fMRI DDQN launcher](https://github.com/botcs/RC_RL/commit/fdb74ee6eb343ab861413ca4d8f5d4077e70cd6e), [evaluation launcher](https://github.com/botcs/RC_RL/commit/a4f1da7d054c3374e64bfbf6b24569c844f22575), and [fMRI game definitions](https://github.com/botcs/RC_RL/commit/3388f9caa9c611e5cd1dbd262c38df24087c9963). | This is preserved upstream fMRI baseline work on `fmri`, not a linear predecessor of current `master`; repository creation date is not scientific origin. |
| 2025-09-15 | `botcs/RC_RL` repository created; same-day [environment fixes](https://github.com/botcs/RC_RL/commit/99566fd07cb5d31da16802b2b2422109af289441), [Python 2 to 3 work](https://github.com/botcs/RC_RL/commit/9ce1dfa795291aa47376d9ae606f8503b75347a9), [working human replay](https://github.com/botcs/RC_RL/commit/a52eb2db6bbd379b4970a14aa3f9f37fefce3e3c). | Start of the current team's retained revival work. Its first commit has parent `8df431a1b5311ad8ff9d1a55e0a3a04076128ab3` (2019-09-05), not the later `fmri` tip. |
| 2025-09-24 to 2025-09-30 | [RL reproduction](https://github.com/botcs/RC_RL/commit/51935e80d4edc19649df73fc1ee92c90c957c705), then [one-hot input representation](https://github.com/botcs/RC_RL/commit/0bb49b3c940aaabd91a5d362a8067e67db11b3d5), [per-level training budget fix](https://github.com/botcs/RC_RL/commit/dd7050c7de3a9690b884a677934344a34da52c36), and [level padding](https://github.com/botcs/RC_RL/commit/23761b5398f17df549131bbf8d4fec2b117041e4). | DDQN reproduction involved scientific input/protocol changes as well as dependency repair. |
| 2025-10-01 meeting; committed 2025-10-02 | [Momchil meeting notes](https://github.com/botcs/RC_RL/blob/4d896df2e5160d80d206766771fc69e9b6107333/momchil-oct1-meeting-notes.md), alongside [recovered game/level descriptions](https://github.com/botcs/RC_RL/commit/6562d1092c54cf0c4152e7624a50f836d4a66475). | Explicitly identifies the original human/EMPA source, `fmri_playsPostproc.py` and `empa_plays_post3`; discusses frozen versus dynamically trained DDQN and possible LLM representations. These are rough meeting notes, not a validated dataset schema. |
| 2025-10-02 | A-Andrews/EfficientZeroV2 GitHub fork created; [initial BMRC/Ray adaptations](https://github.com/A-Andrews/EfficientZeroV2/commit/145acd7c4dfe6c61b0c703f44eaa7ce1c1b5801f). | Separate EfficientZero development starts. |
| 2025-10-06 | [First VGDL integration](https://github.com/A-Andrews/EfficientZeroV2/commit/09e62d2cd059eb1e78f3e50deac0ae8984aa764c), `development` branch. | EfficientZero is connected to the RC_RL environment, not reimplemented inside RC_RL. |
| 2025-10-15 to 2025-10-16 | [VGDL sweeps](https://github.com/A-Andrews/EfficientZeroV2/commit/7f331f0af07d721dd820a6c0972fc8b8e87b6f1c), then [Avoid George training changes](https://github.com/A-Andrews/EfficientZeroV2/commit/7d232324a89590baf8dd3e1b0291c2722d672cf1). RC_RL [e20f67c](https://github.com/botcs/RC_RL/commit/e20f67c6ce7abc00038a28bac4436d33fb8e3292), authored by Austin on Oct 16, adds optional level transformations to `VGDLEnv.reset` and `utils.load_game`. | Matching changes across the two repositories establish an actual integration lineage. The RC_RL branch's `DDQN_test.sh` uses the EZ environment but is a DDQN launcher, not itself an EfficientZero trainer. |
| 2025-10-16 | RC_RL [README warning and BSON parsing update](https://github.com/botcs/RC_RL/commit/15586a43fe054cdca89edab46fd8efef91228a9c), [original source submodule](https://github.com/botcs/RC_RL/commit/e4b519561bb10c603699b387d00b16445472e22b), [DDQN state tracing](https://github.com/botcs/RC_RL/commit/e15d8de25398f0dff20b9eaa69dac8147ceaaae8). | The warning appeared after the revival work began, and coexists with meaningful baseline feature/replay development. |
| 2025-10-28 to 2025-10-29 | [BMRC setup](https://github.com/A-Andrews/EfficientZeroV2/commit/9126a223ded37a3f877ca46f0a13211469c453b2) is merged into `main`; substantive [integrated VGDL implementation](https://github.com/A-Andrews/EfficientZeroV2/commit/bf13e302dd3f72c6bbaa7b15ae9354eccc89c482) continues on `integrated-development`. | Inspecting only the default branch misses the research implementation. |
| 2025-11-13 | [Curriculum training added](https://github.com/A-Andrews/EfficientZeroV2/commit/8597088064cb5b9e0f3d6f366d87d850e306bd92). | Establishes committed curriculum support by November. |
| 2025-12-11 to 2025-12-22 | [Record evaluation actions](https://github.com/A-Andrews/EfficientZeroV2/commit/9a4e74dadcf88f6e491b35236564c595d65e06c4), [action trimming fix](https://github.com/A-Andrews/EfficientZeroV2/commit/7aca357906b283056de809dc58c95a3057314bf4). | Evaluation trace work precedes later consolidated behavioral analyses. |
| 2026-01-12 to 2026-01-17 | [Curriculum testing](https://github.com/A-Andrews/EfficientZeroV2/commit/f95f15da7d4197cdf5127d66cef945025407ea5b), [pre-S3 preparation](https://github.com/A-Andrews/EfficientZeroV2/commit/fbe03511bbdefd11feedb66015bebe5a490ce693), and [environment capture](https://github.com/A-Andrews/EfficientZeroV2/commit/29157d4892afd9467b1bd0994de1355086145490). RC_RL adds [extra warmup/pretraining levels](https://github.com/botcs/RC_RL/commit/b1e33768b9f1d799d7780a7e6556d328be5174ab) on Jan 17. | The current collaborator branch and RC_RL EZ branch form a preserved January implementation. Extra levels are baseline warmup material, not additional human experimental levels. |
| 2026-01-15 and 2026-01-29 | Selected backed-up W&B run metadata records `ez/train.py`, A-Andrews remote, and source revisions `f95f15d` / `29157d4` respectively (details below). | Actual training jobs used the identified implementation. This alone does not connect those runs to the final paper's selected artifacts. |
| 2026-01-25 and 2026-04-17 | RC_RL [DDQN sampling](https://github.com/botcs/RC_RL/commit/616ff3f43effaba815ec13a5915bf8975d488904), [successful sampling record](https://github.com/botcs/RC_RL/commit/07bd59a6612da5672289fc9c28f23c885d464a22), [replay fixes](https://github.com/botcs/RC_RL/commit/6186f36313be956fe2ea35f85a0a70ea31c5b387). | RC_RL remained active beyond the early reconstruction phase. |
| 2026-05-02 | llm-vgdl commit `f34d6de95b22bfb682f1845c86ab078f1709ef65` adds the submission behavioral pipeline and plotting, including the consolidated baseline inputs. | Later analysis integration is separate from baseline algorithm development. The commit date is not the training date. |

## Ancestry check

`git branch -r --contains` finds each of `fdb74ee6eb343ab861413ca4d8f5d4077e70cd6e`, `a4f1da7d054c3374e64bfbf6b24569c844f22575`, and `3388f9caa9c611e5cd1dbd262c38df24087c9963` only on `origin/fmri`. The merge base of `origin/master` and `origin/fmri` is `8df431a1b5311ad8ff9d1a55e0a3a04076128ab3`, also the direct parent of the first current-team commit `99566fd07cb5d31da16802b2b2422109af289441`. Thus the chronology above distinguishes a preserved parallel upstream branch from the ancestry of today's work.

## Direct dependency evidence

At `A-Andrews/EfficientZeroV2` revision `29157d4892afd9467b1bd0994de1355086145490`:

- [`ez/envs/vgdl/__init__.py`](https://github.com/A-Andrews/EfficientZeroV2/blob/29157d4892afd9467b1bd0994de1355086145490/ez/envs/vgdl/__init__.py) reads `RC_RL_PATH`, explicitly loads RC_RL's `utils.py`, imports `VGDLEnv`, and provides the `RawVGDL` Gym adapter. It also patches in a loader for non-contiguous level IDs.
- [`ez/config/exp/vgdl.yaml`](https://github.com/A-Andrews/EfficientZeroV2/blob/29157d4892afd9467b1bd0994de1355086145490/ez/config/exp/vgdl.yaml) points `game_folder` at `RC_RL/all_games_recovered`, selects RGB image input `[3,96,96]`, and lists the default curriculum `[9,10,11,12,0,1,2,3,4,5,6,7,8]`.
- [`scripts/vgdl_pretrain.sh`](https://github.com/A-Andrews/EfficientZeroV2/blob/29157d4892afd9467b1bd0994de1355086145490/scripts/vgdl_pretrain.sh) selects `vgfmri4_pretrain` with level IDs 0–20.
- The older [`development` wrapper](https://github.com/A-Andrews/EfficientZeroV2/blob/510a61860c2259df442c5fcafdca248d419a6105/ez/envs/vgdl_env.py) uses `RC_RL_ROOT`, imports the same environment, and supplies `reset(level_transform=...)`, directly matching the Oct 16 RC_RL branch change.

The two main wrapper/config S3 blobs, uploaded Feb 24, are byte-for-byte identical to the January GitHub revision. SHA256:

- `ez/envs/vgdl/__init__.py`: `528214c3324194f60f246332b208f8da233fa83cae88a237f750a570e6219b9d`
- `ez/config/exp/vgdl.yaml`: `0616580cf28acc2d4978f488ca7855e7c4556d727783d13b49576dbec667a6e6`

This comparison demonstrates why upload date must not be substituted for source creation date.

Selected training metadata, read only from these small S3 objects:

| S3 key below `brain-wide_strategies/EfficientZeroV2/wandb/` | Recorded start (UTC) | Recorded source commit |
|---|---|---|
| `run-20260115_144456-mcfaluzi/files/wandb-metadata.json` | `2026-01-15T14:44:56.490934Z` | `f95f15da7d4197cdf5127d66cef945025407ea5b` |
| `run-20260129_132906-ckp86at9/files/wandb-metadata.json` | `2026-01-29T13:29:06.389305Z` | `29157d4892afd9467b1bd0994de1355086145490` |

Both report program `ez/train.py` and the A-Andrews origin. Host/user/environment details were not needed or copied.

## Remaining uncertainty and release implications

1. **The exact final EfficientZero feature-extraction version is not yet established.** We have identified the real training/integration repository and actual historical runs. We have not matched final S3 feature artifacts, chosen checkpoints, and run IDs to source commits, nor found the final human-trajectory feature extractor in this January source tree. Do not claim complete end-to-end reproduction yet.
2. **The preserved configuration and current manuscript describe different versions.** The local manuscript at `NEURIPS_WRITING/PLAN_A_MASTER_OVERLEAF/neurips_2026.tex` says per-sprite-type channel inputs and three warmup levels. The preserved January wrapper renders RGB frames and its default configuration has four warmup level IDs (9–12). This may reflect later experiments, overrides, another unlocated extractor/configuration, or documentation drift. The audit has insufficient evidence to decide which. Resolve by matching the final checkpoints and feature datasets to actual configs before presenting the January code as the paper's final implementation.
3. **The README source pointer is useful but insufficient.** The `tsividis/vgdl` reference gives original human/EMPA context. It neither replaces the current team's RC_RL preprocessing nor captures their later DDQN/EfficientZero work.
4. **Meeting notes are historical evidence of plans, not validated cohort facts.** In particular, their rough pilot/cohort count differs from subsequently checked source data. The project's validated cohort split remains sub-01..11 vs sub-12..32; this note does not amend that.
5. **Do not archive these baseline research branches as cleanup during this audit.** User's initial branch-archiving scope named `tomov23-analysis`; it did not authorize archiving RC_RL or the collaborator's EfficientZero repository. Their non-default branches contain substantive code absent from defaults.
6. **Public release is still held.** Future integration should pin both baseline repositories with their roles and versions, preserve separate upstream licenses (official EfficientZero is GPL-3.0), and include the exact final feature-generation provenance once resolved. No vendoring or publication was done here.
