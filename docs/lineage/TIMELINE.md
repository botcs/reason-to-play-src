# Reason to Play: code and analysis lineage

> Historical archaeology report captured before the final source recovery and integration.
> See the [current reproduction guide](../reproducibility.md) and [artifact audit](../release/scientific-provenance.md) for subsequent findings and release state.

Historical reconstruction, 27 September 2026. Prepared from Git history,
repository metadata, archived execution records, and Csaba's account.
This is a provenance report, not a declaration that historical pipelines
produce equivalent results. Subsequent recovery and integration are documented
in the current reproduction guide linked above.

## The four strands

1. **Original human experiment and data.** Tomov's experiment is associated
   with OpenNeuro **ds004323**. The project's own RC_RL README identifies
   `tsividis/vgdl`, branch **refactor_fMRI_cannon**, as the original experimental
   and fMRI-analysis reference. It is distinct from the RL repository that
   initially served as the development starting point.
2. **RL baselines, data recovery and raw preprocessing.** `botcs/RC_RL`
   descends from `tomov/RC_RL`, itself a fork of `ACampero/RC_RL`. Csaba's
   September 2025 work modernized/repaired that code and recovered participant
   gameplay. The actual fMRIPrep launcher lives here, together with DDQN and
   later EfficientZero-related environment changes. The EfficientZero algorithm
   and training code live in Austin Andrews's separate
   `A-Andrews/EfficientZeroV2` fork, which imports the RC_RL environment.
3. **Neural analysis.** `botcs/tomov23-analysis` started as Csaba's AWS-backed,
   stage-based RSA pipeline in October 2025. Sreejan's later imports introduced
   parcelwise and then voxelwise encoding. According to Csaba, this drew on
   Sreejan's prior Nature Communications work, whose official source is
   `tsumers/bert-brains`. Shared authorship and methods are verified; the exact
   file-import relationship is not yet established.
4. **LLM agents, harness, replay and activations.** January 2026 work in
   `botcs/llm-vgdl` derives from Cédric Colas's private `ccolas/infer-vgdl`.
   Colas's `language_and_experience` is a later public snapshot of that source
   lineage. It should not be presented as the repository cloned in January.

## Dated chronology

Dates below are recorded commit dates unless another evidence type is stated.
A commit can postdate offline development; repository creation and research
start are separate events. Recruitment dates come from Csaba's account.

| When | What happened | Evidence |
| --- | --- | --- |
| Before this project | RL implementation ancestry runs through ACampero to `tomov/RC_RL`. Original human/fMRI code lives in the separate VGDL reference branch. | [Public RL source](https://github.com/tomov/RC_RL); [experiment reference](https://github.com/tsividis/vgdl/tree/refactor_fMRI_cannon). |
| 29 June 2024 | Kumar, Sumers and colleagues publish *Shared functional specialization in transformer-based language models and the human brain*. Official code is `bert-brains`; its dataset is Narratives ds002345, not VGDL ds004323. | [Article](https://www.nature.com/articles/s41467-024-49173-5); [archived code](https://doi.org/10.5281/zenodo.10863840). |
| **15 September 2025** | Csaba's first recorded RC_RL work: environment repair, Python 2→3 migration and human replay. This supports the reported September project start. | [99566fd](https://github.com/botcs/RC_RL/commit/99566fd07cb5d31da16802b2b2422109af289441), followed by `9ce1dfa` and `a52eb2d`. Repository created the same day. |
| 19 September–early October 2025 | fMRI exploration notes and visualization; DDQN input, reward, determinism and training-protocol repairs; recovery of participant game/level descriptions. | [fMRI updates b5156d4](https://github.com/botcs/RC_RL/commit/b5156d4cf6bb79e553e3350517e9bcb650e2a5f0); [game recovery 6562d10](https://github.com/botcs/RC_RL/commit/6562d10). |
| **2–6 October 2025** | Austin's separate EfficientZeroV2 fork is created on 2 October; its first VGDL integration is committed on 6 October. The trainer imports the environment from RC_RL. | [A-Andrews/EfficientZeroV2](https://github.com/A-Andrews/EfficientZeroV2); [first VGDL integration 09e62d2](https://github.com/A-Andrews/EfficientZeroV2/commit/09e62d2cd059eb1e78f3e50deac0ae8984aa764c). |
| **16 October 2025** | The README explicitly identifies the original experiment code in `tsividis/vgdl`. Csaba commits the actual **fMRIPrep 24.1.0 launcher**; Austin commits EfficientZero-related environment interfaces on a separate branch. | [reference correction 15586a4](https://github.com/botcs/RC_RL/commit/15586a43fe054cdca89edab46fd8efef91228a9c); [launcher 969141d](https://github.com/botcs/RC_RL/commit/969141d09369ecd783965fadbbe2a1920065cf24); [EZ support e20f67c](https://github.com/botcs/RC_RL/commit/e20f67c). |
| 13–28 October 2025 | Retained derivative configs record fMRIPrep invocations for all 32 participants. Some early subject-02 attempts precede the launcher commit. | 38 S3 TOMLs plus derivative `dataset_description.json`; see the preprocessing evidence below. |
| **28 October 2025** | `tomov23-analysis` begins by uploading already-computed fMRIPrep outputs and adding RC_RL for DDQN extraction. It is a downstream analysis repository, not where raw preprocessing originated. | [root 5c53c93](https://github.com/botcs/tomov23-analysis/commit/5c53c931fce464c5f708e3f416db4330be434fa8); [RC_RL dependency 6f69f9c](https://github.com/botcs/tomov23-analysis/commit/6f69f9c). |
| Late October–November 2025 | Csaba develops named extraction, HRF, RDM, similarity and QC stages, Docker images, config-driven orchestration, S3 paths and AWS Batch infrastructure. | [detailed dated analysis history](tomov23-analysis.md). |
| 29 October–13 November 2025 | Austin's `integrated-development` branch adds the integrated VGDL trainer and then curriculum support. The default branch does not contain the substantive research integration. | [integration bf13e30](https://github.com/A-Andrews/EfficientZeroV2/commit/bf13e302dd3f72c6bbaa7b15ae9354eccc89c482); [curriculum 8597088](https://github.com/A-Andrews/EfficientZeroV2/commit/8597088064cb5b9e0f3d6f366d87d850e306bd92). |
| Around November 2025 | Csaba recruits Sreejan to develop neural alignment/encoding. | Csaba's account; Git does not establish the recruitment date. |
| 14 November 2025 | A separate `tomov23-analysis/llm` scratch branch explores text/grid-based feature extraction. This predates the later autonomous LLM harness and should not be conflated with it. | [dafb69b](https://github.com/botcs/tomov23-analysis/commit/dafb69b2430d18d51c6ce3c386b5376e73a2e47a). |
| **17–18 December 2025** | First Sreejan-authored encoding import: Schaefer parcels, feature alignment, HRR, banded-ridge fitting and SLURM submission. Post-fMRIPrep smoothing/high-pass handling follows. | [3199db9](https://github.com/botcs/tomov23-analysis/commit/3199db9946ada6328fab419d2c5f9fb1bb9c5ca3); [718e09c](https://github.com/botcs/tomov23-analysis/commit/718e09c). These dates do not rule out November development. |
| **11 January 2026** | Csaba begins the LLM-agent branch from Colas's private source: VGFMRI conversion followed by a new `src/llm_eval` implementation. | [49a2bca](https://github.com/botcs/llm-vgdl/commit/49a2bca26ccd03aca5d3d67be9b6d2f3fb71b11c); [initial harness 68fc1024](https://github.com/botcs/llm-vgdl/commit/68fc1024013508633d0eeb00780ae6d5a3cbe74a). |
| 16–29 January 2026 | Minimal LLM scaffolding and replay, cross-trial history, hidden-state extraction, parallel extraction and DeepSeek V3.2 support. In parallel, Austin's EZ branch adds pretraining/warmup levels on 17 January. | [scaffolding ff67a04](https://github.com/botcs/llm-vgdl/commit/ff67a04dd9949e8e267a1f89ccb53daf344ca6b9); [replay 0a918605](https://github.com/botcs/llm-vgdl/commit/0a9186054e068387131d4d9c02579062413910ea); [extraction 63df9438](https://github.com/botcs/llm-vgdl/commit/63df9438); [EZ b1e3376](https://github.com/botcs/RC_RL/commit/b1e3376). |
| 15–29 January 2026 | Preserved W&B metadata identifies actual EfficientZero training starts on 15 and 29 January. The second records source `29157d4`; backed-up wrapper/config files match this source revision byte for byte. | [preserved EZ source 29157d4](https://github.com/A-Andrews/EfficientZeroV2/tree/29157d4892afd9467b1bd0994de1355086145490); [baseline provenance report](baselines.md). These runs are not yet matched to the paper's final selected artifacts. |
| **28 January 2026** | Sreejan commits the voxelwise pipeline, including AR(1) whitening, first-volume removal and EfficientZero feature support. | [3e0f53a](https://github.com/botcs/tomov23-analysis/commit/3e0f53a750c6544fa9a6261adeb04c6441f08803). |
| **12 February 2026** | Colas's public `language_and_experience` source snapshot is committed. The GitHub repository was created on 9 February; exact visibility-transition time is not inferred from that metadata. | [public first commit edb30bc](https://github.com/ccolas/language_and_experience/commit/edb30bc9815efad268605d337d63601d4f6f39a8). |
| **23 March 2026** | Sreejan imports the organized `encoding_model_slurm` pipeline on a separate branch from the earlier AWS history. | [26fa74c](https://github.com/botcs/tomov23-analysis/commit/26fa74ce715781bfdd7a6f6a4bc2a6685e5a8678). |
| March–April 2026 | LLM replay becomes a two-phase imputation/extraction workflow. Multi-turn gameplay, explicit rationale modes and the sliding-window extractor replace earlier harness conventions. | [phase separation 6a2d0aaf](https://github.com/botcs/llm-vgdl/commit/6a2d0aaf); [remove single turn 1717f7e7](https://github.com/botcs/llm-vgdl/commit/1717f7e7); [rationale modes 2d5d8e84](https://github.com/botcs/llm-vgdl/commit/2d5d8e84); [window extractor 13610e70](https://github.com/botcs/llm-vgdl/commit/13610e70). |
| **7–9 May 2026** | The project website is deployed, then the public research code is separated into `reason-to-play-src`. This public export is distinct from the full development history. | [website 115dce3](https://github.com/botcs/reason-to-play/commit/115dce3); [public source 9e5b2c8](https://github.com/botcs/reason-to-play-src/commit/9e5b2c8); [website source move e20c204](https://github.com/botcs/reason-to-play/commit/e20c204). |
| **31 July 2026** | `tomov23-analysis/main` becomes the reduced supplementary-code snapshot. The working reproduction branch continues separately. | [62c2021](https://github.com/botcs/tomov23-analysis/commit/62c202130d057e585b6bf7978f621fcd0255919c); tag `encoding-suppmat-v1.0`. |
| September 2026 | Acceptance and release archaeology; S3 inventory, code completeness and publication planning begin. This lineage audit establishes the source relationships used for consolidation. | Author-reported acceptance; current audit. |

## Raw preprocessing: original code and execution evidence

The original entry point is
[`RC_RL/fmriprep_pipeline/run_fmriprep.sh` at 969141d](https://github.com/botcs/RC_RL/blob/969141d09369ecd783965fadbbe2a1920065cf24/fmriprep_pipeline/run_fmriprep.sh).
It launches Docker image `nipreps/fmriprep:24.1.0` with MNI152NLin2009cAsym
2 mm and anatomical outputs, eight threads, 24 GB, no FreeSurfer surface
reconstruction, registration DOF 9, forced BBR, all confound components,
zero dummy scans and seed 23.

The archived S3 outputs independently record fMRIPrep 24.1.0. The newest
recorded config per subject agrees on these central scientific settings.
One early subject-02 invocation used DOF 6. All 38 configs should remain part
of provenance. Config timestamps establish recorded invocations, not exact
file-by-file assignment of all final derivatives. The launcher specifies
Docker while recorded environment detection says Singularity; image digest
and complete execution-environment identity have not been recovered.
The [invocation index](fmriprep-invocations.csv) lists all 38 configs, their
central settings and source S3 URIs. Timestamps embedded in run UUIDs retain
the source's convention; no timezone has been inferred for them.

Historical launcher revisions also explain why copying its present state is
insufficient: the November 14 revision changes the `all` subject list to just
`22`, and its README retains older exclusions. That is an invocation/edit
history, not evidence that only one subject was processed or that those old
exclusions apply to the final study.

The previously reconstructed wrapper in the unmerged release PR is a new
implementation based on recovered TOMLs. It is **not** the original launcher.
Any future consolidation must reconcile it with this newly located source.

## Exact Git ancestry versus imported scientific code

`botcs/RC_RL` is not flagged as a GitHub fork, but its first Csaba commit has
parent **8df431a1b5311ad8ff9d1a55e0a3a04076128ab3**, exactly the inspected
`tomov/RC_RL` head. Git ancestry establishes derivation despite the hosting
metadata.

There is also a preserved upstream `fmri` branch with DDQN work from 2021–2022.
Those commits are not ancestors of current `master`: both lines share the
2019 base. This is another reason that a default-branch view alone gives an
incomplete picture. The original experiment-source pointer and these retained
baseline branches describe different parts of the project's history.

The first January `llm-vgdl` change has parent
**64dc4f6d4621acdf289819f198ba1ca318b621dd**, verified in private
`ccolas/infer-vgdl` (Colas, 21 August 2025). This is the concrete upstream
boundary. The `botcs/llm-vgdl` GitHub repository was created on 27 January,
after the recorded January 11 development commits.

The EfficientZero fork is independently identified by the S3 backup's Git
remote and branch metadata, direct RC_RL imports, exact wrapper/config matches,
and recorded training-run source commits. This establishes the training
implementation's origin. It does not yet identify the final feature extractor
or connect every selected checkpoint to the manuscript. The preserved January
RGB/four-warmup default differs from the current manuscript's
per-sprite-channel/three-warmup description; the producing experiment revision
must be recovered before treating either as the final configuration. See
[the baseline report](baselines.md).

The analysis repository has a more complicated structure:

```text
AWS/RSA history
  Dec 03 d0fc2a0
    |-- Dec 17 3199db9 -> Dec 18 718e09c -> Jan 28 3e0f53a
    |                                      encoding_model branch
    |
    `-- Dec 11 d8eaa5a
          |-- Mar 23 26fa74c -> later reproduction/geometry work
          |                     encoding_model-reproduction_temporal branch
          |
          `-- Jul 31 62c2021
                         submitted main, encoding-suppmat-v1.0
```

The March and July snapshots are sibling Git children. Nevertheless, their
preprocessing file has the same Git blob, **3f78dbfa617cca456b0e9301a879e93820a7ea77**:
content reuse is proven. Their aligners and encoders differ. Thus an arrow
for source reuse must be distinguished from a Git-parent arrow.

Meaningful differences include parcel versus voxel targets, lag direction,
regularization grids, default nuisance bands and whether an aligner creates
the base BOLD/behavioral data or expects it already present. These are reasons
to identify the producing revision of each result, not judgments about the
quality of an individual contributor's work. Details and commit links are in
[the analysis-history report](tomov23-analysis.md).

For the Nature source, Csaba reports adaptation from Sreejan's prior work.
Official code attribution, shared authorship and methodological similarities
are verified. A direct file copy into the first VGDL encoding import has not
been demonstrated. The diagram therefore labels that edge as a reported
adaptation rather than verified Git ancestry. See
[the Nature-source report](nature-source.md).

## Release state and outstanding archaeology

- No original analysis branch has been deleted. Three archive tags and a
  verified Git bundle were created before the hold.
- The preprocessing PR is unmerged; the public code consolidation and parent
  submodule-pin changes remain uncommitted. No dataset has been uploaded.
- The earlier website/Hugging Face **paper** link change was already merged
  and deployed before the hold. The actual dataset link remains pending.
- Open questions are the final EfficientZero extractor/checkpoint/config
  mapping, the Nature-to-VGDL source adaptation, and the exact model/
  checkpoint/config/analysis revision that produced each selected S3 result.

Private repository links require access. This report and the standalone
diagrams can be shared independently; they contain no credentials or data
payloads. Older meeting notes and READMEs are evidence of what was believed
at the time, not substitutes for the verified dataset invariants.
