# Sources

**TL;DR:** Reason to Play builds on the VGDL-fMRI human experiment, RC_RL and
EfficientZero baselines, earlier neural analyses, and Cedric Colas's LLM harness.
This page records their origins; the [workflow guide](../reproducibility.md)
provides commands.

![Project lineage](lineage.svg)

![Neural-analysis branch relationships](lineage-analysis-branches.svg)

[Download the two-page PDF](lineage-all.pdf)

## Sources and implementations

| Research component | Origin | Current implementation or reference |
| --- | --- | --- |
| Original human experiment and EMPA | [tsividis/vgdl, refactor_fMRI_cannon](https://github.com/tsividis/vgdl/tree/refactor_fMRI_cannon); [OpenNeuro ds004323 v1.0.0](https://doi.org/10.18112/openneuro.ds004323.v1.0.0) | [Original behavioural schema](tomov23-behavior-notes.md); [public JSON specification](../data-format.md) |
| DDQN and baseline VGDL environments | [tomov/RC_RL](https://github.com/tomov/RC_RL), followed by [botcs/RC_RL](https://github.com/botcs/RC_RL) | [Baseline guide](../guides/baselines.md); [source pins](../../agents/ddqn/sources.json) |
| EfficientZero integration | Austin Andrews's [EfficientZeroV2 fork](https://github.com/A-Andrews/EfficientZeroV2/tree/29157d4892afd9467b1bd0994de1355086145490), based on [EfficientZeroV2](https://github.com/Shengjiewang-Jason/EfficientZeroV2) and using RC_RL environments | [Source pins](../../agents/efficientzero/sources.json); [extraction and training guide](../guides/baselines.md#efficientzero-hidden-features) |
| Raw fMRI preprocessing | [RC_RL launcher, October 2025](https://github.com/botcs/RC_RL/blob/969141d09369ecd783965fadbbe2a1920065cf24/fmriprep_pipeline/run_fmriprep.sh) | [Launcher/configuration records](../../reconstruction/tomov23/configs); [preprocessing guide](../guides/fmri-preprocessing.md) |
| Neural analysis | [botcs/tomov23-analysis](https://github.com/botcs/tomov23-analysis), with reported methodological influence from [bert-brains](https://github.com/tsumers/bert-brains) | [Per-file source map](code-origins.json); [analysis methods](../analysis-methods.md) |
| Language-model harness and game interpreter | January 2026 fork of Colas's private `ccolas/infer-vgdl`; related public [language_and_experience](https://github.com/ccolas/language_and_experience) | `agents/lrm/`, `environments/vgdl/`, `environments/definitions/`, `agents/lrm/prompts/` |
| DeepSeek extraction runtimes | Upstream V3.2 and V4 code with local loading/feature-hook adaptations | [V3.2 source record](../../agents/lrm/backends/deepseek_v32/PROVENANCE.json), [V4 source record](../../agents/lrm/backends/deepseek_v4/PROVENANCE.json); [runtime and license notices](../../THIRD_PARTY.md); [extraction guide](../reproducibility.md#optional-deepseek-runtimes) |

These sources establish code origins and methodological influences.
Unresolved links to the checkpoints and configurations behind reported results
are listed in [reproduction limits](../reproduction-limits.md).

## Research timeline

Dates are commit dates unless stated otherwise; work may have begun earlier.

| Date | Research development | Evidence |
| --- | --- | --- |
| 29 June 2024 | Kumar, Sumers and colleagues publish *Shared functional specialization in transformer-based language models and the human brain*, using Narratives fMRI. | [Nature Communications paper](https://www.nature.com/articles/s41467-024-49173-5), [official source archive](https://doi.org/10.5281/zenodo.10863840) |
| 15 September 2025 | Csaba's RC_RL work begins with environment, Python 3 and participant-replay changes. | [First current-team commit 99566fd](https://github.com/botcs/RC_RL/commit/99566fd07cb5d31da16802b2b2422109af289441) |
| September–October 2025 | DDQN input representation, training protocol and tracing work; identification of the original human/EMPA source. | [One-hot inputs](https://github.com/botcs/RC_RL/commit/0bb49b3c940aaabd91a5d362a8067e67db11b3d5), [source-reference correction](https://github.com/botcs/RC_RL/commit/15586a43fe054cdca89edab46fd8efef91228a9c) |
| 6 October 2025 | Austin's EfficientZero fork adds VGDL integration. | [09e62d2](https://github.com/A-Andrews/EfficientZeroV2/commit/09e62d2cd059eb1e78f3e50deac0ae8984aa764c) |
| 13–28 October 2025 | Retained fMRIPrep configurations record invocations across all 32 participants; the RC_RL launcher is committed on 16 October. | [Invocation index](fmriprep-invocations.csv), [launcher 969141d](https://github.com/botcs/RC_RL/commit/969141d09369ecd783965fadbbe2a1920065cf24) |
| 28 October–December 2025 | `tomov23-analysis` begins with existing fMRIPrep derivatives and develops an AWS-integrated staged RSA pipeline: features, HRF convolution, RDMs, similarity and QC. | [Root commit 5c53c93](https://github.com/botcs/tomov23-analysis/commit/5c53c931fce464c5f708e3f416db4330be434fa8), [DDQN dependency](https://github.com/botcs/tomov23-analysis/commit/6f69f9c) |
| Around November 2025 | Sreejan joins to develop neural alignment and encoding. | Author's account; Git does not establish recruitment dates. |
| 17–18 December 2025 | First committed Sreejan-authored parcelwise encoding pipeline, alignment and post-fMRIPrep preprocessing. | [3199db9](https://github.com/botcs/tomov23-analysis/commit/3199db9946ada6328fab419d2c5f9fb1bb9c5ca3), [718e09c](https://github.com/botcs/tomov23-analysis/commit/718e09c) |
| 11 January 2026 | Csaba starts the LLM game-agent harness from Colas's private source. | [Fork boundary 49a2bca](https://github.com/botcs/llm-vgdl/commit/49a2bca26ccd03aca5d3d67be9b6d2f3fb71b11c), [initial harness](https://github.com/botcs/llm-vgdl/commit/68fc1024013508633d0eeb00780ae6d5a3cbe74a) |
| January 2026 | Replay, hidden-state extraction and DeepSeek V3.2 support develop alongside baseline curriculum work. Voxelwise neural analysis is committed on 28 January. | [Feature extraction](https://github.com/botcs/llm-vgdl/commit/63df9438), [voxelwise pipeline 3e0f53a](https://github.com/botcs/tomov23-analysis/commit/3e0f53a750c6544fa9a6261adeb04c6441f08803) |
| 12 February 2026 | Colas's related public `language_and_experience` snapshot is committed. It is distinct from the source used in January. | [edb30bc](https://github.com/ccolas/language_and_experience/commit/edb30bc9815efad268605d337d63601d4f6f39a8) |
| March–April 2026 | Separate SLURM encoding import; LLM replay separates imputation/extraction and adopts explicit rationale modes and sliding windows. | [Encoding import 26fa74c](https://github.com/botcs/tomov23-analysis/commit/26fa74ce715781bfdd7a6f6a4bc2a6685e5a8678), [phase separation](https://github.com/botcs/llm-vgdl/commit/6a2d0aaf), [sliding windows](https://github.com/botcs/llm-vgdl/commit/13610e70) |
| May 2026 | Project website and separate public research-code repository appear. | [Website source](https://github.com/botcs/reason-to-play/commit/115dce3), [research-code export](https://github.com/botcs/reason-to-play-src/commit/9e5b2c8) |
| 31 July 2026 | The supplementary neural-code snapshot is committed separately from the continuing research branch. | [62c2021](https://github.com/botcs/tomov23-analysis/commit/62c202130d057e585b6bf7978f621fcd0255919c) |

## Reinforcement-learning baselines

Current-team RC_RL work starts from Tomov revision
`8df431a1b5311ad8ff9d1a55e0a3a04076128ab3`. The separate upstream `fmri` branch
contains 2021–2022 DDQN/fMRI work outside that `master` ancestry. RC_RL's human
experiment reference points to `tsividis/vgdl`; RC_RL contains the later baseline
and environment work.

Austin Andrews's EfficientZero fork
[imports RC_RL's environment](https://github.com/A-Andrews/EfficientZeroV2/blob/29157d4892afd9467b1bd0994de1355086145490/ez/envs/vgdl/__init__.py)
as a dependency. Its `development` and `integrated-development` branches contain
[October integration](https://github.com/A-Andrews/EfficientZeroV2/commit/bf13e302dd3f72c6bbaa7b15ae9354eccc89c482)
and [November curriculum support](https://github.com/A-Andrews/EfficientZeroV2/commit/8597088064cb5b9e0f3d6f366d87d850e306bd92).

The January configuration uses RGB observations and four warmup levels (9–12).
Its connection to the paper's checkpoints and result layers remains
[unresolved](../reproduction-limits.md#baseline-checkpoints-and-layer-mappings).

## Neural-analysis sources

The December parcelwise and January voxelwise pipelines descend from the
AWS/RSA history. The March `encoding_model_slurm` import and July supplementary
snapshot share parent `d8eaa5a6c7fc3382967cb9e004748c89c99eb526` and preprocessing
blob `3f78dbfa617cca456b0e9301a879e93820a7ea77`; their aligners and encoders differ.

Sreejan's preceding Nature paper used Narratives **ds002345** and
`tsumers/bert-brains`. The project author reports adapting its banded ridge
regression, delayed features, correlation scoring and nuisance inputs. Direct
source-file copying has not been established; the diagrams label this link as
reported adaptation.

The [official Zenodo archive](https://doi.org/10.5281/zenodo.10863840) corresponds
to [the camera-ready source](https://github.com/tsumers/bert-brains/tree/26d4da6c1419fc8e635351ddb692897e2d0a1cc5).
The repository retains GPL-3.0; archive metadata states CC-BY-4.0. Source-file
notices still apply. See [analysis methods](../analysis-methods.md) for current
VGDL analysis settings.

## fMRI preprocessing sources

The original raw-data launcher belongs to RC_RL. It specifies fMRIPrep 24.1.0,
MNI152NLin2009cAsym 2 mm output, registration DOF 9, forced BBR, zero dummy
scans, seed 23 and no FreeSurfer surface reconstruction. Its shell and README
are retained with [source checksums](../../reconstruction/tomov23/configs/original-launcher/source.json).

The 38 configurations cover all 32 participants. An early sub-02 invocation
used DOF 6; later records use DOF 9. The launcher uses Docker and the execution
records report Singularity; exact producing image digests remain unknown.

Early `tomov23-analysis` stages processed existing derivatives, applying
T1w-to-MNI transforms with `antsApplyTransforms`. The current
[preprocessing guide](../guides/fmri-preprocessing.md) uses directly generated
MNI derivatives.

## Game translation

Current game descriptions use the Colas interpreter's dialect, shared with the
browser interpreter. Each replay contains one translated `game_description`.
Recorded trajectories establish what participants saw; replay rendering does
not simulate the original experiment again.

[`convert_tomov23_games.py`](convert_tomov23_games.py) preserves the historical
Tomov-to-Colas translation script. Its last recorded edit is
[6dfe6576, 22 March 2026](https://github.com/botcs/llm-vgdl/commit/6dfe65760f3934eb1eae6a5765e14a67ad1297c7).
Source snapshot SHA-256: `2a09720843c5b665c43b92671cd19e54421a17955c46623582ae3d9228fc9696`.
The included entrypoint takes explicit input/output directories and validates
with the current packaged interpreter; its translation rules follow that source.

The script translates sprite hierarchies, colours, level mappings, scoring,
boundary interactions and termination parameters. Its history includes January
conversion fixes, February's Zelda indentation correction and March's scoring
effect change. Later manual edits include plaqueAttack scoring, so use
[`environments/definitions/`](../../environments/definitions) for current games.

To inspect the conversion, supply explicit input and output directories:

```sh
mkdir -p /tmp/translated-vgdl-inspection
python docs/sources/convert_tomov23_games.py \
  --source /path/to/RC_RL/all_games \
  --target /tmp/translated-vgdl-inspection \
  --dry-run
```

`--dry-run` lists files without validating the conversion; omit it to write to
the target. The [original behavioural reference](tomov23-behavior-notes.md)
documents additional source fields and known documentation errors. For released
human files, use the [public JSON specification](../data-format.md).

## Evidence and figures

Solid arrows show Git ancestry; dotted arrows show reused content or
dependencies; the dashed Nature-to-VGDL arrow shows reported adaptation.
Dates mark milestones on an unscaled axis. Private repository links require access.

[`generate_lineage.py`](generate_lineage.py) uses Matplotlib to write both figures
as SVG, PNG and PDF, plus the combined `lineage-all.pdf`, beside the script.
