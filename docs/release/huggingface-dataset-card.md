---
pretty_name: VGDL-fMRI — Reason to Play
license: mit
language:
- en
tags:
- neurips-2026
- neuroscience
- fmri
- video-games
- reinforcement-learning
- model-representations
---

# VGDL-fMRI: Reason to Play

**Publication is in progress.** The `files` catalogue lists verified downloads;
`planned_files` describes the complete selection. Use a dataset commit to pin
your downloads.

VGDL-fMRI connects human video-game learning, fMRI recordings, model gameplay
and model representations. It accompanies
[Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners](https://openreview.net/forum?id=Y1oX1yuaWM),
accepted at **NeurIPS 2026**.

[Research code](https://github.com/botcs/reason-to-play-src) ·
[Interactive results](https://botcs.github.io/reason-to-play/) ·
[Original human dataset](https://doi.org/10.18112/openneuro.ds004323.v1.0.0)

## Contents

The dataset contains self-contained human behaviour, model runs and the inputs
for behavioural and neural analyses. The release manifest records file counts,
sizes and checksums.

| Component | Contents |
| --- | --- |
| Human behavior | One complete human JSON per participant and game, covering levels and attempts across scanner runs, with every recorded engine state, original timestamps, inputs/events, play IDs and nullable outcomes; selected task behaviour from 32 participants |
| Neural analysis inputs | Processed BOLD, DDQN/theory arrays, nuisance variables, sample order, model features sampled at scanner times, and the atlas/common mask used for ROI analysis |
| Model representations | Headline LLM activations, earlier-cohort comparisons and selected controls |
| Generated behavior | Model gameplay with its system prompts, observations, actions and rationales in each replay file |
| Baselines and analyses | Candidate baseline traces, selected checkpoints and neural/behavioral results |
| Catalogues and source records | Dataset paths and checksums, participant, model and condition IDs, source attribution and model/analysis settings |

```text
behavior/
  human/sub-XX/GAME/CONDITION.human.replay.json.gz
  lrm/
  ddqn/episode-history.json
  efficientzero/GAME/episodes.csv
  empa/GAME/trial-XX/level-YY.json
features/
  lrm/
  ddqn/
  efficientzero/
  empa/theory-regressors.json.gz
neural/
  sub-XX/
    bold.npz
    samples.npz
    nuisance.npz
    model-features/
      ddqn.npz
      efficientzero.npz
      lrm/MODEL/CONDITION--SELECTION--STREAM.npz
      empa/hrr.npz
      empa/hrr-decomposed.npz
  atlas/
results/
  behavioral/
  neural/
checkpoints/
reconstruction/tomov23/fmriprep/
website-assets/
catalog/
manifest.jsonl.gz
```

`behavior/` holds recorded gameplay, including replay JSON files. `features/`
holds activations associated with recorded frames or selected prompt steps.
`neural/sub-XX/model-features/` holds representations sampled at that
participant's retained scanner times. These time bases are distinct.

Each participant's `bold.npz` contains processed BOLD and its voxel mask;
`samples.npz` contains ordered sample/play identities and timing;
`nuisance.npz` contains measured button, score and time variables.
DDQN, EfficientZero, LRM and EMPA arrays are peers under `model-features/`.
Selected fMRIPrep outputs under `reconstruction/` support optional
reconstruction from raw MRI; ordinary analysis starts with these processed inputs.
Website assets contain the browser's RDM exports and indexes. Replay links
refer to the files under `behavior/`.

EfficientZero provides **32 scanner-sampled archives with 11 named hooks each**.
All 6,642 plays contributing to the 46,581 retained scanner samples have source
features for every included hook. The raw trace collection covers 6,985 of the
6,994 task attempts; the nine unavailable traces concern attempts outside this
retained sample selection. Per-layer coverage describes source availability,
not whether every scanner bin has a nonzero feature vector.

The adjacent alignment association records the extractor hook names and their
coverage. These names do not establish a mapping to the numeric layer labels in
archived paper results, or identify the exact checkpoint that produced every
original trace. Each EfficientZero archive's manifest entry also records its human
JSON inputs and checksums, sampling policy, and BOLD/sample association.

Raw MRI is available from [OpenNeuro ds004323, version 1.0.0](https://doi.org/10.18112/openneuro.ds004323.v1.0.0).
Historical and unreported feature extractions are excluded. The selection
also omits **630 raw random-initialization and context-ablation tensors** used
by paper controls. Their prompt records, selected aligned features and results
remain included. Recomputing their alignment requires the omitted tensors or
another extraction run; exact regeneration is not guaranteed.

## Behavioral analysis inputs

`behavior/ddqn/episode-history.json` contains all 170,546 recorded episodes
from the 88 selected DDQN runs. The separate
`results/behavioral/retained-episode-table.csv` contains the table used for the
retained behavioral summaries; its DDQN rows contain 32,818 sampled episodes
from those same runs. Use that table to render its summaries and the full
histories when an analysis requires every episode.

Model gameplay consists of 193 selected, self-contained replay files.
Continued runs include their preceding gameplay. Attempt numbers can restart
within a continued run; chronological frame boundaries distinguish attempts.

## Human behaviour files and prompt records

The human data contains **576 JSON files**: 192 participant/game pairs across
32 participants, with three prompt conditions each. Together these files occupy
**411,040,889 compressed bytes** (411 MB); one condition is approximately 137 MB.

Each human behaviour JSON contains one participant's plays of one game across
levels, attempts and scanner runs. The path is
`behavior/human/sub-XX/GAME/CONDITION.human.replay.json.gz`. The three human
prompt conditions (`elaborate`, `minimal`, `oracle`) each embed the measured
trajectory; analysis readers select `elaborate` by default to count each
observation once. Every play retains its original identity, scanner-run
association and timing. A frame is one recorded engine state, including states
with no action. Timestamps retain the original clock; states are not invented
in gaps between plays. Each file carries the translated game description used
by the Colas-based Python engine and browser interpreter, its trajectory and
scanner metadata. The file stores sprite positions rather than screenshots
or an additional ASCII trajectory.

Human files retain fractional positions from participants' recorded
rendering rectangles. Inspect the trajectories with the
[web replay viewer](https://botcs.github.io/reason-to-play/replay.html);
Python code supplies numerical and image inputs for analysis and model runs.

Each replay embeds its per-game conversation alongside its trajectory. Selected
steps retain the original play ID and frame index for correspondence with model
features; opening the replay does not require fetching a separate human file.
The release includes action-only human prompts and model-generated gameplay;
`.narration` and `.imputed` replay files are excluded because they were not used
in the paper. Exact prompts remain model inputs. Model-generated gameplay
rationales are never participant reports.
See the [data format guide](https://github.com/botcs/reason-to-play-src/blob/main/docs/data-format.md)
for schemas, observation identities and the meanings of frames and actions.
The [game-translation history](https://github.com/botcs/reason-to-play-src/blob/main/docs/sources/README.md#game-translation)
includes the historical translator; the
[original behavioural schema reference](https://github.com/botcs/reason-to-play-src/blob/main/docs/sources/tomov23-behavior-notes.md)
supports further source-data archaeology.

## Catalogues and downloads

The publication package provides file and human-play catalogues. Each artifact
row records its dataset path, source attribution and publication state, with a
content checksum once the payload has been byte-verified. Planned entries can
have a null checksum until their bytes have been read and checked. The human-play catalogue contains **6,994 task attempts**
covering **1,661,744 recorded engine states**, counted once across prompt
conditions. Practice attempts are excluded from this task dataset; upper-level
and cohort flags remain available for choosing an analysis.

During incremental publication, `planned_files` describes the complete
selection. `files` contains only uploaded, byte-verified payloads. `human_plays` is a metadata
table; its availability does not imply that every referenced payload is online.
All configurations use a `data` split, which is not a train/test split.

Load the catalogues with ordinary Python calls:

```python
from datasets import load_dataset

revision = "REPLACE_WITH_VERIFIED_DATASET_COMMIT"
plan = load_dataset(
    "csbotos/reason-to-play", "planned_files", split="data", revision=revision
)
plays = load_dataset(
    "csbotos/reason-to-play", "human_plays", split="data", revision=revision
)
```

Catalogue loading fetches metadata. In the file catalogue, `release_path` and
`sha256` identify a payload. In `human_plays`, the corresponding columns are
`source_release_path` and `source_payload_sha256`; several attempts can point to
the same participant/game JSON. These hashes describe the compressed bytes.
The release manifest also records each human file's schema,
counts, checksum and participant/model/condition IDs. Original S3 object identifiers are
source attribution; use dataset-relative paths for Hugging Face downloads.

## Using the research code

The code supports model gameplay and human-to-model neural comparisons through
Python functions and direct module calls. The
[workflow guide](https://github.com/botcs/reason-to-play-src/blob/main/docs/reproducibility.md)
contains installation and execution instructions for gameplay, human replay,
activation extraction, fMRI preprocessing and analysis. The top-level
[AGENTS.md](https://github.com/botcs/reason-to-play-src/blob/main/AGENTS.md)
helps coding agents navigate the repository. Commands and implementation paths
are maintained in the code documentation.

The [dataset analysis guide](https://github.com/botcs/reason-to-play-src/blob/main/docs/guides/dataset-analysis.md)
starts from the downloaded database: ordinary behavioral and neural analysis
requires neither OpenNeuro BSON nor raw MRI, private AWS access or an online
atlas download. Optional raw-data reconstruction is documented separately.
EfficientZero feature extraction uses the included inference and environment
code. Its optional pinned upstream training submodule is needed only for
training; initialization and its separate environment are covered in the
[baseline guide](https://github.com/botcs/reason-to-play-src/blob/main/docs/guides/baselines.md#efficientzero-training).
Full retraining and exact producing-revision attribution for every checkpoint
remain outside the validated scope.

Separate model-feature archives have an adjacent `.npz.alignment.json`
association record. Download it with the archive: the encoding reader checks
the feature hash and its binding to both `bold.npz` and `samples.npz`,
including their sample order, before fitting. BOLD and nuisance files also have
adjacent associations; download those with the participant files. Input coverage is recorded separately from alignment verification;
zero-filled rows for unavailable source plays are not model observations.
Encoding results preserve declared coverage in `alignment_verification_json`.
Coverage intervals refer to the original ordered samples. The documented
fitting mask remains a separate numerical rule; see the dataset analysis guide
for its policy and the recorded sample counts.

Use the code and dataset revisions recorded for the release when reproducing
paper results. New experiments should state their own cohorts, models,
conditions and analysis settings.

## Interpretation and limitations

Human outcomes are three-valued; death events distinguish avatar deaths from
incomplete recordings. Engine frames, non-idle keypress frames and model
decisions are separate clocks. Practice/cohort/level exclusions must remain
explicit. Generated rationales are model outputs, not participant reports.

The [reproduction limits](https://github.com/botcs/reason-to-play-src/blob/main/docs/reproduction-limits.md)
describes missing feature/result cells, unresolved checkpoint attribution and
the EfficientZero 15-hook versus 11-layer-label mapping. These gaps limit exact
paper reproduction. Code smoke tests do not establish full MRI preprocessing or
GPU-scale feature reproduction.

## License and citation

MIT covers original project code and newly created artifacts. Original
OpenNeuro records remain CC0. Third-party source, game assets and model weights
retain their own terms; component metadata preserves these distinctions. The
redistributed AAL SPM12 atlas retains its upstream GPL notice and source archive;
it is not relicensed under MIT.

Please cite the **NeurIPS 2026** paper using
[CITATION.cff](https://github.com/botcs/reason-to-play-src/blob/main/CITATION.cff),
the canonical citation maintained with the research code. The publication
package carries a copy of this file; proceedings volume, pages and DOI are left
unset until confirmed.

Also cite Tomov et al., *The neural architecture of theory-based reinforcement
learning* (2023), and [OpenNeuro ds004323 v1.0.0](https://doi.org/10.18112/openneuro.ds004323.v1.0.0).
For reproducibility questions, use the
[code issue tracker](https://github.com/botcs/reason-to-play-src/issues).
