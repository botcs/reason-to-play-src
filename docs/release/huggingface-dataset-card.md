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

Human video-game learning, fMRI recordings, model gameplay and representations for
[Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners](https://openreview.net/forum?id=Y1oX1yuaWM),
accepted at **NeurIPS 2026**.

[Research code](https://github.com/botcs/reason-to-play-src) ·
[Interactive results](https://botcs.github.io/reason-to-play/) ·
[Original human dataset](https://doi.org/10.18112/openneuro.ds004323.v1.0.0)

**TL;DR:** Explore the replays on the website, or download the human recordings,
model features and processed fMRI inputs for analysis. The complete release is
**33,557 files, 4.80 TB** (4,804,604,182,915 bytes); use the catalogues below to
select files and pin your downloads to a dataset commit.

## Contents

| Component | Contents |
| --- | --- |
| Human behavior | Recorded trajectories and prompts from 32 participants |
| Neural analysis inputs | Processed BOLD, ordered samples, nuisance variables, model features sampled at scanner times, and the atlas/common mask for ROI analysis |
| Model representations | Headline LRM activations, earlier-cohort comparisons and selected controls |
| Generated behavior | Gameplay with system prompts, observations, actions and rationales |
| Baselines and analyses | Candidate baseline traces, selected checkpoints and neural/behavioral results |
| Catalogues | File paths, sizes, checksums, participant/model/condition IDs, source attribution and analysis settings |

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

`behavior/` holds gameplay; `features/` holds activations at recorded frames or
selected prompt steps; `neural/sub-XX/model-features/` holds representations at
retained scanner times. These time bases are distinct. Optional fMRIPrep outputs
under `reconstruction/` support processing from
[raw OpenNeuro MRI](https://doi.org/10.18112/openneuro.ds004323.v1.0.0).

`website-assets/` contains RDM exports, indexes and **384 compact replay files**
(147,759,422 bytes). The copies under `replays/human/` and `replays/lrm/` retain
displayed frames, fractional positions, timing and complete conversations;
they omit engine and raw-input fields unused by the browser. Use the complete
`behavior/` recordings for analysis. See the
[replay format](https://github.com/botcs/reason-to-play-src/blob/main/docs/data-format.md#website-replay-copies).

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

The **576 human JSON files** cover 192 participant/game pairs and three prompt
conditions (`elaborate`, `minimal`, `oracle`). They occupy **411,040,889 compressed
bytes** (411 MB); one condition is approximately 137 MB. Conditions repeat the
same measured trajectory. Analysis readers select `elaborate` by default to
count each observation once.

Each `behavior/human/sub-XX/GAME/CONDITION.human.replay.json.gz` contains all
plays across levels, attempts and scanner runs for that participant/game, with
the translated game description, exact prompts and scanner metadata. It retains
play IDs, nullable outcomes, original timestamps, inputs/events and every
recorded engine state, including states with no action. No states are invented
between plays. Sprite positions retain fractional rendering coordinates;
screenshots and duplicate ASCII trajectories are not stored.

Selected prompt steps retain their play ID and frame index for correspondence
with model features. The release includes action-only human prompts and
model-generated gameplay. `.narration` and `.imputed` files were not used in the
paper and are excluded. Generated rationales are not participant reports.
See the [data format guide](https://github.com/botcs/reason-to-play-src/blob/main/docs/data-format.md)
for schemas and the [source references](https://github.com/botcs/reason-to-play-src/blob/main/docs/sources/README.md)
for game translation and the original behavioral schema.

## Catalogues and downloads

`files` lists every released payload with its path, source attribution, verified
byte count and SHA-256 checksum. `human_plays` links **6,994 task attempts** and
**1,661,744 recorded engine states** to their replay files, counted once across
prompt conditions. Practice attempts are excluded; cohort and upper-level flags
support further selection. Both catalogues use a `data` split, not train/test
splits.

```python
from datasets import load_dataset

revision = "0c674c3ff19b64a55f3fba6d862f5fb828292b74"
files = load_dataset(
    "csbotos/reason-to-play", "files", split="data", revision=revision
)
plays = load_dataset(
    "csbotos/reason-to-play", "human_plays", split="data", revision=revision
)
```

This fetches metadata. In `files`, the `release_path` column identifies each
payload and `sha256` hashes its downloaded bytes. The corresponding `human_plays`
columns are `source_release_path` and `source_payload_sha256`; several attempts
can share one JSON file. Use these dataset-relative paths for downloads;
original S3 identifiers record source attribution.

For example, download one participant/game file at the same revision:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="csbotos/reason-to-play",
    repo_type="dataset",
    filename="behavior/human/sub-01/bait_vgfmri3/elaborate.human.replay.json.gz",
    revision=revision,
)
```

## Using the research code

The
[workflow guide](https://github.com/botcs/reason-to-play-src/blob/main/docs/reproducibility.md)
provides installation and direct Python commands for gameplay, replay,
activation extraction, fMRI processing and analysis. Start with the
[dataset analysis guide](https://github.com/botcs/reason-to-play-src/blob/main/docs/guides/dataset-analysis.md)
for behavioral and neural comparisons from these downloads; raw MRI, BSON,
private AWS access and additional atlas downloads are unnecessary.

For neural fits, download a participant's `bold.npz` (BOLD and voxel mask),
`samples.npz` (ordered play/sample identities and timing), `nuisance.npz`
(button, score and time variables) and selected `model-features/` archives.
Download the adjacent `.npz.alignment.json` records for BOLD, nuisance and
features too. Readers verify file hashes and the exact BOLD/sample order before
fitting. ROI analysis uses the supplied atlas and common mask across all 32
participants, including when analyzing a smaller cohort.

Coverage is separate from alignment: zero-filled intervals for unavailable
source plays are not model observations. Results retain coverage in
`alignment_verification_json`, with intervals in the original sample order.
The fitting mask is a separate numerical rule documented in the analysis guide.

## Interpretation and limitations

Human outcomes are three-valued; check death events before classifying a null
outcome as incomplete. Engine frames, non-idle keypress frames and model
decisions are separate clocks. Keep cohort, level and practice exclusions
explicit when selecting comparisons.

EfficientZero has **32 scanner-sampled archives with 11 named hooks each**.
All 6,642 plays contributing to 46,581 retained scanner samples have source
features for every included hook. Raw traces cover 6,985 of 6,994 task attempts;
the nine missing traces fall outside this retained selection. Coverage describes
source availability, not nonzero feature vectors in every scanner bin. The
alignment records identify hooks and coverage; manifest entries identify human
inputs and checksums, sampling policy and BOLD/sample associations.

The extractor has 15 hooks, but the exact mapping of the 11 selected hooks to
archived numeric layer labels remains unverified, as do the producing checkpoint
and revision for every original trace. Included EfficientZero code supports
extraction; the optional [training submodule](https://github.com/botcs/reason-to-play-src/blob/main/docs/guides/baselines.md#efficientzero-training)
is needed only for training. Full retraining has not been validated.

Historical and unreported extractions are excluded, as are **630 raw
random-initialization and context-ablation tensors** used by paper controls.
Their prompts, selected aligned features and results remain included.
Recomputing alignment requires the omitted tensors or another extraction;
exact regeneration is not guaranteed. The
[reproduction limits](https://github.com/botcs/reason-to-play-src/blob/main/docs/reproduction-limits.md)
also document missing feature/result cells and checkpoint attribution gaps.
Smoke tests do not establish full MRI preprocessing or GPU-scale reproduction.

Use recorded code and dataset revisions for paper reproduction. State cohorts,
models, conditions and analysis settings explicitly for new experiments.

## License and citation

MIT covers original project code and newly created artifacts. Original
OpenNeuro records remain CC0. Third-party source, game assets and model weights
retain their own terms; component metadata preserves these distinctions. The
redistributed AAL SPM12 atlas retains its upstream GPL notice and source archive;
it is not relicensed under MIT.

Please cite the **NeurIPS 2026** paper using
[CITATION.cff](https://github.com/botcs/reason-to-play-src/blob/main/CITATION.cff),
the canonical citation maintained with the research code.

Also cite Tomov et al., *The neural architecture of theory-based reinforcement
learning* (2023), and [OpenNeuro ds004323 v1.0.0](https://doi.org/10.18112/openneuro.ds004323.v1.0.0).
For reproducibility questions, use the
[code issue tracker](https://github.com/botcs/reason-to-play-src/issues).
