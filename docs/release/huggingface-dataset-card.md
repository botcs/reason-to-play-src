---
pretty_name: Reason to Play
license: mit
language:
- en
tags:
- arxiv:2605.08019
- fmri
- reinforcement-learning
- cognitive-science
---

# Reason to Play — dataset card draft

**Prepared draft; dataset not published by this code release.** Intended target:
`csbotos/reason-to-play`. Complete this card against the actual uploaded subset
and immutable dataset/code revisions. The MIT label covers original project
contributions; inherited component terms remain in force.

This dataset is planned to accompany
[Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners](https://arxiv.org/abs/2605.08019).
It connects human video-game learning, fMRI, model gameplay and model
representations. [Code](https://github.com/botcs/reason-to-play-src),
[interactive results](https://botcs.github.io/reason-to-play/) and
[raw OpenNeuro snapshot](https://doi.org/10.18112/openneuro.ds004323.v1.0.0)
are separate resources.

## Planned contents and access

Prepared behavior, replays, aligned analysis inputs and summaries form the
proposed core. Optional tiers contain preprocessing outputs, raw activations,
supplementary/historical conditions and candidate baseline artifacts. The
current plan is approximately 511 GB core / 7.91 TB total selected from a
24.53 TB archive. These are planning figures, not uploaded-byte claims.

The planned catalogue configurations are `files` and `human_plays`, each with
split **`data`**. This name is organizational, not a scientific train/test
split. Configure the dataset YAML with the catalogue paths that actually exist
when publishing; no nonexistent `data_files` paths are advertised in this draft.

After publication and verification, the intended loading interface is:

```python
from datasets import load_dataset

dataset_commit = "REPLACE_WITH_VERIFIED_DATASET_COMMIT"
files = load_dataset(
    "csbotos/reason-to-play", "files", split="data", revision=dataset_commit
)
human_plays = load_dataset(
    "csbotos/reason-to-play", "human_plays", split="data", revision=dataset_commit
)
```

These catalogue examples do not download every scientific payload. The manifest
must distinguish metadata-verified, byte-verified and published artifacts.
Private source URIs are provenance, not public payload URLs. Record the actual
uploaded coverage and test this interface before claiming availability.

## Interpretation and limits

Human outcomes are three-valued; death events distinguish avatar deaths from
incomplete recordings. Engine frames, non-idle keypress frames and model
decisions are separate clocks. Practice/cohort/level exclusions must remain
explicit. Generated rationales are model outputs, not participant reports.

Keep historical and headline conditions separate. Missing feature/result cells,
unresolved checkpoint attribution and the EfficientZero 15-hook versus
11-layer-label mapping remain documented limitations. Code smoke tests do not
establish full MRI preprocessing or GPU-scale feature reproduction.

## License, provenance and citation

Original project code and newly created artifacts use MIT. Original OpenNeuro
records remain CC0. Third-party game assets, source and model weights retain
their own terms; the dataset's component-level metadata must preserve these
boundaries. Link the exact code revision and source snapshot, and include the
paper citation from the code repository.

Before publication, fill in the dataset commit, actual staged/uploaded totals,
catalogue schema and available configurations, verification date, component
licenses and exact coverage limitations. See the
[publication plan](huggingface-plan.md) and [manifest workflow](manifest-guide.md).
