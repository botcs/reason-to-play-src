# Hugging Face dataset publication plan

The intended owner is the author's personal account, with proposed dataset
name **`csbotos/reason-to-play`**. The
[paper page](https://huggingface.co/papers/2605.08019) already exists. This plan
responds to the [dataset invitation](https://github.com/botcs/reason-to-play-src/issues/1).
The dataset and payloads have not been published by this code release.

The prepared dataset working package is separate from Git. It contains the
detailed inventory, selection policy, source manifest, catalogue candidates and
staging evidence. This repository publishes the code, schema examples and
[workflow](manifest-guide.md), not participant tables or object-level storage
metadata. The [dataset-card draft](huggingface-dataset-card.md) must be completed
against the subset actually uploaded.

## Proposed package

The 24.53 TB source archive has a proposed **7.91 TB** selection with a
**511 GB core**. Optional tiers keep large raw features, preprocessing,
supplementary/historical conditions and baseline candidates separately
downloadable. About 10.97 TB of earlier RSA intermediates are excluded and
5.16 TB of older EfficientZero traces remain deferred. These are rounded
planning totals, not public hosting or verified-upload claims.

Raw MRI is linked to the pinned
[OpenNeuro ds004323 v1.0.0](https://doi.org/10.18112/openneuro.ds004323.v1.0.0)
snapshot. The planned `files` catalogue exposes artifact discovery/verification;
`human_plays` exposes per-attempt outcome/clock/comparison metadata. Each uses
split **`data`**, with no invented train/test meaning. Large scientific arrays
remain downloadable payloads rather than browser cells. Historical tables and
corrected catalogues retain distinct provenance and semantics.

The author selected MIT jointly for original code and newly created research
artifacts. Raw OpenNeuro data remain CC0; inherited code, assets and model weights
retain their terms. [THIRD_PARTY.md](../../THIRD_PARTY.md) records the source
boundaries. Unresolved baseline attribution and coverage remain visible in
[scientific provenance](scientific-provenance.md).

## Publication sequence

1. Review the actual dataset selection and intended first upload separately
   from the code release. Keep workstation backups, credentials, environments
   and unrelated intermediate generations outside the dataset.
2. Freeze source generations and stage the chosen tier. Hash actual payloads,
   validate staged files and preserve scientific metadata and known gaps.
   Use the explicit policy and size limits in the manifest workflow.
3. Confirm authenticated write access to the selected personal namespace and
   hosting capacity for the actual tier. Consult the
   [Hub storage guidance](https://huggingface.co/docs/hub/storage-limits);
   an invitation does not establish an allocation for multi-terabyte uploads.
4. Prepare `files` and `human_plays` using split `data`, then load the local
   package by configuration name with `datasets.load_dataset`. Distinguish
   selected-but-unavailable entries from uploaded payloads. Retain actual
   verification status rather than turning a proposed path into a download URL.
5. Upload the reviewed staged subset using the
   [resumable upload workflow](https://huggingface.co/docs/huggingface_hub/guides/upload).
   Record dataset commit, code commit, manifest hash and uploaded totals.
6. Verify the public configurations at that dataset commit, download a small
   scientific sample, and compare its checksum/shape/timing metadata. Complete
   the card with actual coverage, inherited terms and remaining limitations.
7. Add the verified dataset URL to the website and paper links. Reply to the
   invitation only after the response is authorized; no reply is sent by these
   scripts or this document.

The [GitHub Project](https://github.com/users/botcs/projects/1) tracks status
and priority. This document specifies the publication workflow rather than
duplicating the task tracker.
