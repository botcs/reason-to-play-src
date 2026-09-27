# Derivative inventory summary

The research storage audit on 27 September 2026 found approximately **24.53 TB**
of current objects. An inventory listing records what was observed; it is not
an atomic snapshot, payload validation or evidence of public availability.
Detailed inventories and object identities remain in the separate dataset
release working package, outside this code-only repository.

The proposed selection is approximately **7.91 TB**, including a **511 GB core**
(511.6 GB before rounding). The core groups prepared behavior, selected replays,
aligned analysis inputs and result summaries. Optional tiers cover preprocessing
outputs, raw activations, supplementary/historical conditions, and candidate
baseline features and checkpoints. These are proposed download sizes, not
uploaded or fully byte-verified totals.

The source archive also contains about **10.97 TB of earlier RSA intermediates**
excluded from the proposed release and **5.16 TB of older EfficientZero traces**
deferred pending selection/provenance review. These categories explain much of
the difference between the archive and proposed release; they are not an exact
partition because the displayed totals are rounded and other exclusions exist.

| Component | Release requirement |
| --- | --- |
| Raw human data | Link the pinned OpenNeuro ds004323 v1.0.0 snapshot and DOI |
| Human behavior | Preserve original outcomes, play identity, clocks and practice/cohort flags |
| Model behavior | Retain selected run, prompt, seed, advancement and completion metadata |
| fMRI derivatives | Preserve preprocessing version/options, masks, confounds, timing and QC |
| Aligned data | State feature axes and the mapping from timestamps to scanner volumes |
| Raw features | Keep model identity/revision, extraction settings and prompt sidecars |
| Results | Preserve condition, layer, subject, fit/nuisance metadata and analysis provenance |
| Baseline candidates | Keep uncertain checkpoint/layer attribution explicit |

Raw MRI remains available from
[OpenNeuro ds004323 v1.0.0](https://doi.org/10.18112/openneuro.ds004323.v1.0.0).
Source code and game definitions are versioned here. Historical and current
feature generations remain distinct; directory counts are not model counts.

Use the [manifest workflow](manifest-guide.md) to rebuild a local inventory,
apply an explicit policy, verify source generations and stage selected bytes.
The checked-in example rules illustrate syntax only. They do not reproduce
the privately prepared selection or authorize publishing its data. Known
scientific limitations are summarized in [scientific provenance](scientific-provenance.md).
