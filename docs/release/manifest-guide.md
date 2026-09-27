# Inventory, manifest and staging workflow

This code-only repository ships release tooling and configuration examples.
The actual inventory, selection policy, source manifest, participant catalogues,
historical tables and staged payloads are maintained separately. No Hugging Face
dataset upload is claimed. The proposed target is `csbotos/reason-to-play`.

The [inventory summary](derivative-inventory.md) describes the proposed scale:
24.53 TB in source storage, a 7.91 TB selection and a 511 GB core. A proposed
canonical path is not a public URL or a claim that bytes have been downloaded.

## Build and verify a selection

Run from the repository root. Supply your authorized source bucket and reviewed
policy explicitly. The scripts use listing, HEAD and GET operations and do not
modify source objects. Keep generated files in local working storage, not Git.

```bash
pip install -r requirements-release.txt
python scripts/release/index_derivatives.py scan \
    --bucket YOUR_BUCKET --region YOUR_REGION --output out/inventory
python scripts/release/prepare_manifest.py build \
    --inventory out/inventory/s3-objects.jsonl.gz \
    --policy /absolute/path/to/reviewed-release-policy.json \
    --output out/selected
python scripts/release/prepare_manifest.py freeze \
    --manifest out/selected/manifest.jsonl.gz \
    --region YOUR_REGION \
    --output out/selected/source-verified.jsonl.gz
python scripts/release/prepare_manifest.py validate \
    --manifest out/selected/source-verified.jsonl.gz
```

[`policy.example.json`](policy.example.json) illustrates the ordered selection
schema with synthetic names. Replace it with reviewed source/component rules;
the example is not the paper's complete release policy. Unknown objects remain
unselected until an explicit rule applies. For a broader local inventory audit,
the separate index builder also requires explicit rules:

```bash
python scripts/release/index_derivatives.py build \
    --input out/inventory/s3-objects.jsonl.gz \
    --rules /absolute/path/to/reviewed-inventory-rules.json \
    --output out/index
```

See [`asset-rules.example.json`](asset-rules.example.json) for that schema.
Full bucket listings are paginated but not atomic snapshots. An artifact ID
identifies an observed source generation; it is not a payload checksum. ETags
are opaque. `payload.sha256` is filled only after the bytes have been read.

`freeze` pins a non-null S3 VersionId where available. Unversioned sources are
marked as such and checked by metadata/conditional requests; a source change
fails staging. `freeze --reuse-pins PREVIOUS.jsonl.gz` can reuse matching
immutable pins. Source errors remain explicit and cause a nonzero exit.

## Stage selected bytes and build catalogues

Choose a component defined by your policy. Staging defaults to a 1 GB byte
budget; larger selections require an explicit `--max-bytes` and enough local
space. For example, if your policy defines `encoding_summary`:

```bash
python scripts/release/prepare_manifest.py stage \
    --manifest out/selected/source-verified.jsonl.gz \
    --region YOUR_REGION \
    --component encoding_summary --directory out/hf-stage \
    --output out/selected/summary-staged.jsonl.gz
python scripts/release/prepare_manifest.py validate \
    --manifest out/selected/summary-staged.jsonl.gz --require-staged
python scripts/release/prepare_manifest.py merge-verified \
    --manifest out/selected/source-verified.jsonl.gz \
    --verified out/selected/summary-staged.jsonl.gz \
    --output out/selected/release-manifest.jsonl.gz
python scripts/release/prepare_manifest.py catalogue \
    --manifest out/selected/release-manifest.jsonl.gz \
    --output out/hf-stage/catalog/files/part-00000.parquet
```

`--tier` and `--component` can be repeated. Staging hashes streamed bytes,
checks size and known hashes, and atomically promotes completed files. Cached
bytes are reused only when size and SHA-256 match. Merging verified rows rejects
changed source identities or destinations. A catalogue may describe selected
but unstaged objects; its verification/publication fields must distinguish them.

After separately staging the human behavior component and merging its verified
rows, the optional per-play catalogue can be built locally:

```bash
python scripts/release/build_behavior_catalogue.py \
    --manifest out/selected/release-manifest.jsonl.gz \
    --stage-root out/hf-stage \
    --output out/hf-stage/catalog/human_plays --workers 4
```

The builder preserves all source plays and expresses exclusions as flags. It
retains nullable original outcomes and distinguishes engine frames from non-idle
keypress frames. This command generates participant-derived data, which belongs
to the separate dataset publication process rather than this Git release.

## Path and publication contract

Canonical destinations are relative to one release root. Each source path
segment is percent-encoded reversibly; separators remain separators. Source
keys are unchanged. Model IDs and scientific condition fields remain explicit
metadata rather than being reconstructed from sanitized filenames.

| Component | Proposed canonical root |
| --- | --- |
| Prepared human records | `behavior/human/` |
| Recorded replays | `replays/human/`, `replays/generative/` |
| Preprocessing outputs | `derivatives/fmri/fmriprep/` |
| Aligned analysis inputs | `derivatives/fmri/aligned/` |
| Encoding outputs | `analysis/encoding/` |
| Raw model features | `features/raw/pytorch/` |

A release root is not automatically a pipeline workdir. Use the explicit paths
in the [reproduction guide](../reproducibility.md), retaining original subject,
cohort, game, layer and timing identifiers. Never infer a missing correspondence
by renaming files.

The planned Hugging Face configurations are `files` and `human_plays`, each with
split name **`data`**. This is an organizational split, not a scientific
train/test partition; `all` is reserved by the datasets loader. Before publishing,
test the actual local package by configuration name, then test the uploaded
repository at its immutable dataset commit. See the [publication plan](huggingface-plan.md).
