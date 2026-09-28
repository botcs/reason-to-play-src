# Inventory, manifest and staging workflow

This guide covers dataset catalogues and tools for maintaining a source
inventory. Dataset files and generated catalogues belong in the dataset,
not the code repository. The [dataset card](huggingface-dataset-card.md) records
publication status for `csbotos/reason-to-play`.

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

## Assemble the dataset layout

Use an explicit path map to combine selected source objects with generated
JSON, atlas/mask files and other local inputs. Source identifiers and current
payload identities are separate fields. One schema-version-2
`manifest.jsonl.gz` describes the complete dataset; the human-play catalogue
indexes attempts within its human files.

```bash
python scripts/release/prepare_manifest.py assemble \
    --source-manifest out/selected/source-verified.jsonl.gz \
    --path-map /absolute/path/to/source-path-map.jsonl \
    --local-files /absolute/path/to/local-files.jsonl \
    --source-files /absolute/path/to/verified-cache.jsonl \
    --release-id YOUR_RELEASE_ID \
    --output out/dataset --audit-output out/private-receipts --workers 32
```

Each path-map row identifies a source by its frozen `artifact_id`, gives an
explicit include/exclude decision, and supplies the selected `release_path`,
component and experiment metadata. Local-input rows supply `local_path`,
`release_path`, `component`, `metadata`, `license`, `provenance`, `tier` and
`selection_reason`. Cache receipts additionally pin local bytes to the original
source generation and checksum. Local filesystem paths remain in private
receipts outside the dataset.

The assembler checks path and identity collisions, required input references,
source generations and local payload hashes. It writes
`catalog/files/planned_files.parquet` and `catalog/metadata.json` from the same
selection. Source objects whose payloads have not been staged retain that
status; an inventory size or ETag does not become a SHA-256 checksum.

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

Build a per-play catalogue from the human JSON files:

```bash
python scripts/release/build_behavior_catalogue.py \
    --input /data/reason-to-play/behavior/human \
    --output /data/catalog/human_plays --workers 32
```

Adjust `--workers` to the machine’s available CPU and memory; the default is 1.
The builder selects one prompt condition (`elaborate` by default), includes its
plays once, and expresses analysis exclusions as flags. It
retains nullable outcomes and distinguishes engine frames from non-idle
keypress frames. Use `--condition minimal` or `--condition oracle` to select
another condition. `--input` also accepts the dataset root or a single human
JSON file. Each catalogue row preserves the play’s original ordinal within its
scanner run and identifies its current compressed JSON through
`source_release_path`, `source_payload_sha256` and `source_artifact_id`.
No source archive or separate prompt file is needed.

## Human rows in the manifest

The unified manifest contains one row per participant/game/prompt-condition
file with `component: "human_behavior"`. Its payload schema is
`reason-to-play/human-replay`, version 1; the containing manifest row uses
schema version 2:

| Field | Meaning |
| --- | --- |
| `release_path` | Dataset-relative path to the human JSON |
| `artifact_id` | `sha256:` followed by the compressed payload checksum |
| `payload` | Human JSON schema/version, compressed byte count and SHA-256 |
| `metadata` | Participant, game, prompt condition and play/frame/step counts |
| `provenance.measurement_sha256` | Hash of the recorded behavior; agrees across prompt conditions for one participant/game |
| `provenance.recorded_behavior` | Original human study and associated OpenNeuro dataset/version |
| `provenance.upstream_replay` | Source attribution: frozen source artifact ID, source checksum and exact S3 bucket/key/version |

The current payload identity and the upstream source artifact identity have
separate meanings. Source inventory IDs identify an observed source generation;
they are not content checksums. Download the current `release_path` from the
dataset and verify `payload.sha256`. The upstream S3 location is provenance,
not a required consumer input.

## Path and publication contract

Dataset destinations are relative to one release root. Original source keys
remain in provenance metadata. Model IDs and experimental condition fields remain
explicit rather than being reconstructed from filenames. The manifest supplies
the exact path for every feature, BOLD input, atlas, result and website asset;
follow the [dataset analysis guide](../guides/dataset-analysis.md) to select the
inputs for a workflow. Human files use
`behavior/human/sub-XX/GAME/CONDITION.human.replay.json.gz`.

The planned Hugging Face configurations are `planned_files`, `files` and
`human_plays`, each with
split name **`data`**. This is an organizational split, not a train/test partition; `all` is reserved by the datasets loader. Before publishing,
test the actual local package by configuration name, then test the uploaded
repository at its immutable dataset commit. See the [dataset card](huggingface-dataset-card.md) for catalogue contents.
