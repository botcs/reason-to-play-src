#!/usr/bin/env python3
"""Build, pin, stage and validate a release without modifying source S3 objects.

Selection, source-version verification and payload hashing are separate facts.
The manifest never treats an ETag or an inventory listing as a content hash.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
from itertools import islice
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tempfile
from urllib.parse import quote


SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 2
FORBIDDEN = {".git", ".aws", ".ssh", ".venv", "__pycache__", ".ipynb_checkpoints"}
FORBIDDEN_FILES = {
    ".env",
    "credentials",
    "token",
    "stored_tokens",
    ".DS_Store",
    "license.txt",
}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@contextmanager
def write_rows(path):
    """Deterministic gzip bytes; publish each completed file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".manifest-", delete=False
        ) as raw:
            partial = Path(raw.name)
            if path.suffix == ".gz":
                with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as gz:
                    with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as text:
                        yield text
            else:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="\n") as text:
                    yield text
        partial.replace(path)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def emit(handle, row):
    handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def safe_relative(value):
    parts = value.split("/")
    return (
        bool(value)
        and not value.startswith("/")
        and "\\" not in value
        and not any(ord(c) < 32 or ord(c) == 127 for c in value)
        and all(p not in {"", ".", ".."} for p in parts)
    )


def safe_source(key):
    return safe_relative(key) and not (
        set(key.split("/")) & FORBIDDEN or key.split("/")[-1] in FORBIDDEN_FILES
    )


def load_policy(path):
    policy = json.loads(Path(path).read_text())
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported selection-policy schema")
    identifiers = set()
    for rule in policy["rules"]:
        if rule["id"] in identifiers:
            raise ValueError(f"Duplicate rule ID: {rule['id']}")
        identifiers.add(rule["id"])
        prefix = rule["source_prefix"]
        if prefix and not safe_relative(prefix.rstrip("/")):
            raise ValueError(f"Unsafe source prefix in {rule['id']}")
        if rule["decision"] not in {"include", "optional", "defer", "exclude"}:
            raise ValueError(f"Invalid decision in {rule['id']}")
        if rule["decision"] in {"include", "optional"} and not safe_relative(
            rule["release_root"]
        ):
            raise ValueError(f"Unsafe release root in {rule['id']}")
        re.compile(rule.get("match", ".*"))
    return policy


def pick_rule(key, policy):
    if not safe_source(key):
        return {
            "id": "unsafe-or-private-material",
            "decision": "exclude",
            "component": "excluded",
            "reason": "Unsafe path, cache, credentials or non-redistributable local license file",
        }
    for rule in policy["rules"]:
        prefix = rule["source_prefix"]
        if key.startswith(prefix) and re.fullmatch(
            rule.get("match", ".*"), key[len(prefix) :]
        ):
            return rule
    return {
        "id": "unmapped",
        "decision": "defer",
        "component": "unmapped",
        "reason": "No explicit release rule",
    }


def make_row(source, rule, policy):
    source_uri = f"s3://{source['bucket']}/{source['key']}"
    suffix = source["key"][len(rule["source_prefix"]) :]
    release_path = rule["release_root"] + (
        "/" + "/".join(quote(p, safe="-._~") for p in suffix.split("/"))
        if suffix
        else ""
    )
    # Stable identity for the listed generation; later pinning and byte hashing
    # enrich the same record without changing references to its artifact ID.
    identity = json.dumps(
        [source_uri, source["etag"], source["size_bytes"], source.get("last_modified")],
        separators=(",", ":"),
    )
    key = source["key"]
    subject = re.search(r"(?:^|/|\|)(?:sub-|subj)(\d+)(?=/|_|\||\.)", key)
    metadata = dict(rule.get("metadata", {}))
    metadata["subject"] = f"sub-{int(subject[1]):02d}" if subject else None
    metadata["model_path_id"] = next(
        (p for p in key.split("/") if p.startswith("model-")), None
    )
    tokens = key.replace("|", "/").split("/")
    for name, values in {
        "rationale_mode": {"action-only", "copied-reasoning", "prompted-rationale"},
        "suggestion_level": {"minimal", "elaborate", "oracle"},
    }.items():
        metadata[name] = next((p for p in tokens if p in values), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": policy["release_id"],
        "artifact_id": "sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
        "decision": rule["decision"],
        "tier": rule["tier"],
        "component": rule["component"],
        "selection_rule": rule["id"],
        "selection_reason": rule["reason"],
        "release_path": release_path,
        "source_uri": source_uri,
        "source": {**source, "version_id": None, "verification": "listed-only"},
        "payload": {"sha256": None, "validation": "not-staged"},
        "license": rule.get("license", policy.get("derivative_license", "pending")),
        "provenance": {
            "status": rule.get(
                "provenance_status", "source-listed; producing-run mapping incomplete"
            ),
            "references": rule.get("evidence", []),
            "producer_commit": rule.get("producer_commit"),
        },
        "metadata": metadata,
    }


def build(args):
    source, output = Path(args.inventory).resolve(), Path(args.output).resolve()
    if output == source or output in source.parents:
        raise ValueError("Output directory must not contain the source inventory")
    output.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.policy)
    totals = Counter(
        inventory_objects=0, inventory_bytes=0, selected_objects=0, selected_bytes=0
    )
    groups = defaultdict(lambda: {"objects": 0, "bytes": 0})
    coverage = defaultdict(lambda: {"objects": 0, "bytes": 0})
    with tempfile.TemporaryDirectory(prefix="manifest-build-", dir=output) as temp:
        temp = Path(temp)
        db = sqlite3.connect(temp / "unique.sqlite")
        db.execute("CREATE TABLE sources (uri TEXT PRIMARY KEY)")
        db.execute("CREATE TABLE outputs (path TEXT PRIMARY KEY, artifact TEXT UNIQUE)")
        try:
            with write_rows(temp / "manifest.jsonl.gz") as manifest:
                for raw in read_rows(source):
                    if (
                        not isinstance(raw.get("size_bytes"), int)
                        or raw["size_bytes"] < 0
                    ):
                        raise ValueError("Invalid object size")
                    uri = f"s3://{raw['bucket']}/{raw['key']}"
                    try:
                        db.execute("INSERT INTO sources VALUES (?)", (uri,))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"Duplicate source object: {uri}") from exc
                    rule = pick_rule(raw["key"], policy)
                    group = groups[(rule["id"], rule["decision"], rule["component"])]
                    group["objects"] += 1
                    group["bytes"] += raw["size_bytes"]
                    totals["inventory_objects"] += 1
                    totals["inventory_bytes"] += raw["size_bytes"]
                    if rule["decision"] not in {"include", "optional"}:
                        continue
                    row = make_row(raw, rule, policy)
                    if not safe_relative(row["release_path"]):
                        raise ValueError(f"Unsafe release path: {row['release_path']}")
                    try:
                        db.execute(
                            "INSERT INTO outputs VALUES (?,?)",
                            (row["release_path"], row["artifact_id"]),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"Release path collision: {row['release_path']}"
                        ) from exc
                    emit(manifest, row)
                    totals["selected_objects"] += 1
                    totals["selected_bytes"] += raw["size_bytes"]
                    c = coverage[
                        (
                            row["tier"],
                            row["component"],
                            row["metadata"]["subject"] or "",
                            row["metadata"]["model_path_id"] or "",
                        )
                    ]
                    c["objects"] += 1
                    c["bytes"] += raw["size_bytes"]
        finally:
            db.close()
        summary = {
            "schema_version": SCHEMA_VERSION,
            "release_id": policy["release_id"],
            "target_repository": policy["target_repository"],
            "publication_status": "prepared; not uploaded",
            "inventory_sha256": digest_file(source),
            "policy_sha256": digest_file(args.policy),
            "manifest_sha256": digest_file(temp / "manifest.jsonl.gz"),
            "integrity": "Selection only. Pin source versions with freeze; compute payload SHA-256 with stage. ETags are opaque.",
            **dict(totals),
            "groups": [
                {"rule": key[0], "decision": key[1], "component": key[2], **value}
                for key, value in sorted(groups.items())
            ],
        }
        write_json(temp / "summary.json", summary)
        write_json(temp / "policy.json", policy)
        with (temp / "coverage.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["tier", "component", "subject", "model_path_id", "objects", "bytes"]
            )
            writer.writerows(
                [*key, value["objects"], value["bytes"]]
                for key, value in sorted(coverage.items())
            )
        for name in (
            "manifest.jsonl.gz",
            "summary.json",
            "policy.json",
            "coverage.csv",
        ):
            (temp / name).replace(output / name)
    write_json(
        output / "checksums.json",
        {
            p.name: digest_file(p)
            for p in sorted(output.iterdir())
            if p.name
            in {"manifest.jsonl.gz", "summary.json", "policy.json", "coverage.csv"}
        },
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("selected_objects", "selected_bytes", "manifest_sha256")
            },
            indent=2,
        )
    )


def selected_rows(args):
    for row in read_rows(args.manifest):
        if args.tier and row["tier"] not in args.tier:
            continue
        if args.component and row["component"] not in args.component:
            continue
        yield row


def check_row(row):
    if row.get("schema_version") == DATASET_SCHEMA_VERSION:
        return check_dataset_row(row)
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    if not safe_relative(row["release_path"]) or not safe_source(row["source"]["key"]):
        raise ValueError("Unsafe path in manifest")
    if row["decision"] not in {"include", "optional"}:
        raise ValueError("Manifest contains an unselected artifact")
    if row["source_uri"] != f"s3://{row['source']['bucket']}/{row['source']['key']}":
        raise ValueError("Source URI does not match the source object")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", row["artifact_id"]):
        raise ValueError("Invalid artifact identity")
    if row["source"].get("verification") == "version-pinned" and not row["source"].get(
        "version_id"
    ):
        raise ValueError("Pinned source is missing its version ID")
    if type(row["source"]["size_bytes"]) is not int or row["source"]["size_bytes"] < 0:
        raise ValueError("Invalid payload size")
    sha = row["payload"].get("sha256")
    if sha is not None and not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("Invalid payload SHA-256")


def payload_size(row):
    """Released bytes can differ from the inputs used to generate an artifact."""
    if row.get("schema_version") == DATASET_SCHEMA_VERSION:
        return row["payload"]["size_bytes"]
    return row["source"]["size_bytes"]


def check_dataset_row(row):
    if not safe_relative(row["release_path"]):
        raise ValueError("Unsafe path in manifest")
    if row["decision"] not in {"include", "optional"}:
        raise ValueError("Manifest contains an unselected artifact")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", row["artifact_id"]):
        raise ValueError("Invalid artifact identity")
    size = row["payload"].get("size_bytes")
    if type(size) is not int or size < 0:
        raise ValueError("Invalid payload size")
    sha = row["payload"].get("sha256")
    if sha is not None and not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("Invalid payload SHA-256")
    if row["payload"].get("validation") == "bytes-verified" and not sha:
        raise ValueError("Verified payload is missing its SHA-256")
    source = row["source"]
    if source["kind"] == "s3":
        if not safe_source(source["key"]):
            raise ValueError("Unsafe source path")
        if row["source_uri"] != f"s3://{source['bucket']}/{source['key']}":
            raise ValueError("Source URI does not match the source object")
        if size != source["size_bytes"]:
            raise ValueError("Untransformed source and payload sizes differ")
        if source.get("verification") == "version-pinned" and not source.get(
            "version_id"
        ):
            raise ValueError("Pinned source is missing its version ID")
        if row["artifact_id_scheme"] != "frozen-source-identity":
            raise ValueError("Incorrect source artifact identity scheme")
    elif source["kind"] == "generated":
        if row["source_uri"] is not None or "local_path" in source:
            raise ValueError("Generated records must not expose private file locations")
        if (
            row["artifact_id_scheme"] != "payload-sha256"
            or row["artifact_id"] != f"sha256:{sha}"
        ):
            raise ValueError("Generated artifact identity must match its payload")
        if row["payload"]["validation"] != "bytes-verified" or not sha:
            raise ValueError("Generated payload must be byte-verified")
    else:
        raise ValueError("Unknown source kind")
    if row.get("published") is not False:
        raise ValueError("Preparation cannot assert remote publication")


def validate(args):
    ids, paths, sources = set(), set(), set()
    counts = Counter()
    for row in selected_rows(args):
        check_row(row)
        for value, seen, label in (
            (row["artifact_id"], ids, "artifact ID"),
            (row["release_path"], paths, "release path"),
            (row["source_uri"], sources, "source URI"),
        ):
            if value is None:
                continue
            if value in seen and not (
                label == "artifact ID"
                and row.get("artifact_id_scheme") == "payload-sha256"
            ):
                raise ValueError(f"Duplicate {label}: {value}")
            seen.add(value)
        counts["objects"] += 1
        counts["bytes"] += payload_size(row)
        counts["version_pinned"] += bool(row["source"].get("version_id"))
        counts["payload_hashed"] += bool(row["payload"].get("sha256"))
        if args.require_staged and (
            row["payload"]["validation"] != "bytes-verified"
            or not row["payload"]["sha256"]
        ):
            raise ValueError(f"Payload not verified: {row['release_path']}")
    if not counts["objects"]:
        raise ValueError("Selection is empty")
    print(json.dumps(dict(counts), indent=2))


def pin_source(row, client):
    check_row(row)
    src = row["source"]
    if src.get("kind") == "generated":
        return row
    # IfMatch protects against a mutable key changing after the listing.
    request = {"Bucket": src["bucket"], "Key": src["key"], "IfMatch": src["etag"]}
    if src.get("version_id"):
        request["VersionId"] = src["version_id"]
    head = client.head_object(**request)
    modified = head["LastModified"].isoformat()
    if (
        head["ContentLength"] != src["size_bytes"]
        or head["ETag"].strip('"') != src["etag"]
        or modified != src.get("last_modified")
        or (src.get("version_id") and head.get("VersionId") != src["version_id"])
    ):
        raise ValueError(f"Source changed since inventory: {row['source_uri']}")
    version = head.get("VersionId")
    row = {
        **row,
        "source": {
            **src,
            "version_id": version if version and version != "null" else None,
            "verification": "version-pinned"
            if version and version != "null"
            else "head-verified-unversioned",
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return row


def freeze(args):
    import boto3
    from botocore.config import Config

    if Path(args.manifest).resolve() == Path(args.output).resolve():
        raise ValueError("Write pinned records to a new manifest")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    client = boto3.client(
        "s3", region_name=args.region, config=Config(max_pool_connections=args.workers)
    )
    errors, count = 0, 0
    previous = {}
    if args.reuse_pins:
        for row in read_rows(args.reuse_pins):
            check_row(row)
            if row["source"].get("verification") == "version-pinned":
                if row["artifact_id"] in previous:
                    raise ValueError("Duplicate artifact in --reuse-pins")
                previous[row["artifact_id"]] = row

    def one(row):
        try:
            cached = previous.get(row["artifact_id"])
            if cached and all(
                cached["source"].get(k) == row["source"].get(k)
                for k in ("bucket", "key", "etag", "size_bytes", "last_modified")
            ):
                # Reuse the earlier observation of this immutable generation,
                # preserving its verification timestamp, not claiming a new HEAD.
                return {**row, "source": cached["source"], "payload": cached["payload"]}
            return pin_source(row, client)
        except Exception as exc:
            return {
                **row,
                "source": {
                    **row["source"],
                    "verification": "failed",
                    "verification_error": str(exc),
                },
            }

    with (
        write_rows(args.output) as output,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        iterator = iter(selected_rows(args))
        while batch := list(islice(iterator, 2 * args.workers)):
            # Python 3.12 executor.map eagerly consumes its input. Bound the
            # queue so progress and memory use do not depend on manifest size.
            for row in pool.map(one, batch):
                emit(output, row)
                count += 1
                errors += row["source"]["verification"] == "failed"
                if count % 250 == 0:
                    print(
                        f"Checked {count:,} source objects; {errors} errors", flush=True
                    )
    print(json.dumps({"checked": count, "errors": errors}))
    if errors or not count:
        raise SystemExit(1)


def destination(root, relative):
    if not safe_relative(relative):
        raise ValueError("Unsafe destination path")
    root = Path(root).resolve()
    path = (root / PurePosixPath(relative)).resolve()
    if root not in path.parents:
        raise ValueError("Destination escapes staging root")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def stage_object(row, client, root):
    check_row(row)
    src = row["source"]
    if src.get("kind") == "generated":
        target = destination(root, row["release_path"])
        if (
            target.exists()
            and target.stat().st_size == payload_size(row)
            and digest_file(target) == row["payload"]["sha256"]
        ):
            return row
        raise ValueError(
            "Generated payload must be staged from its local assembly receipt"
        )
    if src.get("verification") not in {"version-pinned", "head-verified-unversioned"}:
        raise ValueError("Freeze/verify source metadata before staging")
    target = destination(root, row["release_path"])
    expected = row["payload"].get("sha256")
    if target.exists() and expected:
        if (
            target.stat().st_size == src["size_bytes"]
            and digest_file(target) == expected
        ):
            return {
                **row,
                "payload": {
                    **row["payload"],
                    "sha256": expected,
                    "validation": "bytes-verified",
                },
            }
    kwargs = {"Bucket": src["bucket"], "Key": src["key"]}
    if src.get("version_id"):
        kwargs["VersionId"] = src["version_id"]
    else:
        kwargs["IfMatch"] = src["etag"]
    response = client.get_object(**kwargs)
    body = response["Body"]
    partial = None
    try:
        if (
            response["ContentLength"] != src["size_bytes"]
            or response["ETag"].strip('"') != src["etag"]
            or response["LastModified"].isoformat() != src.get("last_modified")
            or (
                src.get("version_id") and response.get("VersionId") != src["version_id"]
            )
        ):
            raise ValueError("Downloaded source metadata differs from the manifest")
        h, size = hashlib.sha256(), 0
        # A unique temporary name cannot overwrite a selected *.partial artifact.
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".stage-", delete=False
        ) as f:
            partial = Path(f.name)
            for chunk in iter(lambda: body.read(8 * 1024 * 1024), b""):
                h.update(chunk)
                size += len(chunk)
                f.write(chunk)
        if size != src["size_bytes"]:
            raise ValueError("Truncated payload")
        sha = h.hexdigest()
        if expected and sha != expected:
            raise ValueError("Payload SHA-256 mismatch")
        partial.replace(target)
        return {
            **row,
            "payload": {
                **row["payload"],
                "sha256": sha,
                "validation": "bytes-verified",
            },
        }
    finally:
        body.close()
        if partial is not None:
            partial.unlink(missing_ok=True)


def stage(args):
    import boto3

    chosen = list(selected_rows(args))
    needed = sum(payload_size(r) for r in chosen)
    if not chosen:
        raise ValueError("Selection is empty")
    if needed > args.max_bytes:
        raise ValueError(
            f"Selection needs {needed:,} bytes, exceeding --max-bytes={args.max_bytes:,}; select a component/tier or explicitly raise the budget"
        )
    if Path(args.manifest).resolve() == Path(args.output).resolve():
        raise ValueError("Write hashed records to a new manifest")
    targets = set()
    for row in chosen:
        check_row(row)
        target = destination(args.directory, row["release_path"])
        if target in targets or target in {
            Path(args.output).resolve(),
            Path(args.manifest).resolve(),
        }:
            raise ValueError(
                "Staging paths collide with another artifact or a manifest"
            )
        targets.add(target)
    client = boto3.client("s3", region_name=args.region)
    with write_rows(args.output) as output:
        for i, row in enumerate(chosen, 1):
            emit(output, stage_object(row, client, args.directory))
            print(f"Staged {i}/{len(chosen)}: {row['release_path']}", flush=True)


def merge_verified(args):
    """Enrich selected records with bytes verified by a separate staging pass."""
    if Path(args.manifest).resolve() == Path(args.output).resolve():
        raise ValueError("Write enriched records to a new manifest")
    verified = {}
    for filename in args.verified:
        for row in read_rows(filename):
            check_row(row)
            if row["payload"].get("validation") != "bytes-verified" or not row[
                "payload"
            ].get("sha256"):
                raise ValueError("Only byte-verified records can enrich a manifest")
            identity = row["artifact_id"]
            if identity in verified and verified[identity] != row:
                raise ValueError("Conflicting verified records")
            verified[identity] = row
    count = 0
    with write_rows(args.output) as output:
        for row in read_rows(args.manifest):
            check_row(row)
            evidence = verified.pop(row["artifact_id"], None)
            if evidence:
                if row["release_path"] != evidence["release_path"] or any(
                    row["source"].get(k) != evidence["source"].get(k)
                    for k in (
                        "bucket",
                        "key",
                        "etag",
                        "size_bytes",
                        "last_modified",
                        "version_id",
                    )
                ):
                    raise ValueError(
                        "Verified payload does not match the selected source generation/path"
                    )
                row = {**row, "payload": evidence["payload"]}
                count += 1
            emit(output, row)
        if verified:
            raise ValueError(
                "Verified records include artifacts absent from the selected manifest"
            )
    print(f"Added payload SHA-256 evidence to {count:,} selected records")


def catalogue(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for row in selected_rows(args):
        check_row(row)
        records.append(
            {
                "artifact_id": row["artifact_id"],
                "release_path": row["release_path"],
                "tier": row["tier"],
                "component": row["component"],
                "decision": row["decision"],
                "size_bytes": payload_size(row),
                "artifact_id_scheme": row.get(
                    "artifact_id_scheme", "frozen-source-identity"
                ),
                "source_kind": row["source"].get("kind", "s3"),
                "sha256": row["payload"]["sha256"],
                "source_uri": row["source_uri"],
                "source_version_id": row["source"].get("version_id"),
                "source_verification": row["source"]["verification"],
                "payload_validation": row["payload"]["validation"],
                "subject": row["metadata"].get("subject"),
                "source_model_path_id": row["metadata"].get(
                    "source_model_path_id", row["metadata"].get("model_path_id")
                ),
                **{
                    name: row["metadata"].get(name)
                    for name in (
                        "model_family",
                        "model_id",
                        "game",
                        "condition",
                        "suggestion_level",
                        "rationale_mode",
                        "action_selection",
                        "stream",
                        "timebase",
                        "paper_role",
                        "seed",
                        "checkpoint_id",
                        "bold_release_path",
                        "bold_sha256",
                        "samples_release_path",
                        "samples_sha256",
                        "sample_order_sha256",
                        "sample_order_status",
                        "source_model_id",
                        "source_game",
                        "run_id",
                        "run",
                        "source_play_id",
                        "source_play_index",
                        "layer",
                        "level",
                        "fit_condition",
                        "shuffle_unit",
                        "artifact_role",
                        "weight_initialization",
                        "context_fraction",
                        "frames",
                        "plays",
                        "steps",
                    )
                },
                "license": row["license"],
                "provenance_status": row["provenance"]["status"],
                # A proposed HF path is not an existing public download URL.
                "published": False,
            }
        )
    schema = (
        pa.schema(
            [
                (
                    name,
                    pa.int64()
                    if name
                    in {
                        "size_bytes",
                        "seed",
                        "source_play_index",
                        "layer",
                        "level",
                        "frames",
                        "plays",
                        "steps",
                    }
                    else pa.bool_()
                    if name == "published"
                    else pa.float64()
                    if name == "context_fraction"
                    else pa.string(),
                )
                for name in records[0]
            ]
        )
        if records
        else None
    )
    if not records:
        raise ValueError("Selection is empty")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(records, schema=schema), output, compression="zstd"
    )
    print(f"Wrote {len(records):,} file records to {output}")


def dataset_source_row(row, mapping, release_id):
    """Apply an explicit reviewed path/metadata map without altering source bytes."""
    check_row(row)
    metadata = {**row.get("metadata", {}), **mapping.get("metadata", {})}
    original_model_path = metadata.pop("model_path_id", None)
    if original_model_path is not None:
        metadata.setdefault("source_model_path_id", original_model_path)
    if metadata.get("source_model_id") and not metadata.get("model_id"):
        raise ValueError(
            "An explicit public model_id is required for a source model alias"
        )
    provenance = {**row.get("provenance", {}), **mapping.get("provenance", {})}
    # Reviewed package-relative references replace earlier preparation documents.
    provenance["references"] = mapping.get("provenance", {}).get("references", [])
    result = {
        **row,
        "schema_version": DATASET_SCHEMA_VERSION,
        "release_id": release_id,
        "artifact_id_scheme": "frozen-source-identity",
        "source": {**row["source"], "kind": "s3"},
        "payload": {**row["payload"], "size_bytes": row["source"]["size_bytes"]},
        "release_path": mapping["release_path"],
        "metadata": metadata,
        "provenance": provenance,
        "selection_reason": mapping["selection_reason"],
        "published": False,
    }
    for name in ("component", "decision", "tier", "selection_rule"):
        if name in mapping:
            result[name] = mapping[name]
    check_row(result)
    return result


def local_dataset_row(addition, release_id):
    """Hash a generated/local artifact; its private path never enters the manifest."""
    path = Path(addition["local_path"])
    if not path.is_file():
        raise ValueError(f"Missing local artifact: {path}")
    size = path.stat().st_size
    sha = digest_file(path)
    payload = {
        **addition.get("payload", {}),
        "size_bytes": size,
        "sha256": sha,
        "validation": "bytes-verified",
    }
    expected = addition.get("payload", {})
    if expected.get("sha256", sha) != sha or expected.get("size_bytes", size) != size:
        raise ValueError(
            f"Local artifact differs from supplied verification: {addition['release_path']}"
        )
    row = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "release_id": release_id,
        "artifact_id": f"sha256:{sha}",
        "artifact_id_scheme": "payload-sha256",
        "decision": addition.get("decision", "include"),
        "tier": addition.get("tier", "core"),
        "component": addition["component"],
        "selection_rule": addition.get("selection_rule", "local-release-input"),
        "selection_reason": addition["selection_reason"],
        "release_path": addition["release_path"],
        "source_uri": None,
        "source": {"kind": "generated", "verification": "local-bytes-verified"},
        "payload": payload,
        "metadata": addition.get("metadata", {}),
        "provenance": {
            "status": "local payload verified",
            **addition.get("provenance", {}),
        },
        "license": addition["license"],
        "published": False,
    }
    check_row(row)
    return row


def check_public_dataset_paths(rows):
    paths, sources, identities = set(), set(), {}
    for row in rows:
        check_row(row)
        path = row["release_path"]
        if re.search(r"%[0-9a-fA-F]{2}|\||\s", path):
            raise ValueError(
                f"Public path contains an encoded condition or whitespace: {path}"
            )
        if path.endswith((".bson", ".pkl")) or any(
            tag in path for tag in (".imputed.", ".narration.")
        ):
            raise ValueError(f"Unsupported public dataset payload: {path}")
        if path in paths:
            raise ValueError(f"Release path collision: {path}")
        paths.add(path)
        source = row.get("source_uri")
        if source is not None:
            if source in sources:
                raise ValueError(f"Source selected more than once: {source}")
            sources.add(source)
        identity = row["artifact_id"]
        previous = identities.get(identity)
        if previous and not (
            previous.get("artifact_id_scheme")
            == row.get("artifact_id_scheme")
            == "payload-sha256"
            and previous["payload"] == row["payload"]
        ):
            raise ValueError(f"Artifact identity collision: {identity}")
        identities[identity] = row
    by_path = {row["release_path"]: row for row in rows}
    for row in rows:
        metadata = row.get("metadata", {})
        for name in ("bold", "samples"):
            path = metadata.get(f"{name}_release_path")
            if path:
                if path not in by_path:
                    raise ValueError(
                        f"Missing {name} input referenced by {row['release_path']}"
                    )
                expected_sha = metadata.get(f"{name}_sha256")
                actual_sha = by_path[path]["payload"].get("sha256")
                if not expected_sha or expected_sha != actual_sha:
                    raise ValueError(
                        f"{name} checksum does not match selected input for {row['release_path']}"
                    )


def verify_local_source(item, row):
    """Reuse bytes only with evidence tying them to the selected S3 generation."""
    if row["source"].get("kind") != "s3":
        raise ValueError("Local source receipt must reference an S3 artifact")
    expected = row["payload"].get("sha256")
    receipt_sha = item.get("payload", {}).get("sha256")
    if not expected:
        evidence = item.get("source", {})
        if (
            any(
                evidence.get(key) != row["source"].get(key)
                for key in (
                    "bucket",
                    "key",
                    "version_id",
                    "etag",
                    "size_bytes",
                    "last_modified",
                )
            )
            or not receipt_sha
        ):
            raise ValueError(
                "Cached bytes need a matching source-generation receipt and SHA-256"
            )
        expected = receipt_sha
    elif receipt_sha is not None and receipt_sha != expected:
        raise ValueError("Cached source checksum conflicts with the manifest")
    path = Path(item["local_path"])
    if (
        not path.is_file()
        or path.stat().st_size != payload_size(row)
        or digest_file(path) != expected
    ):
        raise ValueError(
            f"Cached bytes do not match the selected source: {row['release_path']}"
        )
    row["payload"] = {
        **row["payload"],
        "sha256": expected,
        "validation": "bytes-verified",
    }
    return item, row


def apply_payload_evidence(item, row):
    """Record a verified download without implying that bytes remain staged."""
    evidence = item.get("source", {})
    payload = item.get("payload", {})
    digest = payload.get("sha256")
    if row["source"].get("kind") != "s3" or any(
        evidence.get(key) != row["source"].get(key)
        for key in (
            "bucket",
            "key",
            "version_id",
            "etag",
            "size_bytes",
            "last_modified",
        )
    ):
        raise ValueError(
            "Payload evidence does not match the selected source generation"
        )
    if (
        payload.get("validation") != "bytes-verified"
        or payload.get("size_bytes") != payload_size(row)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("Payload evidence requires a verified size and SHA-256")
    expected = row["payload"].get("sha256")
    if expected is not None and expected != digest:
        raise ValueError("Payload evidence checksum conflicts with the manifest")
    row["payload"].update(sha256=digest, validation="bytes-verified")


def write_dataset_card(template, root):
    """Preserve approved prose and derive available HF metadata configurations."""
    import yaml

    text = Path(template).read_text()
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Dataset card requires YAML frontmatter")
    metadata = yaml.safe_load(match[1])
    if not isinstance(metadata, dict):
        raise ValueError("Dataset card frontmatter must be a mapping")
    configs = [
        {
            "config_name": "planned_files",
            "default": True,
            "data_files": [
                {"split": "data", "path": "catalog/files/planned_files.parquet"}
            ],
        }
    ]
    if (root / "catalog/human_plays/human_plays.parquet").is_file():
        configs.append(
            {
                "config_name": "human_plays",
                "data_files": [
                    {"split": "data", "path": "catalog/human_plays/human_plays.parquet"}
                ],
            }
        )
    metadata["configs"] = configs
    # A files configuration would imply uploaded payload availability. Assembly
    # provides only planned_files and the independently useful human-play table.
    output = (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        + "---\n"
        + text[match.end() :]
    )
    (root / "README.md").write_text(output)


def assemble(args):
    """Combine an explicit source selection with verified generated payloads."""
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    root = Path(args.output).resolve()
    audit = Path(args.audit_output).resolve()
    if audit == root or root in audit.parents:
        raise ValueError("Private receipts must be outside the dataset directory")
    root.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for filename in args.path_map:
        for item in read_rows(filename):
            key = item["artifact_id"]
            if key in mapping:
                raise ValueError(f"Repeated mapping for {key}")
            if item["action"] not in {"include", "exclude"} or not item.get(
                "selection_reason"
            ):
                raise ValueError(
                    "Every source mapping needs an explicit action and reason"
                )
            mapping[key] = item
    rows, decisions = [], []
    for row in read_rows(args.source_manifest):
        item = mapping.pop(row["artifact_id"], None)
        if item is None:
            raise ValueError(
                f"Source artifact has no reviewed mapping: {row['release_path']}"
            )
        decisions.append(
            {"artifact_id": row["artifact_id"], "source_uri": row["source_uri"], **item}
        )
        if item["action"] == "include":
            rows.append(dataset_source_row(row, item, args.release_id))
    if mapping:
        raise ValueError("Path map contains artifacts absent from the source manifest")
    additions = []
    for filename in args.local_files or []:
        additions.extend(read_rows(filename))
    if args.human_manifest:
        if not args.human_directory:
            raise ValueError("--human-manifest requires --human-directory")
        human_root = Path(args.human_directory)
        for row in read_rows(args.human_manifest):
            relative = row["release_path"]
            if not safe_relative(relative):
                raise ValueError("Unsafe human file path")
            additions.append(
                {
                    "local_path": str(human_root / relative),
                    "release_path": relative,
                    "component": "human_behavior",
                    "metadata": {
                        **row["metadata"],
                        "condition": row["metadata"]["suggestion_level"],
                        "timebase": "recorded-engine-frame",
                        "paper_role": "human-reference",
                    },
                    "payload": row["payload"],
                    "license": row["license"],
                    "provenance": row["provenance"],
                    "selection_reason": "Participant/game trajectory and exact action-only prompts; one self-contained file per condition",
                }
            )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        local_rows = list(
            pool.map(lambda item: local_dataset_row(item, args.release_id), additions)
        )
    rows.extend(local_rows)
    staged_sources = []
    if getattr(args, "source_files", None):
        by_source = {
            row["source_uri"]: row for row in rows if row["source_uri"] is not None
        }
        cached = [
            item for filename in args.source_files for item in read_rows(filename)
        ]
        if len({item["source_uri"] for item in cached}) != len(cached):
            raise ValueError("Duplicate local source receipt")
        if any(item["source_uri"] not in by_source for item in cached):
            raise ValueError("Local source receipt refers to an unselected artifact")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            staged_sources = list(
                pool.map(
                    lambda item: verify_local_source(
                        item, by_source[item["source_uri"]]
                    ),
                    cached,
                )
            )
    if args.enrichments:
        by_source = {
            row["source_uri"]: row for row in rows if row["source_uri"] is not None
        }
        by_path = {row["release_path"]: row for row in rows}
        for filename in args.enrichments:
            for item in read_rows(filename):
                row = (
                    by_source.get(item.get("source_uri"))
                    if item.get("source_uri")
                    else by_path.get(item.get("release_path"))
                )
                if row is None:
                    raise ValueError("Enrichment refers to an unselected source")
                if (
                    item.get("release_path")
                    and item["release_path"] != row["release_path"]
                ):
                    raise ValueError("Enrichment source and public path disagree")
                row["metadata"].update(item.get("metadata", {}))
                row["provenance"].update(item.get("provenance", {}))
                if "payload" in item:
                    apply_payload_evidence(item, row)
    rows.sort(key=lambda row: row["release_path"])
    check_public_dataset_paths(rows)
    total = sum(payload_size(row) for row in rows)
    if total > args.max_bytes:
        raise ValueError(
            f"Selected payloads exceed capacity: {total} > {args.max_bytes}"
        )
    # Validate every destination before copying any local payloads. Source rows
    # remain planned; no S3 or HF request is made by this operation.
    reserved = {
        "README.md",
        "manifest.jsonl.gz",
        "catalog/files/planned_files.parquet",
        "catalog/metadata.json",
    }
    for row in rows:
        if row["release_path"] in reserved:
            raise ValueError("Payload path collides with generated catalogue metadata")
    previous_manifest = root / "manifest.jsonl.gz"
    if previous_manifest.exists():
        selected_paths = {row["release_path"] for row in rows}
        stale = [
            row["release_path"]
            for row in read_rows(previous_manifest)
            if row["release_path"] not in selected_paths
            and (root / row["release_path"]).exists()
        ]
        if stale:
            raise ValueError(
                "Output contains payloads absent from this selection; use a fresh directory or remove verified stale copies: "
                + ", ".join(stale[:5])
            )

    replacement_paths = {row["release_path"] for row in local_rows} | {
        row["release_path"] for _, row in staged_sources
    }

    def verify_retained_target(row):
        if row["release_path"] in replacement_paths:
            return
        target = root / row["release_path"]
        if not target.exists():
            return
        target = destination(root, row["release_path"])
        expected = row["payload"].get("sha256")
        if (
            not target.is_file()
            or not expected
            or target.stat().st_size != payload_size(row)
            or digest_file(target) != expected
        ):
            raise ValueError(
                "Existing payload does not match this selection; provide verified replacement bytes: "
                + row["release_path"]
            )
        row["payload"]["validation"] = "bytes-verified"
        return {"local_path": str(target), "source_uri": row["source_uri"]}, row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        staged_sources.extend(
            item for item in pool.map(verify_retained_target, rows) if item is not None
        )

    def copy_local(pair):
        addition, row = pair
        original = Path(addition["local_path"]).resolve()
        target = destination(root, row["release_path"])
        if target == original:
            return
        if (
            target.exists()
            and target.stat().st_size == payload_size(row)
            and digest_file(target) == row["payload"]["sha256"]
        ):
            return
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".stage-", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(original, temporary)
            if (
                temporary.stat().st_size != payload_size(row)
                or digest_file(temporary) != row["payload"]["sha256"]
            ):
                raise ValueError("Local source changed during staging")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(copy_local, [*zip(additions, local_rows), *staged_sources]))
    with write_rows(root / "manifest.jsonl.gz") as handle:
        for row in rows:
            emit(handle, row)
    catalogue(
        argparse.Namespace(
            manifest=root / "manifest.jsonl.gz",
            tier=None,
            component=None,
            output=root / "catalog/files/planned_files.parquet",
        )
    )
    groups = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in rows:
        groups[row["component"]]["files"] += 1
        groups[row["component"]]["bytes"] += payload_size(row)
    summary = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "release_id": args.release_id,
        "publication_status": "planned; no payload uploads performed",
        "files": len(rows),
        "payload_bytes": total,
        "generated_local_files": len(local_rows),
        "generated_local_bytes": sum(payload_size(row) for row in local_rows),
        "source_files": len(rows) - len(local_rows),
        "staged_source_files": len(staged_sources),
        "staged_source_bytes": sum(payload_size(row) for _, row in staged_sources),
        "payloads_with_sha256": sum(bool(row["payload"].get("sha256")) for row in rows),
        "manifest_sha256": digest_file(root / "manifest.jsonl.gz"),
        "components": dict(sorted(groups.items())),
    }
    write_json(root / "catalog/metadata.json", summary)
    if getattr(args, "card", None):
        write_dataset_card(args.card, root)
    audit.mkdir(parents=True, exist_ok=True)
    with write_rows(audit / "source-decisions.jsonl.gz") as handle:
        for item in decisions:
            emit(handle, item)
    with write_rows(audit / "local-staging.jsonl.gz") as handle:
        for addition, row in [*zip(additions, local_rows), *staged_sources]:
            emit(
                handle,
                {
                    "local_path": str(Path(addition["local_path"]).resolve()),
                    "release_path": row["release_path"],
                    "source_uri": row["source_uri"],
                    "sha256": row["payload"]["sha256"],
                    "size_bytes": payload_size(row),
                },
            )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "assemble",
        help="Assemble an explicit source-to-dataset map and verified local artifacts",
    )
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--path-map", action="append", required=True)
    p.add_argument("--local-files", action="append")
    p.add_argument("--human-manifest")
    p.add_argument("--human-directory")
    p.add_argument(
        "--card",
        help="Approved dataset-card Markdown; metadata configurations are generated from staged catalogues",
    )
    p.add_argument("--enrichments", action="append")
    p.add_argument(
        "--source-files",
        action="append",
        help="Private local-cache receipts tying bytes to selected source generations",
    )
    p.add_argument("--release-id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--audit-output",
        required=True,
        help="Private receipts outside the dataset directory",
    )
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--max-bytes", type=int, default=5_000_000_000_000)
    p.set_defaults(func=assemble)
    p = commands.add_parser(
        "build", help="Select exact source objects with a versioned policy"
    )
    p.add_argument("--inventory", required=True)
    p.add_argument(
        "--policy",
        required=True,
        help="Explicit selection-policy JSON; see docs/release/policy.example.json for synthetic syntax",
    )
    p.add_argument("--output", required=True)
    p.set_defaults(func=build)
    p = commands.add_parser(
        "merge-verified",
        help="Add staged payload hashes without replacing selection metadata",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--verified", action="append", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=merge_verified)
    for name, function in (
        ("validate", validate),
        ("freeze", freeze),
        ("stage", stage),
        ("catalogue", catalogue),
    ):
        p = commands.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--tier", action="append")
        p.add_argument("--component", action="append")
        p.set_defaults(func=function)
        if name == "validate":
            p.add_argument("--require-staged", action="store_true")
        else:
            p.add_argument("--output", required=True)
        if name in {"freeze", "stage"}:
            p.add_argument("--region", default="eu-west-2")
        if name == "freeze":
            p.add_argument("--workers", type=int, default=32)
            p.add_argument(
                "--reuse-pins",
                help="Reuse matching version IDs from a previously verified manifest; preserves original check timestamps",
            )
        if name == "stage":
            p.add_argument("--directory", required=True)
            p.add_argument("--max-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
