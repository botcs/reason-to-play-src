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
import sqlite3
import tempfile
from urllib.parse import quote


SCHEMA_VERSION = 1
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
            if value in seen:
                raise ValueError(f"Duplicate {label}: {value}")
            seen.add(value)
        counts["objects"] += 1
        counts["bytes"] += row["source"]["size_bytes"]
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
                "payload": {"sha256": expected, "validation": "bytes-verified"},
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
        return {**row, "payload": {"sha256": sha, "validation": "bytes-verified"}}
    finally:
        body.close()
        if partial is not None:
            partial.unlink(missing_ok=True)


def stage(args):
    import boto3

    chosen = list(selected_rows(args))
    needed = sum(r["source"]["size_bytes"] for r in chosen)
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
                "size_bytes": row["source"]["size_bytes"],
                "sha256": row["payload"]["sha256"],
                "source_uri": row["source_uri"],
                "source_version_id": row["source"].get("version_id"),
                "source_verification": row["source"]["verification"],
                "payload_validation": row["payload"]["validation"],
                "subject": row["metadata"].get("subject"),
                "model_path_id": row["metadata"].get("model_path_id"),
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
                    if name == "size_bytes"
                    else pa.bool_()
                    if name == "published"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
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
            p.add_argument("--workers", type=int, default=16)
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
