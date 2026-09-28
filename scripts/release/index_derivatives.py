#!/usr/bin/env python3
"""Read-only S3 inventory and reversible release-path manifest builder.

No object bodies are downloaded; no remote objects are modified. A candidate
is a proposed release asset, not a verified input to the paper analyses.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from urllib.parse import quote


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def scan_objects(client, bucket, prefix=""):
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            yield {
                "bucket": bucket,
                "key": obj["Key"],
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
                "etag": obj["ETag"].strip('"'),
                "storage_class": obj.get("StorageClass"),
                "checksum_algorithms": obj.get("ChecksumAlgorithm", []),
                "checksum_type": obj.get("ChecksumType"),
            }


def scan(args):
    import boto3  # Offline manifest building needs only the standard library.

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = now()
    count = size = 0
    client = boto3.client("s3", region_name=args.region)
    partial = output / "s3-objects.jsonl.gz.partial"
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        for row in scan_objects(client, args.bucket, args.prefix):
            handle.write(json.dumps(row) + "\n")
            count += 1
            size += row["size_bytes"]
            if count % 10000 == 0:
                print(f"Indexed {count:,} objects", flush=True)
    partial.replace(output / "s3-objects.jsonl.gz")
    write_json(
        output / "scan.json",
        {
            "schema_version": 1,
            "bucket": args.bucket,
            "prefix": args.prefix,
            "region": args.region,
            "started_at": started,
            "completed_at": now(),
            "objects": count,
            "bytes": size,
            "snapshot_semantics": "Current keys, non-atomic ListObjectsV2 scan; no historical versions",
            "integrity": "ETags are opaque; object content checksums have not been verified",
        },
    )


def load_rules(path):
    rules = json.loads(path.read_text())
    if rules.get("schema_version") != 1:
        raise ValueError("Unsupported rules schema")
    for prefix, rule in rules["prefixes"].items():
        if rule["disposition"] not in {"candidate", "review", "exclude"}:
            raise ValueError(f"Invalid disposition for {prefix}")
        root = rule["release_root"]
        if root.startswith("/") or any(p in {"", ".", ".."} for p in root.split("/")):
            raise ValueError(f"Unsafe release root for {prefix}")
    return rules


def classify(row, rules):
    key = row["key"]
    parts = key.split("/")
    prefix = parts[0]
    rule = rules["prefixes"].get(prefix)
    disposition = rule["disposition"] if rule else "review"
    reason = rule["reason"] if rule else "Unmapped prefix; explicit review required"
    release_path = None
    unsafe = (
        key.startswith("/")
        or "\\" in key
        or any(ord(c) < 32 or ord(c) == 127 for c in key)
        or any(p in {"", ".", ".."} for p in parts)
    )
    if key.endswith("/"):
        disposition, reason = "exclude", "Directory marker, not a data file"
    elif unsafe:
        disposition, reason = (
            "exclude",
            "Unsafe portable path; source key retained for audit",
        )
    elif any(
        p in {".git", ".aws", ".ssh", "__pycache__", ".ipynb_checkpoints", ".venv"}
        for p in parts
    ) or parts[-1] in {".env", "credentials", "token", "stored_tokens", ".DS_Store"}:
        disposition, reason = (
            "exclude",
            "Repository/cache/credential material is not a dataset asset",
        )
    elif rule:
        # Encode each path component reversibly (e.g. | -> %7C). Never infer
        # model IDs or rename game aliases from a lossy filename slug.
        suffix = "/".join(
            quote(p, safe="-._~") for p in (parts[1:] if len(parts) > 1 else parts)
        )
        release_path = f"{rule['release_root']}/{suffix}"
    subject_match = re.search(r"(?:^|/|\|)(?:sub-|subj)(\d+)(?:/|_|\||\.)", key)
    tokens = [token for part in parts for token in part.split("|")]
    model_match = next((p for p in parts if p.startswith("model-")), None)
    game_match = next(
        (
            p
            for p in tokens
            if re.fullmatch(r"(?:vgfmri[34]_[A-Za-z]+|[A-Za-z]+_vgfmri[34])", p)
        ),
        None,
    )
    return {
        **row,
        "source_uri": f"s3://{row['bucket']}/{key}",
        "source_prefix": prefix,
        "release_path": release_path,
        "component": rule["component"] if rule else "unclassified",
        "disposition": disposition,
        "reason": reason,
        "subject": f"sub-{int(subject_match[1]):02d}" if subject_match else None,
        "model_path_id": model_match,
        "source_cohort": parts[1]
        if prefix == "extract_model_features_to_py" and len(parts) > 2
        else None,
        "extraction_variant": next(
            (p for p in parts if p in {"all", "compressed"}), None
        ),
        "game_path_id": game_match,
        "rationale_mode": next(
            (
                p
                for p in tokens
                if p in {"action-only", "copied-reasoning", "prompted-rationale"}
            ),
            None,
        ),
        "suggestion_level": next(
            (p for p in tokens if p in {"minimal", "elaborate", "oracle"}), None
        ),
        "provenance_status": "object-metadata-only",
    }


def build(args):
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source in {output / "objects.jsonl.gz", output / "release-candidates.jsonl.gz"}:
        raise ValueError("Output would overwrite the source inventory")
    output.mkdir(parents=True, exist_ok=True)
    rules = load_rules(Path(args.rules))
    prefixes = {}
    dispositions = defaultdict(lambda: {"objects": 0, "bytes": 0})
    coverage = defaultdict(lambda: {"objects": 0, "bytes": 0, "subjects": set()})
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix=".inventory-", dir=output) as temp:
        temp = Path(temp)
        db = sqlite3.connect(temp / "paths.sqlite")
        db.execute("CREATE TABLE paths (path TEXT PRIMARY KEY, source TEXT UNIQUE)")
        db.execute("CREATE TABLE sources (uri TEXT PRIMARY KEY)")
        try:
            with (
                gzip.open(temp / "objects.jsonl.gz", "wt", encoding="utf-8") as all_out,
                gzip.open(
                    temp / "release-candidates.jsonl.gz", "wt", encoding="utf-8"
                ) as candidate_out,
            ):
                for raw in rows(source):
                    row = classify(raw, rules)
                    db.execute("INSERT INTO sources VALUES (?)", (row["source_uri"],))
                    if row["release_path"] is not None:
                        try:
                            db.execute(
                                "INSERT INTO paths VALUES (?, ?)",
                                (row["release_path"], row["source_uri"]),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(
                                f"Release path collision: {row['release_path']}"
                            ) from exc
                    encoded = json.dumps(row) + "\n"
                    all_out.write(encoded)
                    if row["disposition"] == "candidate":
                        candidate_out.write(encoded)
                    counts["objects"] += 1
                    counts["bytes"] += row["size_bytes"]
                    d = dispositions[row["disposition"]]
                    d["objects"] += 1
                    d["bytes"] += row["size_bytes"]
                    p = prefixes.setdefault(
                        row["source_prefix"],
                        {
                            "objects": 0,
                            "bytes": 0,
                            "component": row["component"],
                            "dispositions": Counter(),
                            "examples": [],
                        },
                    )
                    p["objects"] += 1
                    p["bytes"] += row["size_bytes"]
                    p["dispositions"][row["disposition"]] += 1
                    if len(p["examples"]) < 3:
                        p["examples"].append(row["key"])
                    group = (
                        row["source_prefix"],
                        row["source_cohort"] or "",
                        row["model_path_id"] or "",
                        row["extraction_variant"] or "",
                        row["rationale_mode"] or "",
                        row["suggestion_level"] or "",
                    )
                    c = coverage[group]
                    c["objects"] += 1
                    c["bytes"] += row["size_bytes"]
                    if row["subject"]:
                        c["subjects"].add(row["subject"])
        finally:
            db.close()
        summary = {
            "schema_version": 1,
            "created_at": now(),
            "source_inventory": source.name,
            **counts,
            "dispositions": dict(dispositions),
            "prefixes": prefixes,
            "release_status": "Proposed selection; not a publication approval or paper completeness claim",
            "integrity": "No payloads downloaded; ETag is not a verified content checksum",
            "path_policy": "Reversible component mapping; source suffixes percent-encoded per segment; duplicate paths rejected",
        }
        write_json(temp / "summary.json", summary)
        write_json(temp / "asset-rules.json", rules)
        with (temp / "prefixes.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "source_prefix",
                    "component",
                    "objects",
                    "bytes",
                    "candidate_objects",
                    "review_objects",
                    "excluded_objects",
                ]
            )
            for prefix, p in sorted(prefixes.items()):
                writer.writerow(
                    [
                        prefix,
                        p["component"],
                        p["objects"],
                        p["bytes"],
                        *[
                            p["dispositions"][d]
                            for d in ("candidate", "review", "exclude")
                        ],
                    ]
                )
        with (temp / "coverage.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "source_prefix",
                    "source_cohort",
                    "model_path_id",
                    "extraction_variant",
                    "rationale_mode",
                    "suggestion_level",
                    "objects",
                    "bytes",
                    "subject_count",
                    "subjects",
                ]
            )
            for group, c in sorted(coverage.items()):
                writer.writerow(
                    [
                        *group,
                        c["objects"],
                        c["bytes"],
                        len(c["subjects"]),
                        ";".join(sorted(c["subjects"])),
                    ]
                )
        for name in (
            "objects.jsonl.gz",
            "release-candidates.jsonl.gz",
            "summary.json",
            "asset-rules.json",
            "prefixes.csv",
            "coverage.csv",
        ):
            (temp / name).replace(output / name)
    print(json.dumps({**counts, "dispositions": dict(dispositions)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan", help="List all current S3 object metadata")
    scan_parser.add_argument("--bucket", required=True)
    scan_parser.add_argument("--region", default="eu-west-2")
    scan_parser.add_argument("--prefix", default="")
    scan_parser.add_argument("--output", required=True)
    build_parser = sub.add_parser(
        "build", help="Build manifests offline from a completed scan"
    )
    build_parser.add_argument("--input", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument(
        "--rules",
        required=True,
        help="Explicit prefix-classification JSON; see docs/release/asset-rules.example.json for synthetic syntax",
    )
    args = parser.parse_args()
    (scan if args.command == "scan" else build)(args)


if __name__ == "__main__":
    main()
