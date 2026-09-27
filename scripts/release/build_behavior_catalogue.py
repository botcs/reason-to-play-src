#!/usr/bin/env python3
"""Build a typed, per-attempt human-play catalogue from staged BSON sources.

Keep every original play, including practice and levels outside the common 0–8
comparison. Outcome classification delegates to the behavioral analysis exporter.
Engine frames and non-idle keypress frames are separate measurements. No raw game
states, participant timestamps or credentials are copied into the catalogue.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import re
import sys
import zlib

import bson
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.build_episodes import (  # noqa: E402
    GAMES,
    game_identity,
    human_outcome,
)


PLAY_PATH = re.compile(r"behavior/human/plays/(sub-\d{2})/run-(\d{2})\.bson")
SCHEMA = pa.schema(
    [
        pa.field("original_id", pa.string(), nullable=False),
        pa.field("subject", pa.string(), nullable=False),
        pa.field("run", pa.int16(), nullable=False),
        pa.field("play_id", pa.int32(), nullable=False),
        pa.field("source_document_index", pa.int32(), nullable=False),
        pa.field("game", pa.string(), nullable=False),
        pa.field("game_variant", pa.string(), nullable=False),
        pa.field("cohort", pa.string(), nullable=False),
        pa.field("level", pa.int16(), nullable=False),
        pa.field("original_win", pa.bool_(), nullable=True),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("engine_frames", pa.int64(), nullable=False),
        pa.field("keypress_frames", pa.int64(), nullable=False),
        pa.field("state_count", pa.int64(), nullable=False),
        pa.field("is_practice", pa.bool_(), nullable=False),
        pa.field("is_common_level", pa.bool_(), nullable=False),
        pa.field("is_primary_encoding_cohort", pa.bool_(), nullable=False),
        pa.field("in_common_comparison", pa.bool_(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
        pa.field("source_release_path", pa.string(), nullable=False),
        pa.field("source_payload_sha256", pa.string(), nullable=False),
    ],
    metadata={
        b"catalogue_schema_version": b"1",
        b"original_win": b"Original play document win: true, false or null; never terminal zstate win",
        b"outcome": b"win, avatar_died, loss, incomplete; avatar death checked before null/incomplete",
        b"engine_frames": b"Number of contiguous engine updates: len(states)-1; states[i].gt must equal i",
        b"keypress_frames": b"Count of states with keyPressType not null; not a decision count or elapsed time",
        b"is_practice": b"run=0 or game=sokoban",
        b"is_common_level": b"0<=level<=8; independent of practice flag",
        b"is_primary_encoding_cohort": b"Participants sub-12 through sub-32",
        b"in_common_comparison": b"Common level, non-practice, and game in the seven study games",
        b"source_document_index": b"Zero-based ordinal within the source BSON file; original play_id is kept separately",
    },
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def human_sources(manifest: Path) -> list[dict]:
    """Use verified staged evidence; do not infer an S3 identity from a filename."""
    opener = gzip.open if manifest.suffix == ".gz" else open
    sources = {}
    with opener(manifest, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            path = row["release_path"]
            if not PLAY_PATH.fullmatch(path):
                continue
            if path in sources:
                raise ValueError(f"Duplicate manifest source path: {path}")
            if row.get("source", {}).get("verification") not in {
                "version-pinned",
                "head-verified-unversioned",
            }:
                raise ValueError(f"Source is not frozen: {path}")
            payload = row.get("payload", {})
            if payload.get("validation") != "bytes-verified" or not re.fullmatch(
                r"[0-9a-f]{64}", payload.get("sha256", "")
            ):
                raise ValueError(f"Source payload is not verified: {path}")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", row["artifact_id"]):
                raise ValueError(f"Invalid source artifact ID: {path}")
            sources[path] = row
    if not sources:
        raise ValueError("Manifest has no verified human play BSON sources")
    return [sources[path] for path in sorted(sources)]


def play_row(doc: dict, source: dict, document_index: int) -> dict:
    match = PLAY_PATH.fullmatch(source["release_path"])
    subject, run = match[1], int(match[2])
    if int(doc["subj_id"]) != int(subject.removeprefix("sub-")):
        raise ValueError(
            f"Subject does not match source path: {source['release_path']}"
        )
    if int(doc["run_id"]) != run:
        raise ValueError(f"Run does not match source path: {source['release_path']}")
    if doc["_id"] is None:
        raise ValueError("Missing original play ID")
    original_id = str(doc["_id"])
    if not original_id:
        raise ValueError("Missing original play ID")
    game, cohort = game_identity(doc["game_name"])
    level = int(doc["level_id"])
    states = bson.decode(zlib.decompress(doc["zstates"]))["states"]
    if not states:
        raise ValueError(f"Empty zstates in play {original_id}")
    if any(state["gt"] != i for i, state in enumerate(states)):
        raise ValueError(f"Non-contiguous engine clock in play {original_id}")
    practice = run == 0 or game == "sokoban"
    common_level = 0 <= level <= 8
    return {
        "original_id": original_id,
        "subject": subject,
        "run": run,
        "play_id": int(doc["play_id"]),
        "source_document_index": document_index,
        "game": game,
        "game_variant": doc["game_name"],
        "cohort": cohort,
        "level": level,
        "original_win": doc["win"],
        "outcome": human_outcome(doc, states),
        "engine_frames": len(states) - 1,
        "keypress_frames": sum(
            state.get("keyPressType") is not None for state in states
        ),
        "state_count": len(states),
        "is_practice": practice,
        "is_common_level": common_level,
        "is_primary_encoding_cohort": 12 <= int(doc["subj_id"]) <= 32,
        "in_common_comparison": common_level and not practice and game in GAMES,
        "source_artifact_id": source["artifact_id"],
        "source_release_path": source["release_path"],
        "source_payload_sha256": source["payload"]["sha256"],
    }


def read_source(arguments: tuple[Path, dict]) -> list[dict]:
    stage_root, source = arguments
    path = stage_root / source["release_path"]
    if sha256_file(path) != source["payload"]["sha256"]:
        raise ValueError(f"Staged bytes differ from manifest: {source['release_path']}")
    with path.open("rb") as stream:
        return [
            play_row(doc, source, index)
            for index, doc in enumerate(bson.decode_file_iter(stream))
        ]


def build_catalogue(
    stage_root: Path, manifest: Path, output: Path, progress=None, workers: int = 1
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    sources = human_sources(manifest)
    rows, identities = [], set()
    arguments = [(stage_root, source) for source in sources]
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        batches = (
            pool.map(read_source, arguments) if pool else map(read_source, arguments)
        )
        for index, batch in enumerate(batches):
            for row in batch:
                if row["original_id"] in identities:
                    raise ValueError(
                        f"Duplicate original play ID: {row['original_id']}"
                    )
                identities.add(row["original_id"])
                rows.append(row)
            if progress:
                progress(index + 1, len(sources), len(rows))
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    if not rows:
        raise ValueError("Human source files contain no plays")
    schema = SCHEMA.with_metadata(
        {**SCHEMA.metadata, b"input_manifest_sha256": sha256_file(manifest).encode()}
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    table.validate(full=True)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "human_plays.parquet"
    partial = output / "human_plays.parquet.partial"
    pq.write_table(table, partial, compression="zstd")
    partial.replace(target)
    report = {
        "schema_version": 1,
        "table": "human_plays",
        "input_manifest_sha256": sha256_file(manifest),
        "source_file_count": len(sources),
        "row_count": len(rows),
        "unique_original_ids": len(identities),
        "subject_count": len({row["subject"] for row in rows}),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "original_win_null_count": sum(row["original_win"] is None for row in rows),
        "practice_count": sum(row["is_practice"] for row in rows),
        "outside_common_levels_count": sum(not row["is_common_level"] for row in rows),
        "common_comparison_count": sum(row["in_common_comparison"] for row in rows),
        "cohort_counts": dict(sorted(Counter(row["cohort"] for row in rows).items())),
        "parquet": {
            "file": target.name,
            "sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
        },
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
        "definitions": {
            key.decode(): value.decode() for key, value in schema.metadata.items()
        },
        "validation": "All source bytes rehashed; unique original IDs; contiguous engine clocks; full Arrow table validation",
        "selection": "All source plays retained; no practice, cohort or level filtering",
    }
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel BSON decoders; output remains in manifest path/document order",
    )
    args = parser.parse_args()

    def progress(done, total, rows):
        if done % 16 == 0 or done == total:
            print(f"Read {done}/{total} source files; {rows} plays", flush=True)

    report = build_catalogue(
        args.stage_root, args.manifest, args.output, progress, args.workers
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "row_count",
                    "subject_count",
                    "outcomes",
                    "practice_count",
                    "outside_common_levels_count",
                    "parquet",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
