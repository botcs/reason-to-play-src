#!/usr/bin/env python3
"""Build a typed human-play catalogue from self-contained human replay JSON.

Select one prompt condition so repeated observations are counted once. Keep every
play present in those files and expose practice, cohort and level flags rather
than silently filtering them. Frame clocks and nullable outcomes follow the
shared behavior reader. Paths and hashes identify the exact compressed replay.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reason_to_play.data.behavior import GAMES, behavior_root, game_identity  # noqa: E402
from reason_to_play.data.replay_behavior import (  # noqa: E402
    CONDITIONS,
    read_record,
    record_plays,
    recording_path,
    replay_paths,
)


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
        pa.field("prompt_condition", pa.string(), nullable=False),
    ],
    metadata={
        b"catalogue_schema_version": b"3",
        b"input_format": b"human-replay-json",
        b"artifact_id_scheme": b"payload-sha256",
        b"original_win": b"Original play win: true, false or null; never the terminal frame win",
        b"outcome": b"win, avatar_died, loss, incomplete; avatar death checked before null/incomplete",
        b"engine_frames": b"Contiguous engine updates: state_count-1; frame time must equal its play-relative index",
        b"keypress_frames": b"States with keyPressType not null; not decisions or elapsed time",
        b"is_practice": b"run=0 or game=sokoban",
        b"is_common_level": b"0<=level<=8; independent of practice flag",
        b"is_primary_encoding_cohort": b"Participants sub-12 through sub-32",
        b"in_common_comparison": b"Common level, non-practice, and game in the seven study games",
        b"source_document_index": b"Original zero-based play ordinal within its scanner run; not the per-game file position or play_id",
        b"source_release_path": b"Self-contained replay path relative to dataset root",
        b"source_payload_sha256": b"SHA256 of the exact compressed replay used to build this row",
        b"prompt_condition": b"One selected prompt condition; trajectories are not counted again for other conditions",
    },
)
MEASUREMENT_FIELDS = SCHEMA.names[: SCHEMA.names.index("source_artifact_id")]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_source(arguments) -> tuple[list[dict], dict]:
    path, expected_relative = arguments
    digest = sha256_file(path)
    record = read_record(path, expand=False)
    relative = recording_path(record)
    if expected_relative is not None and relative != expected_relative:
        raise ValueError(f"Replay identity differs from its path: {path}")
    source = {
        "release_path": "behavior/human/" + relative,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }
    condition = record["meta"]["suggestion_level"]
    # Delta encoding changes only sprite dictionaries. The catalogue uses clocks,
    # keypresses and outcome events, all stored in full for every frame. Project
    # away sprites before using the shared reader so this metadata-only operation
    # does not expand millions of sprite snapshots or load the game engine.
    for frame in record["states"]:
        frame["sprites"] = {}
    rows = []
    for play in record_plays(record):
        game, cohort = game_identity(play["game_name"])
        run, level = int(play["run_id"]), int(play["level_id"])
        subject = int(play["subj_id"])
        frames = play["states"]
        practice = run == 0 or game == "sokoban"
        common = 0 <= level <= 8
        rows.append(
            {
                "original_id": play["_id"],
                "subject": f"sub-{subject:02d}",
                "run": run,
                "play_id": int(play["play_id"]),
                "source_document_index": play["_canonical"]["source_document_index"],
                "game": game,
                "game_variant": play["game_name"],
                "cohort": cohort,
                "level": level,
                "original_win": play["win"],
                "outcome": play["_canonical"]["outcome"],
                "engine_frames": len(frames) - 1,
                "keypress_frames": sum(
                    frame.get("keyPressType") is not None for frame in frames
                ),
                "state_count": len(frames),
                "is_practice": practice,
                "is_common_level": common,
                "is_primary_encoding_cohort": 12 <= subject <= 32,
                "in_common_comparison": common and not practice and game in GAMES,
                "source_artifact_id": "sha256:" + digest,
                "source_release_path": source["release_path"],
                "source_payload_sha256": digest,
                "prompt_condition": condition,
            }
        )
    if sha256_file(path) != digest:
        raise ValueError(f"Replay bytes changed while cataloguing: {path}")
    return rows, source


def build_catalogue(
    root: Path,
    output: Path,
    *,
    condition: str = "elaborate",
    progress=None,
    workers: int = 1,
) -> dict:
    """Read a release root, human directory or standalone human replay.

    Directory inputs select one condition (default: elaborate). A standalone file
    always selects that file's own condition, which is recorded in the output.
    Original play identities and run ordinals must be unique across the selection.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown human prompt condition: {condition!r}")
    root, output = behavior_root(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Catalogue output must be a new directory: {output}")
    paths = replay_paths(root, condition=condition)
    if not paths:
        raise FileNotFoundError(f"No {condition} human replay files in {root}")
    arguments = [
        (path, None if root.is_file() else path.relative_to(root).as_posix())
        for path in paths
    ]
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else None
    )
    rows, sources, identities, ordinals = [], [], set(), set()
    try:
        batches = (
            pool.map(read_source, arguments) if pool else map(read_source, arguments)
        )
        for index, (batch, source) in enumerate(batches):
            for row in batch:
                if row["original_id"] in identities:
                    raise ValueError(
                        f"Duplicate original play ID: {row['original_id']}"
                    )
                ordinal = row["subject"], row["run"], row["source_document_index"]
                if ordinal in ordinals:
                    raise ValueError(f"Duplicate original run ordinal: {ordinal}")
                identities.add(row["original_id"])
                ordinals.add(ordinal)
                rows.append(row)
            sources.append(source)
            if progress:
                progress(index + 1, len(paths), len(rows))
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    rows.sort(
        key=lambda row: (row["subject"], row["run"], row["source_document_index"])
    )
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    table.validate(full=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".human-catalogue-", dir=output.parent
    ) as tmp:
        target = Path(tmp) / "human_plays.parquet"
        pq.write_table(table, target, compression="zstd")
        report = {
            "schema_version": 3,
            "table": "human_plays",
            "input_format": "human-replay-json",
            "artifact_id_scheme": "payload-sha256",
            "prompt_condition": rows[0]["prompt_condition"],
            "source_file_count": len(sources),
            "row_count": len(rows),
            "unique_original_ids": len(identities),
            "subject_count": len({row["subject"] for row in rows}),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
            "original_win_null_count": sum(row["original_win"] is None for row in rows),
            "practice_count": sum(row["is_practice"] for row in rows),
            "outside_common_levels_count": sum(
                not row["is_common_level"] for row in rows
            ),
            "common_comparison_count": sum(row["in_common_comparison"] for row in rows),
            "cohort_counts": dict(
                sorted(Counter(row["cohort"] for row in rows).items())
            ),
            "canonical_inputs": sources,
            "parquet": {
                "file": target.name,
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            },
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in SCHEMA
            ],
            "definitions": {
                key.decode(): value.decode() for key, value in SCHEMA.metadata.items()
            },
            "validation": "Replays rehashed before and after reading; unique original IDs and run ordinals; original frame clocks and outcomes verified by shared reader; full Arrow table validation",
            "selection": "Every play in one prompt condition; no practice, cohort or level filtering",
        }
        (Path(tmp) / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        Path(tmp).rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dataset root, behavior/human directory, or standalone human replay",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="elaborate")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    def progress(done, total, rows):
        if done % 16 == 0 or done == total:
            print(f"Read {done}/{total} replay files; {rows} plays", flush=True)

    report = build_catalogue(
        args.input,
        args.output,
        condition=args.condition,
        workers=args.workers,
        progress=progress,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "row_count",
                    "subject_count",
                    "outcomes",
                    "prompt_condition",
                    "parquet",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
