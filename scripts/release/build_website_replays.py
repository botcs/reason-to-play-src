#!/usr/bin/env python3
# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Build compact browser replay copies from checksum-pinned local research files.

All top-level fields, conversation records, timing, outcomes and translated game
text are retained. Sprite engine fields unused by rendering and two raw human input
fields are omitted. Sprite-type deltas preserve every visible frame and fractional
coordinate. Research input files are read only. This module does not publish.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path, PurePosixPath
import resource
import time

SPRITE_OMIT = frozenset({"lastmove", "cooldown", "speed", "_age"})
FRAME_OMIT = frozenset({"keystate", "kill_list_ID"})
INDEX_PATH = "website-assets/replays/manifest.json"


def json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ValueError(f"Unsafe relative path: {relative!r}")
    target = Path(root) / relative
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes root: {relative!r}")
    return target


def canonical_path(relative):
    """Resolve the catalogue's compact or complete replay path to research data."""
    if relative.startswith("website-assets/replays/"):
        relative = "behavior/" + relative.removeprefix("website-assets/replays/")
    if not relative.startswith(("behavior/human/", "behavior/lrm/")):
        raise ValueError("Catalogue replay must identify human or LRM behavior")
    return relative


def frames(data):
    """Expand only the current sprite dictionary; never deep-copy trajectories."""
    previous = {}
    encoded = data.get("delta_encoded", False)
    for index, frame in enumerate(data["states"]):
        if encoded:
            if index == 0 and not isinstance(frame.get("sprites"), dict):
                raise ValueError("First delta frame needs a full sprite dictionary")
            current = dict(previous)
            for key, value in frame.get("sprites", {}).items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
        else:
            current = frame["sprites"]
        if not isinstance(current, dict) or any(
            not isinstance(v, list) for v in current.values()
        ):
            raise ValueError("Invalid sprite dictionary")
        previous = current
        yield frame, current


def update_digest(digest, value):
    payload = json_bytes(value)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def compact_replay(data):
    """Return a browser replay and equivalence evidence without mutating input."""
    if not isinstance(data.get("states"), list) or not data["states"]:
        raise ValueError("Replay requires a non-empty states array")
    compact = {
        key: value
        for key, value in data.items()
        if key not in {"states", "delta_encoded"}
    }
    top_hash = hashlib.sha256(json_bytes(compact)).hexdigest()
    state_hash = hashlib.sha256()
    dropped = Counter()
    output = []
    previous = {}
    fractional = 0
    sprite_count = 0
    # Delta sources can share large unchanged arrays; cache projections by their
    # live identity. Objects remain owned by data until this function returns.
    projected_lists = {}
    for frame, sprites in frames(data):
        projected = {}
        for key, values in sprites.items():
            identity = id(values)
            cached = projected_lists.get(identity)
            if cached is None:
                cached = [
                    {k: v for k, v in sprite.items() if k not in SPRITE_OMIT}
                    for sprite in values
                ]
                # Cache only delta input arrays. Full-state files have distinct
                # arrays per frame; caching those defeats bounded memory.
                if data.get("delta_encoded"):
                    projected_lists[identity] = cached
            projected[key] = cached
            sprite_count += len(values)
            for sprite in values:
                fractional += any(
                    isinstance(sprite.get(axis), (int, float)) and sprite[axis] % 1 != 0
                    for axis in ("col", "row")
                )
        # Counts describe fields physically present in the source JSON, not
        # repetition introduced by delta expansion.
        for values in frame.get("sprites", {}).values():
            for sprite in values or []:
                dropped.update("sprite." + key for key in SPRITE_OMIT if key in sprite)
        dropped.update("frame." + key for key in FRAME_OMIT if key in frame)
        retained = {
            key: value
            for key, value in frame.items()
            if key not in FRAME_OMIT and key != "sprites"
        }
        update_digest(state_hash, dict(retained, sprites=projected))
        if not output:
            retained["sprites"] = projected
        else:
            delta = {
                key: value
                for key, value in projected.items()
                if key not in previous or value != previous[key]
            }
            delta.update((key, None) for key in previous if key not in projected)
            if delta:
                retained["sprites"] = delta
        output.append(retained)
        previous = projected
    compact["states"] = output
    compact["delta_encoded"] = True
    return compact, {
        "top_level_sha256": top_hash,
        "retained_frames_sha256": state_hash.hexdigest(),
        "frames": len(output),
        "steps": len(data.get("steps", [])),
        "expanded_sprite_records": sprite_count,
        "fractional_sprite_records": fractional,
        "omitted_stored_fields": dict(sorted(dropped.items())),
    }


def verify_replay(data, evidence):
    """Verify serialized output independently from the projection/encoding loop."""
    top = {
        key: value
        for key, value in data.items()
        if key not in {"states", "delta_encoded"}
    }
    if hashlib.sha256(json_bytes(top)).hexdigest() != evidence["top_level_sha256"]:
        raise ValueError("Top-level content or conversation changed")
    digest = hashlib.sha256()
    count = 0
    for frame, sprites in frames(data):
        if FRAME_OMIT.intersection(frame):
            raise ValueError("Omitted raw input field remains")
        if any(
            SPRITE_OMIT.intersection(sprite)
            for values in sprites.values()
            for sprite in values
        ):
            raise ValueError("Omitted engine field remains")
        update_digest(digest, dict(frame, sprites=sprites))
        count += 1
    if (
        count != evidence["frames"]
        or digest.hexdigest() != evidence["retained_frames_sha256"]
    ):
        raise ValueError("A retained frame field changed during serialization")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(data) + b"\n")


def generated_row(relative, path, source):
    digest = sha256_file(path)
    return {
        "schema_version": 2,
        "release_id": "reason-to-play-neurips2026",
        "release_path": relative,
        "artifact_id": "sha256:" + digest,
        "artifact_id_scheme": "payload-sha256",
        "component": "website_assets",
        "decision": "include",
        "tier": "website",
        "license": "MIT",
        "published": False,
        "source": {
            "kind": "generated",
            "verification": "local-bytes-verified",
            "generator_path": "scripts/release/build_website_replays.py",
            "generator_sha256": sha256_file(__file__),
            **source,
        },
        "source_uri": None,
        "payload": {
            "sha256": digest,
            "size_bytes": Path(path).stat().st_size,
            "validation": "bytes-verified",
        },
        "selection_rule": "catalogue-browser-copy",
        "selection_reason": "Compact browser playback asset; complete research recording remains at its canonical behavior path",
    }


def build_one(arguments):
    root, output_root, row = arguments
    started = time.monotonic()
    relative = row["release_path"]
    source = safe_path(root, relative)
    expected = row["payload"]
    if (
        source.stat().st_size != expected["size_bytes"]
        or sha256_file(source) != expected["sha256"]
    ):
        raise ValueError(f"Source checksum mismatch: {relative}")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    compact, evidence = compact_replay(data)
    del data
    output_relative = "website-assets/replays/" + relative.removeprefix("behavior/")
    target = safe_path(output_root, output_relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
            ) as compressed:
                # iterencode avoids building a second huge JSON string in memory.
                encoder = json.JSONEncoder(
                    ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                for chunk in encoder.iterencode(compact):
                    compressed.write(chunk.encode("utf-8"))
        del compact
        with gzip.open(temporary, "rt", encoding="utf-8") as stream:
            verify_replay(json.load(stream), evidence)
        if sha256_file(source) != expected["sha256"]:
            raise ValueError(f"Source changed while building: {relative}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    source_pin = {
        "canonical_release_path": relative,
        "canonical_artifact_id": row["artifact_id"],
        "canonical_sha256": expected["sha256"],
    }
    addition = generated_row(output_relative, target, source_pin)
    evidence.update(source_pin)
    evidence.update(
        {
            "release_path": output_relative,
            "source_size_bytes": expected["size_bytes"],
            "size_bytes": addition["payload"]["size_bytes"],
            "sha256": addition["payload"]["sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    )
    return addition, evidence


def build_website_assets(root, output, *, workers=32, select=None):
    """Create website copies selected by the dataset's current catalogue index."""
    root, output = Path(root), Path(output)
    dataset = output / "dataset"
    if dataset.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must be outside the source dataset")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    index_file = root / INDEX_PATH
    index_sha = sha256_file(index_file)
    release_file = root / "manifest.jsonl.gz"
    release_sha = sha256_file(release_file)
    index = json.loads(index_file.read_text())
    wanted = {
        canonical_path(path)
        for group in index.values()
        for entry in group.values()
        for path in entry["replays"].values()
    }
    if select is not None:
        if set(select) - wanted:
            raise ValueError("Selected path is absent from the catalogue index")
        wanted &= set(select)
    if not wanted:
        raise ValueError("Catalogue selection is empty")
    rows = {}
    with gzip.open(release_file, "rt") as stream:
        for row in map(json.loads, stream):
            if row["release_path"] in wanted:
                if row["release_path"] in rows:
                    raise ValueError("Duplicate release path in source manifest")
                rows[row["release_path"]] = row
    if rows.keys() != wanted:
        raise ValueError("Catalogue references an absent manifest path")
    output.mkdir(parents=True, exist_ok=True)
    arguments = [(str(root), str(dataset), rows[path]) for path in sorted(rows)]
    started = time.monotonic()
    additions, evidence = [], []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        futures = [pool.submit(build_one, argument) for argument in arguments]
        for future in as_completed(futures):
            row, result = future.result()
            additions.append(row)
            evidence.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(additions),
                        "total": len(rows),
                        "release_path": row["release_path"],
                        "source_bytes": result["source_size_bytes"],
                        "output_bytes": result["size_bytes"],
                        "peak_rss_kib": result["worker_peak_rss_kib"],
                    }
                ),
                flush=True,
            )
    additions.sort(key=lambda row: row["release_path"])
    evidence.sort(key=lambda row: row["release_path"])
    if sha256_file(index_file) != index_sha or sha256_file(release_file) != release_sha:
        raise ValueError("Input manifest changed while building")
    if select is None:
        for group in index.values():
            for entry in group.values():
                entry["replays"] = {
                    game: "website-assets/replays/"
                    + canonical_path(path).removeprefix("behavior/")
                    for game, path in entry["replays"].items()
                }
        write_json(dataset / INDEX_PATH, index)
        index_row = generated_row(
            INDEX_PATH,
            dataset / INDEX_PATH,
            {
                "canonical_release_path": INDEX_PATH,
                "canonical_sha256": index_sha,
                "canonical_manifest_sha256": release_sha,
            },
        )
        write_json(output / "index-row.json", index_row)
    with (output / "additions.jsonl").open("wb") as stream:
        for row in additions:
            stream.write(json_bytes(row) + b"\n")
    with (output / "source-map.jsonl").open("wb") as stream:
        for row in evidence:
            stream.write(json_bytes(row) + b"\n")
    omitted = Counter()
    for result in evidence:
        omitted.update(result["omitted_stored_fields"])
    report = {
        "files": len(additions),
        "workers": workers,
        "source_manifest_sha256": release_sha,
        "source_index_sha256": index_sha,
        "generator_sha256": sha256_file(__file__),
        "source_size_bytes": sum(row["source_size_bytes"] for row in evidence),
        "size_bytes": sum(row["size_bytes"] for row in evidence),
        "frames": sum(row["frames"] for row in evidence),
        "steps": sum(row["steps"] for row in evidence),
        "expanded_sprite_records": sum(
            row["expanded_sprite_records"] for row in evidence
        ),
        "fractional_sprite_records": sum(
            row["fractional_sprite_records"] for row in evidence
        ),
        "largest_worker_peak_rss_kib": max(
            row["worker_peak_rss_kib"] for row in evidence
        ),
        "conservative_32_worker_rss_bytes": 32
        * 1024
        * max(row["worker_peak_rss_kib"] for row in evidence),
        "omitted_stored_fields": dict(sorted(omitted.items())),
        "verification": "Every retained frame field and all top-level/conversation fields match after serialized output is reloaded and deltas expanded",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--select",
        action="append",
        help="Build selected paths only; omit index replacement",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_website_assets(
                args.dataset_root, args.output, workers=args.workers, select=args.select
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
