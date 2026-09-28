"""Sample EfficientZero traces at the scanner samples in a released BOLD archive.

Human JSON supplies original play IDs, frame clocks and scanner start times.
The existing base archive fixes the retained samples; it is never rewritten.
The default selects four representation and seven initial value/policy hooks,
matching the archived job configuration. This does not establish the mapping to
numbered layers in archived result tables. ``--include-dynamics`` additionally
samples the four sparse MCTS dynamics/reward hooks at their recorded timesteps.

Example::

    python -m analysis.neural.align_efficientzero \\
        --base-data dataset/analysis/neural/inputs/sub-13/bold-ddqn-theory.npz \\
        --behavior-dir dataset/behavior/human \\
        --trace-dir dataset/features/efficientzero \\
        --output efficientzero.npz --workers 32
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import re
import tempfile
import types

import numpy as np

from analysis.neural.alignment import (
    SAMPLE_FIELDS,
    bind_to_base,
    file_sha256,
    validate_feature_coverage,
)
from data.game_ids import canonical_game_id
from data.values import decode_value
from human.behavior import validate_frame_clocks
from analysis.neural.prepare_inputs import (
    EZ_DYNAMICS_REWARD_LAYERS,
    EZ_LAYERS,
    EZ_REPRESENTATION_LAYERS,
    EZ_VALUE_POLICY_LAYERS,
    align_features_to_trs,
    get_play_timing,
)

STUDY_LAYERS = tuple(EZ_REPRESENTATION_LAYERS + EZ_VALUE_POLICY_LAYERS)
ALL_LAYERS = tuple(EZ_LAYERS)


class StoredObjectId:
    """Decode an archived 12-byte identity without loading its database library."""

    def __setstate__(self, value):
        if not isinstance(value, bytes) or len(value) != 12:
            raise ValueError("Invalid stored play identity")
        self.value = value

    def __str__(self):
        return self.value.hex()


def _legacy_bytes(value, encoding="latin1"):
    if encoding not in ("latin1", "latin-1"):
        raise ValueError("Unsupported archived byte encoding")
    return value.encode(encoding)


class _TraceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("bson.objectid", "ObjectId"):
            return StoredObjectId
        if (module, name) == ("_codecs", "encode"):
            return _legacy_bytes
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch._utils" and name in {
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
        }:
            import torch

            return getattr(torch._utils, name)
        if module == "torch" and name in {
            "BFloat16Storage",
            "HalfStorage",
            "FloatStorage",
            "DoubleStorage",
            "LongStorage",
            "IntStorage",
            "ShortStorage",
            "CharStorage",
            "ByteStorage",
            "BoolStorage",
        }:
            import torch

            return getattr(torch, name)
        raise ValueError(f"Unsupported archived trace global: {module}.{name}")


def load_trace_document(path):
    """Read a trusted released PyTorch trace, including inert legacy ID values.

    Only tensor reconstruction, ordered mappings and inert byte IDs are allowed
    in archived pickle metadata. No database package, checkpoint, or engine is
    required.
    """
    import torch

    reader = types.ModuleType("efficientzero_trace_pickle")
    reader.Unpickler = _TraceUnpickler
    reader.load = lambda stream, **kw: _TraceUnpickler(stream, **kw).load()
    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=reader
    )


@dataclass
class PlayAlignment:
    play_id: str
    subject: str
    run_id: int
    game: str
    level: int
    document_index: int | None
    sample_start: int
    sample_stop: int
    timing: dict
    play_index: int | None = None


@dataclass
class AlignmentContext:
    base_path: Path
    base: dict
    binding: dict
    plays: dict[str, PlayAlignment]


def _human_clocks(root, subject):
    """Read one compressed game at a time without expanding its sprite snapshots."""
    from human.behavior import read_record, recording_path, replay_paths

    paths = replay_paths(root, subject=subject)
    if not paths:
        raise FileNotFoundError(f"No elaborate human replay files for {subject}")
    seen_ids, seen_ordinals = set(), set()
    for path in paths:
        record = read_record(path, expand=False)
        if record["subject"] != subject:
            raise ValueError(f"Human recording names a different subject: {path}")
        states = record["states"]
        if record["total_frames"] != len(states):
            raise ValueError(f"Human recording frame count differs: {path}")
        previous_end = 0
        for stored in record["plays"]:
            start, count = stored["state_start"], stored["state_count"]
            if (
                type(start) is not int
                or type(count) is not int
                or count < 1
                or start != previous_end
                or start + count > len(states)
            ):
                raise ValueError(f"Invalid human play frame bounds: {path}")
            previous_end = start + count
            play = {
                key: decode_value(stored[key])
                for key in (
                    "_id",
                    "subj_id",
                    "run_id",
                    "game_name",
                    "level_id",
                    "scanner",
                )
            }
            if "play_id" in stored:
                play["play_id"] = stored["play_id"]
            pid = play["_id"]
            ordinal = stored["source_document_index"]
            ordinal_key = int(play["subj_id"]), int(play["run_id"]), ordinal
            if (
                not isinstance(pid, str)
                or not pid
                or type(ordinal) is not int
                or ordinal < 0
                or pid in seen_ids
                or ordinal_key in seen_ordinals
            ):
                raise ValueError(f"Invalid or repeated human play identity: {path}")
            if int(play["subj_id"]) != int(subject.removeprefix("sub-")):
                raise ValueError(f"Human play/recording subject differs: {path}")
            if canonical_game_id(play["game_name"]) != canonical_game_id(
                record["game"]
            ):
                raise ValueError(f"Human play/recording game differs: {path}")
            seen_ids.add(pid)
            seen_ordinals.add(ordinal_key)
            frames = states[start : start + count]
            if any(frame["source_play_id"] != pid for frame in frames):
                raise ValueError(f"Human frame/play identity differs: {path}")
            play["states"] = [
                {"gt": frame["time"], "ts": frame["realworld_ts"]} for frame in frames
            ]
            validate_frame_clocks(play["states"], pid)
            play["_canonical"] = {
                "source_document_index": ordinal,
                "source_recording": recording_path(record),
            }
            yield play
        if previous_end != len(states):
            raise ValueError(f"Unassigned human frames: {path}")


def prepare_alignment(base_path, behavior_dir=None, *, plays=None, runs=None):
    """Verify base identities and reconstruct frame-to-sample maps once per subject.

    ``plays`` and ``runs`` may supply already decoded human records. Otherwise
    ``behavior_dir`` supplies the self-contained JSON files. Only small identity
    arrays are loaded from the BOLD archive, never its voxel/feature arrays.
    """
    base_path = Path(base_path)
    with np.load(base_path, allow_pickle=False) as source:
        base = {key: source[key] for key in SAMPLE_FIELDS}
        if "n_volumes" in source:
            base["n_volumes"] = source["n_volumes"]
    subject = str(base["subject"])
    subject_number = int(subject.removeprefix("sub-"))
    if plays is None:
        if behavior_dir is None:
            raise ValueError("Provide human behavior_dir or decoded plays and runs")
        plays = _human_clocks(behavior_dir, subject)
    play_documents = {}
    for play in plays.values() if isinstance(plays, dict) else plays:
        pid = str(play["_id"])
        if pid in play_documents:
            raise ValueError(f"Duplicate original play identity: {pid}")
        play_documents[pid] = play
    if runs is None:
        runs = {}
        for play in play_documents.values():
            scanner = play.get("scanner")
            if not isinstance(scanner, dict):
                raise ValueError("Human recording is missing its scanner clock")
            key = int(play["subj_id"]), int(play["run_id"])
            if key in runs and runs[key] != scanner:
                raise ValueError(f"Human recordings disagree on scanner run {key}")
            runs[key] = scanner
    play_ids = [str(pid) for pid in base["play_ids"]]
    if not play_ids or len(play_ids) != len(set(play_ids)):
        raise ValueError("Base must contain unique retained play identities")
    n_samples = len(base["tr_play_idx"])
    boundaries = base["play_boundaries"]
    if (
        boundaries.shape != (len(play_ids) + 1,)
        or boundaries[0] != 0
        or boundaries[-1] != n_samples
        or not np.array_equal(np.diff(boundaries), base["play_n_volumes"])
        or np.any(np.diff(boundaries) <= 0)
        or ("n_volumes" in base and int(base["n_volumes"]) != n_samples)
    ):
        raise ValueError("Base play boundaries disagree with retained sample counts")
    for field in ("tr_run_idx", "tr_game_idx", "tr_level_idx"):
        if base[field].shape != (n_samples,):
            raise ValueError(f"Base {field} has the wrong sample count")
    for field in ("play_n_volumes", "play_game_idx", "play_levels"):
        if base[field].shape != (len(play_ids),):
            raise ValueError(f"Base {field} has the wrong play count")
    tr = float(base["tr"])
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("Base TR must be positive and finite")
    ar1 = bool(base["ar1_corrected"])
    starts, run_ends = {}, {}
    for index, pid in enumerate(play_ids):
        if pid not in play_documents:
            raise ValueError(f"Retained play {pid} is absent from human recordings")
        play = play_documents[pid]
        run = int(play["run_id"])
        if int(play["subj_id"]) != subject_number:
            raise ValueError(f"Human subject differs from base for play {pid}")
        game_index = int(base["play_game_idx"][index])
        if not 0 <= game_index < len(base["game_names"]):
            raise ValueError(f"Base game index is invalid for play {pid}")
        game = str(base["game_names"][game_index])
        if canonical_game_id(play["game_name"]) != canonical_game_id(game):
            raise ValueError(f"Human game differs from base for play {pid}")
        if int(play["level_id"]) != int(base["play_levels"][index]):
            raise ValueError(f"Human level differs from base for play {pid}")
        start, stop = map(int, boundaries[index : index + 2])
        for field, expected in (
            ("tr_run_idx", run),
            ("tr_game_idx", game_index),
            ("tr_level_idx", int(play["level_id"])),
            ("tr_play_idx", index),
        ):
            if not np.all(base[field][start:stop] == expected):
                raise ValueError(f"Base {field} disagrees with human play {pid}")
        scanner = runs.get((subject_number, run))
        if scanner is None:
            raise ValueError(f"Scanner clock missing for {subject}, run {run}")
        if (
            int(scanner["subj_id"]) != subject_number
            or int(scanner["run_id"]) != run
            or not np.isfinite(scanner["scan_start_ts"])
        ):
            raise ValueError(f"Scanner identity/clock disagrees for play {pid}")
        validate_frame_clocks(play["states"], pid)
        clocks = np.asarray([state["ts"] for state in play["states"]])
        if np.any(np.diff(clocks) < 0):
            raise ValueError(f"Nonmonotonic recorded frame timestamps for play {pid}")
        timing = get_play_timing(play, scanner, tr, 10**9, ar1)
        count = stop - start
        if timing is None or count > timing["n_volumes"]:
            raise ValueError(f"Retained sample count exceeds recorded timing for {pid}")
        starts[pid] = int(timing["volume_offset"])
        run_ends[run] = max(run_ends.get(run, 0), starts[pid] + count)
    mapped = {}
    for index, pid in enumerate(play_ids):
        play = play_documents[pid]
        run = int(play["run_id"])
        timing = get_play_timing(
            play, runs[(subject_number, run)], tr, run_ends[run], ar1
        )
        start, stop = map(int, boundaries[index : index + 2])
        if (
            timing is None
            or timing["n_volumes"] != stop - start
            or timing["volume_offset"] != starts[pid]
        ):
            raise ValueError(
                f"Recorded frame timing changes retained samples for {pid}"
            )
        mapped[pid] = PlayAlignment(
            pid,
            subject,
            run,
            str(base["game_names"][int(base["play_game_idx"][index])]),
            int(base["play_levels"][index]),
            play.get("_canonical", {}).get("source_document_index"),
            start,
            stop,
            timing,
            int(play["play_id"]) if "play_id" in play else None,
        )
    return AlignmentContext(base_path, base, bind_to_base(base_path, base), mapped)


def _identity(value):
    if isinstance(value, dict) and set(value) == {"stored_object_id_hex"}:
        value = value["stored_object_id_hex"]
    result = str(value)
    if not re.fullmatch(r"[a-fA-F0-9]{24}", result):
        raise ValueError(f"Invalid original trace play identity: {result!r}")
    return result.lower()


def validate_trace_identity(metadata, play):
    """Check measured identity inside the trace; directory names are not evidence."""
    if _identity(metadata.get("play_key")) != play.play_id.lower():
        raise ValueError(f"Trace original play identity differs for {play.play_id}")
    if int(metadata["subj_id"]) != int(play.subject.removeprefix("sub-")):
        raise ValueError(f"Trace subject differs for play {play.play_id}")
    if int(metadata["run_id"]) != play.run_id:
        raise ValueError(f"Trace run differs for play {play.play_id}")
    if canonical_game_id(metadata["game_name"]) != canonical_game_id(play.game):
        raise ValueError(f"Trace game differs for play {play.play_id}")
    if play.play_index is not None and int(metadata["play_id"]) != play.play_index:
        raise ValueError(f"Trace per-level play index differs for {play.play_id}")
    if "level_id" in metadata and int(metadata["level_id"]) != play.level:
        raise ValueError(f"Trace level differs for play {play.play_id}")
    if metadata.get("trace_human_actions_only", False):
        raise ValueError("Action-only traces cannot represent every recorded frame")


def _activation(entry):
    value = entry["activation"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError("EfficientZero activation is empty or nonfinite")
    return result


def align_trace(document, play, *, layers=STUDY_LAYERS, method="average"):
    """Return aligned hook arrays and reasons for unavailable hooks for one play.

    Representation/initial value-policy rows must enumerate every original frame
    in order. Missing hooks or incomplete frame coverage return an explicit gap;
    contradictory identities, duplicate/out-of-range timesteps and invalid
    values raise. Sparse dynamics/reward calls are averaged per recorded frame
    before scanner sampling, retaining zeroes on frames without an MCTS call.
    """
    if method not in ("average", "last"):
        raise ValueError("Alignment method must be average or last")
    layers = tuple(layers)
    if not layers or len(layers) != len(set(layers)) or set(layers) - set(ALL_LAYERS):
        raise ValueError("Select unique known EfficientZero hooks")
    validate_trace_identity(document["metadata"], play)
    traces = document["traces"]
    n_frames = len(play.timing["state_timestamps"])
    aligned, missing = {}, {}
    for layer in layers:
        entries = traces.get(layer, [])
        if layer in EZ_VALUE_POLICY_LAYERS:
            entries = [entry for entry in entries if entry.get("phase") == "initial"]
        if not entries:
            missing[layer] = "hook-absent-or-no-initial-calls"
            continue
        timesteps = []
        for entry in entries:
            timestep = entry.get("timestep")
            if (
                isinstance(timestep, (bool, np.bool_))
                or not isinstance(timestep, (int, np.integer))
                or not 0 <= timestep < n_frames
            ):
                raise ValueError(f"Invalid recorded frame index in {layer}")
            timesteps.append(int(timestep))
        if layer not in EZ_DYNAMICS_REWARD_LAYERS:
            if len(set(timesteps)) != len(timesteps):
                raise ValueError(f"Duplicate recorded frame index in {layer}")
            if timesteps != sorted(timesteps):
                raise ValueError(f"Unordered recorded frame indices in {layer}")
            if timesteps != list(range(n_frames)):
                missing[layer] = (
                    f"incomplete-frame-coverage:{len(timesteps)}/{n_frames}"
                )
                continue
            if any(entry.get("phase") != "initial" for entry in entries):
                raise ValueError(f"Non-initial representation frame in {layer}")
            activations = np.stack([_activation(entry) for entry in entries])
        else:
            # Keep the original float32, per-timestep mean (not a mean of means).
            grouped = {}
            for timestep, entry in zip(timesteps, entries, strict=True):
                grouped.setdefault(timestep, []).append(_activation(entry))
            n_features = len(next(iter(grouped.values()))[0])
            activations = np.zeros((n_frames, n_features), dtype=np.float32)
            for timestep, values in grouped.items():
                activations[timestep] = np.mean(np.stack(values), axis=0)
        aligned[layer] = align_features_to_trs(
            activations,
            play.timing["volume_indices"],
            play.timing["valid_mask"],
            play.sample_stop - play.sample_start,
            method,
        )
    return aligned, missing


def feature_key(layer):
    return f"ez_{layer.replace('.', '_')}_aligned"


def _coverage(context, missing_ids):
    unknown = set(missing_ids) - context.plays.keys()
    if unknown:
        raise ValueError(f"Unknown missing feature play identities: {sorted(unknown)}")
    intervals = [
        {
            "play_id": pid,
            "sample_start": play.sample_start,
            "sample_stop": play.sample_stop,
            "game": play.game,
            "level": play.level,
        }
        for pid, play in context.plays.items()
        if pid in missing_ids
    ]
    result = {
        "complete": not intervals,
        "retained_sample_count": len(context.base["tr_play_idx"]),
        "missing_feature_sample_count": sum(
            item["sample_stop"] - item["sample_start"] for item in intervals
        ),
        "missing_feature_play_ids": [item["play_id"] for item in intervals],
        "missing_feature_sample_intervals": intervals,
        "missing_feature_policy": "zero-fill-original-aligned-samples",
    }
    return validate_feature_coverage(result, context.base)


def write_aligned_features(
    output_path,
    context,
    arrays,
    missing_by_layer,
    *,
    source_records=None,
    method="average",
):
    """Write one feature archive and a byte-pinned coverage/binding sidecar.

    ``arrays`` maps original hook names to full float32 sample matrices (including
    memmaps). ``missing_by_layer`` lists absent play IDs for each hook. Arrays for
    missing plays must be zero; no sample is dropped or reordered.
    """
    output_path = Path(output_path)
    base_path = context.base_path.resolve()
    for candidate in (output_path, Path(str(output_path) + ".alignment.json")):
        if candidate.resolve() == base_path or (
            candidate.exists() and candidate.samefile(base_path)
        ):
            raise ValueError(
                "EfficientZero output must not overwrite the BOLD/base archive"
            )
    if method not in ("average", "last"):
        raise ValueError("Alignment method must be average or last")
    layers = list(arrays)
    if not layers or set(layers) - set(ALL_LAYERS):
        raise ValueError("Output requires named EfficientZero hook arrays")
    if set(missing_by_layer) != set(layers):
        raise ValueError("Declare missing-play coverage for every output hook")
    n_samples = len(context.base["tr_play_idx"])
    coverage_by_layer = {}
    for layer, array in arrays.items():
        if (
            array.ndim != 2
            or array.shape[0] != n_samples
            or array.shape[1] < 1
            or array.dtype != np.float32
        ):
            raise ValueError(f"Invalid aligned array shape/dtype for {layer}")
        # Bounded validation also works for subject-sized memory-mapped matrices.
        for first in range(0, n_samples, 16):
            if not np.all(np.isfinite(array[first : first + 16])):
                raise ValueError(f"Nonfinite aligned values for {layer}")
        coverage = _coverage(context, set(missing_by_layer[layer]))
        for interval in coverage["missing_feature_sample_intervals"]:
            if np.any(array[interval["sample_start"] : interval["sample_stop"]]):
                raise ValueError(f"Missing-feature interval contains values in {layer}")
        coverage_by_layer[feature_key(layer).removesuffix("_aligned")] = coverage
    if file_sha256(context.base_path) != str(context.binding["alignment_base_sha256"]):
        raise ValueError("Base archive changed during feature alignment")
    aggregate = _coverage(
        context, {pid for missing in missing_by_layer.values() for pid in missing}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "reason-to-play/alignment-binding",
        "schema_version": 1,
        "base_sha256": str(context.binding["alignment_base_sha256"]),
        "sample_order_sha256": str(context.binding["alignment_samples_sha256"]),
        "verification": {
            "status": "verified",
            "method": "original-play-identities-and-recorded-frame-clocks",
            "retained_samples": n_samples,
        },
        "feature_coverage": aggregate,
        "feature_coverage_by_layer": coverage_by_layer,
        "layers": layers,
        "hook_to_array": {layer: feature_key(layer) for layer in layers},
        "sampling": {
            "method": method,
            "scanner_volume": "round((recorded_timestamp - scanner_start) / TR)",
            "ar1_volume_offset": -1 if bool(context.base["ar1_corrected"]) else 0,
            "empty_sample_policy": "zero",
            "dynamics_reward_frame_policy": "mean-recorded-calls-per-frame-zero-when-absent",
        },
    }
    if source_records is not None:
        document["sources"] = source_records
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".npz", delete=False
    ) as handle:
        np.savez_compressed(
            handle,
            **{feature_key(layer): value for layer, value in arrays.items()},
            **context.binding,
            efficientzero_layer_names=np.asarray(layers),
            efficientzero_alignment_method=np.array(method),
        )
        handle.flush()
        document["feature_sha256"] = file_sha256(handle.name)
        Path(handle.name).replace(output_path)
    association_path = Path(str(output_path) + ".alignment.json")
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output_path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(association_path)
    return document


def discover_traces(root):
    """Find paths containing exact play IDs, independent of subject folder names."""
    result = {}
    for path in sorted(Path(root).rglob("traces.pt")):
        match = re.fullmatch(r"play-(\w{24})|play\d+_key(\w{24})", path.parent.name)
        if match is None:
            continue
        pid = _identity(match[1] or match[2])
        if pid in result:
            raise ValueError(f"Multiple traces for original play {pid}")
        result[pid] = path
    return result


def _align_file(job):
    path, play, layers, method = job
    arrays, missing = align_trace(
        load_trace_document(path), play, layers=layers, method=method
    )
    return play.play_id, arrays, missing, file_sha256(path)


def _bounded_results(pool, jobs, workers):
    """Keep at most one in-flight trace per worker, including completed results."""
    jobs = iter(jobs)
    pending = set()
    for _ in range(workers):
        job = next(jobs, None)
        if job is not None:
            pending.add(pool.submit(_align_file, job))
    while pending:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            yield future.result()
            job = next(jobs, None)
            if job is not None:
                pending.add(pool.submit(_align_file, job))


def _worker_init():
    import torch

    torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--method", choices=("average", "last"), default="average")
    parser.add_argument("--include-dynamics", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    context = prepare_alignment(args.base_data, args.behavior_dir)
    paths = discover_traces(args.trace_dir)
    layers = ALL_LAYERS if args.include_dynamics else STUDY_LAYERS
    absent = set(context.plays) - paths.keys()
    missing = {layer: set(absent) for layer in layers}
    arrays, sources, reasons = {}, [], {}
    jobs = [
        (paths[pid], play, layers, args.method)
        for pid, play in context.plays.items()
        if pid in paths
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init
    ) as pool:
        for pid, values, gaps, digest in _bounded_results(pool, jobs, args.workers):
            play = context.plays[pid]
            sources.append({"play_id": pid, "sha256": digest})
            reasons[pid] = gaps
            for layer in layers:
                if layer in gaps:
                    missing[layer].add(pid)
                    continue
                values_for_layer = values[layer]
                if layer not in arrays:
                    arrays[layer] = np.zeros(
                        (len(context.base["tr_play_idx"]), values_for_layer.shape[1]),
                        dtype=np.float32,
                    )
                if arrays[layer].shape[1] != values_for_layer.shape[1]:
                    raise ValueError(f"Feature dimension changes across plays: {layer}")
                arrays[layer][play.sample_start : play.sample_stop] = values_for_layer
    if set(arrays) != set(layers):
        raise ValueError(f"No usable features for hooks: {set(layers) - set(arrays)}")
    document = write_aligned_features(
        args.output,
        context,
        {layer: arrays[layer] for layer in layers},
        missing,
        source_records=sorted(sources, key=lambda item: item["play_id"]),
        method=args.method,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "coverage": document["feature_coverage"],
                "missing_hook_reasons": {
                    pid: gaps for pid, gaps in reasons.items() if gaps
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
