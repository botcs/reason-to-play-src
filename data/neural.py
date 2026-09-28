"""Read and write participant BOLD, scanner samples, nuisance and model features."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np


SAMPLE_FIELDS = (
    "subject",
    "tr",
    "ar1_corrected",
    "game_names",
    "play_ids",
    "play_boundaries",
    "play_n_volumes",
    "play_game_idx",
    "play_levels",
    "tr_run_idx",
    "tr_game_idx",
    "tr_level_idx",
    "tr_play_idx",
)
BINDING_FIELDS = (
    "alignment_binding_version",
    "alignment_bold_sha256",
    "alignment_samples_sha256",
    "alignment_sample_order_sha256",
)
BOLD_FIELDS = ("voxel_ts", "mask", "mask_affine", "n_voxels")
SAMPLE_ARCHIVE_FIELDS = SAMPLE_FIELDS + (
    "game_boundaries",
    "game_n_levels",
    "game_n_volumes",
    "n_games",
    "n_plays",
    "n_volumes",
    "play_partitions",
    "max_level",
    "states_lost_to_ar1",
    "total_empty_volumes",
    "mean_states_per_volume",
    "scanner_volume_index",
)
NUISANCE_FIELDS = (
    "keystates",
    "keystate_columns",
    "any_keypress",
    "scores",
    "score_deltas",
    "time_in_play",
    "time_in_experiment",
)
DDQN_FIELDS = (
    "conv1_aligned",
    "conv2_aligned",
    "fc1_aligned",
    "q_values_aligned",
    "ddqn_layers",
    "ddqn_layer_dims",
    "conv_layer_names",
    "conv_channels",
    "conv_target_sizes",
    "alignment_method",
    "max_timestamp_diff_ms",
    "mean_timestamp_diff_ms",
    "play_has_ddqn",
)
HRR_FIELDS = (
    "hrr_aligned",
    "hrr_dim",
    "hrr_seed",
    "hrr_matched_plays",
    "hrr_missing_plays",
)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_order_sha256(data):
    """Hash recorded sample identities and timing conventions, never feature values."""
    missing = [key for key in SAMPLE_FIELDS if key not in data]
    if missing:
        raise ValueError(f"Samples archive lacks sample identity fields: {missing}")
    digest = hashlib.sha256(b"reason-to-play/scanner-samples/v1\n")
    for key in SAMPLE_FIELDS:
        array = np.asarray(data[key])
        if array.dtype.hasobject:
            raise ValueError(f"Object-valued sample identity field: {key}")
        header = json.dumps([key, array.dtype.str, array.shape], separators=(",", ":"))
        value = np.ascontiguousarray(array).tobytes()
        digest.update(len(header.encode()).to_bytes(8, "little"))
        digest.update(header.encode())
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    return digest.hexdigest()


def validate_feature_coverage(coverage, base):
    """Check declared missing plays in the original, unfiltered base sample order."""
    if not isinstance(coverage, dict):
        raise ValueError("Feature coverage must be an object")
    if coverage.get("status") == "unknown":
        if (
            type(coverage.get("retained_sample_count")) is not int
            or coverage["retained_sample_count"] != len(base["tr_play_idx"])
            or not isinstance(coverage.get("reason"), str)
            or not coverage["reason"].strip()
        ):
            raise ValueError(
                "Unknown coverage requires exact sample count and a reason"
            )
        if set(coverage) != {"status", "retained_sample_count", "reason"}:
            raise ValueError(
                "Unknown coverage must not claim completeness or missing intervals"
            )
        return coverage
    required = {
        "complete",
        "retained_sample_count",
        "missing_feature_sample_count",
        "missing_feature_play_ids",
        "missing_feature_sample_intervals",
    }
    if not required.issubset(coverage):
        raise ValueError("Feature coverage lacks required counts or intervals")
    for key in ("retained_sample_count", "missing_feature_sample_count"):
        if type(coverage[key]) is not int or coverage[key] < 0:
            raise ValueError(f"Feature coverage {key} must be a nonnegative integer")
    if type(coverage["complete"]) is not bool:
        raise ValueError("Feature coverage complete must be a boolean")
    if coverage["retained_sample_count"] != len(base["tr_play_idx"]):
        raise ValueError("Feature coverage sample count differs from the base archive")
    ids = coverage["missing_feature_play_ids"]
    intervals = coverage["missing_feature_sample_intervals"]
    absent_games = coverage.get("absent_game_files", [])
    if (
        not isinstance(absent_games, list)
        or any(not isinstance(game, str) or not game for game in absent_games)
        or len(absent_games) != len(set(absent_games))
    ):
        raise ValueError(
            "Feature coverage absent_game_files must list unique game names"
        )
    if (
        not isinstance(ids, list)
        or any(not isinstance(pid, str) for pid in ids)
        or len(ids) != len(set(ids))
        or not isinstance(intervals, list)
    ):
        raise ValueError(
            "Feature coverage requires unique play IDs and an interval list"
        )
    play_indices = {str(pid): index for index, pid in enumerate(base["play_ids"])}
    seen = set()
    missing_count = 0
    for interval in intervals:
        if not isinstance(interval, dict):
            raise ValueError("Feature coverage interval must be an object")
        pid = interval.get("play_id")
        if not isinstance(pid, str) or pid not in play_indices or pid in seen:
            raise ValueError(
                "Feature coverage interval has an unknown or duplicate play"
            )
        index = play_indices[pid]
        start, stop = interval.get("sample_start"), interval.get("sample_stop")
        if (
            type(start) is not int
            or type(stop) is not int
            or start != int(base["play_boundaries"][index])
            or stop != int(base["play_boundaries"][index + 1])
        ):
            raise ValueError(
                "Feature coverage interval disagrees with base play boundaries"
            )
        if "game" in interval and interval["game"] != str(
            base["game_names"][int(base["play_game_idx"][index])]
        ):
            raise ValueError("Feature coverage interval names a different game")
        if "level" in interval and (
            type(interval["level"]) is not int
            or interval["level"] != int(base["play_levels"][index])
        ):
            raise ValueError("Feature coverage interval names a different level")
        seen.add(pid)
        missing_count += stop - start
    if (
        seen != set(ids)
        or coverage["missing_feature_sample_count"] != missing_count
        or coverage["complete"] != (not ids and not absent_games)
    ):
        raise ValueError("Feature coverage completeness, play IDs and counts disagree")
    return coverage


def released_llm_path(layer):
    """Resolve study model labels to the documented dataset path; unknowns are explicit."""
    import re

    match = re.fullmatch(
        r"llm_(.+)__(all|compressed)__(main|attn|mlp)_layer_\d+", layer
    )
    if match is None:
        return None
    model, selection, stream = match.groups()
    condition = "elaborate"
    for suffix, name in {
        "_sugmin": "minimal",
        "_sugorc": "oracle",
        "_random": "elaborate-random-init",
        "_abl01": "elaborate-context-fraction-0.1",
        "_abl05": "elaborate-context-fraction-0.5",
    }.items():
        if model.endswith(suffix):
            model, condition = model.removesuffix(suffix), name
            break
    model = {
        "qwen35_9b": "qwen3.5-9b",
        "qwen35_27b": "qwen3.5-27b",
        "qwen35_35b_a3b": "qwen3.5-35b-a3b",
        "qwen35_122b_a10b": "qwen3.5-122b-a10b",
        "dsv32": "deepseek-v3.2",
        "dsv4_flash": "deepseek-v4-flash",
        "dsv4_pro": "deepseek-v4-pro",
    }.get(model)
    if model is None:
        return None
    return (
        Path("model-features")
        / "lrm"
        / model
        / f"{condition}--{selection}--{stream}.npz"
    )


def _read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return dict(source)


def _association_path(path):
    return Path(str(path) + ".alignment.json")


def _read_association(path):
    document_path = _association_path(path)
    if not document_path.is_file():
        raise ValueError(f"Download {document_path.name} alongside {Path(path).name}")
    document = json.loads(document_path.read_text())
    if document.get("verification", {}).get("status") != "verified":
        raise ValueError(f"Alignment association is not verified: {document_path}")
    return document


def _write_json(path, document):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_npz(path, arrays):
    for name, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"Object-valued arrays are not supported: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    temporary.replace(path)


def _digests(subject_dir, samples):
    root = Path(subject_dir)
    return {
        "bold_sha256": file_sha256(root / "bold.npz"),
        "samples_sha256": file_sha256(root / "samples.npz"),
        "sample_order_sha256": sample_order_sha256(samples),
    }


def validate_samples(samples):
    """Check that participant identities partition the recorded scanner rows."""
    sample_order_sha256(samples)
    count = len(samples["tr_play_idx"])
    boundaries = np.asarray(samples["play_boundaries"])
    if (
        boundaries.ndim != 1
        or len(boundaries) < 2
        or not np.issubdtype(boundaries.dtype, np.integer)
        or boundaries[0] != 0
        or boundaries[-1] != count
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("Sample play boundaries must partition every row")
    lengths = np.diff(boundaries)
    ids = list(map(str, samples["play_ids"]))
    if len(ids) != len(lengths) or len(ids) != len(set(ids)):
        raise ValueError("Sample play IDs must be unique and match play boundaries")
    if not np.array_equal(samples["play_n_volumes"], lengths):
        raise ValueError("Sample play lengths disagree with play boundaries")
    if not np.array_equal(
        samples["tr_play_idx"], np.repeat(np.arange(len(ids)), lengths)
    ):
        raise ValueError("Sample play indices disagree with play boundaries")
    for row_key, play_key in (
        ("tr_game_idx", "play_game_idx"),
        ("tr_level_idx", "play_levels"),
    ):
        if len(samples[play_key]) != len(ids) or not np.array_equal(
            samples[row_key], np.repeat(samples[play_key], lengths)
        ):
            raise ValueError(f"Sample {row_key} disagrees with per-play identities")
    for key in (
        "tr_play_idx",
        "tr_game_idx",
        "tr_level_idx",
        "tr_run_idx",
        "scanner_volume_index",
    ):
        if key not in samples:
            continue
        array = np.asarray(samples[key])
        if (
            array.shape != (count,)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0)
        ):
            raise ValueError(f"Sample {key} requires one nonnegative integer per row")
    if np.any(np.asarray(samples["tr_game_idx"]) >= len(samples["game_names"])):
        raise ValueError("Sample game index exceeds game names")
    if "n_volumes" in samples and int(samples["n_volumes"]) != count:
        raise ValueError("Sample n_volumes disagrees with recorded rows")


def load_samples(subject_dir):
    """Read only scanner identities, after checking their association with BOLD."""
    root = Path(subject_dir)
    samples = _read_npz(root / "samples.npz")
    validate_samples(samples)
    document = _read_association(root / "bold.npz")
    if (document.get("schema"), document.get("schema_version")) != (
        "reason-to-play/bold-samples",
        1,
    ):
        raise ValueError("Unsupported BOLD/sample association")
    for key, digest in _digests(root, samples).items():
        if document.get(key) != digest:
            raise ValueError(f"BOLD/sample association has different {key}")
    return samples


def bind_to_samples(subject_dir, samples=None):
    """Pin exact participant BOLD bytes, sample bytes and ordered identities."""
    if samples is None:
        samples = load_samples(subject_dir)
    digests = _digests(subject_dir, samples)
    return {
        "alignment_binding_version": np.array(2),
        **{f"alignment_{key}": np.array(value) for key, value in digests.items()},
    }


def validate_binding(binding, samples, subject_dir, *, digests=None):
    """Reject mismatched embedded anchors; absence is allowed only alongside JSON."""
    if "alignment_base_sha256" in binding:
        raise ValueError("Feature archive contains an obsolete base binding")
    present = [key for key in BINDING_FIELDS if key in binding]
    if not present:
        return False
    if len(present) != len(BINDING_FIELDS):
        raise ValueError("Incomplete feature-to-samples alignment binding")
    if any(np.asarray(binding[key]).shape != () for key in BINDING_FIELDS):
        raise ValueError("Alignment binding fields must be scalars")
    if int(binding["alignment_binding_version"]) != 2:
        raise ValueError("Unsupported alignment binding version")
    for key, digest in (digests or _digests(subject_dir, samples)).items():
        if str(binding[f"alignment_{key}"]) != digest:
            raise ValueError(f"Feature archive has different {key}")
    return True


def external_binding(path):
    """Read the mandatory byte-pinned feature or nuisance association."""
    document = _read_association(path)
    if (document.get("schema"), document.get("schema_version")) != (
        "reason-to-play/alignment-binding",
        2,
    ):
        raise ValueError(f"Unsupported alignment association: {path}")
    if document.get("feature_sha256") != file_sha256(path):
        raise ValueError(f"Alignment association names different feature bytes: {path}")
    if "feature_coverage" not in document:
        raise ValueError("Feature association lacks coverage evidence")
    return document


def coverage_from_missing(samples, play_ids):
    """Describe source availability in the unchanged participant sample order."""
    missing = set(map(str, play_ids))
    if missing - set(map(str, samples["play_ids"])):
        raise ValueError("Missing-feature coverage names an unknown play")
    intervals = []
    for index, pid in enumerate(samples["play_ids"]):
        if str(pid) in missing:
            intervals.append(
                {
                    "play_id": str(pid),
                    "sample_start": int(samples["play_boundaries"][index]),
                    "sample_stop": int(samples["play_boundaries"][index + 1]),
                }
            )
    coverage = {
        "complete": not missing,
        "retained_sample_count": len(samples["tr_play_idx"]),
        "missing_feature_sample_count": sum(
            i["sample_stop"] - i["sample_start"] for i in intervals
        ),
        "missing_feature_play_ids": [i["play_id"] for i in intervals],
        "missing_feature_sample_intervals": intervals,
    }
    return validate_feature_coverage(coverage, samples)


def unknown_coverage(samples, reason):
    return validate_feature_coverage(
        {
            "status": "unknown",
            "retained_sample_count": len(samples["tr_play_idx"]),
            "reason": reason,
        },
        samples,
    )


def _check_feature_arrays(arrays, samples):
    count = len(samples["tr_play_idx"])
    for name, value in arrays.items():
        if name.endswith("_aligned"):
            value = np.asarray(value)
            if value.ndim != 2 or value.shape[0] != count or value.shape[1] == 0:
                raise ValueError(f"Feature {name} has incompatible sample dimensions")


def _check_coverage(coverage, by_layer, arrays, samples, layer=None):
    selected = validate_feature_coverage(coverage, samples)
    if by_layer is not None:
        if not isinstance(by_layer, dict) or not by_layer:
            raise ValueError("Per-layer feature coverage must be a mapping")
        for name, value in by_layer.items():
            if not isinstance(name, str) or f"{name}_aligned" not in arrays:
                raise ValueError("Per-layer coverage names an absent feature")
            validate_feature_coverage(value, samples)
        if layer is not None:
            if layer not in by_layer:
                raise ValueError("Requested layer lacks declared feature coverage")
            selected = by_layer[layer]
    elif "efficientzero_layer_names" in arrays:
        raise ValueError(
            "EfficientZero requires feature_coverage_by_layer in its .npz.alignment.json"
        )
    return selected


def write_core_inputs(subject_dir, bold, samples, nuisance, *, nuisance_coverage=None):
    """Write separate participant measurements, identities and nuisance arrays."""
    root = Path(subject_dir)
    validate_samples(samples)
    if set(bold) - set(BOLD_FIELDS) or set(nuisance) - set(NUISANCE_FIELDS):
        raise ValueError("Unexpected array ownership in BOLD or nuisance inputs")
    if set(samples) - set(SAMPLE_ARCHIVE_FIELDS):
        raise ValueError("Unexpected array ownership in scanner samples")
    _write_npz(root / "samples.npz", samples)
    _write_npz(root / "bold.npz", bold)
    _write_json(
        _association_path(root / "bold.npz"),
        {
            "schema": "reason-to-play/bold-samples",
            "schema_version": 1,
            **_digests(root, samples),
            "verification": {"status": "verified"},
        },
    )
    write_feature_archive(
        root / "nuisance.npz",
        nuisance,
        root,
        nuisance_coverage
        if nuisance_coverage is not None
        else unknown_coverage(samples, "Source coverage was not recorded"),
    )


def write_feature_archive(
    path, arrays, subject_dir, coverage, *, coverage_by_layer=None, metadata=None
):
    """Write sampled features and their exact BOLD/sample and coverage association."""
    path, root = Path(path), Path(subject_dir)
    protected_inputs = [
        root / "bold.npz",
        root / "samples.npz",
        _association_path(root / "bold.npz"),
    ]
    nuisance_output = path.absolute() == (root / "nuisance.npz").absolute() and not (
        set(arrays) - set(NUISANCE_FIELDS) - set(BINDING_FIELDS)
    )
    if not nuisance_output:
        protected_inputs.extend(
            (root / "nuisance.npz", _association_path(root / "nuisance.npz"))
        )
    for protected in protected_inputs:
        for candidate in (path, _association_path(path)):
            if candidate.resolve() == protected.resolve() or (
                candidate.exists()
                and protected.exists()
                and candidate.samefile(protected)
            ):
                raise ValueError(
                    "Feature output must not overwrite participant inputs or associations"
                )
    samples = load_samples(root)
    _check_feature_arrays(arrays, samples)
    _check_coverage(coverage, coverage_by_layer, arrays, samples)
    binding = bind_to_samples(root, samples)
    if any(key in arrays for key in (*BINDING_FIELDS, "alignment_base_sha256")):
        validate_binding(arrays, samples, root)
    payload = {**arrays, **binding}
    document = {
        "schema": "reason-to-play/alignment-binding",
        "schema_version": 2,
        **_digests(root, samples),
        "verification": {"status": "verified"},
        "feature_coverage": coverage,
    }
    if coverage_by_layer is not None:
        document["feature_coverage_by_layer"] = coverage_by_layer
    if metadata:
        forbidden = set(metadata) & (set(document) - {"verification"})
        if forbidden or "feature_sha256" in metadata:
            raise ValueError("Feature metadata cannot replace binding or coverage")
        if metadata.get("verification", {}).get("status", "verified") != "verified":
            raise ValueError("Feature verification must be verified")
        document.update(metadata)
    _write_npz(path, payload)
    document["feature_sha256"] = file_sha256(path)
    _write_json(_association_path(path), document)
    return document


def feature_path(subject_dir, layer):
    root = Path(subject_dir)
    if layer in ("conv1", "conv2", "fc1", "q_values"):
        relative = Path("model-features/ddqn.npz")
    elif layer == "hrr":
        relative = Path("model-features/empa/hrr.npz")
    elif layer.startswith("hrr_"):
        relative = Path("model-features/empa/hrr-decomposed.npz")
    elif layer.startswith("ez_") or layer == "ez":
        relative = Path("model-features/efficientzero.npz")
    else:
        relative = released_llm_path(layer)
    if relative is None:
        raise ValueError(f"Provide an explicit feature file for layer {layer}")
    return root / relative


def input_paths(subject_dir, feature_file, layer):
    root = Path(subject_dir)
    paths = [root / name for name in ("bold.npz", "samples.npz", "nuisance.npz")]
    paths.append(
        Path(feature_file) if feature_file is not None else feature_path(root, layer)
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_inputs(subject_dir, feature_file, layer):
    """Load four explicit participant inputs after checking identities and coverage."""
    paths = input_paths(subject_dir, feature_file, layer)
    samples = load_samples(subject_dir)
    digests = _digests(subject_dir, samples)
    data = _read_npz(paths[0])
    if set(data) - set(BOLD_FIELDS):
        raise ValueError("BOLD archive contains arrays belonging to another input")
    if data["voxel_ts"].shape[1] != len(samples["tr_play_idx"]):
        raise ValueError("BOLD and scanner sample counts differ")
    data.update(samples)
    statuses = []
    for path, role in zip(paths[2:], ("nuisance", "feature"), strict=True):
        with np.load(path, allow_pickle=False) as source:
            array_names = set(source.files)
            arrays = {
                key: source[key]
                for key in source.files
                if not key.endswith("_aligned") or key == f"{layer}_aligned"
            }
        association = external_binding(path)
        for key, digest in digests.items():
            if association.get(key) != digest:
                raise ValueError(f"{role} archive has different {key}")
        validate_binding(arrays, samples, subject_dir, digests=digests)
        _check_feature_arrays(arrays, samples)
        selected = _check_coverage(
            association["feature_coverage"],
            association.get("feature_coverage_by_layer"),
            array_names,
            samples,
            layer if role == "feature" else None,
        )
        if role == "nuisance":
            if array_names - set(NUISANCE_FIELDS) - set(BINDING_FIELDS):
                raise ValueError(
                    "Nuisance archive contains arrays belonging to another input"
                )
            for key in set(arrays) - {"keystate_columns"} - set(BINDING_FIELDS):
                if np.asarray(arrays[key]).shape[0] != len(samples["tr_play_idx"]):
                    raise ValueError(f"Nuisance {key} has incompatible sample count")
        for key, value in arrays.items():
            if key in BINDING_FIELDS:
                continue
            if key in data and not np.array_equal(data[key], value):
                raise ValueError(
                    f"Feature archive conflicts with participant field {key}"
                )
            data[key] = value
        statuses.append(
            {
                "path": str(path),
                "status": "verified",
                "feature_coverage": selected,
                "coverage_sample_order": "samples.npz",
            }
        )
    if f"{layer}_aligned" not in data:
        raise ValueError(f"Layer '{layer}' not found in requested feature file")
    data["alignment_verification_json"] = json.dumps(statuses, sort_keys=True)
    return data


def read_feature_binding(path, subject_dir, samples=None):
    """Validate a feature association without loading its activation matrices."""
    if samples is None:
        samples = load_samples(subject_dir)
    digests = _digests(subject_dir, samples)
    document = external_binding(path)
    for key, digest in digests.items():
        if document.get(key) != digest:
            raise ValueError(f"Feature archive has different {key}")
    with np.load(path, allow_pickle=False) as source:
        binding = {key: source[key] for key in BINDING_FIELDS if key in source}
        if "alignment_base_sha256" in source:
            raise ValueError("Feature archive contains an obsolete base binding")
        validate_binding(binding, samples, subject_dir, digests=digests)
        _check_coverage(
            document["feature_coverage"],
            document.get("feature_coverage_by_layer"),
            source,
            samples,
        )
    return document
