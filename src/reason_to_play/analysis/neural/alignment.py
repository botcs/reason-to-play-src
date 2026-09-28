"""Identify the exact BOLD archive and ordered scanner samples for features."""

from __future__ import annotations

import hashlib
import json
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
    "alignment_base_sha256",
    "alignment_samples_sha256",
)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_order_sha256(data):
    """Hash recorded sample identities and timing conventions, never feature values."""
    missing = [key for key in SAMPLE_FIELDS if key not in data]
    if missing:
        raise ValueError(f"Base archive lacks sample identity fields: {missing}")
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


def bind_to_base(path, data):
    """Metadata for a sidecar generated against this exact base archive."""
    return {
        "alignment_binding_version": np.array(1),
        "alignment_base_sha256": np.array(file_sha256(path)),
        "alignment_samples_sha256": np.array(sample_order_sha256(data)),
    }


def validate_binding(sidecar, base, base_path, *, base_sha256=None):
    """Reject incomplete or unequal bindings; return False for unbound archives."""
    present = [key for key in BINDING_FIELDS if key in sidecar]
    if not present:
        return False
    if len(present) != len(BINDING_FIELDS):
        raise ValueError("Incomplete feature-to-base alignment binding")
    if any(np.asarray(sidecar[key]).shape != () for key in BINDING_FIELDS):
        raise ValueError("Alignment binding fields must be scalars")
    if int(sidecar["alignment_binding_version"]) != 1:
        raise ValueError("Unsupported feature-to-base alignment binding version")
    expected = base_sha256 or file_sha256(base_path)
    if str(sidecar["alignment_base_sha256"]) != expected:
        raise ValueError("Feature sidecar belongs to a different base archive")
    if str(sidecar["alignment_samples_sha256"]) != sample_order_sha256(base):
        raise ValueError("Feature sidecar has a different scanner sample order")
    return True


def external_binding(path):
    """Read a byte-pinned association established by checking an archived payload."""
    document_path = Path(str(path) + ".alignment.json")
    if not document_path.is_file():
        return None
    document = json.loads(document_path.read_text())
    if (document.get("schema"), document.get("schema_version")) != (
        "reason-to-play/alignment-binding",
        1,
    ):
        raise ValueError(f"Unsupported alignment association: {document_path}")
    if document.get("verification", {}).get("status") != "verified":
        raise ValueError(f"Alignment association is not verified: {document_path}")
    if document.get("feature_sha256") != file_sha256(path):
        raise ValueError(f"Alignment association names different feature bytes: {path}")
    return {
        "alignment_binding_version": 1,
        "alignment_base_sha256": document["base_sha256"],
        "alignment_samples_sha256": document["sample_order_sha256"],
    }


def released_llm_path(layer, subject):
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
        / condition
        / selection
        / stream
        / f"{subject}.npz"
    )
