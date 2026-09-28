#!/usr/bin/env python3
"""
Align LLM features to an existing processed BOLD/base archive.

The base supplies BOLD timing, AR(1) status, retained volumes and play order.
Canonical human recordings supply original play identities, all frame timestamps
and scanner start times. Output files contain one aligned LLM feature source
each. PyTorch is required only for .pt feature inputs. Human measurements are
read from the self-contained JSON dataset.

Usage:
    python -m reason_to_play.fmri.align_llm \\
        --subject sub-13 \\
        --aligned-data ./dataset/analysis-inputs/sub-13/bold-ddqn-theory.npz \\
        --behavior-dir ./dataset/behavior/human \\
        --llm-source "name=dsv3,dir=/path/to/llm_dsv3" \\
        --output-dir /path/to/out
"""

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from reason_to_play.analysis.neural.alignment import bind_to_base, validate_binding


def _torch():
    """Load the optional tensor reader only when a .pt input is requested."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Reading .pt model features requires PyTorch; install a suitable "
            "PyTorch build or reason-to-play[features]."
        ) from exc
    return torch


def _behavior_api():
    from reason_to_play.data import behavior

    return behavior


def play_states(play_doc):
    """Read original measured frames from a human JSON record."""
    return _behavior_api().play_states(play_doc)


def load_canonical_plays(root, subject):
    result = {}
    for play in _behavior_api().iter_plays(root, subject=subject):
        key = str(play["_id"])
        if key in result:
            raise ValueError(f"Duplicate original play ID: {key}")
        result[key] = play
    if not result:
        raise ValueError(f"No canonical plays for {subject} in {root}")
    return result


# =============================================================================
# Configuration — identical to original
# =============================================================================

LLM_LAYERS = list(range(1, 65))
LLM_LAYER_NAMES = [f"layer_{i}" for i in LLM_LAYERS]

# Multi-turn .pt files store three activation streams as separate keys.
# `stream` argument across this module ('main' | 'attn' | 'mlp') resolves
# to the corresponding key here.
MULTITURN_STREAM_KEYS = {
    "main": "features",
    "attn": "features_attn",
    "mlp": "features_mlp",
}

GAME_ORDER = [
    "vgfmri4_avoidgeorge",
    "vgfmri4_bait",
    "vgfmri4_chase",
    "vgfmri4_helper",
    "vgfmri4_lemmings",
    "vgfmri4_zelda",
]


# =============================================================================
# Game name normalization for multi-turn format
# =============================================================================
# Per-level feature directories: vgfmri4_avoidgeorge/ etc. (lowercase, prefix=vgfmri4)
# Per-game feature files: avoidGeorge_vgfmri4.pt  (camelCase, suffix=vgfmri4)
# This maps between them so analysis joins use the vgfmri-prefixed names.

_CANONICAL_TO_MULTITURN = {
    "vgfmri4_avoidgeorge": "avoidGeorge_vgfmri4",
    "vgfmri4_bait": "bait_vgfmri4",
    "vgfmri4_chase": "chase_vgfmri4",
    "vgfmri4_helper": "helper_vgfmri4",
    "vgfmri4_lemmings": "lemmings_vgfmri4",
    "vgfmri4_zelda": "zelda_vgfmri4",
    # vgfmri3 variants (in case they come up later)
    "vgfmri3_avoidgeorge": "avoidGeorge_vgfmri3",
    "vgfmri3_bait": "bait_vgfmri3",
    "vgfmri3_chase": "chase_vgfmri3",
    "vgfmri3_helper": "helper_vgfmri3",
    "vgfmri3_lemmings": "lemmings_vgfmri3",
    "vgfmri3_zelda": "zelda_vgfmri3",
    "vgfmri3_plaqueattack": "plaqueAttack_vgfmri3",
}


def _multiturn_file_for(llm_dir: Path, subject: str, game: str) -> Optional[Path]:
    """Return path to the multiturn per-game .pt if it exists, else None."""
    mt_name = _CANONICAL_TO_MULTITURN.get(game.lower())
    if mt_name is None:
        return None
    p = llm_dir / subject / f"{mt_name}.pt"
    return p if p.exists() else None


# =============================================================================
# LLM Source Configuration — verbatim from original
# =============================================================================


@dataclass
class LLMSourceConfig:
    """
    Configuration for a single LLM feature source.

    Attributes:
        name: Unique identifier for this source.
        directory: Path to the directory containing LLM features for this source.
        subsample: Subsampling factor applied BEFORE alignment.
        layers: List of layer indices to extract. If None, uses auto-discovery.
        stream: Which activation stream to extract from multi-turn .pt files.
                One of {'main', 'attn', 'mlp'}. Maps to the .pt keys
                'features', 'features_attn', 'features_mlp' respectively.
                Ignored for per-level feature sources.
    """

    name: str
    directory: Path
    subsample: int = 1
    layers: List[int] = None
    stream: str = "main"

    def __post_init__(self):
        if self.layers is not None:
            self.layer_names = [f"layer_{i}" for i in self.layers]
        else:
            self.layer_names = None
        if self.subsample < 1:
            raise ValueError(f"subsample must be >= 1, got {self.subsample}")
        if self.stream not in ("main", "attn", "mlp"):
            raise ValueError(
                f"stream must be one of 'main', 'attn', 'mlp'; got {self.stream!r}"
            )


# =============================================================================
# LLM Loading Functions — verbatim from original
# =============================================================================


def discover_llm_layers(llm_dir: Path, subject: str) -> List[int]:
    """Auto-discover available layer indices. Dispatches by format."""
    if is_multiturn_source(llm_dir, subject):
        return discover_multiturn_layers(llm_dir, subject)
    return _discover_llm_layers_legacy(llm_dir, subject)


def _discover_llm_layers_legacy(llm_dir: Path, subject: str) -> List[int]:
    """Auto-discover available layer indices by peeking at one data file."""
    subj_dir = llm_dir / subject
    if not subj_dir.exists():
        return []

    for game_dir in sorted(subj_dir.iterdir()):
        if not game_dir.is_dir() or not game_dir.name.startswith("vgfmri"):
            continue

        level_files = sorted(game_dir.glob("level_*.pt"))
        if not level_files:
            level_files = sorted(game_dir.glob("level_*.npz"))
        if not level_files:
            continue

        try:
            f = level_files[0]
            if f.suffix == ".pt":
                data = _torch().load(f, map_location="cpu", weights_only=False)
                keys = list(data.keys())
            else:
                data = np.load(f, allow_pickle=True)
                keys = list(data.files)

            layer_indices = set()
            for key in keys:
                match = re.match(r"play_\d+_(?:model_)?layer_(\d+)(?:_post)?$", key)
                if match:
                    layer_indices.add(int(match.group(1)))

            if layer_indices:
                result = sorted(layer_indices)
                logging.info(
                    f"    Auto-discovered {len(result)} layers from {f.name}: "
                    f"{result[0]}-{result[-1]}"
                )
                return result
        except Exception as e:
            logging.warning(f"    Failed to discover layers from {f}: {e}")
            continue

    return []


def load_llm_features_for_level(
    llm_dir: Path,
    subject: str,
    game: str,
    level: int,
    layers: list = None,
    stream: str = "main",
) -> dict:
    """Load LLM features for a specific game/level.

    Dispatches between per-game and per-level feature loaders by directory layout.
    `stream` selects which activation tensor to extract from multi-turn .pt files
    ('main' | 'attn' | 'mlp'); ignored for per-level feature sources.

    Returns dict keyed by local play idx.
    """
    if is_multiturn_source(llm_dir, subject):
        return load_multiturn_features_for_level(
            llm_dir, subject, game, level, layers=layers, stream=stream
        )
    if stream != "main":
        logging.warning(
            f"    Per-level LLM source at {llm_dir} does not support stream={stream!r}; "
            f"loading default features."
        )
    return _load_llm_features_for_level_legacy(
        llm_dir, subject, game, level, layers=layers
    )


def _load_llm_features_for_level_legacy(
    llm_dir: Path, subject: str, game: str, level: int, layers: list = None
) -> dict:
    """Load LLM features for a specific game/level. Returns dict keyed by local play idx."""
    if layers is None:
        layers = LLM_LAYER_NAMES

    level_file_pt = llm_dir / subject / game / f"level_{level:02d}.pt"
    level_file_npz = llm_dir / subject / game / f"level_{level:02d}.npz"

    level_file = None
    is_torch = False

    if level_file_pt.exists():
        level_file = level_file_pt
        is_torch = True
    elif level_file_npz.exists():
        level_file = level_file_npz
        is_torch = False
    else:
        return {}

    torch = _torch() if is_torch else None
    try:
        if is_torch:
            data = torch.load(level_file, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                converted = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.bfloat16:
                            v = v.to(torch.float32)
                        converted[k] = v.numpy()
                    else:
                        converted[k] = v
                data = converted
        else:
            data = dict(np.load(level_file, allow_pickle=True))
    except Exception as e:
        logging.warning(f"Failed to load LLM features from {level_file}: {e}")
        return {}

    play_indices_found = set()
    for key in data.keys():
        if key.startswith("play_") and "_behavioral_timestamps" in key:
            try:
                idx = int(key.split("_")[1])
                play_indices_found.add(idx)
            except (ValueError, IndexError):
                pass

    if not play_indices_found:
        for key in data.keys():
            match = re.match(r"play_(\d+)_(?:model_)?layer_\d+", key)
            if match:
                play_indices_found.add(int(match.group(1)))

    if not play_indices_found:
        return {}

    sorted_indices = sorted(play_indices_found)
    llm_by_local_index = {}

    for local_idx, global_idx in enumerate(sorted_indices):
        activations = {}
        for layer_name in layers:
            key = f"play_{global_idx}_model_{layer_name}"
            if key not in data:
                key = f"play_{global_idx}_{layer_name}"
            if key in data:
                act = data[key]
                if torch is not None and isinstance(act, torch.Tensor):
                    if act.dtype == torch.bfloat16:
                        act = act.to(torch.float32)
                    act = act.numpy()
                activations[layer_name] = act.astype(np.float32)

        if not activations:
            continue

        timestamps_key = f"play_{global_idx}_behavioral_timestamps"
        if timestamps_key not in data:
            continue
        timestamps = data[timestamps_key]
        if torch is not None and isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.numpy()

        num_states_key = f"play_{global_idx}_behavioral_num_states"
        num_states = int(data.get(num_states_key, len(timestamps)))

        llm_by_local_index[local_idx] = {
            "activations": activations,
            "timestamps": timestamps,
            "metadata": {
                "num_states": num_states,
                "local_index": local_idx,
                "global_index": global_idx,
            },
        }

    return llm_by_local_index


# =============================================================================
# Multi-turn format loader
# =============================================================================
# Per-game features store one .pt per game with:
#   features       : (n_states, n_layers, hidden_dim)  bfloat16
#   features_attn  : same shape
#   features_mlp   : same shape
#   metadata       : list of n_states dicts, each with
#                    {play_id, step_num, level_id, attempt, trial_idx,
#                     realworld_ts, ...}
#
# We pick `features` (residual stream) to match the old pipeline's single-tensor
# convention. Per-play slicing is driven by `metadata[i].play_id`.
#
# Cache: the .pt files are large (hundreds of MB to a few GB) so we cache the
# fully-loaded dict per (subject, game) and reuse across level calls.


_MULTITURN_CACHE: Dict[Tuple[str, str, str], dict] = {}
_MULTITURN_LAYERS_CACHE: Dict[Path, List[int]] = {}


def _load_multiturn_game_file(pt_path: Path, subject: str, game: str) -> dict:
    """Load (and cache) one game's multiturn .pt file.

    Cache key includes the resolved file path so that different feature
    *variants* of the same (subject, game) — e.g. `llm/` vs `llm_compressed/`
    — do not collide. Without the path component the second variant would
    silently receive cached data from the first.
    """
    key = (str(pt_path.resolve()), subject, game.lower())
    if key in _MULTITURN_CACHE:
        return _MULTITURN_CACHE[key]
    logging.info(
        f"    Loading multiturn file {pt_path.name} ({pt_path.stat().st_size / 1024**3:.2f} GB) ..."
    )
    data = _torch().load(pt_path, map_location="cpu", weights_only=False)
    _MULTITURN_CACHE[key] = data
    return data


def discover_multiturn_layers(llm_dir: Path, subject: str) -> List[int]:
    """Look at first available game file and read session['num_layers'] (or features.shape[1])."""
    subj_dir = llm_dir / subject
    if not subj_dir.exists():
        return []
    for f in sorted(subj_dir.glob("*_vgfmri*.pt")):
        if f in _MULTITURN_LAYERS_CACHE:
            return _MULTITURN_LAYERS_CACHE[f]
        try:
            # Just peek the session dict to get num_layers without loading tensors.
            # torch.load always loads the whole file, but we keep it in cache so
            # the main loader doesn't pay again.
            data = _torch().load(f, map_location="cpu", weights_only=False)
            sess = data.get("session", {})
            n_layers = sess.get("num_layers")
            if n_layers is None and "features" in data:
                n_layers = int(data["features"].shape[1])
            if n_layers is None:
                continue
            result = list(range(1, n_layers + 1))
            _MULTITURN_LAYERS_CACHE[f] = result
            # Also cache the whole file so the upcoming load_for_level doesn't re-read
            # Infer game key from filename for cache insertion
            return result
        except Exception as e:
            logging.warning(f"    multiturn discover failed on {f}: {e}")
            continue
    return []


def load_multiturn_features_for_level(
    llm_dir: Path,
    subject: str,
    game: str,
    level: int,
    layers: list = None,
    stream: str = "main",
) -> dict:
    """Multi-turn equivalent of load_llm_features_for_level.

    Returns dict keyed by local play index (0, 1, 2, ...) within the level,
    using the shared feature-loader return structure.

    `stream` selects which activation tensor to extract from the .pt:
        'main' -> 'features'        (residual stream output, default)
        'attn' -> 'features_attn'   (attention block output)
        'mlp'  -> 'features_mlp'    (MLP block output)
    """
    if stream not in MULTITURN_STREAM_KEYS:
        raise ValueError(
            f"Unknown stream {stream!r}; expected one of {list(MULTITURN_STREAM_KEYS)}"
        )
    feature_key = MULTITURN_STREAM_KEYS[stream]

    pt_path = _multiturn_file_for(llm_dir, subject, game)
    if pt_path is None:
        return {}

    torch = _torch()
    data = _load_multiturn_game_file(pt_path, subject, game)
    if feature_key not in data or "metadata" not in data:
        return {}

    feats = data[feature_key]  # (n_states, n_layers, hidden_dim)
    md = data["metadata"]
    n_layers_file = int(feats.shape[1])
    if len(md) != int(feats.shape[0]):
        raise ValueError("Feature rows and metadata rows differ")

    # Determine which layers to return. `layers` is passed as e.g. ['layer_1', ..., 'layer_32'].
    if layers is None:
        layers_idx_1based = list(range(1, n_layers_file + 1))
    else:
        layers_idx_1based = []
        for ln in layers:
            m = re.match(r"layer_(\d+)$", ln)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= n_layers_file:
                    layers_idx_1based.append(idx)

    # Group states by play_id, keeping their tensor positions.
    from collections import defaultdict

    states_by_play: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
    for state_pos, entry in enumerate(md):
        if int(entry["level_id"]) != int(level):
            continue
        states_by_play[entry["play_id"]].append((state_pos, entry))

    if not states_by_play:
        return {}

    # Order plays by trial_idx of their first state (stable, matches experiment order)
    ordered_play_ids = sorted(
        states_by_play.keys(),
        key=lambda pid: (
            int(states_by_play[pid][0][1]["trial_idx"]),
            states_by_play[pid][0][0],
        ),
    )

    llm_by_local_index: Dict[int, Dict] = {}

    for local_idx, play_id in enumerate(ordered_play_ids):
        records = states_by_play[play_id]
        # Sort states within a play by their position in the global metadata
        # (which is the order they were extracted — game step order).
        records.sort(key=lambda x: x[0])

        positions = [r[0] for r in records]
        entries = [r[1] for r in records]

        # Slice features: (n_states_this_play, n_layers, hidden_dim)
        # features is (n_total_states, n_layers, hidden_dim). Using a tensor index,
        # then extracting per-layer views.
        positions_tensor = torch.as_tensor(positions, dtype=torch.long)
        play_feats = feats.index_select(
            0, positions_tensor
        )  # (n_states, n_layers, hidden_dim)
        # Convert to float32 once
        if play_feats.dtype == torch.bfloat16:
            play_feats = play_feats.to(torch.float32)
        play_feats_np = play_feats.numpy()  # (n_states, n_layers, hidden_dim)

        activations = {}
        for idx_1based in layers_idx_1based:
            idx_0based = idx_1based - 1
            activations[f"layer_{idx_1based}"] = play_feats_np[:, idx_0based, :].astype(
                np.float32
            )

        timestamps = np.array(
            [float(e["realworld_ts"]) for e in entries], dtype=np.float64
        )

        llm_by_local_index[local_idx] = {
            "activations": activations,
            "timestamps": timestamps,
            "metadata": {
                "num_states": len(entries),
                "local_index": local_idx,
                "global_index": local_idx,  # no distinction in multiturn format
                "play_id": play_id,
                "trial_idx": int(entries[0].get("trial_idx", -1)),
                "source_references": [
                    {field: entry.get(field) for field in SOURCE_REFERENCE_FIELDS}
                    for entry in entries
                ],
            },
        }

    return llm_by_local_index


SOURCE_REFERENCE_FIELDS = (
    "source_play_id",
    "source_recording",
    "source_frame_index",
    "source_document_index",
)


def match_multiturn_plays(features_by_index, source_plays, *, slop_seconds=2.0):
    """Resolve feature groups before aligning; never select the first possible play.

    Linked features require exact original IDs and per-frame clocks. Historical
    groups named sub-XX_RUN_DOCUMENT use the recorded participant, run and
    original document ordinal, with exact frame-clock validation. Other groups
    retain their timestamp-range rule and must match one original play.
    Two groups cannot occupy the same play.
    Supply the complete subject/game/level inventory, including plays absent from
    BOLD or feature inputs, so a reduced cohort cannot hide ambiguity.
    """
    source_times = {
        str(pid): np.asarray(
            [frame["ts"] for frame in play_states(play)], dtype=np.float64
        )
        for pid, play in source_plays.items()
    }
    matched = {}
    for feature in features_by_index.values():
        times = np.asarray(feature["timestamps"], dtype=np.float64)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError(
                "Feature timestamps must be a finite one-dimensional array"
            )
        if not len(times):
            continue
        references = feature["metadata"].get("source_references")
        has_references = references is not None and any(
            any(ref.get(field) is not None for field in SOURCE_REFERENCE_FIELDS)
            for ref in references
        )
        # run_replay constructs this identifier before extraction; DOCUMENT is
        # the original run document ordinal, not the game-local attempt index.
        composite = re.fullmatch(
            r"sub-(\d{2})_(\d+)_(\d+)", str(feature["metadata"].get("play_id", ""))
        )
        composite_pid = None
        if composite is not None:
            subject, run, document = map(int, composite.groups())
            candidates = [
                pid
                for pid, play in source_plays.items()
                if int(play["subj_id"]) == subject
                and int(play["run_id"]) == run
                and play.get("_canonical", {}).get("source_document_index") == document
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Composite feature identity must name exactly one original "
                    "participant/run/document in the source game/level"
                )
            composite_pid = candidates[0]
            if not np.isin(times, source_times[composite_pid]).all():
                raise ValueError(
                    "Feature timestamps disagree with the composite original play identity"
                )
        if has_references:
            if len(references) != len(times) or any(
                any(ref.get(field) is None for field in SOURCE_REFERENCE_FIELDS)
                for ref in references
            ):
                raise ValueError(
                    "Linked features require complete source references on every row"
                )
            ids = {ref["source_play_id"] for ref in references}
            if len(ids) != 1:
                raise ValueError(
                    "One feature play contains conflicting original play IDs"
                )
            pid = next(iter(ids))
            if composite_pid is not None and composite_pid != pid:
                raise ValueError(
                    "Linked and composite original play identities disagree"
                )
            if pid not in source_plays:
                raise ValueError(
                    f"Linked feature play {pid!r} is absent from the source game/level"
                )
            source = source_plays[pid]
            ordinal = source.get("_canonical", {}).get("source_document_index")
            if ordinal is None:
                raise ValueError(
                    "Linked features require canonical human input with source document ordinals"
                )
            from reason_to_play.data.replay_behavior import (
                CONDITIONS,
                play_recording_path,
            )

            recording_path = Path(play_recording_path(source))
            # Prompt variants embed the same measured trajectory. Allow only
            # those filenames within this participant/game directory; original
            # play IDs, run ordinals and exact frame clocks are checked below.
            accepted_paths = {
                recording_path.with_name(f"{condition}.human.replay.json.gz").as_posix()
                for condition in CONDITIONS
            }
            frames = []
            for ref in references:
                frame = ref["source_frame_index"]
                document = ref["source_document_index"]
                if (
                    isinstance(frame, bool)
                    or not isinstance(frame, (int, np.integer))
                    or frame < 0
                    or frame >= len(source_times[pid])
                ):
                    raise ValueError(f"Invalid original frame index for play {pid}")
                if (
                    isinstance(document, bool)
                    or not isinstance(document, (int, np.integer))
                    or document != ordinal
                    or ref["source_recording"] not in accepted_paths
                ):
                    raise ValueError(
                        f"Source recording/document identity disagrees for play {pid}"
                    )
                frames.append(int(frame))
            if len(set(frames)) != len(frames):
                raise ValueError(f"Duplicate original frame references for play {pid}")
            if not np.array_equal(times, source_times[pid][frames]):
                raise ValueError(
                    f"Feature timestamps differ from referenced original frames for play {pid}"
                )
        elif composite_pid is not None:
            pid = composite_pid
        else:
            candidates = [
                pid
                for pid, original in source_times.items()
                if len(original)
                and times.min() >= original.min() - slop_seconds
                and times.max() <= original.max() + slop_seconds
            ]
            if len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous timestamp-only feature references match multiple original plays: {candidates}; "
                    "supply features with explicit source references"
                )
            if not candidates:
                logging.warning(
                    "Timestamp-only feature play %s has no source timestamp match",
                    feature["metadata"].get("play_id"),
                )
                continue
            pid = candidates[0]
        if pid in matched:
            raise ValueError(f"Multiple feature groups match original play {pid}")
        matched[pid] = feature
    return matched


def find_multiturn_games_and_levels(llm_dir: Path, subject: str) -> dict:
    """Multi-turn equivalent of find_llm_games_and_levels. Returns
    {canonical_game_name: sorted_levels}.
    """
    subj_dir = llm_dir / subject
    if not subj_dir.exists():
        return {}

    # Reverse map: multiturn filename stem -> canonical game name
    mt_to_canonical = {v: k for k, v in _CANONICAL_TO_MULTITURN.items()}

    games_levels = {}
    for pt_file in subj_dir.glob("*_vgfmri*.pt"):
        stem = pt_file.stem
        canonical = mt_to_canonical.get(stem)
        if canonical is None:
            continue
        # Load metadata to find which levels this file contains
        try:
            data = _load_multiturn_game_file(pt_file, subject, canonical)
            md = data.get("metadata", [])
            levels = sorted({int(m["level_id"]) for m in md})
            if levels:
                games_levels[canonical] = levels
        except Exception as e:
            logging.warning(f"    multiturn find failed on {pt_file}: {e}")
            continue

    return games_levels


def is_multiturn_source(llm_dir: Path, subject: str) -> bool:
    """Detect whether this source uses the new multiturn per-game format."""
    subj_dir = llm_dir / subject
    if not subj_dir.exists():
        return False
    # Multi-turn format: <game>_vgfmri*.pt directly in subject dir
    has_mt = any(subj_dir.glob("*_vgfmri*.pt"))
    # Per-level files: <game>/level_*.pt or <game>/level_*.npz
    has_legacy = any(
        d.is_dir() and (any(d.glob("level_*.pt")) or any(d.glob("level_*.npz")))
        for d in subj_dir.iterdir()
    )
    if has_mt and has_legacy:
        # Prefer multi-turn format if both exist (unusual)
        logging.warning(
            f"    Both per-game and per-level features detected for {subject}; using per-game files."
        )
    return has_mt


def find_llm_games_and_levels(llm_dir: Path, subject: str) -> dict:
    """Find all games and levels available for a subject in LLM features.

    Dispatches to per-game or per-level features based on directory layout.
    """
    if is_multiturn_source(llm_dir, subject):
        return find_multiturn_games_and_levels(llm_dir, subject)
    return _find_llm_games_and_levels_legacy(llm_dir, subject)


def _find_llm_games_and_levels_legacy(llm_dir: Path, subject: str) -> dict:
    """Find all games and levels available for a subject in LLM features."""
    if llm_dir is None or not llm_dir.exists():
        return {}

    subj_dir = llm_dir / subject
    if not subj_dir.exists():
        return {}

    games_levels = {}
    for game_dir in subj_dir.iterdir():
        if not game_dir.is_dir() or not game_dir.name.startswith("vgfmri"):
            continue

        levels = []
        for f in game_dir.glob("level_*.pt"):
            levels.append(int(f.stem.split("_")[1]))
        for f in game_dir.glob("level_*.npz"):
            level_num = int(f.stem.split("_")[1])
            if level_num not in levels:
                levels.append(level_num)

        if levels:
            games_levels[game_dir.name] = sorted(levels)

    return games_levels


# =============================================================================
# LLM Feature Manager — verbatim from original
# =============================================================================


class LLMFeatureManager:
    """Manages loading and preprocessing for multiple LLM feature sources."""

    def __init__(self, sources: List[LLMSourceConfig], subject: str):
        self.sources = {src.name: src for src in sources}
        self.subject = subject
        self.source_metadata = {}
        # Tracks per-play timestamp consistency with original human frames.
        # Populated for per-game features; per-level features use frame-index lookup.
        self.timestamp_audit = {"pass": 0, "fail": 0, "examples": []}
        for src in sources:
            self._initialize_source(src)

    def _initialize_source(self, src: LLMSourceConfig) -> None:
        if not src.directory.exists():
            logging.warning(
                f"LLM source '{src.name}' directory not found: {src.directory}"
            )
            return

        games_levels = find_llm_games_and_levels(src.directory, self.subject)
        if not games_levels:
            logging.warning(
                f"LLM source '{src.name}': No data found for {self.subject}"
            )
            return

        if src.layers is None or src.layer_names is None:
            discovered = discover_llm_layers(src.directory, self.subject)
            if discovered:
                src.layers = discovered
                src.layer_names = [f"layer_{i}" for i in src.layers]
                logging.info(
                    f"    LLM source '{src.name}': auto-discovered {len(discovered)} layers "
                    f"(layer_{discovered[0]} to layer_{discovered[-1]})"
                )
            else:
                logging.warning(
                    f"    LLM source '{src.name}': auto-discovery failed, "
                    f"falling back to global LLM_LAYERS default ({len(LLM_LAYERS)} layers)"
                )
                src.layers = list(LLM_LAYERS)
                src.layer_names = [f"layer_{i}" for i in src.layers]

        n_features_by_layer = {}
        for game, levels in games_levels.items():
            sample_data = load_llm_features_for_level(
                src.directory,
                self.subject,
                game,
                levels[0],
                src.layer_names,
                stream=src.stream,
            )
            if sample_data:
                first_play = list(sample_data.values())[0]
                for layer_name, activations in first_play["activations"].items():
                    n_features_by_layer[layer_name] = activations.shape[1]
                break

        self.source_metadata[src.name] = {
            "config": src,
            "games_levels": games_levels,
            "n_features_by_layer": n_features_by_layer,
            "layer_names": src.layer_names,
            "is_multiturn": is_multiturn_source(src.directory, self.subject),
        }

        logging.info(
            f"    LLM source '{src.name}': {len(games_levels)} games, "
            f"subsample={src.subsample}, stream={src.stream}, layers={src.layers}"
        )
        for layer_name, n_feat in n_features_by_layer.items():
            logging.info(f"      {layer_name}: {n_feat} features")

    def has_sources(self) -> bool:
        return len(self.source_metadata) > 0

    def load_for_level(
        self, game: str, level: int, *, subsample=True
    ) -> Dict[str, Dict[int, Dict]]:
        result = {}
        for src_name, meta in self.source_metadata.items():
            src = meta["config"]
            games_levels = meta["games_levels"]
            levels = games_levels.get(game, games_levels.get(game.lower(), ()))
            if level not in levels:
                result[src_name] = {}
                continue
            llm_by_local_index = load_llm_features_for_level(
                src.directory,
                self.subject,
                game,
                level,
                meta["layer_names"],
                stream=src.stream,
            )
            if subsample and src.subsample > 1 and llm_by_local_index:
                llm_by_local_index = self._apply_subsampling(
                    llm_by_local_index, src.subsample
                )
            result[src_name] = llm_by_local_index
        return result

    def _apply_subsampling(
        self, llm_by_local_index: Dict[int, Dict], subsample_factor: int
    ) -> Dict[int, Dict]:
        for local_idx in llm_by_local_index:
            play_data = llm_by_local_index[local_idx]
            n_states = len(play_data["timestamps"])
            subsample_mask = np.arange(n_states) % subsample_factor == 0
            play_data["timestamps"] = play_data["timestamps"][subsample_mask]
            references = play_data["metadata"].get("source_references")
            if references is not None:
                play_data["metadata"]["source_references"] = [
                    ref
                    for ref, keep in zip(references, subsample_mask, strict=True)
                    if keep
                ]
            play_data["metadata"]["num_states_original"] = n_states
            play_data["metadata"]["num_states"] = int(subsample_mask.sum())
            play_data["metadata"]["subsample_factor"] = subsample_factor
            for layer_name in list(play_data["activations"].keys()):
                play_data["activations"][layer_name] = play_data["activations"][
                    layer_name
                ][subsample_mask]
        return llm_by_local_index

    def get_all_storage_keys(self) -> List[str]:
        keys = []
        for src_name, meta in self.source_metadata.items():
            for layer_name in meta["layer_names"]:
                keys.append(f"{src_name}_{layer_name}")
        return keys

    def get_n_features(self, src_name: str, layer_name: str) -> int:
        if src_name not in self.source_metadata:
            return 1
        return self.source_metadata[src_name]["n_features_by_layer"].get(layer_name, 1)

    def get_source_names(self) -> List[str]:
        return list(self.source_metadata.keys())


# =============================================================================
# Measured human play readers
# =============================================================================


def get_play_timing(
    play_doc: dict,
    run_doc: dict,
    tr: float,
    n_volumes_whitened: int,
    ar1_corrected: bool = True,
) -> dict:
    """Compute timing info for a play: which TR each state maps to."""
    states = play_states(play_doc)
    state_timestamps = np.array([s.get("ts", 0) for s in states])
    scan_start_ts = run_doc["scan_start_ts"]
    onsets = state_timestamps - scan_start_ts

    volume_indices_original = np.round(onsets / tr).astype(int)

    if ar1_corrected:
        volume_indices = volume_indices_original - 1
        valid_mask = (volume_indices >= 0) & (volume_indices < n_volumes_whitened)
    else:
        volume_indices = volume_indices_original
        valid_mask = (volume_indices >= 0) & (volume_indices < n_volumes_whitened)

    valid_indices = volume_indices[valid_mask]
    if len(valid_indices) == 0:
        return None

    vol_min = valid_indices.min()
    vol_max = valid_indices.max()
    n_volumes = vol_max - vol_min + 1
    volume_indices_adjusted = volume_indices - vol_min

    return {
        "volume_indices": volume_indices_adjusted,
        "valid_mask": valid_mask,
        "n_volumes": n_volumes,
        "volume_offset": vol_min,
        "onsets": onsets,
        "scan_start_ts": scan_start_ts,
        "state_timestamps": state_timestamps,
        "n_states_at_tr0": int((volume_indices_original == 0).sum()),
    }


# =============================================================================
# Verification Functions — verbatim subset from original
# =============================================================================


def verify_state_temporal_order(timestamps: np.ndarray, play_id: str) -> bool:
    diffs = np.diff(timestamps)
    violations = np.where(diffs < 0)[0]
    if len(violations) > 0:
        logging.warning(
            f"Play {play_id}: {len(violations)} temporal order violation(s) detected"
        )
        for v in violations[:3]:
            logging.warning(
                f"  ts[{v}]={timestamps[v]:.3f} -> ts[{v + 1}]={timestamps[v + 1]:.3f} (diff={diffs[v]:.3f}s)"
            )
        return False
    return True


def verify_no_invalid_values(arrays: dict, context: str) -> None:
    for key, arr in arrays.items():
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError(f"{context}[{key}] contains NaN/Inf values")


# =============================================================================
# Alignment Function — verbatim from original
# =============================================================================


def align_llm_features_by_timestamp(
    activations: np.ndarray,
    llm_timestamps: np.ndarray,
    scan_start_ts: float,
    tr: float,
    vol_offset: int,
    n_vols: int,
    n_volumes_whitened: int,
    ar1_corrected: bool = True,
    method: str = "average",
    debug: bool = False,
    play_id: str = None,
    forward_fill_empty: bool = False,
) -> np.ndarray:
    """Align LLM features to TRs using their own timestamps.

    forward_fill_empty: If True, any TRs that received no LLM states will be
      filled with the most recent populated TR's values (last observation
      carried forward). Applied only WITHIN this play — callers must not use
      this across play boundaries, since values would spuriously leak between
      plays, levels, or games. Leading empty TRs (before the first populated
      one) stay at zero.
    """
    n_llm_states, n_features = activations.shape
    onsets = llm_timestamps - scan_start_ts

    if debug:
        logging.info(
            f"    [DEBUG LLM {play_id}] n_llm_states={n_llm_states}, n_features={n_features}"
        )
        logging.info(
            f"    [DEBUG LLM {play_id}] llm_timestamps: min={llm_timestamps.min():.3f}, max={llm_timestamps.max():.3f}"
        )
        logging.info(f"    [DEBUG LLM {play_id}] scan_start_ts={scan_start_ts:.3f}")

    volume_indices_original = np.round(onsets / tr).astype(int)

    if ar1_corrected:
        volume_indices = volume_indices_original - 1
        valid_mask = (volume_indices >= 0) & (volume_indices < n_volumes_whitened)
    else:
        volume_indices = volume_indices_original
        valid_mask = (volume_indices >= 0) & (volume_indices < n_volumes_whitened)

    volume_indices_local = volume_indices - vol_offset

    aligned = np.zeros((n_vols, n_features), dtype=np.float32)
    counts = np.zeros(n_vols, dtype=np.float32)

    for state_idx in range(n_llm_states):
        if not valid_mask[state_idx]:
            continue
        vol = volume_indices_local[state_idx]
        if vol < 0 or vol >= n_vols:
            continue
        if method == "average":
            aligned[vol] += activations[state_idx]
            counts[vol] += 1
        elif method == "last":
            aligned[vol] = activations[state_idx]
            counts[vol] = 1

    if method == "average":
        nonzero = counts > 0
        aligned[nonzero] /= counts[nonzero, np.newaxis]

    n_filled = 0
    if forward_fill_empty:
        # LOCF: after the first populated TR, carry last value forward through
        # empty TRs. Leading empties (before any data) stay at zero.
        populated = counts > 0
        first_populated_idx = None
        for i in range(n_vols):
            if populated[i]:
                first_populated_idx = i
                break

        if first_populated_idx is not None:
            last_value = aligned[first_populated_idx].copy()
            for i in range(first_populated_idx + 1, n_vols):
                if populated[i]:
                    last_value = aligned[i].copy()
                else:
                    aligned[i] = last_value
                    n_filled += 1

    if debug:
        n_nonzero_trs = (counts > 0).sum()
        fill_msg = f", forward-filled {n_filled}" if forward_fill_empty else ""
        logging.info(
            f"    [DEBUG LLM {play_id}] TRs with data: {n_nonzero_trs}/{n_vols}{fill_msg}"
        )

    return aligned


# =============================================================================
# Main Processing — MODIFIED from original process_subject:
#   - Takes aligned_data_path instead of preprocessed_dir + model_features_dir
#   - Loads BOLD/timing metadata and play ordering from aligned_data.npz
#   - DDQN, EZ, HRR, and behavioral blocks removed
#   - Only writes aligned_llm_{source}.npz files
# =============================================================================


def process_subject(
    subject: str,
    aligned_data_path: Path,
    output_dir: Path = None,
    llm_sources: List[LLMSourceConfig] = None,
    method: str = "average",
    incremental: bool = False,
    behavior_dir: Path = None,
) -> List[Path]:
    """Process LLM-only alignment for one subject."""
    if output_dir is None:
        raise ValueError("output_dir is required")
    if behavior_dir is None:
        raise ValueError("Provide behavior_dir with self-contained human JSON files")
    logging.info(f"Processing {subject}")
    logging.info(f"  Alignment method: {method}")

    subj_str = subject if subject.startswith("sub-") else f"sub-{int(subject):02d}"
    subj_num = int(subj_str.replace("sub-", ""))

    if not aligned_data_path.is_file():
        raise FileNotFoundError(aligned_data_path)
    base = np.load(aligned_data_path, allow_pickle=True)
    binding = bind_to_base(aligned_data_path, base)

    # -------------------------------------------------------------------------
    # Incremental mode: filter out already-aligned LLM sources
    # -------------------------------------------------------------------------
    output_subdir = output_dir / subject
    if incremental and llm_sources:
        new_sources = []
        for src in llm_sources:
            llm_file = output_subdir / f"aligned_llm_{src.name}.npz"
            if llm_file.exists():
                with np.load(llm_file, allow_pickle=False) as existing:
                    if not validate_binding(
                        existing,
                        base,
                        aligned_data_path,
                        base_sha256=str(binding["alignment_base_sha256"]),
                    ):
                        raise ValueError(
                            f"Cannot resume unbound feature archive: {llm_file}"
                        )
                logging.info(f"  [incremental] Skipping {src.name} — already aligned")
            else:
                new_sources.append(src)
        n_skipped = len(llm_sources) - len(new_sources)
        if n_skipped > 0:
            logging.info(
                f"  [incremental] {n_skipped} source(s) skipped, {len(new_sources)} remaining"
            )
        llm_sources = new_sources
        if not llm_sources:
            logging.info(f"  [incremental] Nothing to do — skipping {subject}")
            return []

    if not llm_sources:
        raise ValueError("Must provide at least one LLM source")

    # -------------------------------------------------------------------------
    # NEW: load BOLD/timing metadata from aligned_data.npz instead of from
    # preprocessed runs + DDQN feature files
    # -------------------------------------------------------------------------
    if not aligned_data_path.exists():
        raise FileNotFoundError(f"aligned_data.npz not found: {aligned_data_path}")
    logging.info(f"  Loading base alignment: {aligned_data_path}")
    tr = float(base["tr"])
    ar1_corrected = bool(base["ar1_corrected"])
    n_volumes_total = int(base["n_volumes"])
    game_names = [str(g) for g in base["game_names"]]
    play_ids_base = [str(p) for p in base["play_ids"]]
    play_game_idx_base = base["play_game_idx"].astype(int)
    play_levels_base = base["play_levels"].astype(int)
    play_n_volumes_base = base["play_n_volumes"].astype(int)

    n_plays_total = len(play_ids_base)
    logging.info(f"    tr={tr}, ar1_corrected={ar1_corrected}")
    logging.info(f"    n_volumes={n_volumes_total}, n_plays={n_plays_total}")
    logging.info(f"    games: {game_names}")

    # -------------------------------------------------------------------------
    # NEW: group play_ids by (game_idx, level) in original order, to drive the
    # per-level inner loop. This replaces the role of load_model_features_for_level
    # from the original script, which was the source of play ordering per level.
    # -------------------------------------------------------------------------
    plays_by_game_level: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for i, pid in enumerate(play_ids_base):
        plays_by_game_level[
            (int(play_game_idx_base[i]), int(play_levels_base[i]))
        ].append(pid)

    # Lookup from original play ID; run identity comes from human recordings.
    expected_n_vols_by_play_id = {
        pid: int(play_n_volumes_base[i]) for i, pid in enumerate(play_ids_base)
    }

    # -------------------------------------------------------------------------
    # Derive per-run n_volumes_whitened from the base file.
    # aligned_data doesn't store this directly; the safe value is the maximum
    # (vol_offset + retained n_vols) ever used in this run. Raw behavioral
    # timestamps may extend past the final scan volume: use unbounded timing
    # only for the start offset, and keep the base file's retained length.
    # -------------------------------------------------------------------------
    all_plays_by_id = load_canonical_plays(behavior_dir, subj_str)
    runs_by_key = _behavior_api().load_runs(behavior_dir)

    # Derive per-run n_volumes_whitened bound
    per_run_vol_max: Dict[int, int] = {}
    for i, pid in enumerate(play_ids_base):
        play_doc = all_plays_by_id.get(pid)
        if play_doc is None:
            raise RuntimeError(f"Play {pid} missing from source behavior records")
        run_id = int(play_doc["run_id"])
        run_doc = runs_by_key.get((subj_num, run_id))
        if run_doc is None:
            raise RuntimeError(f"Scanner run missing ({subj_num}, {run_id})")
        # Calculate the start offset without extending scan-truncated plays.
        tmp_timing = get_play_timing(
            play_doc, run_doc, tr, n_volumes_whitened=10**9, ar1_corrected=ar1_corrected
        )
        retained_n_volumes = int(play_n_volumes_base[i])
        if (
            tmp_timing is None
            or retained_n_volumes <= 0
            or retained_n_volumes > int(tmp_timing["n_volumes"])
        ):
            raise RuntimeError(
                f"Play {pid} retained length {retained_n_volumes} is inconsistent "
                "with its behavioral timestamps"
            )
        end = int(tmp_timing["volume_offset"]) + retained_n_volumes
        per_run_vol_max[run_id] = max(per_run_vol_max.get(run_id, 0), end)

    logging.info(f"  Per-run vol_max (used as n_volumes_whitened): {per_run_vol_max}")

    # -------------------------------------------------------------------------
    # game_list: from base file order
    # -------------------------------------------------------------------------
    game_list = list(game_names)
    logging.info(f"  Found {len(game_list)} games: {game_list}")

    # -------------------------------------------------------------------------
    # Initialize LLM Feature Manager — verbatim from original
    # -------------------------------------------------------------------------
    logging.info(
        f"  Initializing LLM feature manager with {len(llm_sources)} source(s)..."
    )
    llm_manager = LLMFeatureManager(llm_sources, subject)
    has_llm = llm_manager.has_sources()
    if not has_llm:
        raise RuntimeError("No LLM sources successfully initialized")

    # -------------------------------------------------------------------------
    # Accumulators (LLM only)
    # -------------------------------------------------------------------------
    all_llm_aligned = {key: [] for key in llm_manager.get_all_storage_keys()}

    global_play_idx = 0
    total_volumes = 0
    llm_stats = defaultdict(lambda: {"matched": 0, "missing": 0})

    # -------------------------------------------------------------------------
    # Main loop — mirrors original iteration order: game → level → play
    # -------------------------------------------------------------------------
    for game_idx, game in enumerate(game_list):
        logging.info(f"  Processing {game}")
        # Levels for this game: whichever appear in the base file for this game
        levels = sorted(
            {
                int(play_levels_base[i])
                for i in range(n_plays_total)
                if int(play_game_idx_base[i]) == game_idx
            }
        )
        if not levels:
            continue
        logging.info(f"    Processing {len(levels)} levels: {levels}")

        for level in levels:
            play_ids_this_level = plays_by_game_level[(game_idx, level)]
            if not play_ids_this_level:
                continue

            # Load LLM features for this level from ALL sources
            llm_by_source = llm_manager.load_for_level(game, level, subsample=False)

            level_plays = {
                pid: play
                for pid, play in all_plays_by_id.items()
                if int(play["subj_id"]) == subj_num
                and play["game_name"].lower() == game.lower()
                and int(play["level_id"]) == level
            }
            multiturn_by_source = {
                name: match_multiturn_plays(llm_by_source.get(name, {}), level_plays)
                for name, metadata in llm_manager.source_metadata.items()
                if metadata.get("is_multiturn", False)
            }
            # Validate every source row before selecting the requested subsample.
            # Mapping values share these dictionaries, preserving row order.
            for name, metadata in llm_manager.source_metadata.items():
                factor = metadata["config"].subsample
                if factor > 1 and llm_by_source.get(name):
                    llm_manager._apply_subsampling(llm_by_source[name], factor)

            llm_counts_str = ", ".join(
                f"{src}: {len(plays)}" for src, plays in llm_by_source.items()
            )
            logging.info(
                f"      Level {level}: {len(play_ids_this_level)} plays (LLM: {llm_counts_str})"
            )

            for play_idx_in_level, play_id in enumerate(play_ids_this_level):
                play_doc = all_plays_by_id[play_id]
                run_id = int(play_doc["run_id"])
                run_doc = runs_by_key[(subj_num, run_id)]
                n_vols_whitened_bound = per_run_vol_max[run_id]

                timing = get_play_timing(
                    play_doc,
                    run_doc,
                    tr,
                    n_volumes_whitened=n_vols_whitened_bound,
                    ar1_corrected=ar1_corrected,
                )
                if timing is None:
                    raise RuntimeError(
                        f"Play {play_id} has no valid TRs — inconsistent with aligned_data.npz"
                    )

                n_vols = int(timing["n_volumes"])
                expected = expected_n_vols_by_play_id[play_id]
                if n_vols != expected:
                    raise RuntimeError(
                        f"Play {play_id} n_vols mismatch: computed {n_vols}, "
                        f"aligned_data expected {expected}. Timing reconstruction "
                        f"diverges from original."
                    )

                # Temporal-order check (non-fatal in original; we raise here since
                # skipped plays would produce a row-count mismatch downstream)
                if not verify_state_temporal_order(timing["state_timestamps"], play_id):
                    raise RuntimeError(
                        f"Play {play_id} has temporal order violation but was included "
                        f"in aligned_data.npz — cannot reproduce alignment safely."
                    )

                # Original frame timestamps serve as the DDQN-timestamp
                # equivalent: the original script verifies DDQN timestamps match
                # these within 100 ms (verify_timestamp_alignment).
                ddqn_timestamps = timing["state_timestamps"]
                n_ddqn_states = len(ddqn_timestamps)

                vol_offset = int(timing["volume_offset"])

                # -------------------------------------------------------------
                # Align each source with the retained averaging/fill operations.
                # -------------------------------------------------------------
                for src_name, meta in llm_manager.source_metadata.items():
                    llm_by_local_index = llm_by_source.get(src_name, {})
                    is_multiturn_source_flag = meta.get("is_multiturn", False)

                    # Multiturn rows are matched once per level using exact source
                    # identities, or an unambiguous timestamp-only match.
                    if is_multiturn_source_flag:
                        llm_play = multiturn_by_source[src_name].get(play_id)
                    else:
                        llm_play = llm_by_local_index.get(play_idx_in_level)

                    if llm_play is not None:
                        llm_frame_indices = llm_play["timestamps"]

                        debug_llm = (global_play_idx < 3) or (
                            game_idx > 0
                            and level == levels[0]
                            and play_idx_in_level < 2
                        )
                        if debug_llm:
                            logging.info(
                                f"    [DEBUG {src_name} {play_id}] "
                                f"n_llm={len(llm_frame_indices)}, n_ddqn={n_ddqn_states}"
                            )

                        # Detect which format this LLM source uses.
                        # - Per-level format: `timestamps` are integer frame indices into the
                        #   play's DDQN state array. We look them up to get real ts.
                        # - Per-game format: `timestamps` are absolute
                        #   wall-clock timestamps (unix seconds, ~1.6e9). Pass straight
                        #   through to align_llm_features_by_timestamp.
                        is_multiturn_ts = (
                            len(llm_frame_indices) > 0
                            and np.max(llm_frame_indices)
                            > 1e8  # > ~3 years = real unix ts
                        )

                        if is_multiturn_ts:
                            # Per-game features: use the timestamps as-is.
                            valid_llm_mask = np.ones(len(llm_frame_indices), dtype=bool)
                            real_timestamps = llm_frame_indices.astype(np.float64)

                            # Check the timestamp range. Exact referenced
                            # frame clocks were verified before subsampling; the
                            # 2-second tolerance applies only to timestamp-only matching.
                            SLOP_SEC = 2.0
                            frame_min = float(ddqn_timestamps.min())
                            frame_max = float(ddqn_timestamps.max())
                            llm_min = float(real_timestamps.min())
                            llm_max = float(real_timestamps.max())

                            if (
                                llm_min < frame_min - SLOP_SEC
                                or llm_max > frame_max + SLOP_SEC
                            ):
                                # Counter rather than raise -- some plays may
                                # legitimately have edge-case timing differences.
                                # Track the count and fail at the end if too many.
                                llm_manager.timestamp_audit["fail"] += 1
                                llm_manager.timestamp_audit["examples"].append(
                                    {
                                        "play_id": play_id,
                                        "src_name": src_name,
                                        "frame_range": (frame_min, frame_max),
                                        "llm_range": (llm_min, llm_max),
                                        "frame_lead_gap": frame_min
                                        - llm_min,  # +ve = LLM observations start before recorded human frames
                                        "frame_trail_gap": llm_max
                                        - frame_max,  # +ve = LLM ends after recorded frames
                                    }
                                )
                            else:
                                llm_manager.timestamp_audit["pass"] += 1
                        else:
                            # Per-level features: timestamps are indices into ddqn_timestamps
                            llm_to_ddqn_indices = llm_frame_indices.astype(int)
                            valid_llm_mask = (llm_to_ddqn_indices >= 0) & (
                                llm_to_ddqn_indices < n_ddqn_states
                            )
                            real_timestamps = np.zeros(len(llm_frame_indices))
                            real_timestamps[valid_llm_mask] = ddqn_timestamps[
                                llm_to_ddqn_indices[valid_llm_mask]
                            ]

                        llm_matched_any = False
                        for layer_idx, layer_name in enumerate(meta["layer_names"]):
                            key = f"{src_name}_{layer_name}"

                            if layer_name in llm_play["activations"]:
                                llm_activations = llm_play["activations"][layer_name]
                                valid_activations = llm_activations[valid_llm_mask]
                                valid_timestamps = real_timestamps[valid_llm_mask]

                                if len(valid_activations) > 0:
                                    llm_aligned = align_llm_features_by_timestamp(
                                        activations=valid_activations,
                                        llm_timestamps=valid_timestamps,
                                        scan_start_ts=timing["scan_start_ts"],
                                        tr=tr,
                                        vol_offset=vol_offset,
                                        n_vols=n_vols,
                                        n_volumes_whitened=n_vols_whitened_bound,
                                        ar1_corrected=ar1_corrected,
                                        method=method,
                                        debug=(debug_llm and layer_idx == 0),
                                        play_id=f"{src_name}_{play_id}",
                                        # Multi-turn extractions only embed actions
                                        # (sparse in time); fill in-between TRs with
                                        # LOCF. Never crosses play boundaries because
                                        # this fn runs once per play.
                                        forward_fill_empty=is_multiturn_ts,
                                    )
                                else:
                                    llm_aligned = np.zeros(
                                        (n_vols, llm_activations.shape[1]),
                                        dtype=np.float32,
                                    )

                                all_llm_aligned[key].append(llm_aligned)
                                llm_matched_any = True
                            else:
                                n_features = llm_manager.get_n_features(
                                    src_name, layer_name
                                )
                                all_llm_aligned[key].append(
                                    np.zeros((n_vols, n_features), dtype=np.float32)
                                )

                        if llm_matched_any:
                            llm_stats[src_name]["matched"] += 1
                        else:
                            llm_stats[src_name]["missing"] += 1
                    else:
                        for layer_name in meta["layer_names"]:
                            key = f"{src_name}_{layer_name}"
                            n_features = llm_manager.get_n_features(
                                src_name, layer_name
                            )
                            all_llm_aligned[key].append(
                                np.zeros((n_vols, n_features), dtype=np.float32)
                            )
                        llm_stats[src_name]["missing"] += 1

                total_volumes += n_vols
                global_play_idx += 1

    if global_play_idx == 0:
        raise ValueError(f"No plays processed for {subject}")

    logging.info(f"  Total: {global_play_idx} plays, {total_volumes} volumes")

    # -------------------------------------------------------------------------
    # Concatenate LLM features
    # -------------------------------------------------------------------------
    llm_concatenated = {}
    for key in llm_manager.get_all_storage_keys():
        if all_llm_aligned[key]:
            llm_concatenated[key] = np.vstack(all_llm_aligned[key])
            logging.info(f"    llm_{key}: {llm_concatenated[key].shape}")

    for src_name in llm_manager.get_source_names():
        stats = llm_stats[src_name]
        logging.info(
            f"    LLM '{src_name}' stats: matched={stats['matched']}, missing={stats['missing']}"
        )

    # Multi-turn timestamp audit: assert all per-play LLM timestamps fell
    # within the original frame timestamp bounds. Fails the alignment
    # if too many plays violate, since this would indicate a play_id
    # misalignment between feature rows and source human recordings.
    audit = llm_manager.timestamp_audit
    total_audit = audit["pass"] + audit["fail"]
    if total_audit > 0:
        if audit["fail"] == 0:
            logging.info(
                f"  Timestamp audit: ✓ all {audit['pass']} multi-turn plays "
                f"fall within source recording timestamp bounds"
            )
        else:
            # Show the first few failure examples
            logging.error(
                f"  Timestamp audit: ✗ {audit['fail']} of {total_audit} multi-turn "
                f"plays have LLM timestamps outside source recording bounds (slop=2.0s)"
            )
            for i, ex in enumerate(audit["examples"][:5]):
                logging.error(
                    f"    play={ex['play_id']} src={ex['src_name']}: "
                    f"Frames [{ex['frame_range'][0]:.3f} .. {ex['frame_range'][1]:.3f}]  "
                    f"LLM [{ex['llm_range'][0]:.3f} .. {ex['llm_range'][1]:.3f}]  "
                    f"lead_gap={ex['frame_lead_gap']:+.3f}s  "
                    f"trail_gap={ex['frame_trail_gap']:+.3f}s"
                )
            if len(audit["examples"]) > 5:
                logging.error(f"    ... and {len(audit['examples']) - 5} more")
            # Hard-fail unless override
            fail_threshold = max(1, int(0.01 * total_audit))  # 1% tolerance
            if audit["fail"] > fail_threshold:
                raise RuntimeError(
                    f"Timestamp audit failure: {audit['fail']}/{total_audit} multi-turn "
                    f"plays have LLM timestamps outside source recording bounds. "
                    f"This indicates a play_id misalignment between the LLM extraction "
                    f"and human recordings — alignment cannot be trusted."
                )

    # Consistency: each array must have n_volumes_total rows
    for key, arr in llm_concatenated.items():
        if arr.shape[0] != n_volumes_total:
            raise RuntimeError(
                f"Row count for {key} is {arr.shape[0]} but aligned_data has "
                f"{n_volumes_total} volumes"
            )

    if llm_concatenated:
        verify_no_invalid_values(llm_concatenated, "LLM")
    logging.info("  Alignment verification: ✓ All checks passed")

    # -------------------------------------------------------------------------
    # Save per-source LLM files — LIFTED VERBATIM from original
    # -------------------------------------------------------------------------
    output_subdir.mkdir(parents=True, exist_ok=True)
    written_files = []

    for src_name, meta in llm_manager.source_metadata.items():
        src = meta["config"]
        prefix = f"llm_{src_name}"

        llm_save_dict = dict(binding)
        for layer_name in meta["layer_names"]:
            key = f"{src_name}_{layer_name}"
            if key in llm_concatenated:
                llm_save_dict[f"llm_{key}_aligned"] = llm_concatenated[key]

        llm_save_dict[f"{prefix}_layers"] = np.array(meta["layer_names"], dtype="U32")
        llm_save_dict[f"{prefix}_layer_indices"] = np.array(src.layers, dtype=int)
        llm_save_dict[f"{prefix}_subsample"] = src.subsample
        llm_save_dict[f"{prefix}_matched"] = llm_stats[src_name]["matched"]
        llm_save_dict[f"{prefix}_missing"] = llm_stats[src_name]["missing"]
        llm_save_dict[f"{prefix}_n_features"] = np.array(
            [
                (layer, meta["n_features_by_layer"].get(layer, -1))
                for layer in meta["layer_names"]
            ],
            dtype=[("layer", "U32"), ("n_features", "i4")],
        )

        llm_file = output_subdir / f"aligned_llm_{src_name}.npz"
        np.savez_compressed(llm_file, **llm_save_dict)
        logging.info(
            f"  Saved LLM: {llm_file.name} ({llm_file.stat().st_size / 1024**2:.1f} MB)"
        )
        written_files.append(llm_file)

    logging.info(f"  All files saved to: {output_subdir}")
    return written_files


# =============================================================================
# CLI Argument Parsing — verbatim from original
# =============================================================================


def parse_llm_source_arg(source_str: str) -> LLMSourceConfig:
    """Parse an LLM source specification string from command line."""
    parts = {}
    for part in source_str.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            parts[key.strip()] = value.strip()

    if "name" not in parts:
        raise ValueError(
            f"LLM source must specify 'name': {source_str}\n"
            f"Expected format: name=<name>,dir=<path>[,subsample=<int>][,layers=<l1>:<l2>:...][,stream=main|attn|mlp]"
        )
    if "dir" not in parts:
        raise ValueError(
            f"LLM source must specify 'dir': {source_str}\n"
            f"Expected format: name=<name>,dir=<path>[,subsample=<int>][,layers=<l1>:<l2>:...][,stream=main|attn|mlp]"
        )

    subsample = int(parts.get("subsample", 1))

    layers = None
    if "layers" in parts:
        layers = [int(x) for x in parts["layers"].split(":")]

    stream = parts.get("stream", "main")
    if stream not in MULTITURN_STREAM_KEYS:
        raise ValueError(
            f"LLM source 'stream' must be one of {list(MULTITURN_STREAM_KEYS)}; "
            f"got {stream!r} in: {source_str}"
        )

    return LLMSourceConfig(
        name=parts["name"],
        directory=Path(parts["dir"]),
        subsample=subsample,
        layers=layers,
        stream=stream,
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Align LLM features using an existing processed BOLD/base archive"
    )
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument(
        "--aligned-data",
        required=True,
        help="Processed BOLD/base NPZ with scanner timing and original play order",
    )
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        required=True,
        help="Canonical human recordings root or downloaded dataset root",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--llm-source",
        action="append",
        dest="llm_sources_raw",
        default=[],
        metavar="SPEC",
        help=(
            "LLM source specification. Can be repeated for multiple sources.\n"
            'Format: "name=<name>,dir=<path>[,subsample=<int>][,layers=<l1>:<l2>:...][,stream=main|attn|mlp]"'
        ),
    )
    parser.add_argument(
        "--method",
        default="average",
        choices=["average", "last"],
        help="Alignment method",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip LLM sources that already have output files",
    )

    args = parser.parse_args()

    if not args.llm_sources_raw:
        raise SystemExit("Must provide at least one --llm-source")

    llm_sources = []
    for src_str in args.llm_sources_raw:
        try:
            llm_sources.append(parse_llm_source_arg(src_str))
        except ValueError as e:
            logging.error(f"Invalid --llm-source argument: {e}")
            raise
    logging.info(f"Configured {len(llm_sources)} LLM source(s)")

    process_subject(
        subject=args.subject,
        aligned_data_path=Path(args.aligned_data),
        output_dir=Path(args.output_dir),
        llm_sources=llm_sources,
        method=args.method,
        incremental=args.incremental,
        behavior_dir=args.behavior_dir,
    )
    logging.info(f"Completed {args.subject}")
