#!/usr/bin/env python3
"""
Stage 2: Align — Map computational model activations to preprocessed BOLD timeseries.

Aligns activations from multiple model families (DDQN, EfficientZero, LLMs) and
HRR theory embeddings to the fMRI volume timeseries using behavioral timestamps
recorded during scanning.

Model feature sources:
    - DDQN: Conv and FC layer activations (1:1 with game states)
    - EfficientZero: Representation, value/policy, dynamics/reward layers
      with phase-aware extraction
    - LLM: Multi-source support (e.g., DeepSeek-R1 family, Qwen) with
      per-source subsampling and auto-discovered layer counts
    - HRR: Holographic Reduced Representations of EMPA theory strings
    - Behavioral: Keystates, scores, time-in-play (nuisance regressors)

Input:
    - preprocessed/ — Voxelwise BOLD timeseries from Stage 1 (AR(1) whitened)
    - DDQN features — Per-state activations with behavioral timestamps
    - EfficientZero traces — Optional .pt files per play
    - LLM features — Optional, one or more model sources
    - Canonical behavior/human JSON recordings and scanner-run metadata
    - features/theory/empa/source-regressors.json.gz — Optional EMPA theory strings for HRR

Output:
    Per-subject directory containing:
    - bold-ddqn-theory.npz: BOLD + DDQN + HRR + behavioral + metadata
    - aligned_llm_{source}.npz: One file per LLM source
    - aligned_ez.npz: EfficientZero features (if available)
    - aligned_hrr_decomposed.npz: Sprite/interaction/termination sub-vectors

Usage:
    python -m reason_to_play.fmri.align_baselines \\
        --subject sub-12 \\
        --preprocessed-dir ./workdir/preprocessed \\
        --model-features-dir ./workdir/ddqn_features \\
        --behavior-dir ./dataset/behavior/human \\
        --regressors-json ./dataset/features/theory/empa/source-regressors.json.gz \\
        --llm-source "name=r1_qwen_7b,dir=./llm_features/7B" \\
        --llm-source "name=r1_qwen_32b,dir=./llm_features/32B" \\
        --output-dir ./workdir/aligned_data
"""

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from reason_to_play.analysis.neural.alignment import (
    bind_to_base,
    sample_order_sha256,
    validate_binding,
)


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
# Configuration
# =============================================================================

# DDQN layers to extract
DDQN_LAYERS = ["conv1", "conv2", "fc1", "q_values"]

# Channel counts for conv layers (for adaptive pooling)
DDQN_CONV_CHANNELS = {
    "conv1": 32,
    "conv2": 64,
}

HRR_DIM = 348
HRR_SEED = 42

# LLM layers used when feature files do not provide layer configuration.
# The loader discovers layers from data files (see discover_llm_layers()).
# Only used if LLMSourceConfig.layers is None AND auto-discovery fails.
LLM_LAYERS = list(range(1, 65))
LLM_LAYER_NAMES = [f"layer_{i}" for i in LLM_LAYERS]

# EfficientZero layers organized by type
# - representation: 1:1 with game states, no filtering needed
# - value_policy: filter for phase='initial' to get 1:1
# - dynamics/reward: aggregate by timestep (sparse)
EZ_REPRESENTATION_LAYERS = [
    "representation.downsample_net.resblocks1.0",
    "representation.downsample_net.resblocks2.0",
    "representation.downsample_net.resblocks3.0",
    "representation.resblocks.0",
]

EZ_VALUE_POLICY_LAYERS = [
    "value_policy.resblocks.0",
    "value_policy.fc_values.0.0",
    "value_policy.fc_values.0.1",
    "value_policy.fc_values.0.3",
    "value_policy.fc_policy.0",
    "value_policy.fc_policy.1",
    "value_policy.fc_policy.3",
]

EZ_DYNAMICS_REWARD_LAYERS = [
    "dynamics.resblocks.0",
    "reward.fc.0",
    "reward.fc.1",
    "reward.fc.3",
]

# All EZ layers combined
EZ_LAYERS = (
    EZ_REPRESENTATION_LAYERS + EZ_VALUE_POLICY_LAYERS + EZ_DYNAMICS_REWARD_LAYERS
)

GAME_ORDER = [
    "vgfmri4_avoidgeorge",
    "vgfmri4_bait",
    "vgfmri4_chase",
    "vgfmri4_helper",
    "vgfmri4_lemmings",
    "vgfmri4_zelda",
]


# =============================================================================
# LLM Source Configuration
# =============================================================================


@dataclass
class LLMSourceConfig:
    """
    Configuration for a single LLM feature source.

    This dataclass encapsulates all settings needed to load and process
    features from one LLM model. Multiple instances can be used to compare
    different models or the same model with different subsampling rates.

    Attributes:
        name: Unique identifier for this source (e.g., "deepseek32b", "deepseek685b").
              Used as prefix in output keys and logging.
        directory: Path to the directory containing LLM features for this source.
                   Expected structure: {directory}/{subject}/{game}/level_XX.pt
        subsample: Subsampling factor applied BEFORE alignment.
                   1 = no subsampling (use all frames)
                   2 = keep every 2nd frame (e.g., convert 10-frame to 20-frame sampling)
                   N = keep every Nth frame
        layers: List of layer indices to extract. If None, uses global LLM_LAYERS default.

    Example:
        >>> cfg = LLMSourceConfig(
        ...     name="deepseek32b_sub2",
        ...     directory=Path("./llm_features/32B"),
        ...     subsample=2,  # Subsample to match 685B's 20-frame rate
        ...     layers=[1, 8, 16, 24, 32, 40, 48, 56, 64]
        ... )
    """

    name: str
    directory: Path
    subsample: int = 1
    layers: List[int] = None

    def __post_init__(self):
        """Initialize derived attributes after dataclass creation."""
        # If layers explicitly provided, compute layer names now.
        # If None, defer to auto-discovery in LLMFeatureManager._initialize_source.
        if self.layers is not None:
            self.layer_names = [f"layer_{i}" for i in self.layers]
        else:
            self.layer_names = None  # Will be set during initialization

        # Validate subsample factor
        if self.subsample < 1:
            raise ValueError(f"subsample must be >= 1, got {self.subsample}")


def _game_directory(root, subject, game):
    from reason_to_play.data.behavior import canonical_game_id

    candidates = [root / subject / canonical_game_id(game), root / subject / game]
    found = list(dict.fromkeys(path for path in candidates if path.is_dir()))
    if len(found) > 1:
        raise ValueError(f"Multiple feature directories for {subject}/{game}: {found}")
    return found[0] if found else candidates[0]


def _level_files(directory):
    found = {}
    for path in directory.glob("level*.npz"):
        match = re.fullmatch(r"level[-_](\d+)", path.stem)
        if match:
            level = int(match[1])
            if level in found:
                raise ValueError(
                    f"Duplicate feature files for level {level}: {directory}"
                )
            found[level] = path
    return found


# =============================================================================
# Conv Layer Pooling Functions
# =============================================================================


def infer_spatial_dims(n_flat: int, n_channels: int) -> Tuple[int, int]:
    """Infer spatial dimensions (H, W) from flattened conv output size."""
    spatial_size = n_flat // n_channels
    w = 30
    h = spatial_size // w
    if h * w == spatial_size:
        return h, w
    for w_try in [30, 25, 20, 15, 10]:
        if spatial_size % w_try == 0:
            return spatial_size // w_try, w_try
    h = int(np.sqrt(spatial_size))
    while h > 0 and spatial_size % h != 0:
        h -= 1
    return (h, spatial_size // h) if h > 0 else (1, spatial_size)


def find_common_conv_resolution(
    model_dir: Path, subject: str, games: list, n_channels: dict
) -> dict:
    """Find the minimum spatial resolution across all games for each conv layer."""
    layer_dims = {layer: [] for layer in n_channels.keys()}
    for game in games:
        game_dir = _game_directory(model_dir, subject, game)
        if not game_dir.exists():
            continue
        npz_files = list(_level_files(game_dir).values())
        if not npz_files:
            continue
        data = np.load(npz_files[0])
        for layer, n_ch in n_channels.items():
            for key in data.keys():
                if f"model_{layer}" in key:
                    n_flat = data[key].shape[1]
                    h, w = infer_spatial_dims(n_flat, n_ch)
                    layer_dims[layer].append((h, w, game))
                    break
        data.close()

    common_resolution = {}
    for layer, dims in layer_dims.items():
        if dims:
            min_h = min(d[0] for d in dims)
            min_w = min(d[1] for d in dims)
            common_resolution[layer] = (min_h, min_w)
            dims_str = ", ".join([f"{g}: {h}x{w}" for h, w, g in dims])
            logging.info(f"    {layer}: {dims_str} -> common = ({min_h}, {min_w})")
    return common_resolution


def adaptive_avg_pool_conv(
    activation: np.ndarray, n_channels: int, target_size: Tuple[int, int]
) -> np.ndarray:
    """Adaptive average pooling for conv activations to a target spatial size."""
    n_states, n_flat = activation.shape
    h, w = infer_spatial_dims(n_flat, n_channels)
    target_h, target_w = target_size
    if h == target_h and w == target_w:
        return activation.astype(np.float32)
    reshaped = activation.reshape(n_states, n_channels, h, w)
    pooled = np.zeros((n_states, n_channels, target_h, target_w), dtype=np.float32)
    for i in range(target_h):
        for j in range(target_w):
            h_start = (i * h) // target_h
            h_end = ((i + 1) * h) // target_h
            w_start = (j * w) // target_w
            w_end = ((j + 1) * w) // target_w
            if h_end <= h_start:
                h_end = h_start + 1
            if w_end <= w_start:
                w_end = w_start + 1
            pooled[:, :, i, j] = reshaped[:, :, h_start:h_end, w_start:w_end].mean(
                axis=(2, 3)
            )
    return pooled.reshape(n_states, -1).astype(np.float32)


# =============================================================================
# HRR Embedding Class
# =============================================================================


class HRREncoder:
    """Holographic Reduced Representation encoder for EMPA theories."""

    def __init__(self, dim: int = 348, seed: int = None):
        self.dim = dim
        self.sigma = 1.0 / np.sqrt(dim)
        self._rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self._vocab = {}
        for role in [
            "type",
            "color",
            "name",
            "effect",
            "agent",
            "patient",
            "generic",
            "rule_type",
            "sprite1",
            "sprite2",
            "outcome",
            "count",
        ]:
            self._get_or_create_vector(f"ROLE_{role}")

    def _get_or_create_vector(self, token: str) -> np.ndarray:
        if token not in self._vocab:
            self._vocab[token] = self._rng.normal(0, self.sigma, self.dim)
        return self._vocab[token]

    def _circular_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _bind(self, role: str, filler: str) -> np.ndarray:
        return self._circular_convolution(
            self._get_or_create_vector(f"ROLE_{role}"),
            self._get_or_create_vector(filler),
        )

    def _bind_with_vector(self, role: str, filler_vec: np.ndarray) -> np.ndarray:
        return self._circular_convolution(
            self._get_or_create_vector(f"ROLE_{role}"), filler_vec
        )

    def parse_theory_string(self, theory_str: str) -> dict:
        result = {"interactions": [], "terminations": [], "classes": {}}
        if not theory_str or not isinstance(theory_str, str):
            return result
        lines = theory_str.strip().split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == "InteractionSet:":
                current_section = "interactions"
            elif line == "TerminationSet:":
                current_section = "terminations"
            elif line == "Class assignments:":
                current_section = "classes"
            elif current_section == "interactions":
                match = re.match(
                    r"^(\w+)\s+(\w+)\s+(\w+)\s+(\{.*?\})\s+generic:\s*(True|False)",
                    line,
                )
                if match:
                    effect, agent, patient, params_str, generic = match.groups()
                    result["interactions"].append(
                        {
                            "effect": effect,
                            "agent": agent,
                            "patient": patient,
                            "generic": generic == "True",
                            "params": params_str,
                        }
                    )
            elif current_section == "terminations":
                parts = line.split()
                if len(parts) >= 3:
                    rule_type = parts[0]
                    if rule_type == "NoveltyRule" and len(parts) >= 4:
                        result["terminations"].append(
                            {
                                "rule_type": rule_type,
                                "sprite1": parts[1],
                                "sprite2": parts[2],
                                "explored": parts[3] if len(parts) > 3 else "True",
                                "outcome": parts[4] if len(parts) > 4 else "None",
                            }
                        )
                    elif rule_type == "SpriteCounterRule" and len(parts) >= 4:
                        result["terminations"].append(
                            {
                                "rule_type": rule_type,
                                "sprite": parts[1],
                                "count": parts[2],
                                "outcome": parts[3],
                            }
                        )
                    elif rule_type == "MultiSpriteCounterRule" and len(parts) >= 5:
                        result["terminations"].append(
                            {
                                "rule_type": rule_type,
                                "sprite1": parts[1],
                                "sprite2": parts[2],
                                "count": parts[3],
                                "outcome": parts[4],
                            }
                        )
                    else:
                        result["terminations"].append(
                            {"rule_type": rule_type, "raw": line}
                        )
            elif current_section == "classes":
                match = re.match(r"^(\w+):\s*\['(\w+)'\]:\s*<class '([^']+)'>", line)
                if match:
                    name, color, vgdl_class = match.groups()
                    vgdl_type = (
                        vgdl_class.split(".")[-1] if "." in vgdl_class else vgdl_class
                    )
                    result["classes"][name] = {"color": color, "vgdl_type": vgdl_type}
        return result

    def encode_theory(self, theory_str: str) -> np.ndarray:
        if not theory_str or not isinstance(theory_str, str):
            return np.zeros(self.dim)
        parsed = self.parse_theory_string(theory_str)
        sprite_vectors = {}
        for class_name, class_info in parsed["classes"].items():
            emb = np.zeros(self.dim)
            emb += self._bind("name", class_name)
            if "color" in class_info:
                emb += self._bind("color", class_info["color"])
            if "vgdl_type" in class_info:
                emb += self._bind("type", class_info["vgdl_type"])
            sprite_vectors[class_name] = self._normalize(emb)
        sprite_set_vec = np.zeros(self.dim)
        for sv in sprite_vectors.values():
            sprite_set_vec += sv
        sprite_set_vec = (
            self._normalize(sprite_set_vec) if sprite_vectors else sprite_set_vec
        )
        interaction_set_vec = np.zeros(self.dim)
        for inter in parsed["interactions"]:
            emb = np.zeros(self.dim)
            emb += self._bind("effect", inter["effect"])
            agent = inter["agent"]
            emb += (
                self._bind_with_vector("agent", sprite_vectors[agent])
                if agent in sprite_vectors
                else self._bind("agent", agent)
            )
            patient = inter["patient"]
            emb += (
                self._bind_with_vector("patient", sprite_vectors[patient])
                if patient in sprite_vectors
                else self._bind("patient", patient)
            )
            emb += self._bind(
                "generic",
                "generic_true" if inter.get("generic", True) else "generic_false",
            )
            interaction_set_vec += self._normalize(emb)
        interaction_set_vec = (
            self._normalize(interaction_set_vec)
            if parsed["interactions"]
            else interaction_set_vec
        )
        termination_set_vec = np.zeros(self.dim)
        for term in parsed["terminations"]:
            emb = np.zeros(self.dim)
            emb += self._bind("rule_type", term["rule_type"])
            for key in ["sprite", "sprite1"]:
                if key in term:
                    s = term[key]
                    emb += (
                        self._bind_with_vector("sprite1", sprite_vectors[s])
                        if s in sprite_vectors
                        else self._bind("sprite1", s)
                    )
            if "sprite2" in term:
                s = term["sprite2"]
                emb += (
                    self._bind_with_vector("sprite2", sprite_vectors[s])
                    if s in sprite_vectors
                    else self._bind("sprite2", s)
                )
            if "outcome" in term:
                emb += self._bind("outcome", str(term["outcome"]))
            if "count" in term:
                emb += self._bind("count", f"count_{term['count']}")
            termination_set_vec += self._normalize(emb)
        termination_set_vec = (
            self._normalize(termination_set_vec)
            if parsed["terminations"]
            else termination_set_vec
        )
        return self._normalize(
            sprite_set_vec + interaction_set_vec + termination_set_vec
        )

    def encode_theory_decomposed(self, theory_str: str) -> tuple:
        """Encode theory and return the three sub-vectors separately.

        Returns:
            (sprite_set_vec, interaction_set_vec, termination_set_vec)
            Each is a normalized vector of shape (self.dim,).
        """
        if not theory_str or not isinstance(theory_str, str):
            z = np.zeros(self.dim)
            return z.copy(), z.copy(), z.copy()
        parsed = self.parse_theory_string(theory_str)
        sprite_vectors = {}
        for class_name, class_info in parsed["classes"].items():
            emb = np.zeros(self.dim)
            emb += self._bind("name", class_name)
            if "color" in class_info:
                emb += self._bind("color", class_info["color"])
            if "vgdl_type" in class_info:
                emb += self._bind("type", class_info["vgdl_type"])
            sprite_vectors[class_name] = self._normalize(emb)
        sprite_set_vec = np.zeros(self.dim)
        for sv in sprite_vectors.values():
            sprite_set_vec += sv
        sprite_set_vec = (
            self._normalize(sprite_set_vec) if sprite_vectors else sprite_set_vec
        )
        interaction_set_vec = np.zeros(self.dim)
        for inter in parsed["interactions"]:
            emb = np.zeros(self.dim)
            emb += self._bind("effect", inter["effect"])
            agent = inter["agent"]
            emb += (
                self._bind_with_vector("agent", sprite_vectors[agent])
                if agent in sprite_vectors
                else self._bind("agent", agent)
            )
            patient = inter["patient"]
            emb += (
                self._bind_with_vector("patient", sprite_vectors[patient])
                if patient in sprite_vectors
                else self._bind("patient", patient)
            )
            emb += self._bind(
                "generic",
                "generic_true" if inter.get("generic", True) else "generic_false",
            )
            interaction_set_vec += self._normalize(emb)
        interaction_set_vec = (
            self._normalize(interaction_set_vec)
            if parsed["interactions"]
            else interaction_set_vec
        )
        termination_set_vec = np.zeros(self.dim)
        for term in parsed["terminations"]:
            emb = np.zeros(self.dim)
            emb += self._bind("rule_type", term["rule_type"])
            for key in ["sprite", "sprite1"]:
                if key in term:
                    s = term[key]
                    emb += (
                        self._bind_with_vector("sprite1", sprite_vectors[s])
                        if s in sprite_vectors
                        else self._bind("sprite1", s)
                    )
            if "sprite2" in term:
                s = term["sprite2"]
                emb += (
                    self._bind_with_vector("sprite2", sprite_vectors[s])
                    if s in sprite_vectors
                    else self._bind("sprite2", s)
                )
            if "outcome" in term:
                emb += self._bind("outcome", str(term["outcome"]))
            if "count" in term:
                emb += self._bind("count", f"count_{term['count']}")
            termination_set_vec += self._normalize(emb)
        termination_set_vec = (
            self._normalize(termination_set_vec)
            if parsed["terminations"]
            else termination_set_vec
        )
        return sprite_set_vec, interaction_set_vec, termination_set_vec


def encode_theory_sequence(encoder: HRREncoder, theory_strs: list) -> np.ndarray:
    return np.array([encoder.encode_theory(ts) for ts in theory_strs])


def encode_theory_sequence_decomposed(encoder: HRREncoder, theory_strs: list) -> tuple:
    """Encode a sequence of theories, returning three arrays of sub-vectors.

    Returns:
        (sprites, interactions, terminations) each of shape (n_states, dim)
    """
    sprites, interactions, terminations = [], [], []
    for ts in theory_strs:
        s, i, t = encoder.encode_theory_decomposed(ts)
        sprites.append(s)
        interactions.append(i)
        terminations.append(t)
    return np.array(sprites), np.array(interactions), np.array(terminations)


# =============================================================================
# EfficientZero Loading Functions
# =============================================================================


def load_ez_traces(
    trace_path: Path, layers: list = None, n_expected_states: int = None
) -> Optional[Dict]:
    """
    Load EfficientZero traces from a .pt file with phase-aware extraction.
    """
    if not trace_path.exists():
        return None
    torch = _torch()

    if layers is None:
        layers = EZ_LAYERS

    try:
        from reason_to_play.fmri.align_efficientzero import load_trace_document

        data = load_trace_document(trace_path)
    except Exception as e:
        logging.warning(f"Failed to load {trace_path}: {e}")
        return None

    metadata = data.get("metadata", {})
    traces = data.get("traces", OrderedDict())
    if not traces:
        return None

    # Determine n_states from representation layer (ground truth for game states)
    n_states = None
    for rep_layer in EZ_REPRESENTATION_LAYERS:
        if rep_layer in traces:
            n_states = len(traces[rep_layer])
            break

    if n_states is None:
        first_layer = list(traces.keys())[0]
        n_states = len(traces[first_layer])

    activations_by_layer = {}

    for layer in layers:
        if layer not in traces:
            continue

        layer_data = traces[layer]

        if layer in EZ_REPRESENTATION_LAYERS:
            layer_activations = []
            for entry in layer_data:
                act = entry["activation"]
                if isinstance(act, torch.Tensor):
                    act = act.numpy()
                layer_activations.append(act)
            activations_by_layer[layer] = np.array(layer_activations, dtype=np.float32)

        elif layer in EZ_VALUE_POLICY_LAYERS:
            layer_activations = []
            for entry in layer_data:
                if entry.get("phase") == "initial":
                    act = entry["activation"]
                    if isinstance(act, torch.Tensor):
                        act = act.numpy()
                    layer_activations.append(act)

            if layer_activations:
                activations_by_layer[layer] = np.array(
                    layer_activations, dtype=np.float32
                )

        elif layer in EZ_DYNAMICS_REWARD_LAYERS:
            timestep_activations = defaultdict(list)
            n_features = None

            for entry in layer_data:
                timestep = entry.get("timestep", 0)
                act = entry["activation"]
                if isinstance(act, torch.Tensor):
                    act = act.numpy()
                if n_features is None:
                    n_features = act.shape[0] if act.ndim == 1 else act.size
                timestep_activations[timestep].append(act.flatten())

            if n_features is None:
                continue

            result = np.zeros((n_states, n_features), dtype=np.float32)
            for timestep, acts in timestep_activations.items():
                if 0 <= timestep < n_states:
                    result[timestep] = np.mean(acts, axis=0)

            activations_by_layer[layer] = result

    if not activations_by_layer:
        return None

    return {
        "metadata": metadata,
        "activations": activations_by_layer,
        "n_timesteps": n_states,
        "layers": list(activations_by_layer.keys()),
    }


def find_available_ez_plays(ez_dir: Path, subject: str) -> Tuple[Dict, bool]:
    """
    Find all available EfficientZero trace files for a subject.

    Supports two directory structures:
    - Ordinal paths: play0/, play1/, ... (requires state-count matching)
    - Play-ID paths: play0_key{original_id}/, ... (direct matching by play_id)

    Returns:
        Tuple of:
        - Dict mapping:
          - Play-ID paths: play_id (str) -> (trace_path, game_name, run_id)
          - Ordinal paths: (game, run, play_idx) tuple -> (trace_path,)
        - Boolean indicating whether paths contain play IDs

    Note: This function does NOT load the trace files - it only discovers paths.
    Traces are loaded lazily when needed during alignment.
    """
    if ez_dir is None or not ez_dir.exists():
        return {}, False

    subj_num = (
        int(subject.replace("sub-", "")) if subject.startswith("sub-") else int(subject)
    )
    subj_str = f"subj{subj_num}"
    available = {}
    has_play_ids = False

    from reason_to_play.data.behavior import game_identity

    for game_dir in ez_dir.iterdir():
        if not game_dir.is_dir():
            continue
        try:
            game, cohort = game_identity(game_dir.name)
        except ValueError:
            continue
        game_name = f"{cohort}_{game}"
        candidates = [game_dir / f"sub-{subj_num:02d}", game_dir / subj_str]
        candidates = [path for path in candidates if path.is_dir()]
        if len(candidates) > 1:
            raise ValueError(
                f"Duplicate EfficientZero subject directories: {candidates}"
            )
        if not candidates:
            continue
        subj_dir = candidates[0]
        for run_dir in subj_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run"):
                continue
            run_id = int(run_dir.name.removeprefix("run").removeprefix("-"))
            for play_dir in run_dir.iterdir():
                if not play_dir.is_dir() or not play_dir.name.startswith("play"):
                    continue
                trace_path = play_dir / "traces.pt"
                if not trace_path.exists():
                    continue

                # Check for a play ID in play{idx}_key{original_id}
                play_dir_name = play_dir.name
                if play_dir_name.startswith("play-") or "_key" in play_dir_name:
                    # Extract play_id directly without loading the trace file
                    has_play_ids = True
                    play_id = (
                        play_dir_name.removeprefix("play-")
                        if play_dir_name.startswith("play-")
                        else play_dir_name.split("_key")[1]
                    )
                    if not re.fullmatch(r"[a-fA-F0-9]{24}", play_id):
                        raise ValueError(f"Invalid original play identity: {play_dir}")
                    if play_id in available:
                        raise ValueError(
                            f"Duplicate EfficientZero trace for original play {play_id}"
                        )
                    available[play_id] = (trace_path, game_name, run_id)
                else:
                    # Ordinal path: play{idx}; load later to match by frame count
                    ez_play_idx = int(play_dir_name.replace("play", ""))
                    available[(game_name, run_id, ez_play_idx)] = (trace_path,)

    if has_play_ids:
        logging.info(
            f"    Using EZ paths with embedded play IDs ({len(available)} plays)"
        )
    else:
        logging.info(f"    Using ordinal EZ paths ({len(available)} plays)")

    return available, has_play_ids


def build_ez_to_behavioral_mapping(
    ez_available: Dict[Tuple[str, int, int], Tuple[Path]],
    behavioral_plays: List[Dict],
    subject: str,
) -> Dict[Tuple[str, int, int], str]:
    """
    Build mapping from EZ (game, run, ez_play_idx) -> original behavioral play ID.

    Ordinal paths require this frame-count mapping.
    Paths with embedded play IDs support direct lookup.

    This function loads each trace file to get state counts for matching.
    """
    subj_num = (
        str(int(subject.replace("sub-", "")))
        if subject.startswith("sub-")
        else str(int(subject))
    )

    # Load state counts for each EZ play (required for matching)
    logging.info("      Loading EZ trace files to get state counts...")
    ez_by_game_run = {}
    for (game, run_id, ez_play_idx), (path,) in ez_available.items():
        ez_data = load_ez_traces(path, layers=EZ_LAYERS)
        if ez_data is None:
            continue
        n_states = ez_data["n_timesteps"]
        key = (game, run_id)
        if key not in ez_by_game_run:
            ez_by_game_run[key] = []
        ez_by_game_run[key].append((ez_play_idx, n_states, path))

    for key in ez_by_game_run:
        ez_by_game_run[key].sort(key=lambda x: x[0])

    beh_by_game_run = {}
    for play in behavioral_plays:
        if str(play.get("subj_id")) != subj_num:
            continue
        game = play.get("game_name")
        run_id = play.get("run_id")
        if game is None or run_id is None:
            continue
        try:
            n_states = len(play_states(play))
        except Exception:
            continue
        key = (game, run_id)
        if key not in beh_by_game_run:
            beh_by_game_run[key] = []
        beh_by_game_run[key].append(
            {
                "_id": str(play["_id"]),
                "n_states": n_states,
                "start_time": play.get("start_time", 0),
            }
        )
    for key in beh_by_game_run:
        beh_by_game_run[key].sort(key=lambda x: x["start_time"])
    mapping = {}
    match_stats = {"matched": 0, "unmatched": 0}
    for (game, run_id), ez_plays in ez_by_game_run.items():
        beh_plays = beh_by_game_run.get((game, run_id), [])
        if not beh_plays:
            for ez_play_idx, n_states, path in ez_plays:
                mapping[(game, run_id, ez_play_idx)] = None
                match_stats["unmatched"] += 1
            continue
        beh_idx = 0
        for ez_play_idx, ez_n_states, path in ez_plays:
            matched = False
            while beh_idx < len(beh_plays):
                if beh_plays[beh_idx]["n_states"] == ez_n_states:
                    mapping[(game, run_id, ez_play_idx)] = beh_plays[beh_idx]["_id"]
                    match_stats["matched"] += 1
                    matched = True
                    beh_idx += 1
                    break
                beh_idx += 1
            if not matched:
                mapping[(game, run_id, ez_play_idx)] = None
                match_stats["unmatched"] += 1
    logging.info(
        f"    EZ->Behavioral mapping: {match_stats['matched']} matched, {match_stats['unmatched']} unmatched"
    )
    return mapping


# =============================================================================
# LLM Loading Functions
# =============================================================================


def discover_llm_layers(llm_dir: Path, subject: str) -> List[int]:
    """
    Auto-discover available layer indices by peeking at one data file.

    Scans the first available .pt or .npz file for a subject and extracts
    all unique layer indices from keys matching 'play_X_model_layer_Y' or
    'play_X_layer_Y' patterns.

    Args:
        llm_dir: Root directory for this LLM source
        subject: Subject ID (e.g., 'sub-12')

    Returns:
        Sorted list of layer indices found, or empty list if discovery fails.
    """
    subj_dir = llm_dir / subject
    if not subj_dir.exists():
        return []

    for game_dir in sorted(subj_dir.iterdir()):
        if not game_dir.is_dir() or not game_dir.name.startswith("vgfmri"):
            continue

        # Try .pt files first, then .npz
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
                # Match: play_X_model_layer_Y or play_X_layer_Y (with optional _post)
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
    llm_dir: Path, subject: str, game: str, level: int, layers: list = None
) -> dict:
    """
    Load LLM features for a specific game/level.

    Returns features indexed by LOCAL play position (0, 1, 2, ...) within the level,
    regardless of the global play index stored in the file.

    Note: LLM files use global play indices across the game (e.g., level 1 might have
    plays 3,4,5,6,7 instead of 0,1,2,3,4), so we discover actual indices and map them
    to local positions.

    Returns:
        Dict mapping local_play_index (0, 1, 2...) -> {'activations': {...}, ...}
    """
    if layers is None:
        layers = LLM_LAYER_NAMES

    # Try both .pt and .npz extensions
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
            # Convert torch tensors to numpy, handling BFloat16
            if isinstance(data, dict):
                converted = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        # Convert BFloat16 to Float32 before numpy conversion
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

    # Discover actual play indices in the file (they may not start at 0)
    # Look for keys like 'play_N_behavioral_timestamps'
    play_indices_found = set()
    for key in data.keys():
        if key.startswith("play_") and "_behavioral_timestamps" in key:
            try:
                idx = int(key.split("_")[1])
                play_indices_found.add(idx)
            except (ValueError, IndexError):
                pass

    # Fallback: discover play indices from layer keys if no timestamp keys found
    if not play_indices_found:
        for key in data.keys():
            match = re.match(r"play_(\d+)_(?:model_)?layer_\d+", key)
            if match:
                play_indices_found.add(int(match.group(1)))

    if not play_indices_found:
        return {}

    # Sort indices to maintain order
    sorted_indices = sorted(play_indices_found)

    # Return features indexed by LOCAL position (0, 1, 2, ...)
    # mapping from the actual global indices in the file
    llm_by_local_index = {}

    for local_idx, global_idx in enumerate(sorted_indices):
        # Extract activations for requested layers
        activations = {}
        for layer_name in layers:
            # Try both key formats: with and without 'model_' prefix
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

        # Get timestamps
        timestamps_key = f"play_{global_idx}_behavioral_timestamps"
        if timestamps_key not in data:
            continue
        timestamps = data[timestamps_key]
        if torch is not None and isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.numpy()

        # Get num_states
        num_states_key = f"play_{global_idx}_behavioral_num_states"
        num_states = int(data.get(num_states_key, len(timestamps)))

        # Store by LOCAL index position
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


def find_llm_games_and_levels(llm_dir: Path, subject: str) -> dict:
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

        # Check for both .pt and .npz files
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
# LLM Feature Manager (Multi-Source Support)
# =============================================================================


class LLMFeatureManager:
    """
    Manages loading and preprocessing for multiple LLM feature sources.

    Orchestrates multiple LLM sources by:
    1. Tracking source configurations (directory, subsample rate, layers)
    2. Calling existing loading functions per source
    3. Applying per-source subsampling after loading, before alignment
    4. Providing unified access to feature dimensions for output allocation

    Architecture:
        LLMFeatureManager
            ├── source_1 (LLMSourceConfig)
            │   ├── load via load_llm_features_for_level()
            │   └── subsample if configured
            ├── source_2 (LLMSourceConfig)
            │   ├── load via load_llm_features_for_level()
            │   └── subsample if configured
            └── ...

    Usage:
        >>> sources = [
        ...     LLMSourceConfig(name="ds32b", directory=Path("./32b")),
        ...     LLMSourceConfig(name="ds685b", directory=Path("./685b")),
        ... ]
        >>> manager = LLMFeatureManager(sources, subject="sub-01")
        >>> llm_by_source = manager.load_for_level("vgfmri4_chase", level=0)
    """

    def __init__(self, sources: List[LLMSourceConfig], subject: str):
        """
        Initialize the manager with a list of LLM sources.

        Args:
            sources: List of LLMSourceConfig objects, one per LLM source
            subject: Subject ID (e.g., "sub-01") for locating feature files
        """
        self.sources = {src.name: src for src in sources}
        self.subject = subject

        # Metadata populated during initialization
        # Maps source_name -> {config, games_levels, n_features_by_layer, layer_names}
        self.source_metadata = {}

        # Initialize each source (discover available data, get feature dims)
        for src in sources:
            self._initialize_source(src)

    def _initialize_source(self, src: LLMSourceConfig) -> None:
        """
        Discover available data and feature dimensions for one source.

        Validates the source directory, discovers available games/levels and
        layer indices, and gets feature dimensions from the first available file.
        """
        # Validate directory
        if not src.directory.exists():
            logging.warning(
                f"LLM source '{src.name}' directory not found: {src.directory}"
            )
            return

        # Discover available games and levels
        games_levels = find_llm_games_and_levels(src.directory, self.subject)

        if not games_levels:
            logging.warning(
                f"LLM source '{src.name}': No data found for {self.subject}"
            )
            return

        # Auto-discover layers if not explicitly specified
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
                # Final fallback to global default
                logging.warning(
                    f"    LLM source '{src.name}': auto-discovery failed, "
                    f"falling back to global LLM_LAYERS default ({len(LLM_LAYERS)} layers)"
                )
                src.layers = list(LLM_LAYERS)
                src.layer_names = [f"layer_{i}" for i in src.layers]

        # Get feature dimensions from first available file
        n_features_by_layer = {}

        for game, levels in games_levels.items():
            sample_data = load_llm_features_for_level(
                src.directory,
                self.subject,
                game,
                levels[0],  # First available level
                src.layer_names,
            )

            if sample_data:
                # Extract feature dimensions from first play
                first_play = list(sample_data.values())[0]
                for layer_name, activations in first_play["activations"].items():
                    n_features_by_layer[layer_name] = activations.shape[1]
                break  # Only need one file to get dimensions

        # Store metadata
        self.source_metadata[src.name] = {
            "config": src,
            "games_levels": games_levels,
            "n_features_by_layer": n_features_by_layer,
            "layer_names": src.layer_names,
        }

        # Log initialization summary
        logging.info(
            f"    LLM source '{src.name}': "
            f"{len(games_levels)} games, "
            f"subsample={src.subsample}, "
            f"layers={src.layers}"
        )
        for layer_name, n_feat in n_features_by_layer.items():
            logging.info(f"      {layer_name}: {n_feat} features")

    def has_sources(self) -> bool:
        """
        Check if any sources were successfully initialized.

        Returns:
            True if at least one source has valid metadata, False otherwise
        """
        return len(self.source_metadata) > 0

    def load_for_level(self, game: str, level: int) -> Dict[str, Dict[int, Dict]]:
        """
        Load LLM features for all sources for a given game/level.

        This method:
        1. Calls existing load_llm_features_for_level() for each source
        2. Applies per-source subsampling if configured
        3. Returns all data organized by source name

        The alignment function receives data with potentially fewer states
        (if subsampled) but is otherwise unchanged.

        Args:
            game: Game name (e.g., "vgfmri4_chase")
            level: Level number (0-8)

        Returns:
            Dict mapping source_name -> {local_play_idx -> play_data}

            Each play_data dict contains:
                - 'activations': {layer_name: np.ndarray}
                - 'timestamps': np.ndarray of frame indices
                - 'metadata': dict with num_states, etc.
        """
        result = {}

        for src_name, meta in self.source_metadata.items():
            src = meta["config"]
            games_levels = meta["games_levels"]

            # Check if this source has data for this game/level
            if game not in games_levels or level not in games_levels[game]:
                result[src_name] = {}
                continue

            # Load features for this level
            llm_by_local_index = load_llm_features_for_level(
                src.directory, self.subject, game, level, meta["layer_names"]
            )

            # Apply per-source subsampling if configured

            if src.subsample > 1 and llm_by_local_index:
                llm_by_local_index = self._apply_subsampling(
                    llm_by_local_index, src.subsample
                )

            result[src_name] = llm_by_local_index

        return result

    def _apply_subsampling(
        self, llm_by_local_index: Dict[int, Dict], subsample_factor: int
    ) -> Dict[int, Dict]:
        """
        Subsample LLM features by keeping every Nth frame.

        This is applied AFTER loading, BEFORE alignment. The alignment
        function receives fewer states but processes them identically.

        Subsampling behavior:
            - subsample_factor=1: Keep all frames [0, 1, 2, 3, 4, ...]
            - subsample_factor=2: Keep frames [0, 2, 4, 6, 8, ...]
            - subsample_factor=N: Keep frames [0, N, 2N, 3N, ...]

        Args:
            llm_by_local_index: Dict mapping play index to play data
            subsample_factor: Keep every Nth frame

        Returns:
            Same structure with subsampled timestamps and activations
        """
        for local_idx in llm_by_local_index:
            play_data = llm_by_local_index[local_idx]
            n_states = len(play_data["timestamps"])

            # Keep every Nth frame
            subsample_mask = np.arange(n_states) % subsample_factor == 0

            play_data["timestamps"] = play_data["timestamps"][subsample_mask]

            play_data["metadata"]["num_states_original"] = n_states
            play_data["metadata"]["num_states"] = int(subsample_mask.sum())
            play_data["metadata"]["subsample_factor"] = subsample_factor

            for layer_name in list(play_data["activations"].keys()):
                play_data["activations"][layer_name] = play_data["activations"][
                    layer_name
                ][subsample_mask]

        return llm_by_local_index

    def get_all_storage_keys(self) -> List[str]:
        """
        Get all unique storage keys for aligned features.

        These keys are used for:
        - Accumulator initialization
        - Output array naming in saved .npz file

        Returns:
            List of strings in format "{source_name}_{layer_name}"

        Example:
            >>> manager.get_all_storage_keys()
            ['ds32b_layer_1', 'ds32b_layer_8', ..., 'ds685b_layer_1', ...]
        """
        keys = []
        for src_name, meta in self.source_metadata.items():
            for layer_name in meta["layer_names"]:
                keys.append(f"{src_name}_{layer_name}")
        return keys

    def get_n_features(self, src_name: str, layer_name: str) -> int:
        """
        Get feature dimension for a specific source/layer combination.

        Used for pre-allocating zero arrays when a play has no data.

        Args:
            src_name: Source name (e.g., "ds32b")
            layer_name: Layer name (e.g., "layer_8")

        Returns:
            Number of features for this layer, or 1 if unknown
        """
        if src_name not in self.source_metadata:
            return 1
        return self.source_metadata[src_name]["n_features_by_layer"].get(layer_name, 1)

    def get_source_names(self) -> List[str]:
        """
        Get list of successfully initialized source names.

        Returns:
            List of source name strings
        """
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
# Verification Functions
# =============================================================================


def verify_timestamp_alignment(
    model_timestamps: np.ndarray,
    behavioral_timestamps: np.ndarray,
    play_id: str,
    tolerance_ms: float = 100.0,
) -> float:
    if len(model_timestamps) != len(behavioral_timestamps):
        raise ValueError(f"Play {play_id}: Timestamp count mismatch")
    diffs_ms = np.abs(model_timestamps - behavioral_timestamps) * 1000
    max_diff = diffs_ms.max()
    if max_diff > tolerance_ms:
        raise ValueError(f"Play {play_id}: Timestamp mismatch exceeds {tolerance_ms}ms")
    return max_diff


def verify_volume_coverage(
    volume_indices: np.ndarray, valid_mask: np.ndarray, n_volumes: int, play_id: str
) -> dict:
    valid_indices = volume_indices[valid_mask]
    valid_indices_clipped = np.clip(valid_indices, 0, n_volumes - 1)
    states_per_volume = np.bincount(valid_indices_clipped, minlength=n_volumes)
    return {
        "n_empty_volumes": int((states_per_volume == 0).sum()),
        "min_states": int(states_per_volume.min()),
        "max_states": int(states_per_volume.max()),
        "mean_states": float(states_per_volume.mean()),
    }


def verify_partitions(play_levels: np.ndarray, play_partitions: np.ndarray) -> None:
    expected_partitions = play_levels // 3
    if not np.array_equal(play_partitions, expected_partitions):
        raise ValueError("Partition assignment errors")


def verify_no_invalid_values(arrays: dict, context: str) -> None:
    for key, arr in arrays.items():
        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError(f"{context}[{key}] contains NaN/Inf values")


def verify_state_temporal_order(timestamps: np.ndarray, play_id: str) -> bool:
    """
    Check if states are in temporal order.

    Returns:
        True if order is valid, False if there are violations
    """
    diffs = np.diff(timestamps)
    violations = np.where(diffs < 0)[0]
    if len(violations) > 0:
        # Log details about the violations
        logging.warning(
            f"Play {play_id}: {len(violations)} temporal order violation(s) detected"
        )
        for v in violations[:3]:  # Show first 3
            logging.warning(
                f"  ts[{v}]={timestamps[v]:.3f} -> ts[{v + 1}]={timestamps[v + 1]:.3f} (diff={diffs[v]:.3f}s)"
            )
        return False
    return True


def verify_play_temporal_order(all_play_metadata: list) -> None:
    current_game = None
    last_level = -1
    for play in all_play_metadata:
        if play["game_name"] != current_game:
            current_game = play["game_name"]
            last_level = -1
        if play["level_id"] < last_level:
            raise ValueError(f"Play order violation in {current_game}")
        last_level = play["level_id"]


def verify_concatenation_order(
    tr_game_idx: np.ndarray, tr_level_idx: np.ndarray, tr_play_idx: np.ndarray
) -> None:
    for t in range(1, len(tr_game_idx)):
        if tr_game_idx[t] < tr_game_idx[t - 1]:
            raise ValueError(f"Game order violation at TR {t}")
        if (
            tr_game_idx[t] == tr_game_idx[t - 1]
            and tr_level_idx[t] < tr_level_idx[t - 1]
        ):
            raise ValueError(f"Level order violation at TR {t}")


def extract_theory_strings(regressor_doc: dict):
    if not regressor_doc:
        return None, None
    regressors = regressor_doc.get("regressors", {})
    theory_str_data = regressors.get("theory_str", [])
    if not theory_str_data:
        return None, None
    theory_strs = []
    timestamps = []
    for item in theory_str_data:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            theory_strs.append(item[0])
            timestamps.append(item[2])
        elif isinstance(item, (list, tuple)):
            theory_strs.append(item[0])
            timestamps.append(None)
        else:
            theory_strs.append(item)
            timestamps.append(None)
    return theory_strs, np.array(timestamps) if all(
        t is not None for t in timestamps
    ) else None


def extract_behavioral_features(play_doc: dict) -> dict:
    states = play_states(play_doc)
    n_states = len(states)
    key_codes = [273, 274, 276, 275, 32]
    keystates = np.zeros((n_states, 5), dtype=np.float32)
    scores = np.zeros(n_states, dtype=np.float32)
    for i, state in enumerate(states):
        ks = state.get("keystate")
        if ks is not None:
            if isinstance(ks, list):
                for j, key_code in enumerate(key_codes):
                    if key_code < len(ks) and ks[key_code]:
                        keystates[i, j] = 1.0
            elif isinstance(ks, dict):
                for j, key_code in enumerate(key_codes):
                    if ks.get(key_code, False):
                        keystates[i, j] = 1.0
        scores[i] = state.get("score", 0)
    return {
        "keystates": keystates,
        "any_keypress": keystates.any(axis=1).astype(np.float32),
        "scores": scores,
        "score_deltas": np.diff(scores, prepend=scores[0]),
        "time_in_play": np.arange(n_states, dtype=np.float32) / max(n_states - 1, 1),
        "n_states": n_states,
    }


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_preprocessed_bold(preprocessed_dir: Path, subject: str, run: int) -> dict:
    """Load preprocessed voxelwise BOLD from Stage 1."""
    subj_dir = preprocessed_dir / subject
    run_file = subj_dir / f"run-{run:02d}_voxelwise.npz"
    if not run_file.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {run_file}")
    data = np.load(run_file, allow_pickle=True)
    ar1_corrected = bool(data.get("preproc_ar1_corrected", False))
    return {
        "voxel_ts": data["voxel_ts"],
        "voxel_ts_zscored": data["voxel_ts_zscored"],
        "mask": data["mask"],
        "mask_affine": data["mask_affine"],
        "n_voxels": int(data["n_voxels"]),
        "n_volumes": int(data["n_volumes"]),
        "n_volumes_original": int(data.get("n_volumes_original", data["n_volumes"])),
        "tr": float(data["tr"]),
        "ar1_corrected": ar1_corrected,
        "ar1_rho": float(data.get("preproc_ar1_rho", 0.0)),
    }


def find_all_runs(preprocessed_dir: Path, subject: str) -> list:
    """Find all available preprocessed runs for a subject."""
    subj_dir = preprocessed_dir / subject
    if not subj_dir.exists():
        return []
    runs = []
    for f in subj_dir.glob("run-*_voxelwise.npz"):
        run_str = f.stem.split("_")[0]
        runs.append(int(run_str.replace("run-", "")))
    return sorted(runs)


def load_model_features_for_level(
    model_dir: Path,
    subject: str,
    game: str,
    level: int,
    conv_channels: dict = None,
    conv_target_sizes: dict = None,
) -> dict:
    """Load DDQN model features for a specific game/level."""
    level_file = _level_files(_game_directory(model_dir, subject, game)).get(level)
    if level_file is None:
        return {}
    data = np.load(level_file, allow_pickle=True)
    num_plays = int(data["num_plays"])
    model_by_play = {}
    for idx in range(num_plays):
        play_id = str(data[f"play_{idx}_metadata_play_id"])
        activations = {}
        for layer in DDQN_LAYERS:
            key = f"play_{idx}_model_{layer}"
            if key in data:
                act = data[key]
                if conv_channels and conv_target_sizes and layer in conv_channels:
                    act = adaptive_avg_pool_conv(
                        act, conv_channels[layer], conv_target_sizes[layer]
                    )
                activations[layer] = act.astype(np.float32)
        if activations:
            run_id_raw = data[f"play_{idx}_metadata_run_id"]
            model_by_play[play_id] = {
                "activations": activations,
                "timestamps": data[f"play_{idx}_behavioral_timestamps"],
                "metadata": {
                    "play_id": play_id,
                    "subj_id": int(data[f"play_{idx}_metadata_subj_id"]),
                    "game_name": str(data[f"play_{idx}_metadata_game_name"]),
                    "level_id": int(data[f"play_{idx}_metadata_level_id"]),
                    "run_id": int(run_id_raw) if run_id_raw != -1 else None,
                    "num_states": int(data[f"play_{idx}_behavioral_num_states"]),
                },
            }
    data.close()
    return model_by_play


def find_games_and_levels(model_dir: Path, subject: str) -> dict:
    """Discover canonical public names and supported source names without duplicates."""
    from reason_to_play.data.behavior import game_identity

    subj_dir = model_dir / subject
    if not subj_dir.exists():
        return {}
    games_levels = {}
    for game_dir in subj_dir.iterdir():
        if not game_dir.is_dir():
            continue
        try:
            game, cohort = game_identity(game_dir.name)
        except ValueError:
            continue
        identity = f"{cohort}_{game}"
        levels = _level_files(game_dir)
        if levels:
            if identity in games_levels:
                raise ValueError(f"Multiple feature directories for {identity}")
            games_levels[identity] = sorted(levels)
    return games_levels


# =============================================================================
# Alignment Functions
# =============================================================================


def align_features_to_trs(
    activations: np.ndarray,
    volume_indices: np.ndarray,
    valid_mask: np.ndarray,
    n_volumes: int,
    method: str = "average",
) -> np.ndarray:
    n_states, n_features = activations.shape
    aligned = np.zeros((n_volumes, n_features), dtype=np.float32)
    counts = np.zeros(n_volumes, dtype=np.float32)
    for state_idx in range(n_states):
        if not valid_mask[state_idx]:
            continue
        vol = volume_indices[state_idx]
        if vol < 0 or vol >= n_volumes:
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
    return aligned


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
) -> np.ndarray:
    """
    Align LLM features to TRs using their own timestamps.

    LLM features may be subsampled (e.g., every 10th frame), so we align them
    directly by timestamp rather than requiring matching state counts with DDQN.

    Args:
        activations: (n_llm_states, n_features) LLM activations
        llm_timestamps: (n_llm_states,) timestamps for each LLM state
        scan_start_ts: Scan start timestamp from embedded scanner metadata
        tr: Repetition time in seconds
        vol_offset: Starting volume index for this play
        n_vols: Number of volumes for this play
        n_volumes_whitened: Total volumes in whitened data
        ar1_corrected: Whether data has been AR(1) whitened
        method: 'average' or 'last'
        debug: If True, print debug info
        play_id: Play ID for debug messages

    Returns:
        (n_vols, n_features) aligned features
    """
    n_llm_states, n_features = activations.shape

    # Compute onsets relative to scan start
    onsets = llm_timestamps - scan_start_ts

    if debug:
        logging.info(
            f"    [DEBUG LLM {play_id}] n_llm_states={n_llm_states}, n_features={n_features}"
        )
        logging.info(
            f"    [DEBUG LLM {play_id}] llm_timestamps: min={llm_timestamps.min():.3f}, max={llm_timestamps.max():.3f}"
        )
        logging.info(f"    [DEBUG LLM {play_id}] scan_start_ts={scan_start_ts:.3f}")
        logging.info(
            f"    [DEBUG LLM {play_id}] onsets: min={onsets.min():.3f}, max={onsets.max():.3f}"
        )
        logging.info(
            f"    [DEBUG LLM {play_id}] TR={tr}, vol_offset={vol_offset}, n_vols={n_vols}, n_volumes_whitened={n_volumes_whitened}"
        )

    # Compute volume indices in original space
    volume_indices_original = np.round(onsets / tr).astype(int)

    if debug:
        logging.info(
            f"    [DEBUG LLM {play_id}] volume_indices_original: min={volume_indices_original.min()}, max={volume_indices_original.max()}"
        )

    if ar1_corrected:
        # AR(1) whitening removes the first TR
        volume_indices = volume_indices_original - 1
        valid_mask = (volume_indices >= 0) & (volume_indices < n_volumes_whitened)
    else:
        volume_indices = volume_indices_original
        valid_mask = (volume_indices >= 0) & (volume_indices < n_volumes_whitened)

    if debug:
        logging.info(
            f"    [DEBUG LLM {play_id}] After AR1 correction: volume_indices min={volume_indices.min()}, max={volume_indices.max()}"
        )
        logging.info(
            f"    [DEBUG LLM {play_id}] valid_mask: {valid_mask.sum()}/{len(valid_mask)} states valid"
        )

    # Adjust to play's local volume indices
    volume_indices_local = volume_indices - vol_offset

    if debug:
        logging.info(
            f"    [DEBUG LLM {play_id}] volume_indices_local: min={volume_indices_local.min()}, max={volume_indices_local.max()}"
        )
        in_range = (
            (volume_indices_local >= 0) & (volume_indices_local < n_vols) & valid_mask
        )
        logging.info(
            f"    [DEBUG LLM {play_id}] States in play's TR range: {in_range.sum()}/{len(in_range)}"
        )

    # Align to TRs
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

    if debug:
        n_nonzero_trs = (counts > 0).sum()
        logging.info(
            f"    [DEBUG LLM {play_id}] TRs with data: {n_nonzero_trs}/{n_vols} ({100 * n_nonzero_trs / n_vols:.1f}%)"
        )
        logging.info(
            f"    [DEBUG LLM {play_id}] aligned sum={aligned.sum():.3f}, max={aligned.max():.6f}"
        )

    return aligned


def align_hrr_to_trs(
    hrr_embeddings: np.ndarray,
    volume_indices: np.ndarray,
    valid_mask: np.ndarray,
    n_volumes: int,
    method: str = "average",
) -> np.ndarray:
    aligned = align_features_to_trs(
        hrr_embeddings, volume_indices, valid_mask, n_volumes, method
    )
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return aligned / norms


def align_behavioral_to_trs(
    behavioral: dict,
    volume_indices: np.ndarray,
    valid_mask: np.ndarray,
    n_volumes: int,
    method: str = "average",
) -> dict:
    n_states = behavioral["n_states"]
    keystates_aligned = np.zeros((n_volumes, 5), dtype=np.float32)
    any_keypress_aligned = np.zeros(n_volumes, dtype=np.float32)
    scores_aligned = np.zeros(n_volumes, dtype=np.float32)
    score_deltas_aligned = np.zeros(n_volumes, dtype=np.float32)
    time_in_play_aligned = np.zeros(n_volumes, dtype=np.float32)
    counts = np.zeros(n_volumes, dtype=np.float32)
    for state_idx in range(n_states):
        if not valid_mask[state_idx]:
            continue
        vol = volume_indices[state_idx]
        if vol < 0 or vol >= n_volumes:
            continue
        if method == "average":
            keystates_aligned[vol] += behavioral["keystates"][state_idx]
            any_keypress_aligned[vol] += behavioral["any_keypress"][state_idx]
            scores_aligned[vol] += behavioral["scores"][state_idx]
            score_deltas_aligned[vol] += behavioral["score_deltas"][state_idx]
            time_in_play_aligned[vol] += behavioral["time_in_play"][state_idx]
            counts[vol] += 1
        elif method == "last":
            keystates_aligned[vol] = behavioral["keystates"][state_idx]
            any_keypress_aligned[vol] = behavioral["any_keypress"][state_idx]
            scores_aligned[vol] = behavioral["scores"][state_idx]
            score_deltas_aligned[vol] = behavioral["score_deltas"][state_idx]
            time_in_play_aligned[vol] = behavioral["time_in_play"][state_idx]
            counts[vol] = 1
    if method == "average":
        nonzero = counts > 0
        keystates_aligned[nonzero] /= counts[nonzero, np.newaxis]
        any_keypress_aligned[nonzero] /= counts[nonzero]
        scores_aligned[nonzero] /= counts[nonzero]
        score_deltas_aligned[nonzero] /= counts[nonzero]
        time_in_play_aligned[nonzero] /= counts[nonzero]
    return {
        "keystates": keystates_aligned,
        "any_keypress": (any_keypress_aligned > 0.5).astype(np.float32),
        "scores": scores_aligned,
        "score_deltas": score_deltas_aligned,
        "time_in_play": time_in_play_aligned,
    }


def align_regressors_to_states(
    theory_strs: list,
    reg_timestamps: np.ndarray,
    model_timestamps: np.ndarray,
    tolerance_ms: float = 100,
) -> tuple:
    n_model = len(model_timestamps)
    aligned_theories = [None] * n_model
    reg_timestamps = np.array(reg_timestamps)
    model_timestamps = np.array(model_timestamps)
    matched_count = 0
    for reg_idx, reg_ts in enumerate(reg_timestamps):
        diffs = np.abs(model_timestamps - reg_ts)
        nearest_idx = np.argmin(diffs)
        if diffs[nearest_idx] * 1000 < tolerance_ms:
            aligned_theories[nearest_idx] = theory_strs[reg_idx]
            matched_count += 1
    last_valid = None
    for i in range(n_model):
        if aligned_theories[i] is not None:
            last_valid = aligned_theories[i]
        elif last_valid is not None:
            aligned_theories[i] = last_valid
    first_valid = next((t for t in aligned_theories if t is not None), "")
    aligned_theories = [t if t is not None else first_valid for t in aligned_theories]
    return aligned_theories, matched_count


# =============================================================================
# Main Processing
# =============================================================================


def process_subject(
    subject: str,
    preprocessed_dir: Path,
    model_features_dir: Path,
    output_dir: Path = None,
    ez_features_dir: Path = None,
    llm_sources: List[LLMSourceConfig] = None,
    method: str = "average",
    max_level: int = None,
    require_ez: bool = False,
    incremental: bool = False,
    behavior_dir: Path = None,
    regressors_json_path: Path = None,
) -> Path:
    """Process alignment for one subject."""
    if output_dir is None:
        raise ValueError("output_dir is required")
    if behavior_dir is None:
        raise ValueError("Provide behavior_dir with self-contained human JSON files")
    logging.info(f"Processing {subject}")
    logging.info(f"  Alignment method: {method}")
    if max_level is not None:
        logging.info(f"  Max level: {max_level}")
    if require_ez:
        logging.info("  Require EZ: Only processing plays with EfficientZero features")

    subj_str = subject if subject.startswith("sub-") else f"sub-{int(subject):02d}"
    subj_num = int(subj_str.replace("sub-", ""))

    # Incremental mode: skip base file + filter out already-aligned LLM sources
    output_subdir = output_dir / subject
    base_file = output_subdir / "bold-ddqn-theory.npz"
    skip_base_save = False
    skip_hrr_decomp_save = False
    skip_ez_save = False

    if incremental and base_file.is_file():
        with np.load(base_file, allow_pickle=True) as existing_base:
            identity = bind_to_base(base_file, existing_base)
            for existing_path in output_subdir.glob("aligned_*.npz"):
                if existing_path == base_file:
                    continue
                with np.load(existing_path, allow_pickle=True) as existing_features:
                    if not validate_binding(
                        existing_features,
                        existing_base,
                        base_file,
                        base_sha256=str(identity["alignment_base_sha256"]),
                    ):
                        raise ValueError(
                            f"Cannot resume unbound feature archive: {existing_path}"
                        )

    if incremental:
        # Skip base file if it already exists
        if base_file.exists():
            skip_base_save = True
            logging.info("  [incremental] Base file exists, will skip re-saving")

        # Skip HRR decomposed file if it already exists
        hrr_decomp_file = output_subdir / "aligned_hrr_decomposed.npz"
        if hrr_decomp_file.exists():
            skip_hrr_decomp_save = True
            logging.info(
                "  [incremental] HRR decomposed file exists, will skip re-saving"
            )

        # Skip EZ file if it already exists
        ez_out_file = output_subdir / "aligned_ez.npz"
        if ez_out_file.exists():
            skip_ez_save = True
            logging.info("  [incremental] EZ file exists, will skip re-saving")

        # Filter LLM sources to only those without existing output files
        if llm_sources:
            new_sources = []
            for src in llm_sources:
                llm_file = output_subdir / f"aligned_llm_{src.name}.npz"
                if llm_file.exists():
                    logging.info(
                        f"  [incremental] Skipping {src.name} — already aligned"
                    )
                else:
                    new_sources.append(src)

            n_skipped = len(llm_sources) - len(new_sources)
            if n_skipped > 0:
                logging.info(
                    f"  [incremental] {n_skipped} source(s) skipped, {len(new_sources)} remaining"
                )
            llm_sources = new_sources if new_sources else None

        # If nothing to do, return early
        # Check if EZ still needs processing (dir specified but file doesn't exist)
        needs_ez = ez_features_dir is not None and not skip_ez_save
        if skip_base_save and skip_hrr_decomp_save and not llm_sources and not needs_ez:
            logging.info(f"  [incremental] Everything up to date — skipping {subject}")
            return base_file
    runs = find_all_runs(preprocessed_dir, subject)
    if not runs:
        raise ValueError(f"No preprocessed runs found for {subject}")
    logging.info(f"  Found {len(runs)} preprocessed runs: {runs}")

    all_bold = {}
    first_run_data = None
    for run in runs:
        bold_data = load_preprocessed_bold(preprocessed_dir, subject, run)
        all_bold[run] = bold_data
        if first_run_data is None:
            first_run_data = bold_data
        logging.info(
            f"    Run {run}: {bold_data['n_volumes']} vols ({bold_data['n_volumes_original']} orig), "
            f"{bold_data['n_voxels']} voxels, AR1={bold_data['ar1_corrected']} (rho={bold_data['ar1_rho']:.3f})"
        )

    tr = first_run_data["tr"]
    ar1_corrected = first_run_data["ar1_corrected"]
    mask_affine = first_run_data["mask_affine"]

    common_mask = first_run_data["mask"].copy()
    for run, bold_data in all_bold.items():
        if bold_data["mask"].shape != common_mask.shape or not np.allclose(
            bold_data["mask_affine"], mask_affine
        ):
            raise ValueError(f"Run {run} has a different voxel grid/affine")
        if not np.isclose(bold_data["tr"], tr):
            raise ValueError(f"Run {run} has a different TR")
        if bold_data["ar1_corrected"] != ar1_corrected:
            raise ValueError(f"Run {run} has a different AR(1) correction setting")
        common_mask = common_mask & bold_data["mask"]

    n_voxels = int(common_mask.sum())
    if n_voxels == 0:
        raise ValueError("No common brain-mask voxels across runs")
    mask = common_mask
    logging.info(f"  Common mask: {n_voxels} voxels (intersection of all runs)")

    for run, bold_data in all_bold.items():
        run_mask = bold_data["mask"]
        common_indices = np.where(common_mask.ravel())[0]
        run_indices = np.where(run_mask.ravel())[0]
        common_to_run = np.searchsorted(run_indices, common_indices)
        bold_data["voxel_ts_common"] = bold_data["voxel_ts"][common_to_run, :]
        bold_data["voxel_ts_zscored_common"] = bold_data["voxel_ts_zscored"][
            common_to_run, :
        ]
        bold_data["n_voxels_common"] = n_voxels

    runs_by_key = _behavior_api().load_runs(behavior_dir)

    games_levels = find_games_and_levels(model_features_dir, subject)
    if not games_levels:
        raise ValueError(f"No model features found for {subject}")

    def sort_key(g):
        try:
            return GAME_ORDER.index(g)
        except ValueError:
            return len(GAME_ORDER)

    game_list = sorted(games_levels.keys(), key=sort_key)
    logging.info(f"  Found {len(game_list)} games: {game_list}")

    logging.info("  Finding common conv resolution across games...")
    conv_target_sizes = find_common_conv_resolution(
        model_features_dir, subject, game_list, DDQN_CONV_CHANNELS
    )

    conv_layer_dims = {}
    for layer, (h, w) in conv_target_sizes.items():
        n_ch = DDQN_CONV_CHANNELS[layer]
        conv_layer_dims[layer] = n_ch * h * w
        logging.info(
            f"    {layer} output: {n_ch} ch × {h}×{w} = {conv_layer_dims[layer]} features"
        )

    all_plays_by_id = load_canonical_plays(behavior_dir, subj_str)
    all_plays_list = list(all_plays_by_id.values())

    # EfficientZero setup
    ez_available = {}
    ez_new_format = False
    ez_to_beh_mapping = {}
    beh_to_ez_mapping = {}
    has_ez = False
    ez_n_features_by_layer = {}

    if ez_features_dir and ez_features_dir.exists():
        logging.info("  Finding EfficientZero plays...")
        ez_available, ez_new_format = find_available_ez_plays(ez_features_dir, subject)

        if ez_available:
            has_ez = True

            # Get feature dimensions from first available file
            if ez_new_format:
                # Play-ID paths: keys are play_ids (strings)
                first_play_id = list(ez_available.keys())[0]
                first_ez_path, _, _ = ez_available[first_play_id]
            else:
                # Ordinal paths: keys are (game, run, play_idx) tuples
                first_key = list(ez_available.keys())[0]
                (first_ez_path,) = ez_available[first_key]

            first_ez_data = load_ez_traces(first_ez_path, layers=EZ_LAYERS)
            if first_ez_data:
                for layer_name in first_ez_data["layers"]:
                    ez_n_features_by_layer[layer_name] = first_ez_data["activations"][
                        layer_name
                    ].shape[1]
                    logging.info(
                        f"    EZ layer '{layer_name}': {ez_n_features_by_layer[layer_name]} features"
                    )

            # Build a frame-count mapping for ordinal paths
            if not ez_new_format:
                logging.info(
                    "  Building EZ -> Behavioral mapping from ordinal paths..."
                )
                ez_to_beh_mapping = build_ez_to_behavioral_mapping(
                    ez_available, all_plays_list, subject
                )
                beh_to_ez_mapping = {
                    v: k for k, v in ez_to_beh_mapping.items() if v is not None
                }
            else:
                # Paths contain play_ids, so mapping is direct
                logging.info("  Using direct play_id mapping from EZ paths")

    if require_ez and not has_ez:
        raise ValueError(
            f"--require-ez specified but no EZ features found for {subject}"
        )

    # Initialize LLM Feature Manager (Multi-Source Support)
    llm_manager = None
    has_llm = False

    if llm_sources:
        logging.info(
            f"  Initializing LLM feature manager with {len(llm_sources)} source(s)..."
        )
        llm_manager = LLMFeatureManager(llm_sources, subject)
        has_llm = llm_manager.has_sources()

        if not has_llm:
            logging.warning("  No LLM sources successfully initialized")

    has_hrr = False
    regressors_by_play = {}
    hrr_encoder = None
    if regressors_json_path is not None:
        for doc in _behavior_api().iter_regressors(regressors_json_path):
            key = str(doc["play_key"])
            if key in regressors_by_play:
                raise ValueError(f"Duplicate regressor play ID: {key}")
            regressors_by_play[key] = doc
    if regressors_by_play:
        hrr_encoder = HRREncoder(dim=HRR_DIM, seed=HRR_SEED)
        has_hrr = True
        logging.info(f"    HRR encoder initialized: dim={HRR_DIM}")

    # Accumulators
    all_voxel_ts = []
    all_aligned = {layer: [] for layer in DDQN_LAYERS}
    all_ez_aligned = {layer: [] for layer in EZ_LAYERS}
    # LLM Accumulators: One list per (source, layer) combination
    all_llm_aligned = {}
    if llm_manager:
        for key in llm_manager.get_all_storage_keys():
            all_llm_aligned[key] = []
    all_hrr_aligned = []
    all_hrr_sprites_aligned = []
    all_hrr_interactions_aligned = []
    all_hrr_terminations_aligned = []
    all_keystates = []
    all_any_keypress = []
    all_scores = []
    all_score_deltas = []
    all_time_in_play = []
    all_play_metadata = []
    tr_game_idx = []
    tr_level_idx = []
    tr_play_idx = []
    tr_run_idx = []
    game_names = []
    game_n_levels = []
    game_n_volumes = []
    game_boundaries = [0]

    global_play_idx = 0
    total_volumes = 0
    total_states_lost_to_ar1 = 0
    hrr_stats = {"matched": 0, "missing": 0, "state_mismatch": 0}
    ez_stats = {"matched": 0, "missing": 0, "state_mismatch": 0, "skipped": 0}
    # Per-source LLM statistics (uses defaultdict for automatic initialization)
    llm_stats = defaultdict(lambda: {"matched": 0, "missing": 0})
    all_timestamp_diffs = []
    all_coverage_stats = []
    layer_dims = dict(conv_layer_dims)

    for game_idx, game in enumerate(game_list):
        logging.info(f"  Processing {game}")
        game_names.append(game)
        levels = games_levels[game]
        if max_level is not None:
            levels = [item for item in levels if item <= max_level]
        if not levels:
            game_n_levels.append(0)
            game_n_volumes.append(0)
            game_boundaries.append(total_volumes)
            continue

        logging.info(f"    Processing {len(levels)} levels: {levels}")
        game_vol_count = 0

        for level in levels:
            model_by_play = load_model_features_for_level(
                model_features_dir,
                subject,
                game,
                level,
                conv_channels=DDQN_CONV_CHANNELS,
                conv_target_sizes=conv_target_sizes,
            )
            if not model_by_play:
                continue

            # Load LLM features for this level from ALL sources
            # Manager handles per-source subsampling automatically
            llm_by_source = {}
            if llm_manager:
                llm_by_source = llm_manager.load_for_level(game, level)

            # Log with per-source counts
            if llm_by_source:
                llm_counts_str = ", ".join(
                    f"{src}: {len(plays)}" for src, plays in llm_by_source.items()
                )
            else:
                llm_counts_str = "none"
            logging.info(
                f"      Level {level}: {len(model_by_play)} plays (LLM: {llm_counts_str})"
            )

            # Iterate with index to match LLM by position
            for play_idx_in_level, (play_id, model_play) in enumerate(
                model_by_play.items()
            ):
                run_id = model_play["metadata"]["run_id"]
                if run_id is None or run_id not in all_bold:
                    continue
                if play_id not in all_plays_by_id:
                    continue

                play_doc = all_plays_by_id[play_id]

                # Check EZ availability based on format
                if ez_new_format:
                    # Play-ID paths: direct lookup by play_id
                    has_ez_for_play = play_id in ez_available
                    ez_path = ez_available[play_id][0] if has_ez_for_play else None
                else:
                    # Ordinal paths: lookup via frame-count mapping
                    ez_key = beh_to_ez_mapping.get(play_id)
                    has_ez_for_play = ez_key is not None
                    ez_path = ez_available[ez_key][0] if has_ez_for_play else None

                if require_ez and not has_ez_for_play:
                    ez_stats["skipped"] += 1
                    continue

                run_key = (subj_num, run_id)
                if run_key not in runs_by_key:
                    raise ValueError(
                        f"Scanner run missing for {subj_str} run-{run_id:02d}, play {play_id}; "
                        "provide explicit verified scanner-clock metadata or deliberately "
                        "exclude this run from the analysis inputs"
                    )

                run_doc = runs_by_key[run_key]
                bold_run = all_bold[run_id]

                timing = get_play_timing(
                    play_doc,
                    run_doc,
                    tr,
                    bold_run["n_volumes"],
                    bold_run["ar1_corrected"],
                )
                if timing is None:
                    continue

                total_states_lost_to_ar1 += timing.get("n_states_at_tr0", 0)

                n_model_states = model_play["metadata"]["num_states"]
                if n_model_states != len(timing["volume_indices"]):
                    continue

                n_vols = timing["n_volumes"]
                volume_indices = timing["volume_indices"]
                valid_mask = timing["valid_mask"]
                vol_offset = timing["volume_offset"]

                max_ts_diff = verify_timestamp_alignment(
                    model_play["timestamps"], timing["state_timestamps"], play_id
                )
                all_timestamp_diffs.append(max_ts_diff)

                # Skip plays with temporal order violations (data recording issues)
                if not verify_state_temporal_order(timing["state_timestamps"], play_id):
                    logging.warning(
                        f"    Skipping play {play_id} due to temporal order violation"
                    )
                    continue

                coverage = verify_volume_coverage(
                    volume_indices, valid_mask, n_vols, play_id
                )
                all_coverage_stats.append(coverage)

                vol_end = vol_offset + n_vols
                if vol_end > bold_run["n_volumes"]:
                    continue

                all_voxel_ts.append(
                    bold_run["voxel_ts_zscored_common"][:, vol_offset:vol_end]
                )

                for layer in DDQN_LAYERS:
                    if layer in model_play["activations"]:
                        aligned = align_features_to_trs(
                            model_play["activations"][layer],
                            volume_indices,
                            valid_mask,
                            n_vols,
                            method,
                        )
                        all_aligned[layer].append(aligned)
                        if layer not in layer_dims:
                            layer_dims[layer] = aligned.shape[1]

                # Align EfficientZero features
                if has_ez and has_ez_for_play:
                    ez_data = load_ez_traces(
                        ez_path, layers=EZ_LAYERS, n_expected_states=n_model_states
                    )

                    if ez_data:
                        ez_matched_any = False
                        for ez_layer in EZ_LAYERS:
                            if ez_layer in ez_data["activations"]:
                                layer_acts = ez_data["activations"][ez_layer]
                                if layer_acts.shape[0] == n_model_states:
                                    ez_aligned = align_features_to_trs(
                                        layer_acts,
                                        volume_indices,
                                        valid_mask,
                                        n_vols,
                                        method,
                                    )
                                    all_ez_aligned[ez_layer].append(ez_aligned)
                                    ez_matched_any = True
                                else:
                                    n_features = ez_n_features_by_layer.get(
                                        ez_layer, layer_acts.shape[1]
                                    )
                                    all_ez_aligned[ez_layer].append(
                                        np.zeros((n_vols, n_features), dtype=np.float32)
                                    )
                            else:
                                n_features = ez_n_features_by_layer.get(ez_layer, 1)
                                all_ez_aligned[ez_layer].append(
                                    np.zeros((n_vols, n_features), dtype=np.float32)
                                )

                        if ez_matched_any:
                            ez_stats["matched"] += 1
                        else:
                            ez_stats["state_mismatch"] += 1
                    else:
                        ez_stats["missing"] += 1
                        for ez_layer in EZ_LAYERS:
                            n_features = ez_n_features_by_layer.get(ez_layer, 1)
                            all_ez_aligned[ez_layer].append(
                                np.zeros((n_vols, n_features), dtype=np.float32)
                            )
                elif has_ez:
                    ez_stats["missing"] += 1
                    for ez_layer in EZ_LAYERS:
                        n_features = ez_n_features_by_layer.get(ez_layer, 1)
                        all_ez_aligned[ez_layer].append(
                            np.zeros((n_vols, n_features), dtype=np.float32)
                        )

                # Align LLM features for EACH source
                #
                # Iterate over sources and align each layer

                if llm_manager:
                    for src_name, meta in llm_manager.source_metadata.items():
                        llm_by_local_index = llm_by_source.get(src_name, {})

                        if play_idx_in_level in llm_by_local_index:
                            # This play has LLM data from this source
                            llm_play = llm_by_local_index[play_idx_in_level]
                            llm_frame_indices = llm_play["timestamps"]
                            ddqn_timestamps = model_play["timestamps"]
                            n_ddqn_states = len(ddqn_timestamps)

                            # Debug logging for first few plays
                            debug_llm = (global_play_idx < 3) or (
                                game_idx > 0
                                and game_vol_count == 0
                                and play_idx_in_level < 2
                            )

                            if debug_llm:
                                logging.info(
                                    f"    [DEBUG {src_name} {play_id}] "
                                    f"n_llm={len(llm_frame_indices)}, n_ddqn={n_ddqn_states}"
                                )

                            # Map LLM frame indices to DDQN states
                            # (maps LLM frame indices to DDQN state indices)
                            llm_to_ddqn_indices = llm_frame_indices.astype(int)
                            valid_llm_mask = (llm_to_ddqn_indices >= 0) & (
                                llm_to_ddqn_indices < n_ddqn_states
                            )

                            # Get real timestamps for valid LLM states
                            real_timestamps = np.zeros(len(llm_frame_indices))
                            real_timestamps[valid_llm_mask] = ddqn_timestamps[
                                llm_to_ddqn_indices[valid_llm_mask]
                            ]

                            # Align each layer to TRs
                            llm_matched_any = False

                            for layer_idx, layer_name in enumerate(meta["layer_names"]):
                                key = f"{src_name}_{layer_name}"

                                if layer_name in llm_play["activations"]:
                                    llm_activations = llm_play["activations"][
                                        layer_name
                                    ]
                                    valid_activations = llm_activations[valid_llm_mask]
                                    valid_timestamps = real_timestamps[valid_llm_mask]

                                    if len(valid_activations) > 0:
                                        # Align to TRs
                                        llm_aligned = align_llm_features_by_timestamp(
                                            activations=valid_activations,
                                            llm_timestamps=valid_timestamps,
                                            scan_start_ts=timing["scan_start_ts"],
                                            tr=tr,
                                            vol_offset=vol_offset,
                                            n_vols=n_vols,
                                            n_volumes_whitened=bold_run["n_volumes"],
                                            ar1_corrected=bold_run["ar1_corrected"],
                                            method=method,
                                            debug=(debug_llm and layer_idx == 0),
                                            play_id=f"{src_name}_{play_id}",
                                        )
                                    else:
                                        llm_aligned = np.zeros(
                                            (n_vols, llm_activations.shape[1]),
                                            dtype=np.float32,
                                        )

                                    all_llm_aligned[key].append(llm_aligned)
                                    llm_matched_any = True
                                else:
                                    # Layer not in data - append zeros
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
                            # No data for this play from this source
                            # Append zero arrays to maintain alignment
                            for layer_name in meta["layer_names"]:
                                key = f"{src_name}_{layer_name}"
                                n_features = llm_manager.get_n_features(
                                    src_name, layer_name
                                )
                                all_llm_aligned[key].append(
                                    np.zeros((n_vols, n_features), dtype=np.float32)
                                )
                            llm_stats[src_name]["missing"] += 1

                if has_hrr and play_id in regressors_by_play:
                    theory_strs, reg_timestamps = extract_theory_strings(
                        regressors_by_play[play_id]
                    )
                    if theory_strs is not None and reg_timestamps is not None:
                        aligned_theories, match_count = align_regressors_to_states(
                            theory_strs, reg_timestamps, model_play["timestamps"]
                        )
                        if match_count > 0:
                            hrr_embeddings = encode_theory_sequence(
                                hrr_encoder, aligned_theories
                            )
                            all_hrr_aligned.append(
                                align_hrr_to_trs(
                                    hrr_embeddings,
                                    volume_indices,
                                    valid_mask,
                                    n_vols,
                                    method,
                                )
                            )
                            spr, inter, term = encode_theory_sequence_decomposed(
                                hrr_encoder, aligned_theories
                            )
                            all_hrr_sprites_aligned.append(
                                align_hrr_to_trs(
                                    spr, volume_indices, valid_mask, n_vols, method
                                )
                            )
                            all_hrr_interactions_aligned.append(
                                align_hrr_to_trs(
                                    inter, volume_indices, valid_mask, n_vols, method
                                )
                            )
                            all_hrr_terminations_aligned.append(
                                align_hrr_to_trs(
                                    term, volume_indices, valid_mask, n_vols, method
                                )
                            )
                            hrr_stats["matched"] += 1
                        else:
                            hrr_stats["state_mismatch"] += 1
                            all_hrr_aligned.append(
                                np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                            )
                            all_hrr_sprites_aligned.append(
                                np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                            )
                            all_hrr_interactions_aligned.append(
                                np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                            )
                            all_hrr_terminations_aligned.append(
                                np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                            )
                    elif theory_strs is not None and len(theory_strs) == n_model_states:
                        hrr_embeddings = encode_theory_sequence(
                            hrr_encoder, theory_strs
                        )
                        all_hrr_aligned.append(
                            align_hrr_to_trs(
                                hrr_embeddings,
                                volume_indices,
                                valid_mask,
                                n_vols,
                                method,
                            )
                        )
                        spr, inter, term = encode_theory_sequence_decomposed(
                            hrr_encoder, theory_strs
                        )
                        all_hrr_sprites_aligned.append(
                            align_hrr_to_trs(
                                spr, volume_indices, valid_mask, n_vols, method
                            )
                        )
                        all_hrr_interactions_aligned.append(
                            align_hrr_to_trs(
                                inter, volume_indices, valid_mask, n_vols, method
                            )
                        )
                        all_hrr_terminations_aligned.append(
                            align_hrr_to_trs(
                                term, volume_indices, valid_mask, n_vols, method
                            )
                        )
                        hrr_stats["matched"] += 1
                    else:
                        hrr_stats["missing"] += 1
                        all_hrr_aligned.append(
                            np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                        )
                        all_hrr_sprites_aligned.append(
                            np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                        )
                        all_hrr_interactions_aligned.append(
                            np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                        )
                        all_hrr_terminations_aligned.append(
                            np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                        )
                elif has_hrr:
                    hrr_stats["missing"] += 1
                    all_hrr_aligned.append(
                        np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                    )
                    all_hrr_sprites_aligned.append(
                        np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                    )
                    all_hrr_interactions_aligned.append(
                        np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                    )
                    all_hrr_terminations_aligned.append(
                        np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                    )
                else:
                    all_hrr_aligned.append(
                        np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                    )
                    all_hrr_sprites_aligned.append(
                        np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                    )
                    all_hrr_interactions_aligned.append(
                        np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                    )
                    all_hrr_terminations_aligned.append(
                        np.zeros((n_vols, HRR_DIM), dtype=np.float32)
                    )

                beh_features = extract_behavioral_features(play_doc)
                if beh_features and beh_features["n_states"] == n_model_states:
                    beh_aligned = align_behavioral_to_trs(
                        beh_features, volume_indices, valid_mask, n_vols, method
                    )
                    all_keystates.append(beh_aligned["keystates"])
                    all_any_keypress.append(beh_aligned["any_keypress"])
                    all_scores.append(beh_aligned["scores"])
                    all_score_deltas.append(beh_aligned["score_deltas"])
                    all_time_in_play.append(beh_aligned["time_in_play"])
                else:
                    all_keystates.append(np.zeros((n_vols, 5), dtype=np.float32))
                    all_any_keypress.append(np.zeros(n_vols, dtype=np.float32))
                    all_scores.append(np.full(n_vols, np.nan, dtype=np.float32))
                    all_score_deltas.append(np.zeros(n_vols, dtype=np.float32))
                    all_time_in_play.append(np.linspace(0, 1, n_vols, dtype=np.float32))

                # Track which LLM sources have data for this play
                llm_sources_for_play = []
                if llm_manager:
                    for src_name in llm_manager.get_source_names():
                        if play_idx_in_level in llm_by_source.get(src_name, {}):
                            llm_sources_for_play.append(src_name)

                all_play_metadata.append(
                    {
                        "play_id": play_id,
                        "game_name": game,
                        "game_idx": game_idx,
                        "level_id": level,
                        "run_id": run_id,
                        "n_volumes": n_vols,
                        "n_states": n_model_states,
                        "has_ez": has_ez_for_play,
                        "has_llm": len(llm_sources_for_play) > 0,
                        "llm_sources": llm_sources_for_play,
                    }
                )

                for _ in range(n_vols):
                    tr_game_idx.append(game_idx)
                    tr_level_idx.append(level)
                    tr_play_idx.append(global_play_idx)
                    tr_run_idx.append(run_id)

                game_vol_count += n_vols
                total_volumes += n_vols
                global_play_idx += 1

        game_n_levels.append(len(levels))
        game_n_volumes.append(game_vol_count)
        game_boundaries.append(total_volumes)
        logging.info(f"    {game}: {len(levels)} levels, {game_vol_count} TRs")

    if global_play_idx == 0:
        raise ValueError(f"No plays processed for {subject}")

    logging.info(f"  Total: {global_play_idx} plays, {total_volumes} volumes")
    if ar1_corrected:
        logging.info(f"  States lost to AR(1) (at TR 0): {total_states_lost_to_ar1}")

    voxel_ts_concat = np.concatenate(all_voxel_ts, axis=1)
    concatenated = {
        layer: np.vstack(all_aligned[layer])
        for layer in DDQN_LAYERS
        if all_aligned[layer]
    }
    for layer, arr in concatenated.items():
        logging.info(f"    {layer}: {arr.shape}")

    # EZ features
    ez_concatenated = {}
    if has_ez:
        for ez_layer in EZ_LAYERS:
            if all_ez_aligned[ez_layer]:
                ez_concatenated[ez_layer] = np.vstack(all_ez_aligned[ez_layer])
                logging.info(f"    ez_{ez_layer}: {ez_concatenated[ez_layer].shape}")
        logging.info(
            f"    EZ stats: matched={ez_stats['matched']}, missing={ez_stats['missing']}, state_mismatch={ez_stats['state_mismatch']}"
        )

    # LLM Features: Concatenate for each (source, layer) combination
    llm_concatenated = {}

    if llm_manager:
        for key in llm_manager.get_all_storage_keys():
            if all_llm_aligned[key]:
                llm_concatenated[key] = np.vstack(all_llm_aligned[key])
                logging.info(f"    llm_{key}: {llm_concatenated[key].shape}")

        # Log per-source statistics
        for src_name in llm_manager.get_source_names():
            stats = llm_stats[src_name]
            logging.info(
                f"    LLM '{src_name}' stats: "
                f"matched={stats['matched']}, missing={stats['missing']}"
            )

    hrr_concat = (
        np.vstack(all_hrr_aligned)
        if all_hrr_aligned
        else np.zeros((total_volumes, HRR_DIM))
    )
    logging.info(f"    hrr: {hrr_concat.shape}")

    # Decomposed HRR sub-vectors
    hrr_sprites_concat = (
        np.vstack(all_hrr_sprites_aligned)
        if all_hrr_sprites_aligned
        else np.zeros((total_volumes, HRR_DIM))
    )
    hrr_interactions_concat = (
        np.vstack(all_hrr_interactions_aligned)
        if all_hrr_interactions_aligned
        else np.zeros((total_volumes, HRR_DIM))
    )
    hrr_terminations_concat = (
        np.vstack(all_hrr_terminations_aligned)
        if all_hrr_terminations_aligned
        else np.zeros((total_volumes, HRR_DIM))
    )
    logging.info(
        f"    hrr decomposed: sprites={hrr_sprites_concat.shape}, interactions={hrr_interactions_concat.shape}, terminations={hrr_terminations_concat.shape}"
    )

    keystates_concat = np.vstack(all_keystates)
    any_keypress_concat = np.concatenate(all_any_keypress)
    scores_concat = np.concatenate(all_scores)
    score_deltas_concat = np.concatenate(all_score_deltas)
    time_in_play_concat = np.concatenate(all_time_in_play)

    play_ids = np.array([m["play_id"] for m in all_play_metadata], dtype="U24")
    play_game_idx = np.array([m["game_idx"] for m in all_play_metadata], dtype=int)
    play_levels = np.array([m["level_id"] for m in all_play_metadata], dtype=int)
    play_n_volumes = np.array([m["n_volumes"] for m in all_play_metadata], dtype=int)
    play_boundaries = np.concatenate([[0], np.cumsum(play_n_volumes)])
    play_partitions = play_levels // 3
    play_has_ez = np.array([m["has_ez"] for m in all_play_metadata], dtype=bool)
    play_has_llm = np.array(
        [m.get("has_llm", False) for m in all_play_metadata], dtype=bool
    )
    # Per-play list of which LLM sources have data (list of lists -> object array)
    play_llm_sources = [m.get("llm_sources", []) for m in all_play_metadata]
    time_in_experiment = np.arange(total_volumes, dtype=np.float32) / max(
        total_volumes - 1, 1
    )

    verify_play_temporal_order(all_play_metadata)
    verify_partitions(play_levels, play_partitions)
    verify_no_invalid_values(concatenated, "Features")
    verify_no_invalid_values({"hrr": hrr_concat}, "HRR")
    if ez_concatenated:
        verify_no_invalid_values(ez_concatenated, "EZ")
    if llm_concatenated:
        verify_no_invalid_values(llm_concatenated, "LLM")
    verify_concatenation_order(
        np.array(tr_game_idx), np.array(tr_level_idx), np.array(tr_play_idx)
    )
    logging.info("  Alignment verification: ✓ All checks passed")

    # Save: base file (shared data) + one file per LLM source
    output_subdir = output_dir / subject
    output_subdir.mkdir(parents=True, exist_ok=True)

    # --- Base save dict: everything except LLM features ---
    save_dict = {
        "voxel_ts": voxel_ts_concat,
        "n_voxels": n_voxels,
        "mask": mask,
        "mask_affine": mask_affine,
        **{f"{layer}_aligned": arr for layer, arr in concatenated.items()},
        "has_ez_data": has_ez,
        "ez_new_format": ez_new_format,
        "ez_layers": np.array(EZ_LAYERS, dtype="U64")
        if has_ez
        else np.array([], dtype="U64"),
        "ez_representation_layers": np.array(EZ_REPRESENTATION_LAYERS, dtype="U64"),
        "ez_value_policy_layers": np.array(EZ_VALUE_POLICY_LAYERS, dtype="U64"),
        "ez_dynamics_reward_layers": np.array(EZ_DYNAMICS_REWARD_LAYERS, dtype="U64"),
        "has_llm_data": has_llm,
        "llm_source_names": np.array(
            llm_manager.get_source_names() if llm_manager else [], dtype="U64"
        ),
        "hrr_aligned": hrr_concat,
        "hrr_dim": HRR_DIM,
        "hrr_seed": HRR_SEED,
        "hrr_matched_plays": hrr_stats["matched"],
        "hrr_missing_plays": hrr_stats["missing"],
        "keystates": keystates_concat,
        "any_keypress": any_keypress_concat,
        "scores": scores_concat,
        "score_deltas": score_deltas_concat,
        "time_in_play": time_in_play_concat,
        "time_in_experiment": time_in_experiment,
        "tr_game_idx": np.array(tr_game_idx, dtype=int),
        "tr_level_idx": np.array(tr_level_idx, dtype=int),
        "tr_play_idx": np.array(tr_play_idx, dtype=int),
        "tr_run_idx": np.array(tr_run_idx, dtype=int),
        "play_ids": play_ids,
        "play_game_idx": play_game_idx,
        "play_levels": play_levels,
        "play_partitions": play_partitions,
        "play_n_volumes": play_n_volumes,
        "play_boundaries": play_boundaries,
        "play_has_ez": play_has_ez,
        "play_has_llm": play_has_llm,
        "play_llm_sources": np.array(play_llm_sources, dtype=object),
        "game_names": np.array(game_names, dtype="U32"),
        "game_n_levels": np.array(game_n_levels, dtype=int),
        "game_n_volumes": np.array(game_n_volumes, dtype=int),
        "game_boundaries": np.array(game_boundaries, dtype=int),
        "n_volumes": total_volumes,
        "n_plays": len(all_play_metadata),
        "n_games": len(game_names),
        "subject": subject,
        "tr": tr,
        "alignment_method": method,
        "max_level": max_level if max_level is not None else -1,
        "require_ez": require_ez,
        "ar1_corrected": ar1_corrected,
        "states_lost_to_ar1": total_states_lost_to_ar1,
        "ddqn_layers": np.array(list(concatenated.keys()), dtype="U16"),
        "ddqn_layer_dims": np.array(
            [layer_dims.get(item, -1) for item in concatenated.keys()], dtype=int
        ),
        "conv_layer_names": np.array(list(DDQN_CONV_CHANNELS.keys()), dtype="U16"),
        "conv_channels": np.array(
            [DDQN_CONV_CHANNELS[item] for item in DDQN_CONV_CHANNELS.keys()], dtype=int
        ),
        "conv_target_sizes": np.array(
            [
                conv_target_sizes.get(item, (-1, -1))
                for item in DDQN_CONV_CHANNELS.keys()
            ]
        ),
        "keystate_columns": np.array(
            ["up", "down", "left", "right", "space"], dtype="U8"
        ),
        "has_behavioral_data": True,
        "has_hrr_data": has_hrr,
        "ez_matched_plays": ez_stats["matched"],
        "ez_missing_plays": ez_stats["missing"],
        "ez_skipped_plays": ez_stats["skipped"],
        "max_timestamp_diff_ms": max(all_timestamp_diffs) if all_timestamp_diffs else 0,
        "mean_timestamp_diff_ms": float(np.mean(all_timestamp_diffs))
        if all_timestamp_diffs
        else 0,
        "total_empty_volumes": sum(c["n_empty_volumes"] for c in all_coverage_stats)
        if all_coverage_stats
        else 0,
        "mean_states_per_volume": float(
            np.mean([c["mean_states"] for c in all_coverage_stats])
        )
        if all_coverage_stats
        else 0,
    }

    # NOTE: EZ feature arrays are saved in a separate aligned_ez.npz file (not in base)
    # EZ metadata (has_ez_data, ez_layers, play_has_ez, stats) is still in base save_dict above

    # Save base file (BOLD + DDQN + HRR + EZ + behavioral + metadata, NO LLM)
    base_file = output_subdir / "bold-ddqn-theory.npz"
    if skip_base_save:
        logging.info("  [incremental] Skipping base file save (already exists)")
    else:
        np.savez_compressed(base_file, **save_dict)
        logging.info(
            f"  Saved base: {base_file.name} ({base_file.stat().st_size / 1024**2:.1f} MB)"
        )

    # Features are computed in save_dict's sample order, including during resume.
    with np.load(base_file, allow_pickle=True) as stored_base:
        if sample_order_sha256(stored_base) != sample_order_sha256(save_dict):
            raise ValueError(
                "Existing base has a different sample order; use a new output directory"
            )
        binding = bind_to_base(base_file, stored_base)

    # Save per-source LLM files
    if llm_manager and llm_concatenated:
        for src_name, meta in llm_manager.source_metadata.items():
            src = meta["config"]
            prefix = f"llm_{src_name}"

            # Build per-source save dict with only this source's layers
            llm_save_dict = dict(binding)
            for layer_name in meta["layer_names"]:
                key = f"{src_name}_{layer_name}"
                if key in llm_concatenated:
                    llm_save_dict[f"llm_{key}_aligned"] = llm_concatenated[key]

            # Add source metadata
            llm_save_dict[f"{prefix}_layers"] = np.array(
                meta["layer_names"], dtype="U32"
            )
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

    # Save decomposed HRR file (sprites, interactions, terminations)
    if has_hrr:
        hrr_decomp_file = output_subdir / "aligned_hrr_decomposed.npz"
        if skip_hrr_decomp_save:
            logging.info(
                "  [incremental] Skipping HRR decomposed file save (already exists)"
            )
        else:
            hrr_decomp_dict = {
                **binding,
                "hrr_sprites_aligned": hrr_sprites_concat,
                "hrr_interactions_aligned": hrr_interactions_concat,
                "hrr_terminations_aligned": hrr_terminations_concat,
                "hrr_dim": HRR_DIM,
                "hrr_seed": HRR_SEED,
                "hrr_matched_plays": hrr_stats["matched"],
                "hrr_missing_plays": hrr_stats["missing"],
            }
            np.savez_compressed(hrr_decomp_file, **hrr_decomp_dict)
            logging.info(
                f"  Saved HRR decomposed: {hrr_decomp_file.name} ({hrr_decomp_file.stat().st_size / 1024**2:.1f} MB)"
            )

    # Save EZ features as separate file (like LLM sources)
    if has_ez and ez_concatenated:
        ez_out_path = output_subdir / "aligned_ez.npz"
        if skip_ez_save:
            logging.info("  [incremental] Skipping EZ file save (already exists)")
        else:
            ez_save_dict = dict(binding)
            for ez_layer, arr in ez_concatenated.items():
                safe_key = f"ez_{ez_layer.replace('.', '_')}_aligned"
                ez_save_dict[safe_key] = arr
            ez_save_dict["ez_n_features_by_layer"] = np.array(
                [(layer, ez_n_features_by_layer.get(layer, -1)) for layer in EZ_LAYERS],
                dtype=[("layer", "U64"), ("n_features", "i4")],
            )
            ez_save_dict["ez_layers"] = np.array(EZ_LAYERS, dtype="U64")
            ez_save_dict["ez_representation_layers"] = np.array(
                EZ_REPRESENTATION_LAYERS, dtype="U64"
            )
            ez_save_dict["ez_value_policy_layers"] = np.array(
                EZ_VALUE_POLICY_LAYERS, dtype="U64"
            )
            ez_save_dict["ez_dynamics_reward_layers"] = np.array(
                EZ_DYNAMICS_REWARD_LAYERS, dtype="U64"
            )
            ez_save_dict["ez_matched_plays"] = ez_stats["matched"]
            ez_save_dict["ez_missing_plays"] = ez_stats["missing"]
            ez_save_dict["ez_skipped_plays"] = ez_stats["skipped"]
            ez_save_dict["ez_new_format"] = ez_new_format
            np.savez_compressed(ez_out_path, **ez_save_dict)
            logging.info(
                f"  Saved EZ: {ez_out_path.name} ({ez_out_path.stat().st_size / 1024**2:.1f} MB)"
            )

    logging.info(f"  All files saved to: {output_subdir}")
    return base_file


# =============================================================================
# CLI Argument Parsing Helpers
# =============================================================================


def parse_llm_source_arg(source_str: str) -> LLMSourceConfig:
    """
    Parse an LLM source specification string from command line.

    Format:
        "name=<name>,dir=<path>[,subsample=<int>][,layers=<l1>:<l2>:<l3>:...]"

    Required fields:
        - name: Unique identifier for this source
        - dir: Path to feature directory

    Optional fields:
        - subsample: Integer >= 1 (default: 1)
        - layers: Colon-separated layer indices (default: use LLM_LAYERS global)

    Examples:
        "name=deepseek32b,dir=./llm_features/32B"
        "name=deepseek32b_sub2,dir=./llm_features/32B,subsample=2"
        "name=deepseek685b,dir=./llm_features/685B,layers=1:8:16:24:32"
    """
    # Parse key=value pairs from comma-separated string
    parts = {}
    for part in source_str.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            parts[key.strip()] = value.strip()

    # Validate required fields
    if "name" not in parts:
        raise ValueError(
            f"LLM source must specify 'name': {source_str}\n"
            f"Expected format: name=<name>,dir=<path>[,subsample=<int>][,layers=<l1>:<l2>:...]"
        )
    if "dir" not in parts:
        raise ValueError(
            f"LLM source must specify 'dir': {source_str}\n"
            f"Expected format: name=<name>,dir=<path>[,subsample=<int>][,layers=<l1>:<l2>:...]"
        )

    # Parse optional fields with defaults
    subsample = int(parts.get("subsample", 1))

    layers = None
    if "layers" in parts:
        layers = [int(x) for x in parts["layers"].split(":")]

    # Construct and return config object
    return LLMSourceConfig(
        name=parts["name"],
        directory=Path(parts["dir"]),
        subsample=subsample,
        layers=layers,
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Align model features to preprocessed voxelwise BOLD"
    )
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument(
        "--preprocessed-dir", required=True, help="Directory with preprocessed BOLD"
    )
    parser.add_argument(
        "--model-features-dir", required=True, help="Directory with DDQN features"
    )
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        required=True,
        help="Canonical human recordings root or downloaded dataset root",
    )
    parser.add_argument(
        "--regressors-json", type=Path, help="Canonical EMPA regressors .json.gz"
    )
    parser.add_argument(
        "--output-dir", default="./workdir/aligned_data", help="Output directory"
    )
    parser.add_argument(
        "--ez-features-dir", default=None, help="Directory with EfficientZero features"
    )
    # LLM Source Arguments (Multi-Source Support)
    parser.add_argument(
        "--llm-source",
        action="append",
        dest="llm_sources_raw",
        default=[],
        metavar="SPEC",
        help=(
            "LLM source specification. Can be repeated for multiple sources.\n"
            'Format: "name=<name>,dir=<path>[,subsample=<int>][,layers=<l1>:<l2>:...]"\n'
            "Examples:\n"
            '  --llm-source "name=ds32b,dir=./llm_32b"\n'
            '  --llm-source "name=ds32b_sub2,dir=./llm_32b,subsample=2"\n'
            '  --llm-source "name=ds685b,dir=./llm_685b,layers=1:8:16:24:32:40:48:56:64"'
        ),
    )
    parser.add_argument(
        "--llm-features-dir",
        default=None,
        help="[DEPRECATED] Single LLM directory. Use --llm-source instead.",
    )
    parser.add_argument(
        "--method",
        default="average",
        choices=["average", "last"],
        help="Alignment method",
    )
    parser.add_argument(
        "--max-level", type=int, default=None, help="Maximum level to include"
    )
    parser.add_argument(
        "--require-ez", action="store_true", help="Only process plays with EZ features"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip bold-ddqn-theory.npz if it exists; skip LLM sources that already have output files",
    )

    args = parser.parse_args()

    # Build LLM sources list from arguments
    llm_sources = []

    # Named LLM sources: --llm-source (repeatable)
    if args.llm_sources_raw:
        for src_str in args.llm_sources_raw:
            try:
                llm_sources.append(parse_llm_source_arg(src_str))
            except ValueError as e:
                logging.error(f"Invalid --llm-source argument: {e}")
                raise
        logging.info(f"Configured {len(llm_sources)} LLM source(s)")

    # Single unnamed source: --llm-features-dir (deprecated)
    elif args.llm_features_dir:
        logging.warning(
            "--llm-features-dir is deprecated and will be removed in a future version. "
            "Use --llm-source instead."
        )
        # Derive a name from the directory name
        dir_path = Path(args.llm_features_dir)
        derived_name = dir_path.name.replace("-", "_").replace(".", "_")

        llm_sources.append(
            LLMSourceConfig(
                name=derived_name,
                directory=dir_path,
                subsample=1,
                layers=None,  # Use default
            )
        )
        logging.info(
            f"  Converted deprecated arg to: "
            f"--llm-source name={derived_name},dir={dir_path}"
        )

    process_subject(
        subject=args.subject,
        preprocessed_dir=Path(args.preprocessed_dir),
        model_features_dir=Path(args.model_features_dir),
        output_dir=Path(args.output_dir),
        ez_features_dir=Path(args.ez_features_dir) if args.ez_features_dir else None,
        llm_sources=llm_sources if llm_sources else None,
        method=args.method,
        max_level=args.max_level,
        require_ez=args.require_ez,
        incremental=args.incremental,
        behavior_dir=args.behavior_dir,
        regressors_json_path=args.regressors_json,
    )
    logging.info(f"Completed {args.subject}")
