#!/usr/bin/env python3
"""
Encoding model for DDQN features predicting voxelwise BOLD responses.

KEY CHANGE in v4: PCA is done BEFORE lagging, not after. 
Criterion: if n_features_raw * 4 > n_train, apply PCA to raw features.

This script fits a banded ridge regression model to predict voxel-wise BOLD
responses from DDQN model features, with nuisance regressors for game/level
identity (and optionally button presses and time).

Features:
- Lagged features (2-5 TRs into the PAST) to capture HRF delay
- Banded ridge regression with separate regularization for feature groups:
  * Band 1: Main features (DDQN fc1 or HRR)
  * Band 2: Game + Level identity 
  * (Optional) Band 3: Button presses
  * (Optional) Band 4: Time (time_in_play + time_in_experiment)
- Leave-one-partition-out cross-validation (following Tomov et al.)
- Separate evaluation of each band's contribution
- Intelligent batching for voxelwise data (~60k targets)
- Optional --shuffle flag for control analysis (shuffles features within games)

Usage:
    python encoding_model_v4.py \
        --subject sub-01 \
        --data-dir ./workdir/aligned_data \
        --layer fc1 \
        --output-dir ./workdir/encoding_results
    
    # With nuisance bands (button presses + time):
    python encoding_model_v4.py \
        --subject sub-01 \
        --data-dir ./workdir/aligned_data \
        --layer fc1 \
        --output-dir ./workdir/encoding_results \
        --include-nuisance-bands

    # Shuffle control analysis:
    python encoding_model_v4.py \
        --subject sub-01 \
        --data-dir ./workdir/aligned_data \
        --layer fc1 \
        --output-dir ./workdir/encoding_results \
        --shuffle

Output:
    {output_dir}/{subject}/encoding_results_{layer}.npz (or _shuffled.npz) containing:
    - performances_full: (n_voxels, n_partitions) correlation per partition/voxel
    - performances_mean_full: (n_voxels,) mean across partitions
    - performances_{band}: per-band performance arrays
    - partition_ids: which partitions were tested
    - mask, mask_affine: for reconstruction to NIfTI
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from reason_to_play.analysis.neural.alignment import (
    external_binding,
    file_sha256,
    released_llm_path,
    validate_binding,
    validate_feature_coverage,
)

# Himalaya imports
from himalaya.ridge import GroupRidgeCV
from himalaya.scoring import correlation_score
from himalaya.backend import set_backend

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# =============================================================================
# Configuration
# =============================================================================

LAGS = [2, 3, 4, 5]  # TRs into the PAST (HRF peaks ~4-6s after neural activity)
N_LAGS = len(LAGS)

# Default bands (always included)
DEFAULT_BANDS = ["main"]

# Optional nuisance bands
NUISANCE_BANDS = ["button", "time", "identity"]

# Feature budget: total features after lagging should be < this fraction of n_train
FEATURE_BUDGET_RATIO = 0.95


# =============================================================================
# Feature Engineering
# =============================================================================


def create_lagged_features(
    features: np.ndarray, play_boundaries: np.ndarray, lags: list = LAGS
) -> np.ndarray:
    """
    Create lagged versions of features, using PAST TRs to predict current BOLD.

    For HRF modeling, we want features from the PAST to predict current BOLD:
    - BOLD[t] is predicted by features[t-2], features[t-3], features[t-4], features[t-5]
    - This captures the ~4-6s delay of the hemodynamic response

    At play boundaries, we pad with the first valid value (no bleeding across plays).
    """
    n_volumes, n_features = features.shape
    n_lags = len(lags)

    lagged = np.zeros((n_volumes, n_features * n_lags), dtype=np.float32)
    n_plays = len(play_boundaries) - 1

    for play_idx in range(n_plays):
        start = play_boundaries[play_idx]
        end = play_boundaries[play_idx + 1]

        for lag_idx, lag in enumerate(lags):
            feat_start = lag_idx * n_features
            feat_end = feat_start + n_features

            for t in range(start, end):
                src_t = t - lag
                if src_t >= start:
                    lagged[t, feat_start:feat_end] = features[src_t]
                else:
                    lagged[t, feat_start:feat_end] = features[start]

    return lagged


def create_onehot(indices: np.ndarray, n_classes: int) -> np.ndarray:
    """Create one-hot encoding from integer indices."""
    onehot = np.zeros((len(indices), n_classes), dtype=np.float32)
    onehot[np.arange(len(indices)), indices] = 1.0
    return onehot


def validate_aligned_data(data: dict, layer: str, subject: str | None = None) -> None:
    """Reject mismatched row identities before constructing folds or lagged data."""
    if subject is not None and "subject" in data:
        actual = np.asarray(data["subject"])
        if actual.shape != () or str(actual.item()) != subject:
            raise ValueError(f"Aligned subject metadata does not match {subject}")
    target_key = "voxel_ts" if "voxel_ts" in data else "parcel_ts"
    targets = np.asarray(data[target_key])
    if targets.ndim != 2 or not all(targets.shape):
        raise ValueError(f"{target_key} must have targets × volumes shape")
    n_volumes = targets.shape[1]
    feature_key = f"{layer}_aligned"
    if feature_key not in data:
        raise ValueError(f"Layer '{layer}' not found in aligned inputs")
    features = np.asarray(data[feature_key])
    if features.ndim != 2 or features.shape[0] != n_volumes or features.shape[1] == 0:
        raise ValueError(
            f"{feature_key} does not match the BOLD volume count or feature shape"
        )
    for key in ("n_volumes",):
        if key in data and int(data[key]) != n_volumes:
            raise ValueError(f"{key} disagrees with the BOLD volume count")
    for key in ("tr_level_idx", "tr_game_idx", "tr_play_idx"):
        value = np.asarray(data[key])
        if value.shape != (n_volumes,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{key} must contain one integer per BOLD volume")
        if (value < 0).any():
            raise ValueError(f"{key} contains negative indices")
    boundaries = np.asarray(data["play_boundaries"])
    if (
        boundaries.ndim != 1
        or len(boundaries) < 2
        or not np.issubdtype(boundaries.dtype, np.integer)
        or boundaries[0] != 0
        or boundaries[-1] != n_volumes
        or (np.diff(boundaries) <= 0).any()
    ):
        raise ValueError(
            "play_boundaries must partition every BOLD volume exactly once"
        )
    lengths = np.diff(boundaries)
    n_plays = len(lengths)
    for key in ("play_levels", "play_game_idx"):
        if np.asarray(data[key]).shape != (n_plays,):
            raise ValueError(f"{key} does not match play boundaries")
    if not np.array_equal(data["tr_play_idx"], np.repeat(np.arange(n_plays), lengths)):
        raise ValueError("TR play identities disagree with play boundaries")
    for tr_key, play_key in (
        ("tr_level_idx", "play_levels"),
        ("tr_game_idx", "play_game_idx"),
    ):
        if not np.array_equal(data[tr_key], np.repeat(data[play_key], lengths)):
            raise ValueError(
                f"{tr_key} disagrees with per-play identity; lagging could cross CV folds"
            )
    if "play_n_volumes" in data and not np.array_equal(data["play_n_volumes"], lengths):
        raise ValueError("play_n_volumes disagrees with play boundaries")
    if np.any(data["tr_game_idx"] >= int(data["n_games"])):
        raise ValueError("Game indices exceed n_games")
    n_targets_key = "n_voxels" if target_key == "voxel_ts" else "n_parcels"
    if n_targets_key in data and int(data[n_targets_key]) != targets.shape[0]:
        raise ValueError(f"{n_targets_key} disagrees with BOLD targets")
    if (
        target_key == "voxel_ts"
        and "mask" in data
        and int(np.asarray(data["mask"], dtype=bool).sum()) != targets.shape[0]
    ):
        raise ValueError("BOLD target count disagrees with voxel mask")


def merge_feature_sidecar(data: dict, sidecar: dict, path: Path) -> dict:
    """Feature sidecars may add arrays, but may not replace alignment identities."""
    for key in data.keys() & sidecar.keys():
        left, right = np.asarray(data[key]), np.asarray(sidecar[key])
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(
                f"Feature sidecar {path} conflicts with base alignment field {key}"
            )
    return {**data, **sidecar}


def encoding_input_paths(subject, data_dir, layer, base_data=None, feature_files=None):
    """Resolve explicit paths, retaining the historical subject-directory layout."""
    if base_data is None:
        if data_dir is None:
            raise ValueError("Provide --base-data or --data-dir")
        root = Path(data_dir)
        if (root / "analysis/neural/inputs").is_dir():
            root = root / "analysis/neural/inputs"
        clean_base = root / subject / "bold-ddqn-theory.npz"
        base_data = (
            clean_base if clean_base.is_file() else root / subject / "aligned_data.npz"
        )
    paths = [Path(base_data)]
    if feature_files is not None:
        paths.extend(Path(path) for path in feature_files)
    else:
        sidecar = None
        if layer.startswith("llm_"):
            match = re.fullmatch(r"llm_(.+)_layer_\d+", layer)
            if match:
                sidecar = f"aligned_llm_{match.group(1)}.npz"
        elif layer.startswith("ez_") or layer == "ez":
            sidecar = "aligned_ez.npz"
        elif layer.startswith("hrr_"):
            sidecar = "aligned_hrr_decomposed.npz"
        if sidecar:
            clean = released_llm_path(layer, subject)
            if layer.startswith("hrr_"):
                clean = Path("model-features/theory/hrr-decomposed") / f"{subject}.npz"
            candidate = paths[0].parent.parent / clean if clean is not None else None
            if candidate is None or not candidate.is_file():
                candidate = paths[0].parent / sidecar
            if candidate.exists():
                paths.append(candidate)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_aligned_data(paths, subject, layer, *, require_binding=True):
    with np.load(paths[0], allow_pickle=True) as source:
        data = dict(source)
    base = data
    binding_status = []
    base_digest = file_sha256(paths[0]) if len(paths) > 1 else None
    for path in paths[1:]:
        with np.load(path, allow_pickle=True) as source:
            sidecar = dict(source)
        verified = validate_binding(sidecar, base, paths[0], base_sha256=base_digest)
        association = external_binding(path)
        coverage = None
        if association is not None:
            validate_binding(association, base, paths[0], base_sha256=base_digest)
            if "feature_coverage" in association:
                coverage = validate_feature_coverage(
                    association["feature_coverage"], base
                )
            verified = True
        if not verified:
            if require_binding:
                raise ValueError(
                    f"Feature sidecar lacks a recorded base/sample-order binding: {path}"
                )
            logging.warning(
                "Unbound feature archive %s: sample-order association is unverified",
                path,
            )
        binding_status.append(
            {
                "path": str(path),
                "status": "verified" if verified else "unverified",
                "feature_coverage": coverage,
                "coverage_sample_order": "original-base-archive",
            }
        )
        data = merge_feature_sidecar(data, sidecar, path)
    data["alignment_verification_json"] = json.dumps(binding_status, sort_keys=True)
    validate_aligned_data(data, layer, subject)
    return data


def input_fingerprints(paths):
    """Content identities for resume; filenames/working directories may change."""
    identities = []
    for index, path in enumerate(paths):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        identities.append(
            {"role": "base" if index == 0 else "feature", "sha256": digest.hexdigest()}
        )
        association_path = Path(str(path) + ".alignment.json")
        if index and association_path.is_file():
            identities.append(
                {
                    "role": "alignment-association",
                    "sha256": file_sha256(association_path),
                }
            )
    return json.dumps(identities, sort_keys=True)


def build_design_matrices(
    data: dict, layer: str = "fc1", include_nuisance_bands: bool = False
) -> dict:
    """
    Build all design matrices from aligned data.

    KEY CHANGE: Stores BOTH raw (unlagged) and lagged features.
    PCA-before-lagging is handled in the CV loop to ensure proper train/test separation.

    Creates feature bands for banded ridge regression:
    - Always: Main features (DDQN, HRR, EZ, or an LLM layer)
    - Optional: Game/level identity, button presses, and time features
    """
    validate_aligned_data(data, layer)
    logging.info("Building design matrices...")

    # Extract BOLD data (targets) - handle both voxelwise and parcellated
    if "voxel_ts" in data:
        Y = data["voxel_ts"].T  # (n_volumes, n_voxels)
        n_targets = data["n_voxels"] if "n_voxels" in data else Y.shape[1]
        target_type = "voxel"
    else:
        Y = data["parcel_ts"].T  # (n_volumes, n_parcels)
        n_targets = data["n_parcels"] if "n_parcels" in data else Y.shape[1]
        target_type = "parcel"

    n_volumes = Y.shape[0]
    logging.info(f"  Y (BOLD): {Y.shape} ({target_type}s)")

    play_boundaries = data["play_boundaries"]

    # Determine which bands to build
    band_names = list(DEFAULT_BANDS)
    if include_nuisance_bands:
        band_names.extend(NUISANCE_BANDS)
    logging.info(f"  Bands: {band_names}")

    result = {
        "Y": Y,
        "n_volumes": n_volumes,
        "n_targets": n_targets,
        "target_type": target_type,
        "band_names": band_names,
        "play_boundaries": play_boundaries,
        "play_levels": data["play_levels"],
        "play_game_idx": data["play_game_idx"],
        "tr_level_idx": data["tr_level_idx"],
        "tr_game_idx": data["tr_game_idx"],
        "game_names": data["game_names"],
    }

    # Store mask info if available (for voxelwise reconstruction)
    if "mask" in data:
        result["mask"] = data["mask"]
        result["mask_affine"] = data["mask_affine"]

    # --- Band 1: Main features (always included) ---
    layer_key = f"{layer}_aligned"
    if layer_key not in data:
        available = [k for k in data.keys() if "_aligned" in k]
        raise ValueError(f"Layer '{layer}' not found. Available: {available}")

    X_main_raw = data[layer_key]
    logging.info(f"  X_main raw ({layer}): {X_main_raw.shape}")

    # Create mask for valid (non-zero) timepoints
    # Some timepoints may have no features (e.g., EZ data not available for some plays)
    valid_mask = X_main_raw.sum(axis=1) != 0
    n_valid = valid_mask.sum()
    n_total = len(valid_mask)
    logging.info(
        f"  Valid timepoints: {n_valid} / {n_total} ({100 * n_valid / n_total:.1f}%)"
    )
    if n_valid < n_total:
        logging.warning(
            f"  {n_total - n_valid} timepoints have zero features and will be excluded"
        )
    result["valid_mask"] = valid_mask

    # Store RAW features for PCA-before-lagging in CV loop
    result["X_main_raw"] = X_main_raw
    result["n_features_main_raw"] = X_main_raw.shape[1]

    # Also store lagged version (for reference/fallback)
    X_main = create_lagged_features(X_main_raw, play_boundaries)
    result["X_main"] = X_main
    result["n_features_main"] = X_main.shape[1]
    logging.info(f"  X_main lagged: {X_main.shape}")

    # --- Game + Level identity (fitted only with nuisance bands) ---
    n_games = int(data["n_games"])
    game_onehot = create_onehot(data["tr_game_idx"], n_games)

    level_idx = data["tr_level_idx"]
    unique_levels = np.unique(level_idx)
    level_to_idx = {lv: i for i, lv in enumerate(unique_levels)}
    level_idx_remapped = np.array([level_to_idx[lv] for lv in level_idx])
    level_onehot = create_onehot(level_idx_remapped, len(unique_levels))

    X_identity_raw = np.hstack([game_onehot, level_onehot])

    # Store RAW features
    result["X_identity_raw"] = X_identity_raw
    result["n_features_identity_raw"] = X_identity_raw.shape[1]

    X_identity = create_lagged_features(X_identity_raw, play_boundaries)
    result["X_identity"] = X_identity
    result["n_features_identity"] = X_identity.shape[1]
    result["unique_levels"] = unique_levels
    logging.info(
        f"  X_identity raw: {X_identity_raw.shape}, lagged: {X_identity.shape}"
    )

    # --- Optional bands ---
    if include_nuisance_bands:
        # Band 3: Button presses
        if "keystates" in data:
            X_button_raw = data["keystates"]  # (n_volumes, 5)
        else:
            X_button_raw = data["any_keypress"].reshape(-1, 1)

        result["X_button_raw"] = X_button_raw
        result["n_features_button_raw"] = X_button_raw.shape[1]

        X_button = create_lagged_features(X_button_raw, play_boundaries)
        result["X_button"] = X_button
        result["n_features_button"] = X_button.shape[1]
        logging.info(f"  X_button raw: {X_button_raw.shape}, lagged: {X_button.shape}")

        # Band 4: Time features
        time_in_play = data["time_in_play"].reshape(-1, 1)
        time_in_experiment = data["time_in_experiment"].reshape(-1, 1)
        X_time_raw = np.hstack([time_in_play, time_in_experiment])

        result["X_time_raw"] = X_time_raw
        result["n_features_time_raw"] = X_time_raw.shape[1]

        X_time = create_lagged_features(X_time_raw, play_boundaries)
        result["X_time"] = X_time
        result["n_features_time"] = X_time.shape[1]
        logging.info(f"  X_time raw: {X_time_raw.shape}, lagged: {X_time.shape}")

    # Partition index for CV
    tr_level_idx = data["tr_level_idx"]
    result["tr_partition_idx"] = tr_level_idx // 3  # 0-2 -> 0, 3-5 -> 1, 6-8 -> 2

    return result


def filter_to_max_level(data: dict, max_level: int) -> dict:
    """Subset every aligned feature family and remap play indices together."""
    data = dict(data)
    keep = np.asarray(data["tr_level_idx"]) <= max_level
    if not keep.any():
        raise ValueError(f"No volumes remain at levels 0-{max_level}")
    tr_keys = {
        "keystates",
        "any_keypress",
        "scores",
        "score_deltas",
        "time_in_play",
        "time_in_experiment",
        "tr_game_idx",
        "tr_level_idx",
        "tr_play_idx",
        "tr_run_idx",
    }
    for key, value in list(data.items()):
        if key.endswith("_aligned") or key in tr_keys:
            if np.asarray(value).shape[0] != len(keep):
                raise ValueError(f"{key} does not match the BOLD volume count")
            data[key] = value[keep]
    for key in ("voxel_ts", "parcel_ts"):
        if key in data:
            data[key] = data[key][:, keep]
    included, remapped = np.unique(data["tr_play_idx"], return_inverse=True)
    data["tr_play_idx"] = remapped
    for key in (
        "play_ids",
        "play_levels",
        "play_game_idx",
        "play_partitions",
        "play_has_ez",
        "play_has_llm",
        "play_llm_sources",
    ):
        if key in data:
            data[key] = data[key][included]
    changes = np.flatnonzero(np.diff(remapped)) + 1
    data["play_boundaries"] = np.r_[0, changes, len(remapped)]
    data["play_n_volumes"] = np.diff(data["play_boundaries"])
    data["n_volumes"] = int(keep.sum())
    data["n_plays"] = len(included)
    return data


def zscore_by_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Z-score both arrays using training data statistics."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1)

    valid_cols = std > 1e-6

    X_train_z = np.zeros_like(X_train)
    X_test_z = np.zeros_like(X_test)

    if valid_cols.any():
        X_train_z[:, valid_cols] = (X_train[:, valid_cols] - mean[valid_cols]) / std[
            valid_cols
        ]
        X_test_z[:, valid_cols] = (X_test[:, valid_cols] - mean[valid_cols]) / std[
            valid_cols
        ]

    return X_train_z, X_test_z


def shuffle_within_games(
    features: np.ndarray, tr_game_idx: np.ndarray, seed: int = None
) -> np.ndarray:
    """
    Shuffle features within each game to destroy temporal structure while preserving
    game-level statistics. This is a control analysis to test whether temporal
    alignment matters for prediction.

    Args:
        features: (n_volumes, n_features) array
        tr_game_idx: (n_volumes,) game index for each TR
        seed: random seed for reproducibility

    Returns:
        Shuffled features array (copy, original unchanged)
    """
    rng = np.random.RandomState(seed)
    features_shuffled = features.copy()

    unique_games = np.unique(tr_game_idx)
    for game_idx in unique_games:
        game_mask = tr_game_idx == game_idx
        game_indices = np.where(game_mask)[0]

        # Shuffle indices within this game
        shuffled_indices = rng.permutation(game_indices)

        # Apply shuffle
        features_shuffled[game_indices] = features[shuffled_indices]

    return features_shuffled


def shuffle_within_plays(
    features: np.ndarray, play_boundaries: np.ndarray, seed: int = None
) -> np.ndarray:
    """
    Shuffle features within each play. Destroys within-episode temporal structure
    while preserving play-level statistics. Stronger control than within-game
    shuffling — even within-game state-action dependencies are scrambled.

    Args:
        features: (n_volumes, n_features) array
        play_boundaries: (n_plays + 1,) array — TR indices where each play starts/ends.
                         play k spans [play_boundaries[k], play_boundaries[k+1]).
        seed: random seed for reproducibility

    Returns:
        Shuffled features array (copy, original unchanged)
    """
    rng = np.random.RandomState(seed)
    features_shuffled = features.copy()
    n_plays = len(play_boundaries) - 1

    for play_idx in range(n_plays):
        start = play_boundaries[play_idx]
        end = play_boundaries[play_idx + 1]
        if end - start <= 1:
            continue  # nothing to shuffle in a 1-TR play
        play_indices = np.arange(start, end)
        shuffled_indices = rng.permutation(play_indices)
        features_shuffled[play_indices] = features[shuffled_indices]

    return features_shuffled


def shuffle_within_levels(
    features: np.ndarray,
    tr_level_idx: np.ndarray,
    tr_game_idx: np.ndarray,
    seed: int = None,
) -> np.ndarray:
    """
    Shuffle features within each (game, level) combination. Destroys TR-level
    temporal structure across all plays of the same level within a game.

    Note: Use (game, level) jointly, not level index alone — same level number
    in different games is a different stimulus context.

    Args:
        features: (n_volumes, n_features) array
        tr_level_idx: (n_volumes,) level index for each TR
        tr_game_idx: (n_volumes,) game index for each TR
        seed: random seed for reproducibility

    Returns:
        Shuffled features array (copy, original unchanged)
    """
    rng = np.random.RandomState(seed)
    features_shuffled = features.copy()

    # Make a compound key: (game, level) tuples
    n_games = tr_game_idx.max() + 1
    compound_key = tr_game_idx * n_games * 10 + tr_level_idx  # unique per (game, level)

    for key in np.unique(compound_key):
        mask = compound_key == key
        indices = np.where(mask)[0]
        if len(indices) <= 1:
            continue
        shuffled_indices = rng.permutation(indices)
        features_shuffled[indices] = features[shuffled_indices]

    return features_shuffled


# =============================================================================
# Cross-Validation
# =============================================================================


class PartitionCV:
    """Leave-one-partition-out cross-validator."""

    def __init__(self, partitions: list, tr_partition: np.ndarray):
        self.partitions = partitions
        self.tr_partition = tr_partition

    def split(self, X, y=None, groups=None):
        for partition in self.partitions:
            test_mask = self.tr_partition == partition
            train_mask = ~test_mask
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(test_idx) > 0 and len(train_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.partitions)


# =============================================================================
# Main Encoding Model
# =============================================================================


def run_encoding_model(
    subject: str,
    data_dir: Path,
    output_dir: Path,
    layer: str = "fc1",
    max_level: int = None,
    include_nuisance_bands: bool = False,
    shuffle: bool = False,
    shuffle_scope: str = "games",
    shuffle_seed: int = 42,
    n_targets_batch: int = 500,
    n_alphas_batch: int = 10,
    n_iter: int = 100,
    backend: str = "numpy",
    seed: int | None = None,
    base_data: Path | None = None,
    feature_files: list[Path] | None = None,
    allow_unverified_alignment: bool = False,
) -> dict:
    """
    Run banded ridge encoding model for one subject.

    KEY CHANGE: PCA is applied BEFORE lagging when n_features_raw * n_lags > n_train.

    Uses leave-one-partition-out cross-validation following Tomov et al.
    Partitions: levels 0-2, 3-5, 6-8 (each partition spans all games)

    Default: main features only.
    Optional (with --include-nuisance-bands): game/level identity, button, and time.

    Args:
        subject: Subject ID
        data_dir: Directory containing aligned_data.npz
        output_dir: Output directory
        layer: Which model layer to use ('fc1' or 'hrr')
        max_level: Maximum level to include (e.g., 5 for levels 0-5)
        include_nuisance_bands: Include identity, button, and time bands
        shuffle: If True, shuffle main features within games (control analysis)
        shuffle_seed: Historical within-level shuffle seed. Use seed for all scopes.
        n_targets_batch: Batch size for targets/voxels
        n_alphas_batch: Batch size for alpha values
        n_iter: Number of random search iterations
        backend: Himalaya backend
        seed: Seed both ridge search and shuffle for fresh reproducible reruns.
              None preserves the submitted unseeded ridge/game/play behavior.

    Returns:
        Dict of {band_name: (performances, performances_mean)}
    """
    logging.info(f"Running encoding model for {subject}")
    logging.info(f"  Layer: {layer}")
    logging.info(f"  Lags: {LAGS} TRs (into the past)")
    logging.info(f"  Include nuisance bands: {include_nuisance_bands}")
    logging.info(f"  Backend: {backend}")

    # Set himalaya backend
    try:
        backend_obj = set_backend(backend, on_error="warn")
        logging.info(f"  Using backend: {backend_obj}")
    except Exception as e:
        logging.warning(f"  Could not set backend {backend}: {e}, using numpy")
        backend_obj = set_backend("numpy")

    paths = encoding_input_paths(subject, data_dir, layer, base_data, feature_files)
    source_fingerprints = input_fingerprints(paths)
    data = load_aligned_data(
        paths, subject, layer, require_binding=not allow_unverified_alignment
    )
    if n_iter < 1 or n_targets_batch < 1 or n_alphas_batch < 1:
        raise ValueError("Iteration and batch counts must be positive")
    if seed is not None and not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if shuffle_scope not in {"games", "plays", "levels"}:
        raise ValueError(f"Unknown shuffle scope: {shuffle_scope}")
    effective_shuffle_seed = (
        seed
        if seed is not None
        else (shuffle_seed if shuffle_scope == "levels" else None)
    )

    if shuffle:
        logging.info(
            "  Shuffle scope=%s, effective seed=%s",
            shuffle_scope,
            effective_shuffle_seed,
        )
    # Determine data type
    is_voxelwise = "voxel_ts" in data
    logging.info(f"  Data type: {'voxelwise' if is_voxelwise else 'parcellated'}")

    if max_level is not None:
        data = filter_to_max_level(data, max_level)

    # Apply within-game shuffle if requested (CONTROL ANALYSIS)
    if shuffle:
        layer_key = f"{layer}_aligned"
        if layer_key in data:
            if shuffle_scope == "games":
                logging.info(
                    f"  *** SHUFFLE CONTROL (seed={effective_shuffle_seed}): shuffling {layer_key} within games ***"
                )
                data[layer_key] = shuffle_within_games(
                    data[layer_key], data["tr_game_idx"], seed=effective_shuffle_seed
                )
            elif shuffle_scope == "plays":
                play_changes = np.where(np.diff(data["tr_play_idx"]) != 0)[0] + 1
                play_boundaries = np.concatenate(
                    [[0], play_changes, [len(data["tr_play_idx"])]]
                )
                logging.info(
                    f"  *** SHUFFLE CONTROL (within PLAYS, seed={effective_shuffle_seed}): "
                    f"shuffling {layer_key} within {len(play_boundaries) - 1} plays ***"
                )
                data[layer_key] = shuffle_within_plays(
                    data[layer_key], play_boundaries, seed=effective_shuffle_seed
                )
            else:
                n_unique_levels = len(
                    np.unique(
                        data["tr_game_idx"].astype(np.int64) * 100
                        + data["tr_level_idx"]
                    )
                )
                logging.info(
                    f"  *** SHUFFLE CONTROL (within LEVELS, seed={effective_shuffle_seed}): "
                    f"shuffling within {n_unique_levels} (game, level) pairs ***"
                )
                data[layer_key] = shuffle_within_levels(
                    data[layer_key],
                    data["tr_level_idx"],
                    data["tr_game_idx"],
                    seed=effective_shuffle_seed,
                )

            logging.info(
                "  *** Temporal structure destroyed - this is a control analysis ***"
            )

    # Build design matrices (stores both RAW and lagged features)
    dm = build_design_matrices(data, layer, include_nuisance_bands)

    band_names = dm["band_names"]
    Y = dm["Y"]
    n_volumes = dm["n_volumes"]
    n_targets = dm["n_targets"]
    tr_partition_idx = dm["tr_partition_idx"]
    play_boundaries = dm["play_boundaries"]
    valid_mask = dm["valid_mask"]

    logging.info(f"  Target shape: {Y.shape} ({n_targets} {dm['target_type']}s)")

    # Get unique partitions for CV
    unique_partitions = sorted(np.unique(tr_partition_idx))
    n_folds = len(unique_partitions)
    if n_folds < 2:
        raise ValueError("Encoding requires at least two level partitions")

    logging.info(f"  Cross-validation: {n_folds} partitions")
    for p in unique_partitions:
        n_trs = (tr_partition_idx == p).sum()
        n_valid_trs = ((tr_partition_idx == p) & valid_mask).sum()
        logging.info(
            f"    Partition {p} (levels {p * 3}-{p * 3 + 2}): {n_trs} TRs ({n_valid_trs} valid)"
        )

    # Storage for results
    performances_by_band = {
        band: np.zeros((n_targets, n_folds), dtype=np.float32) for band in band_names
    }
    performances_full = np.zeros((n_targets, n_folds), dtype=np.float32)

    fold_info = []

    # Batching parameters for voxelwise data
    if is_voxelwise:
        logging.info(
            f"  Voxelwise batching: n_targets_batch={n_targets_batch}, n_alphas_batch={n_alphas_batch}"
        )

    # CV loop
    for fold_idx, test_partition in enumerate(unique_partitions):
        logging.info(
            f"  Fold {fold_idx + 1}/{n_folds}: hold out partition {test_partition}"
        )

        # Apply valid_mask to exclude zero-feature timepoints
        test_mask = (tr_partition_idx == test_partition) & valid_mask
        train_mask = (tr_partition_idx != test_partition) & valid_mask

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_test < 2 or n_train < 2:
            raise ValueError(
                f"Partition {test_partition} has insufficient valid samples: "
                f"{n_train} train, {n_test} test"
            )

        logging.info(
            f"    Train: {n_train} TRs, Test: {n_test} TRs (after excluding invalid)"
        )

        # =====================================================================
        # KEY CHANGE: PCA BEFORE LAGGING
        # =====================================================================
        # For each band:
        # 1. Get raw (unlagged) features
        # 2. Split into train/test
        # 3. Z-score using train stats
        # 4. If n_features_raw * n_lags > n_train, apply PCA to raw features
        # 5. Reconstruct full feature matrix (with PCA-reduced features)
        # 6. Apply lagging
        # 7. Re-extract train/test from lagged features
        # =====================================================================

        X_train_bands = []
        X_test_bands = []
        pca_info = {}

        for band_name in band_names:
            # Get raw (unlagged) features
            X_raw = dm[f"X_{band_name}_raw"]
            n_features_raw = X_raw.shape[1]

            # Split raw features
            X_train_raw = X_raw[train_mask]
            X_test_raw = X_raw[test_mask]

            # Z-score raw features using train stats
            X_train_raw_z, X_test_raw_z = zscore_by_train(X_train_raw, X_test_raw)

            # Handle NaN/Inf
            X_train_raw_z = np.nan_to_num(
                X_train_raw_z, nan=0.0, posinf=0.0, neginf=0.0
            )
            X_test_raw_z = np.nan_to_num(X_test_raw_z, nan=0.0, posinf=0.0, neginf=0.0)

            # Check if PCA needed: n_features_raw * n_lags > n_train
            n_features_after_lag = n_features_raw * N_LAGS
            needs_pca = n_features_after_lag > n_train

            if needs_pca:
                # Determine n_components so that n_components * n_lags < n_train * FEATURE_BUDGET_RATIO
                max_total_features = int(n_train * FEATURE_BUDGET_RATIO)
                n_components = max_total_features // N_LAGS
                n_components = min(n_components, n_features_raw, n_train - 1)
                n_components = max(n_components, 1)  # At least 1 component

                logging.info(
                    f"    PCA on {band_name} (BEFORE lagging): {n_features_raw} -> {n_components} components"
                )
                logging.info(
                    f"      Reason: {n_features_raw}×{N_LAGS}={n_features_after_lag} > {n_train} train TRs"
                )
                logging.info(
                    f"      After lagging: {n_components}×{N_LAGS}={n_components * N_LAGS} features"
                )

                # Clip extreme values before PCA
                X_train_raw_z = np.clip(X_train_raw_z, -10, 10)
                X_test_raw_z = np.clip(X_test_raw_z, -10, 10)

                # Fit PCA on train, transform both
                pca = PCA(n_components=n_components, svd_solver="full")
                X_train_raw_pca = pca.fit_transform(X_train_raw_z)
                X_test_raw_pca = pca.transform(X_test_raw_z)

                pca_info[band_name] = {
                    "n_components": n_components,
                    "n_features_raw": n_features_raw,
                    "explained_variance_ratio": pca.explained_variance_ratio_.sum(),
                }
                logging.info(
                    f"      Explained variance: {pca.explained_variance_ratio_.sum():.2%}"
                )

                # Reconstruct full feature matrix for lagging
                n_features_reduced = X_train_raw_pca.shape[1]
                X_full_reduced = np.zeros(
                    (n_volumes, n_features_reduced), dtype=np.float32
                )
                X_full_reduced[train_mask] = X_train_raw_pca
                X_full_reduced[test_mask] = X_test_raw_pca

            else:
                # No PCA needed - reconstruct full z-scored raw features
                X_full_reduced = np.zeros((n_volumes, n_features_raw), dtype=np.float32)
                X_full_reduced[train_mask] = X_train_raw_z
                X_full_reduced[test_mask] = X_test_raw_z

            # Now apply lagging to the full (possibly reduced) feature matrix
            X_lagged = create_lagged_features(X_full_reduced, play_boundaries)

            # Re-extract train/test from lagged features
            X_train_lagged = X_lagged[train_mask]
            X_test_lagged = X_lagged[test_mask]

            X_train_bands.append(X_train_lagged.astype(np.float32))
            X_test_bands.append(X_test_lagged.astype(np.float32))

        # Split and z-score targets
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        Y_train, Y_test = zscore_by_train(Y_train, Y_test)
        Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        Y_test = np.nan_to_num(Y_test, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

        # Log feature band shapes
        total_features = sum(xb.shape[1] for xb in X_train_bands)
        logging.info("    Feature bands (after PCA-before-lag):")
        for i, band_name in enumerate(band_names):
            logging.info(f"      {band_name}: {X_train_bands[i].shape}")
        logging.info(
            f"    Total features: {total_features}, n_train: {n_train}, ratio: {total_features / n_train:.2%}"
        )

        # Inner CV for hyperparameter tuning
        # Need to remap train indices to 0..n_train-1 for inner CV
        train_tr_partition = tr_partition_idx[train_mask]
        train_partitions = sorted(set(train_tr_partition))
        inner_cv = PartitionCV(train_partitions, train_tr_partition)

        # Banded ridge regression with batching for voxelwise data
        actual_n_targets_batch = (
            n_targets_batch
            if "cuda" not in backend
            else min(n_targets_batch * 4, n_targets)
        )

        # Alpha grid
        alphas = np.logspace(-5, 10, 20)

        model = GroupRidgeCV(
            groups="input",
            cv=inner_cv,
            fit_intercept=False,
            solver="random_search",
            random_state=seed,
            solver_params=dict(
                score_func=correlation_score,
                n_iter=n_iter,
                alphas=alphas,
                local_alpha=True,
                jitter_alphas=True,
                n_targets_batch=actual_n_targets_batch,
                n_targets_batch_refit=actual_n_targets_batch,
                n_alphas_batch=n_alphas_batch,
                progress_bar=True,
                conservative=False,
            ),
        )

        logging.info(
            f"    Using n_targets_batch={actual_n_targets_batch}, n_iter={n_iter}, alphas={alphas[0]:.0e}-{alphas[-1]:.0e}"
        )

        try:
            model.fit(X_train_bands, Y_train)

            # Get predictions split by band
            Y_pred_bands = model.predict(X_test_bands, split=True)
            Y_pred_full = model.predict(X_test_bands)

            # Convert to numpy if using torch backend
            if hasattr(Y_pred_full, "cpu"):
                Y_pred_full = Y_pred_full.cpu().numpy()
                Y_pred_bands = [yp.cpu().numpy() for yp in Y_pred_bands]

            # Ensure Y_test is numpy
            if hasattr(Y_test, "cpu"):
                Y_test_np = Y_test.cpu().numpy()
            else:
                Y_test_np = Y_test

            # Diagnostic: check prediction statistics
            logging.info("    Prediction stats:")
            logging.info(
                f"      Y_test: mean={Y_test_np.mean():.4f}, std={Y_test_np.std():.4f}"
            )
            logging.info(
                f"      Y_pred_full: mean={Y_pred_full.mean():.4f}, std={Y_pred_full.std():.4f}"
            )

            for band_idx, band_name in enumerate(band_names):
                yp = Y_pred_bands[band_idx]
                logging.info(
                    f"      Y_pred_{band_name}: mean={yp.mean():.4f}, std={yp.std():.4f}"
                )

            # Compute correlations efficiently using vectorized operations
            for band_idx, band_name in enumerate(band_names):
                Y_pred_band = Y_pred_bands[band_idx]

                y_true_centered = Y_test_np - Y_test_np.mean(axis=0)
                y_pred_centered = Y_pred_band - Y_pred_band.mean(axis=0)

                numerator = (y_true_centered * y_pred_centered).sum(axis=0)
                denom_true = np.sqrt((y_true_centered**2).sum(axis=0))
                denom_pred = np.sqrt((y_pred_centered**2).sum(axis=0))
                denom = denom_true * denom_pred

                r = np.zeros(n_targets, dtype=np.float32)
                valid = denom > 1e-10
                r[valid] = numerator[valid] / denom[valid]
                r = np.clip(r, -1.0, 1.0)
                r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

                performances_by_band[band_name][:, fold_idx] = r

            # Full model correlations
            y_true_centered = Y_test_np - Y_test_np.mean(axis=0)
            y_pred_centered = Y_pred_full - Y_pred_full.mean(axis=0)

            numerator = (y_true_centered * y_pred_centered).sum(axis=0)
            denom_true = np.sqrt((y_true_centered**2).sum(axis=0))
            denom_pred = np.sqrt((y_pred_centered**2).sum(axis=0))
            denom = denom_true * denom_pred

            r_full = np.zeros(n_targets, dtype=np.float32)
            valid = denom > 1e-10
            r_full[valid] = numerator[valid] / denom[valid]
            r_full = np.clip(r_full, -1.0, 1.0)
            r_full = np.nan_to_num(r_full, nan=0.0, posinf=0.0, neginf=0.0)

            performances_full[:, fold_idx] = r_full

            # Log band weights
            if hasattr(model, "deltas_"):
                deltas = model.deltas_
                if hasattr(deltas, "cpu"):
                    deltas = deltas.cpu().numpy()
                logging.info("    Band weights (log10 scale, mean across targets):")
                for band_idx, band_name in enumerate(band_names):
                    mean_delta = deltas[band_idx].mean()
                    logging.info(
                        f"      {band_name}: log10={mean_delta:.4f} (10^x = {10**mean_delta:.2e})"
                    )

        except Exception as e:
            raise RuntimeError(
                f"{subject}, layer {layer}, partition {test_partition} failed; "
                "no result file was written"
            ) from e

        fold_info.append(
            {
                "partition": test_partition,
                "n_train": n_train,
                "n_test": n_test,
                "pca_info": pca_info,
            }
        )

        # Log summary
        logging.info("    Results by band:")
        for band_name in band_names:
            mean_r = performances_by_band[band_name][:, fold_idx].mean()
            max_r = performances_by_band[band_name][:, fold_idx].max()
            n_pos = (performances_by_band[band_name][:, fold_idx] > 0).sum()
            logging.info(
                f"      {band_name}: mean r={mean_r:.4f}, max r={max_r:.4f}, positive={n_pos}/{n_targets}"
            )

        mean_r_full = performances_full[:, fold_idx].mean()
        max_r_full = performances_full[:, fold_idx].max()
        n_pos_full = (performances_full[:, fold_idx] > 0).sum()
        logging.info(
            f"      FULL: mean r={mean_r_full:.4f}, max r={max_r_full:.4f}, positive={n_pos_full}/{n_targets}"
        )

    # Compute mean performance across partitions
    performances_mean_by_band = {
        band: perf.mean(axis=1) for band, perf in performances_by_band.items()
    }
    performances_mean_full = performances_full.mean(axis=1)

    # Log overall results
    logging.info("\n  === Overall Results ===")
    for band_name in band_names:
        mean_r = performances_mean_by_band[band_name].mean()
        max_r = performances_mean_by_band[band_name].max()
        n_above_05 = (performances_mean_by_band[band_name] > 0.05).sum()
        n_above_10 = (performances_mean_by_band[band_name] > 0.10).sum()
        logging.info(
            f"  {band_name}: mean={mean_r:.4f}, max={max_r:.4f}, >0.05: {n_above_05}, >0.10: {n_above_10}"
        )

    mean_r_full = performances_mean_full.mean()
    max_r_full = performances_mean_full.max()
    n_above_05_full = (performances_mean_full > 0.05).sum()
    n_above_10_full = (performances_mean_full > 0.10).sum()
    logging.info(
        f"  FULL: mean={mean_r_full:.4f}, max={max_r_full:.4f}, >0.05: {n_above_05_full}, >0.10: {n_above_10_full}"
    )

    # Save results
    output_subdir = output_dir / subject
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Determine output suffix
    suffix = (
        f"_shuffled_{shuffle_scope}"
        if shuffle
        else ("_with_nuisance" if include_nuisance_bands else "")
    )
    output_file = output_subdir / f"encoding_results_{layer}{suffix}.npz"

    partition_ids = np.array([f["partition"] for f in fold_info])

    save_dict = {
        # Full model performance
        "completion_status": "complete",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_files_json": source_fingerprints,
        "alignment_verification_json": data["alignment_verification_json"],
        "allow_unverified_alignment": allow_unverified_alignment,
        "runtime_versions_json": json.dumps(
            {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "himalaya")
            },
            sort_keys=True,
        ),
        "n_iter": n_iter,
        "alphas": np.logspace(-5, 10, 20),
        "requested_backend": backend,
        "actual_backend": backend_obj.name,
        "seed": seed if seed is not None else -1,
        "effective_shuffle_seed": effective_shuffle_seed
        if shuffle and effective_shuffle_seed is not None
        else -1,
        "n_targets_batch": n_targets_batch,
        "n_alphas_batch": n_alphas_batch,
        "randomness_policy": (
            "explicit seed controls ridge search and shuffle for this execution"
            if seed is not None
            else "submitted: unseeded ridge search and game/play shuffle; within-level shuffle uses shuffle_seed"
        ),
        "performances_full": performances_full,
        "performances_mean_full": performances_mean_full,
        # Metadata
        "partition_ids": partition_ids,
        "n_folds": n_folds,
        "lags": np.array(LAGS),
        "layer": layer,
        "max_level": max_level if max_level is not None else -1,
        "subject": subject,
        "n_targets": n_targets,
        "target_type": dm["target_type"],
        "n_volumes": dm["n_volumes"],
        "n_valid_volumes": int(valid_mask.sum()),
        "valid_sample_policy": "feature-row-sum-nonzero-after-level-selection-and-shuffle",
        "game_names": dm["game_names"],
        "band_names": np.array(band_names),
        "include_nuisance_bands": include_nuisance_bands,
        "pca_before_lag": True,  # Flag indicating this version
        "feature_budget_ratio": FEATURE_BUDGET_RATIO,
        # Shuffle control info
        "shuffle": shuffle,
        "shuffle_seed": shuffle_seed if shuffle else -1,
        # Feature dimensions (raw)
        "n_features_main_raw": dm["n_features_main_raw"],
        "n_features_identity_raw": dm["n_features_identity_raw"],
    }
    save_dict["shuffle_scope"] = shuffle_scope if shuffle else "none"

    # Add per-band performances
    for band_name in band_names:
        save_dict[f"performances_{band_name}"] = performances_by_band[band_name]
        save_dict[f"performances_mean_{band_name}"] = performances_mean_by_band[
            band_name
        ]

    # Add optional feature dimensions
    if include_nuisance_bands:
        save_dict["n_features_button_raw"] = dm["n_features_button_raw"]
        save_dict["n_features_time_raw"] = dm["n_features_time_raw"]

    # Add mask info for voxelwise reconstruction
    if "mask" in dm:
        save_dict["mask"] = dm["mask"]
        save_dict["mask_affine"] = dm["mask_affine"]

    np.savez_compressed(output_file, **save_dict)
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logging.info(f"  Saved: {output_file} ({file_size_mb:.1f} MB)")

    return {
        **{
            band: (performances_by_band[band], performances_mean_by_band[band])
            for band in band_names
        },
        "full": (performances_full, performances_mean_full),
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Run banded ridge encoding model (v3: PCA before lagging)"
    )
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument(
        "--data-dir", help="Historical layout root containing subject/aligned_data.npz"
    )
    parser.add_argument(
        "--base-data", type=Path, help="Explicit aligned BOLD/base NPZ path"
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        action="append",
        help="Explicit feature sidecar NPZ; repeat for additional feature files",
    )
    parser.add_argument(
        "--output-dir", default="./workdir/encoding_results", help="Output directory"
    )
    parser.add_argument(
        "--layer",
        default="fc1",
        help=(
            'Which features to use. Can be a single layer name (e.g. "fc1") '
            'or a comma-separated list (e.g. "fc1,hrr,llm_dsv3_legacy_layer_30"). '
            "Each layer runs as a separate encoding model; intermediate state "
            "(backend setup, Python imports) is shared."
        ),
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help="Maximum level to include (e.g., 5 for levels 0-5)",
    )
    parser.add_argument(
        "--include-nuisance-bands",
        action="store_true",
        help="Include game/level identity, button presses, and time bands (default: False)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle features within games as control analysis (default: False)",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Historical within-level shuffle seed (default: 42); --seed controls ridge and every shuffle scope",
    )
    parser.add_argument(
        "--shuffle-scope",
        choices=["games", "plays", "levels"],
        default="games",
        help="Shuffle scope: games (broadest), levels (within-level across plays), "
        "or plays (narrowest). Default: games.",
    )
    parser.add_argument(
        "--n-targets-batch",
        type=int,
        default=500,
        help="Batch size for targets/voxels (default: 500)",
    )
    parser.add_argument(
        "--n-alphas-batch",
        type=int,
        default=10,
        help="Batch size for alpha values (default: 10)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Number of random search iterations (default: 100)",
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "torch", "torch_cuda", "cupy"],
        help="Himalaya backend (default: numpy)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed ridge search and every shuffle scope for a fresh reproducible rerun; overrides --shuffle-seed. Default preserves submitted randomness.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing results only if successful completion and run settings match",
    )
    parser.add_argument(
        "--allow-unverified-alignment",
        action="store_true",
        help="Permit archived sidecars without base/sample-order bindings; result metadata records unverified association",
    )
    args = parser.parse_args()
    if args.base_data is None and args.data_dir is None:
        parser.error("provide --base-data or --data-dir")

    # Parse layer list (supports comma-separated)
    layers = [item.strip() for item in args.layer.split(",") if item.strip()]
    if not layers:
        raise SystemExit("No layers specified via --layer")

    logging.info(f"Running encoding for {len(layers)} layer(s): {layers}")

    output_subdir = Path(args.output_dir) / args.subject

    failed_layers = []
    for i, layer in enumerate(layers):
        logging.info("")
        logging.info(f"========== Layer {i + 1}/{len(layers)}: {layer} ==========")

        # Resume logic: if the output .npz for this layer already exists locally
        # (pre-downloaded from S3 by the bootstrap before the retry), skip it.
        suffix = (
            f"_shuffled_{args.shuffle_scope}"
            if args.shuffle
            else ("_with_nuisance" if args.include_nuisance_bands else "")
        )
        expected_output = output_subdir / f"encoding_results_{layer}{suffix}.npz"
        if expected_output.exists():
            if not args.resume:
                raise FileExistsError(
                    f"{expected_output}: use a new output directory or --resume"
                )
            with np.load(expected_output, allow_pickle=False) as existing:
                expected = {
                    "completion_status": "complete",
                    "input_files_json": input_fingerprints(
                        encoding_input_paths(
                            args.subject,
                            args.data_dir,
                            layer,
                            args.base_data,
                            args.feature_file,
                        )
                    ),
                    "subject": args.subject,
                    "layer": layer,
                    "source_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "n_iter": args.n_iter,
                    "allow_unverified_alignment": args.allow_unverified_alignment,
                    "seed": args.seed if args.seed is not None else -1,
                    "n_targets_batch": args.n_targets_batch,
                    "n_alphas_batch": args.n_alphas_batch,
                    "include_nuisance_bands": args.include_nuisance_bands,
                    "max_level": args.max_level if args.max_level is not None else -1,
                    "shuffle": args.shuffle,
                    "shuffle_seed": args.shuffle_seed if args.shuffle else -1,
                    "shuffle_scope": args.shuffle_scope if args.shuffle else "none",
                    "requested_backend": args.backend,
                }
                if any(
                    key not in existing or existing[key].item() != value
                    for key, value in expected.items()
                ):
                    raise ValueError(
                        f"Cannot resume mismatched or unverified result: {expected_output}"
                    )
            logging.info("  [resume] Verified completed output: %s", expected_output)
            continue

        try:
            run_encoding_model(
                subject=args.subject,
                data_dir=Path(args.data_dir) if args.data_dir is not None else None,
                base_data=args.base_data,
                feature_files=args.feature_file,
                allow_unverified_alignment=args.allow_unverified_alignment,
                output_dir=Path(args.output_dir),
                layer=layer,
                max_level=args.max_level,
                include_nuisance_bands=args.include_nuisance_bands,
                shuffle=args.shuffle,
                shuffle_scope=args.shuffle_scope,
                shuffle_seed=args.shuffle_seed,
                n_targets_batch=args.n_targets_batch,
                n_alphas_batch=args.n_alphas_batch,
                n_iter=args.n_iter,
                backend=args.backend,
                seed=args.seed,
            )
        except Exception as e:
            # Finish independent layers, but report the bundle as failed below.
            failed_layers.append(layer)
            logging.error(f"Layer {layer} FAILED: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if failed_layers:
        raise SystemExit(
            f"Encoding failed for {len(failed_layers)} layer(s): {', '.join(failed_layers)}"
        )
    logging.info("Done!")
