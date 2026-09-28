"""
Shared utilities for extracting BOLD data (native or MNI) aligned with behavioral states.

Human JSON replays supply behavior and scanner metadata. These functions
accept local roots and do not acquire data or configure logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from human.behavior import play_states

if TYPE_CHECKING:
    import nibabel as nib


def load_bold_nifti(subject: str, run: int, data_path: str, space: str) -> Path:
    """
    Load BOLD nifti from fmriprep output.

    Args:
        subject: Subject ID (e.g., 'sub-01')
        run: Run number
        data_path: Base path to data (local directory)
        space: Space identifier ('T1w' for native, 'MNI152NLin2009cAsym' for MNI)

    Returns:
        Path to nifti file
    """
    subj_str = subject if subject.startswith("sub-") else f"sub-{int(subject):02d}"

    # Build filename based on space
    if space == "T1w":
        # Native space: space-T1w_desc-preproc_bold.nii.gz
        filename = (
            f"{subj_str}_task-gameplay_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz"
        )
    elif space == "MNI152NLin2009cAsym":
        # MNI space: space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
        filename = f"{subj_str}_task-gameplay_run-{run:02d}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    else:
        raise ValueError(f"Unknown space: {space}")

    base_path = Path(data_path)
    local_file = base_path / "fmriprep" / subj_str / "func" / filename

    if not local_file.exists():
        raise FileNotFoundError(f"BOLD not found: {local_file}")

    logging.info(f"Using local file: {local_file}")
    return local_file


def load_brain_mask_nifti(subject: str, run: int, data_path: str, space: str) -> Path:
    """
    Load brain mask nifti from fmriprep output.

    Args:
        subject: Subject ID (e.g., 'sub-01')
        run: Run number
        data_path: Base path to data (local directory)
        space: Space identifier ('T1w' for native, 'MNI152NLin2009cAsym' for MNI)

    Returns:
        Path to nifti mask file
    """
    subj_str = subject if subject.startswith("sub-") else f"sub-{int(subject):02d}"

    # Build filename based on space
    if space == "T1w":
        # Native space: space-T1w_desc-brain_mask.nii.gz
        filename = (
            f"{subj_str}_task-gameplay_run-{run:02d}_space-T1w_desc-brain_mask.nii.gz"
        )
    elif space == "MNI152NLin2009cAsym":
        # MNI space: space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
        filename = f"{subj_str}_task-gameplay_run-{run:02d}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
    else:
        raise ValueError(f"Unknown space: {space}")

    base_path = Path(data_path)
    local_file = base_path / "fmriprep" / subj_str / "func" / filename

    if not local_file.exists():
        raise FileNotFoundError(f"Brain mask not found: {local_file}")

    logging.info(f"Using mask file: {local_file}")
    return local_file


def compute_temporal_alignment(
    state_timestamps: List[float],
    scan_start_ts: float,
    num_volumes: int,
    tr: float = 2.0,
) -> Dict:
    """
    Compute which fMRI volume corresponds to each behavioral state.

    Args:
        state_timestamps: Unix timestamps for behavioral states
        scan_start_ts: Unix timestamp when fMRI scan started
        num_volumes: Total number of fMRI volumes
        tr: Repetition time in seconds

    Returns:
        dict with volume_indices, onsets, valid_mask, scan_start_ts
    """
    onsets = np.array(state_timestamps) - scan_start_ts
    volume_indices = np.round(onsets / tr).astype(int)
    valid_mask = (volume_indices >= 0) & (volume_indices < num_volumes)

    return {
        "volume_indices": volume_indices,
        "onsets": onsets,
        "valid_mask": valid_mask,
        "scan_start_ts": scan_start_ts,
    }


def extract_bold_for_play(
    play_doc: Dict,
    bold_img: nib.Nifti1Image,
    tr: float,
    scan_start_ts: float,
    space: str,
) -> Dict:
    """
    Extract BOLD data with temporal alignment to behavioral states.

    BOLD data dimensions: (x, y, z, time)
      - x, y, z: spatial dimensions (voxels)
      - time: temporal dimension (volumes/TRs)

    Args:
        play_doc: Document from plays collection
        bold_img: Nibabel image object for BOLD
        tr: Repetition time in seconds
        scan_start_ts: Unix timestamp when fMRI scan started
        space: Space identifier ('T1w' or 'MNI152NLin2009cAsym')

    Returns:
        Dict with bold_data (contiguous slice), behavioral, temporal_alignment, metadata
    """
    play_id = play_doc["_id"]
    subj_id = play_doc["subj_id"]
    game_name = play_doc["game_name"]
    level_id = play_doc["level_id"]
    run_id = play_doc.get("run_id")
    human_win = play_doc.get("win")
    human_score = play_doc.get("score")

    # Decompress states
    states = play_states(play_doc)
    state_timestamps = [state.get("ts", 0) for state in states]

    # Load BOLD data
    bold_data = bold_img.get_fdata()
    num_volumes = bold_data.shape[3]

    # Compute temporal alignment
    alignment = compute_temporal_alignment(
        state_timestamps, scan_start_ts, num_volumes, tr
    )

    valid_mask = alignment["valid_mask"]
    volume_indices = np.array(alignment["volume_indices"])

    # Extract contiguous slice of BOLD data
    valid_volume_indices = volume_indices[valid_mask]
    if len(valid_volume_indices) == 0:
        raise ValueError(f"Play {play_id} has no valid states within fMRI scan")

    vol_min = valid_volume_indices.min()
    vol_max = valid_volume_indices.max()

    # Extract contiguous slice [vol_min, vol_max] along time axis
    bold_slice = bold_data[:, :, :, vol_min : vol_max + 1]

    # Adjust volume indices to be relative to the slice
    volume_indices_adjusted = volume_indices - vol_min

    # Behavioral data
    behavioral = {
        "win": human_win,
        "score": human_score,
        "timestamps": state_timestamps,
        "num_states": len(states),
        "num_valid_states": int(valid_mask.sum()),
    }

    # Package results
    result = {
        "bold_data": bold_slice,
        "behavioral": behavioral,
        "temporal_alignment": {
            "volume_indices": volume_indices_adjusted,
            "volume_offset": int(vol_min),
            "onsets": alignment["onsets"],
            "valid_mask": valid_mask,
            "scan_start_ts": scan_start_ts,
        },
        "metadata": {
            "play_id": str(play_id),
            "subj_id": int(subj_id),
            "run_id": int(run_id) if run_id is not None else None,
            "game_name": game_name,
            "level_id": int(level_id),
            "tr": tr,
            "space": space,
        },
    }

    return result


def save_bold_data_for_level(
    plays_data: List[Dict],
    brain_mask: np.ndarray,
    output_dir: str,
    subject: str,
    game_name: str,
    level: int,
):
    """
    Save BOLD data for all rollouts of a level to a single file.

    Args:
        plays_data: List of dicts, each with bold_data, behavioral, temporal_alignment, metadata
        brain_mask: (x, y, z) brain mask array from fMRIPrep
        output_dir: Base output directory
        subject: Subject ID (e.g., 'sub-01')
        game_name: Game name
        level: Level number
    """
    # Create directory structure: subject/game/
    subj_str = subject if subject.startswith("sub-") else f"sub-{int(subject):02d}"
    subj_dir = Path(output_dir) / subj_str
    game_dir = subj_dir / game_name
    game_dir.mkdir(parents=True, exist_ok=True)

    # Save as npz file: sub-XX/game/level_YY.npz
    output_file = game_dir / f"level_{level:02d}.npz"

    # Build save dict with indexed keys for each play
    save_dict = {}
    play_ids = []

    for idx, data in enumerate(plays_data):
        play_ids.append(data["metadata"]["play_id"])

        # Save each play's data with index prefix
        save_dict[f"play_{idx}_bold_data"] = data["bold_data"]

        # Behavioral
        save_dict[f"play_{idx}_behavioral_win"] = data["behavioral"]["win"]
        save_dict[f"play_{idx}_behavioral_score"] = data["behavioral"]["score"]
        save_dict[f"play_{idx}_behavioral_timestamps"] = np.array(
            data["behavioral"]["timestamps"]
        )
        save_dict[f"play_{idx}_behavioral_num_states"] = data["behavioral"][
            "num_states"
        ]
        save_dict[f"play_{idx}_behavioral_num_valid_states"] = data["behavioral"][
            "num_valid_states"
        ]

        # Temporal alignment
        save_dict[f"play_{idx}_temporal_volume_indices"] = np.array(
            data["temporal_alignment"]["volume_indices"]
        )
        save_dict[f"play_{idx}_temporal_volume_offset"] = data["temporal_alignment"][
            "volume_offset"
        ]
        save_dict[f"play_{idx}_temporal_onsets"] = np.array(
            data["temporal_alignment"]["onsets"]
        )
        save_dict[f"play_{idx}_temporal_valid_mask"] = np.array(
            data["temporal_alignment"]["valid_mask"]
        )
        save_dict[f"play_{idx}_temporal_scan_start_ts"] = data["temporal_alignment"][
            "scan_start_ts"
        ]

        # Metadata
        save_dict[f"play_{idx}_metadata_play_id"] = data["metadata"]["play_id"]
        save_dict[f"play_{idx}_metadata_subj_id"] = data["metadata"]["subj_id"]
        save_dict[f"play_{idx}_metadata_run_id"] = (
            data["metadata"]["run_id"] if data["metadata"]["run_id"] is not None else -1
        )
        save_dict[f"play_{idx}_metadata_game_name"] = data["metadata"]["game_name"]
        save_dict[f"play_{idx}_metadata_level_id"] = data["metadata"]["level_id"]
        save_dict[f"play_{idx}_metadata_tr"] = data["metadata"]["tr"]
        save_dict[f"play_{idx}_metadata_space"] = data["metadata"]["space"]

    # Save brain mask (shared across all plays from same run)
    save_dict["brain_mask"] = brain_mask

    # Save play IDs array for easy reference
    save_dict["play_ids"] = np.array(play_ids, dtype="U24")
    save_dict["num_plays"] = len(plays_data)

    np.savez_compressed(output_file, **save_dict)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logging.info(f"Saved: {file_size_mb:.1f}MB -> {output_file.name}")

    return output_file
