#!/usr/bin/env python3
"""
Stage 1: Preprocess — Apply SPM12-style preprocessing to fMRIPrep BOLD data.

Applies the preprocessing pipeline from Tomov et al. (2023) to fMRIPrep output,
producing voxelwise timeseries ready for encoding model analysis.

Preprocessing steps:
    1. Spatial smoothing (8mm FWHM Gaussian kernel)
    2. High-pass filtering (1/128 Hz cutoff, removes slow drift)
    3. Confound regression (6 motion params + derivatives, CSF, white matter)
    4. AR(1) pre-whitening (temporal autocorrelation correction)
    5. Z-scoring per voxel

Input:
    fMRIPrep output directory containing per subject/run:
        - *_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
        - *_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
        - *_desc-confounds_timeseries.tsv

Output:
    NPZ file per subject/run containing:
        - voxel_ts: (n_voxels, n_volumes) masked voxel timeseries (AR(1) whitened)
        - voxel_ts_zscored: (n_voxels, n_volumes) z-scored version
        - mask, mask_affine: boolean brain mask and affine for NIfTI reconstruction
        - preprocessing_params: dict of all parameters used (including ar1_rho)

Notes:
    - All data is in MNI152NLin2009cAsym_res-2 space (2mm isotropic), so the 3D
      grid is consistent across subjects. The per-run brain mask varies slightly.
    - AR(1) whitening (y'[t] = y[t] - rho*y[t-1]) drops the first timepoint,
      so n_volumes = n_volumes_original - 1.

Usage:
    python -m human.neural.process_bold --subject sub-01 --fmriprep-dir ./fmriprep --output-dir ./preprocessed
    python -m human.neural.process_bold --subject sub-01 --fmriprep-dir ./fmriprep --output-dir ./preprocessed --no-ar1

Reference:
    Tomov, M. S., et al. (2023). Neural architecture of theory-based reinforcement
    learning. Nature Human Behaviour.
"""

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import smooth_img, clean_img

# =============================================================================
# Configuration
# =============================================================================

# Preprocessing parameters (Tomov et al. 2023)
SMOOTHING_FWHM = 8.0  # mm, spatial smoothing
HIGH_PASS_FREQ = 1 / 128  # Hz (~0.0078 Hz), removes slow drift
TR = 2.0  # seconds (will be read from data if available)


# =============================================================================
# Helper Functions
# =============================================================================


def find_bold_file(fmriprep_dir: Path, subject: str, run: int) -> Path:
    """Find the preprocessed BOLD file for a subject/run."""
    subj_func_dir = fmriprep_dir / subject / "func"

    # Pattern: sub-XX_task-gameplay_run-YY_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
    pattern = f"{subject}_task-gameplay_run-{run:02d}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    bold_file = subj_func_dir / pattern

    if not bold_file.exists():
        raise FileNotFoundError(f"BOLD file not found: {bold_file}")

    return bold_file


def find_mask_file(fmriprep_dir: Path, subject: str, run: int) -> Path:
    """Find the brain mask file for a subject/run."""
    subj_func_dir = fmriprep_dir / subject / "func"

    pattern = f"{subject}_task-gameplay_run-{run:02d}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
    mask_file = subj_func_dir / pattern

    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    return mask_file


def find_confounds_file(fmriprep_dir: Path, subject: str, run: int) -> Path:
    """Find the confounds file for a subject/run (optional)."""
    subj_func_dir = fmriprep_dir / subject / "func"

    pattern = f"{subject}_task-gameplay_run-{run:02d}_desc-confounds_timeseries.tsv"
    confounds_file = subj_func_dir / pattern

    return confounds_file if confounds_file.exists() else None


def find_all_runs(fmriprep_dir: Path, subject: str) -> list:
    """Find all available runs for a subject."""
    subj_func_dir = fmriprep_dir / subject / "func"

    if not subj_func_dir.exists():
        return []

    runs = []
    for f in subj_func_dir.glob(
        f"{subject}_task-gameplay_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    ):
        # Extract run number from filename
        parts = f.stem.split("_")
        for part in parts:
            if part.startswith("run-"):
                run_num = int(part.replace("run-", ""))
                runs.append(run_num)
                break

    return sorted(runs)


def load_confounds(confounds_file: Path, n_volumes: int) -> tuple:
    """
    Load and prepare confound regressors from fMRIPrep.

    Following common practice, we include:
    - 6 motion parameters (trans_x/y/z, rot_x/y/z)
    - Their temporal derivatives (if available)
    - CSF and WM mean signals (aCompCor or mean signals)

    Args:
        confounds_file: Path to *_desc-confounds_timeseries.tsv
        n_volumes: Expected number of volumes

    Returns:
        Tuple of (confounds_array, confound_names) or (None, []) if no confounds found
    """
    df = pd.read_csv(confounds_file, sep="\t")

    if len(df) != n_volumes:
        raise ValueError(
            f"Confounds have {len(df)} rows but BOLD has {n_volumes} volumes"
        )

    confound_cols = []

    # Motion parameters (6 rigid body parameters)
    motion_cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    for col in motion_cols:
        if col in df.columns:
            confound_cols.append(col)

    # Motion derivatives (if available)
    motion_deriv_cols = [f"{col}_derivative1" for col in motion_cols]
    for col in motion_deriv_cols:
        if col in df.columns:
            confound_cols.append(col)

    # CSF and WM signals (physiological noise)
    physio_cols = ["csf", "white_matter"]
    for col in physio_cols:
        if col in df.columns:
            confound_cols.append(col)

    if not confound_cols:
        logging.warning("  No standard confound columns found!")
        return None, []

    logging.info(f"  Using {len(confound_cols)} confound regressors:")
    logging.info(f"    {confound_cols}")

    # Extract confounds
    confounds = df[confound_cols].values.astype(np.float64)

    # Handle NaN values (common in first row for derivatives)
    # Replace with 0 (no motion at first timepoint)
    nan_mask = np.isnan(confounds)
    if nan_mask.any():
        n_nans = nan_mask.sum()
        logging.info(f"  Replacing {n_nans} NaN values in confounds with 0")
        confounds[nan_mask] = 0

    return confounds, confound_cols


def get_tr_from_nifti(img: nib.Nifti1Image) -> float:
    """Extract TR from NIfTI header."""
    # Try to get from header
    zooms = img.header.get_zooms()
    if len(zooms) >= 4:
        tr = zooms[3]
        if tr > 0:
            return float(tr)

    # Default fallback
    logging.warning(f"Could not read TR from header, using default: {TR}s")
    return TR


def estimate_global_ar1(data: np.ndarray) -> float:
    """
    Estimate global AR(1) coefficient from data (SPM12-style).

    SPM estimates AR(1) from OLS residuals after removing low-frequency drift.
    Since we've already high-pass filtered, we estimate from the filtered data
    using a simple mean model.

    Args:
        data: (n_voxels, n_volumes) timeseries (already high-pass filtered)

    Returns:
        Global AR(1) coefficient (single value pooled across voxels)
    """
    n_voxels, n_volumes = data.shape

    # Estimate AR(1) per voxel: rho = corr(y[t], y[t-1])
    y_t = data[:, 1:]  # y[t]
    y_t1 = data[:, :-1]  # y[t-1]

    # Center each voxel's timeseries
    y_t_centered = y_t - y_t.mean(axis=1, keepdims=True)
    y_t1_centered = y_t1 - y_t1.mean(axis=1, keepdims=True)

    # Pearson correlation per voxel
    numerator = (y_t_centered * y_t1_centered).sum(axis=1)
    denominator = np.sqrt(
        (y_t_centered**2).sum(axis=1) * (y_t1_centered**2).sum(axis=1)
    )
    denominator[denominator == 0] = 1  # Avoid division by zero

    rho_per_voxel = numerator / denominator

    # Global estimate: SPM uses a pooled/averaged estimate across brain voxels
    # Using median for robustness to outliers
    rho_global = float(np.median(rho_per_voxel))

    return rho_global


def ar1_prewhiten(data: np.ndarray, rho: float) -> np.ndarray:
    """
    Apply AR(1) pre-whitening transformation (SPM12-style).

    The whitening filter transforms: y_whitened[t] = y[t] - rho * y[t-1]
    This removes temporal autocorrelation from the data.

    Note: This loses the first timepoint, reducing n_volumes by 1.

    Args:
        data: (n_voxels, n_volumes) timeseries
        rho: AR(1) coefficient

    Returns:
        Whitened data (n_voxels, n_volumes-1)
    """
    # Apply whitening filter: y'[t] = y[t] - rho * y[t-1]
    whitened = data[:, 1:] - rho * data[:, :-1]
    return whitened


# =============================================================================
# Main Processing Function
# =============================================================================


def preprocess_voxelwise(
    subject: str,
    run: int,
    fmriprep_dir: Path,
    output_dir: Path,
    smoothing_fwhm: float = SMOOTHING_FWHM,
    high_pass_freq: float = HIGH_PASS_FREQ,
    detrend: bool = True,
    standardize: bool = False,  # We'll do our own z-scoring per voxel
    use_confounds: bool = True,
    ar1_correct: bool = True,
) -> Path:
    """
    Preprocess BOLD data for one subject/run and save voxelwise timeseries.

    Preprocessing steps (following Tomov et al. 2023):
    1. Spatial smoothing (8mm FWHM)
    2. Confound regression (motion, CSF, WM)
    3. High-pass filtering (1/128 Hz)
    4. Linear detrending
    5. AR(1) pre-whitening (temporal autocorrelation correction)
    6. Z-scoring per voxel

    Args:
        subject: Subject ID (e.g., 'sub-01')
        run: Run number
        fmriprep_dir: Path to fMRIPrep output directory
        output_dir: Output directory
        smoothing_fwhm: Smoothing kernel FWHM in mm
        high_pass_freq: High-pass filter cutoff in Hz
        detrend: Whether to detrend the data
        standardize: Whether to standardize (z-score) during cleaning
        use_confounds: Whether to regress out confounds (motion, CSF, WM)
        ar1_correct: Whether to apply AR(1) pre-whitening (Tomov et al. 2023)

    Returns:
        Path to output NPZ file
    """
    logging.info(f"Processing {subject} run-{run:02d}")

    # Find input files
    bold_file = find_bold_file(fmriprep_dir, subject, run)
    mask_file = find_mask_file(fmriprep_dir, subject, run)
    confounds_file = find_confounds_file(fmriprep_dir, subject, run)

    logging.info(f"  BOLD: {bold_file.name}")
    logging.info(f"  Mask: {mask_file.name}")
    if confounds_file:
        logging.info(f"  Confounds: {confounds_file.name}")

    # Load BOLD data
    logging.info("  Loading BOLD data...")
    bold_img = nib.load(str(bold_file))
    tr = get_tr_from_nifti(bold_img)
    n_volumes = bold_img.shape[3]
    logging.info(f"  TR: {tr:.2f}s, Shape: {bold_img.shape}")

    # Load brain mask
    mask_img = nib.load(str(mask_file))
    mask_data = mask_img.get_fdata().astype(bool)
    if mask_img.shape != bold_img.shape[:3] or not np.allclose(
        mask_img.affine, bold_img.affine
    ):
        raise ValueError("BOLD and brain mask must have matching grids and affines")
    n_voxels_in_mask = mask_data.sum()
    if n_voxels_in_mask == 0:
        raise ValueError("Brain mask is empty")
    logging.info(f"  Brain mask: {n_voxels_in_mask} voxels")

    # Load confounds if requested
    confounds = None
    confound_names = []
    if use_confounds:
        if confounds_file is None:
            raise FileNotFoundError(
                f"Confounds are required for {subject} run-{run:02d}; "
                "use --no-confounds only for an explicitly different preprocessing run"
            )
        confounds, confound_names = load_confounds(confounds_file, n_volumes)
        if confounds is None:
            raise ValueError(f"No usable confound columns in {confounds_file}")

    # ==========================================================================
    # Step 1: Spatial Smoothing
    # ==========================================================================
    logging.info(f"  Applying spatial smoothing (FWHM={smoothing_fwhm}mm)...")
    bold_smoothed = smooth_img(bold_img, fwhm=smoothing_fwhm)

    # ==========================================================================
    # Step 2: High-pass Filtering + Detrending + Confound Regression
    # ==========================================================================
    logging.info(
        f"  Applying high-pass filter (cutoff={high_pass_freq:.4f} Hz = {1 / high_pass_freq:.0f}s)..."
    )
    if confounds is not None:
        logging.info(f"  Regressing out {confounds.shape[1]} confound regressors...")

    bold_cleaned = clean_img(
        bold_smoothed,
        detrend=detrend,
        standardize=standardize,
        high_pass=high_pass_freq,
        t_r=tr,
        mask_img=mask_img,
        confounds=confounds,
    )

    # Get cleaned data as array
    bold_data = bold_cleaned.get_fdata()
    logging.info(f"  Preprocessed shape: {bold_data.shape}")

    # ==========================================================================
    # Step 3: Extract masked voxels
    # ==========================================================================
    logging.info("  Extracting masked voxel timeseries...")
    # Shape: (n_voxels, n_volumes)
    voxel_ts = bold_data[mask_data].astype(np.float32)
    logging.info(f"  Voxel timeseries shape: {voxel_ts.shape}")

    # ==========================================================================
    # Step 4: AR(1) pre-whitening (temporal autocorrelation correction)
    # ==========================================================================
    ar1_rho = None
    n_volumes_original = n_volumes

    if ar1_correct:
        logging.info("  Estimating AR(1) coefficient...")
        ar1_rho = estimate_global_ar1(voxel_ts)
        logging.info(f"    Global AR(1) rho = {ar1_rho:.4f}")

        logging.info("  Applying AR(1) pre-whitening...")
        voxel_ts = ar1_prewhiten(voxel_ts, ar1_rho)
        n_volumes = voxel_ts.shape[1]  # Reduced by 1
        logging.info(f"    Whitened shape: {voxel_ts.shape} (lost first TR)")

    # ==========================================================================
    # Step 5: Z-score per voxel (for encoding model)
    # ==========================================================================
    logging.info("  Z-scoring voxel timeseries...")
    voxel_ts_mean = voxel_ts.mean(axis=1, keepdims=True)
    voxel_ts_std = voxel_ts.std(axis=1, keepdims=True)
    voxel_ts_std[voxel_ts_std == 0] = 1  # Avoid division by zero
    voxel_ts_zscored = (voxel_ts - voxel_ts_mean) / voxel_ts_std

    # ==========================================================================
    # Save Output
    # ==========================================================================
    output_subdir = output_dir / subject
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_file = output_subdir / f"run-{run:02d}_voxelwise.npz"

    preprocessing_params = {
        "smoothing_fwhm_mm": smoothing_fwhm,
        "high_pass_freq_hz": high_pass_freq,
        "high_pass_period_s": 1 / high_pass_freq,
        "detrend": detrend,
        "standardize": standardize,
        "confounds_regressed": use_confounds and confounds is not None,
        "n_confounds": len(confound_names),
        "ar1_corrected": ar1_correct,
        "ar1_rho": ar1_rho if ar1_rho is not None else 0.0,
    }

    np.savez_compressed(
        output_file,
        # Main data
        voxel_ts=voxel_ts,  # (n_voxels, n_volumes)
        voxel_ts_zscored=voxel_ts_zscored,  # (n_voxels, n_volumes)
        # Mask for reconstruction
        mask=mask_data,  # (x, y, z) boolean
        mask_affine=mask_img.affine,  # (4, 4) affine transform
        mask_shape=np.array(mask_data.shape),  # (3,) original 3D shape
        # Metadata
        subject=subject,
        run=run,
        tr=tr,
        n_volumes=n_volumes,
        n_volumes_original=n_volumes_original,
        n_voxels=n_voxels_in_mask,
        bold_shape=np.array(bold_data.shape),
        # Confound info
        confound_names=np.array(confound_names, dtype="U32")
        if confound_names
        else np.array([]),
        # Preprocessing parameters
        **{f"preproc_{k}": v for k, v in preprocessing_params.items()},
    )

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logging.info(f"  Saved: {output_file.name} ({file_size_mb:.1f} MB)")

    return output_file


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Preprocess fMRIPrep BOLD data and save voxelwise timeseries"
    )
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run number (if not specified, process all runs)",
    )
    parser.add_argument(
        "--fmriprep-dir", required=True, help="Path to fMRIPrep output directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./workdir/preprocessed",
        help="Output directory (default: ./workdir/preprocessed)",
    )
    parser.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=SMOOTHING_FWHM,
        help=f"Smoothing kernel FWHM in mm (default: {SMOOTHING_FWHM})",
    )
    parser.add_argument(
        "--high-pass",
        type=float,
        default=HIGH_PASS_FREQ,
        help=f"High-pass filter cutoff in Hz (default: {HIGH_PASS_FREQ:.4f})",
    )
    parser.add_argument(
        "--no-detrend", action="store_true", help="Disable linear detrending"
    )
    parser.add_argument(
        "--no-confounds",
        action="store_true",
        help="Disable confound regression (motion, CSF, WM)",
    )
    parser.add_argument(
        "--no-ar1",
        action="store_true",
        help="Disable AR(1) pre-whitening (temporal autocorrelation correction)",
    )

    args = parser.parse_args()

    fmriprep_dir = Path(args.fmriprep_dir)
    output_dir = Path(args.output_dir)

    # Determine which runs to process
    if args.run is not None:
        runs = [args.run]
    else:
        runs = find_all_runs(fmriprep_dir, args.subject)
        if not runs:
            raise ValueError(f"No runs found for {args.subject} in {fmriprep_dir}")
        logging.info(f"Found {len(runs)} runs for {args.subject}: {runs}")

    # Process each run
    for run in runs:
        preprocess_voxelwise(
            subject=args.subject,
            run=run,
            fmriprep_dir=fmriprep_dir,
            output_dir=output_dir,
            smoothing_fwhm=args.smoothing_fwhm,
            high_pass_freq=args.high_pass,
            detrend=not args.no_detrend,
            use_confounds=not args.no_confounds,
            ar1_correct=not args.no_ar1,
        )

    logging.info(f"Completed {args.subject}: {len(runs)} runs processed")
