#!/usr/bin/env python3
"""
Parse encoding results from the variant × stream sweep into a long-format CSV.

Input layout (flat per-subject):
    {results_dir}/sub-{XX}/encoding_results_llm_{model}__{variant}__{stream}_layer_{N}_with_nuisance.npz

Each .npz contains:
    performances_main:     (n_voxels, n_partitions)  - main band correlations
    performances_full:     (n_voxels, n_partitions)  - full-model correlations
    performances_identity: (n_voxels, n_partitions)
    performances_button:   (n_voxels, n_partitions)  if include_nuisance_bands
    performances_time:     (n_voxels, n_partitions)  if include_nuisance_bands
    mask:                  (97, 115, 97) bool        - which voxels in MNI grid
    mask_affine:           (4, 4)                    - voxel -> MNI mm
    band_names, layer, subject, ...

Output CSV columns:
    model           e.g. qwen35_9b
    variant         all | compressed
    stream          main | attn | mlp
    fit_condition   main-only | with-nuisance (fitted model, distinct from band)
    family          e.g. Qwen3.5
    backbone        e.g. Qwen
    n_layers        total layers in the model (32, 40, 64)
    layer_idx       1..n_layers
    layer_depth     (layer_idx-1) / (n_layers-1) in [0, 1]
    band            'main' or 'full'  (which band the performance is for)
    ROI             AAL ROI short name
    side            left | right
    subject         sub-13, sub-25, ...
    partition       0, 1, 2  (CV fold)
    performance     correlation (Pearson r)
    n_voxels        # voxels in this (ROI, side) for this subject

Usage:
    python load_and_parse_streams.py \\
        --results-dir ./encoding_data \\
        --atlas ~/atlases/aal/ROI_MNI_V4.nii \\
        --output ./encoding_roi_streams.csv \\
        --workers 8

    # Filter to specific (model, variant, stream) combos:
    python load_and_parse_streams.py --results-dir ./encoding_data \\
        --models qwen35_9b --variants all compressed --streams main attn mlp \\
        --output qwen9b_full.csv

    # Only emit main-band rows (faster, smaller CSV):
    python load_and_parse_streams.py --results-dir ./encoding_data \\
        --bands main --output qwen9b_main.csv

    # Also emit one nifti per (model, subject) at the subject's best layer
    # (selected by highest whole-brain mean main-band r):
    python load_and_parse_streams.py --results-dir ./encoding_data \\
        --atlas ~/atlases/aal/ROI_MNI_V4.nii \\
        --bands main --output rois.csv \\
        --nifti-dir ./best_layer_niftis --nifti-band main
"""

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# =============================================================================
# Model metadata and ROI mapping (same as legacy parser)
# =============================================================================

# =============================================================================
# Model metadata
# =============================================================================
# model_id -> (n_layers, family, backbone)
#
# Layer counts mirror LLM_LAYER_COUNTS in the upstream alignment code.
# `family` is the label used for grouping in plots; `backbone` is the underlying
# architecture vendor. Both are free-form strings — adjust to taste, e.g. if
# you'd rather group all MoE Qwens together or split ablations into per-fraction
# families. Unknown models fall through to (None, model_name, None) and emit
# rows with NaN layer_depth, so add new entries here when sources are added.

MODEL_INFO = {
    # Qwen3.5 instruct (dense)
    "qwen35_9b": (32, "Qwen3.5", "Qwen"),
    "qwen35_27b": (64, "Qwen3.5", "Qwen"),
    # Qwen3.5 instruct (MoE)
    "qwen35_35b_a3b": (40, "Qwen3.5-MoE", "Qwen"),
    "qwen35_122b_a10b": (48, "Qwen3.5-MoE", "Qwen"),
    # Qwen3.5 base (no RLHF)
    "qwen35_9b_base": (32, "Qwen3.5-Base", "Qwen"),
    "qwen35_35b_a3b_base": (40, "Qwen3.5-MoE-Base", "Qwen"),
    # Qwen3.5 ablations (27B, expert-fraction)
    "qwen35_27b_abl05": (64, "Qwen3.5-Abl-0.5", "Qwen"),
    "qwen35_27b_abl01": (64, "Qwen3.5-Abl-0.1", "Qwen"),
    # 'qwen35_27b_abl07':     (64, 'Qwen3.5-Abl-0.7',  'Qwen'),   # excluded
    # Qwen3.5 random-init controls
    "qwen35_9b_random": (32, "Qwen3.5-Random", "Qwen"),
    "qwen35_27b_random": (64, "Qwen3.5-Random", "Qwen"),
    "qwen35_35b_a3b_random": (40, "Qwen3.5-Random", "Qwen"),
    # Qwen3.5-9B shuffling ablations (synthesized at filename-parse time from
    # _shuffled_{games,levels,plays} suffixes — see FILENAME_PATTERN handling).
    "qwen35_9b_shuf_games": (32, "Qwen3.5-Shuf-Games", "Qwen"),
    "qwen35_9b_shuf_levels": (32, "Qwen3.5-Shuf-Levels", "Qwen"),
    "qwen35_9b_shuf_plays": (32, "Qwen3.5-Shuf-Plays", "Qwen"),
    # Qwen3.5 suggestion-prompt ablations
    "qwen35_9b_sugmin": (32, "Qwen3.5-SugMin", "Qwen"),
    "qwen35_9b_sugorc": (32, "Qwen3.5-SugOrc", "Qwen"),
    "qwen35_27b_sugmin": (64, "Qwen3.5-SugMin", "Qwen"),
    "qwen35_27b_sugorc": (64, "Qwen3.5-SugOrc", "Qwen"),
    "qwen35_35b_a3b_sugmin": (40, "Qwen3.5-SugMin", "Qwen"),
    "qwen35_35b_a3b_sugorc": (40, "Qwen3.5-SugOrc", "Qwen"),
    # DeepSeek R1-Distill (Qwen backbone)
    # 'ds32b_mt':             (64, 'R1-Distill-MT',    'Qwen'),     # legacy, dropped
    # 'ds32b_legacy':         (64, 'R1-Distill',       'Qwen'),     # legacy, dropped
    # DeepSeek V3 / V4
    "dsv32": (61, "DeepSeek-V3.2", "DeepSeek"),
    # 'dsv3_legacy':          (61, 'V3-legacy',        'DeepSeek'),  # legacy, dropped
    "dsv4_flash": (43, "DeepSeek-V4-Flash", "DeepSeek"),
    "dsv4_flash_base": (43, "DeepSeek-V4-Flash-Base", "DeepSeek"),
    "dsv4_pro": (61, "DeepSeek-V4-Pro", "DeepSeek"),
}

# =============================================================================
# AAL ROI grouping — built dynamically from the AAL label table (ROI_MNI_V4.txt)
# =============================================================================
#
# ROI_GROUPING maps each short name → list of AAL "base names" (the region name
# without the _L / _R suffix as it appears in ROI_MNI_V4.txt). At load time we
# resolve each base to its (left, right) codes, or 'midline' for unlateralized
# structures (vermis in AAL v4).
#
# Each AAL code may belong to multiple short names (e.g., Caudate_L → both
# 'Caudate' and 'dStriatum'); the per-file worker resolves voxels with
# np.isin against the code list of each (roi, side), so overlap is supported.

ROI_GROUPING = {
    # Frontal
    "PreCG": ["Precentral"],
    "SFG": ["Frontal_Sup"],
    "MFG": ["Frontal_Mid"],
    "IFGoperc": ["Frontal_Inf_Oper"],
    "IFGtriang": ["Frontal_Inf_Tri"],
    "ROL": ["Rolandic_Oper"],
    "SMA": ["Supp_Motor_Area"],
    "OFC": ["Frontal_Med_Orb", "Rectus"],  # medial OFC only
    # Parietal
    "PoCG": ["Postcentral"],
    "IPG": ["Parietal_Inf"],
    "SMG": ["SupraMarginal"],
    "AG": ["Angular"],
    "PCUN": ["Precuneus"],
    # Occipital
    "CAL": ["Calcarine"],
    "CUN": ["Cuneus"],
    "LING": ["Lingual"],
    "SOG": ["Occipital_Sup"],
    "MOG": ["Occipital_Mid"],
    "IOG": ["Occipital_Inf"],
    "FFG": ["Fusiform"],
    # Temporal
    "MTG": ["Temporal_Mid"],
    # Subcortical
    "Caudate": ["Caudate"],
    "Putamen": ["Putamen"],
    "dStriatum": ["Caudate", "Putamen"],
    # Cerebellum (lateralized lobules + midline vermis)
    "Cerebellum": [
        "Cerebelum_Crus1",
        "Cerebelum_Crus2",
        "Cerebelum_3",
        "Cerebelum_4_5",
        "Cerebelum_6",
        "Cerebelum_7b",
        "Cerebelum_8",
        "Cerebelum_9",
        "Cerebelum_10",
        "Vermis_1_2",
        "Vermis_3",
        "Vermis_4_5",
        "Vermis_6",
        "Vermis_7",
        "Vermis_8",
        "Vermis_9",
        "Vermis_10",
    ],
}


def load_aal_label_table(atlas_txt_path):
    """Parse ROI_MNI_V4.txt → {full_region_name: integer_code}.

    Tab- or whitespace-separated; three tokens per line:
        <abbrev>  <Region_Name>  <code>
    Blank lines and lines starting with '#' are skipped.
    """
    name_to_code = {}
    with open(atlas_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                code = int(parts[-1])
            except ValueError:
                continue
            name_to_code[parts[-2]] = code
    if not name_to_code:
        raise ValueError(f"No labels parsed from {atlas_txt_path!r}")
    return name_to_code


def build_lateralized_roi_codes(atlas_txt_path, grouping=ROI_GROUPING):
    """Build {(short_name, side): sorted list of AAL codes}.

    side ∈ {'left', 'right', 'midline'}. 'midline' is used for AAL bases that
    have no _L / _R variant (the cerebellar vermis in AAL v4).

    Raises KeyError if any base name in `grouping` is missing from the atlas.
    """
    name_to_code = load_aal_label_table(atlas_txt_path)
    out = {}
    for short_name, bases in grouping.items():
        left, right, midline = [], [], []
        for base in bases:
            l_full, r_full = base + "_L", base + "_R"
            l_in, r_in, m_in = (
                l_full in name_to_code,
                r_full in name_to_code,
                base in name_to_code,
            )
            if not (l_in or r_in or m_in):
                raise KeyError(
                    f"AAL region '{base}' (in ROI '{short_name}') "
                    f"not found in {atlas_txt_path}"
                )
            if l_in:
                left.append(name_to_code[l_full])
            if r_in:
                right.append(name_to_code[r_full])
            if m_in:
                midline.append(name_to_code[base])
        if left:
            out[(short_name, "left")] = sorted(left)
        if right:
            out[(short_name, "right")] = sorted(right)
        if midline:
            out[(short_name, "midline")] = sorted(midline)
    return out


def _default_atlas_labels_path(atlas_path):
    """Derive ROI_MNI_V4.txt path from a sibling ROI_MNI_V4.nii(.gz) path."""
    s = str(atlas_path)
    for ext in (".nii.gz", ".nii"):
        if s.endswith(ext):
            return s[: -len(ext)] + ".txt"
    return s + ".txt"


FILENAME_PATTERN = re.compile(
    r"^encoding_results_llm_(?P<model>[a-z0-9_]+?)__(?P<variant>[a-z]+)__(?P<stream>[a-z]+)"
    r"_layer_(?P<layer>\d+)(?:_(?P<suffix>[a-z_]+))?\.npz$"
)

# Filename suffix → action (suffix is the captured group, NOT including the leading '_'):
#   None              → unsuffixed file; pass through as-is
#   'with_nuisance'   → standard nuisance-regressed file; pass through as-is
#   'shuffled_games'  → shuffling ablation; rename model to {model}_shuf_games
#   'shuffled_levels' →                "                    {model}_shuf_levels
#   'shuffled_plays'  →                "                    {model}_shuf_plays
#   anything else     → unrecognized; skip with a warning
SHUFFLE_SUFFIX_PREFIX = "shuffled_"
KNOWN_PASSTHROUGH_SUFFIXES = {"with_nuisance"}


# =============================================================================
# File discovery
# =============================================================================


def get_fit_condition(path):
    """Read fit settings, including shuffles whose filenames omit this setting."""
    with np.load(path, allow_pickle=False) as data:
        bands = (
            {str(name) for name in data["band_names"]} if "band_names" in data else None
        )
        if bands is not None and (
            "main" not in bands or not bands <= {"main", "identity", "button", "time"}
        ):
            raise ValueError(f"Unsupported band_names in {path}: {sorted(bands)}")
        if "include_nuisance_bands" in data:
            value = data["include_nuisance_bands"]
            if value.shape != () or value.item() not in (True, False):
                raise ValueError(f"Invalid include_nuisance_bands in {path}")
            nuisance = bool(value.item())
            if bands is not None and nuisance != (bands != {"main"}):
                raise ValueError(f"Conflicting nuisance-band metadata in {path}")
        elif bands is not None and "main" in bands:
            nuisance = bands != {"main"}
        else:
            raise ValueError(f"Missing fit-condition metadata in {path}")
    return "with-nuisance" if nuisance else "main-only"


def find_npz_files(
    results_dirs,
    models=None,
    variants=None,
    streams=None,
    subjects=None,
    allowed_layers=None,
    allowed_layers_per_model=None,
    baseline_map=None,
    fit_condition=None,
):
    """Walk one or more results_dir/sub-XX/ trees and return parsed-metadata items.

    `results_dirs` may be a single path or an iterable of paths. Each is walked
    independently; items from later directories that have the same key
    (subject, model, variant, stream, layer, fit_condition) as an earlier item are dropped
    (with a log message). This keeps `--results-dir A B` clean when both A and
    B contain the same baseline files (e.g. shuffled_encoding_data/ usually
    contains a copy of the unshuffled `_with_nuisance` baseline alongside the
    `_shuffled_*` files).

    Filename suffix handling:
      * unsuffixed and `_with_nuisance` files pass through normally
      * `_shuffled_{games,levels,plays}` files are re-tagged with model name
        `{model}_shuf_{games|levels|plays}` so they appear as their own model
        in the CSV (they're a shuffling ablation, not a 'variant' of qwen35_9b
        in the encoding-pipeline sense, since the pipeline reserves 'variant'
        for all/compressed)
      * any other suffix produces a one-time warning and the file is skipped

    Fit conditions are read from result metadata. Mixed main-only and nuisance
    fits require explicit selection with fit_condition; they are never deduped
    or combined into the same CSV. Optional filters keep matching items only.

    allowed_layers: set of int — applied to all models if allowed_layers_per_model
                    doesn't list that model.
    allowed_layers_per_model: dict[model_name -> set[int]] — per-model overrides.
    """
    if isinstance(results_dirs, (str, Path)):
        results_dirs = [results_dirs]
    items = []
    conditions_found = set()
    seen_key_to_path = {}  # model/layer/condition identity -> first filepath
    n_dup = 0
    unknown_suffixes_seen = set()

    for results_dir in results_dirs:
        results_dir = Path(results_dir)
        if not results_dir.is_dir():
            logging.warning(f"--results-dir entry is not a directory: {results_dir}")
            continue
        for sub_dir in sorted(results_dir.iterdir()):
            if not (sub_dir.is_dir() and sub_dir.name.startswith("sub-")):
                continue
            subject = sub_dir.name
            if subjects and subject not in subjects:
                continue
            for f in sorted(sub_dir.iterdir()):
                mt = FILENAME_PATTERN.match(f.name)
                baseline_metadata = None
                if mt:
                    model = mt.group("model")
                    variant = mt.group("variant")
                    stream = mt.group("stream")
                    layer = int(mt.group("layer"))
                    suffix = mt.group("suffix")
                else:
                    for feature_name, metadata in (baseline_map or {}).items():
                        baseline_match = re.fullmatch(
                            rf"encoding_results_{re.escape(feature_name)}"
                            r"(?:_(with_nuisance|shuffled_games|shuffled_levels|shuffled_plays))?\.npz",
                            f.name,
                        )
                        if baseline_match:
                            baseline_metadata = metadata
                            model = metadata["model"]
                            variant = metadata.get("variant", "all")
                            stream = metadata.get("stream", "main")
                            layer = metadata["layer_idx"]
                            suffix = baseline_match.group(1)
                            break
                    if baseline_metadata is None:
                        continue

                # Suffix dispatch
                if suffix is None or suffix in KNOWN_PASSTHROUGH_SUFFIXES:
                    pass  # keep model name as-is
                elif suffix.startswith(SHUFFLE_SUFFIX_PREFIX):
                    shuffle_type = suffix[
                        len(SHUFFLE_SUFFIX_PREFIX) :
                    ]  # 'games', 'levels', 'plays'
                    model = f"{model}_shuf_{shuffle_type}"
                else:
                    if suffix not in unknown_suffixes_seen:
                        unknown_suffixes_seen.add(suffix)
                        logging.warning(
                            f"Unknown filename suffix '_{suffix}' (e.g. {f.name}); "
                            f"skipping all files with this suffix. To support it, "
                            f"extend the suffix dispatch in find_npz_files()."
                        )
                    continue

                if models and model not in models:
                    continue
                if variants and variant not in variants:
                    continue
                if streams and stream not in streams:
                    continue

                # Layer filter: per-model takes precedence; else fall back to global allowed_layers
                if allowed_layers_per_model and model in allowed_layers_per_model:
                    if layer not in allowed_layers_per_model[model]:
                        continue
                elif allowed_layers is not None:
                    if layer not in allowed_layers:
                        continue

                condition = get_fit_condition(f)
                if suffix == "with_nuisance" and condition != "with-nuisance":
                    raise ValueError(
                        f"Filename and fit-condition metadata disagree: {f}"
                    )
                if fit_condition is not None and condition != fit_condition:
                    continue
                conditions_found.add(condition)
                if len(conditions_found) > 1:
                    raise ValueError(
                        "Mixed main-only and nuisance fits; select --fit-condition "
                        "main-only or --fit-condition with-nuisance and write separate CSVs"
                    )

                key = (subject, model, variant, stream, layer, condition)
                if key in seen_key_to_path:
                    n_dup += 1
                    logging.debug(
                        f"  Dedup: dropping {f} (already have "
                        f"{seen_key_to_path[key]} for key={key})"
                    )
                    continue
                seen_key_to_path[key] = f

                items.append(
                    {
                        "filepath": f,
                        "subject": subject,
                        "model": model,
                        "variant": variant,
                        "stream": stream,
                        "fit_condition": condition,
                        "layer_idx": layer,
                        "baseline_metadata": baseline_metadata,
                    }
                )

    if n_dup:
        logging.info(
            f"  Deduped {n_dup} files across results-dirs (same "
            f"(subject, model, variant, stream, layer, fit_condition) found more than once; "
            f"first-seen wins)"
        )
    return items


# =============================================================================
# Depth-k layer selection
# =============================================================================
#
# "Depth-k" = pick k layers per model whose normalized depths (layer_idx-1)/(n-1)
# are as close as possible to evenly-spaced targets {0, 1/(k-1), ..., 1}. Used to
# get cross-model-comparable layer samples — e.g., depth=0.5 is the middle layer
# regardless of whether the model has 32 or 64 layers.
#
# We do "closest available" rather than "canonical formula": for each evenly-
# spaced target depth, pick the available layer closest to it, no replacement.
# This gracefully handles three situations:
#   (a) model has exactly k layers at canonical positions → all picked
#   (b) model has more than k available → picks the k closest to canonical
#   (c) model has fewer than k available, or coverage gaps → picks what it can
#       and the log shows which target depths went unfilled


def pick_depth_k_layers(available_layers, n_total, k=7):
    """From `available_layers`, pick k whose depths are closest to evenly spaced
    targets in [1, n_total]. Greedy nearest-neighbor; ties broken by smaller idx.

    Returns a sorted list of length min(k, len(available_layers)).
    """
    available = sorted(set(int(item) for item in available_layers))
    if len(available) <= k:
        return available
    targets = np.linspace(1, n_total, k)
    selected = []
    selected_set = set()
    for t in targets:
        candidates = [item for item in available if item not in selected_set]
        if not candidates:
            break
        best = min(candidates, key=lambda item: (abs(item - t), item))
        selected.append(best)
        selected_set.add(best)
    return sorted(selected)


def select_depth_k_per_model(items, k, model_info):
    """For each model in `items`, pick k depth-k layers using the INTERSECTION
    of layers available across subjects.

    Heterogeneous coverage (some subjects extracted more layers than others)
    used to cause two silent failure modes when this function aggregated
    layers as a UNION across subjects:
      1. depth-k could pick a layer that only some subjects have, so per_model
         selection later averaged over fewer subjects than expected at that
         layer.
      2. when that layer happened to win per_model, the nifti writer would
         emit "No file for sub-XX layer N" warnings and silently produce
         niftis only for the subset of subjects who had that layer.

    Now we only consider layers present for EVERY subject in `items`, and
    warn about which subjects had extras (so you can decide whether to
    homogenize or accept the restriction). The restricted set is what the
    log calls "common".

    Returns: dict {model_name: set of int layer indices to keep}.
    """
    from collections import defaultdict

    by_model_subj = defaultdict(lambda: defaultdict(set))
    for it in items:
        by_model_subj[it["model"]][it["subject"]].add(it["layer_idx"])

    out = {}
    for model in sorted(by_model_subj):
        per_subj = by_model_subj[model]
        subjects_for_model = sorted(per_subj)
        common_layers = set.intersection(*per_subj.values())
        union_layers = set.union(*per_subj.values())
        extras = union_layers - common_layers

        n_total = model_info.get(model, (None,))[0]

        if extras:
            # Surface which subjects have the extras — first 3 examples is enough
            extras_by_subj = {
                s: sorted(per_subj[s] - common_layers)
                for s in subjects_for_model
                if per_subj[s] - common_layers
            }
            sample = list(extras_by_subj.items())[:3]
            sample_str = "; ".join(
                f"{s} has +{exs[:6]}{'...' if len(exs) > 6 else ''}"
                for s, exs in sample
            )
            n_with_extras = len(extras_by_subj)
            logging.warning(
                f"  depth-{k} {model}: {len(extras)} layer(s) present for only "
                f"{n_with_extras}/{len(subjects_for_model)} subjects "
                f"({sample_str}{'; ...' if n_with_extras > 3 else ''}); "
                f"restricting to {len(common_layers)}-layer intersection"
            )

        if n_total is None:
            logging.warning(
                f"  depth-{k}: model {model!r} not in MODEL_INFO (no n_layers); "
                f"keeping all {len(common_layers)} common layers"
            )
            out[model] = set(common_layers)
            continue

        picked = pick_depth_k_layers(common_layers, n_total, k)
        out[model] = set(picked)
        depths = [(item - 1) / (n_total - 1) if n_total > 1 else 0.0 for item in picked]
        depth_str = "  ".join(f"L{item}({d:.2f})" for item, d in zip(picked, depths))
        suffix = ""
        if len(picked) < k:
            suffix = f"  ⚠ only {len(picked)}/{k} available"
        elif depths and depths[-1] < 0.99:
            suffix = (
                f"  ⚠ max depth {depths[-1]:.2f} — high-depth layers missing from data"
            )
        logging.info(
            f"  depth-{k}  {model:<24} (n={n_total:>2}, "
            f"common={len(common_layers):>2}): {depth_str}{suffix}"
        )
    return out


# =============================================================================
# Atlas resampling
# =============================================================================


def get_resampled_atlas(atlas_path, ref_mask_shape, ref_affine):
    """Resample AAL atlas to match the encoding mask grid."""
    atlas_img = nib.load(atlas_path)
    ref_img = nib.Nifti1Image(np.zeros(ref_mask_shape, dtype=np.float32), ref_affine)
    try:
        # Newer nilearn
        resampled = resample_to_img(
            atlas_img, ref_img, interpolation="nearest", force_resample=True
        )
    except TypeError:
        # Older nilearn
        resampled = resample_to_img(atlas_img, ref_img, interpolation="nearest")
    return resampled.get_fdata().astype(np.int32)


# =============================================================================
# Per-file processor: load npz, map to ROIs, return rows
# =============================================================================

# =============================================================================
# Common-mask intersection (mirrors load_and_parse_data_baselines.py)
# =============================================================================


def collect_subject_masks(items):
    """Load one representative mask per subject (from any of their files).

    Returns: {subject: (mask: bool ndarray, mask_affine: ndarray, mask_sum: int)}
    """
    subject_masks = {}
    for it in items:
        s = it["subject"]
        with np.load(it["filepath"], allow_pickle=True) as d:
            mask = np.asarray(d["mask"], dtype=bool)
            affine = np.asarray(d["mask_affine"])
        if s in subject_masks:
            reference_mask, reference_affine, _ = subject_masks[s]
            if not np.array_equal(mask, reference_mask) or not np.allclose(
                affine, reference_affine
            ):
                raise ValueError(
                    f"Inconsistent encoding masks for {s}: {it['filepath']}"
                )
        else:
            subject_masks[s] = (mask, affine, int(mask.sum()))
    return subject_masks


def build_common_mask_and_indexing(subject_masks):
    """Bitwise-AND masks across subjects; build per-subject re-indexing arrays.

    Returns:
        common_mask:    bool ndarray, same shape as subject masks
        common_affine:  affine of the (consistent) input grid
        common_to_subj: {subject: int64 ndarray of length n_common}
                        common_to_subj[s][j] = row-major position of the j-th
                        common-mask voxel inside subject s's flat mask order.
    """
    subjects = sorted(subject_masks.keys())
    if not subjects:
        raise ValueError("No subject masks provided")
    ref_mask, ref_affine, _ = subject_masks[subjects[0]]
    for s in subjects[1:]:
        m, a, _ = subject_masks[s]
        if m.shape != ref_mask.shape:
            raise ValueError(
                f"Subject {s} mask shape {m.shape} != reference {ref_mask.shape}; "
                f"all subjects must share the same MNI grid."
            )
        if not np.allclose(a, ref_affine, atol=1e-4):
            raise ValueError(f"Subject {s} mask affine differs from the common grid")

    # AND masks together. Bool ops are cheap; this is ~30 × 1MB.
    common_mask = ref_mask.copy()
    for s in subjects[1:]:
        common_mask &= subject_masks[s][0]

    n_common = int(common_mask.sum())
    if n_common == 0:
        raise ValueError(f"Intersection mask is empty across {len(subjects)} subjects.")

    # Build int64 translation arrays. Trick: for each subject, fill a 3D
    # volume so that volume[True_voxel_i] = i (row-major), then index it at
    # the common-mask coordinates. Since common ⊆ subject by construction,
    # every result is non-negative.
    common_indices_3d = np.argwhere(common_mask)  # (n_common, 3), row-major
    common_to_subj = {}
    for s in subjects:
        m, _, n_subj = subject_masks[s]
        flat_idx_vol = np.full(m.shape, -1, dtype=np.int64)
        flat_idx_vol[m] = np.arange(n_subj, dtype=np.int64)
        c2s = flat_idx_vol[
            common_indices_3d[:, 0], common_indices_3d[:, 1], common_indices_3d[:, 2]
        ]
        if (c2s < 0).any():
            raise RuntimeError(f"Internal error: common mask not subset of {s} mask")
        common_to_subj[s] = c2s

    per_subj_counts = {s: subject_masks[s][2] for s in subjects}
    logging.info(
        f"Common mask: {n_common} voxels (intersection of {len(subjects)} subjects). "
        f"Per-subject voxel counts (min={min(per_subj_counts.values())}, "
        f"max={max(per_subj_counts.values())}, common/min="
        f"{n_common / max(min(per_subj_counts.values()), 1):.3f})"
    )
    return common_mask, ref_affine, common_to_subj


def precompute_roi_voxel_masks(common_mask, common_affine, atlas_path, roi_lateralized):
    """Resample atlas to common-mask grid; build {(roi, side): bool over n_common}.

    Done once globally so each (roi, side) gets a fixed boolean selector over
    the common-mask voxel order, identical across all subjects/files.
    """
    atlas_3d = get_resampled_atlas(atlas_path, common_mask.shape, common_affine)
    coords = np.argwhere(common_mask)
    voxel_codes = atlas_3d[coords[:, 0], coords[:, 1], coords[:, 2]]

    roi_voxel_masks = {}
    for (roi_short, side), codes in roi_lateralized.items():
        m = np.isin(voxel_codes, np.asarray(codes, dtype=voxel_codes.dtype))
        if m.any():
            roi_voxel_masks[(roi_short, side)] = m
    return roi_voxel_masks


# =============================================================================
# Per-file processor: load npz, project to common-mask voxels, return rows
# =============================================================================
#
# Worker globals — set once per worker process via _worker_init (initializer=).
# This avoids pickling the (potentially large) shared dicts on every submit().

_W_ROI_MASKS = None  # {(roi, side): bool ndarray of length n_common}
_W_COMMON_TO_SUBJ = None  # {subject: int64 ndarray of length n_common}
_W_EXPECTED_SUMS = None  # {subject: int} — expected mask.sum() for sanity check


def _worker_init(roi_voxel_masks, common_to_subj, expected_sums):
    global _W_ROI_MASKS, _W_COMMON_TO_SUBJ, _W_EXPECTED_SUMS
    _W_ROI_MASKS = roi_voxel_masks
    _W_COMMON_TO_SUBJ = common_to_subj
    _W_EXPECTED_SUMS = expected_sums


def process_one_file(args):
    """Worker: load one .npz, project to common-mask voxels, compute per-ROI means."""
    item, bands_to_emit = args

    f = item["filepath"]
    subject = item["subject"]
    model = item["model"]
    variant = item["variant"]
    stream = item["stream"]
    layer_idx = item["layer_idx"]

    n_layers, family, backbone = MODEL_INFO.get(model, (None, model, None))
    baseline_metadata = item.get("baseline_metadata")
    if baseline_metadata:
        n_layers = baseline_metadata["n_layers"]
        family = baseline_metadata["family"]
        backbone = baseline_metadata.get("backbone")
    # layer_depth: 0 at first layer, 1 at last layer. Matches the convention
    # used in load_and_parse_data_baselines.py and in the depth-k log output.
    # (Old formula `layer_idx / n_layers` gave (1/n, 1] — inconsistent with the
    # depth-k log AND incorrect since the first layer should be at depth 0.)
    layer_depth = (
        baseline_metadata["layer_depth"]
        if baseline_metadata
        else ((layer_idx - 1) / (n_layers - 1) if n_layers and n_layers > 1 else None)
    )

    try:
        data = np.load(f, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load encoding result {f}") from e

    # Sanity check: this file's mask must agree with the per-subject reference,
    # otherwise the per-subject re-indexing is invalid. Cheap version: compare
    # popcount; full equality check would be n_subj_voxels comparisons.
    file_mask = np.asarray(data["mask"], dtype=bool)
    file_sum = int(file_mask.sum())
    expected = _W_EXPECTED_SUMS.get(subject) if _W_EXPECTED_SUMS else None
    if expected is not None and file_sum != expected:
        raise ValueError(
            f"Mask mismatch in {f}: {file_sum} voxels, subject reference {expected}"
        )

    c2s = _W_COMMON_TO_SUBJ[subject]  # (n_common,) int64

    rows = []
    for band in bands_to_emit:
        perf_key = f"performances_{band}"
        if perf_key not in data.files:
            continue
        perf_subj = data[perf_key]  # (n_subj_voxels, n_partitions)
        perf_common = perf_subj[c2s, :]  # (n_common,        n_partitions)
        n_partitions = perf_common.shape[1]

        for (roi_short, side), vox_mask in _W_ROI_MASKS.items():
            n_vox = int(vox_mask.sum())
            if n_vox == 0:
                continue
            for partition in range(n_partitions):
                vals = perf_common[vox_mask, partition]
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "stream": stream,
                        "fit_condition": item["fit_condition"],
                        "family": family,
                        "backbone": backbone,
                        "n_layers": n_layers,
                        "layer_idx": layer_idx,
                        "layer_depth": round(layer_depth, 4)
                        if layer_depth is not None
                        else None,
                        "band": band,
                        "ROI": roi_short,
                        "side": side,
                        "subject": subject,
                        "partition": partition,
                        "performance": float(np.mean(vals)),
                        "n_voxels": n_vox,
                    }
                )

    return rows


# =============================================================================
# Main
# =============================================================================


def load_baseline_map(path):
    """Require declared layer semantics; numeric historical EZ labels are ambiguous."""
    mappings = json.loads(Path(path).read_text())
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("Baseline map must be a nonempty object keyed by feature name")
    for feature, metadata in mappings.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", feature):
            raise ValueError(f"Invalid baseline feature name: {feature!r}")
        required = {"model", "family", "layer_idx", "n_layers", "layer_depth"}
        if not isinstance(metadata, dict) or required - metadata.keys():
            raise ValueError(f"Baseline {feature} needs {sorted(required)}")
        if not isinstance(metadata["layer_idx"], int) or metadata["layer_idx"] < 0:
            raise ValueError(f"Invalid layer index for {feature}")
        if not isinstance(metadata["n_layers"], int) or metadata["n_layers"] < 1:
            raise ValueError(f"Invalid layer count for {feature}")
        if not 0 <= metadata["layer_depth"] <= 1:
            raise ValueError(f"Invalid normalized layer depth for {feature}")
    return mappings


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        nargs="+",
        help="One or more dirs containing sub-XX/encoding_results_*.npz. "
        "When multiple are given they are walked in order and any "
        "duplicate (subject, model, variant, stream, layer) is kept "
        "only from the first directory it appears in. Useful for "
        "merging e.g. ./encoding_data and ./shuffled_encoding_data "
        "in a single run.",
    )
    parser.add_argument(
        "--atlas",
        required=True,
        type=Path,
        help="AAL atlas .nii path (e.g. ROI_MNI_V4.nii)",
    )
    parser.add_argument(
        "--atlas-labels",
        default=None,
        type=Path,
        help="Path to ROI_MNI_V4.txt (AAL label table). "
        "If omitted, derived from --atlas by swapping the "
        ".nii(.gz) extension for .txt.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--baseline-map",
        type=Path,
        help="JSON mapping baseline feature names to explicit model/layer metadata",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter to specific models (default: all found)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=["all", "compressed"],
        help="Filter to specific variants (default: both)",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=None,
        choices=["main", "attn", "mlp"],
        help="Filter to specific streams (default: all 3)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Filter to specific subjects (default: all found)",
    )
    parser.add_argument(
        "--fit-condition",
        choices=["main-only", "with-nuisance"],
        default=None,
        help="Select fitted nuisance setting, distinct from --bands. Required "
        "when inputs contain both main-only and nuisance fits; default accepts "
        "one consistent condition and records it in the CSV.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["main", "full"],
        choices=["main", "full", "identity", "button", "time"],
        help="Which bands to emit rows for (default: main, full)",
    )
    parser.add_argument(
        "--allowed-layers",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to specific layer indices applied to ALL models "
        "(e.g. --allowed-layers 1 5 11 16 21 27 32). Useful for "
        "enforcing depth-7 (or other) subsampling uniformly.",
    )
    parser.add_argument(
        "--allowed-layers-per-model",
        nargs="+",
        default=None,
        metavar="MODEL=L1,L2,...",
        help="Per-model layer restriction, e.g. "
        "--allowed-layers-per-model qwen35_9b=1,5,11,16,21,27,32 "
        "qwen35_27b=1,11,21,32,43,53,64. Overrides --allowed-layers "
        "for the listed models; unlisted models keep all layers.",
    )
    parser.add_argument(
        "--depth-k",
        type=int,
        default=None,
        metavar="K",
        help="Pick K layers per model whose normalized depths "
        "(layer_idx-1)/(n_layers-1) are closest to evenly-spaced "
        "targets in [0, 1]. Applied AFTER --allowed-layers / "
        "--allowed-layers-per-model. Typical: --depth-k 7. "
        "Models not in MODEL_INFO are kept as-is (all layers).",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel worker processes (default: 4)"
    )
    parser.add_argument(
        "--nifti-dir",
        type=Path,
        default=None,
        help="If set, also emit one nifti per (model, subject) at the "
        "subject's best layer (selected by whole-brain mean of "
        "--nifti-band r). Folder layout: {nifti_dir}/{model}/{subject}_layer{N}.nii.gz",
    )
    parser.add_argument(
        "--nifti-models",
        nargs="+",
        default=None,
        help="Restrict nifti output to these models (default: same "
        "set as the CSV). Useful when you want a complete CSV "
        "across all models but niftis for only a subset. "
        "Example: --nifti-models qwen35_35b_a3b dsv4_pro",
    )
    parser.add_argument(
        "--nifti-band",
        default="main",
        choices=["main", "full", "identity", "button", "time"],
        help="Which band's voxelwise r values to write to nifti (default: main)",
    )
    parser.add_argument(
        "--nifti-selection",
        default="per_subject",
        choices=["per_subject", "per_model", "per_voxel"],
        help="Best-layer selection rule: "
        "per_subject (each subject's own best layer), "
        "per_model (group-mean best layer applied to all subjects), "
        'per_voxel (max across layers per voxel — a "max-projection" map). '
        "Default: per_subject.",
    )
    parser.add_argument(
        "--nifti-layers",
        nargs="+",
        type=int,
        default=None,
        help="Restrict nifti best-layer selection to these layer indices. "
        "Useful when CSV is parsed from a mix of dense and sparsely-sampled "
        "sweeps and you want consistent layer choices. "
        "Affects ONLY nifti output, not CSV. Example: --nifti-layers 1 5 11 16 21 27 32",
    )
    args = parser.parse_args()

    # Parse --allowed-layers-per-model into a dict
    allowed_per_model = None
    if args.allowed_layers_per_model:
        allowed_per_model = {}
        for entry in args.allowed_layers_per_model:
            if "=" not in entry:
                logging.error(
                    f"Invalid --allowed-layers-per-model entry: {entry!r}. "
                    f"Expected format: MODEL=L1,L2,L3"
                )
                return
            model, layers_str = entry.split("=", 1)
            layers = set()
            for tok in layers_str.split(","):
                tok = tok.strip()
                if tok:
                    layers.add(int(tok))
            allowed_per_model[model] = layers
        logging.info(
            f"  --allowed-layers-per-model: "
            f"{ {m: sorted(ls) for m, ls in allowed_per_model.items()} }"
        )
    allowed_layers_set = set(args.allowed_layers) if args.allowed_layers else None
    if allowed_layers_set:
        logging.info(
            f"  --allowed-layers (applied to unlisted models): {sorted(allowed_layers_set)}"
        )

    # Discovery
    items = find_npz_files(
        args.results_dir,
        models=set(args.models) if args.models else None,
        variants=set(args.variants) if args.variants else None,
        streams=set(args.streams) if args.streams else None,
        subjects=set(args.subjects) if args.subjects else None,
        allowed_layers=allowed_layers_set,
        allowed_layers_per_model=allowed_per_model,
        baseline_map=load_baseline_map(args.baseline_map)
        if args.baseline_map
        else None,
        fit_condition=args.fit_condition,
    )
    if not items:
        raise ValueError(
            f"No .npz files found under {args.results_dir} matching filters"
        )

    # Optional: depth-k layer selection (per-model, post-discovery)
    if args.depth_k:
        logging.info(f"Applying depth-{args.depth_k} layer selection per model...")
        depth_k_keep = select_depth_k_per_model(items, args.depth_k, MODEL_INFO)
        before = len(items)
        items = [it for it in items if it["layer_idx"] in depth_k_keep[it["model"]]]
        logging.info(f"  depth-{args.depth_k} filter: {before} → {len(items)} files")
        if not items:
            raise ValueError("No files remain after depth-k filter")

    logging.info(f"Found {len(items)} files to process")
    logging.info(f"  models:   {sorted(set(i['model'] for i in items))}")
    logging.info(f"  variants: {sorted(set(i['variant'] for i in items))}")
    logging.info(f"  streams:  {sorted(set(i['stream'] for i in items))}")
    logging.info(f"  fit condition: {items[0]['fit_condition']}")
    logging.info(f"  subjects: {sorted(set(i['subject'] for i in items))}")
    logging.info(f"  bands:    {args.bands}")

    # Build the ROI grouping once from the AAL label table.
    atlas_labels = args.atlas_labels
    if atlas_labels is None:
        atlas_labels = Path(_default_atlas_labels_path(args.atlas))
    roi_lateralized = build_lateralized_roi_codes(atlas_labels)
    logging.info(
        f"  Loaded {len(roi_lateralized)} (ROI, side) entries from {atlas_labels}"
    )

    # =========================================================================
    # Compute the cross-subject intersection mask, then precompute (over
    # common-mask voxel order) the per-(roi, side) boolean selectors and the
    # per-subject re-indexing arrays. This mirrors load_and_parse_data_baselines.py
    # and ensures that every (ROI, side) refers to the same set of voxels for
    # every subject — making n_voxels constant across subjects within an ROI
    # and group-level statistics over voxel sets well-defined.
    # =========================================================================
    logging.info("Collecting subject masks for intersection...")
    subject_masks = collect_subject_masks(items)
    if not subject_masks:
        raise ValueError("No subject masks could be loaded")
    common_mask, common_affine, common_to_subj = build_common_mask_and_indexing(
        subject_masks
    )
    roi_voxel_masks = precompute_roi_voxel_masks(
        common_mask, common_affine, str(args.atlas), roi_lateralized
    )
    expected_sums = {s: subject_masks[s][2] for s in subject_masks}
    logging.info(
        f"  Active (ROI, side) entries (≥1 voxel in common mask): "
        f"{len(roi_voxel_masks)} / {len(roi_lateralized)}"
    )
    dropped = set(roi_lateralized) - set(roi_voxel_masks)
    if dropped:
        logging.warning(f"  Dropped (no common-mask coverage): {sorted(dropped)}")

    # Build worker arg list — only per-file metadata + bands now; everything
    # shared lives in worker globals set by _worker_init.
    worker_args = [(item, args.bands) for item in items]

    all_rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(roi_voxel_masks, common_to_subj, expected_sums),
        ) as exe:
            futures = {exe.submit(process_one_file, wa): wa for wa in worker_args}
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="processing"
            ):
                rows = fut.result()  # propagate failures; do not publish a partial CSV
                all_rows.extend(rows)
    else:
        # Serial — set the worker globals in this process directly.
        _worker_init(roi_voxel_masks, common_to_subj, expected_sums)
        for wa in tqdm(worker_args, desc="processing"):
            rows = process_one_file(wa)
            all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No ROI rows emitted; check atlas codes and file structure")

    df = pd.DataFrame(all_rows)
    logging.info(f"Output shape: {df.shape}")
    logging.info(
        f"Cells: {df.groupby(['model', 'variant', 'stream', 'subject', 'band']).size().head(10)}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logging.info(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")

    # =========================================================================
    # Optional: emit best-layer niftis
    # =========================================================================
    if args.nifti_dir is not None:
        nifti_items = items
        if args.nifti_models:
            requested = set(args.nifti_models)
            available = set(it["model"] for it in items)
            unrecognized = requested - available
            if unrecognized:
                logging.warning(
                    f"--nifti-models contains models with no items in the CSV: "
                    f"{sorted(unrecognized)}. Available: {sorted(available)}"
                )
            nifti_items = [it for it in items if it["model"] in requested]
            logging.info(
                f"Nifti scope: {len(nifti_items)}/{len(items)} files "
                f"({sorted(set(it['model'] for it in nifti_items))})"
            )
        if not nifti_items:
            logging.warning(
                "No items match --nifti-models filter; skipping nifti output"
            )
        else:
            write_best_layer_niftis(
                nifti_items,
                args.nifti_dir,
                args.nifti_band,
                args.nifti_selection,
                common_to_subj=common_to_subj,
            )


def write_best_layer_niftis(
    items, nifti_dir, nifti_band, selection, common_to_subj=None
):
    """For each (model, subject), pick best layer and write a 3D nifti of voxelwise r.

    Selection rules:
        per_subject: each subject's own best layer (highest mean over voxels of r)
        per_model:   group-mean best layer applied to all subjects of that model
        per_voxel:   for each voxel, take the max r across layers within (model, subject)
                     (no single "best layer" — a max-projection map)

    For 'per_subject' and 'per_model' selection, the whole-brain mean used to score
    each layer is computed over the common (intersection) mask voxels — passed via
    `common_to_subj`. This makes layer scores comparable across subjects for the
    'per_model' rule. The output nifti volume is written on the subject's own mask
    (visualization is per-subject, so subject coverage is the natural domain).
    """
    nifti_dir = Path(nifti_dir)
    nifti_dir.mkdir(parents=True, exist_ok=True)

    perf_key = f"performances_mean_{nifti_band}"
    fallback_key = (
        f"performances_{nifti_band}"  # if mean not stored, average partitions
    )

    # Group items by (model, variant, stream, subject)
    # We typically only do this for one (variant, stream) at a time, but support multi
    from collections import defaultdict

    by_msvs = defaultdict(list)  # (model, variant, stream, subject) -> [items]
    for it in items:
        by_msvs[(it["model"], it["variant"], it["stream"], it["subject"])].append(it)

    # Step 1: compute whole-brain mean r per (model, variant, stream, subject, layer)
    # — restricted to common-mask voxels for cross-subject comparability.
    logging.info(
        "Computing whole-brain mean r per (model, subject, layer) for nifti selection..."
    )
    wholebrain = {}  # (m, v, s, subj, layer) -> mean r
    for (m, v, s, subj), grp_items in by_msvs.items():
        c2s = common_to_subj.get(subj) if common_to_subj else None
        for it in grp_items:
            try:
                d = np.load(it["filepath"], allow_pickle=True)
                if perf_key in d.files:
                    voxel_r = d[perf_key]  # (n_subj_voxels,)
                else:
                    voxel_r = d[fallback_key].mean(axis=1)
                # Restrict to common mask voxels (if available) for fairness
                if c2s is not None:
                    voxel_r_common = voxel_r[c2s]
                else:
                    voxel_r_common = voxel_r
                wholebrain[(m, v, s, subj, it["layer_idx"])] = float(
                    np.mean(voxel_r_common)
                )
            except Exception as e:
                logging.warning(f"Failed reading {it['filepath']}: {e}")

    # Step 2: pick best layer per the selection rule
    best_layer = {}  # (m, v, s, subj) -> layer_idx
    if selection == "per_subject":
        for (m, v, s, subj), grp_items in by_msvs.items():
            scores = {
                it["layer_idx"]: wholebrain.get(
                    (m, v, s, subj, it["layer_idx"]), -np.inf
                )
                for it in grp_items
            }
            best_layer[(m, v, s, subj)] = max(scores, key=scores.get)
    elif selection == "per_model":
        # Pool across subjects within (model, variant, stream)
        by_mvs = defaultdict(
            lambda: defaultdict(list)
        )  # (m,v,s) -> layer -> [r values]
        for (m, v, s, subj), grp_items in by_msvs.items():
            for it in grp_items:
                key = (m, v, s, subj, it["layer_idx"])
                if key in wholebrain:
                    by_mvs[(m, v, s)][it["layer_idx"]].append(wholebrain[key])
        # For each (m,v,s), find layer with highest mean across subjects
        best_per_mvs = {}
        for mvs, layer_map in by_mvs.items():
            mean_per_layer = {item: np.mean(rs) for item, rs in layer_map.items()}
            best_per_mvs[mvs] = max(mean_per_layer, key=mean_per_layer.get)
        # Apply to all subjects
        for m, v, s, subj in by_msvs:
            best_layer[(m, v, s, subj)] = best_per_mvs[(m, v, s)]
    # per_voxel handled separately in step 3

    # Step 3: write niftis
    n_written = 0
    if selection == "per_voxel":
        # Per-voxel max across layers — load all layers, stack, take max
        for (m, v, s, subj), grp_items in tqdm(by_msvs.items(), desc="writing niftis"):
            try:
                voxel_max = None
                mask_3d = None
                mask_affine = None
                for it in grp_items:
                    d = np.load(it["filepath"], allow_pickle=True)
                    if mask_3d is None:
                        mask_3d = d["mask"]
                        mask_affine = d["mask_affine"]
                    if perf_key in d.files:
                        v_r = d[perf_key]
                    else:
                        v_r = d[fallback_key].mean(axis=1)
                    if voxel_max is None:
                        voxel_max = v_r.copy()
                    else:
                        voxel_max = np.maximum(voxel_max, v_r)
                _write_nifti(
                    nifti_dir,
                    m,
                    subj,
                    "maxlayer",
                    nifti_band,
                    voxel_max,
                    mask_3d,
                    mask_affine,
                    v=v,
                    s=s,
                )
                n_written += 1
            except Exception as e:
                logging.warning(f"Failed nifti for {m}/{subj}: {e}")
    else:
        for (m, v, s, subj), layer_idx in tqdm(
            best_layer.items(), desc="writing niftis"
        ):
            # Find the matching item to get the file path
            target = next(
                (it for it in by_msvs[(m, v, s, subj)] if it["layer_idx"] == layer_idx),
                None,
            )
            if target is None:
                logging.warning(f"No file for {m}/{subj} layer {layer_idx}")
                continue
            try:
                d = np.load(target["filepath"], allow_pickle=True)
                if perf_key in d.files:
                    voxel_r = d[perf_key]
                else:
                    voxel_r = d[fallback_key].mean(axis=1)
                _write_nifti(
                    nifti_dir,
                    m,
                    subj,
                    layer_idx,
                    nifti_band,
                    voxel_r,
                    d["mask"],
                    d["mask_affine"],
                    v=v,
                    s=s,
                )
                n_written += 1
            except Exception as e:
                logging.warning(f"Failed nifti for {m}/{subj} layer {layer_idx}: {e}")

    logging.info(f"Wrote {n_written} niftis under {nifti_dir}")


def _write_nifti(
    nifti_dir,
    model,
    subject,
    layer_idx,
    band,
    voxel_r,
    mask_3d,
    mask_affine,
    v="all",
    s="main",
):
    """Reconstruct a 3D volume from (n_voxels,) r values and write nifti.gz."""
    model_dir = nifti_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    vol = np.zeros(mask_3d.shape, dtype=np.float32)
    vol[mask_3d.astype(bool)] = voxel_r.astype(np.float32)

    img = nib.Nifti1Image(vol, affine=mask_affine)

    # Filename includes layer + band for clarity
    if isinstance(layer_idx, int):
        fname = f"{subject}_{v}_{s}_layer{layer_idx:02d}_{band}_r.nii.gz"
    else:  # 'maxlayer' for per_voxel
        fname = f"{subject}_{v}_{s}_{layer_idx}_{band}_r.nii.gz"
    out = model_dir / fname
    nib.save(img, str(out))


if __name__ == "__main__":
    main()
