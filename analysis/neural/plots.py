"""Encoding accuracy bar plots (per-subject best-layer Pearson r, band=main).

Three figure types, each in baselines + llms variants:

  aggregate   -- one bar per model, best layer across all ROIs
  rois        -- per-ROI panel grid, one bar per model in each panel
  groups      -- ROIs grouped into functional regions, clustered bars

Usage:
    python analysis/neural/plots.py            # all
    python analysis/neural/plots.py aggregate
    python analysis/neural/plots.py rois
    python analysis/neural/plots.py groups
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Matplotlib style (matches plot_behavioural.py) ----------------------------
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 24
plt.rcParams["axes.labelsize"] = 26
plt.rcParams["axes.titlesize"] = 26
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linewidth"] = 0.5

# -- Colors and labels (from plot_behavioural.py) ------------------------------

ALL_COLORS = {
    "ddqn": "#CB181D",
    "ez": "#7A0177",
    "hrr": "#31A354",
    "dsv32": "#E6550D",
    "dsv4_flash": "#FD8D3C",
    "dsv4_pro": "#FDAE6B",
    "qwen35_9b": "#6A51A3",
    "qwen35_27b": "#807DBA",
    "qwen35_35b_a3b": "#9E9AC8",
    "qwen35_122b_a10b": "#BCBDDC",
}

LABELS = {
    "ddqn": "DDQN",
    "ez": "EfficientZero",
    "hrr": "HRR",
    "dsv32": "V3.2",
    "dsv4_flash": "V4-Flash",
    "dsv4_pro": "V4-Pro",
    "qwen35_9b": "Q-9B",
    "qwen35_27b": "Q-27B",
    "qwen35_35b_a3b": "Q-35B",
    "qwen35_122b_a10b": "Q-122B",
}

LABELS_LONG = {
    "ddqn": "DDQN",
    "ez": "EfficientZero",
    "hrr": "HRR",
    "dsv32": "DeepSeek-V3.2",
    "dsv4_flash": "DeepSeek V4-Flash",
    "dsv4_pro": "DeepSeek V4-Pro",
    "qwen35_9b": "Qwen3.5-9B",
    "qwen35_27b": "Qwen3.5-27B",
    "qwen35_35b_a3b": "Qwen3.5-35B",
    "qwen35_122b_a10b": "Qwen3.5-122B",
}

BASELINES_ORDER = ["ddqn", "ez", "hrr", "dsv32", "qwen35_35b_a3b"]

LLMS_ORDER = [
    "qwen35_9b",
    "qwen35_27b",
    "qwen35_35b_a3b",
    "qwen35_122b_a10b",
    "dsv32",
    "dsv4_flash",
    "dsv4_pro",
]

# ROIs grouped by functional region (order from Sreejan's script)
ROI_GROUPS: dict[str, list[str]] = {
    "Frontal": ["IFGtriang", "IFGoperc", "MFG", "SFG", "OFC"],
    "Motor": ["PreCG", "SMA", "PoCG", "ROL"],
    "Parietal": ["IPG", "AG", "SMG", "PCUN"],
    "Visual": ["IOG", "MOG", "SOG", "FFG", "MTG"],
    "Early Visual": ["LING", "CAL", "CUN"],
    "Striatal": ["Caudate", "Putamen"],
}

ALL_ROIS = [roi for rois in ROI_GROUPS.values() for roi in rois]

# HRR only predicts frontal and parietal regions (confirmed by Sreejan/Momchil)
MODEL_VALID_GROUPS: dict[str, list[str]] = {}


def _valid_groups(model: str) -> list[str]:
    return MODEL_VALID_GROUPS.get(model, list(ROI_GROUPS.keys()))


def _valid_rois(model: str) -> list[str]:
    groups = _valid_groups(model)
    return [roi for g in groups for roi in ROI_GROUPS[g]]


OUTPUT_DIR = Path("out/figures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def model_label(model: str, *, long: bool = False) -> str:
    """Keep historical labels; new conditions retain their explicit identity."""
    return (LABELS_LONG if long else LABELS).get(model, model)


def model_color(model: str) -> str:
    if model in ALL_COLORS:
        return ALL_COLORS[model]
    # Unlike Python's hash(), this is stable across processes and CSV ordering.
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]
    index = int.from_bytes(hashlib.sha256(model.encode()).digest()[:4], "big")
    return palette[index % len(palette)]


def pick_variant(model: str, df: pd.DataFrame) -> str | None:
    avail = df[df["model"] == model]["variant"].dropna().unique()
    if len(avail) == 0:
        return None
    if model.startswith("qwen") and "all" in avail:
        return "all"
    if model.startswith("ds") and "compressed" in avail:
        return "compressed"
    if model not in LABELS and len(avail) > 1:
        raise ValueError(
            f"multiple variants for {model}: {list(avail)}; select --variant"
        )
    return avail[0]


def _filter_model(df: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = df[df["model"] == model]
    variant = pick_variant(model, sub)
    if variant is not None:
        sub = sub[sub["variant"] == variant]
    if len(sub) == 0:
        raise ValueError(f"no data for {model} (variant={variant})")
    return sub


def per_subject_best_layer(df: pd.DataFrame, model: str) -> pd.Series:
    sub = _filter_model(df, model)
    per_layer = (
        sub.groupby(["subject", "layer_idx"])["performance"].mean().reset_index()
    )
    best = per_layer.loc[per_layer.groupby("subject")["performance"].idxmax()]
    return best.set_index("subject")["performance"]


def per_subject_best_layer_roi(
    df: pd.DataFrame,
    model: str,
    roi: str,
) -> pd.Series:
    sub = _filter_model(df, model)
    sub = sub[sub["ROI"] == roi]
    if len(sub) == 0:
        return pd.Series(dtype=float)
    per_layer = (
        sub.groupby(["subject", "layer_idx"])["performance"].mean().reset_index()
    )
    best = per_layer.loc[per_layer.groupby("subject")["performance"].idxmax()]
    return best.set_index("subject")["performance"]


def per_subject_best_layer_rois(
    df: pd.DataFrame,
    model: str,
    rois: list[str],
) -> pd.Series:
    sub = _filter_model(df, model)
    sub = sub[sub["ROI"].isin(rois)]
    if len(sub) == 0:
        return pd.Series(dtype=float)
    per_layer = (
        sub.groupby(["subject", "layer_idx"])["performance"].mean().reset_index()
    )
    best = per_layer.loc[per_layer.groupby("subject")["performance"].idxmax()]
    return best.set_index("subject")["performance"]


def _scores_from_series(scores: pd.Series, model: str) -> dict:
    return {
        "model": model,
        "label": model_label(model),
        "label_long": model_label(model, long=True),
        "color": model_color(model),
        "mean": scores.mean(),
        "sem": scores.std(ddof=1) / np.sqrt(len(scores)),
        "n": len(scores),
    }


# ---------------------------------------------------------------------------
# Aggregate bars
# ---------------------------------------------------------------------------


def plot_aggregate(
    df: pd.DataFrame,
    models: list[str],
    output: Path,
    title: str,
) -> None:
    rows = [
        _scores_from_series(per_subject_best_layer_rois(df, m, _valid_rois(m)), m)
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(rows))
    means = [r["mean"] for r in rows]

    ax.bar(
        xs,
        means,
        yerr=[r["sem"] for r in rows],
        color=[r["color"] for r in rows],
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        width=0.7,
        zorder=3,
        error_kw={"linewidth": 1.5},
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label_long"] for r in rows], rotation=30, ha="right")
    ax.set_ylabel(r"Encoding accuracy (Pearson $r$)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    lower = min(0.0, min(means) * 1.30)
    upper = max(0.0, max(means) * 1.30)
    ax.set_ylim(lower, upper if upper > lower else 0.01)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"wrote {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-ROI grid (2 rows: row1 = Frontal/Parietal/Occipito-temporal,
#                        row2 = Early visual/Subcortical)
# ---------------------------------------------------------------------------

ROI_GRID_ROWS = [
    ["Frontal", "Motor", "Parietal"],  # 5 + 4 + 4 = 13
    ["Visual", "Early Visual", "Striatal"],  # 5 + 3 + 2 = 10
]


def _plot_roi_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    models: list[str],
    groups: list[str],
    global_max: float,
) -> None:
    n_models = len(models)
    bar_width = 0.8 / n_models
    roi_centers: list[float] = []
    group_spans: list[tuple[float, float, str]] = []

    pos = 0.0
    for gi, gname in enumerate(groups):
        rois = ROI_GROUPS[gname]
        if gi > 0:
            pos += 1.0
        group_start = pos
        for ri, roi in enumerate(rois):
            center = pos + ri
            roi_centers.append(center)
            for mi, model in enumerate(models):
                x = center + (mi - n_models / 2 + 0.5) * bar_width
                if gname not in _valid_groups(model):
                    continue
                scores = per_subject_best_layer_roi(df, model, roi)
                if scores.empty:
                    continue
                mean = scores.mean()
                sem = (
                    (scores.std(ddof=1) / np.sqrt(len(scores)))
                    if len(scores) > 0
                    else 0.0
                )
                ax.bar(
                    x,
                    mean,
                    yerr=sem,
                    width=bar_width * 0.85,
                    color=model_color(model),
                    edgecolor="black",
                    linewidth=0.4,
                    capsize=2,
                    zorder=3,
                    error_kw={"linewidth": 0.8},
                )
        group_end = pos + len(rois) - 1
        group_spans.append((group_start, group_end, gname))
        pos += len(rois)

    all_rois_flat = [roi for g in groups for roi in ROI_GROUPS[g]]
    ax.set_xticks(roi_centers)
    ax.set_xticklabels(all_rois_flat, rotation=45, ha="right", fontsize=28)
    ax.axhline(0, color="black", linewidth=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.set_ylim(0, global_max * 1.10 if global_max > 0 else 0.01)
    ax.set_xlim(roi_centers[0] - 0.6, roi_centers[-1] + 0.6)

    for si, (start, end, gname) in enumerate(group_spans):
        cx = (start + end) / 2
        ax.text(
            cx,
            1.02,
            gname,
            ha="center",
            va="bottom",
            fontsize=28,
            fontstyle="italic",
            transform=ax.get_xaxis_transform(),
        )
        if si > 0:
            prev_end = group_spans[si - 1][1]
            ax.axvline(
                (prev_end + start) / 2, color="gray", linestyle="--", linewidth=1.0
            )


def _row_max(
    df: pd.DataFrame,
    models: list[str],
    groups: list[str],
) -> float:
    row_max = 0.0
    for gname in groups:
        for roi in ROI_GROUPS[gname]:
            for model in models:
                if gname not in _valid_groups(model):
                    continue
                scores = per_subject_best_layer_roi(df, model, roi)
                if len(scores) > 0:
                    sem = scores.std(ddof=1) / np.sqrt(len(scores))
                    val = scores.mean() + (sem if np.isfinite(sem) else 0.0)
                    row_max = max(row_max, val)
    return row_max


def plot_rois(
    df: pd.DataFrame,
    models: list[str],
    output: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(22, 12),
    )

    for row_idx, groups in enumerate(ROI_GRID_ROWS):
        row_max = _row_max(df, models, groups)
        _plot_roi_panel(axes[row_idx], df, models, groups, row_max)

    axes[0].set_ylabel(r"Encoding accuracy (Pearson $r$)")
    axes[1].set_ylabel(r"Encoding accuracy (Pearson $r$)")

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=model_color(m), edgecolor="black", linewidth=0.6
        )
        for m in models
    ]
    fig.legend(
        handles,
        [model_label(m) for m in models],
        loc="lower center",
        fontsize=20,
        ncol=len(models),
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grouped-by-region
# ---------------------------------------------------------------------------


def _per_subject_selected_layer_roi(
    df: pd.DataFrame,
    model: str,
    roi: str,
    selection: str,
) -> pd.Series:
    """For a single ROI, rank layers by mean across subjects, pick the
    worst/median/best layer, then return per-subject scores at that layer."""
    sub = _filter_model(df, model)
    sub = sub[sub["ROI"] == roi]
    if len(sub) == 0:
        return pd.Series(dtype=float)
    layer_ranking = sub.groupby("layer_idx")["performance"].mean().sort_values()
    if selection == "worst":
        chosen = layer_ranking.index[0]
    elif selection == "median":
        chosen = layer_ranking.index[len(layer_ranking) // 2]
    elif selection == "best":
        chosen = layer_ranking.index[-1]
    else:
        raise ValueError(f"unknown selection: {selection}")
    at_layer = sub[sub["layer_idx"] == chosen]
    return at_layer.groupby("subject")["performance"].mean()


def _group_scores_for_selection(
    df: pd.DataFrame,
    model: str,
    rois: list[str],
    selection: str,
) -> pd.Series:
    """For each ROI pick worst/median/best layer independently, get per-subject
    scores, then average across ROIs -> one value per subject."""
    per_roi = [
        _per_subject_selected_layer_roi(df, model, roi, selection) for roi in rois
    ]
    per_roi = [s for s in per_roi if len(s) > 0]
    if not per_roi:
        return pd.Series(dtype=float)
    combined = pd.concat(per_roi, axis=1)
    return combined.mean(axis=1)


def _model_n_layers(df: pd.DataFrame, model: str) -> int:
    sub = _filter_model(df, model)
    return sub["layer_idx"].nunique()


def plot_groups(
    df: pd.DataFrame,
    models: list[str],
    output: Path,
    title: str,
    best_only: bool = False,
) -> None:
    group_names = list(ROI_GROUPS.keys())
    n_groups = len(group_names)

    all_selections = ["worst", "median", "best"]
    all_hatches = ["\\\\", "//", ""]

    # Per-model: single-layer models get 1 bar, multi-layer get 3 (or 1 if best_only)
    model_nlayers = {m: _model_n_layers(df, m) for m in models}
    model_slots: list[list[tuple[str, str]]] = []
    for m in models:
        if best_only or model_nlayers[m] == 1:
            model_slots.append([("best", "")])
        else:
            model_slots.append(list(zip(all_selections, all_hatches)))

    n_bars = sum(len(s) for s in model_slots)
    bar_width = 0.8 / n_bars

    fig, axes = plt.subplots(
        1,
        n_groups,
        figsize=(3.5 * n_groups, 6),
        sharey=False,
    )

    for gi, gname in enumerate(group_names):
        ax = axes[gi]
        rois = ROI_GROUPS[gname]
        local_min = 0.0
        local_max = 0.0
        bar_idx = 0
        for mi, model in enumerate(models):
            color = model_color(model)
            for sel, hatch in model_slots[mi]:
                if gname not in _valid_groups(model):
                    bar_idx += 1
                    continue
                scores = _group_scores_for_selection(df, model, rois, sel)
                if scores.empty:
                    bar_idx += 1
                    continue
                mean = scores.mean()
                sem = scores.std(ddof=1) / np.sqrt(len(scores))
                local_max = max(local_max, mean + (sem if np.isfinite(sem) else 0.0))
                local_min = min(local_min, mean - (sem if np.isfinite(sem) else 0.0))
                x = (bar_idx - n_bars / 2 + 0.5) * bar_width
                ax.bar(
                    x,
                    mean,
                    yerr=sem,
                    width=bar_width * 0.85,
                    color=color,
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=3,
                    error_kw={"linewidth": 1.0, "capsize": 2},
                )
                bar_idx += 1

        ax.set_title(gname, fontsize=22)
        ax.set_xticks([])
        ax.axhline(0, color="black", linewidth=0.8, zorder=4)
        margin = (local_max - local_min) * 0.10 or 0.01
        ax.set_ylim(local_min - margin, local_max + margin)
        ax.set_axisbelow(True)
        if gi == 0:
            ax.set_ylabel(r"Encoding accuracy (Pearson $r$)")

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=model_color(m), edgecolor="black", linewidth=0.6
        )
        for m in models
    ]
    labels = [model_label(m) for m in models]
    has_triplet = any(len(s) > 1 for s in model_slots)
    if has_triplet:
        for sname, hatch in zip(all_selections, all_hatches):
            handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="gray",
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.6,
                )
            )
            labels.append(sname + " layer")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        fontsize=16,
        ncol=len(handles),
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined encoding groups (baselines + LLMs in one figure)
# ---------------------------------------------------------------------------


def plot_groups_combined(df: pd.DataFrame, output: Path) -> None:
    """Single-row figure: baselines (left) + LLMs (right), one subplot per region.

    Top row (baselines): multi-layer models show worst/median/best via hatching;
    single-layer models (HRR) show one solid bar.
    Bottom row (LLMs): best layer only, solid bars.
    """
    group_names = list(ROI_GROUPS.keys())
    n_groups = len(group_names)

    from matplotlib.gridspec import GridSpec

    all_selections = ["worst", "median", "best"]
    all_hatches = ["\\\\", "//", ""]

    # 4 rows: bars, legend, bars, legend
    fig = plt.figure(figsize=(3.5 * n_groups, 10))
    gs = GridSpec(
        4,
        n_groups,
        figure=fig,
        height_ratios=[1, 0.15, 1, 0.15],
        hspace=0.3,
        wspace=0.3,
    )

    row_configs = [
        (0, BASELINES_ORDER, "Cross-Paradigm", False),
        (2, LLMS_ORDER, "LLMs", True),
    ]

    for gs_row, models, row_title, best_only in row_configs:
        model_nlayers = {m: _model_n_layers(df, m) for m in models}
        model_slots: list[list[tuple[str, str]]] = []
        for m in models:
            if best_only or model_nlayers[m] == 1:
                model_slots.append([("best", "")])
            else:
                model_slots.append(list(zip(all_selections, all_hatches)))

        n_bars = sum(len(s) for s in model_slots)
        bar_width = 0.8 / n_bars

        axes_row = [fig.add_subplot(gs[gs_row, gi]) for gi in range(n_groups)]

        for gi, gname in enumerate(group_names):
            ax = axes_row[gi]
            rois = ROI_GROUPS[gname]
            local_min = 0.0
            local_max = 0.0
            bar_idx = 0
            for mi, model in enumerate(models):
                color = model_color(model)
                for sel, hatch in model_slots[mi]:
                    if gname not in _valid_groups(model):
                        bar_idx += 1
                        continue
                    scores = _group_scores_for_selection(df, model, rois, sel)
                    if scores.empty:
                        bar_idx += 1
                        continue
                    mean = scores.mean()
                    sem = scores.std(ddof=1) / np.sqrt(len(scores))
                    local_max = max(
                        local_max, mean + (sem if np.isfinite(sem) else 0.0)
                    )
                    local_min = min(
                        local_min, mean - (sem if np.isfinite(sem) else 0.0)
                    )
                    x = (bar_idx - n_bars / 2 + 0.5) * bar_width
                    ax.bar(
                        x,
                        mean,
                        yerr=sem,
                        width=bar_width * 0.85,
                        color=color,
                        hatch=hatch,
                        edgecolor="black",
                        linewidth=0.4,
                        zorder=3,
                        error_kw={"linewidth": 1.0, "capsize": 2},
                    )
                    bar_idx += 1

            ax.set_title(gname if gs_row == 0 else "", fontsize=22)
            ax.set_xticks([])
            ax.axhline(0, color="black", linewidth=0.8, zorder=4)
            margin = (local_max - local_min) * 0.10 or 0.01
            ax.set_ylim(local_min - margin, local_max + margin)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            if gi == 0:
                ax.set_ylabel(row_title, fontsize=24, fontweight="bold")

        # Legend row below bars
        legend_ax = fig.add_subplot(gs[gs_row + 1, :])
        legend_ax.axis("off")
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=model_color(m), edgecolor="black", linewidth=0.6
            )
            for m in models
        ]
        labels = [model_label(m) for m in models]
        if not best_only:
            has_triplet = any(model_nlayers[m] > 1 for m in models)
            if has_triplet:
                for sname, hatch in zip(all_selections, all_hatches):
                    handles.append(
                        plt.Rectangle(
                            (0, 0),
                            1,
                            1,
                            facecolor="gray",
                            hatch=hatch,
                            edgecolor="black",
                            linewidth=0.6,
                        )
                    )
                    labels.append(sname + " layer")
        legend_ax.legend(
            handles,
            labels,
            loc="center",
            fontsize=16,
            framealpha=0.9,
            ncol=len(handles),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Permutation tests: trained vs controls at brain-group level
# ---------------------------------------------------------------------------

# Panel A: Qwen 9B trained vs control conditions
PERM_9B_PAIRS = [
    ("qwen35_9b", "qwen35_9b_random", "Random init"),
    ("qwen35_9b", "qwen35_9b_shuf_plays", "Shuffle plays"),
    ("qwen35_9b", "qwen35_9b_shuf_levels", "Shuffle levels"),
    ("qwen35_9b", "qwen35_9b_shuf_games", "Shuffle games"),
]

# Panel B: trained vs random-init across model sizes
PERM_SCALE_PAIRS = [
    ("qwen35_9b", "qwen35_9b_random", "9B"),
    ("qwen35_27b", "qwen35_27b_random", "27B"),
    ("qwen35_35b_a3b", "qwen35_35b_a3b_random", "35B"),
]

CONTROL_COLOR = "#bdc3c7"


def _perm_scores(
    df: pd.DataFrame,
    model: str,
    rois: list[str],
) -> tuple[float, float]:
    scores = per_subject_best_layer_rois(df, model, rois)
    if len(scores) == 0:
        return 0.0, 0.0
    return scores.mean(), scores.std(ddof=1) / np.sqrt(len(scores))


def plot_permutation(df: pd.DataFrame, output: Path) -> None:
    group_names = list(ROI_GROUPS.keys())
    n_groups = len(group_names)

    fig, axes = plt.subplots(
        2,
        n_groups,
        figsize=(3.5 * n_groups, 7),
        sharey=False,
    )

    # Row 0: 9B control types
    for gi, gname in enumerate(group_names):
        ax = axes[0, gi]
        rois = ROI_GROUPS[gname]
        n_pairs = len(PERM_9B_PAIRS)
        bar_width = 0.35

        for pi, (trained, control, label) in enumerate(PERM_9B_PAIRS):
            t_mean, t_sem = _perm_scores(df, trained, rois)
            c_mean, c_sem = _perm_scores(df, control, rois)
            x = pi
            ax.bar(
                x - bar_width / 2,
                t_mean,
                yerr=t_sem,
                width=bar_width,
                color=model_color(trained),
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
                zorder=3,
                error_kw={"linewidth": 1.0},
            )
            ax.bar(
                x + bar_width / 2,
                c_mean,
                yerr=c_sem,
                width=bar_width,
                color=CONTROL_COLOR,
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
                zorder=3,
                error_kw={"linewidth": 1.0},
            )

        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels(
            [p[2] for p in PERM_9B_PAIRS],
            rotation=30,
            ha="right",
            fontsize=20,
        )
        ax.set_title(gname, fontsize=22)
        ax.axhline(0, color="black", linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Row 1: random-init across sizes
    for gi, gname in enumerate(group_names):
        ax = axes[1, gi]
        rois = ROI_GROUPS[gname]
        n_pairs = len(PERM_SCALE_PAIRS)
        bar_width = 0.35

        for pi, (trained, control, label) in enumerate(PERM_SCALE_PAIRS):
            t_mean, t_sem = _perm_scores(df, trained, rois)
            c_mean, c_sem = _perm_scores(df, control, rois)
            x = pi
            ax.bar(
                x - bar_width / 2,
                t_mean,
                yerr=t_sem,
                width=bar_width,
                color=model_color(trained),
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
                zorder=3,
                error_kw={"linewidth": 1.0},
            )
            ax.bar(
                x + bar_width / 2,
                c_mean,
                yerr=c_sem,
                width=bar_width,
                color=CONTROL_COLOR,
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
                zorder=3,
                error_kw={"linewidth": 1.0},
            )

        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels(
            [p[2] for p in PERM_SCALE_PAIRS],
            fontsize=16,
        )
        ax.set_title(gname, fontsize=22)
        ax.axhline(0, color="black", linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Shared ylabel across both rows
    fig.text(
        0.01,
        0.5,
        r"Encoding accuracy (Pearson $r$)",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=26,
    )

    # Shared legend: one entry per trained model + one for control
    seen = {}
    legend_handles = []
    legend_labels = []
    for pairs in [PERM_9B_PAIRS, PERM_SCALE_PAIRS]:
        for trained, _, _ in pairs:
            if trained not in seen:
                seen[trained] = True
                legend_handles.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=model_color(trained),
                        edgecolor="black",
                        linewidth=0.6,
                    )
                )
                legend_labels.append(model_label(trained))
    legend_handles.append(
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=CONTROL_COLOR, edgecolor="black", linewidth=0.6
        )
    )
    legend_labels.append("Control")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        fontsize=18,
        ncol=len(legend_handles),
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def select_encoding_rows(
    df: pd.DataFrame,
    *,
    selection: dict | None = None,
    stream: str | None = None,
    subjects: list[str] | None = None,
    variant: str | None = None,
    fit_condition: str | None = None,
) -> tuple[pd.DataFrame, list[str] | None]:
    """Select declared analysis conditions before any averaging or layer selection."""
    required = {
        "model",
        "variant",
        "subject",
        "layer_idx",
        "ROI",
        "band",
        "performance",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"missing encoding columns: {sorted(missing_columns)}")
    selection_models = None
    band = "main"
    if selection is not None:
        required_config = {
            "subjects",
            "stream",
            "band",
            "fit_condition",
            "model_conditions",
        }
        if required_config - selection.keys():
            raise ValueError(f"selection config requires {sorted(required_config)}")
        for key, override in (
            ("subjects", subjects),
            ("stream", stream),
            ("fit_condition", fit_condition),
        ):
            if override is not None and override != selection[key]:
                raise ValueError(
                    f"--{key.replace('_', '-')} conflicts with --selection-config"
                )
        subjects, stream, fit_condition, band = (
            selection[key] for key in ("subjects", "stream", "fit_condition", "band")
        )
        if (
            not isinstance(subjects, list)
            or not subjects
            or len(subjects) != len(set(subjects))
        ):
            raise ValueError("selection subjects must be a nonempty list of unique IDs")
        if band != "main":
            raise ValueError(
                "these figures describe main-band encoding; selection band must be main"
            )
        conditions = selection["model_conditions"]
        selection_models = [condition["model"] for condition in conditions]
        if not selection_models or len(selection_models) != len(set(selection_models)):
            raise ValueError("model_conditions must declare each model exactly once")
        if variant is not None and any(
            condition["variant"] != variant for condition in conditions
        ):
            raise ValueError(
                "--variant conflicts with per-model variants in --selection-config"
            )
        df = df[df["model"].isin(selection_models)]
    df = df[df["band"] == band].copy()
    if subjects is not None:
        df = df[df["subject"].isin(subjects)]
    elif any(
        str(value).startswith("sub-")
        and str(value)[4:].isdigit()
        and int(str(value)[4:]) <= 11
        for value in df["subject"].unique()
    ) and any(
        str(value).startswith("sub-")
        and str(value)[4:].isdigit()
        and int(str(value)[4:]) >= 12
        for value in df["subject"].unique()
    ):
        raise ValueError(
            "mixed participant cohorts; select --subjects or --selection-config"
        )
    if selection is not None:
        if "stream" not in df:
            raise ValueError("selection config requires a stream column")
        parts = []
        for condition in selection["model_conditions"]:
            # A null declaration selects genuinely unlabeled baseline rows;
            # it must not relabel them as a known LLM feature stream.
            model_rows = df[df["model"] == condition["model"]]
            selected_stream = condition.get("stream", stream)
            match = (
                model_rows["stream"].isna()
                if selected_stream is None
                else model_rows["stream"] == selected_stream
            )
            parts.append(model_rows[match])
        df = pd.concat(parts, ignore_index=True)
    elif stream is not None:
        if "stream" not in df:
            raise ValueError("--stream/selection config requires a stream column")
        df = df[df["stream"] == stream]
    elif "stream" in df and df["stream"].dropna().nunique() > 1:
        raise ValueError("mixed feature streams; select --stream or --selection-config")
    if "fit_condition" not in df:
        declared = (selection or {}).get("historical_fit_condition")
        if declared is not None:
            if (
                declared not in {"main-only", "with-nuisance"}
                or declared != fit_condition
            ):
                raise ValueError(
                    "historical_fit_condition must match the explicitly selected condition"
                )
            warnings.warn(
                "CSV has no fit_condition column; using the explicitly declared historical_fit_condition",
                stacklevel=2,
            )
            df["fit_condition"] = declared
        elif fit_condition is not None:
            raise ValueError(
                "--fit-condition/selection config requires fit_condition metadata; an audited legacy table needs explicit historical_fit_condition in the config"
            )
        elif selection is not None:
            warnings.warn(
                "CSV has no fit_condition metadata; the null selection preserves it as unknown and only reproduces the supplied table",
                stacklevel=2,
            )
    if "fit_condition" in df:
        if df["fit_condition"].isna().any():
            raise ValueError(
                "fit_condition is missing on some rows; separate incomplete metadata"
            )
        if fit_condition is not None:
            df = df[df["fit_condition"] == fit_condition]
        elif df["fit_condition"].nunique() > 1:
            raise ValueError("mixed fit conditions; select --fit-condition")
    if variant is not None:
        df = df[df["variant"] == variant]
    if selection is not None:
        selected_parts = []
        expected_cells = set()
        for condition in selection["model_conditions"]:
            model, selected_variant = condition["model"], condition["variant"]
            layers = condition["expected_layers"]
            if (
                not isinstance(layers, list)
                or not layers
                or len(layers) != len(set(layers))
                or any(not isinstance(layer, int) or layer < 0 for layer in layers)
            ):
                raise ValueError(
                    f"expected_layers for {model} must be an explicit nonempty list of unique integers"
                )
            selected_parts.append(
                df[
                    (df["model"] == model)
                    & (
                        df["variant"].isna()
                        if selected_variant is None
                        else df["variant"] == selected_variant
                    )
                    & df["layer_idx"].isin(layers)
                ]
            )
            expected_cells.update(
                (model, subject, layer) for subject in subjects for layer in layers
            )
        df = pd.concat(selected_parts, ignore_index=True)
        present = set(
            df[["model", "subject", "layer_idx"]].itertuples(index=False, name=None)
        )
        missing = expected_cells - present
        allowed = {
            (entry["model"], entry["subject"], entry["layer_idx"])
            for entry in selection.get("allow_missing_cells", [])
        }
        if allowed - expected_cells:
            raise ValueError(
                "allow_missing_cells contains cells outside the selected experiment"
            )
        if missing - allowed:
            raise ValueError(
                f"missing expected model/subject/layer cells: {sorted(missing - allowed)[:12]}"
            )
        if missing:
            warnings.warn(
                f"Explicitly acknowledged missing cells: {sorted(missing)}",
                stacklevel=2,
            )
    if df.empty:
        raise ValueError("no main-band data remain after filtering")
    if subjects is not None and set(subjects) - set(df["subject"]):
        raise ValueError(
            f"selected subjects absent after filtering: {sorted(set(subjects) - set(df['subject']))}"
        )
    identity = [
        column
        for column in (
            "model",
            "variant",
            "stream",
            "fit_condition",
            "subject",
            "layer_idx",
            "ROI",
            "side",
            "partition",
            "band",
        )
        if column in df
    ]
    if df.duplicated(identity).any():
        raise ValueError(
            "duplicate encoding cells; select one result source before plotting"
        )
    if not np.isfinite(pd.to_numeric(df["performance"], errors="raise")).all():
        raise ValueError("non-finite encoding performance values")
    return df, selection_models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "figures",
        nargs="*",
        default=["all"],
    )
    p.add_argument(
        "--csv",
        default="out/analysis/master_encoding_data.csv",
    )
    p.add_argument(
        "--outdir",
        default=str(OUTPUT_DIR),
    )
    p.add_argument(
        "--tex",
        action="store_true",
        help="Render labels using an installed LaTeX distribution",
    )
    p.add_argument(
        "--models",
        nargs="+",
        help="Exact model IDs, in display order; write selected-model figures",
    )
    p.add_argument("--variant", help="Select one exact feature variant before plotting")
    p.add_argument(
        "--stream", choices=["main", "attn", "mlp"], help="Select one feature stream"
    )
    p.add_argument("--subjects", nargs="+", help="Explicit participant IDs")
    p.add_argument(
        "--selection-config",
        type=Path,
        help="JSON specifying subjects, stream, band, fit condition and per-model variants/layers",
    )
    p.add_argument(
        "--fit-condition",
        choices=["main-only", "with-nuisance"],
        help="Select one fit condition; mixed-condition tables require this flag",
    )
    args = p.parse_args()
    plt.rcParams["text.usetex"] = args.tex

    figs = set(args.figures)
    df = pd.read_csv(args.csv, low_memory=False)
    try:
        selection = (
            json.loads(args.selection_config.read_text())
            if args.selection_config
            else None
        )
        if selection is not None and selection.get("input_sha256"):
            with open(args.csv, "rb") as source:
                actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
            if actual_hash != selection["input_sha256"]:
                raise ValueError("CSV checksum does not match selection input_sha256")
        if selection is None and args.models:
            df = df[df["model"].isin(args.models)]
        df, config_models = select_encoding_rows(
            df,
            selection=selection,
            stream=args.stream,
            subjects=args.subjects,
            variant=args.variant,
            fit_condition=args.fit_condition,
        )
    except (ValueError, KeyError, TypeError) as error:
        p.error(str(error))
    if config_models is not None:
        if args.models is not None and args.models != config_models:
            p.error(
                "--models conflicts with --selection-config; edit the experiment selection"
            )
        args.models = config_models

    available = set(df["model"].dropna())
    historical_models = set(BASELINES_ORDER + LLMS_ORDER)
    selected_models = args.models
    if selected_models is None and not historical_models.issubset(available):
        historical_order = list(dict.fromkeys(BASELINES_ORDER + LLMS_ORDER))
        selected_models = [m for m in historical_order if m in available]
        selected_models += sorted(available - historical_models)
        print("Partial table: plotting available models:", ", ".join(selected_models))
    if selected_models is not None:
        selected_models = list(dict.fromkeys(selected_models))
        missing = set(selected_models) - available
        if missing:
            p.error(f"selected models absent after filtering: {sorted(missing)}")
        if "all" in figs:
            figs = {"aggregate", "rois", "groups"}
        unsupported = figs - {"aggregate", "rois", "groups"}
        if unsupported:
            p.error(
                "selected/partial tables support aggregate, rois and groups; "
                f"unsupported figures: {sorted(unsupported)}"
            )
        outdir = Path(args.outdir)
        plotters = {
            "aggregate": plot_aggregate,
            "rois": plot_rois,
            "groups": plot_groups,
        }
        for figure in ("aggregate", "rois", "groups"):
            if figure in figs:
                plotters[figure](
                    df,
                    selected_models,
                    outdir / f"encoding_{figure}_selected.pdf",
                    "Encoding accuracy -- selected models",
                )
        return

    if "all" in figs:
        figs = {"aggregate", "rois", "groups", "groups_combined", "permutation"}
    outdir = Path(args.outdir)

    if "aggregate" in figs:
        plot_aggregate(
            df,
            BASELINES_ORDER,
            outdir / "encoding_cross_paradigm.pdf",
            "Encoding accuracy -- cross-paradigm",
        )
        plot_aggregate(
            df,
            LLMS_ORDER,
            outdir / "encoding_llms.pdf",
            "Encoding accuracy -- LLMs",
        )

    if "rois" in figs:
        plot_rois(
            df,
            BASELINES_ORDER,
            outdir / "encoding_rois_cross_paradigm.pdf",
            "Encoding accuracy by ROI -- cross-paradigm",
        )
        plot_rois(
            df,
            LLMS_ORDER,
            outdir / "encoding_rois_llms.pdf",
            "Encoding accuracy by ROI -- LLMs",
        )

    if "groups" in figs:
        plot_groups(
            df,
            BASELINES_ORDER,
            outdir / "encoding_groups_cross_paradigm.pdf",
            "Encoding accuracy by region -- cross-paradigm",
        )
        plot_groups(
            df,
            LLMS_ORDER,
            outdir / "encoding_groups_llms.pdf",
            "Encoding accuracy by region -- LLMs",
            best_only=True,
        )

    if "groups_combined" in figs:
        plot_groups_combined(df, outdir / "encoding_groups_combined.pdf")

    if "permutation" in figs:
        plot_permutation(df, outdir / "encoding_permutation.pdf")


if __name__ == "__main__":
    main()
