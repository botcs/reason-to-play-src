#!/usr/bin/env python3
"""Plot behavioural figures from a local episodes.csv (no cloud credentials).

Inputs contain one row per agent/game/level, with JSON arrays of episode
steps and outcomes. See docs/reproducibility.md for clock conventions.

Figures (each produced in two variants):
  km          -- Kaplan-Meier survival curves
  discovery   -- Discovery step-count histograms
  execution   -- Execution step-count histograms

Variants:
  _baselines  -- best 2 LLMs (V4-Pro + Q-35B) + all baselines
  _llms       -- all 8 LLMs, no baselines

Layout: 2 rows x 6 per-game panels (vgfmri3 top, vgfmri4 bottom)
+ 1 aggregate panel on the right spanning both rows.

Usage:
    python scripts/analysis/plot_behavioural.py km
    python scripts/analysis/plot_behavioural.py discovery
    python scripts/analysis/plot_behavioural.py execution
    python scripts/analysis/plot_behavioural.py all
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

# -- Matplotlib style (matches scripts/plot_step_to_win_histogram.py) ---------
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
plt.rcParams["lines.linewidth"] = 3.5
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

MARKER_SIZE = 36
MARKER_EDGE_WIDTH = 2.0
ANNOTATION_FONTSIZE = 24
LINE_WIDTH_PRIMARY = 5.5
LINE_WIDTH_SECONDARY = 5.0

CACHE_DIR = _PROJECT_ROOT / "out" / "behavioural_cache"
OUTPUT_DIR = _PROJECT_ROOT / "out" / "figures"

VGFMRI3_GAMES = ["bait", "chase", "helper", "lemmings", "zelda", "plaqueattack"]
VGFMRI4_GAMES = ["bait", "chase", "helper", "lemmings", "zelda", "avoidgeorge"]

DISPLAY_NAMES = {
    "avoidgeorge": "Avoid George",
    "bait": "Bait",
    "chase": "Chase",
    "helper": "Helper",
    "lemmings": "Lemmings",
    "plaqueattack": "Plaque Attack",
    "zelda": "Zelda",
}

# -- Agent definitions --------------------------------------------------------

BASELINE_ORDER = ["Human", "DDQN", "EfficientZero", "EMPA"]

ALL_LLM_MODELS = [
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.5-35b-a3b",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-397b-a17b",
]

BEST_LLM_MODELS = [
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.5-35b-a3b",
]

LLM_VARIANT = ("copied-reasoning", "elaborate")

BASELINE_COLORS = {
    "Human": "#2171B5",
    "DDQN": "#CB181D",
    "EfficientZero": "#7A0177",
    "EMPA": "#31A354",
    "EMPA 2.0": "#74C476",
}

LLM_COLORS = {
    "deepseek/deepseek-v3.2": "#E6550D",
    "deepseek/deepseek-v4-flash": "#FD8D3C",
    "deepseek/deepseek-v4-pro": "#FDAE6B",
    "qwen/qwen3.5-9b": "#6A51A3",
    "qwen/qwen3.5-27b": "#807DBA",
    "qwen/qwen3.5-35b-a3b": "#9E9AC8",
    "qwen/qwen3.5-122b-a10b": "#BCBDDC",
    "qwen/qwen3.5-397b-a17b": "#DADAEB",
}

LLM_SHORT = {
    "deepseek/deepseek-v3.2": "DeepSeek-V3.2",
    "deepseek/deepseek-v4-flash": "DeepSeek V4-Flash",
    "deepseek/deepseek-v4-pro": "DeepSeek V4-Pro",
    "qwen/qwen3.5-9b": "Qwen3.5-9B",
    "qwen/qwen3.5-27b": "Qwen3.5-27B",
    "qwen/qwen3.5-35b-a3b": "Qwen3.5-35B",
    "qwen/qwen3.5-122b-a10b": "Qwen3.5-122B",
    "qwen/qwen3.5-397b-a17b": "Qwen3.5-397B",
}

LLM_MARKERS = {
    "deepseek/deepseek-v3.2": ">",
    "deepseek/deepseek-v4-flash": "h",
    "deepseek/deepseek-v4-pro": "p",
    "qwen/qwen3.5-9b": "o",
    "qwen/qwen3.5-27b": "s",
    "qwen/qwen3.5-35b-a3b": "^",
    "qwen/qwen3.5-122b-a10b": "v",
    "qwen/qwen3.5-397b-a17b": "P",
}

BASELINE_MARKERS = {
    "Human": "*",
    "DDQN": "d",
    "EfficientZero": "X",
    "EMPA": "P",
    "EMPA 2.0": "^",
}


# ---------------------------------------------------------------------------
# FinalScoreAnnotator (from https://gist.github.com/botcs/8157060321e39baaf9862f5242f28e44)
# ---------------------------------------------------------------------------


class FinalScoreAnnotator:
    """Annotates curves with their final values, avoiding overlaps via L-BFGS-B."""

    def __init__(self, ax=None):
        self.ax = ax or plt.gca()
        self.lines = []
        self.last_ys = []
        self.last_xs = []
        self.colors = []
        self.markers_list = []

    def plot(self, x, y, label=None, color=None, marker=None, **kwargs):
        line_idx = len(self.lines)
        if color is None:
            color = f"C{line_idx}"
        line_kw = {"alpha": 0.7, **kwargs}
        marker_kw = {"alpha": 0.8, "zorder": 5, **kwargs}
        (line,) = self.ax.plot(x, y, color=color, **line_kw)
        self.ax.plot(
            x[-1],
            y[-1],
            color=color,
            marker=marker,
            markersize=MARKER_SIZE,
            markeredgecolor="black",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=label,
            **marker_kw,
        )
        self.lines.append(line)
        self.last_ys.append(y[-1])
        self.last_xs.append(x[-1])
        self.colors.append(color)
        self.markers_list.append(marker)
        return line

    def _get_fontsize_in_data_coords(self, fontsize):
        temp_text = self.ax.text(0, 0, "99.9", fontsize=fontsize)
        plt.draw()
        bbox = temp_text.get_window_extent()
        bbox_data = bbox.transformed(self.ax.transData.inverted())
        tw = bbox_data.xmax - bbox_data.xmin
        th = bbox_data.ymax - bbox_data.ymin
        temp_text.remove()
        plt.draw()
        return tw, th

    def annotate(self, fontsize=10, fmt="{:>4.0f}", **kwargs):
        if not self.last_ys:
            return

        tw, th = self._get_fontsize_in_data_coords(fontsize * 1.2)
        orig = np.array(self.last_ys)

        # Sort indices by value
        order = np.argsort(orig)

        # Group entries whose values are closer than th (would overlap)
        groups: list[list[int]] = []
        for idx in order:
            if groups and abs(orig[idx] - orig[groups[-1][-1]]) < th:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        pos_x = max(self.last_xs)
        max_pos = orig.max()
        for group in groups:
            if len(group) == 1:
                i = group[0]
                txt = fmt.format(self.last_ys[i] * 100).strip()
                self.ax.annotate(
                    txt,
                    xy=(pos_x, orig[i]),
                    xytext=(20, 0),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=self.colors[i],
                    va="center",
                    ha="left",
                )
            elif len(group) == 2:
                # Two colliding: push apart with top/bottom alignment
                bot, top = group[0], group[-1]
                bot_val = fmt.format(self.last_ys[bot] * 100).strip()
                top_val = fmt.format(self.last_ys[top] * 100).strip()
                mid = (orig[bot] + orig[top]) / 2
                self.ax.annotate(
                    bot_val,
                    xy=(pos_x, mid),
                    xytext=(20, 0),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=self.colors[bot],
                    va="top",
                    ha="left",
                )
                self.ax.annotate(
                    top_val,
                    xy=(pos_x, mid),
                    xytext=(20, 0),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=self.colors[top],
                    va="bottom",
                    ha="left",
                )
                max_pos = max(max_pos, mid + th)
            else:
                # 3+ colliding: only show bottom and top, drop middle
                bot, top = group[0], group[-1]
                bot_val = fmt.format(self.last_ys[bot] * 100).strip()
                top_val = fmt.format(self.last_ys[top] * 100).strip()
                mid = (orig[bot] + orig[top]) / 2
                self.ax.annotate(
                    bot_val,
                    xy=(pos_x, mid),
                    xytext=(20, 0),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=self.colors[bot],
                    va="top",
                    ha="left",
                )
                self.ax.annotate(
                    top_val,
                    xy=(pos_x, mid),
                    xytext=(20, 0),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=self.colors[top],
                    va="bottom",
                    ha="left",
                )
                max_pos = max(max_pos, mid + th)

        ylim = self.ax.get_ylim()
        xlim = self.ax.get_xlim()
        self.ax.set_ylim(ylim[0], max(max_pos + th * 1.5, ylim[1]))
        self.ax.set_xlim(xlim[0], xlim[1] * 3)


class EndpointMarkerAnnotator:
    """Places markers at curve endpoints, right-justified to plot edge.

    Colliding markers (same y within threshold) are staggered
    horizontally leftward, one marker-width per collision.
    """

    def __init__(self, ax=None):
        self.ax = ax or plt.gca()
        self.entries = []  # (final_x, final_y, color, marker, label)

    def plot(self, x, y, label=None, color=None, marker=None, **kwargs):
        line_idx = len(self.entries)
        if color is None:
            color = f"C{line_idx}"
        line_kw = {"alpha": 0.7, **kwargs}
        (line,) = self.ax.plot(x, y, color=color, **line_kw)
        self.entries.append((x[-1], y[-1], color, marker, label))
        return line

    def fill_between(self, x, y_lo, y_hi, **kwargs):
        self.ax.fill_between(x, y_lo, y_hi, **kwargs)

    def place_markers(self, marker_size=None, edge_width=None, x_budget=None):
        if not self.entries:
            return
        if marker_size is None:
            marker_size = MARKER_SIZE
        if edge_width is None:
            edge_width = MARKER_EDGE_WIDTH

        xlim = self.ax.get_xlim()
        if x_budget is None:
            x_budget = xlim[1]

        # Sort entries by final_y
        sorted_entries = sorted(self.entries, key=lambda e: e[1])

        # Convert marker_size (points) to data coordinates via display transform
        fig = self.ax.get_figure()
        fig.canvas.draw()
        # Get two display-space points separated by marker_size pixels
        dpi = fig.dpi
        marker_px = marker_size * dpi / 72.0  # points -> pixels
        inv = self.ax.transData.inverted()
        origin = self.ax.transData.transform((0, 0))
        y_pt = inv.transform((origin[0], origin[1] + marker_px))
        x_pt = inv.transform((origin[0] + marker_px * 1.5, origin[1]))
        y_radius = abs(y_pt[1])
        x_step = abs(x_pt[0])

        # Place markers greedily: each one tries the rightmost column,
        # shifts left one step if it overlaps ANY already-placed marker,
        # repeats until clear.
        placed = []  # list of (x_pos, y_pos)
        for _, fy, color, marker, label in sorted_entries:
            x_pos = x_budget
            while any(
                abs(x_pos - px) < x_step * 0.5 and abs(fy - py) < y_radius
                for px, py in placed
            ):
                x_pos -= x_step
            placed.append((x_pos, fy))
            self.ax.plot(
                x_pos,
                fy,
                marker=marker,
                color=color,
                markersize=marker_size,
                markeredgecolor="black",
                markeredgewidth=edge_width,
                alpha=0.85,
                zorder=10,
                linestyle="none",
                label=label,
                clip_on=False,
            )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        print(f"Episode CSV not found: {cache_path}")
        print("Use scripts/analysis/build_episodes.py or the released episode CSV.")
        raise SystemExit(1)
    df = pd.read_csv(cache_path)
    df["episode_steps"] = df["episode_steps"].apply(json.loads)
    df["episode_outcomes"] = df["episode_outcomes"].apply(json.loads)
    return df


# ---------------------------------------------------------------------------
# Episode analysis: compute survival/discovery/execution from raw episodes
# ---------------------------------------------------------------------------

WIN_OUTCOMES = {"win"}


def compute_discovery(ep_steps: list[int], ep_outcomes: list[str]) -> int | None:
    """Cumulative steps until first win (within-level). None if no win."""
    cumulative = 0
    for steps, outcome in zip(ep_steps, ep_outcomes):
        if outcome in WIN_OUTCOMES:
            return cumulative + steps
        cumulative += steps
    return None


def compute_execution(ep_steps: list[int], ep_outcomes: list[str]) -> list[int]:
    """Steps for each win after the first."""
    found_first = False
    exec_steps = []
    for steps, outcome in zip(ep_steps, ep_outcomes):
        if outcome in WIN_OUTCOMES:
            if found_first:
                exec_steps.append(steps)
            else:
                found_first = True
    return exec_steps


def has_consecutive_wins(ep_outcomes: list[str], n: int = 2) -> bool:
    streak = 0
    for outcome in ep_outcomes:
        if outcome in WIN_OUTCOMES:
            streak += 1
            if streak >= n:
                return True
        else:
            streak = 0
    return False


def derive_survival(
    df: pd.DataFrame,
    censor_budget: int = 1200,
    include_ez_curriculum: bool = False,
    censor_after_failure: bool = True,
    consecutive_wins_required: int = 2,
) -> pd.DataFrame:
    """Derive survival table from raw episode data.

    Returns a DataFrame with columns: all identity columns + time, observed,
    discovery_steps.
    """
    rows = []
    identity_cols = [
        "agent_type",
        "model",
        "rationale_mode",
        "suggestion_level",
        "instance_id",
        "seed",
        "game",
        "cohort",
        "owner",
    ]

    # Group by instance (one agent run on one game)
    group_keys = [
        "agent_type",
        "model",
        "rationale_mode",
        "suggestion_level",
        "instance_id",
        "game",
        "cohort",
    ]
    for group_vals, group_df in df.groupby(group_keys):
        agent_type = group_vals[0]

        identity = {col: group_df.iloc[0][col] for col in identity_cols}

        # Separate EZ warmup curriculum (level > 8, EZ only) from game
        # levels (0-8). Levels 9-12 only exist for EfficientZero (Austin's
        # warmup curriculum); any level>8 rows on humans/LRMs are data
        # artefacts and must not be added to discovery time, since that
        # would silently shift the human reference distribution used in
        # the EMD computation.
        curriculum_overhead = 0
        if include_ez_curriculum and agent_type == "ez":
            for _, row in group_df.iterrows():
                if row["level"] > 8:
                    curriculum_overhead += sum(row["episode_steps"])

        game_levels = group_df[group_df["level"] <= 8].sort_values("level")

        # Realized total steps for this instance (all levels combined).
        # Used as censor time for unreached/unsolved levels -- more honest
        # than the nominal budget for runs that ended early (e.g. V4-Pro
        # hitting the API cost cap).
        realized_steps = (
            sum(sum(row["episode_steps"]) for _, row in game_levels.iterrows())
            + curriculum_overhead
        )
        instance_censor = max(realized_steps, 1)

        # Track mastery for censoring
        first_failure = None

        for level in range(9):
            level_row = game_levels[game_levels["level"] == level]

            if (
                first_failure is not None
                and level > first_failure
                and censor_after_failure
            ):
                rows.append(
                    {
                        **identity,
                        "level": level,
                        "time": instance_censor,
                        "observed": 0,
                        "discovery_steps": None,
                    }
                )
                continue

            if len(level_row) == 0:
                rows.append(
                    {
                        **identity,
                        "level": level,
                        "time": instance_censor,
                        "observed": 0,
                        "discovery_steps": None,
                    }
                )
                if censor_after_failure:
                    first_failure = level
                continue

            ep_steps = level_row.iloc[0]["episode_steps"]
            ep_outcomes = level_row.iloc[0]["episode_outcomes"]

            disc = compute_discovery(ep_steps, ep_outcomes)
            if disc is not None:
                time = disc
                # Add curriculum overhead to the first game level
                if (
                    level == int(game_levels.iloc[0]["level"])
                    and curriculum_overhead > 0
                ):
                    time += curriculum_overhead
                    disc += curriculum_overhead
                rows.append(
                    {
                        **identity,
                        "level": level,
                        "time": time,
                        "observed": 1,
                        "discovery_steps": disc,
                    }
                )
            else:
                if censor_after_failure:
                    total = instance_censor
                else:
                    total = sum(ep_steps)
                    if (
                        level == int(game_levels.iloc[0]["level"])
                        and curriculum_overhead > 0
                    ):
                        total += curriculum_overhead
                rows.append(
                    {
                        **identity,
                        "level": level,
                        "time": total,
                        "observed": 0,
                        "discovery_steps": None,
                    }
                )

            # Check mastery for censoring
            if censor_after_failure and not has_consecutive_wins(
                ep_outcomes, consecutive_wins_required
            ):
                first_failure = level

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KM estimator
# ---------------------------------------------------------------------------


def fix_censoring(times: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Re-cap censored times to >= max event time + 1."""
    times = times.copy()
    event_mask = observed == 1
    if not event_mask.any():
        return times
    max_event = times[event_mask].max()
    censor_mask = observed == 0
    times[censor_mask & (times <= max_event)] = max_event + 1
    return times


def km_cdf(times: np.ndarray, observed: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier CDF (= 1 - survival) on a time grid."""
    if len(times) == 0:
        return np.zeros(len(grid))

    times = fix_censoring(times, observed)
    order = np.argsort(times)
    t_sorted = times[order]
    o_sorted = observed[order]

    event_times = np.unique(t_sorted[o_sorted == 1])
    if len(event_times) == 0:
        return np.zeros(len(grid))

    survival = 1.0
    step_t = [0.0]
    step_s = [1.0]
    for t_i in event_times:
        at_risk = np.sum(t_sorted >= t_i)
        events = np.sum((t_sorted == t_i) & (o_sorted == 1))
        if at_risk > 0:
            survival *= 1 - events / at_risk
        step_t.append(t_i)
        step_s.append(survival)

    step_t = np.array(step_t)
    step_s = np.array(step_s)

    cdf = np.zeros(len(grid))
    for i, g in enumerate(grid):
        idx = np.searchsorted(step_t, g, side="right") - 1
        idx = max(0, min(idx, len(step_s) - 1))
        cdf[i] = 1.0 - step_s[idx]

    return cdf


def km_cdf_per_game_pool(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Compute KM CDF per (game, cohort), then average across games."""
    cdfs = []
    for (game, cohort), sub in df.groupby(["game", "cohort"]):
        if len(sub) == 0:
            continue
        cdf = km_cdf(
            sub["time"].values.astype(float), sub["observed"].values.astype(float), grid
        )
        cdfs.append(cdf)
    if not cdfs:
        return np.zeros(len(grid))
    return np.mean(cdfs, axis=0)


# ---------------------------------------------------------------------------
# Exclusion + selection helpers
# ---------------------------------------------------------------------------


def parse_exclusions(
    exclude_args: list[str],
) -> list[tuple[str | None, str | None, str | None]]:
    """Parse --exclude arguments into (agent_type, cohort, game) tuples.

    Formats:
        vgfmri3              -- exclude entire cohort (all agents)
        ez:vgfmri3           -- exclude agent_type from cohort
        ez:vgfmri3:bait      -- exclude agent_type from cohort+game
    """
    exclusions = []
    for arg in exclude_args:
        parts = arg.split(":")
        if len(parts) == 1:
            # Could be a cohort name or an agent_type
            if parts[0].startswith("vgfmri"):
                exclusions.append((None, parts[0], None))
            else:
                exclusions.append((parts[0], None, None))
        elif len(parts) == 2:
            exclusions.append((parts[0], parts[1], None))
        else:
            exclusions.append((parts[0], parts[1], parts[2]))
    return exclusions


def apply_exclusions(
    df: pd.DataFrame, exclusions: list[tuple[str | None, str | None, str | None]]
) -> pd.DataFrame:
    if not exclusions:
        return df
    mask = pd.Series(True, index=df.index)
    for agent_type, cohort, game in exclusions:
        exc = pd.Series(True, index=df.index)
        if agent_type is not None:
            exc = exc & (df["agent_type"] == agent_type)
        if cohort is not None:
            exc = exc & (df["cohort"] == cohort)
        if game is not None:
            exc = exc & (df["game"] == game)
        mask = mask & ~exc
    return df[mask]


def _sel(
    df: pd.DataFrame, game: str | None = None, cohort: str | None = None
) -> pd.DataFrame:
    out = df
    if game is not None:
        out = out[out["game"] == game]
    if cohort is not None:
        out = out[out["cohort"] == cohort]
    return out


def _select_agent(
    df: pd.DataFrame, model: str, agent_type: str | None = None
) -> pd.DataFrame:
    s = df[df["model"] == model]
    if agent_type == "llm":
        rm, sl = LLM_VARIANT
        s = s[(s["rationale_mode"] == rm) & (s["suggestion_level"] == sl)]
    return s


def _game_cohort_list(surv: pd.DataFrame) -> list[tuple[str, str]]:
    """Build game-cohort list from the cohorts actually present in data."""
    cohorts = sorted(surv["cohort"].unique())
    pairs = []
    if "vgfmri3" in cohorts:
        pairs.extend([(g, "vgfmri3") for g in VGFMRI3_GAMES])
    if "vgfmri4" in cohorts:
        pairs.extend([(g, "vgfmri4") for g in VGFMRI4_GAMES])
    return pairs


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------


def make_figure(n_rows: int, n_cols: int):
    fig = plt.figure(figsize=(36, 2.5 * (n_rows + 1)))
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[0.8] * n_cols + [3.5],
        hspace=0.45,
        wspace=0.3,
    )
    axes = []
    for row in range(n_rows):
        for col in range(n_cols):
            axes.append(fig.add_subplot(gs[row, col]))
    ax_agg = fig.add_subplot(gs[:, n_cols])
    return fig, axes, ax_agg


# ---------------------------------------------------------------------------
# KM plotting
# ---------------------------------------------------------------------------


def _compute_cdf(
    s: pd.DataFrame, t_grid: np.ndarray, per_game_pool: bool, game: str | None
) -> np.ndarray:
    """Compute KM CDF, optionally averaging per-game CDFs."""
    if per_game_pool and game is None:
        return km_cdf_per_game_pool(s, t_grid)
    return km_cdf(
        s["time"].values.astype(float), s["observed"].values.astype(float), t_grid
    )


def _plot_km_panel(
    ax: plt.Axes,
    surv: pd.DataFrame,
    baselines: list[str],
    llms: list[str],
    game: str | None,
    cohort: str | None,
    annotate: bool = False,
    per_game_pool: bool = False,
) -> None:
    sub = _sel(surv, game, cohort)
    t_grid = np.logspace(0, 5, 400)
    ann = FinalScoreAnnotator(ax) if annotate else None

    # Plot non-human agents first, then human on top
    human_deferred = None
    for model_name in baselines:
        if model_name == "Human":
            human_deferred = ("baseline", model_name)
            continue
        s = _select_agent(sub, model_name)
        if len(s) == 0:
            continue
        cdf = _compute_cdf(s, t_grid, per_game_pool, game)
        color = BASELINE_COLORS.get(model_name, "#888")
        marker = BASELINE_MARKERS.get(model_name, "o")
        label = model_name
        if ann:
            ann.plot(
                t_grid,
                cdf,
                label=label,
                color=color,
                marker=marker,
                linewidth=LINE_WIDTH_PRIMARY,
            )
        else:
            ax.plot(
                t_grid,
                cdf,
                label=label,
                color=color,
                linewidth=LINE_WIDTH_SECONDARY,
                alpha=0.7,
            )

    for model_name in llms:
        if model_name == "Human":
            human_deferred = ("llm", model_name)
            continue
        s = _select_agent(sub, model_name, "llm")
        if len(s) == 0:
            continue
        cdf = _compute_cdf(s, t_grid, per_game_pool, game)
        color = LLM_COLORS.get(model_name, "#888")
        marker = LLM_MARKERS.get(model_name, "o")
        label = LLM_SHORT.get(model_name, model_name)
        if ann:
            ann.plot(
                t_grid,
                cdf,
                label=label,
                color=color,
                marker=marker,
                linewidth=LINE_WIDTH_PRIMARY,
            )
        else:
            ax.plot(
                t_grid,
                cdf,
                label=label,
                color=color,
                linewidth=LINE_WIDTH_SECONDARY,
                alpha=0.7,
            )

    if human_deferred is not None:
        s = _select_agent(sub, "Human")
        if len(s) > 0:
            cdf = _compute_cdf(s, t_grid, per_game_pool, game)
            color = BASELINE_COLORS["Human"]
            marker = BASELINE_MARKERS["Human"]
            if ann:
                ann.plot(
                    t_grid,
                    cdf,
                    label="Human",
                    color=color,
                    marker=marker,
                    linewidth=LINE_WIDTH_PRIMARY,
                    zorder=10,
                )
            else:
                ax.plot(
                    t_grid,
                    cdf,
                    label="Human",
                    color=color,
                    linewidth=LINE_WIDTH_PRIMARY,
                    alpha=0.9,
                    zorder=10,
                )

    ax.set_xscale("log")
    if not annotate:
        ax.set_ylim(-0.05, 1.05)

    if ann:
        ann.annotate(fontsize=ANNOTATION_FONTSIZE, fmt="{:>4.0f}")


def plot_km(surv: pd.DataFrame, output_dir: Path, per_game_pool: bool = False) -> None:
    pool_tag = "pgp" if per_game_pool else "raw"
    gc_list = _game_cohort_list(surv)
    cohorts = sorted(set(c for _, c in gc_list))
    n_rows = len(cohorts)
    n_cols = max(sum(1 for _, c in gc_list if c == ch) for ch in cohorts)

    for variant_name, baselines, llms in [
        ("cross_paradigm", BASELINE_ORDER, BEST_LLM_MODELS),
        ("llms", ["Human"], ALL_LLM_MODELS),
    ]:
        fig, axes, ax_agg = make_figure(n_rows, n_cols)

        for idx, (game, cohort) in enumerate(gc_list):
            ax = axes[idx]
            _plot_km_panel(ax, surv, baselines, llms, game, cohort)
            title = DISPLAY_NAMES.get(game, game)
            cl = cohort.replace("vgfmri", "v")
            ax.set_title(r"" + title + r" (" + cl + r")")
            if idx % n_cols == 0:
                ax.set_ylabel(r"$P(\mathrm{first\ win} \leq t)$")
            if idx >= n_cols * (n_rows - 1):
                ax.set_xlabel(r"Steps")

        agg_title = (
            r"All Games (per-game avg)" if per_game_pool else r"All Games (pooled)"
        )
        _plot_km_panel(
            ax_agg,
            surv,
            baselines,
            llms,
            None,
            None,
            annotate=True,
            per_game_pool=per_game_pool,
        )
        ax_agg.set_title(agg_title)
        ax_agg.set_xlabel(r"Steps")
        ax_agg.set_ylabel(r"$P(\mathrm{first\ win} \leq t)$")

        handles, labels = ax_agg.get_legend_handles_labels()
        ncol = 5 if variant_name == "llms" else len(handles)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
        )

        fig.subplots_adjust(bottom=0.25)
        out = output_dir / f"behavioural_km_{variant_name}_{pool_tag}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Histogram plotting
# ---------------------------------------------------------------------------


def _kde(vals: np.ndarray, grid: np.ndarray, bw: float = 0.3) -> np.ndarray:
    """Gaussian KDE in log10-space, proper density on log-scale."""
    if len(vals) < 2:
        return np.zeros(len(grid))
    log_vals = np.log10(np.maximum(vals, 1))
    if np.std(log_vals) < 1e-10:
        return np.zeros(len(grid))
    log_grid = np.log10(np.maximum(grid, 1))
    kde = np.zeros(len(grid))
    for v in log_vals:
        kde += np.exp(-0.5 * ((log_grid - v) / bw) ** 2)
    kde /= len(log_vals) * bw * np.sqrt(2 * np.pi)
    return kde


def _plot_hist_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    baselines: list[str],
    llms: list[str],
    game: str | None,
    cohort: str | None,
    grid: np.ndarray,
    show_markers: bool = False,
) -> None:
    sub = _sel(df, game, cohort)

    for model_name in baselines:
        s = _select_agent(sub, model_name)
        vals = s[value_col].dropna().values.astype(float)
        if len(vals) == 0:
            continue
        color = BASELINE_COLORS.get(model_name, "#888")
        marker = BASELINE_MARKERS.get(model_name, "o")
        kde = _kde(vals, grid)
        if model_name == "Human":
            ax.fill_between(grid, kde, alpha=0.25, color=color)
            ax.plot(
                grid,
                kde,
                color=color,
                linewidth=LINE_WIDTH_PRIMARY,
                label=None if show_markers else model_name,
            )
        else:
            ax.plot(
                grid,
                kde,
                color=color,
                linewidth=LINE_WIDTH_SECONDARY,
                alpha=0.7,
                label=None if show_markers else model_name,
            )
        if show_markers:
            peak_idx = np.argmax(kde) if kde.max() > 0 else 0
            ax.plot(
                grid[peak_idx],
                kde[peak_idx],
                marker=marker,
                color=color,
                markersize=MARKER_SIZE * 0.7,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=10,
                linestyle="none",
                alpha=0.7,
            )
            ax.plot(
                [],
                [],
                marker=marker,
                color=color,
                markersize=MARKER_SIZE * 0.7,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linewidth=LINE_WIDTH_SECONDARY,
                label=model_name,
            )

    for model_name in llms:
        s = _select_agent(sub, model_name, "llm")
        vals = s[value_col].dropna().values.astype(float)
        if len(vals) == 0:
            continue
        label = LLM_SHORT.get(model_name, model_name)
        color = LLM_COLORS.get(model_name, "#888")
        marker = LLM_MARKERS.get(model_name, "o")
        kde = _kde(vals, grid)
        ax.plot(
            grid,
            kde,
            color=color,
            linewidth=LINE_WIDTH_SECONDARY,
            alpha=0.7,
            label=None if show_markers else label,
        )
        if show_markers:
            peak_idx = np.argmax(kde) if kde.max() > 0 else 0
            ax.plot(
                grid[peak_idx],
                kde[peak_idx],
                marker=marker,
                color=color,
                markersize=MARKER_SIZE * 0.7,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=10,
                linestyle="none",
                alpha=0.7,
            )
            ax.plot(
                [],
                [],
                marker=marker,
                color=color,
                markersize=MARKER_SIZE * 0.7,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linewidth=LINE_WIDTH_SECONDARY,
                label=label,
            )

    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)


def _plot_histogram_figure(
    df: pd.DataFrame,
    value_col: str,
    x_label: str,
    fig_name: str,
    output_dir: Path,
    filter_observed: bool = False,
) -> None:
    grid = np.logspace(0, 5, 500)
    plot_df = df[df["observed"] == 1] if filter_observed else df
    gc_list = _game_cohort_list(plot_df)
    cohorts = sorted(set(c for _, c in gc_list))
    n_rows = len(cohorts)
    n_cols = max(sum(1 for _, c in gc_list if c == ch) for ch in cohorts)

    for variant_name, baselines, llms in [
        ("cross_paradigm", BASELINE_ORDER, BEST_LLM_MODELS),
        ("llms", ["Human"], ALL_LLM_MODELS),
    ]:
        fig, axes, ax_agg = make_figure(n_rows, n_cols)

        for idx, (game, cohort) in enumerate(gc_list):
            ax = axes[idx]
            _plot_hist_panel(
                ax, plot_df, value_col, baselines, llms, game, cohort, grid
            )
            title = DISPLAY_NAMES.get(game, game)
            cl = cohort.replace("vgfmri", "v")
            ax.set_title(r"" + title + r" (" + cl + r")")
            if idx % n_cols == 0:
                ax.set_ylabel(r"Density")
            if idx >= n_cols * (n_rows - 1):
                mid_col = n_cols // 2
                if idx % n_cols == mid_col:
                    ax.set_xlabel(x_label)
                else:
                    ax.set_xlabel("")

        _plot_hist_panel(
            ax_agg,
            plot_df,
            value_col,
            baselines,
            llms,
            None,
            None,
            grid,
            show_markers=True,
        )
        ax_agg.set_title(r"All Games (pooled)")
        ax_agg.set_xlabel(x_label)
        ax_agg.set_ylabel(r"Density")

        # Inset EMD barplot (upper right of aggregate panel)
        human_sub = _select_agent(plot_df, "Human")
        human_vals = np.log1p(human_sub[value_col].dropna().values.astype(float))
        if len(human_vals) >= 2:
            from scipy.stats import wasserstein_distance

            emd_entries = []
            for model_name in baselines:
                if model_name == "Human":
                    continue
                s = _select_agent(plot_df, model_name)
                vals = np.log1p(s[value_col].dropna().values.astype(float))
                if len(vals) >= 2:
                    emd = wasserstein_distance(human_vals, vals)
                    emd_entries.append(
                        (model_name, emd, BASELINE_COLORS.get(model_name, "#888"))
                    )
            for model_name in llms:
                s = _select_agent(plot_df, model_name, "llm")
                vals = np.log1p(s[value_col].dropna().values.astype(float))
                if len(vals) >= 2:
                    emd = wasserstein_distance(human_vals, vals)
                    label = LLM_SHORT.get(model_name, model_name)
                    emd_entries.append((label, emd, LLM_COLORS.get(model_name, "#888")))
            if emd_entries:
                emd_entries.sort(key=lambda x: x[1])
                print(f"  EMD vs Human ({value_col}):")
                for name, emd_val, _ in emd_entries:
                    print(f"    {name:25s} {emd_val:.4f}")
                ax_inset = ax_agg.inset_axes([0.55, 0.5, 0.42, 0.42])
                # Build marker lookup from display name
                marker_lookup = {}
                for k, v in BASELINE_MARKERS.items():
                    marker_lookup[k] = v
                for k, v in LLM_MARKERS.items():
                    marker_lookup[LLM_SHORT.get(k, k)] = v

                emds = [e[1] for e in emd_entries]
                colors = [e[2] for e in emd_entries]
                ax_inset.barh(
                    range(len(emd_entries)),
                    emds,
                    color=colors,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.8,
                )
                ax_inset.set_xscale("linear")
                ax_inset.set_yticks(range(len(emd_entries)))
                # Use markers as ytick labels
                for i, (name, emd_val, color) in enumerate(emd_entries):
                    marker = marker_lookup.get(name, "o")
                    ax_inset.plot(
                        -0.01,
                        i,
                        marker=marker,
                        color=color,
                        markersize=MARKER_SIZE * 0.6,
                        markeredgecolor="black",
                        markeredgewidth=0.8,
                        clip_on=False,
                        transform=ax_inset.get_yaxis_transform(),
                    )
                ax_inset.set_yticklabels([""] * len(emd_entries))
                ax_inset.set_xlabel(r"EMD (log-space)", fontsize=12)
                ax_inset.set_title(r"EMD vs Human", fontsize=15)
                ax_inset.tick_params(labelsize=10)

        handles, labels = ax_agg.get_legend_handles_labels()
        ncol = 5 if variant_name == "llms" else len(handles)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
        )

        fig.subplots_adjust(bottom=0.25)
        out = output_dir / f"behavioural_{fig_name}_{variant_name}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def plot_discovery(surv: pd.DataFrame, output_dir: Path) -> None:
    _plot_histogram_figure(
        surv,
        value_col="discovery_steps",
        x_label=r"Steps to First Win",
        fig_name="discovery",
        output_dir=output_dir,
        filter_observed=True,
    )


def _emd_inset(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    value_col: str,
    baselines: list[str],
    llms: list[str],
) -> None:
    """Add EMD inset bar plot to an axes."""
    from scipy.stats import wasserstein_distance

    human_sub = _select_agent(plot_df, "Human")
    human_vals = np.log1p(human_sub[value_col].dropna().values.astype(float))
    if len(human_vals) < 2:
        return
    emd_entries = []
    marker_lookup = {}
    for k, v in BASELINE_MARKERS.items():
        marker_lookup[k] = v
    for k, v in LLM_MARKERS.items():
        marker_lookup[LLM_SHORT.get(k, k)] = v

    for model_name in baselines:
        if model_name == "Human":
            continue
        s = _select_agent(plot_df, model_name)
        vals = np.log1p(s[value_col].dropna().values.astype(float))
        if len(vals) >= 2:
            emd = wasserstein_distance(human_vals, vals)
            emd_entries.append(
                (model_name, emd, BASELINE_COLORS.get(model_name, "#888"))
            )
    for model_name in llms:
        s = _select_agent(plot_df, model_name, "llm")
        vals = np.log1p(s[value_col].dropna().values.astype(float))
        if len(vals) >= 2:
            emd = wasserstein_distance(human_vals, vals)
            label = LLM_SHORT.get(model_name, model_name)
            emd_entries.append((label, emd, LLM_COLORS.get(model_name, "#888")))
    if not emd_entries:
        return
    emd_entries.sort(key=lambda x: x[1])
    print(f"  EMD vs Human ({value_col}):")
    for name, emd_val, _ in emd_entries:
        print(f"    {name:25s} {emd_val:.4f}")

    ax_inset = ax.inset_axes([0.55, 0.5, 0.42, 0.42])
    emds = [e[1] for e in emd_entries]
    colors = [e[2] for e in emd_entries]
    ax_inset.barh(
        range(len(emd_entries)),
        emds,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    ax_inset.set_xscale("linear")
    ax_inset.set_yticks(range(len(emd_entries)))
    for i, (name, emd_val, color) in enumerate(emd_entries):
        marker = marker_lookup.get(name, "o")
        ax_inset.plot(
            -0.01,
            i,
            marker=marker,
            color=color,
            markersize=MARKER_SIZE * 0.6,
            markeredgecolor="black",
            markeredgewidth=0.8,
            clip_on=False,
            transform=ax_inset.get_yaxis_transform(),
        )
    ax_inset.set_yticklabels([""] * len(emd_entries))
    ax_inset.set_xlabel(r"EMD (log-space)", fontsize=24)
    ax_inset.set_title(r"EMD vs Human", fontsize=28)
    ax_inset.tick_params(labelsize=18)


def plot_discovery_aggregate(surv: pd.DataFrame, output_dir: Path) -> None:
    """Combined baselines+LLMs aggregate-only discovery KDE (main paper figure)."""
    from matplotlib.gridspec import GridSpec

    grid = np.logspace(0, 5, 500)
    plot_df = surv[surv["observed"] == 1]

    fig = plt.figure(figsize=(28, 10))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1, 0.15],
        hspace=0.25,
        wspace=0.15,
        left=0.06,
        right=0.98,
        top=0.95,
        bottom=0.05,
    )

    ax_base = fig.add_subplot(gs[0, 0])
    ax_llm = fig.add_subplot(gs[0, 1], sharey=ax_base)

    # Left: baselines
    _plot_hist_panel(
        ax_base,
        plot_df,
        "discovery_steps",
        BASELINE_ORDER,
        BEST_LLM_MODELS,
        None,
        None,
        grid,
        show_markers=True,
    )
    ax_base.set_title(r"Cross-Paradigm", fontsize=40)
    ax_base.set_xlabel(r"Steps to First Win")
    ax_base.set_ylabel(r"Density")
    ax_base.tick_params(axis="both", labelsize=40)
    _emd_inset(ax_base, plot_df, "discovery_steps", BASELINE_ORDER, BEST_LLM_MODELS)

    # Right: LLMs
    _plot_hist_panel(
        ax_llm,
        plot_df,
        "discovery_steps",
        ["Human"],
        ALL_LLM_MODELS,
        None,
        None,
        grid,
        show_markers=True,
    )
    ax_llm.set_title(r"LLMs", fontsize=40)
    ax_llm.set_xlabel(r"Steps to First Win")
    ax_llm.tick_params(axis="both", labelsize=40)
    _emd_inset(ax_llm, plot_df, "discovery_steps", ["Human"], ALL_LLM_MODELS)

    # Expand ylim by 5% on both ends
    ymin, ymax = ax_base.get_ylim()
    margin = (ymax - ymin) * 0.05
    ax_base.set_ylim(ymin - margin, ymax + margin)

    # Joint legend below both panels
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    handles_b, labels_b = ax_base.get_legend_handles_labels()
    handles_l, labels_l = ax_llm.get_legend_handles_labels()
    seen = set()
    handles_all, labels_all = [], []
    for handle, label in list(zip(handles_b, labels_b)) + list(
        zip(handles_l, labels_l)
    ):
        if label not in seen:
            seen.add(label)
            handles_all.append(handle)
            labels_all.append(label)
    ax_leg.legend(
        handles_all,
        labels_all,
        loc="center",
        fontsize=24,
        framealpha=0.9,
        ncol=min(len(handles_all), 6),
    )

    output = output_dir / "behavioural_discovery_combined.pdf"
    fig.savefig(output)
    print(f"  Saved {output}")
    plt.close(fig)


def _plot_curriculum_panel_epm(
    ax: plt.Axes,
    df: pd.DataFrame,
    baselines: list[str],
    llms: list[str],
    consecutive_wins: int = 2,
) -> EndpointMarkerAnnotator:
    """Curriculum panel using EndpointMarkerAnnotator instead of FinalScoreAnnotator."""
    group_keys = [
        "agent_type",
        "model",
        "rationale_mode",
        "suggestion_level",
        "instance_id",
        "game",
        "cohort",
    ]
    grid = np.linspace(0, CURRICULUM_BUDGET, 500)
    epm = EndpointMarkerAnnotator(ax)

    def _plot_agent(agent_sub, model_name, color, marker, label, agent_type=None):
        instances = agent_sub.groupby(group_keys).ngroups
        if instances == 0:
            return
        trajectories = []
        for keys, _ in agent_sub.groupby(group_keys):
            xs, ys = _compute_curriculum_trajectory(df, keys, consecutive_wins)
            if len(xs) > 1:
                trajectories.append((xs, ys))
        if not trajectories:
            return
        interped = np.zeros((len(trajectories), len(grid)))
        for i, (xs, ys) in enumerate(trajectories):
            for j, g in enumerate(grid):
                idx = np.searchsorted(xs, g, side="right") - 1
                idx = max(0, min(idx, len(ys) - 1))
                interped[i, j] = ys[idx]
        mean_y = np.mean(interped, axis=0)
        sem_y = np.std(interped, axis=0, ddof=1) / np.sqrt(len(interped))
        epm.plot(
            grid,
            mean_y,
            label=label,
            color=color,
            marker=marker,
            linewidth=LINE_WIDTH_PRIMARY,
        )
        epm.fill_between(
            grid, mean_y - sem_y, mean_y + sem_y, color=color, alpha=0.15, zorder=1
        )

    for model_name in baselines:
        if model_name == "Human":
            continue
        s = _select_agent(df, model_name)
        if len(s) == 0:
            continue
        _plot_agent(
            s,
            model_name,
            BASELINE_COLORS.get(model_name, "#888"),
            BASELINE_MARKERS.get(model_name, "o"),
            model_name,
        )

    for model_name in llms:
        if model_name == "Human":
            continue
        s = _select_agent(df, model_name, "llm")
        if len(s) == 0:
            continue
        _plot_agent(
            s,
            model_name,
            LLM_COLORS.get(model_name, "#888"),
            LLM_MARKERS.get(model_name, "o"),
            LLM_SHORT.get(model_name, model_name),
            "llm",
        )

    if "Human" in baselines or "Human" in llms:
        s = _select_agent(df, "Human")
        if len(s) > 0:
            _plot_agent(
                s, "Human", BASELINE_COLORS["Human"], BASELINE_MARKERS["Human"], "Human"
            )

    ax.axhline(8, color="black", linestyle="--", linewidth=4.0, alpha=0.5, zorder=0)
    ax.text(
        CURRICULUM_BUDGET * 0.02,
        8.15,
        r"\textit{max level}",
        fontsize=40,
        color="black",
        alpha=0.6,
        va="bottom",
    )
    ax.set_ylim(-0.3, 9.3)
    ax.set_xlim(0, CURRICULUM_BUDGET)
    return epm


def plot_discovery_curriculum_combined(
    surv: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_dir: Path,
    consecutive_wins: int = 2,
) -> None:
    """2x2 combined figure: discovery (top) + curriculum (bottom) x cross-paradigm + LRMs."""

    F = 2  # font scale factor
    grid = np.logspace(0, 5, 500)
    plot_df = surv[surv["observed"] == 1]

    fig = plt.figure(figsize=(34, 22))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1, 0.02, 1],
        hspace=0.45,
        wspace=0.06,
        left=0.10,
        right=0.98,
        top=0.95,
        bottom=0.05,
    )

    ax_disc_base = fig.add_subplot(gs[0, 0])
    ax_disc_lrm = fig.add_subplot(gs[0, 2], sharey=ax_disc_base)
    ax_curr_base = fig.add_subplot(gs[1, 0])
    ax_curr_lrm = fig.add_subplot(gs[1, 2], sharey=ax_curr_base)

    # -- Row 0: Discovery KDE --
    _plot_hist_panel(
        ax_disc_base,
        plot_df,
        "discovery_steps",
        BASELINE_ORDER,
        BEST_LLM_MODELS,
        None,
        None,
        grid,
        show_markers=True,
    )
    ax_disc_base.set_title(r"Cross-Paradigm", fontsize=32 * F)
    ax_disc_base.set_xlabel(r"Steps to First Win", fontsize=26 * F)
    ax_disc_base.set_ylabel(r"Density", fontsize=26 * F)
    ax_disc_base.tick_params(labelsize=20 * F)
    _emd_inset(
        ax_disc_base, plot_df, "discovery_steps", BASELINE_ORDER, BEST_LLM_MODELS
    )

    _plot_hist_panel(
        ax_disc_lrm,
        plot_df,
        "discovery_steps",
        ["Human"],
        ALL_LLM_MODELS,
        None,
        None,
        grid,
        show_markers=True,
    )
    ax_disc_lrm.set_title(r"LRMs", fontsize=32 * F)
    ax_disc_lrm.set_xlabel(r"Steps to First Win", fontsize=26 * F)
    ax_disc_lrm.tick_params(labelsize=20 * F)
    _emd_inset(ax_disc_lrm, plot_df, "discovery_steps", ["Human"], ALL_LLM_MODELS)

    # Sync discovery y-limits
    ymin, ymax = ax_disc_base.get_ylim()
    margin = (ymax - ymin) * 0.05
    ax_disc_base.set_ylim(ymin - margin, ymax + margin)

    # -- Row 1: Curriculum staircase --
    epm_base = _plot_curriculum_panel_epm(
        ax_curr_base, raw_df, BASELINE_ORDER, BEST_LLM_MODELS, consecutive_wins
    )
    ax_curr_base.set_xlabel(r"Steps", fontsize=26 * F)
    ax_curr_base.set_ylabel(r"Avg Level Reached", fontsize=26 * F)
    ax_curr_base.tick_params(labelsize=20 * F)

    epm_lrm = _plot_curriculum_panel_epm(
        ax_curr_lrm, raw_df, ["Human"], ALL_LLM_MODELS, consecutive_wins
    )
    ax_curr_lrm.set_xlabel(r"Steps", fontsize=26 * F)
    ax_curr_lrm.tick_params(labelsize=20 * F)

    # Place endpoint markers (right-justified, stagger collisions)
    epm_base.place_markers(x_budget=CURRICULUM_BUDGET * 0.98)
    epm_lrm.place_markers(x_budget=CURRICULUM_BUDGET * 0.98)

    # -- Row labels (left of first column y-axis) --
    ax_disc_base.text(
        -0.14,
        0.5,
        r"Learning Efficiency",
        transform=ax_disc_base.transAxes,
        fontsize=28 * F,
        rotation=90,
        va="center",
        ha="center",
    )
    ax_curr_base.text(
        -0.14,
        0.5,
        r"Capability",
        transform=ax_curr_base.transAxes,
        fontsize=28 * F,
        rotation=90,
        va="center",
        ha="center",
    )

    # -- Two legend boxes side by side: baselines (2x2) | LLMs (2x4) --
    handles_b, labels_b = ax_disc_base.get_legend_handles_labels()
    handles_l, labels_l = ax_disc_lrm.get_legend_handles_labels()

    llm_display_names = set(LLM_SHORT.values())
    baseline_handles, baseline_labels = [], []
    for h, lb in zip(handles_b, labels_b):
        if lb not in llm_display_names:
            baseline_handles.append(h)
            baseline_labels.append(lb)

    llm_handles, llm_labels = [], []
    for h, lb in zip(handles_l, labels_l):
        if lb != "Human":
            llm_handles.append(h)
            llm_labels.append(lb)

    leg_kw = dict(
        fontsize=18 * F,
        framealpha=0.95,
        columnspacing=1.5,
        bbox_transform=fig.transFigure,
        handlelength=2.5,
    )

    fig.legend(
        baseline_handles,
        baseline_labels,
        loc="center",
        ncol=2,
        bbox_to_anchor=(0.18, 0.475),
        **leg_kw,
    )
    fig.legend(
        llm_handles,
        llm_labels,
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.65, 0.475),
        **leg_kw,
    )

    output = output_dir / "behavioural_discovery_curriculum_combined.pdf"
    fig.savefig(output, bbox_inches="tight", dpi=150)
    print(f"  Saved {output}")
    plt.close(fig)


def plot_execution(exec_df: pd.DataFrame, output_dir: Path) -> None:
    _plot_histogram_figure(
        exec_df,
        value_col="steps",
        x_label=r"Steps for Subsequent Wins",
        fig_name="execution",
        output_dir=output_dir,
        filter_observed=False,
    )


# ---------------------------------------------------------------------------
# Curriculum progression plotting
# ---------------------------------------------------------------------------


def _compute_curriculum_trajectory(
    df: pd.DataFrame,
    instance_key: tuple,
    consecutive_wins: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (cumulative_steps, level) staircase for one instance.

    Returns arrays suitable for step-plot: the agent stays on a level
    until it achieves consecutive_wins, then advances.
    """
    group_keys = [
        "agent_type",
        "model",
        "rationale_mode",
        "suggestion_level",
        "instance_id",
        "game",
        "cohort",
    ]
    sub = df
    for col, val in zip(group_keys, instance_key):
        sub = sub[sub[col] == val]

    game_levels = sub[sub["level"] <= 8].sort_values("level")

    cum_steps = 0
    xs = [0]
    ys = [0]

    for level in range(9):
        level_row = game_levels[game_levels["level"] == level]
        if len(level_row) == 0:
            break

        ep_steps = level_row.iloc[0]["episode_steps"]
        ep_outcomes = level_row.iloc[0]["episode_outcomes"]

        mastered = False
        streak = 0
        for steps, outcome in zip(ep_steps, ep_outcomes):
            cum_steps += steps
            if outcome in WIN_OUTCOMES:
                streak += 1
                if streak >= consecutive_wins:
                    mastered = True
                    xs.append(cum_steps)
                    ys.append(level + 1)
                    break
            else:
                streak = 0

        if not mastered:
            xs.append(cum_steps)
            ys.append(level)
            break

    return np.array(xs), np.array(ys)


CURRICULUM_BUDGET = 1600


def _plot_curriculum_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    baselines: list[str],
    llms: list[str],
    game: str | None,
    cohort: str | None,
    consecutive_wins: int = 2,
    annotate: bool = False,
) -> None:
    sub = _sel(df, game, cohort)
    group_keys = [
        "agent_type",
        "model",
        "rationale_mode",
        "suggestion_level",
        "instance_id",
        "game",
        "cohort",
    ]
    ann = FinalScoreAnnotator(ax) if annotate else None
    grid = np.linspace(0, CURRICULUM_BUDGET, 500)

    def _plot_agent_mean(
        agent_sub: pd.DataFrame,
        model_name: str,
        color: str,
        marker: str,
        label: str,
        agent_type: str | None = None,
    ):
        instances = agent_sub.groupby(group_keys).ngroups
        if instances == 0:
            return
        trajectories = []
        for keys, _ in agent_sub.groupby(group_keys):
            xs, ys = _compute_curriculum_trajectory(df, keys, consecutive_wins)
            if len(xs) > 1:
                trajectories.append((xs, ys))

        if not trajectories:
            return

        # Interpolate all trajectories onto the fixed-budget grid.
        # Trajectories that end early hold their last level for the
        # remainder -- an agent that ran out of budget on level 1
        # correctly drags the average down.
        interped = np.zeros((len(trajectories), len(grid)))
        for i, (xs, ys) in enumerate(trajectories):
            for j, g in enumerate(grid):
                idx = np.searchsorted(xs, g, side="right") - 1
                idx = max(0, min(idx, len(ys) - 1))
                interped[i, j] = ys[idx]

        mean_y = np.mean(interped, axis=0)
        sem_y = np.std(interped, axis=0, ddof=1) / np.sqrt(len(interped))

        if ann:
            ann.plot(
                grid,
                mean_y,
                label=label,
                color=color,
                marker=marker,
                linewidth=LINE_WIDTH_PRIMARY,
            )
            ax.fill_between(
                grid, mean_y - sem_y, mean_y + sem_y, color=color, alpha=0.15, zorder=1
            )
        else:
            ax.plot(
                grid,
                mean_y,
                label=label,
                color=color,
                linewidth=LINE_WIDTH_SECONDARY,
                alpha=0.7,
            )
            ax.fill_between(
                grid, mean_y - sem_y, mean_y + sem_y, color=color, alpha=0.1, zorder=1
            )

    # Non-human agents first
    for model_name in baselines:
        if model_name == "Human":
            continue
        s = _select_agent(sub, model_name)
        if len(s) == 0:
            continue
        color = BASELINE_COLORS.get(model_name, "#888")
        marker = BASELINE_MARKERS.get(model_name, "o")
        _plot_agent_mean(s, model_name, color, marker, model_name)

    for model_name in llms:
        if model_name == "Human":
            continue
        s = _select_agent(sub, model_name, "llm")
        if len(s) == 0:
            continue
        color = LLM_COLORS.get(model_name, "#888")
        marker = LLM_MARKERS.get(model_name, "o")
        label = LLM_SHORT.get(model_name, model_name)
        _plot_agent_mean(s, model_name, color, marker, label, "llm")

    # Human on top
    if "Human" in baselines or "Human" in llms:
        s = _select_agent(sub, "Human")
        if len(s) > 0:
            color = BASELINE_COLORS["Human"]
            marker = BASELINE_MARKERS["Human"]
            _plot_agent_mean(s, "Human", color, marker, "Human")

    ax.set_ylim(-0.3, 9.3)
    if ann:
        ann.annotate(fontsize=ANNOTATION_FONTSIZE, fmt="{:>3.1f}")


def plot_curriculum(
    raw_df: pd.DataFrame, output_dir: Path, consecutive_wins: int = 2
) -> None:
    gc_list = _game_cohort_list(raw_df)
    cohorts = sorted(set(c for _, c in gc_list))
    n_rows = len(cohorts)
    n_cols = max(sum(1 for _, c in gc_list if c == ch) for ch in cohorts)

    for variant_name, baselines, llms in [
        ("cross_paradigm", BASELINE_ORDER, BEST_LLM_MODELS),
        ("llms", ["Human"], ALL_LLM_MODELS),
    ]:
        fig, axes, ax_agg = make_figure(n_rows, n_cols)

        for idx, (game, cohort) in enumerate(gc_list):
            ax = axes[idx]
            _plot_curriculum_panel(
                ax, raw_df, baselines, llms, game, cohort, consecutive_wins
            )
            title = DISPLAY_NAMES.get(game, game)
            cl = cohort.replace("vgfmri", "v")
            ax.set_title(r"" + title + r" (" + cl + r")")
            if idx % n_cols == 0:
                ax.set_ylabel(r"Level reached")
            if idx >= n_cols * (n_rows - 1):
                ax.set_xlabel(r"Steps")

        _plot_curriculum_panel(
            ax_agg, raw_df, baselines, llms, None, None, consecutive_wins, annotate=True
        )
        ax_agg.set_title(r"All Games (avg)")
        ax_agg.set_xlabel(r"Steps")
        ax_agg.set_ylabel(r"Avg Level Reached")

        for ax in axes:
            ax.set_xlim(0, CURRICULUM_BUDGET)
        ax_agg.set_xlim(0, CURRICULUM_BUDGET)

        handles, labels = ax_agg.get_legend_handles_labels()
        ncol = 5 if variant_name == "llms" else len(handles)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
        )

        fig.subplots_adjust(bottom=0.25)
        out = output_dir / f"behavioural_curriculum_{variant_name}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def plot_rationale_compare(surv: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: solve rate per game, copied-reasoning vs action-only.

    Layout: 2 rows x 6 game panels (vgfmri3 top, vgfmri4 bottom) + aggregate
    panel on the right spanning both rows. Within each panel, paired bars per
    LLM model (solid = copied-reasoning, hatched = action-only). No x-tick
    labels; models identified by colour in a shared legend.
    """
    import matplotlib.patches as mpatches

    gc_list = _game_cohort_list(surv)
    cohorts = sorted(set(c for _, c in gc_list))
    n_rows = len(cohorts)
    n_cols = max(sum(1 for _, c in gc_list if c == ch) for ch in cohorts)

    fig, axes, ax_agg = make_figure(n_rows, n_cols)

    models = ALL_LLM_MODELS
    n_models = len(models)
    bar_width = 0.35
    x = np.arange(n_models)

    llm_surv = surv[surv["agent_type"] == "llm"]

    def _solve_rates(sub: pd.DataFrame, mode: str) -> list[float]:
        rates = []
        for model in models:
            mask = (
                (sub["model"] == model)
                & (sub["rationale_mode"] == mode)
                & (sub["suggestion_level"] == "elaborate")
            )
            m = sub[mask]
            if len(m) == 0:
                rates.append(0.0)
            else:
                rates.append(100.0 * m["observed"].sum() / len(m))
        return rates

    def _plot_panel(
        ax: plt.Axes, sub: pd.DataFrame, title: str, show_ylabel: bool
    ) -> None:
        cr_rates = _solve_rates(sub, "copied-reasoning")
        ao_rates = _solve_rates(sub, "action-only")

        bars_cr = ax.bar(x - bar_width / 2, cr_rates, bar_width)
        bars_ao = ax.bar(x + bar_width / 2, ao_rates, bar_width)

        for bar, model_key in zip(bars_cr, models):
            bar.set_color(LLM_COLORS[model_key])
            bar.set_edgecolor("black")
            bar.set_linewidth(0.8)
        for bar, model_key in zip(bars_ao, models):
            bar.set_color(LLM_COLORS[model_key])
            bar.set_alpha(0.35)
            bar.set_edgecolor("black")
            bar.set_linewidth(0.8)
            bar.set_hatch("//")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_ylim(0, 105)
        if show_ylabel:
            ax.set_ylabel(r"Solve rate (\%)")

    for idx, (game, cohort) in enumerate(gc_list):
        ax = axes[idx]
        sub = llm_surv[(llm_surv["game"] == game) & (llm_surv["cohort"] == cohort)]
        cl = cohort.replace("vgfmri", "v")
        title = DISPLAY_NAMES.get(game, game) + r" (" + cl + r")"
        _plot_panel(ax, sub, title, show_ylabel=(idx % n_cols == 0))

    _plot_panel(ax_agg, llm_surv, r"All Games", show_ylabel=True)

    # Legend: per-model colour patches + condition patches (solid vs hatched)
    handles = []
    labels = []
    for model in models:
        handles.append(
            mpatches.Patch(
                facecolor=LLM_COLORS[model], edgecolor="black", linewidth=0.6
            )
        )
        labels.append(LLM_SHORT[model])
    handles.append(
        mpatches.Patch(facecolor="#888888", edgecolor="black", linewidth=0.8)
    )
    labels.append("Copied-reasoning")
    handles.append(
        mpatches.Patch(
            facecolor="#888888",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.35,
            hatch="//",
        )
    )
    labels.append("Action-only")

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=len(handles),
        frameon=True,
        fontsize=16,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(bottom=0.25)

    out = output_dir / "behavioural_rationale_compare.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "figure",
        choices=[
            "km",
            "discovery",
            "discovery_combined",
            "discovery_curriculum_combined",
            "execution",
            "curriculum",
            "rationale_compare",
            "all",
        ],
        help="Which figure(s) to produce",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["empa2"],
        help="Exclude agent_type:cohort[:game] from plots. "
        "E.g.: --exclude ez:vgfmri3 ddqn:vgfmri3",
    )
    parser.add_argument(
        "--per-game-pool",
        action="store_true",
        help="Aggregate panel averages per-game KM CDFs "
        "instead of pooling all rows (compute-then-pool)",
    )
    parser.add_argument(
        "--raw-pool",
        action="store_true",
        help="Override per-game-pool: pool all rows directly",
    )
    parser.add_argument(
        "--include-ez-curriculum",
        action="store_true",
        help="EfficientZero only: add EZ warmup-curriculum steps "
        "(levels 9-12) to its first game level discovery time. "
        "Never applied to humans or LRMs.",
    )
    parser.add_argument(
        "--forced-blocked-curricula",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retroactively apply blocked-curricula rule: censor levels "
        "after first level without N consecutive wins",
    )
    parser.add_argument(
        "--consecutive-wins",
        type=int,
        default=2,
        help="Required consecutive wins for mastery (default: 2)",
    )
    parser.add_argument(
        "--censor-budget",
        type=int,
        default=4500,
        help="Censor time for unplayed levels (default: 4500, matches total frame budget)",
    )
    parser.add_argument("--csv", type=Path, default=CACHE_DIR / "episodes.csv")
    parser.add_argument(
        "--tex",
        action="store_true",
        help="Render labels using an installed LaTeX distribution",
    )
    args = parser.parse_args()
    plt.rcParams["text.usetex"] = args.tex
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_data(args.csv)
    print(f"Loaded {len(raw_df)} raw episode items")

    if args.exclude:
        exclusions = parse_exclusions(args.exclude)
        n_before = len(raw_df)
        raw_df = apply_exclusions(raw_df, exclusions)
        print(f"Exclusions {args.exclude}: {n_before} -> {len(raw_df)} items")

    print("Computing survival table from raw episodes...")
    surv = derive_survival(
        raw_df,
        censor_budget=args.censor_budget,
        include_ez_curriculum=args.include_ez_curriculum,
        censor_after_failure=args.forced_blocked_curricula,
        consecutive_wins_required=args.consecutive_wins,
    )
    print(f"Survival table: {len(surv)} rows, {surv['observed'].sum()} won")

    figs = (
        [args.figure]
        if args.figure != "all"
        else [
            "km",
            "discovery",
            "discovery_combined",
            "discovery_curriculum_combined",
            "execution",
            "curriculum",
            "rationale_compare",
        ]
    )

    for fig_type in figs:
        print(f"\nPlotting {fig_type}...")
        if fig_type == "km":
            pgp = args.per_game_pool and not args.raw_pool
            plot_km(surv, args.output_dir, per_game_pool=pgp)
        elif fig_type == "discovery":
            plot_discovery(surv, args.output_dir)
        elif fig_type == "discovery_combined":
            plot_discovery_aggregate(surv, args.output_dir)
        elif fig_type == "execution":
            # Build execution data from raw episodes
            exec_rows = []
            # Levels 9-11 exist for the vgfmri3 human cohort only and have no
            # agent counterpart; including them would shift the human reference
            # distribution used in the EMD, exactly as derive_survival() avoids
            # for discovery.
            for _, row in raw_df[raw_df["level"] <= 8].iterrows():
                ex = compute_execution(row["episode_steps"], row["episode_outcomes"])
                for s in ex:
                    exec_rows.append(
                        {
                            "agent_type": row["agent_type"],
                            "model": row["model"],
                            "rationale_mode": row["rationale_mode"],
                            "suggestion_level": row["suggestion_level"],
                            "instance_id": row["instance_id"],
                            "game": row["game"],
                            "cohort": row["cohort"],
                            "level": row["level"],
                            "steps": s,
                        }
                    )
            exec_df = pd.DataFrame(exec_rows)
            plot_execution(exec_df, args.output_dir)
        elif fig_type == "discovery_curriculum_combined":
            plot_discovery_curriculum_combined(
                surv, raw_df, args.output_dir, consecutive_wins=args.consecutive_wins
            )
        elif fig_type == "curriculum":
            plot_curriculum(
                raw_df, args.output_dir, consecutive_wins=args.consecutive_wins
            )
        elif fig_type == "rationale_compare":
            plot_rationale_compare(surv, args.output_dir)


if __name__ == "__main__":
    main()
