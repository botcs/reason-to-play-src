#!/usr/bin/env python3
"""Draw the preceding research's code, data and methodology lineage.

Run with Python + matplotlib. Outputs beside this file. The diagram encodes
milestones, not elapsed time. Links on the SVG/PDF cards point to primary sources.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parent
INK = "#182B43"
MUTED = "#53657A"
PAPER = "#F5F7FA"
WHITE = "#FFFFFF"
GRID = "#DCE4ED"
BLUE = "#2363A4"
TEAL = "#147D79"
PURPLE = "#78539B"
GOLD = "#A46A18"
STYLES = {"git": "solid", "data": (0, (1.2, 2.7)), "reported": (0, (5, 3))}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "svg.fonttype": "none",
        "svg.hashsalt": "reason-to-play-research-lineage",
        "pdf.fonttype": 42,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)


def canvas(width, height, title, subtitle, number):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    ax.text(0.65, height - 0.55, title, fontsize=27, weight="bold", color=INK, va="top")
    ax.text(0.67, height - 1.13, subtitle, fontsize=12.5, color=MUTED, va="top")
    ax.text(
        width - 0.65,
        height - 0.59,
        f"{number:02d}",
        fontsize=25,
        weight="bold",
        color=BLUE,
        ha="right",
        va="top",
    )
    return fig, ax


def section(ax, x, y, title, color=BLUE):
    ax.text(x, y, title.upper(), fontsize=10.8, weight="bold", color=color, va="bottom")


def card(ax, x, y, w, h, title, body, stamp="", color=BLUE, url=None, title_size=14):
    shadow = FancyBboxPatch(
        (x + 0.025, y - 0.035),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.11",
        linewidth=0,
        facecolor="#E5EAF1",
        zorder=2,
    )
    ax.add_patch(shadow)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.11",
        linewidth=0.8,
        edgecolor=GRID,
        facecolor=WHITE,
        zorder=3,
    )
    if url:
        patch.set_url(url)
    ax.add_patch(patch)
    ax.plot(
        [x + 0.02, x + 0.02],
        [y + 0.16, y + h - 0.16],
        color=color,
        lw=3,
        solid_capstyle="round",
        zorder=4,
    )
    ax.text(
        x + 0.24,
        y + h - 0.23,
        title,
        fontsize=title_size,
        weight="bold",
        color=INK,
        va="top",
        zorder=5,
    )
    ax.text(
        x + 0.24,
        y + h - 0.66,
        body,
        fontsize=12.2,
        linespacing=1.48,
        color=MUTED,
        va="top",
        zorder=5,
    )
    if stamp:
        ax.text(
            x + 0.24,
            y + 0.23,
            stamp,
            fontsize=10.4,
            color=color,
            weight="bold",
            va="bottom",
            zorder=5,
        )
    return {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
    }


def edge(ax, start, end, kind="git", color=MUTED, via=None, width=1.6):
    if via:
        pts = [start] + list(via) + [end]
        path = MplPath(pts, [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1))
        patch = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            linestyle=STYLES[kind],
            color=color,
            zorder=2,
        )
    else:
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            linestyle=STYLES[kind],
            color=color,
            shrinkA=4,
            shrinkB=5,
            zorder=2,
        )
    ax.add_patch(patch)


def label(ax, x, y, text, color=MUTED, size=10.0, ha="center"):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        ha=ha,
        va="center",
        linespacing=1.3,
        bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 2.0},
        zorder=6,
    )


def legend(ax, y, width):
    entries = [
        (0.7, "git", "Verified Git ancestry", MUTED),
        (6.3, "data", "Verified data / content / dependency", TEAL),
        (13.1, "reported", "Reported methodological adaptation", PURPLE),
    ]
    for x, kind, text, color in entries:
        edge(ax, (x, y), (x + 0.65, y), kind=kind, color=color)
        ax.text(x + 0.85, y, text, fontsize=10.4, color=MUTED, va="center")
    ax.text(
        0.7,
        y - 0.45,
        "Dates are milestones, not a time scale  ·  Solid lines may abbreviate intermediate commits",
        fontsize=10.1,
        color=MUTED,
    )
    ax.text(
        0.7,
        y - 0.74,
        "Primary links and evidence notes: README.md  ·  SVG/PDF cards link to the cited repositories, datasets or commits",
        fontsize=9.8,
        color=MUTED,
    )


def overview():
    fig, ax = canvas(
        24,
        17.2,
        "Reason to Play — code, data and method lineage",
        "The research's source relationships, with neural-analysis Git ancestry distinguished from file reuse.",
        1,
    )
    cols = [0.7, 6.45, 12.2, 17.95]
    w = 5.0
    section(ax, 0.7, 14.75, "Human experiment, raw data and preprocessing", TEAL)
    a = card(
        ax,
        cols[0],
        12.4,
        w,
        2.05,
        "Human experiment code",
        "tsividis/vgdl\nrefactor_fMRI_cannon\nTomov study infrastructure",
        "PRE-2025",
        TEAL,
        "https://github.com/tsividis/vgdl/tree/refactor_fMRI_cannon",
    )
    b = card(
        ax,
        cols[1],
        12.4,
        w,
        2.05,
        "OpenNeuro ds004323",
        "Original human dataset\nVersion 1.0.0\nRaw BIDS fMRI",
        "PRE-2025  ·  OPENNEURO",
        TEAL,
        "https://openneuro.org/datasets/ds004323/versions/1.0.0",
    )
    c = card(
        ax,
        cols[2],
        12.4,
        w,
        2.05,
        "Raw fMRIPrep launcher",
        "botcs/RC_RL\nActual raw-data launch script\nDDQN / EfficientZero work alongside",
        "16 OCT 2025  ·  969141d",
        TEAL,
        "https://github.com/botcs/RC_RL/commit/969141d",
    )
    d = card(
        ax,
        cols[3],
        12.4,
        w,
        2.05,
        "AWS staged RSA",
        "botcs/tomov23-analysis\nBOLD / model extraction, HRF, RDMs\nAWS orchestration developed next",
        "28 OCT 2025  ·  5c53c93",
        TEAL,
        "https://github.com/botcs/tomov23-analysis/commit/5c53c931fce464c5f708e3f416db4330be434fa8",
    )
    edge(ax, a["right"], b["left"], "data", TEAL)
    edge(ax, b["right"], c["left"], "data", TEAL)
    edge(ax, c["right"], d["left"], "data", TEAL)

    section(ax, 0.7, 11.85, "Reinforcement-learning code ancestry", BLUE)
    e = card(
        ax,
        cols[0],
        9.5,
        w,
        2.05,
        "ACampero/RC_RL",
        "Earlier reinforcement-learning\nimplementation and game tooling",
        "UPSTREAM LINEAGE",
        BLUE,
        "https://github.com/ACampero/RC_RL",
    )
    f = card(
        ax,
        cols[1],
        9.5,
        w,
        2.05,
        "tomov/RC_RL",
        "Inherited study / model code\nVerified upstream base for botcs fork",
        "UPSTREAM BASE  ·  8df431a",
        BLUE,
        "https://github.com/tomov/RC_RL/commit/8df431a",
    )
    g = card(
        ax,
        cols[2],
        9.5,
        w,
        2.05,
        "botcs/RC_RL",
        "First botcs work in the fork\nMRI, DDQN and EfficientZero interfaces\nFeeds the staged analysis project",
        "15 SEP 2025  ·  99566fd",
        BLUE,
        "https://github.com/botcs/RC_RL/commit/99566fd07cb5d31da16802b2b2422109af289441",
    )
    h = card(
        ax,
        cols[3],
        9.5,
        w,
        2.05,
        "Sreejan's encoding pipeline",
        "Parcelwise: 17 Dec 2025 · 3199db9\nVoxelwise: 28 Jan 2026 · 3e0f53a\nPost-fMRIPrep processing and fits",
        "TOMOV23-ANALYSIS  ·  SEE FIGURE 2",
        PURPLE,
        "https://github.com/botcs/tomov23-analysis/commit/3199db9946ada6328fab419d2c5f9fb1bb9c5ca3",
        title_size=13.4,
    )
    edge(ax, e["right"], f["left"], color=BLUE)
    edge(ax, f["right"], g["left"], color=BLUE)
    edge(ax, g["top"], c["bottom"], color=BLUE)
    edge(ax, d["bottom"], h["top"], color=PURPLE)
    label(ax, 20.45, 11.98, "Git branch; details in figure 2", PURPLE, 9.3)

    section(ax, 0.7, 8.95, "Methods and baseline tooling", PURPLE)
    i = card(
        ax,
        0.7,
        6.7,
        w,
        1.95,
        "Nature Communications (2024)",
        "tsumers/bert-brains · Kumar, Sumers et al.\nBanded ridge and delayed features\nNarratives ds002345 (different data)",
        "CODE: APR 2022  /  PAPER: 29 JUN 2024",
        PURPLE,
        "https://www.nature.com/articles/s41467-024-49173-5",
        title_size=13.4,
    )
    ez = card(
        ax,
        cols[1],
        6.7,
        w,
        1.95,
        "A-Andrews/EfficientZeroV2",
        "Austin's fork of EfficientZeroV2\nVGDL / curriculum work: Oct–Nov 2025\nImports RC_RL.VGDLEnv",
        "VGDL BRANCH: 17 JAN 2026  ·  29157d4",
        BLUE,
        "https://github.com/A-Andrews/EfficientZeroV2/tree/29157d4892afd9467b1bd0994de1355086145490",
        title_size=13.6,
    )
    card(
        ax,
        12.2,
        6.7,
        w,
        1.95,
        "Evidence boundary",
        "Shared author and methods verified.\nDirect source-file copying from\nbert-brains has not been established.",
        "ADAPTATION REPORTED BY PROJECT OWNER",
        PURPLE,
        "https://github.com/tsumers/bert-brains",
    )
    card(
        ax,
        cols[3],
        6.7,
        w,
        1.95,
        "Later analysis snapshots",
        "23 Mar 2026 · clean import 26fa74c\n31 Jul 2026 · submission 62c2021\nSibling Git children; some files reused",
        "EXACT TOPOLOGY IN FIGURE 2",
        PURPLE,
        "https://github.com/botcs/tomov23-analysis/commit/62c202130d057e585b6bf7978f621fcd0255919c",
    )
    edge(
        ax,
        i["right"],
        h["left"],
        "reported",
        PURPLE,
        via=[(6.04, 7.675), (6.04, 9.05), (17.58, 9.05), (17.58, 10.525)],
    )
    edge(
        ax,
        g["bottom"],
        ez["right"],
        "data",
        BLUE,
        via=[(14.7, 9.29), (11.82, 9.29), (11.82, 7.675)],
    )
    label(ax, 14.55, 9.065, "reported methodological adaptation", PURPLE, 9.6)

    section(ax, 0.7, 6.03, "Language-model harness and public artifacts", GOLD)
    j = card(
        ax,
        cols[0],
        3.65,
        w,
        2.05,
        "Colas: private infer-vgdl",
        "Original source of the January fork\nUpstream base: 64dc4f6d\nBase commit dated 21 Aug 2025",
        "PRIVATE UPSTREAM",
        GOLD,
    )
    k = card(
        ax,
        cols[1],
        3.65,
        w,
        2.05,
        "botcs/llm-vgdl",
        "Gameplay, replay and activations\nFirst botcs commit: 49a2bca2\nFirst harness: 68fc1024",
        "11 JAN 2026",
        GOLD,
        "https://github.com/botcs/llm-vgdl/commit/68fc1024",
    )
    public_source = card(
        ax,
        cols[2],
        3.65,
        w,
        2.05,
        "reason-to-play-src",
        "Public source repository\nDerived from the LLM harness\nMay source snapshot: 309e9a7",
        "9 MAY 2026  ·  9e5b2c8",
        GOLD,
        "https://github.com/botcs/reason-to-play-src/commit/9e5b2c8",
    )
    m = card(
        ax,
        cols[3],
        3.65,
        w,
        2.05,
        "reason-to-play website",
        "Public project and replay website\nCreated 7 May 2026 · c4583a2\nLinks separate source repo on 9 May",
        "9 MAY 2026  ·  SOURCE LINK e20c204",
        GOLD,
        "https://github.com/botcs/reason-to-play/commit/e20c204",
    )
    edge(ax, j["right"], k["left"], color=GOLD)
    edge(ax, k["right"], public_source["left"], "data", GOLD)
    edge(ax, public_source["right"], m["left"], "data", GOLD)
    card(
        ax,
        cols[0],
        1.95,
        10.75,
        1.27,
        "Later public Colas source: language_and_experience",
        "First source snapshot: 12 Feb 2026 · edb30bc",
        "LATER PARALLEL PUBLIC SNAPSHOT — NOT THE JANUARY FORK POINT",
        GOLD,
        "https://github.com/ccolas/language_and_experience/commit/edb30bc9815efad268605d337d63601d4f6f39a8",
        title_size=13.4,
    )
    ax.text(
        12.25,
        2.95,
        "Reading this figure",
        fontsize=12.5,
        color=INK,
        weight="bold",
        va="top",
    )
    ax.text(
        12.25,
        2.57,
        "These streams have separate histories. A repository snapshot date does not\nestablish when work began, when a model was run, or which results used it.\nThe analysis imports are expanded in the companion branch diagram.",
        fontsize=11.7,
        color=MUTED,
        va="top",
        linespacing=1.5,
    )
    legend(ax, 1.17, 24)
    return fig


def branches():
    fig, ax = canvas(
        24,
        13.7,
        "tomov23-analysis — Git ancestry and file reuse",
        "The March clean import and July submitted main have the same immediate parent. File reuse is a separate connection.",
        2,
    )
    section(ax, 0.7, 11.53, "Early AWS / RSA history", TEAL)
    a = card(
        ax,
        0.7,
        8.5,
        4.6,
        2.2,
        "AWS / RSA project",
        "Root: 28 Oct 2025 · 5c53c93\nStaged extraction and RSA\nShared history through early December",
        "3 DEC 2025  ·  d0fc2a0",
        TEAL,
        "https://github.com/botcs/tomov23-analysis/commit/d0fc2a0d459b5f796ce75fe9582bf0a09f0e99d3",
    )
    section(ax, 6.5, 11.53, "First encoding branch", PURPLE)
    b = card(
        ax,
        6.5,
        9.0,
        4.6,
        2.2,
        "Parcelwise encoding",
        "Sreejan Kumar\nFirst encoding import\nSchaefer parcels and ridge fits",
        "17 DEC 2025  ·  3199db9",
        PURPLE,
        "https://github.com/botcs/tomov23-analysis/commit/3199db9946ada6328fab419d2c5f9fb1bb9c5ca3",
    )
    c = card(
        ax,
        12.3,
        9.0,
        4.6,
        2.2,
        "Voxelwise encoding",
        "Sreejan Kumar\nVoxelwise preprocessing and fits\nVia 18 Dec revision · 718e09c",
        "28 JAN 2026  ·  3e0f53a",
        PURPLE,
        "https://github.com/botcs/tomov23-analysis/commit/3e0f53a750c6544fa9a6261adeb04c6441f08803",
    )
    edge(ax, a["right"], b["left"], color=PURPLE)
    edge(ax, b["right"], c["left"], color=PURPLE)
    label(
        ax,
        19.9,
        10.1,
        "encoding_model branch\nJanuary 2026 branch tip",
        PURPLE,
        12,
    )
    d = card(
        ax,
        0.7,
        4.05,
        4.6,
        2.2,
        "AWS branch continued",
        "botcs\nAdditional extraction and\nRSA maintenance",
        "11 DEC 2025  ·  d8eaa5a",
        TEAL,
        "https://github.com/botcs/tomov23-analysis/commit/d8eaa5a6c7fc3382967cb9e004748c89c99eb526",
    )
    edge(ax, a["bottom"], d["top"], color=TEAL)
    label(ax, 3.0, 7.34, "intermediate Git commits", TEAL, 10.3)

    section(ax, 8.0, 8.01, "Two imports with the same immediate Git parent", PURPLE)
    e = card(
        ax,
        8.0,
        5.35,
        6.1,
        2.2,
        "Clean encoding import",
        "Sreejan Kumar · encoding_model_slurm/\nFull preprocessing, base alignment,\nencoding and decoding; SLURM launchers",
        "23 MAR 2026  ·  26fa74c",
        PURPLE,
        "https://github.com/botcs/tomov23-analysis/commit/26fa74ce715781bfdd7a6f6a4bc2a6685e5a8678",
    )
    f = card(
        ax,
        8.0,
        2.0,
        6.1,
        2.2,
        "Submitted main",
        "botcs · encoding_model_code/\nFive-file supplementary snapshot\nTag: encoding-suppmat-v1.0",
        "31 JUL 2026  ·  62c2021",
        BLUE,
        "https://github.com/botcs/tomov23-analysis/commit/62c202130d057e585b6bf7978f621fcd0255919c",
    )
    edge(ax, d["right"], e["left"], color=PURPLE, via=[(6.6, 5.15), (6.6, 6.45)])
    edge(ax, d["right"], f["left"], color=BLUE, via=[(6.6, 5.15), (6.6, 3.1)])
    label(ax, 6.48, 4.43, "same\nparent", MUTED, 10.3)
    edge(ax, e["bottom"], f["top"], "data", TEAL)
    label(
        ax,
        11.05,
        4.77,
        "preprocess.py reused byte-for-byte\nGit blob: 3f78dbfa617c…",
        TEAL,
        10.5,
    )

    g = card(
        ax,
        17.0,
        5.35,
        6.0,
        2.2,
        "Reproduction research branch",
        "Later July–August edits\nEncoder diagnostics and geometry work\nencoding_model-reproduction_temporal",
        "A SEPARATE DESCENDANT OF THE CLEAN IMPORT",
        PURPLE,
        "https://github.com/botcs/tomov23-analysis/commit/d49655c8cb179e6d7ecd601ccb766eb34331404c",
        title_size=13.6,
    )
    edge(ax, e["right"], g["left"], color=PURPLE)
    card(
        ax,
        17.0,
        2.0,
        6.0,
        2.2,
        "What the content comparison shows",
        "March / July preprocessing: identical.\nAlignment and encoder files: different.\nJuly aligner adds LLMs to an existing base;\nMarch aligner also builds that base.",
        "NO GIT MERGE CONNECTS THE TWO IMPORTS",
        TEAL,
        "https://github.com/botcs/tomov23-analysis/tree/62c202130d057e585b6bf7978f621fcd0255919c",
        title_size=12.9,
    )
    ax.text(
        0.75,
        2.9,
        "Commit dates show recorded imports.\nThe project owner reports recruitment\nin November 2025; the first committed\nencoding import is 17 December.",
        fontsize=11.4,
        color=MUTED,
        linespacing=1.55,
        va="top",
    )
    legend(ax, 1.13, 24)
    return fig


def save(fig, name):
    metadata = {
        "Title": name.replace("-", " "),
        "Author": "Reason to Play",
        "Subject": "Research code, data and methodology lineage",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(OUT / f"{name}.png", dpi=160)
    fig.savefig(
        OUT / f"{name}.svg",
        metadata={
            "Title": metadata["Title"],
            "Description": metadata["Subject"],
            "Date": None,
        },
    )
    svg_path = OUT / f"{name}.svg"
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    fig.savefig(OUT / f"{name}.pdf", metadata=metadata)


if __name__ == "__main__":
    overview_fig = overview()
    branches_fig = branches()
    save(overview_fig, "lineage")
    save(branches_fig, "lineage-analysis-branches")
    with PdfPages(
        OUT / "lineage-all.pdf",
        metadata={
            "Title": "Reason to Play — project lineage",
            "Author": "Reason to Play",
            "Subject": "Overview and neural-analysis branch topology",
            "CreationDate": None,
            "ModDate": None,
        },
    ) as pdf:
        pdf.savefig(overview_fig)
        pdf.savefig(branches_fig)
    plt.close("all")
    print(f"Created PNG, SVG and PDF figures in {OUT}")
