"""Shared, compact Nature-style matplotlib conventions for GraphAtlas figures."""
from __future__ import annotations

from pathlib import Path
from typing import Any


MODEL_COLORS = {
    "graphatlas_c": "#176D8A",
    "graphatlas_c_q_reference": "#176D8A",
    "graphatlas_c_oracle": "#176D8A",
    "graphatlas_certified": "#176D8A",
    "atn": "#176D8A",
    "graphatlas_original": "#475569",
    "graphatlas": "#475569",
    "graphatlas_min_distortion": "#7C5C9E",
    "graphatlas_no_transport": "#C76A2A",
    "graphatlas_free_transition": "#B34D63",
    "geometry_moe": "#7A8F46",
    "ambient_vector_gnn": "#6B7280",
    "gcn": "#9CA3AF",
    "gprgnn": "#64748B",
    "linkx": "#A16207",
    "acmgcn": "#0F766E",
}

MODEL_DISPLAY_NAMES = {
    "graphatlas_c": "ATN",
    "graphatlas_c_q_reference": "ATN",
    "graphatlas_c_oracle": "ATN",
    "graphatlas_certified": "ATN",
    "atn": "ATN",
    "graphatlas_original": "GraphAtlas",
}


def configure_nature_style() -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 9,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def model_color(model: str) -> str:
    return MODEL_COLORS.get(str(model), "#6B7280")


def model_display_name(model: str) -> str:
    value = str(model)
    if value.startswith("graphatlas_c"):
        return "ATN" + value.removeprefix("graphatlas_c").replace("_", " ")
    return MODEL_DISPLAY_NAMES.get(value, value.replace("_", " "))


def nature_panel_label(ax: Any, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=9,
            ha="left", va="top")


def apply_spine_style(ax: Any) -> None:
    for name in ("left", "bottom"):
        ax.spines[name].set_linewidth(0.65)
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.45, alpha=0.55, zorder=0)
    ax.tick_params(width=0.6, length=2.5, pad=2)


def finalize_axes(ax: Any, xlabel: str | None = None, ylabel: str | None = None,
                  title: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
    apply_spine_style(ax)


def save_figure_bundle(fig: Any, output_stem: str | Path) -> list[str]:
    """The user-facing export contract is SVG only; text stays editable."""
    path = Path(output_stem).with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor="white")
    return [str(path)]
