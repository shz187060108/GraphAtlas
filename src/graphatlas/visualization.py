from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from graphatlas.reporting import scientific_gates


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _ci95(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size <= 1:
        return 0.0
    return float(1.96 * array.std(ddof=1) / math.sqrt(array.size))


def _save_figure(fig: Any, output_stem: Path) -> list[str]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 320})):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    return paths


def _synthetic_classification(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[frame["dataset"].astype(str).str.lower().str.startswith("atlas_het")].copy()
    if "task" in result:
        result = result[result["task"].eq("node_classification")]
    return result


def _paired_values(frame: pd.DataFrame, left_model: str, right_model: str, metric: str) -> pd.DataFrame:
    keys = [column for column in ("dataset", "task", "seed", "split") if column in frame]
    if not keys or metric not in frame:
        return pd.DataFrame(columns=["left", "right"])
    left = frame[frame["model"].eq(left_model)][keys + [metric]].rename(columns={metric: "left"})
    right = frame[frame["model"].eq(right_model)][keys + [metric]].rename(columns={metric: "right"})
    return left.merge(right, on=keys).dropna(subset=["left", "right"])


def _short_model(name: str) -> str:
    aliases = {
        "graphatlas": "GraphAtlas",
        "graphatlas_no_transport": "No transport",
        "graphatlas_free_transition": "Free transition",
        "graphatlas_no_metric": "No metric",
        "graphatlas_no_cocycle": "No cocycle",
        "graphatlas_no_overlap": "No overlap",
        "geometry_moe": "Geometry-MoE",
        "gprgnn": "GPR-GNN",
        "resgcn": "ResGCN",
        "mixhop": "MixHop",
        "linkx": "LINKX",
        "h2gcn": "H2GCN",
        "appnp": "APPNP",
        "gcn": "GCN",
        "mlp": "MLP",
        "sgc": "SGC",
    }
    return aliases.get(name, name.replace("_", " ").title())




def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _plot_paired_panel(
    ax: Any,
    paired: pd.DataFrame,
    labels: tuple[str, str],
    ylabel: str,
    title: str,
) -> float | None:
    if paired.empty:
        ax.text(0.5, 0.5, "Insufficient paired runs", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    for _, row in paired.iterrows():
        ax.plot([0, 1], [row["left"], row["right"]], marker="o", alpha=0.72)
    delta = float((paired["right"] - paired["left"]).mean())
    ax.set_xticks([0, 1], list(labels))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.03, 0.96, f"Mean paired change: {delta:+.3f}", transform=ax.transAxes, va="top")
    ax.grid(axis="y", alpha=0.25)
    return delta

def _paired_figure(
    paired: pd.DataFrame,
    labels: tuple[str, str],
    ylabel: str,
    title: str,
    output_stem: Path,
) -> tuple[list[str], float | None]:
    import matplotlib.pyplot as plt

    if paired.empty:
        return [], None
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    for _, row in paired.iterrows():
        ax.plot([0, 1], [row["left"], row["right"]], marker="o", alpha=0.68)
    delta = float((paired["right"] - paired["left"]).mean())
    ax.set_xticks([0, 1], list(labels))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.03, 0.96, f"Mean paired change: {delta:+.3f}", transform=ax.transAxes, va="top")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    files = _save_figure(fig, output_stem)
    plt.close(fig)
    return files, delta


def plot_mechanism_figures(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    synthetic = _synthetic_classification(frame)
    variants = ["graphatlas", "graphatlas_no_transport", "graphatlas_free_transition"]
    required = set(variants)
    present = set(synthetic.get("model", pd.Series(dtype=str)).astype(str))
    if synthetic.empty or not required.issubset(present):
        return {
            "generated": False,
            "reason": "Atlas-Het classification results with full, no-transport, and free-transition variants are required.",
            "files": [],
        }

    means: list[float] = []
    errors: list[float] = []
    for model in variants:
        values = synthetic.loc[synthetic["model"].eq(model), "invariance_error_nonlinear"].astype(float)
        values = values[np.isfinite(values)]
        means.append(float(values.mean()) if not values.empty else float("nan"))
        errors.append(_ci95(values))
    floor = 1e-12
    plotted = np.maximum(np.asarray(means, dtype=float), floor)
    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    x = np.arange(len(variants))
    ax.bar(x, plotted, yerr=errors, capsize=3)
    ax.set_yscale("log")
    ax.set_xticks(x, [_short_model(model) for model in variants], rotation=14, ha="right")
    ax.set_ylabel("Nonlinear coordinate-intervention error")
    ax.set_title("Coordinate reparameterization stability")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    files = _save_figure(fig, output_dir / "mechanism_diagnostics")
    plt.close(fig)

    boundary = _paired_values(synthetic, "graphatlas_no_transport", "graphatlas", "boundary_accuracy")
    boundary_files, boundary_delta = _paired_figure(
        boundary,
        ("No transport", "GraphAtlas"),
        "Boundary-node accuracy",
        "Boundary-node mechanism",
        output_dir / "boundary_node_pairs",
    )
    transition = _paired_values(synthetic, "graphatlas_free_transition", "graphatlas", "test_metric")
    transition_files, transition_delta = _paired_figure(
        transition,
        ("Free transition", "GraphAtlas"),
        "Primary test metric",
        "Composition-induced transition test",
        output_dir / "transition_pairs",
    )

    perf = (
        synthetic.groupby("model", as_index=False)["test_metric"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=True)
        .tail(12)
    )
    perf["ci95"] = 1.96 * perf["std"].fillna(0.0) / np.sqrt(perf["count"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(5.2, max(3.0, 0.32 * len(perf) + 0.8)))
    y = np.arange(len(perf))
    ax.barh(y, perf["mean"], xerr=perf["ci95"], capsize=2)
    ax.set_yticks(y, [_short_model(name) for name in perf["model"]])
    ax.set_xlabel("Primary test metric")
    ax.set_title("Atlas-Het performance context")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    performance_files = _save_figure(fig, output_dir / "atlashet_performance")
    plt.close(fig)

    # Main-paper group figure. It separates mechanism evidence from predictive context
    # so a low intervention error cannot be mistaken for an accuracy claim.
    group, axes = plt.subplots(2, 2, figsize=(7.15, 5.55), constrained_layout=True)
    ax = axes[0, 0]
    bars = ax.bar(x, plotted, yerr=errors, capsize=3, edgecolor="black", linewidth=0.7)
    bars[0].set_hatch("///")
    ax.set_yscale("log")
    ax.set_xticks(x, [_short_model(model) for model in variants], rotation=12, ha="right")
    ax.set_ylabel("Intervention error")
    ax.set_title("Coordinate reparameterization")
    ax.grid(axis="y", alpha=0.25)
    _panel_label(ax, "a")

    ax = axes[0, 1]
    _plot_paired_panel(
        ax, boundary, ("No transport", "GraphAtlas"), "Boundary accuracy", "Boundary nodes"
    )
    _panel_label(ax, "b")

    ax = axes[1, 0]
    _plot_paired_panel(
        ax, transition, ("Free transition", "GraphAtlas"), "Primary metric", "Induced transitions"
    )
    _panel_label(ax, "c")

    ax = axes[1, 1]
    compact = perf.sort_values("mean", ascending=True).tail(8)
    yy = np.arange(len(compact))
    group_bars = ax.barh(yy, compact["mean"], xerr=compact["ci95"], capsize=2, edgecolor="black", linewidth=0.5)
    for bar, model_name in zip(group_bars, compact["model"]):
        if model_name == "graphatlas":
            bar.set_hatch("///")
    ax.set_yticks(yy, [_short_model(name) for name in compact["model"]])
    ax.set_xlabel("Primary test metric")
    ax.set_title("Predictive context")
    ax.grid(axis="x", alpha=0.25)
    _panel_label(ax, "d")

    evaluated_seeds = int(synthetic["seed"].nunique()) if "seed" in synthetic else 0
    group.suptitle("GraphAtlas mechanism diagnostics on Atlas-Het", y=1.025)
    group.text(
        0.5,
        -0.02,
        f"Panels b–d provide diagnostic context from {evaluated_seeds} seed(s); predictive claims require the scientific gates to pass.",
        ha="center",
        va="top",
        fontsize=8,
    )
    group_files = _save_figure(group, output_dir / "graphatlas_mechanism_group")
    plt.close(group)

    full_inv = float(synthetic.loc[synthetic["model"].eq("graphatlas"), "invariance_error_nonlinear"].mean())
    no_transport_inv = float(
        synthetic.loc[synthetic["model"].eq("graphatlas_no_transport"), "invariance_error_nonlinear"].mean()
    )
    free_inv = float(
        synthetic.loc[synthetic["model"].eq("graphatlas_free_transition"), "invariance_error_nonlinear"].mean()
    )
    all_files = files + boundary_files + transition_files + performance_files + group_files
    return {
        "generated": True,
        "files": all_files,
        "group_figure_files": group_files,
        "coordinate_invariance_files": files,
        "boundary_files": boundary_files,
        "transition_files": transition_files,
        "performance_files": performance_files,
        "full_invariance_error": full_inv,
        "no_transport_to_full_ratio": no_transport_inv / max(full_inv, floor),
        "free_transition_to_full_ratio": free_inv / max(full_inv, floor),
        "boundary_pairs": int(len(boundary)),
        "boundary_mean_delta": boundary_delta,
        "transition_pairs": int(len(transition)),
        "transition_mean_delta": transition_delta,
    }


def plot_atlas_diagnostics(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Plot path consistency, metric compatibility, and runtime for atlas variants.

    The figure is generated only when the result table contains at least two
    atlas variants and one finite diagnostic. This prevents empty decorative
    panels from being emitted for baseline-only result tables.
    """
    import matplotlib.pyplot as plt

    synthetic = _synthetic_classification(frame)
    variants = [
        model
        for model in (
            "graphatlas_no_transport",
            "graphatlas_free_transition",
            "graphatlas",
        )
        if model in set(synthetic.get("model", pd.Series(dtype=str)).astype(str))
    ]
    metrics = [
        ("path_consistency_error", "Path-consistency error", True),
        ("metric_compatibility_error", "Metric-compatibility error", True),
        ("runtime_seconds", "Runtime (seconds)", False),
    ]
    available = []
    for metric, label, log_scale in metrics:
        if metric not in synthetic:
            continue
        values = pd.to_numeric(synthetic[metric], errors="coerce")
        if np.isfinite(values).any():
            available.append((metric, label, log_scale))
    if len(variants) < 2 or not available:
        return {
            "generated": False,
            "reason": "At least two atlas variants and one finite atlas diagnostic are required.",
            "files": [],
        }

    fig, axes = plt.subplots(1, len(available), figsize=(2.55 * len(available), 2.75), squeeze=False)
    summaries: dict[str, dict[str, float]] = {}
    for panel_index, (metric, label, log_scale) in enumerate(available):
        ax = axes[0, panel_index]
        means: list[float] = []
        errors: list[float] = []
        summaries[metric] = {}
        for model in variants:
            values = pd.to_numeric(
                synthetic.loc[synthetic["model"].eq(model), metric], errors="coerce"
            )
            values = values[np.isfinite(values)]
            mean = float(values.mean()) if not values.empty else float("nan")
            means.append(mean)
            errors.append(_ci95(values))
            summaries[metric][model] = mean
        plotted = np.asarray(means, dtype=float)
        if log_scale:
            plotted = np.maximum(plotted, 1e-12)
        bars = ax.bar(
            np.arange(len(variants)),
            plotted,
            yerr=errors,
            capsize=2.5,
            edgecolor="black",
            linewidth=0.55,
        )
        for bar, model in zip(bars, variants, strict=True):
            if model == "graphatlas":
                bar.set_hatch("///")
        if log_scale and np.all(np.asarray(plotted) > 0):
            ax.set_yscale("log")
        ax.set_xticks(
            np.arange(len(variants)),
            [_short_model(model) for model in variants],
            rotation=17,
            ha="right",
        )
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        _panel_label(ax, chr(ord("a") + panel_index))
    fig.suptitle("Atlas consistency and efficiency diagnostics", y=1.04)
    fig.tight_layout()
    files = _save_figure(fig, output_dir / "atlas_consistency_efficiency_group")
    plt.close(fig)
    return {
        "generated": True,
        "files": files,
        "metrics": [metric for metric, _, _ in available],
        "means": summaries,
    }


def plot_benchmark_overview(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if frame.empty or "test_metric" not in frame:
        return {"generated": False, "reason": "No primary test metric is available.", "files": []}
    keys = [column for column in ("dataset", "task", "seed", "split") if column in frame]
    if not keys:
        return {"generated": False, "reason": "Run identity columns are unavailable.", "files": []}
    per_run = frame.groupby(keys + ["model"], as_index=False)["test_metric"].mean()
    per_run["rank"] = per_run.groupby(keys)["test_metric"].rank(ascending=False, method="average")
    rank_groups = [column for column in ("task", "model") if column in per_run]
    ranks = per_run.groupby(rank_groups, as_index=False).agg(
        mean_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        evaluated_runs=("rank", "count"),
    )
    tasks = sorted(ranks["task"].unique()) if "task" in ranks else ["all"]
    files: list[str] = []
    for task in tasks:
        subset = ranks[ranks["task"].eq(task)] if "task" in ranks else ranks
        subset = subset.sort_values("mean_rank", ascending=False)
        fig, ax = plt.subplots(figsize=(5.2, max(3.2, 0.3 * len(subset) + 0.8)))
        y = np.arange(len(subset))
        ax.barh(y, subset["mean_rank"], xerr=subset["std_rank"].fillna(0.0), capsize=2)
        ax.set_yticks(y, [_short_model(name) for name in subset["model"]])
        ax.set_xlabel("Mean rank, lower is better")
        ax.set_title(task.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        task_name = str(task).replace(" ", "_").lower()
        files.extend(_save_figure(fig, output_dir / f"benchmark_rank_{task_name}"))
        plt.close(fig)
    return {"generated": True, "files": files, "tasks": tasks, "models": int(ranks["model"].nunique())}


def write_claim_report(frame: pd.DataFrame, output_dir: Path, mechanism: dict[str, Any]) -> Path:
    gates = scientific_gates(frame)
    lines = [
        "# Figure and claim report",
        "",
        "This report is generated from the same result table as the figures.",
        "",
        "## Scientific gates",
        "",
        "| Gate | Status | Value | Criterion |",
        "|---|---:|---:|---|",
    ]
    for gate in gates["gates"]:
        status = "PASS" if gate["passed"] is True else "FAIL" if gate["passed"] is False else "N/A"
        value = gate["value"]
        value_text = f"{value:.6g}" if isinstance(value, float) else str(value)
        lines.append(f"| {gate['name']} | {status} | {value_text} | {gate['criterion']} |")
    lines.extend(["", "## Interpretation", ""])
    if mechanism.get("generated"):
        lines.append(
            "The coordinate-intervention figure validates a mechanism. It does not by itself establish superior predictive performance."
        )
    failed = [gate["name"] for gate in gates["gates"] if gate["passed"] is False]
    if failed:
        lines.append("Unsupported claims in the current result table: " + ", ".join(failed) + ".")
    else:
        lines.append("No evaluated scientific gate failed in the current result table.")
    path = output_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def visualize_results(results_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    _configure_matplotlib()
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    frame = pd.read_csv(results_path)
    if "test_metric" not in frame and "test_accuracy" in frame:
        frame["test_metric"] = frame["test_accuracy"]
    destination = Path(output_dir) if output_dir is not None else results_path.parent / "figures"
    destination.mkdir(parents=True, exist_ok=True)

    mechanism = plot_mechanism_figures(frame, destination)
    atlas_diagnostics = plot_atlas_diagnostics(frame, destination)
    benchmark = plot_benchmark_overview(frame, destination)
    report_path = write_claim_report(frame, destination, mechanism)
    manifest = {
        "results": str(results_path),
        "output_dir": str(destination),
        "mechanism_diagnostics": mechanism,
        "atlas_diagnostics": atlas_diagnostics,
        "benchmark_overview": benchmark,
        "claim_report": str(report_path),
    }
    (destination / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
