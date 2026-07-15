"""Homogeneous atlas figures: one scientific question and one chart grammar per figure."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .figure_data import (
    build_beta_sweep_frame,
    build_counterfactual_frame,
    build_decile_response_frame,
    build_invariance_curve_frame,
    build_phase_diagram_frame,
    build_rank_summary_frame,
    build_risk_calibration_frame,
    build_routing_intervention_frame,
    build_runtime_pareto_frame,
    build_seed_level_slope_frame,
    build_transport_diagnostics_frame,
    load_results_frame,
)
from .figure_style import configure_nature_style, finalize_axes, model_color, model_display_name, nature_panel_label

EXPORT_FORMATS = ("svg", "pdf", "png")
EXPORT_DPI = 450


def _save(fig: Any, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(f".{ext}") for ext in EXPORT_FORMATS]
    for path in paths:
        fig.savefig(path, format=path.suffix[1:], dpi=EXPORT_DPI, facecolor="white", bbox_inches="tight", pad_inches=.025)
    return [str(path) for path in paths]


def _unavailable(ax: Any, reason: str) -> None:
    ax.text(.5, .5, f"Unavailable\n{reason}", ha="center", va="center", transform=ax.transAxes, color="#64748B", fontsize=6)
    ax.set_axis_off()


def _label_axes(axes: Any) -> None:
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes).flat):
        nature_panel_label(ax, label)


def _write(data: pd.DataFrame, root: Path, name: str) -> str:
    path = root / "data" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return str(path)


def _groups(data: pd.DataFrame, columns: list[str], limit: int) -> list[tuple[tuple[Any, ...], pd.DataFrame]]:
    present = [column for column in columns if column in data and data[column].notna().any()]
    if not present:
        return [((), data)]
    return list(data.groupby(present, dropna=False, sort=True))[:limit]


def _line_atlas(data: pd.DataFrame, *, metric: str, name: str, title: str, x: str, y: str, groups: list[str], root: Path,
                supplementary: bool = False, logy: bool = False) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    panels = _groups(data, groups, 12) if not data.empty else [((), data)]
    fig, axes = plt.subplots(3, 4, figsize=(7.25, 5.7), sharex=False, sharey=False, constrained_layout=True)
    _label_axes(axes)
    for ax, key in zip(axes.flat, panels):
        label, item = key
        if item.empty or x not in item or y not in item or item[y].dropna().empty:
            _unavailable(ax, "required analysis rows missing")
            continue
        for model, model_rows in item.groupby("model", dropna=False):
            model_rows = model_rows.sort_values(x)
            summary = model_rows.groupby(x, as_index=False)[y].agg(mean="mean", std="std", n="count")
            summary["ci"] = 1.96 * summary["std"].fillna(0) / np.sqrt(summary["n"].clip(lower=1))
            if logy:
                # Preserve zero-valued stability measurements without asking
                # matplotlib to take log(0); the epsilon is plotting-only.
                summary["mean"] = pd.to_numeric(summary["mean"], errors="coerce").clip(lower=1e-12)
                summary["ci_low"] = (summary["mean"] - summary["ci"]).clip(lower=1e-12)
                summary["ci_high"] = (summary["mean"] + summary["ci"]).clip(lower=1e-12)
            color = model_color(str(model)); label_name = model_display_name(str(model))
            ax.plot(summary[x], summary["mean"], color=color, lw=1.35, label=label_name)
            if summary["ci"].notna().any():
                low = summary.get("ci_low", summary["mean"] - summary["ci"]); high = summary.get("ci_high", summary["mean"] + summary["ci"])
                ax.fill_between(summary[x], low, high, color=color, alpha=.12, linewidth=0)
        ax.set_title(" / ".join(map(str, key[0])) if key[0] else title, loc="left", fontsize=7)
        if logy: ax.set_yscale("log")
        finalize_axes(ax, x.replace("_", " ").title(), y.replace("_", " ").title(), None)
    for ax in list(axes.flat)[len(panels):]:
        _unavailable(ax, "no additional panel condition")
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend: legend.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(.5, 1.01))
    directory = root / ("supplementary" if supplementary else "main")
    return {"files": _save(fig, directory / name), "data": data.to_dict("records"), "question": title}


def _phase_heat_atlas(data: pd.DataFrame, *, value: str, name: str, title: str, root: Path, supplementary: bool = False) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    panels = _groups(data, ["contrast", "dataset"], 12) if not data.empty else [((), data)]
    fig, axes = plt.subplots(3, 4, figsize=(7.25, 5.7), constrained_layout=True)
    _label_axes(axes)
    finite = pd.to_numeric(data.get(value, pd.Series(dtype=float)), errors="coerce") if not data.empty else pd.Series(dtype=float)
    if value in {"mean", "mean_delta", "boundary_gain"}:
        vmax = float(np.nanmax(np.abs(finite))) if finite.notna().any() else 1.
        vmin, vmax = -max(vmax, 1e-9), max(vmax, 1e-9); cmap = "RdBu_r"
    else:
        vmin, vmax, cmap = 0., 1., "viridis"
    for ax, key in zip(axes.flat, panels):
        label, item = key
        item = item.dropna(subset=[value, "overlap", "cross_chart_edge_fraction"]) if not item.empty else item
        if item.empty:
            _unavailable(ax, "phase rows missing")
            continue
        scatter = ax.scatter(item["cross_chart_edge_fraction"], item["overlap"], c=item[value], cmap=cmap, vmin=vmin, vmax=vmax, marker="s", s=75, linewidth=.2, edgecolor="white")
        ax.set_title(" / ".join(map(str, label)) if label else title, loc="left", fontsize=7)
        finalize_axes(ax, "Cross-chart fraction", "Overlap", None)
    for ax in list(axes.flat)[len(panels):]:
        _unavailable(ax, "no additional phase condition")
    if finite.notna().any(): fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=.65, label=value.replace("_", " "))
    directory = root / ("supplementary" if supplementary else "main")
    return {"files": _save(fig, directory / name), "data": data.to_dict("records"), "question": title}


def _slope_atlas(data: pd.DataFrame, *, name: str, title: str, root: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    panels = _groups(data, ["dataset"], 8) if not data.empty else [((), data)]
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 3.9), constrained_layout=True)
    _label_axes(axes)
    for ax, key in zip(axes.flat, panels):
        label, item = key
        if item.empty:
            _unavailable(ax, "paired seed rows missing")
            continue
        for _, row in item.iterrows():
            ax.plot([0, 1], [row["baseline"], row["target"]], color=model_color("graphatlas_c_oracle"), lw=.65, alpha=.42, marker="o", ms=2)
        ax.set_xticks([0, 1], ["Original", "ATN"]); ax.set_title(str(label[0]) if label else title, loc="left", fontsize=7)
        finalize_axes(ax, "Model", "Test metric", None)
    for ax in list(axes.flat)[len(panels):]:
        _unavailable(ax, "no additional dataset")
    directory = root / "main"
    return {"files": _save(fig, directory / name), "data": data.to_dict("records"), "question": title}


def _routing_atlas(data: pd.DataFrame, *, name: str, title: str, root: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    panels = _groups(data, ["dataset"], 12) if not data.empty else [((), data)]
    fig, axes = plt.subplots(3, 4, figsize=(7.25, 5.7), constrained_layout=True)
    _label_axes(axes)
    for ax, key in zip(axes.flat, panels):
        label, item = key
        if item.empty:
            _unavailable(ax, "routing intervention rows missing")
            continue
        modes = list(item["routing_mode"].dropna().astype(str).unique())
        pivot = item.pivot_table(index=[k for k in ("seed", "split") if k in item], columns="routing_mode", values="test_metric", aggfunc="mean")
        for index, row in pivot.iterrows():
            usable = [(mode, row.get(mode)) for mode in modes if pd.notna(row.get(mode))]
            if len(usable) > 1:
                ax.plot(range(len(usable)), [value for _, value in usable], color="#94A3B8", lw=.6, alpha=.45, marker="o", ms=2)
        means = item.groupby("routing_mode", sort=False)["test_metric"].mean().reindex(modes)
        ax.plot(range(len(means)), means, color=model_color("graphatlas_c_oracle"), lw=1.4, marker="o", ms=2.5)
        ax.set_xticks(range(len(modes)), [str(mode).replace("_", " ") for mode in modes], rotation=45, ha="right")
        ax.set_title(str(label[0]) if label else title, loc="left", fontsize=7); finalize_axes(ax, "Routing mode", "Test metric", None)
    for ax in list(axes.flat)[len(panels):]:
        _unavailable(ax, "no additional dataset")
    directory = root / "main"
    return {"files": _save(fig, directory / name), "data": data.to_dict("records"), "question": title}


def _calibration_atlas(data: pd.DataFrame, *, name: str, title: str, root: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    panels = _groups(data, ["dataset"], 12) if not data.empty else [((), data)]
    fig, axes = plt.subplots(3, 4, figsize=(7.25, 5.7), constrained_layout=True)
    _label_axes(axes)
    for ax, key in zip(axes.flat, panels):
        label, item = key
        if item.empty:
            _unavailable(ax, "node risk and error columns are required")
            continue
        for model, rows in item.groupby("model"):
            rows = rows.sort_values("predicted_risk")
            ax.plot(rows["predicted_risk"], rows["error_rate"], marker="o", ms=2, lw=1.0, color=model_color(str(model)), label=model_display_name(str(model)))
        ax.plot([0, 1], [0, 1], ls="--", lw=.6, color="#94A3B8")
        ax.set_title(str(label[0]) if label else title, loc="left", fontsize=7); finalize_axes(ax, "Predicted transport risk", "Observed error rate", None)
    for ax in list(axes.flat)[len(panels):]:
        _unavailable(ax, "no additional dataset")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(.5, 1.01))
    directory = root / "main"
    return {"files": _save(fig, directory / name), "data": data.to_dict("records"), "question": title}


def _ego_atlas(cases_csv: str | Path | None, cases_dir: str | Path | None, root: Path, *, name: str, title: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    cases = pd.read_csv(cases_csv) if cases_csv and Path(cases_csv).exists() else pd.DataFrame()
    if not cases.empty and "case_category" in cases:
        cases = cases[cases["case_category"].isin(["rescued", "failed"])].head(20)
    fig, axes = plt.subplots(4, 5, figsize=(7.25, 5.8), constrained_layout=True)
    _label_axes(axes)
    data = cases.copy()
    for ax, (_, row) in zip(axes.flat, cases.iterrows() if not cases.empty else []):
        case_id = str(row.get("case_id", "")); path = Path(cases_dir or "") / "tensors" / f"{case_id}.npz"
        if not path.exists(): _unavailable(ax, "case tensor missing"); continue
        payload = np.load(path); nodes = payload["node_ids"]; edges = payload["edge_index"]
        theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False); xy = np.column_stack([np.cos(theta), np.sin(theta)])
        for src, dst in edges.T: ax.plot(xy[[src, dst], 0], xy[[src, dst], 1], color="#CBD5E1", lw=.45, zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=10, c="#3B82F6", zorder=2); center = np.where(nodes == int(row["node_id"]))[0]
        if len(center): ax.scatter(xy[center[0], 0], xy[center[0], 1], s=28, c="#D94841", edgecolor="white", linewidth=.4, zorder=3)
        ax.set_title(f"{row.get('case_category', '')} n={int(row['node_id'])}", loc="left", fontsize=6); ax.set_axis_off()
    for ax in list(axes.flat)[len(cases):]: _unavailable(ax, "case not available")
    return {"files": _save(fig, root / "main" / name), "data": data.to_dict("records"), "question": title}


def _build_one(name: str, result: dict[str, Any], root: Path, manifest: dict[str, Any], report: dict[str, Any], source: str) -> None:
    data = pd.DataFrame(result.pop("data", [])); data_path = _write(data, root, name); result.update(name=name, data=data_path)
    result["panels"] = [{"panel_id": label, "scientific_question": result.get("question", ""), "input_files": [source], "input_row_count": len(data), "missing_reason": None if not data.empty else "required source columns or rows unavailable"} for label in "abcdefghijkl"[:12]]
    manifest["figures"].append(result); report["generated"].append(name)


def build_dense_paper_figures(results: str | Path | pd.DataFrame, output_dir: str | Path = "outputs/reports/paper_dense", *, target_model: str = "graphatlas_c_oracle", stress_csv: str | Path | None = None, cases_csv: str | Path | None = None, cases_dir: str | Path | None = None, calibration_csv: str | Path | None = None, include_upper_bound: bool = False, bootstrap_reps: int = 10000, bootstrap_seed: int = 2026, formats: Iterable[str] = EXPORT_FORMATS, dpi: int = 450, figures: Iterable[str] | None = None) -> dict[str, Any]:
    """Build homogeneous main and supplementary atlases from traceable tables.

    No training or interpolation is performed. Missing node-level artifacts are
    represented as unavailable panels instead of being replaced by run-level means.
    """
    global EXPORT_FORMATS, EXPORT_DPI
    EXPORT_FORMATS = tuple(item.lstrip(".") for item in formats); EXPORT_DPI = int(dpi); configure_nature_style()
    root = Path(output_dir)
    for directory in ("main", "supplementary", "data", "manifests"): (root / directory).mkdir(parents=True, exist_ok=True)
    all_rows = load_results_frame(results, include_upper_bound=True)
    formal = all_rows[all_rows["is_formal_result"].astype(bool)].copy() if "is_formal_result" in all_rows else all_rows.copy()
    available = set(formal.get("model", pd.Series(dtype=str)).astype(str)); target = target_model if target_model == "graphatlas_c_oracle" or target_model in available else target_model
    stress = pd.read_csv(stress_csv) if stress_csv and Path(stress_csv).exists() else build_invariance_curve_frame(formal)
    phase = build_phase_diagram_frame(formal, target)
    slopes = build_seed_level_slope_frame(formal, target)
    routing = build_routing_intervention_frame(formal)
    runtime = build_runtime_pareto_frame(formal)
    calibration = build_risk_calibration_frame(pd.read_csv(calibration_csv) if calibration_csv and Path(calibration_csv).exists() else formal)
    chosen = set(figures or [])
    main = [
        ("coordinate_metric_drop_atlas", lambda: _line_atlas(stress[stress.get("metric", pd.Series(index=stress.index)).eq("test_metric_drop")], metric="test_metric_drop", name="coordinate_metric_drop_atlas", title="Coordinate strength and task metric drop", x="strength", y="mean", groups=["transform_family", "dataset"], root=root)),
        ("mechanism_gain_phase_atlas", lambda: _phase_heat_atlas(phase, value="mean", name="mechanism_gain_phase_atlas", title="ATN gain across overlap and cross-chart fraction", root=root)),
        ("certificate_calibration_atlas", lambda: _calibration_atlas(calibration, name="certificate_calibration_atlas", title="Transport risk calibration against observed error", root=root)),
        ("routing_intervention_atlas", lambda: _routing_atlas(routing, name="routing_intervention_atlas", title="Paired performance under routing interventions", root=root)),
        ("real_benchmark_slope_atlas", lambda: _slope_atlas(slopes, name="real_benchmark_slope_atlas", title="Seed-level Original to ATN change", root=root)),
        ("opportunity_response_atlas", lambda: _line_atlas(build_decile_response_frame(formal, x_column="routing_opportunity"), metric="gain", name="opportunity_response_atlas", title="Performance gain by routing-opportunity decile", x="decile", y="gain_mean", groups=["dataset", "model"], root=root)),
        ("rescued_failed_ego_graph_atlas", lambda: _ego_atlas(cases_csv, cases_dir or (Path(cases_csv).parent if cases_csv else None), root, name="rescued_failed_ego_graph_atlas", title="Rescued and failed local ego graphs")),
        ("runtime_scaling_atlas", lambda: _line_atlas(runtime.rename(columns={"runtime_seconds": "runtime"}), metric="runtime", name="runtime_scaling_atlas", title="Runtime scaling with graph size", x="num_nodes", y="runtime", groups=["model"], root=root, logy=True)),
    ]
    supplementary = [
        ("coordinate_logit_stability_atlas", lambda: _line_atlas(stress[stress.get("metric", pd.Series(index=stress.index)).eq("max_logit_error")], metric="max_logit_error", name="coordinate_logit_stability_atlas", title="Coordinate strength and maximum logit error", x="strength", y="mean", groups=["transform_family", "dataset"], root=root, supplementary=True, logy=True)),
        ("coordinate_probability_stability_atlas", lambda: _line_atlas(stress[stress.get("metric", pd.Series(index=stress.index)).eq("mean_probability_error")], metric="mean_probability_error", name="coordinate_probability_stability_atlas", title="Coordinate strength and probability error", x="strength", y="mean", groups=["transform_family", "dataset"], root=root, supplementary=True, logy=True)),
        ("coordinate_flip_atlas", lambda: _line_atlas(stress[stress.get("metric", pd.Series(index=stress.index)).eq("prediction_flip_rate")], metric="prediction_flip_rate", name="coordinate_flip_atlas", title="Coordinate strength and prediction flip rate", x="strength", y="mean", groups=["transform_family", "dataset"], root=root, supplementary=True)),
        ("transport_risk_phase_atlas", lambda: _phase_heat_atlas(phase, value="irreducible_transport_risk", name="transport_risk_phase_atlas", title="Irreducible transport risk phase atlas", root=root, supplementary=True)),
        ("routing_opportunity_phase_atlas", lambda: _phase_heat_atlas(phase, value="routing_opportunity", name="routing_opportunity_phase_atlas", title="Routing opportunity phase atlas", root=root, supplementary=True)),
        ("routing_agreement_phase_atlas", lambda: _phase_heat_atlas(phase, value="oracle_agreement", name="routing_agreement_phase_atlas", title="Route agreement phase atlas", root=root, supplementary=True)),
        ("model_rank_atlas", lambda: _line_atlas(build_rank_summary_frame(formal), metric="rank", name="model_rank_atlas", title="Mean model rank", x="model", y="mean", groups=["model"], root=root, supplementary=True)),
        ("depth_performance_atlas", lambda: _line_atlas(formal, metric="test_metric", name="depth_performance_atlas", title="Depth and task performance", x="num_layers", y="test_metric", groups=["dataset"], root=root, supplementary=True)),
        ("memory_scaling_atlas", lambda: _line_atlas(runtime.rename(columns={"peak_cuda_memory_mb": "memory"}), metric="memory", name="memory_scaling_atlas", title="Memory scaling with graph size", x="num_nodes", y="memory", groups=["model"], root=root, supplementary=True, logy=True)),
        ("transport_diagnostics_atlas", lambda: _line_atlas(build_transport_diagnostics_frame(formal), metric="transportability_mean", name="transport_diagnostics_atlas", title="Transport diagnostics by dataset", x="seed", y="transportability_mean", groups=["dataset"], root=root, supplementary=True)),
    ]
    manifest = {"target_model": target, "formal_rows": int(len(formal)), "excluded_test_selected_rows": int(len(all_rows) - len(formal)), "bootstrap_resamples": int(bootstrap_reps), "bootstrap_seed": int(bootstrap_seed), "outputs": list(EXPORT_FORMATS), "figures": []}
    report = {"generated": [], "failed": [], "formal_source": str(results), "formal_rows": int(len(formal)), "excluded_test_selected_rows": int(len(all_rows) - len(formal)), "unavailable": []}
    for name, builder in [*main, *supplementary]:
        if chosen and name not in chosen: continue
        try:
            _build_one(name, builder(), root, manifest, report, str(results))
        except Exception as exc:
            report["failed"].append({"name": name, "reason": f"{type(exc).__name__}: {exc}"})
            manifest["figures"].append({"name": name, "generated": False, "unavailable_reason": str(exc)})
    (root / "manifests" / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "manifests" / "figure_build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
