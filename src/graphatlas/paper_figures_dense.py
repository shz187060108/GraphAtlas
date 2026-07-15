"""Dense, reproducible paper figures built from existing result tables only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .figure_data import (
    build_beta_sweep_frame, build_counterfactual_frame, build_invariance_curve_frame,
    build_phase_diagram_frame, build_rank_summary_frame, build_real_benchmark_frame,
    build_runtime_pareto_frame, build_transport_diagnostics_frame, load_results_frame,
    paired_effect,
)
from .figure_style import configure_nature_style, finalize_axes, model_color, model_display_name, nature_panel_label
from .reporting import resolve_target_model

EXPORT_FORMATS = ("svg", "pdf", "png")
EXPORT_DPI = 450


def _unavailable(ax, reason: str) -> None:
    ax.text(0.5, 0.5, f"Unavailable\n{reason}", ha="center", va="center", transform=ax.transAxes, color="#64748B", fontsize=6)
    ax.set_axis_off()


def _save(fig, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(f".{ext}") for ext in EXPORT_FORMATS]
    for path in paths:
        fig.savefig(path, format=path.suffix[1:], dpi=EXPORT_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    return [str(path) for path in paths]


def _write(data: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True); data.to_csv(path, index=False); return str(path)


def _empty_panel_data(name: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame([{"panel": name, "available": False, "unavailable_reason": reason}])


def _panels(fig, axes) -> None:
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes).flat):
        nature_panel_label(ax, label)


def _figure1(frame: pd.DataFrame, root: Path, target: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    stress = build_invariance_curve_frame(frame)
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.8), constrained_layout=True); _panels(fig, axes)
    metrics = [("max_logit_error", "Max logit error"), ("mean_probability_error", "Probability error"), ("prediction_flip_rate", "Flip rate"), ("test_metric_drop", "Test metric drop")]
    for ax, (metric, title) in zip(axes[0], metrics):
        subset = stress[stress.get("metric", pd.Series(index=stress.index)).eq(metric)] if not stress.empty else pd.DataFrame()
        if subset.empty or subset["strength"].dropna().empty:
            _unavailable(ax, "coordinate_stress.csv required")
            rows.append({"panel": title, "available": False, "unavailable_reason": "coordinate stress table missing"}); continue
        for model, item in subset.groupby("model"):
            item = item.sort_values("strength"); ax.plot(item["strength"], item["mean"], color=model_color(model), lw=0.7, alpha=.55)
            summary = item.groupby("strength", as_index=False)["mean"].mean(); ax.plot(summary["strength"], summary["mean"], color=model_color(model), lw=1.5, label=model_display_name(model))
            rows.extend(item.assign(panel=title, available=True).to_dict("records"))
        if metric == "max_logit_error": ax.set_yscale("log")
        finalize_axes(ax, "Strength", title, title)
    high = stress[stress.get("transform_strength", pd.Series(index=stress.index)).ge(1.5)] if not stress.empty else pd.DataFrame()
    for ax, (metric, title) in zip(axes[1], [("max_logit_error", "High-strength error"), ("mean_probability_error", "High-strength probability"), ("prediction_flip_rate", "High-strength flips"), ("test_metric_drop", "High-strength metric")]):
        item = high[high.get("metric", pd.Series(index=high.index)).eq(metric)] if not high.empty else pd.DataFrame()
        if item.empty: _unavailable(ax, "no strength >= 1.5 rows")
        else:
            item.boxplot(column="mean", by="model", ax=ax, grid=False, rot=35); ax.set_title(title, loc="left"); ax.set_xlabel(""); ax.set_ylabel("value")
        finalize_axes(ax)
    for ax in axes.flat: ax.get_legend().remove() if ax.get_legend() else None
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(.5, 1.03))
    return {"files": _save(fig, root / "main" / "figure1_coordinate_stress"), "data": rows, "question": "Are coordinate perturbations numerically stable across strength and transform family?"}


def _figure2(frame: pd.DataFrame, root: Path, target: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    phase = build_phase_diagram_frame(frame, target); fig, axes = plt.subplots(3, 4, figsize=(7.25, 5.6), constrained_layout=True); _panels(fig, axes)
    rows = phase.copy(); panels = [("no_transport", "mean", "ATN − no transport"), ("free_transition", "mean", "ATN − free transition"), ("original", "mean", "ATN − original"), ("no_transport", "boundary_gain", "Boundary gain"), ("no_transport", "routing_opportunity", "Routing opportunity"), ("no_transport", "irreducible_transport_risk", "Irreducible risk"), ("no_transport", "mean", "Gain regime"), ("free_transition", "mean", "Transition regime"), ("original", "mean", "Original regime"), ("no_transport", "routing_opportunity", "Opportunity"), ("no_transport", "irreducible_transport_risk", "Risk"), ("no_transport", "mean", "Annotated gain")]
    for ax, (contrast, value, title) in zip(axes.flat, panels):
        item = phase[(phase.get("contrast", pd.Series(index=phase.index)).eq(contrast)) & phase.get(value, pd.Series(index=phase.index)).notna()] if not phase.empty else pd.DataFrame()
        if item.empty: _unavailable(ax, "phase conditions or paired contrast missing"); continue
        scatter = ax.scatter(item["cross_chart_edge_fraction"], item["overlap"], c=item[value], cmap="RdBu_r" if value in {"mean", "boundary_gain"} else "viridis", s=22, edgecolor="white", linewidth=.25)
        for _, row in item.iterrows():
            if len(item) <= 40: ax.text(row["cross_chart_edge_fraction"], row["overlap"], f"{row[value]:+.2f}" if value in {"mean", "boundary_gain"} else f"{row[value]:.2f}", fontsize=4, ha="center", va="center")
        finalize_axes(ax, "Cross-chart fraction", "Overlap", title)
    return {"files": _save(fig, root / "main" / "figure2_regime_atlas"), "data": rows.to_dict("records"), "question": "Where does representability-guided routing have opportunity and irreducible risk?"}


def _figure3(frame: pd.DataFrame, root: Path, target: str, upper: pd.DataFrame | None) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.5), constrained_layout=True); _panels(fig, axes); rows = []
    controls = frame[frame.get("certified_routing_mode", pd.Series(index=frame.index)).notna()].copy() if not frame.empty else pd.DataFrame()
    if controls.empty: controls = frame[frame.get("model_family", pd.Series(index=frame.index)).astype(str).eq("graphatlas_certified")].copy() if not frame.empty else pd.DataFrame()
    modes = sorted(controls.get("certified_routing_mode", pd.Series(dtype=str)).dropna().astype(str).unique()) if not controls.empty else []
    for ax, mode in zip(axes.flat[:6], modes[:6]):
        item = controls[controls["certified_routing_mode"].astype(str).eq(mode)]
        paired = item.groupby("dataset", as_index=False)["test_metric"].mean()
        ax.scatter(np.arange(len(paired)), paired["test_metric"], color=model_color(target), s=18); ax.set_xticks(range(len(paired)), paired["dataset"], rotation=55, ha="right"); finalize_axes(ax, "Dataset", "Test metric", mode); rows.extend(item.assign(panel=mode).to_dict("records"))
    for ax, field, title in zip(axes.flat[6:], ("routing_entropy_mean", "q_route_spread_mean"), ("Routing entropy", "Q route spread")):
        if field not in controls or controls[field].dropna().empty: _unavailable(ax, f"{field} missing")
        else: controls.boxplot(column=field, by="certified_routing_mode", ax=ax, grid=False, rot=35); ax.set_title(title, loc="left"); ax.set_xlabel("")
        finalize_axes(ax)
    if upper is not None and not upper.empty: rows.extend(upper.assign(panel="test_selected_upper_bound").to_dict("records"))
    return {"files": _save(fig, root / "main" / "figure3_routing_causality"), "data": rows, "question": "Does the certificate change routing beyond membership-only controls?"}


def _figure4(frame: pd.DataFrame, root: Path, target: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    real = build_real_benchmark_frame(frame); fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.45), constrained_layout=True); _panels(fig, axes); rows = real.to_dict("records")
    if real.empty:
        for ax in axes.flat: _unavailable(ax, "formal real benchmark rows missing")
        return {"files": _save(fig, root / "main" / "figure4_real_benchmarks"), "data": rows, "question": "How stable is ATN across real datasets?"}
    pivot = real.pivot_table(index="dataset", columns="model", values="mean")
    axes[0, 0].imshow(pivot, aspect="auto", cmap="YlGnBu"); axes[0, 0].set_xticks(range(len(pivot.columns)), [model_display_name(x) for x in pivot.columns], rotation=45, ha="right"); axes[0, 0].set_yticks(range(len(pivot.index)), pivot.index); finalize_axes(axes[0, 0], title="Metric by dataset")
    base = real[real["model"].eq("graphatlas_original")][["dataset", "mean"]].rename(columns={"mean": "baseline"}); atn = real[real["model"].eq(target)][["dataset", "mean"]].rename(columns={"mean": "atn"}); gain = atn.merge(base, on="dataset"); axes[0, 1].scatter(gain["baseline"], gain["atn"] - gain["baseline"], color=model_color(target)); finalize_axes(axes[0, 1], "Original metric", "ATN gain", "Paired gain")
    rank = build_rank_summary_frame(frame); axes[0, 2].scatter(rank["mean"], np.arange(len(rank)), color="#475569") if not rank.empty else _unavailable(axes[0, 2], "rank rows missing"); finalize_axes(axes[0, 2], "Mean rank", title="Mean rank")
    wins = paired_effect(frame, target, "graphatlas_original"); axes[0, 3].hist(wins["delta"], bins=12, color=model_color(target), alpha=.8) if not wins.empty else _unavailable(axes[0, 3], "paired original rows missing"); finalize_axes(axes[0, 3], "ATN − original", "Pairs", "Seed paired deltas")
    for ax, field, title in zip(axes[1], ("boundary_accuracy", "test_metric", "std", "dataset_z"), ("Boundary accuracy", "Metric", "Seed variability", "Within-dataset standardized metric")):
        if field not in real or real[field].dropna().empty: _unavailable(ax, f"{field} missing")
        elif field == "std": ax.scatter(real["std"].fillna(0), real["mean"], c=[model_color(v) for v in real["model"]], s=15)
        else: ax.scatter(np.arange(len(real)), real[field], c=[model_color(v) for v in real["model"]], s=15); ax.set_xticks([])
        finalize_axes(ax, title=title)
    return {"files": _save(fig, root / "main" / "figure4_real_benchmarks"), "data": rows, "question": "Does ATN improve and remain stable across real datasets?"}


def _figure5(frame: pd.DataFrame, root: Path, cases_csv: str | Path | None) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(7.25, 3.7), constrained_layout=True); _panels(fig, axes)
    if cases_csv and Path(cases_csv).exists():
        cases = pd.read_csv(cases_csv); fields = ["rank", "node_id", "probability", "node_transport_risk", "node_q_spread", "node_cross_chart_mass"]
        for ax, field in zip(axes.flat, fields * 2):
            if field not in cases: _unavailable(ax, f"{field} missing")
            else: cases[field].astype(str).value_counts().head(12).plot(kind="bar", ax=ax, color="#176D8A"); ax.tick_params(axis="x", labelrotation=70)
        data = cases.to_dict("records")
    else:
        for ax in axes.flat: _unavailable(ax, "case_index.csv not provided")
        data = []
    return {"files": _save(fig, root / "main" / "figure5_local_case_atlas"), "data": data, "question": "Which local boundary cases are rescued or failed?"}


def _figure6(frame: pd.DataFrame, root: Path, target: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    pareto = build_runtime_pareto_frame(frame); fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.45), constrained_layout=True); _panels(fig, axes); rows = pareto.to_dict("records")
    fields = [("num_layers", "Depth"), ("num_nodes", "Nodes"), ("num_edges", "Edges"), ("membership_topk", "Top-k")]
    for ax, (field, title) in zip(axes[0], fields):
        if field not in pareto or pareto[field].dropna().empty: _unavailable(ax, f"{field} missing")
        else: ax.scatter(pareto[field], pareto["test_metric"], c=[model_color(v) for v in pareto["model"]], s=16); finalize_axes(ax, field, "Test metric", title)
    for ax, (x, y, title) in zip(axes[1], (("num_nodes", "runtime_seconds", "Runtime scaling"), ("num_edges", "peak_cuda_memory_mb", "Memory scaling"), ("runtime_seconds", "test_metric", "Performance-cost"), ("runtime_seconds", "test_metric", "Pareto view"))):
        if x not in pareto or y not in pareto: _unavailable(ax, f"{x}/{y} missing")
        else: ax.scatter(pareto[x].clip(lower=1e-9), pareto[y], c=[model_color(v) for v in pareto["model"]], s=16); ax.set_xscale("log") if x != "runtime_seconds" or x in pareto else None; finalize_axes(ax, x, y, title)
    return {"files": _save(fig, root / "main" / "figure6_scaling_dynamics"), "data": rows, "question": "What is the performance-cost scaling envelope?"}


def build_dense_paper_figures(results: str | Path | pd.DataFrame, output_dir: str | Path = "outputs/reports/paper_dense", *, target_model: str = "graphatlas_c_oracle", stress_csv: str | Path | None = None, cases_csv: str | Path | None = None, upper_bound_results: str | Path | None = None, include_upper_bound: bool = True, bootstrap_reps: int = 10000, bootstrap_seed: int = 2026, formats: Iterable[str] = EXPORT_FORMATS, dpi: int = 450, figures: Iterable[str] | None = None) -> dict[str, Any]:
    """Build all dense figures; missing phases become explicit unavailable panels."""
    global EXPORT_FORMATS, EXPORT_DPI
    EXPORT_FORMATS = tuple(format_name.lstrip(".") for format_name in formats)
    EXPORT_DPI = int(dpi)
    configure_nature_style(); root = Path(output_dir); (root / "main").mkdir(parents=True, exist_ok=True); (root / "supplementary").mkdir(parents=True, exist_ok=True); (root / "data").mkdir(parents=True, exist_ok=True); (root / "manifests").mkdir(parents=True, exist_ok=True)
    all_rows = load_results_frame(results, include_upper_bound=True)
    frame = all_rows[all_rows["is_formal_result"].astype(bool)].copy() if "is_formal_result" in all_rows else all_rows.copy()
    excluded_test_selected_rows = int(len(all_rows) - len(frame))
    available_models = set(frame.get("model", pd.Series(dtype=str)).astype(str))
    # The formal q-reference model is the focal model.  Do not silently
    # replace it with the learned certificate model when its rows are absent.
    target = target_model if target_model == "graphatlas_c_oracle" or target_model in available_models else resolve_target_model(frame, target_model)
    upper = load_results_frame(upper_bound_results, include_upper_bound=True) if upper_bound_results else pd.DataFrame()
    if not upper.empty and "is_formal_result" in upper:
        upper = upper[~upper["is_formal_result"].astype(bool)].copy()
    if include_upper_bound and upper_bound_results is None and "is_formal_result" in all_rows:
        upper = all_rows[~all_rows["is_formal_result"]].copy()
    builders = [("figure1_coordinate_stress", lambda: _figure1(pd.read_csv(stress_csv) if stress_csv and Path(stress_csv).exists() else frame, root, target)), ("figure2_regime_atlas", lambda: _figure2(frame, root, target)), ("figure3_routing_causality", lambda: _figure3(frame, root, target, upper)), ("figure4_real_benchmarks", lambda: _figure4(frame, root, target)), ("figure5_local_case_atlas", lambda: _figure5(frame, root, cases_csv)), ("figure6_scaling_dynamics", lambda: _figure6(frame, root, target))]
    if figures:
        wanted = set(figures); builders = [item for item in builders if item[0] in wanted]
    manifest = {"target_model": target, "formal_rows": int(len(frame)), "excluded_test_selected_rows": excluded_test_selected_rows, "legacy_rows_missing_metadata": int(all_rows.get("legacy_rows_missing_metadata", pd.Series(dtype=bool)).sum()), "figures": [], "bootstrap_resamples": int(bootstrap_reps), "bootstrap_seed": int(bootstrap_seed), "outputs": ", ".join(EXPORT_FORMATS), "dpi": EXPORT_DPI}
    report = {"generated": [], "built_figures": [], "partially_built_figures": [], "failed": [], "failed_figures": [], "unavailable_panels": [], "warnings": [], "formal_source": str(results), "upper_bound_source": str(upper_bound_results) if upper_bound_results else None, "formal_rows": int(len(frame)), "excluded_test_selected_rows": excluded_test_selected_rows}
    for name, builder in builders:
        try:
            result = builder(); data_frame = pd.DataFrame(result.pop("data", [])); data_path = _write(data_frame, root / "data" / f"{name}.csv"); result["data"] = data_path; result["name"] = name
            panel_records = [{"figure_name": name, "panel_id": label, "scientific_question": result.get("question", ""), "input_files": [str(results)], "input_models": sorted(frame.get("model", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()), "filter_expression": "formal_result=True", "pairing_keys": ["dataset", "condition_id", "seed", "split"], "metric_name": "test_metric", "metric_direction": "higher", "normalization": "none", "bootstrap_reps": int(bootstrap_reps), "bootstrap_seed": int(bootstrap_seed), "color_limits": None, "axis_scale": "linear", "plot_epsilon": 1e-12, "output_files": result.get("files", []), "input_row_count": int(len(frame)), "used_row_count": int(len(data_frame)), "missing_reason": None} for label in "abcdefghijklm"[:6]]
            manifest["figures"].append({"name": name, **result, "panels": panel_records}); report["generated"].append(name); report["built_figures"].append(name)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"; manifest["figures"].append({"name": name, "generated": False, "unavailable_reason": reason}); report["failed"].append({"name": name, "reason": reason}); report["failed_figures"].append({"name": name, "reason": reason})
    (root / "manifests" / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "manifests" / "figure_build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
