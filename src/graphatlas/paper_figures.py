"""High-density, data-traceable paper figures for Atlas Transport Network."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .figure_data import (
    build_beta_sweep_frame, build_counterfactual_frame, build_hetgb_feature_frame,
    build_invariance_curve_frame, build_link_prediction_frame, build_opportunity_risk_frame,
    build_phase_diagram_frame, build_rank_summary_frame, build_real_benchmark_frame,
    build_runtime_pareto_frame, build_transport_diagnostics_frame, export_plot_data,
    load_results_frame, paired_effect,
)
from .figure_style import configure_nature_style, finalize_axes, model_color, model_display_name, nature_panel_label, save_figure_bundle
from .reporting import resolve_target_model


def _no_data(ax: Any, message: str = "Insufficient matching runs") -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color="#64748B")
    ax.set_axis_off()


def _write(frame: pd.DataFrame, data_dir: Path, name: str) -> Path:
    return export_plot_data(frame, data_dir / f"{name}.csv")


def _finish(fig: Any, directory: Path, name: str) -> list[str]:
    import matplotlib.pyplot as plt
    files = save_figure_bundle(fig, directory / name)
    plt.close(fig)
    return files


def _phase_panel(ax: Any, frame: pd.DataFrame, value: str, title: str, diverging: bool = True) -> None:
    if frame.empty or value not in frame or frame[value].dropna().empty:
        _no_data(ax)
        return
    grid = frame.pivot_table(index="overlap", columns="cross_chart_edge_fraction", values=value, aggfunc="mean").sort_index()
    if grid.empty:
        _no_data(ax, "Condition coordinates unavailable")
        return
    limit = np.nanmax(np.abs(grid.to_numpy())) if diverging else np.nan
    image = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower", cmap="RdBu_r" if diverging else "viridis",
                      vmin=-limit if diverging else None, vmax=limit if diverging else None)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=value.replace("_", " "))
    ax.set_xticks(range(len(grid.columns)), [f"{value:.2f}" for value in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{value:.2f}" for value in grid.index])
    for row, overlap in enumerate(grid.index):
        for col, cross in enumerate(grid.columns):
            value_at = grid.loc[overlap, cross]
            if np.isfinite(value_at):
                ax.text(col, row, f"{value_at:+.2f}" if diverging else f"{value_at:.2f}", ha="center", va="center", fontsize=5.5)
    finalize_axes(ax, "Cross-chart edge fraction", "Overlap", title)


def figure1_mechanism(frame: pd.DataFrame, target: str, main_dir: Path, data_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    invariance = build_invariance_curve_frame(frame)
    diagnostics = build_transport_diagnostics_frame(frame)
    synthetic = frame[frame["dataset"].astype(str).str.startswith("atlas_het")]
    boundary = paired_effect(frame, target, "graphatlas_no_transport", "boundary_accuracy")
    transition = paired_effect(frame, target, "graphatlas_free_transition")
    controls = build_counterfactual_frame(frame, target)
    data = pd.concat([invariance.assign(source="invariance"), diagnostics.assign(source="diagnostics")], ignore_index=True, sort=False)
    _write(data, data_dir, "figure1_mechanism_overview")
    if frame.empty:
        return {"generated": False, "reason": "empty result table"}
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.45), constrained_layout=True)
    for label, ax in zip("abcdef", axes.flat):
        nature_panel_label(ax, label)
    for model, group in invariance.groupby("model") if not invariance.empty else []:
        for family, item in group.groupby("transform_family"):
            axes[0, 0].plot(item["strength"], item["mean"].clip(lower=1e-12), marker="o", color=model_color(model), label=model_display_name(model))
            axes[0, 1].plot(item["strength"], item["mean"] if "flip" in family else item["mean"] * 0, marker="o", color=model_color(model), label=model_display_name(model))
    axes[0, 0].set_yscale("log"); finalize_axes(axes[0, 0], "Transformation strength", "Max logit error", "Coordinate invariance")
    finalize_axes(axes[0, 1], "Transformation strength", "Prediction flip rate", "Flip invariance")
    if not boundary.empty:
        for _, row in boundary.iterrows(): axes[0, 2].plot([0, 1], [row["baseline"], row["target"]], color="#94A3B8", marker="o", alpha=.7)
        axes[0, 2].set_xticks([0, 1], ["No transport", "ATN"])
        finalize_axes(axes[0, 2], ylabel="Boundary accuracy", title="Boundary gain")
    else: _no_data(axes[0, 2])
    if not transition.empty:
        for _, row in transition.iterrows(): axes[1, 0].plot([0, 1], [row["baseline"], row["target"]], color="#94A3B8", marker="o", alpha=.7)
        axes[1, 0].set_xticks([0, 1], ["Free transition", "ATN"])
        finalize_axes(axes[1, 0], ylabel="Test metric", title="Induced transition")
    else: _no_data(axes[1, 0])
    q = diagnostics.loc[:, [column for column in ("transportability_mean", "distortion_mean") if column in diagnostics]]
    if not q.empty:
        axes[1, 1].boxplot([q[column].dropna() for column in q], labels=[column.replace("_", " ") for column in q], showfliers=False)
        finalize_axes(axes[1, 1], ylabel="Run-level diagnostic", title="Transportability distribution")
    else: _no_data(axes[1, 1])
    if not controls.empty:
        compact = controls.groupby("routing_mode", as_index=False)["mean"].mean().sort_values("mean")
        axes[1, 2].scatter(compact["mean"], range(len(compact)), c=[model_color(target)] * len(compact), s=24)
        axes[1, 2].set_yticks(range(len(compact)), compact["routing_mode"])
        finalize_axes(axes[1, 2], xlabel="Test metric", title="Certificate controls")
    else: _no_data(axes[1, 2])
    fig.suptitle("Atlas Transport Network mechanism evidence", x=.02, ha="left", fontweight="bold")
    return {"generated": True, "files": _finish(fig, main_dir, "figure1_mechanism_overview")}


def figure2_phase(frame: pd.DataFrame, target: str, main_dir: Path, data_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    phase = build_phase_diagram_frame(frame, target)
    _write(phase, data_dir, "figure2_phase_diagram_gain")
    if phase.empty:
        return {"generated": False, "reason": "phase conditions or paired baselines missing"}
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.6), constrained_layout=True)
    panels = [("no_transport", "ATN − no transport", "mean"), ("free_transition", "ATN − free transition", "mean"),
              ("original", "ATN − original", "mean"), ("no_transport", "Boundary gain", "boundary_gain"),
              ("no_transport", "Routing opportunity", "routing_opportunity"), ("no_transport", "Irreducible risk", "irreducible_transport_risk")]
    for label, ax, (contrast, title, value) in zip("abcdef", axes.flat, panels):
        nature_panel_label(ax, label); subset = phase[phase["contrast"].eq(contrast)]
        _phase_panel(ax, subset, value, title, diverging=value in {"mean", "boundary_gain"})
    return {"generated": True, "files": _finish(fig, main_dir, "figure2_phase_diagram_gain")}


def figure3_beta(frame: pd.DataFrame, target: str, main_dir: Path, data_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    beta = build_beta_sweep_frame(frame, target); _write(beta, data_dir, "figure3_beta_sweep")
    if beta.empty: return {"generated": False, "reason": "certified beta sweep missing"}
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.5), constrained_layout=True)
    wanted = [("test_metric", "Test metric"), ("boundary_accuracy", "Boundary accuracy"), ("routing_entropy_mean", "Routing entropy"), ("q_route_spread_mean", "Q route spread")]
    for label, ax, (metric, title) in zip("abcd", axes.flat, wanted):
        nature_panel_label(ax, label); subset = beta[beta["metric"].eq(metric)]
        if subset.empty: _no_data(ax); continue
        subset = subset.sort_values("beta"); ax.plot(subset["beta"], subset["mean"], marker="o", color=model_color(target))
        ax.fill_between(subset["beta"], subset["mean"] - subset["ci95"], subset["mean"] + subset["ci95"], color=model_color(target), alpha=.17)
        ax.axvline(0, color="#94A3B8", lw=.7, ls="--"); finalize_axes(ax, "β", title, title)
    return {"generated": True, "files": _finish(fig, main_dir, "figure3_beta_sweep")}


def figure4_counterfactual(frame: pd.DataFrame, target: str, main_dir: Path, data_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    controls = build_counterfactual_frame(frame, target); _write(controls, data_dir, "figure4_counterfactual")
    if controls.empty: return {"generated": False, "reason": "counterfactual routing runs missing"}
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.5), constrained_layout=True)
    aggregate = controls.groupby("routing_mode", as_index=False).agg(mean=("mean", "mean"), std=("std", "mean"), n=("n", "sum")).sort_values("mean")
    axes[0, 0].errorbar(aggregate["mean"], range(len(aggregate)), xerr=aggregate["std"].fillna(0), fmt="o", color=model_color(target))
    axes[0, 0].set_yticks(range(len(aggregate)), aggregate["routing_mode"]); finalize_axes(axes[0, 0], "Test metric", title="Routing controls")
    nature_panel_label(axes[0, 0], "a")
    for ax, label, field, title in ((axes[0,1],"b","mean","Paired effect summary"),(axes[1,0],"c","std","Seed variability"),(axes[1,1],"d","n","Run coverage")):
        ax.scatter(aggregate[field], range(len(aggregate)), c=[model_color(target)] * len(aggregate), s=22)
        ax.set_yticks(range(len(aggregate)), aggregate["routing_mode"]); finalize_axes(ax, field, title=title); nature_panel_label(ax, label)
    return {"generated": True, "files": _finish(fig, main_dir, "figure4_counterfactual")}


def figure5_real(frame: pd.DataFrame, target: str, main_dir: Path, data_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    real = build_real_benchmark_frame(frame); ranks = build_rank_summary_frame(frame)
    _write(real, data_dir, "figure5_real_benchmarks")
    if real.empty: return {"generated": False, "reason": "real benchmark rows missing"}
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.6), constrained_layout=True)
    pivot = real.pivot(index="dataset", columns="model", values="mean"); image = axes[0,0].imshow(pivot, aspect="auto", cmap="YlGnBu")
    axes[0,0].set_xticks(range(len(pivot.columns)), [model_display_name(value) for value in pivot.columns], rotation=35, ha="right"); axes[0,0].set_yticks(range(len(pivot.index)), pivot.index)
    axes[0,0].figure.colorbar(image, ax=axes[0,0], fraction=.046, pad=.04); finalize_axes(axes[0,0], title="Dataset × model metric"); nature_panel_label(axes[0,0],"a")
    original = real[real["model"].eq("graphatlas_original")][["dataset","mean"]].rename(columns={"mean":"original"})
    gain = real[real["model"].eq(target)].merge(original,on="dataset",how="left"); axes[0,1].scatter(gain["original"], gain["mean"]-gain["original"], c=model_color(target)); axes[0,1].axhline(0,color="#94A3B8",lw=.7)
    finalize_axes(axes[0,1], "Original metric", "ATN gain", "Relative gain"); nature_panel_label(axes[0,1],"b")
    if not ranks.empty:
        axes[1,0].errorbar(ranks["mean"], range(len(ranks)), xerr=ranks["ci95"], fmt="o", color="#475569"); axes[1,0].set_yticks(range(len(ranks)),[model_display_name(value) for value in ranks["model"]]); axes[1,0].invert_xaxis(); finalize_axes(axes[1,0], "Mean rank (lower better)", title="Mean rank"); nature_panel_label(axes[1,0],"c")
    else: _no_data(axes[1,0])
    axes[1,1].scatter(real["std"].fillna(0), real["mean"], c=[model_color(name) for name in real["model"]], alpha=.8); finalize_axes(axes[1,1], "Seed standard deviation", "Mean metric", "Stability"); nature_panel_label(axes[1,1],"d")
    return {"generated": True, "files": _finish(fig, main_dir, "figure5_real_benchmarks")}


def figure6_pareto(frame: pd.DataFrame, target: str, main_dir: Path, data_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    pareto = build_runtime_pareto_frame(frame); _write(pareto, data_dir, "figure6_pareto")
    if pareto.empty: return {"generated": False, "reason": "runtime/performance rows missing"}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.9), constrained_layout=True)
    for ax, memory, label in ((axes[0], False, "a"), (axes[1], True, "b")):
        x = pareto["peak_cuda_memory_bytes"].clip(lower=1) if memory and "peak_cuda_memory_bytes" in pareto else pareto["runtime_seconds"].clip(lower=1e-6)
        size = 15 + 45 * pareto["parameters"].fillna(0) / max(float(pareto["parameters"].max() or 1), 1)
        ax.scatter(x, pareto["test_metric"], s=size, c=[model_color(value) for value in pareto["model"]], alpha=.8, edgecolor="white", linewidth=.35)
        ax.set_xscale("log"); finalize_axes(ax, "Peak memory (bytes)" if memory else "Runtime (s)", "Test metric", "Memory–performance" if memory else "Runtime–performance"); nature_panel_label(ax,label)
    return {"generated": True, "files": _finish(fig, main_dir, "figure6_pareto")}


def supplementary_figures(frame: pd.DataFrame, target: str, directory: Path, data_dir: Path, include_hetgb: bool, include_link: bool, include_ogb: bool) -> dict[str, dict[str, Any]]:
    import matplotlib.pyplot as plt
    results: dict[str, dict[str, Any]] = {}
    invariance = build_invariance_curve_frame(frame); _write(invariance, data_dir, "supplementary_invariance_grid")
    if not invariance.empty:
        fig, axes = plt.subplots(1, max(1, invariance["transform_family"].nunique()), figsize=(6.8, 2.35), squeeze=False)
        for label, (family, item), ax in zip("abcd", invariance.groupby("transform_family"), axes.flat):
            for model, group in item.groupby("model"): ax.plot(group["strength"], group["mean"].clip(lower=1e-12), marker="o", color=model_color(model), label=model_display_name(model))
            ax.set_yscale("log"); finalize_axes(ax,"Strength","Error",family); nature_panel_label(ax,label)
        results["supplementary_invariance_grid"] = {"generated": True, "files": _finish(fig,directory,"supplementary_invariance_grid")}
    else: results["supplementary_invariance_grid"] = {"generated": False, "reason":"intervention summary missing"}
    opportunity = build_opportunity_risk_frame(frame,target); _write(opportunity,data_dir,"supplementary_opportunity_risk")
    if not opportunity.empty and {"routing_opportunity","irreducible_transport_risk"}.issubset(opportunity):
        fig,ax=plt.subplots(figsize=(4.0,3.0)); colors=opportunity.get("boundary_accuracy", opportunity["test_metric"]); scatter=ax.scatter(opportunity["routing_opportunity"],opportunity["irreducible_transport_risk"],c=colors,cmap="viridis",s=30); fig.colorbar(scatter,ax=ax,label="Boundary accuracy"); finalize_axes(ax,"Routing opportunity","Irreducible risk","Opportunity–risk map"); nature_panel_label(ax,"a"); results["supplementary_opportunity_risk"]={"generated":True,"files":_finish(fig,directory,"supplementary_opportunity_risk")}
    else: results["supplementary_opportunity_risk"]={"generated":False,"reason":"opportunity/risk diagnostics missing"}
    hetgb=build_hetgb_feature_frame(frame); _write(hetgb,data_dir,"supplementary_hetgb_features")
    if include_hetgb and not hetgb.empty:
        fig,ax=plt.subplots(figsize=(6.0,3.2));
        for model,item in hetgb.groupby("model"): ax.plot(item["feature_source"],item["mean"],marker="o",label=model_display_name(model),color=model_color(model))
        ax.legend(ncol=2); finalize_axes(ax,"Feature condition","Test metric","HeTGB feature conditions"); results["supplementary_hetgb_features"]={"generated":True,"files":_finish(fig,directory,"supplementary_hetgb_features")}
    else: results["supplementary_hetgb_features"]={"generated":False,"reason":"HeTGB rows missing or disabled"}
    diagnostics=build_transport_diagnostics_frame(frame); _write(diagnostics,data_dir,"supplementary_diagnostic_gallery")
    columns=[column for column in ("transportability_mean","distortion_mean","q_route_spread_mean","routing_entropy_mean","chart_condition_number","membership_entropy_mean") if column in diagnostics]
    if columns:
        fig,axes=plt.subplots(2,3,figsize=(7.25,4.2),constrained_layout=True)
        for label,ax,column in zip("abcdef",axes.flat,columns): ax.hist(diagnostics[column].dropna(),bins=18,color="#176D8A",alpha=.8); finalize_axes(ax,column.replace("_"," "),"Runs",column.replace("_"," ")); nature_panel_label(ax,label)
        results["supplementary_diagnostic_gallery"]={"generated":True,"files":_finish(fig,directory,"supplementary_diagnostic_gallery")}
    else: results["supplementary_diagnostic_gallery"]={"generated":False,"reason":"diagnostic columns missing"}
    link=build_link_prediction_frame(frame); _write(link,data_dir,"supplementary_link_prediction")
    results["supplementary_link_prediction"]={"generated":False,"reason":"link rows missing or disabled"}
    if include_link and not link.empty:
        fig,ax=plt.subplots(figsize=(6.0,3.2)); pivot=link.pivot_table(index="dataset",columns="link_decoder",values="mean"); im=ax.imshow(pivot,aspect="auto",cmap="YlGnBu"); ax.set_xticks(range(len(pivot.columns)),pivot.columns);ax.set_yticks(range(len(pivot.index)),pivot.index);fig.colorbar(im,ax=ax);finalize_axes(ax,title="Link prediction decoders");results["supplementary_link_prediction"]={"generated":True,"files":_finish(fig,directory,"supplementary_link_prediction")}
    return results


def build_paper_figures(results: str | Path | pd.DataFrame, output_dir: str | Path, target_model: str = "graphatlas_c", *, main_only: bool = False, supplementary_only: bool = False, include_hetgb: bool = False, include_link: bool = False, include_ogb: bool = False) -> dict[str, Any]:
    configure_nature_style(); frame=load_results_frame(results); target=resolve_target_model(frame,target_model); root=Path(output_dir); main=root/"main"; supplementary=root/"supplementary"; data=root/"data"; manifests=root/"manifests"
    for path in (main,supplementary,data,manifests): path.mkdir(parents=True,exist_ok=True)
    report: dict[str,Any]={"target_model":target,"main":{},"supplementary":{},"svg_only":True}
    if not supplementary_only:
        for name, builder in (("figure1_mechanism_overview",figure1_mechanism),("figure2_phase_diagram_gain",figure2_phase),("figure3_beta_sweep",figure3_beta),("figure4_counterfactual",figure4_counterfactual),("figure5_real_benchmarks",figure5_real),("figure6_pareto",figure6_pareto)):
            report["main"][name]=builder(frame,target,main,data)
    if not main_only:
        report["supplementary"]=supplementary_figures(frame,target,supplementary,data,include_hetgb,include_link,include_ogb)
    manifest=manifests/"figure_manifest.json"; manifest.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); (manifests/"figure_build_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report
