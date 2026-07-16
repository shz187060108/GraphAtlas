"""Validated, plotting-oriented table builders for Atlas Transport Network figures."""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_results_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.read_csv(results)
    required = {"dataset", "model", "test_metric"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Results table is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    for column in frame.columns:
        if column.endswith(("_metric", "_accuracy", "_error", "_mean", "_std")) or column in {
            "transportability_beta", "runtime_seconds", "parameters", "num_nodes", "num_edges", "seed", "split",
        }:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "condition_id" not in frame:
        frame["condition_id"] = ""
    return frame


def _summary(frame: pd.DataFrame, groups: list[str], value: str = "test_metric") -> pd.DataFrame:
    if frame.empty or value not in frame:
        return pd.DataFrame(columns=[*groups, "mean", "std", "n", "ci95"])
    out = frame.groupby(groups, dropna=False)[value].agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"count": "n"})
    out["ci95"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["n"].clip(lower=1))
    return out


def _condition_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name in ("overlap", "cross_chart_edge_fraction"):
        if name not in out:
            out[name] = np.nan
    parsed = out["condition_id"].astype(str).str.extract(r"ov(?P<overlap>\d+(?:\.\d+)?)_cross(?P<cross>\d+(?:\.\d+)?)")
    out["overlap"] = out["overlap"].fillna(pd.to_numeric(parsed["overlap"], errors="coerce"))
    out["cross_chart_edge_fraction"] = out["cross_chart_edge_fraction"].fillna(pd.to_numeric(parsed["cross"], errors="coerce"))
    return out


def paired_effect(frame: pd.DataFrame, target_model: str, baseline: str, metric: str = "test_metric") -> pd.DataFrame:
    keys = [column for column in ("dataset", "condition_id", "task", "seed", "split") if column in frame]
    if metric not in frame or not keys:
        return pd.DataFrame(columns=[*keys, "target", "baseline", "delta"])
    target = frame.loc[frame["model"].eq(target_model), keys + [metric]].rename(columns={metric: "target"})
    base = frame.loc[frame["model"].eq(baseline), keys + [metric]].rename(columns={metric: "baseline"})
    joined = target.merge(base, on=keys, how="inner").dropna(subset=["target", "baseline"])
    joined["delta"] = joined["target"] - joined["baseline"]
    return joined


def bootstrap_paired_effect(paired: pd.DataFrame, *, resamples: int = 5000, seed: int = 2026) -> dict[str, float]:
    """Deterministic seed-level bootstrap interval for a paired model contrast."""
    values = paired.get("delta", pd.Series(dtype=float)).dropna().to_numpy(dtype=float)
    if not len(values):
        return {"mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return {"mean_delta": float(values.mean()), "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)), "n": int(len(values))}


def build_phase_diagram_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c") -> pd.DataFrame:
    frame = _condition_coordinates(load_results_frame(results))
    rows = []
    for baseline, label in (("graphatlas_no_transport", "no_transport"), ("graphatlas_free_transition", "free_transition"), ("graphatlas_original", "original")):
        paired = paired_effect(frame, target_model, baseline)
        if paired.empty:
            continue
        paired = _condition_coordinates(paired)
        aggregate = _summary(paired, ["condition_id", "overlap", "cross_chart_edge_fraction"], "delta")
        aggregate["contrast"] = label
        boundary = paired_effect(frame, target_model, baseline, "boundary_accuracy")
        if not boundary.empty:
            boundary = _condition_coordinates(boundary)
            boundary = _summary(boundary, ["condition_id", "overlap", "cross_chart_edge_fraction"], "delta").rename(columns={"mean": "boundary_gain"})
            aggregate = aggregate.merge(boundary[["condition_id", "boundary_gain"]], on="condition_id", how="left")
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_beta_sweep_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c") -> pd.DataFrame:
    frame = load_results_frame(results)
    beta = frame.loc[frame["model"].eq(target_model) | frame.get("model_family", pd.Series(index=frame.index, dtype=str)).eq("graphatlas_certified")].copy()
    if beta.empty:
        return pd.DataFrame()
    beta["beta"] = pd.to_numeric(beta.get("transportability_beta"), errors="coerce")
    columns = ["test_metric", "boundary_accuracy", "routing_entropy_mean", "q_route_spread_mean", "distortion_mean"]
    rows = []
    for metric in columns:
        if metric in beta:
            item = _summary(beta, ["beta"], metric)
            item["metric"] = metric
            rows.append(item)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_counterfactual_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c") -> pd.DataFrame:
    frame = load_results_frame(results)
    family = frame["model_family"] if "model_family" in frame else frame["model"]
    controls = frame[family.eq("graphatlas_certified")].copy()
    if controls.empty:
        return pd.DataFrame()
    controls["routing_mode"] = controls.get("certified_routing_mode", "certificate").fillna("certificate")
    return _summary(controls, ["routing_mode", "condition_id"], "test_metric")


def build_invariance_curve_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    metrics = [column for column in ("invariance_error_affine", "invariance_error_nonlinear", "intervention_flip_rate_nonlinear") if column in frame]
    rows = []
    for metric in metrics:
        item = _summary(frame, ["model"], metric)
        item["metric"] = metric
        item["transform_family"] = "affine" if metric.endswith("affine") else "nonlinear"
        item["strength"] = 1.0
        rows.append(item)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_transport_diagnostics_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    columns = [column for column in ("transportability_mean", "distortion_mean", "q_route_spread_mean", "routing_entropy_mean", "chart_condition_number", "membership_entropy_mean") if column in frame]
    return frame[[column for column in ["dataset", "condition_id", "model", "seed", *columns] if column in frame]].copy()


def build_real_benchmark_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    return _summary(frame[~frame["dataset"].astype(str).str.startswith("atlas_het")], ["dataset", "model"], "test_metric")


def build_hetgb_feature_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    subset = frame[frame["dataset"].astype(str).str.startswith("hetgb")].copy()
    if "feature_source" not in subset:
        subset["feature_source"] = subset.get("condition_id", "unknown")
    return _summary(subset, ["dataset", "feature_source", "model"], "test_metric")


def build_link_prediction_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    task = frame["task"] if "task" in frame else pd.Series("", index=frame.index)
    if "link_decoder" not in frame:
        frame = frame.copy()
        frame["link_decoder"] = "dot"
    return _summary(frame[task.eq("link_prediction")], ["dataset", "link_decoder", "model"], "test_metric")


def build_runtime_pareto_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    columns = [column for column in ("dataset", "model", "test_metric", "runtime_seconds", "parameters", "peak_cuda_memory_bytes", "num_nodes") if column in frame]
    return frame[columns].dropna(subset=["test_metric", "runtime_seconds"]).copy()


def build_opportunity_risk_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c") -> pd.DataFrame:
    frame = load_results_frame(results)
    subset = frame[frame["model"].eq(target_model)].copy()
    return subset[[column for column in ("dataset", "condition_id", "test_metric", "routing_opportunity", "irreducible_transport_risk", "boundary_accuracy") if column in subset]]


def build_rank_summary_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    keys = [column for column in ("dataset", "task", "seed", "split") if column in frame]
    if not keys:
        return pd.DataFrame()
    ranks = frame.groupby(keys + ["model"], as_index=False)["test_metric"].mean()
    ranks["rank"] = ranks.groupby(keys)["test_metric"].rank(ascending=False, method="average")
    return _summary(ranks, ["model"], "rank")


def export_plot_data(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
