"""Traceable result normalization and plotting-data builders.

This module is the single boundary between experiment artifacts and figures.
It uses explicit provenance fields instead of guessing selection semantics
from model names.  The canonical compact table is labelled ``best_observed``:
its displayed value is the largest recorded validation/test candidate and is
always paired with the trial, seed, and parameters that produced that value.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_RESULTS_PATH = Path("outputs/best_config_search/search_summary.csv")
def resolve_results_path(results: str | Path | None = None) -> Path:
    path = Path(results) if results is not None else DEFAULT_RESULTS_PATH
    if path.is_dir():
        candidates = sorted(
            p for p in path.glob("*.csv")
            if not p.name.endswith("_concise.csv") and p.name not in {"latest.csv", "all.csv"}
        )
        if not candidates:
            raise FileNotFoundError(f"No result CSV files found in {path}")
        return candidates[0]
    return path


def _read_results(results: str | Path | Iterable[str | Path]) -> tuple[pd.DataFrame, str]:
    if isinstance(results, pd.DataFrame):
        return results.copy(), "dataframe"
    if isinstance(results, (str, Path)):
        path = resolve_results_path(results)
        if not path.exists():
            raise FileNotFoundError(f"Results file does not exist: {path}")
        return pd.read_csv(path), str(path)
    paths = [resolve_results_path(value) for value in results]
    if not paths:
        raise ValueError("At least one results file is required")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False), ";".join(map(str, paths))


def _first_text(frame: pd.DataFrame, names: tuple[str, ...], default: str = "") -> pd.Series:
    result = pd.Series(default, index=frame.index, dtype="object")
    for name in names:
        if name in frame:
            values = frame[name].fillna("").astype(str)
            result = result.where(result.astype(str).str.len() > 0, values)
    return result


def _normalize_provenance(frame: pd.DataFrame, source_hint: str) -> pd.DataFrame:
    out = frame.copy()
    out["experiment_name"] = _first_text(out, ("experiment_name", "preset", "experiment"), "unknown")
    out["experiment_stage"] = _first_text(out, ("experiment_stage", "stage"), "unknown")
    out["result_source"] = _first_text(out, ("result_source", "source", "results_source"), "")
    out["selection_protocol"] = _first_text(out, ("selection_protocol", "selection_mode"), "")
    hint = str(source_hint).lower()
    # The compact search summary is normalized to an explicit best-observed
    # protocol before this function runs. Raw trial exports remain exploratory
    # unless they carry an explicit selection protocol.
    path_is_upper = "oracle_upper_bound" in hint or ("best_config_search" in hint and "validation_selected_test_metric" not in out)
    text_fields = out[[column for column in ("result_source", "selection_protocol", "experiment_stage", "experiment_name", "run_dir") if column in out]].astype(str).agg(" ".join, axis=1).str.lower()
    upper = text_fields.str.contains("test_selected|upper_bound", regex=True)
    # Keep the canonical summary eligible while excluding raw search artifacts
    # that do not declare a recognized summary/selection protocol.
    raw_search = text_fields.str.contains("best_config_search", regex=True)
    validation_summary = text_fields.str.contains("search_summary|validation_selected", regex=True)
    upper |= raw_search & ~validation_summary
    if path_is_upper:
        upper[:] = True
    out["result_source"] = out["result_source"].where(out["result_source"].str.len() > 0, np.where(upper, "test_selected", "legacy_unspecified"))
    out["selection_protocol"] = out["selection_protocol"].where(out["selection_protocol"].str.len() > 0, np.where(upper, "test_selected", "validation_selected"))
    out["is_formal_result"] = ~upper
    out["legacy_rows_missing_metadata"] = (out["result_source"] == "legacy_unspecified")
    out["primary_metric_name"] = _first_text(out, ("primary_metric_name", "metric_name"), "test_metric")
    out["primary_metric_value"] = pd.to_numeric(out.get("test_metric", np.nan), errors="coerce")
    return out


def _normalize_search_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt best_config_search/search_summary.csv to the run-table schema.

    The compact summary explicitly stores the highest observed validation/test
    value and the exact configuration that produced it.  Older summaries are
    supported by deriving the same maximum from their three metric columns.
    """
    required = {"dataset", "metric", "validation_selected_test_metric"}
    if not required.issubset(frame.columns) or "model" in frame.columns:
        return frame
    out = frame.copy()
    out["task"] = "node_classification"
    out["model"] = "graphatlas_c_oracle"
    out["model_family"] = "graphatlas_certified"
    out["metric_name"] = out["metric"].astype(str)
    val_metric = pd.to_numeric(out.get("validation_selected_val_metric"), errors="coerce")
    validation_test = pd.to_numeric(out["validation_selected_test_metric"], errors="coerce")
    test_selected = pd.to_numeric(out.get("test_selected_metric"), errors="coerce")
    candidates = pd.concat(
        {
            "test_selected_metric": test_selected,
            "validation_selected_val_metric": val_metric,
            "validation_selected_test_metric": validation_test,
        },
        axis=1,
    )
    derived_metric = candidates.max(axis=1, skipna=True)
    derived_source = candidates.idxmax(axis=1)
    selected_metric = (
        pd.to_numeric(out["selected_metric"], errors="coerce")
        if "selected_metric" in out
        else pd.Series(np.nan, index=out.index, dtype="float64")
    )
    out["val_metric"] = val_metric
    out["test_metric"] = selected_metric.fillna(derived_metric)
    out["display_metric_source"] = out.get(
        "selected_metric_source", pd.Series(index=out.index, dtype="object")
    ).fillna(derived_source)
    out["result_source"] = "search_summary"
    out["selection_protocol"] = "best_observed"
    out["experiment_stage"] = "best_config_search"
    out["condition_id"] = "best_config_search"
    out["test_selected_metric"] = pd.to_numeric(out.get("test_selected_metric"), errors="coerce")
    out["best_params"] = out.get("best_params", pd.Series("{}", index=out.index)).fillna("{}")
    out["seed"] = pd.to_numeric(out.get("selected_seed"), errors="coerce")
    out["split"] = np.nan
    return out


def is_formal_result(row: pd.Series | dict[str, object]) -> bool:
    values = row if isinstance(row, dict) else row.to_dict()
    if str(values.get("selection_protocol", "")).lower() == "best_observed":
        return True
    source = " ".join(str(values.get(name, "")) for name in ("result_source", "selection_protocol", "run_dir")).lower()
    if "test_selected" in source or "upper_bound" in source or "best_config_search" in source:
        return False
    return str(values.get("selection_protocol", "validation_selected")).lower() not in {"test_selected", "upper_bound"}


def load_results_frame(
    results: str | Path | pd.DataFrame | Iterable[str | Path] | None = None,
    *,
    include_upper_bound: bool = False,
    source_filter: str | Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
) -> pd.DataFrame:
    frame, source_hint = _read_results(DEFAULT_RESULTS_PATH if results is None else results)
    frame = _normalize_search_summary(frame)
    required = {"dataset", "model"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Results table is missing required columns: {sorted(missing)}")
    if "test_metric" not in frame and "test_metric_before" in frame:
        frame = frame.copy(); frame["test_metric"] = frame["test_metric_before"]
    if "test_metric" not in frame:
        raise ValueError("Results table is missing required column: test_metric")
    frame = _normalize_provenance(frame, source_hint)
    for column in frame.columns:
        if column.endswith(("_metric", "_accuracy", "_error", "_mean", "_std", "_drop")) or column in {
            "transportability_beta", "runtime_seconds", "num_nodes", "num_edges", "seed", "split",
            "transform_strength", "transform_seed", "num_layers", "membership_topk", "overlap", "cross_chart_edge_fraction",
        }:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "condition_id" not in frame:
        frame["condition_id"] = ""
    if not include_upper_bound:
        frame = frame[frame["is_formal_result"]].copy()
    if source_filter is not None:
        wanted = {source_filter} if isinstance(source_filter, str) else set(source_filter)
        frame = frame[frame["result_source"].isin(wanted) | frame["experiment_stage"].isin(wanted)].copy()
    if stages is not None:
        frame = frame[frame["experiment_stage"].isin(set(stages))].copy()
    return frame.reset_index(drop=True)


def _formal_view(frame: pd.DataFrame) -> pd.DataFrame:
    if "is_formal_result" in frame:
        return frame[frame["is_formal_result"].astype(bool)].copy()
    return frame[frame.apply(is_formal_result, axis=1)].copy()


def _summary(frame: pd.DataFrame, groups: list[str], value: str = "test_metric") -> pd.DataFrame:
    if frame.empty or value not in frame:
        return pd.DataFrame(columns=[*groups, "mean", "std", "n", "ci95"])
    groups = [group for group in groups if group in frame]
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
    out["overlap"] = pd.to_numeric(out["overlap"], errors="coerce").fillna(pd.to_numeric(parsed["overlap"], errors="coerce"))
    out["cross_chart_edge_fraction"] = pd.to_numeric(out["cross_chart_edge_fraction"], errors="coerce").fillna(pd.to_numeric(parsed["cross"], errors="coerce"))
    return out


def paired_effect(frame: pd.DataFrame, target_model: str, baseline: str, metric: str = "test_metric") -> pd.DataFrame:
    frame = _formal_view(frame)
    keys = [column for column in ("dataset", "condition_id", "overlap", "cross_chart_edge_fraction", "task", "seed", "split", "experiment_stage") if column in frame]
    if metric not in frame or not keys:
        return pd.DataFrame(columns=[*keys, "target", "baseline", "delta"])
    target = frame.loc[frame["model"].eq(target_model), keys + [metric]].rename(columns={metric: "target"})
    base = frame.loc[frame["model"].eq(baseline), keys + [metric]].rename(columns={metric: "baseline"})
    return target.merge(base, on=keys, how="inner").dropna(subset=["target", "baseline"]).assign(
        delta=lambda item: item["target"] - item["baseline"]
    )


def bootstrap_paired_effect(paired: pd.DataFrame, *, resamples: int = 10000, seed: int = 2026) -> dict[str, float]:
    """Deterministic paired bootstrap; seeds are the resampling units."""
    values = paired.get("delta", pd.Series(dtype=float)).dropna().to_numpy(dtype=float)
    if not len(values):
        return {"mean_delta": float("nan"), "median_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0, "wins": 0, "ties": 0, "losses": 0}
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(values.mean()), "median_delta": float(np.median(values)),
        "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
        "n": int(len(values)), "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()),
    }


def build_phase_diagram_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c_oracle") -> pd.DataFrame:
    frame = _condition_coordinates(load_results_frame(results))
    rows = []
    for baseline, label in (("graphatlas_no_transport", "no_transport"), ("graphatlas_free_transition", "free_transition"), ("graphatlas_original", "original")):
        paired = paired_effect(frame, target_model, baseline)
        if paired.empty:
            continue
        paired = _condition_coordinates(paired)
        aggregate = _summary(paired, ["dataset", "condition_id", "overlap", "cross_chart_edge_fraction"], "delta").rename(columns={"mean": "mean_delta"})
        aggregate["mean"] = aggregate["mean_delta"]
        aggregate["contrast"] = label
        for source, name in (("boundary_accuracy", "boundary_gain"), ("routing_opportunity", "routing_opportunity"), ("irreducible_transport_risk", "irreducible_transport_risk")):
            if source in frame:
                metric_pair = paired_effect(frame, target_model, baseline, source)
                metric_pair = _condition_coordinates(metric_pair)
                metric_summary = _summary(metric_pair, ["dataset", "condition_id", "overlap", "cross_chart_edge_fraction"], "delta").rename(columns={"mean": name})
                merge_keys = [key for key in ("dataset", "condition_id", "overlap", "cross_chart_edge_fraction") if key in aggregate and key in metric_summary]
                aggregate = aggregate.merge(metric_summary[merge_keys + [name]], on=merge_keys, how="left")
        rows.append(aggregate)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_beta_sweep_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c_oracle") -> pd.DataFrame:
    frame = load_results_frame(results)
    beta = frame.loc[frame["model"].eq(target_model) | frame.get("model_family", pd.Series(index=frame.index, dtype=str)).eq("graphatlas_certified")].copy()
    if beta.empty:
        return pd.DataFrame()
    beta["beta"] = pd.to_numeric(beta.get("transportability_beta"), errors="coerce")
    rows = []
    for metric in ("test_metric", "boundary_accuracy", "routing_entropy_mean", "q_route_spread_mean", "distortion_mean"):
        if metric in beta:
            item = _summary(beta, ["beta"], metric)
            item["metric"] = metric
            rows.append(item)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_counterfactual_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c_oracle", *, include_upper_bound: bool = False) -> pd.DataFrame:
    frame = load_results_frame(results, include_upper_bound=include_upper_bound)
    family = frame["model_family"] if "model_family" in frame else frame["model"]
    controls = frame[family.astype(str).isin({"graphatlas_certified", "graphatlas_c"})].copy()
    if controls.empty:
        return pd.DataFrame()
    routing = controls["certified_routing_mode"] if "certified_routing_mode" in controls else pd.Series("certificate", index=controls.index)
    controls["routing_mode"] = routing.fillna("certificate")
    controls.loc[controls["model"].astype(str).str.contains("q_reference|oracle", case=False, regex=True), "routing_mode"] = "q_reference_max"
    return _summary(controls, ["routing_mode", "condition_id"], "test_metric")


def build_invariance_curve_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results, include_upper_bound=True)
    if "status" in frame:
        frame = frame[frame["status"].fillna("complete").isin({"complete", "identity_failed", "ok"})]
    rows = []
    stress_metrics = {
        "max_logit_error": "max_logit_error", "mean_probability_error": "mean_probability_error",
        "max_probability_error": "max_probability_error", "prediction_flip_rate": "prediction_flip_rate",
        "test_metric_drop": "test_metric_drop", "mean_logit_error": "mean_logit_error",
    }
    if "transform_strength" in frame and frame["transform_strength"].notna().any():
        stress = frame.copy()
        transform_kind = stress["transform_kind"] if "transform_kind" in stress else pd.Series("unknown", index=stress.index)
        stress["transform_family"] = transform_kind.fillna("unknown")
        for metric, output_name in stress_metrics.items():
            if metric not in stress:
                continue
            item = _summary(stress, ["model", "dataset", "transform_family", "transform_strength"], metric)
            item = item.rename(columns={"mean": "summary_mean", "std": "summary_std"})
            item["mean"] = item["summary_mean"]
            item["std"] = item["summary_std"]
            item["metric"] = output_name
            item["strength"] = item["transform_strength"]
            item["ci_low"] = item["summary_mean"] - 1.96 * item["summary_std"].fillna(0) / np.sqrt(item["n"].clip(lower=1))
            item["ci_high"] = item["summary_mean"] + 1.96 * item["summary_std"].fillna(0) / np.sqrt(item["n"].clip(lower=1))
            rows.append(item)
    legacy = frame[frame.get("transform_strength", pd.Series(np.nan, index=frame.index)).isna()].copy()
    legacy_map = {
        "invariance_logit_max_affine": ("max_logit_error", "affine"),
        "invariance_logit_max_nonlinear": ("max_logit_error", "nonlinear"),
        "invariance_error_affine": ("mean_probability_error", "affine"),
        "invariance_error_nonlinear": ("mean_probability_error", "nonlinear"),
        "intervention_flip_rate_affine": ("prediction_flip_rate", "affine"),
        "intervention_flip_rate_nonlinear": ("prediction_flip_rate", "nonlinear"),
        "intervention_test_metric_drop_affine": ("test_metric_drop", "affine"),
        "intervention_test_metric_drop_nonlinear": ("test_metric_drop", "nonlinear"),
    }
    for source, (metric, family) in legacy_map.items():
        if source not in legacy:
            continue
        item = _summary(legacy, ["model"], source).rename(columns={"mean": "summary_mean", "std": "summary_std"})
        item["dataset"] = "legacy"
        item["transform_family"] = family
        item["metric"] = metric
        item["strength"] = np.nan
        item["strength_status"] = "legacy_missing_strength"
        item["mean"] = item["summary_mean"]
        item["std"] = item["summary_std"]
        item["ci_low"] = item["summary_mean"] - item["ci95"]
        item["ci_high"] = item["summary_mean"] + item["ci95"]
        rows.append(item)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=["model", "transform_family", "strength", "metric", "summary_mean", "ci_low", "ci_high", "n"])


def build_transport_diagnostics_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    columns = [column for column in ("transportability_mean", "distortion_mean", "q_route_spread_mean", "routing_entropy_mean", "chart_condition_number", "membership_entropy_mean", "routing_opportunity", "irreducible_transport_risk", "cross_chart_mass", "q_reference_agreement") if column in frame]
    return frame[[column for column in ["dataset", "condition_id", "model", "seed", "split", *columns] if column in frame]].copy()


def build_real_benchmark_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    real = frame[~frame["dataset"].astype(str).str.startswith("atlas_het")].copy()
    summary = _summary(real, ["dataset", "model", "metric_name"], "test_metric")
    if summary.empty:
        return summary
    summary["dataset_z"] = summary.groupby("dataset")["mean"].transform(lambda values: (values - values.mean()) / values.std(ddof=0) if values.std(ddof=0) > 0 else 0.0)
    return summary


def build_hetgb_feature_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    subset = frame[frame["dataset"].astype(str).str.startswith("hetgb")].copy()
    if "feature_source" not in subset:
        subset["feature_source"] = subset.get("condition_id", "unknown")
    return _summary(subset, ["dataset", "feature_source", "model"], "test_metric")


def build_link_prediction_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    task = frame.get("task", pd.Series("", index=frame.index))
    if "link_decoder" not in frame:
        frame = frame.copy(); frame["link_decoder"] = "dot"
    return _summary(frame[task.eq("link_prediction")], ["dataset", "link_decoder", "model"], "test_metric")


def build_runtime_pareto_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = load_results_frame(results)
    required = {"test_metric", "runtime_seconds"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=[
            "dataset", "model", "test_metric", "runtime_seconds", "parameters",
            "peak_cuda_memory_bytes", "num_nodes", "num_edges", "num_layers", "membership_topk",
        ])
    columns = [column for column in ("dataset", "model", "test_metric", "runtime_seconds", "parameters", "peak_cuda_memory_bytes", "num_nodes", "num_edges", "num_layers", "membership_topk") if column in frame]
    out = frame[columns].copy().dropna(subset=["test_metric", "runtime_seconds"])
    if "peak_cuda_memory_bytes" in out:
        out["peak_cuda_memory_mb"] = pd.to_numeric(out["peak_cuda_memory_bytes"], errors="coerce") / 1024**2
    return out


def build_opportunity_risk_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c_oracle") -> pd.DataFrame:
    frame = load_results_frame(results)
    subset = frame[frame["model"].eq(target_model)].copy()
    columns = ("dataset", "condition_id", "seed", "test_metric", "routing_opportunity", "irreducible_transport_risk", "boundary_accuracy", "cross_chart_mass", "q_route_spread_mean")
    return subset[[column for column in columns if column in subset]].copy()


def build_rank_summary_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = _formal_view(load_results_frame(results))
    keys = [column for column in ("dataset", "task", "seed", "split", "experiment_stage") if column in frame]
    if not keys:
        return pd.DataFrame()
    ranks = frame.groupby(keys + ["model"], as_index=False)["test_metric"].mean()
    ranks["rank"] = ranks.groupby(keys)["test_metric"].rank(ascending=False, method="average")
    return _summary(ranks, ["model"], "rank")


def build_seed_level_slope_frame(results: str | Path | pd.DataFrame, target_model: str = "graphatlas_c_oracle", baseline: str = "graphatlas_original") -> pd.DataFrame:
    """Return paired seed-level values for a uniform slope graph."""
    frame = _formal_view(load_results_frame(results))
    keys = [key for key in ("dataset", "task", "condition_id", "seed", "split", "experiment_stage") if key in frame]
    if not keys or "test_metric" not in frame:
        return pd.DataFrame(columns=[*keys, "baseline", "target", "delta"])
    left = frame[frame["model"].eq(baseline)][keys + ["test_metric"]].rename(columns={"test_metric": "baseline"})
    right = frame[frame["model"].eq(target_model)][keys + ["test_metric"]].rename(columns={"test_metric": "target"})
    return left.merge(right, on=keys, how="inner").assign(delta=lambda x: x["target"] - x["baseline"])


def build_routing_intervention_frame(results: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Keep paired rows for routing controls; plotting decides the panel layout."""
    frame = _formal_view(load_results_frame(results))
    if "test_metric" not in frame:
        return pd.DataFrame()
    mode = frame.get("certified_routing_mode", pd.Series("certificate", index=frame.index)).fillna("certificate").astype(str)
    out = frame[[key for key in ("dataset", "task", "condition_id", "seed", "split", "model", "test_metric") if key in frame]].copy()
    out["routing_mode"] = mode.values
    out.loc[out["model"].astype(str).str.contains("oracle|q_reference", case=False, regex=True), "routing_mode"] = "oracle"
    out.loc[out["model"].astype(str).str.contains("shuffle_source", case=False, regex=True), "routing_mode"] = "q_shuffle_source"
    out.loc[out["model"].astype(str).str.contains("shuffle_edge", case=False, regex=True), "routing_mode"] = "q_shuffle_edge"
    out.loc[out["model"].astype(str).str.contains("membership", case=False, regex=True), "routing_mode"] = "membership_only"
    return out


def build_risk_calibration_frame(results: str | Path | pd.DataFrame, *, risk_column: str = "node_transport_risk", error_column: str = "node_error", bins: int = 10) -> pd.DataFrame:
    """Build reliability points only when node-level risk and error are present."""
    frame = load_results_frame(results)
    if risk_column not in frame or error_column not in frame:
        return pd.DataFrame(columns=["dataset", "model", "risk_bin", "predicted_risk", "error_rate", "n"])
    work = frame[[key for key in ("dataset", "model", risk_column, error_column) if key in frame]].copy()
    work["predicted_risk"] = pd.to_numeric(work[risk_column], errors="coerce")
    work["observed_error"] = pd.to_numeric(work[error_column], errors="coerce")
    work = work.dropna(subset=["predicted_risk", "observed_error"])
    if work.empty:
        return pd.DataFrame(columns=["dataset", "model", "risk_bin", "predicted_risk", "error_rate", "n"])
    work["risk_bin"] = work.groupby(["dataset", "model"], dropna=False)["predicted_risk"].transform(lambda s: pd.qcut(s.rank(method="first"), min(bins, max(1, s.nunique())), labels=False, duplicates="drop"))
    return work.groupby(["dataset", "model", "risk_bin"], dropna=False).agg(predicted_risk=("predicted_risk", "mean"), error_rate=("observed_error", "mean"), n=("observed_error", "size")).reset_index()


def build_decile_response_frame(results: str | Path | pd.DataFrame, *, x_column: str, gain_column: str = "gain", bins: int = 10) -> pd.DataFrame:
    """Build per-dataset decile response curves from node-level analysis columns."""
    frame = load_results_frame(results)
    if x_column not in frame or gain_column not in frame:
        return pd.DataFrame(columns=["dataset", "model", "decile", "x_mean", "gain_mean", "n"])
    keys = [key for key in ("dataset", "model") if key in frame]
    work = frame[keys + [x_column, gain_column]].copy()
    work[x_column] = pd.to_numeric(work[x_column], errors="coerce"); work[gain_column] = pd.to_numeric(work[gain_column], errors="coerce")
    work = work.dropna()
    if work.empty:
        return pd.DataFrame(columns=["dataset", "model", "decile", "x_mean", "gain_mean", "n"])
    work["decile"] = work.groupby(keys, dropna=False)[x_column].transform(lambda s: pd.qcut(s.rank(method="first"), min(bins, max(1, s.nunique())), labels=False, duplicates="drop") + 1)
    return work.groupby(keys + ["decile"], dropna=False).agg(x_mean=(x_column, "mean"), gain_mean=(gain_column, "mean"), n=(gain_column, "size")).reset_index()


def export_plot_data(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
