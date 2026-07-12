from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join([str(part) for part in column if str(part)]) if isinstance(column, tuple) else str(column)
        for column in frame.columns
    ]
    return frame.reset_index()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows: list[str] = []
    for _, row in selected.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if math.isnan(float(value)) else f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *rows])


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    finite = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    finite.sort(key=lambda item: item[1])
    running = 0.0
    count = len(finite)
    for rank, (index, value) in enumerate(finite):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_tests(frame: pd.DataFrame, target_model: str = "graphatlas") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [column for column in ["seed", "split"] if column in frame]
    if not keys:
        return pd.DataFrame()
    group_columns = [column for column in ["dataset", "task"] if column in frame]
    grouped = frame.groupby(group_columns) if group_columns else [((), frame)]
    for group_value, dataset_frame in grouped:
        if not isinstance(group_value, tuple):
            group_value = (group_value,)
        group_metadata = dict(zip(group_columns, group_value, strict=True))
        target = dataset_frame[dataset_frame["model"] == target_model][keys + ["test_metric"]].rename(
            columns={"test_metric": "target"}
        )
        for model in sorted(set(dataset_frame["model"]) - {target_model}):
            baseline = dataset_frame[dataset_frame["model"] == model][keys + ["test_metric"]].rename(
                columns={"test_metric": "baseline"}
            )
            paired = target.merge(baseline, on=keys).dropna()
            if paired.empty:
                continue
            delta = paired["target"] - paired["baseline"]
            t_p = (
                float(stats.ttest_rel(paired["target"], paired["baseline"]).pvalue)
                if len(paired) >= 2
                else float("nan")
            )
            try:
                w_p = (
                    float(stats.wilcoxon(delta).pvalue)
                    if len(paired) >= 2 and np.any(delta != 0)
                    else float("nan")
                )
            except ValueError:
                w_p = float("nan")
            std = float(delta.std(ddof=1)) if len(delta) > 1 else float("nan")
            effect = float(delta.mean() / std) if np.isfinite(std) and std > 0 else float("nan")
            rows.append(
                {
                    **group_metadata,
                    "baseline": model,
                    "n_pairs": len(paired),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                    "paired_effect_d": effect,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_holm_p"] = _holm_adjust(result["wilcoxon_p"].tolist())
    return result


def model_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ["dataset", "task", "seed", "split"] if column in frame]
    if not keys or "model" not in frame or "test_metric" not in frame:
        return pd.DataFrame()
    per_run = frame.groupby(keys + ["model"], as_index=False)["test_metric"].mean()
    per_run["rank"] = per_run.groupby(keys)["test_metric"].rank(method="average", ascending=False)
    rank_groups = [column for column in ["task", "model"] if column in per_run]
    if not rank_groups:
        return pd.DataFrame()
    result = (
        per_run.groupby(rank_groups, as_index=False)
        .agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), evaluated_runs=("rank", "count"))
    )
    sort_columns = (["task"] if "task" in result else []) + ["mean_rank"]
    return result.sort_values(sort_columns).reset_index(drop=True)


def scientific_gates(frame: pd.DataFrame, boundary_margin: float = 0.02) -> dict[str, Any]:
    synthetic = frame[frame["dataset"] == "atlas_het"].copy() if "dataset" in frame else pd.DataFrame()
    if "task" in synthetic:
        synthetic = synthetic[synthetic["task"] == "node_classification"]
    gates: list[dict[str, Any]] = []

    def add(name: str, passed: bool | None, value: Any, criterion: str) -> None:
        gates.append({"name": name, "passed": passed, "value": value, "criterion": criterion})

    full = synthetic[synthetic["model"] == "graphatlas"] if "model" in synthetic else pd.DataFrame()
    if full.empty or "invariance_error_nonlinear" not in full:
        add("full_model_present", False, None, "At least one Atlas-Het GraphAtlas run exists with intervention metrics")
    else:
        inv = float(full["invariance_error_nonlinear"].max())
        add("nonlinear_coordinate_invariance", inv < 1e-6, inv, "maximum nonlinear intervention error < 1e-6")

    for ablation in ["graphatlas_no_transport", "graphatlas_free_transition"]:
        other = synthetic[synthetic["model"] == ablation] if "model" in synthetic else pd.DataFrame()
        if full.empty or other.empty or "invariance_error_nonlinear" not in synthetic:
            add(f"invariance_separates_{ablation}", None, None, "both models must be present")
            continue
        full_inv = float(full["invariance_error_nonlinear"].mean())
        other_inv = float(other["invariance_error_nonlinear"].mean())
        ratio = other_inv / max(full_inv, 1e-15)
        add(f"invariance_separates_{ablation}", ratio >= 10.0, ratio, "ablation error / full error >= 10")

    no_transport = synthetic[synthetic["model"] == "graphatlas_no_transport"] if "model" in synthetic else pd.DataFrame()
    if not full.empty and not no_transport.empty and "boundary_accuracy" in synthetic:
        keys = [column for column in ["seed", "split"] if column in synthetic]
        paired = full[keys + ["boundary_accuracy"]].merge(
            no_transport[keys + ["boundary_accuracy"]], on=keys, suffixes=("_full", "_ablation")
        ).dropna()
        delta = (
            float((paired["boundary_accuracy_full"] - paired["boundary_accuracy_ablation"]).mean())
            if not paired.empty
            else float("nan")
        )
        add(
            "boundary_node_gain",
            bool(np.isfinite(delta) and delta >= boundary_margin),
            delta,
            f"mean paired gain >= {boundary_margin:.3f}",
        )
    else:
        add("boundary_node_gain", None, None, "full and no-transport boundary metrics must exist")

    free = synthetic[synthetic["model"] == "graphatlas_free_transition"] if "model" in synthetic else pd.DataFrame()
    if not full.empty and not free.empty and "test_metric" in synthetic:
        keys = [column for column in ["seed", "split"] if column in synthetic]
        paired = full[keys + ["test_metric"]].merge(
            free[keys + ["test_metric"]], on=keys, suffixes=("_full", "_free")
        ).dropna()
        delta = (
            float((paired["test_metric_full"] - paired["test_metric_free"]).mean())
            if not paired.empty
            else float("nan")
        )
        add("induced_transition_gain", bool(np.isfinite(delta) and delta > 0.0), delta, "mean paired test-metric gain > 0")
    else:
        add("induced_transition_gain", None, None, "full and free-transition runs must exist")

    decisive = [gate for gate in gates if gate["passed"] is not None]
    return {
        "all_decisive_gates_pass": bool(decisive) and all(bool(gate["passed"]) for gate in decisive),
        "passed": sum(gate["passed"] is True for gate in gates),
        "failed": sum(gate["passed"] is False for gate in gates),
        "not_evaluated": sum(gate["passed"] is None for gate in gates),
        "gates": gates,
    }


def summarize_results(
    results_path: str | Path,
    plots: bool = True,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Result table is empty: {path}")
    if "test_metric" not in frame:
        if "test_accuracy" not in frame:
            raise ValueError("Result table must contain test_metric or test_accuracy")
        frame["test_metric"] = frame["test_accuracy"]

    report_dir = Path(output_dir) if output_dir is not None else path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    numeric = [
        column
        for column in [
            "test_metric",
            "test_accuracy",
            "test_roc_auc",
            "test_macro_f1",
            "test_micro_f1",
            "test_weighted_f1",
            "test_balanced_accuracy",
            "test_mcc",
            "test_worst_class_recall",
            "test_nll",
            "test_brier",
            "test_ece",
            "test_average_precision",
            "test_link_mrr",
            "test_link_hits_at_10",
            "test_link_hits_at_50",
            "boundary_accuracy",
            "interior_accuracy",
            "chart_ari",
            "overlap_f1",
            "invariance_error_affine",
            "invariance_error_nonlinear",
            "intervention_flip_rate_nonlinear",
            "intervention_test_metric_drop_nonlinear",
            "intervention_test_metric_nonlinear",
            "mean_active_charts",
            "overlap_ratio",
            "membership_entropy_mean",
            "chart_utilization_min",
            "chart_utilization_max",
            "cocycle_error",
            "path_consistency_error",
            "triple_cocycle_error",
            "metric_compatibility_error",
            "geometry_error",
            "runtime_seconds",
            "parameters",
        ]
        if column in frame
    ]
    group_columns = [column for column in ["dataset", "task", "model"] if column in frame]
    if not group_columns:
        raise ValueError("Result table must contain at least one grouping column")
    grouped = frame.groupby(group_columns)[numeric].agg(["count", "mean", "std"])
    summary = _flatten_columns(grouped)
    if "test_metric_count" in summary:
        summary["test_metric_ci95"] = (
            1.96 * summary["test_metric_std"] / np.sqrt(summary["test_metric_count"].clip(lower=1))
        )
    summary.to_csv(report_dir / "summary.csv", index=False)

    tests = paired_tests(frame)
    tests.to_csv(report_dir / "paired_tests.csv", index=False)
    ranks = model_ranks(frame)
    ranks.to_csv(report_dir / "model_ranks.csv", index=False)
    gates = scientific_gates(frame)
    (report_dir / "scientific_gates.json").write_text(
        json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    display_columns = [
        column
        for column in [
            "dataset",
            "task",
            "model",
            "test_metric_count",
            "test_metric_mean",
            "test_metric_std",
            "test_metric_ci95",
            "boundary_accuracy_mean",
            "invariance_error_nonlinear_mean",
            "path_consistency_error_mean",
            "metric_compatibility_error_mean",
            "runtime_seconds_mean",
        ]
        if column in summary
    ]
    markdown = "# GraphAtlas experiment summary\n\n" + markdown_table(summary, display_columns)
    if not tests.empty:
        markdown += "\n\n# Paired tests against GraphAtlas\n\n" + markdown_table(tests, list(tests.columns))
    if not ranks.empty:
        markdown += "\n\n# Mean model ranks\n\n" + markdown_table(ranks, list(ranks.columns))
    gate_frame = pd.DataFrame(gates["gates"])
    markdown += "\n\n# Scientific claim gates\n\n" + markdown_table(gate_frame, list(gate_frame.columns))
    (report_dir / "summary.md").write_text(markdown + "\n", encoding="utf-8")

    if plots:
        from graphatlas.visualization import visualize_results

        visualize_results(path, report_dir / "figures")
    return summary, tests, gates
