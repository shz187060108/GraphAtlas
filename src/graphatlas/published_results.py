from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

REQUIRED_COLUMNS = [
    "source_id", "paper_title", "venue", "year", "table_id", "source_url",
    "dataset", "dataset_version", "task", "metric", "feature_condition",
    "split_protocol", "model", "mean", "std", "unit", "n_runs",
    "comparison_level", "notes",
]
ALLOWED_LEVELS = {"exact", "contextual", "incompatible"}
ALIASES = {
    "roman-empire": "roman_empire", "roman_empire": "roman_empire",
    "amazon-ratings": "amazon_ratings", "amazon_ratings": "amazon_ratings",
    "chameleon-filtered": "chameleon_filtered", "squirrel-filtered": "squirrel_filtered",
    "roc auc": "roc_auc", "roc-auc": "roc_auc", "auc": "roc_auc",
    "f1 score": "macro_f1", "f1": "macro_f1",
}


def normalize_name(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return ALIASES.get(text, text)


def _to_fraction(mean: float, std: float, unit: str) -> tuple[float, float]:
    if unit == "percent":
        return mean / 100.0, std / 100.0 if np.isfinite(std) else std
    if unit == "fraction":
        return mean, std
    raise ValueError(f"Unsupported result unit: {unit}")


def load_published_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames=[]
    for value in paths:
        path=Path(value)
        if path.is_dir():
            for item in sorted(path.glob("*.csv")):
                frames.append(pd.read_csv(item))
        else:
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    frame=pd.concat(frames, ignore_index=True, sort=False)
    return validate_published_results(frame)


def validate_published_results(frame: pd.DataFrame) -> pd.DataFrame:
    missing=[c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Published result table is missing columns: {missing}")
    out=frame[REQUIRED_COLUMNS].copy()
    for col in ["source_id", "paper_title", "venue", "table_id", "source_url", "dataset_version",
                "task", "feature_condition", "split_protocol", "model", "unit", "comparison_level", "notes"]:
        out[col]=out[col].fillna("").astype(str).str.strip()
    out["dataset"]=out["dataset"].map(normalize_name)
    out["metric"]=out["metric"].map(normalize_name)
    out["task"]=out["task"].map(normalize_name)
    out["year"]=pd.to_numeric(out["year"], errors="raise").astype(int)
    out["mean"]=pd.to_numeric(out["mean"], errors="raise")
    out["std"]=pd.to_numeric(out["std"], errors="coerce")
    out["n_runs"]=pd.to_numeric(out["n_runs"], errors="coerce").astype("Int64")
    invalid=set(out["comparison_level"])-ALLOWED_LEVELS
    if invalid:
        raise ValueError(f"Invalid comparison levels: {sorted(invalid)}")
    means=[]; stds=[]
    for row in out.itertuples(index=False):
        mean,std=_to_fraction(float(row.mean), float(row.std), row.unit)
        if not 0.0 <= mean <= 1.0:
            raise ValueError(f"Score outside [0,1] for {row.source_id}/{row.dataset}/{row.model}: {mean}")
        means.append(mean); stds.append(std)
    out["mean_fraction"]=means
    out["std_fraction"]=stds
    duplicate_keys=["source_id","dataset","dataset_version","task","metric","feature_condition","split_protocol","model"]
    if out.duplicated(duplicate_keys).any():
        dup=out.loc[out.duplicated(duplicate_keys, keep=False), duplicate_keys]
        raise ValueError(f"Duplicate published rows:\n{dup.to_string(index=False)}")
    return out


def load_protocols(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def summarize_graphatlas_results(path: str | Path) -> pd.DataFrame:
    frame=pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    if "test_metric" not in frame:
        raise ValueError("GraphAtlas results require test_metric")
    group=[c for c in ["dataset","task","model","metric_name"] if c in frame]
    summary=(frame.groupby(group, as_index=False)
             .agg(graphatlas_mean=("test_metric","mean"), graphatlas_std=("test_metric","std"),
                  graphatlas_runs=("test_metric","count")))
    summary["dataset"]=summary["dataset"].map(normalize_name)
    summary["task"]=summary["task"].map(normalize_name)
    if "metric_name" in summary:
        summary["metric"]=summary["metric_name"].map(normalize_name)
    else:
        summary["metric"]="accuracy"
    return summary


def build_comparison_bundle(graphatlas_results: str | Path, published_paths: Iterable[str | Path],
                            protocols_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output=Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    project=summarize_graphatlas_results(graphatlas_results)
    published=load_published_results(published_paths)
    protocols=load_protocols(protocols_path).get("datasets", {})
    rows=[]
    for row in published.to_dict("records"):
        rule=protocols.get(row["dataset"], {})
        level=row["comparison_level"]
        reasons=[]
        for field in ["dataset_version","split_protocol","feature_condition","metric"]:
            expected=rule.get(field)
            if expected and str(row[field]) != str(expected):
                reasons.append(f"{field}: published={row[field]}, project={expected}")
        if level == "exact" and reasons:
            level="contextual"
        row["effective_comparison_level"]=level
        row["compatibility_notes"]="; ".join(reasons)
        rows.append(row)
    validated=pd.DataFrame(rows)
    validated.to_csv(output/"published_results_validated.csv", index=False)
    exact=validated[validated["effective_comparison_level"].eq("exact")].copy()
    contextual=validated[validated["effective_comparison_level"].eq("contextual")].copy()
    join_keys=["dataset","task","metric"]
    merged=project.merge(exact, on=join_keys, how="left", suffixes=("_graphatlas","_published"))
    merged["delta_vs_published_mean"]=merged["graphatlas_mean"]-merged["mean_fraction"]
    merged.to_csv(output/"comparison_exact.csv", index=False)
    contextual.to_csv(output/"comparison_contextual.csv", index=False)
    audit={
        "graphatlas_rows": int(len(project)), "published_rows": int(len(validated)),
        "exact_rows": int((validated["effective_comparison_level"]=="exact").sum()),
        "contextual_rows": int((validated["effective_comparison_level"]=="contextual").sum()),
        "policy": "Published aggregate means are never used for paired significance tests or seed-level ranks.",
    }
    (output/"provenance_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    cols=[c for c in ["dataset","metric","model_graphatlas","graphatlas_mean","graphatlas_std","model_published",
                      "mean_fraction","std_fraction","delta_vs_published_mean","source_id","table_id"] if c in merged]
    md=["# Published-result comparison", "", "Only rows with matching dataset version, split protocol, features and metric appear below.",
        "Published aggregates are not used for paired tests.", ""]
    if not merged.empty and cols:
        md.append(merged[cols].to_markdown(index=False))
    else:
        md.append("No exact-match rows are currently available.")
    md += ["", "## Contextual references", "", "These rows are useful for orientation but are not apples-to-apples comparisons.", ""]
    if not contextual.empty:
        ccols=[c for c in ["dataset","metric","model","mean_fraction","std_fraction","feature_condition","split_protocol","source_id","table_id"] if c in contextual]
        md.append(contextual[ccols].to_markdown(index=False))
    (output/"comparison.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    return {"validated": output/"published_results_validated.csv", "exact": output/"comparison_exact.csv",
            "contextual": output/"comparison_contextual.csv", "markdown": output/"comparison.md",
            "audit": output/"provenance_audit.json"}
