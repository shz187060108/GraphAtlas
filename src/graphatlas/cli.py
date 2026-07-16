from __future__ import annotations

import argparse
from pathlib import Path

from graphatlas.config import ExperimentConfig
from graphatlas.datasets import load_dataset
from graphatlas.experiments import run_preset
from graphatlas.figure_data import load_results_frame
from graphatlas.reporting import submission_readiness_gates, summarize_results
from graphatlas.trainer import Trainer


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train one GraphAtlas experiment.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    config.validate()
    data = load_dataset(config.dataset, config.train.seed)
    run_dir = args.run_dir or (
        Path(config.train.output_dir) / config.dataset.name / config.model.name / f"seed_{config.train.seed}"
    )
    _, metrics = Trainer(config).fit(data, run_dir)
    for key, value in metrics.items():
        print(f"{key}: {value}")


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable GraphAtlas experiment matrix.")
    parser.add_argument("--preset", default="full", help="Preset name or YAML path")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path.cwd()
    preset_path = Path(args.preset)
    if not preset_path.suffix:
        preset_path = project_root / "configs" / "presets" / f"{args.preset}.yaml"
    frame = run_preset(
        preset_path,
        force=args.force,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )
    if not frame.empty:
        print(frame.to_string(index=False))


def summarize_main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GraphAtlas experiment results.")
    parser.add_argument("--results", default="outputs/best_config_search/search_summary.csv")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--strict-submission-gates", action="store_true")
    args = parser.parse_args()
    summary, _, gates = summarize_results(args.results, plots=not args.no_plots)
    print(summary.to_string(index=False))
    if args.strict_gates and not gates["all_decisive_gates_pass"]:
        raise SystemExit(2)
    submission = submission_readiness_gates(load_results_frame(args.results))
    if args.strict_submission_gates and not submission["all_submission_gates_pass"]:
        raise SystemExit(3)
