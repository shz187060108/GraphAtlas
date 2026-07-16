#!/usr/bin/env python
"""Remove only superseded reports and low-value legacy result summaries."""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import shutil
from pathlib import Path


ROOT = Path(PROJECT_ROOT)
OUTPUTS = ROOT / "outputs"

LEGACY_DIRECTORIES = (
    OUTPUTS / "reports" / "paper_dense",
    OUTPUTS / "reports" / "paper_figures",
    OUTPUTS / "real_best_single",
)
LEGACY_RESULT_NAMES = {
    "all.csv", "latest.csv", "latest_concise.csv",
    "public_graphatlas_seed0.csv", "selected_seed0.csv",
    "real_best_single.csv", "real_best_single_summary.csv",
    "real_best_single_actor.csv", "real_best_single_actor_concise.csv",
    "real_best_single_h2gb_ieee_cis.csv", "real_best_single_h2gb_ieee_cis_concise.csv",
    "real_best_single_hetgb_actor.csv", "real_best_single_hetgb_actor_concise.csv",
    "real_best_single_hetgb_amazon.csv", "real_best_single_hetgb_amazon_concise.csv",
    "real_best_single_hetgb_texas.csv", "real_best_single_hetgb_texas_concise.csv",
    "real_best_single_questions.csv", "real_best_single_questions_concise.csv",
}
ROOT_TEMP_FILES = (
    ROOT / ".codex_graphatlas_dense_task.md",
    ROOT / "GraphAtlas_Codex_Dense_Experiment_Spec.md",
)


def cleanup_targets() -> list[Path]:
    targets = [path for path in LEGACY_DIRECTORIES if path.exists()]
    targets.extend(
        path for path in (OUTPUTS / "results").glob("*.csv")
        if path.name in LEGACY_RESULT_NAMES
    )
    targets.extend(
        path for path in (OUTPUTS / "manifests").glob("*")
        if (
            path.name.startswith(("public_graphatlas_seed0", "selected_seed0", "protocol_retry_seed0", "roman_p0"))
            or (path.name.startswith("real_best_single_") and "h2gb_pdns" not in path.name)
        )
    )
    targets.extend(path for path in ROOT_TEMP_FILES if path.exists())
    return sorted(set(targets), key=lambda path: str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the printed cleanup plan.")
    args = parser.parse_args()
    targets = cleanup_targets()
    for path in targets:
        print(path.relative_to(ROOT))
    if not args.apply:
        print(f"dry-run: {len(targets)} targets; pass --apply to remove them")
        return
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print(f"removed {len(targets)} superseded artifacts")


if __name__ == "__main__":
    main()
