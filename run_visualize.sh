#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source scripts/runtime_env.sh
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
RESULTS="${1:-outputs/results/complete.csv}"
REPORT_DIR="${2:-outputs/reports/manual}"
python -u scripts/summarize.py --results "${RESULTS}" --output-dir "${REPORT_DIR}"
