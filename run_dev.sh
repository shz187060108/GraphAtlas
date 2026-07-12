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
exec python -u scripts/run_pipeline.py --mode kill "$@"
