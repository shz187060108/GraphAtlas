#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/runtime_env.sh
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -u scripts/run_pipeline.py \
  --preset configs/presets/roman_p0.yaml "$@"
