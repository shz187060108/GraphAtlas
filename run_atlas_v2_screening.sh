#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
source scripts/runtime_env.sh
python -u scripts/run_pipeline.py --preset configs/presets/atlas_v2_screening.yaml "$@"
