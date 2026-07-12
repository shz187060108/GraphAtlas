#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source scripts/runtime_env.sh
python scripts/run_pipeline.py --preset confirmatory "$@"
