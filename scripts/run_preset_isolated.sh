#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/runtime_env.sh
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# Keep one advisory lock across the recursive exec chain.
if [[ "${GRAPHATLAS_LOCK_HELD:-0}" != "1" ]]; then
  LOCK_PATH="${GRAPHATLAS_LOCK_PATH:-/tmp/graphatlas_experiments.lock}"
  export GRAPHATLAS_LOCK_HELD=1
  exec flock -n "$LOCK_PATH" "$0" "$@"
fi

PRESET="${1:-smoke}"
shift || true
LIMIT=""
FORCE=0
CONTINUE_ON_ERROR=0
RETRIES=1
START_INDEX=1
ATTEMPT=0
PREPARED=0
FAILED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    --retries) RETRIES="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --attempt) ATTEMPT="$2"; shift 2 ;;
    --prepared) PREPARED=1; shift ;;
    --failed-count) FAILED="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PRESET_NAME="$(basename "$PRESET")"
PRESET_NAME="${PRESET_NAME%.yaml}"
PLAN="outputs/manifests/${PRESET_NAME}.jsonl"
STATUS="outputs/status/experiment_${PRESET_NAME}.json"
mkdir -p outputs/manifests outputs/status

common_args=("$PRESET" --retries "$RETRIES" --start-index "$START_INDEX" --attempt "$ATTEMPT" --failed-count "$FAILED")
[[ -n "$LIMIT" ]] && common_args+=(--limit "$LIMIT")
[[ "$FORCE" -eq 1 ]] && common_args+=(--force)
[[ "$CONTINUE_ON_ERROR" -eq 1 ]] && common_args+=(--continue-on-error)

if [[ "$PREPARED" -eq 0 ]]; then
  materialize=(python -u scripts/materialize_jobs.py --preset "$PRESET" --plan "$PLAN")
  [[ -n "$LIMIT" ]] && materialize+=(--limit "$LIMIT")
  [[ "$FORCE" -eq 1 ]] && materialize+=(--force)
  "${materialize[@]}"
  # Re-exec before the first training worker. This avoids native-runtime state
  # leaking from one Python child into the next on restrictive HPC images.
  exec "$0" "${common_args[@]}" --prepared
fi

TOTAL="$(grep -cve '^\s*$' "$PLAN")"
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No jobs in $PLAN" >&2
  exit 2
fi

if (( START_INDEX > TOTAL )); then
  completed=$((TOTAL - FAILED))
  cat > "$STATUS" <<JSON
{
  "preset": "$PRESET_NAME",
  "expected_runs": $TOTAL,
  "completed_runs": $completed,
  "failed_runs": $FAILED,
  "remaining_runs": 0,
  "progress_fraction": 1.0,
  "current_index": $TOTAL,
  "complete": $([[ "$FAILED" -eq 0 ]] && echo true || echo false)
}
JSON
  printf '[##############################] 100%%  completed %d/%d, failed %d\n' "$completed" "$TOTAL" "$FAILED"
  if (( FAILED > 0 )); then
    echo "Experiment matrix completed with $FAILED failed job(s)." >&2
  fi
  exec python -u scripts/collect_results.py --plan "$PLAN" --preset-name "$PRESET_NAME"
fi

width=30
filled=$((width * (START_INDEX - 1) / TOTAL))
empty=$((width - filled))
percent=$((100 * (START_INDEX - 1) / TOTAL))
bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
printf '[%s] %3d%%  job %d/%d\n' "$bar" "$percent" "$START_INDEX" "$TOTAL"

set +e
python -u scripts/run_plan_job.py --plan "$PLAN" --index "$START_INDEX"
code=$?
set -e

if (( code != 0 )); then
  if (( ATTEMPT < RETRIES )); then
    echo "Job $START_INDEX failed; restarting a fresh worker from checkpoint." >&2
    next_args=("$PRESET" --retries "$RETRIES" --start-index "$START_INDEX" --attempt "$((ATTEMPT + 1))" --failed-count "$FAILED" --prepared)
    [[ -n "$LIMIT" ]] && next_args+=(--limit "$LIMIT")
    [[ "$FORCE" -eq 1 ]] && next_args+=(--force)
    [[ "$CONTINUE_ON_ERROR" -eq 1 ]] && next_args+=(--continue-on-error)
    exec "$0" "${next_args[@]}"
  fi
  FAILED=$((FAILED + 1))
  if (( CONTINUE_ON_ERROR == 0 )); then
    exit "$code"
  fi
fi

completed=$((START_INDEX - FAILED))
remaining=$((TOTAL - START_INDEX))
cat > "$STATUS" <<JSON
{
  "preset": "$PRESET_NAME",
  "expected_runs": $TOTAL,
  "completed_runs": $completed,
  "failed_runs": $FAILED,
  "remaining_runs": $remaining,
  "progress_fraction": $(awk "BEGIN {print $START_INDEX / $TOTAL}"),
  "current_index": $START_INDEX,
  "complete": false
}
JSON

next_args=("$PRESET" --retries "$RETRIES" --start-index "$((START_INDEX + 1))" --attempt 0 --failed-count "$FAILED" --prepared)
[[ -n "$LIMIT" ]] && next_args+=(--limit "$LIMIT")
[[ "$FORCE" -eq 1 ]] && next_args+=(--force)
[[ "$CONTINUE_ON_ERROR" -eq 1 ]] && next_args+=(--continue-on-error)
exec "$0" "${next_args[@]}"
