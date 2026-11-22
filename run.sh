#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${DRIFT_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    printf '%s\n' 'Python 3 is required. Install dependencies from code/requirements.txt.' >&2
    exit 1
  fi
fi
export PYTHONPATH="$ROOT/code/src:$ROOT/code"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$ROOT/code"
if [[ "${1:-verify}" == "verify" ]]; then
  if [[ $# -gt 0 ]]; then shift; fi
  exec "$PY" "$ROOT/code/verify_bundle.py" "$@"
elif [[ "$1" == "load" ]]; then
  shift
  exec "$PY" "$ROOT/code/load_checkpoint.py" "$@"
elif [[ "$1" == "ensemble" ]]; then
  shift
  exec "$PY" "$ROOT/code/paper_ensemble.py" "$@"
elif [[ "$1" == "python" ]]; then
  shift
  exec "$PY" "$@"
else
  printf '%s\n' 'Usage: ./run.sh verify | load --cell alpha158_csi300 | ensemble --help | python <script> [args]'
  exit 2
fi
