#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${DRIFT_PYTHON:-/root/autodl-tmp/meta-fdg/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3)"; fi
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
elif [[ "$1" == "python" ]]; then
  shift
  exec "$PY" "$@"
else
  printf '%s\n' 'Usage: ./run.sh verify | load --cell alpha158_csi300 | python <script> [args]'
  exit 2
fi
