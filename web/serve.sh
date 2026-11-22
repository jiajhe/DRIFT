#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
export PYTHONPATH="$root/cache/qa-transformers:$root/cache/qa-kernels:$root/cache/marketflow-http:$root/code/TSLM_training:$root/code:$root/code/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export CPATH="$root/cache/qa-headers/usr/include/python3.12:$root/cache/qa-headers/usr/include:$root/cache/qa-headers/usr/include/x86_64-linux-gnu"
export TRITON_CACHE_DIR="$root/cache/qa-triton"
exec "$root/cache/opentslm-venv/bin/python" -m uvicorn api.main:app --host "${MARKETFLOW_HOST:-127.0.0.1}" --port "${MARKETFLOW_PORT:-8765}"
