#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export PYTHONPATH="$root/cache/qa-transformers:$root/cache/qa-kernels:$root/code:$root/code/src"
export CPATH="$root/cache/qa-headers/usr/include/python3.12:$root/cache/qa-headers/usr/include:$root/cache/qa-headers/usr/include/x86_64-linux-gnu"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TRITON_CACHE_DIR="$root/cache/qa-triton"
exec "$root/cache/opentslm-venv/bin/python" -u "$root/code/TSLM_training/v2_train.py" "$@"
