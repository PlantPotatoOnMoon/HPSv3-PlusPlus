#!/usr/bin/env bash
# Evaluate pairwise preference accuracy of the HPSv3++ checkpoint on the clean test sets (a single hpsv3++ checkpoint only)
# Usage: bash eval.sh   (evaluates aes + tf by default)
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1

cd "$(dirname "$0")"

CKPT="${CKPT:-checkpoints/hpsv3++.pth}"
CONFIG="${CONFIG:-hpsv3/config/train_stage2.yaml}"
NPROC="${NPROC:-8}"
BATCH="${BATCH:-4}"
IMG_ROOT="${IMG_ROOT:-datasets}"

for tag in aes tf; do
    echo "================== EVAL ${tag} =================="
    python evaluate/evaluate.py \
        --test_json "datasets/test/${tag}.json" \
        --config_path "${CONFIG}" \
        --checkpoint_path "${CKPT}" \
        --img_root "${IMG_ROOT}" \
        --mode pair \
        --batch_size "${BATCH}" \
        --num_processes "${NPROC}"
done
