#!/usr/bin/env bash
# HPSv3++ Stage 2: semi-supervised adaptive training (open-source version, CSV data entry)
# Usage: bash train_stage2.sh   (defaults to single-node 8 GPUs, override with NPROC)
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

cd "$(dirname "$0")"

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29502}"
CONFIG="${CONFIG:-hpsv3/config/train_stage2.yaml}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_stage2_${TS}.log"

deepspeed --include "localhost:$(seq -s, 0 $((NPROC-1)))" --master_port "${MASTER_PORT}" \
    hpsv3/train_stage2.py \
    --config "${CONFIG}" \
    2>&1 | tee -a "${LOG_FILE}"
