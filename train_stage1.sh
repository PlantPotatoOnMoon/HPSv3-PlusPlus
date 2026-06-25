#!/usr/bin/env bash
# HPSv3++ Stage 1: OGD continual learning (open-source version, CSV data entry)
# Usage: bash train_stage1.sh   (defaults to single-node 8 GPUs, override with NPROC)
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TORCH_SHOW_CPP_STACKTRACES=1

cd "$(dirname "$0")"

NNODES="${NNODES:-1}"
NPROC="${NPROC:-8}"
CONFIG="${CONFIG:-hpsv3/config/train_stage1.yaml}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_stage1_${TS}.log"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC}" \
  hpsv3/train_stage1.py \
  --config "${CONFIG}" \
  2>&1 | tee -a "${LOG_FILE}"
