#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/v12_1 results/v12_1 figures/v12_1

set -a
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
export WANDB_PROJECT="${WANDB_PROJECT:-geometry-of-relativity}"

echo "[v12.1-all] start $(date -Is)"
echo "[v12.1-all] branch $(git branch --show-current)"

.venv/bin/python scripts/run_v12_1_lexical_disentanglement.py "$@" 2>&1 | tee logs/v12_1/run_v12_1_lexical_disentanglement.log

echo "[v12.1-all] complete $(date -Is)"
