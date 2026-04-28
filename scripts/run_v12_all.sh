#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/v12 results/v12 figures/v12

set -a
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
export WANDB_PROJECT="${WANDB_PROJECT:-geometry-of-relativity}"

echo "[v12-all] start $(date -Is)"
echo "[v12-all] branch $(git branch --show-current)"

echo "[v12-all] CPU pass before GPU"
.venv/bin/python scripts/analyze_v12_cpu.py 2>&1 | tee logs/v12/analyze_v12_cpu_pre.log

echo "[v12-all] GPU pass"
.venv/bin/python scripts/run_v12_gpu.py --sections all --batch-size 12 --layer-sweep-prompts 160 --redteam-prompts 160 --transfer-prompts 72 2>&1 | tee logs/v12/run_v12_gpu.log

echo "[v12-all] CPU pass after GPU"
.venv/bin/python scripts/analyze_v12_cpu.py 2>&1 | tee logs/v12/analyze_v12_cpu_post.log

echo "[v12-all] SAE lexical audit"
.venv/bin/python scripts/analyze_v12_sae_lexical_audit.py 2>&1 | tee logs/v12/analyze_v12_sae_lexical_audit.log

echo "[v12-all] complete $(date -Is)"
