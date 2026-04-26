#!/usr/bin/env bash
# v11.5 master: run §A through §I in parallel where possible, sequential where
# they need the GPU. Logs under logs/v11_5/.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/v11_5
LOG=logs/v11_5/master.log
echo "[$(date +%T)] v11.5 master starting" > "$LOG"

# === CPU stage 1 (parallel — no GPU dependency) ===
echo "[$(date +%T)] === CPU stage 1: §B perm-null, §F bootstrap CIs, §H P3d-widened, §G P3c fold-aware ===" | tee -a "$LOG"
(
  for MS in gemma2-2b gemma2-9b; do
    .venv/bin/python scripts/analyze_v11_5_perm_null_taxonomy.py --model-short "$MS" \
      > logs/v11_5/B_perm_${MS}.log 2>&1 &
    .venv/bin/python scripts/analyze_v11_5_bootstrap_cis.py --model-short "$MS" --pair all \
      > logs/v11_5/F_boot_${MS}.log 2>&1 &
    .venv/bin/python scripts/analyze_v11_5_p3d_widened.py --model-short "$MS" --pair all \
      > logs/v11_5/H_p3d_${MS}.log 2>&1 &
    .venv/bin/python scripts/analyze_v11_5_p3c_fold_aware.py --model-short "$MS" --pair all \
      > logs/v11_5/G_p3c_${MS}.log 2>&1 &
  done
  wait
) &
CPU1_PID=$!
echo "[$(date +%T)]   CPU stage 1 launched (pid=$CPU1_PID)" | tee -a "$LOG"

# === GPU stage 1: §A shared-z (small) — start IN PARALLEL with CPU stage 1 ===
echo "[$(date +%T)] === GPU stage 1: §A shared-z (2B, then 9B) ===" | tee -a "$LOG"
.venv/bin/python scripts/analyze_v11_5_shared_z.py --model-short gemma2-2b --batch-size 16 \
  > logs/v11_5/A_shared_gemma2-2b.log 2>&1 \
  || echo "A 2B failed" | tee -a "$LOG"
.venv/bin/python scripts/analyze_v11_5_shared_z.py --model-short gemma2-9b --batch-size 8 \
  > logs/v11_5/A_shared_gemma2-9b.log 2>&1 \
  || echo "A 9B failed" | tee -a "$LOG"

# === GPU stage 2: §I joint head-set ablation ===
echo "[$(date +%T)] === GPU stage 2: §I joint head-set ablation ===" | tee -a "$LOG"
.venv/bin/python scripts/analyze_v11_5_joint_ablation.py --model-short gemma2-2b --batch-size 16 \
  > logs/v11_5/I_joint_abl_gemma2-2b.log 2>&1 \
  || echo "I 2B failed" | tee -a "$LOG"
.venv/bin/python scripts/analyze_v11_5_joint_ablation.py --model-short gemma2-9b --batch-size 8 \
  > logs/v11_5/I_joint_abl_gemma2-9b.log 2>&1 \
  || echo "I 9B failed" | tee -a "$LOG"

# === GPU stage 3: §C/§D multi-seed transfer (the biggest GPU job) ===
echo "[$(date +%T)] === GPU stage 3: §C/§D multi-seed cross-pair transfer ===" | tee -a "$LOG"
.venv/bin/python scripts/analyze_v11_5_multiseed_transfer.py --model-short gemma2-2b --seeds 5 --batch-size 16 \
  > logs/v11_5/CD_transfer_gemma2-2b.log 2>&1 \
  || echo "C/D 2B failed" | tee -a "$LOG"
.venv/bin/python scripts/analyze_v11_5_multiseed_transfer.py --model-short gemma2-9b --seeds 5 --batch-size 8 \
  > logs/v11_5/CD_transfer_gemma2-9b.log 2>&1 \
  || echo "C/D 9B failed" | tee -a "$LOG"

# Wait for CPU stage 1
echo "[$(date +%T)] waiting for CPU stage 1 to finish..." | tee -a "$LOG"
wait "$CPU1_PID" 2>/dev/null || true

# === CPU stage 2: §E SAE token-freq control (after multiseed since SAE download is slow) ===
echo "[$(date +%T)] === CPU stage 2: §E SAE token-freq control ===" | tee -a "$LOG"
for MS in gemma2-2b gemma2-9b; do
  .venv/bin/python scripts/analyze_v11_5_sae_token_freq.py --model-short "$MS" --pair all \
    > logs/v11_5/E_sae_${MS}.log 2>&1 \
    || echo "E $MS failed" | tee -a "$LOG"
done

echo "[$(date +%T)] v11.5 master DONE" | tee -a "$LOG"
ls -la results/v11_5/gemma2-2b/ results/v11_5/gemma2-9b/ 2>&1 | tee -a "$LOG"
