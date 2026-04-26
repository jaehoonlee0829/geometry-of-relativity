#!/usr/bin/env bash
# v11 post-P2 master: wait for P2 sweep, fill gaps, run P3-P6, commit.
set -u
cd "$(dirname "$0")/.."
LOG=logs/v11/post_p2.log
mkdir -p logs/v11
echo "[$(date +%T)] post-P2 master starting" > "$LOG"

# Stage 0: wait for run_v11_p2_full.sh to exit
echo "[$(date +%T)] waiting for run_v11_p2_full.sh to exit..." | tee -a "$LOG"
while pgrep -f "run_v11_p2_full" >/dev/null; do sleep 8; done
echo "[$(date +%T)] main P2 sweep exited" | tee -a "$LOG"

# Stage 1: fill any missing pair extractions (with proper max_seq/bs)
PAIRS=(height age weight size speed wealth experience bmi_abs)
for MODEL in google/gemma-2-2b google/gemma-2-9b; do
  SHORT=$(echo "$MODEL" | sed 's|google/gemma-2-2b|gemma2-2b|; s|google/gemma-2-9b|gemma2-9b|')
  if [[ "$MODEL" == *9b* ]]; then BS=8; else BS=16; fi
  for PAIR in "${PAIRS[@]}"; do
    NPZ="results/v11/${SHORT}/${PAIR}/${SHORT}_${PAIR}_v11_residuals.npz"
    if [ -f "$NPZ" ]; then
      echo "[$(date +%T)] $SHORT/$PAIR: have NPZ, skip" | tee -a "$LOG"
      continue
    fi
    echo "[$(date +%T)] $SHORT/$PAIR: MISSING, re-running with default max_seq=288 bs=$BS..." | tee -a "$LOG"
    if .venv/bin/python scripts/vast_remote/extract_v11_dense.py \
         --model "$MODEL" --pair "$PAIR" --batch-size "$BS" \
         >> logs/v11/p2_${SHORT}_${PAIR}_rerun.log 2>&1; then
      .venv/bin/python scripts/upload_v11_to_hf.py \
        --model "$MODEL" --pair "$PAIR" --skip-existing \
        >> logs/v11/p2_${SHORT}_${PAIR}_rerun.log 2>&1 \
        && echo "[$(date +%T)]   re-run + upload ok" | tee -a "$LOG" \
        || echo "[$(date +%T)]   upload FAILED" | tee -a "$LOG"
    else
      echo "[$(date +%T)]   re-run extract FAILED — see logs/v11/p2_${SHORT}_${PAIR}_rerun.log" | tee -a "$LOG"
    fi
  done
done

# Stage 2: P3 analyses (CPU + some GPU)
echo "[$(date +%T)] === P3 analyses ===" | tee -a "$LOG"
for MS in gemma2-2b gemma2-9b; do
  echo "[$(date +%T)] P3a/b/c/d on $MS..." | tee -a "$LOG"
  .venv/bin/python scripts/analyze_v11_pca.py --model-short "$MS" --pair all \
    > logs/v11/p3a_${MS}.log 2>&1 || echo "p3a $MS failed" | tee -a "$LOG"
  .venv/bin/python scripts/analyze_v11_z_vs_lexical.py --model-short "$MS" --pairs all \
    > logs/v11/p3d_${MS}.log 2>&1 || echo "p3d $MS failed" | tee -a "$LOG"
  .venv/bin/python scripts/analyze_v11_increment_r2.py --model-short "$MS" --pair all \
    > logs/v11/p3c_${MS}.log 2>&1 || echo "p3c $MS failed" | tee -a "$LOG"
done

# Stage 3: P3e cross-pair transfer (GPU). 2B first, then 9B.
echo "[$(date +%T)] === P3e cross-pair transfer ===" | tee -a "$LOG"
for MS in gemma2-2b gemma2-9b; do
  if [[ "$MS" == "gemma2-9b" ]]; then BS=8; else BS=16; fi
  echo "[$(date +%T)] P3e on $MS (bs=$BS)..." | tee -a "$LOG"
  .venv/bin/python scripts/analyze_v11_cross_pair_transfer.py \
      --model-short "$MS" --batch-size "$BS" \
    > logs/v11/p3e_${MS}.log 2>&1 || echo "p3e $MS failed" | tee -a "$LOG"
done

# Stage 4: P4 SAE analysis (CPU after SAE download)
echo "[$(date +%T)] === P4 SAE ===" | tee -a "$LOG"
for MS in gemma2-2b gemma2-9b; do
  echo "[$(date +%T)] P4 SAE on $MS..." | tee -a "$LOG"
  .venv/bin/python scripts/analyze_v11_sae.py --model-short "$MS" --pair all \
    > logs/v11/p4_${MS}.log 2>&1 || echo "p4 $MS failed" | tee -a "$LOG"
done

# Stage 5: P5 head taxonomy + ablation (GPU). 2B uses canonical heads, 9B re-derives.
echo "[$(date +%T)] === P5 head taxonomy + ablation ===" | tee -a "$LOG"
for MS in gemma2-2b gemma2-9b; do
  echo "[$(date +%T)] P5 on $MS..." | tee -a "$LOG"
  .venv/bin/python scripts/analyze_v11_head_taxonomy_and_ablate.py \
      --model-short "$MS" \
    > logs/v11/p5_${MS}.log 2>&1 || echo "p5 $MS failed" | tee -a "$LOG"
done

# Stage 6: P6 critic round + writeup
echo "[$(date +%T)] === P6 critic round ===" | tee -a "$LOG"
.venv/bin/python scripts/run_v11_p6_critics.py \
  > logs/v11/p6_critics.log 2>&1 || echo "p6 critics failed" | tee -a "$LOG"

echo "[$(date +%T)] post-P2 master done." | tee -a "$LOG"
