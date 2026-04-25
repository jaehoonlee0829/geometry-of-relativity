#!/usr/bin/env bash
# v11 P2 — extract + per-pair upload for all 8 pairs × 2 models. Sequential.
# Logs to logs/v11/p2_<model>_<pair>.log. Continues past per-pair failures
# (logs them to logs/v11/p2_failures.txt) so a single bad pair doesn't kill the sweep.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/v11
: > logs/v11/p2_failures.txt

PAIRS=(height age weight size speed wealth experience bmi_abs)
MODELS=(google/gemma-2-2b google/gemma-2-9b)

for MODEL in "${MODELS[@]}"; do
  SHORT=$(echo "$MODEL" | sed 's|google/gemma-2-2b|gemma2-2b|; s|google/gemma-2-9b|gemma2-9b|')
  # 9B at bs=16 may be tight; bs=8 is safe and lazy
  if [[ "$MODEL" == *9b* ]]; then BS=8; MAXSEQ=240; else BS=16; MAXSEQ=224; fi
  for PAIR in "${PAIRS[@]}"; do
    LOG="logs/v11/p2_${SHORT}_${PAIR}.log"
    echo ">>> [$(date +%H:%M:%S)] $SHORT / $PAIR  (bs=$BS, max_seq=$MAXSEQ)"
    if .venv/bin/python scripts/vast_remote/extract_v11_dense.py \
         --model "$MODEL" --pair "$PAIR" \
         --batch-size "$BS" --max-seq "$MAXSEQ" \
         > "$LOG" 2>&1; then
      echo "    extract ok ($(grep -E 'TOTAL elapsed|CELL-MEAN' "$LOG" | head -2 | tr '\n' '|'))"
      if .venv/bin/python scripts/upload_v11_to_hf.py \
           --model "$MODEL" --pair "$PAIR" --skip-existing \
           >> "$LOG" 2>&1; then
        echo "    upload ok"
      else
        echo "    UPLOAD FAILED (see $LOG)"
        echo "$SHORT/$PAIR upload" >> logs/v11/p2_failures.txt
      fi
    else
      echo "    EXTRACT FAILED (see $LOG)"
      echo "$SHORT/$PAIR extract" >> logs/v11/p2_failures.txt
    fi
  done
done

echo ">>> [$(date +%H:%M:%S)] P2 sweep done."
echo ">>> failures: $(wc -l < logs/v11/p2_failures.txt) — $(cat logs/v11/p2_failures.txt | tr '\n' ',' )"
