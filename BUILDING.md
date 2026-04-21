# BUILDING.md — What to run RIGHT NOW

## Active task — v8 GPU session prep

No GPU task currently running. Next GPU session plan is in `docs/NEXT_GPU_SESSION_v8.md`.

### What's ready to run (CPU, no GPU needed)

1. **Fetch Grid B .npz from HF** — `python scripts/fetch_from_hf.py --only v7_xz_grid --data-kind npz`
2. **Regenerate PCA horseshoe** — needs .npz files, then run CPU PCA scripts
3. **Regenerate SVD scree + cross-pair PC1 cosine heatmap** — same dependency

### What needs GPU

See `docs/NEXT_GPU_SESSION_v8.md` for the full plan:
- Priority 1+2: Direct sign classification + top-K tokens (~3 min)
- Priority 4: Cross-template transfer test (~5 min)

### Replot scripts (CPU, already working)

- `python scripts/plots_v7_behavioral.py` — all behavioral plots from v7 jsonl
- `python scripts/replot_v7_from_json.py` — all geometry plots from v7 JSON results
