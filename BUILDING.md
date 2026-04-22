# BUILDING.md — What to run RIGHT NOW

## Active task — v10 GPU session: Dense single-pair deep dive

See `docs/NEXT_GPU_SESSION_v10.md` for full plan.

**Scope:** Height only, Gemma 2 2B, 20×20 grid (400 cells × 10 seeds = 4,000 prompts).
Fix the underpowered dimensionality estimates from v9 (25 cell-means → 400).

### Key questions

1. Does the ID hunchback survive with 400 cell-means? (vs 25 in v9)
2. Are z-features place-cells or linear? (20 z-values can distinguish)
3. How does the model compute z? (attention head analysis)
4. Which layers actively write z? (increment R² decomposition)
5. Do three dimensionality methods (PCA-95%, TWO-NN, Gram) agree?

### Execution order

| # | What | GPU? | Time |
|---|------|------|------|
| 1 | Extract 4000 prompts × 26 layers + attention | GPU | 15 min |
| 2 | Dimensionality: PCA / TWO-NN / Gram on 400 cells | CPU | 10 min |
| 3 | SAE place-cell vs linear on 20 z-values | CPU | 10 min |
| 4 | Attention head analysis | CPU | 20 min |
| 5 | Layer sweep replication on dense grid | CPU+GPU | 15 min |
| 6 | Residual increment decomposition | CPU | 10 min |
