# BUILDING.md — What to run RIGHT NOW

## Active task — none

v10 (DENSE-MANGO) shipped. See FINDINGS §14 and `docs/NEXT_GPU_SESSION_v10.md`.

## Next candidate tasks (pull from TODO.md when ready)

- Paper draft (ICML MI Workshop, May 8 deadline)
- v10 follow-ups, in priority order:
  1. R²(z | x) — properly orthogonalised increment R² (project out
     prior layer's z-direction before measuring "what this layer adds")
  2. SAE place-cell vs linear at L7 (where z-encoding emerges) instead
     of L20 (where it has saturated and been compressed)
  3. Causal verification of the v10 attention head taxonomy via
     ablation (requires fresh forwards: zero out L13h2 and watch
     R²(z) drop downstream)
