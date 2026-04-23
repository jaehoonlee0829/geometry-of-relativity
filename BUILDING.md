# BUILDING.md — What to run RIGHT NOW

## Active task — none

v10 dense-height deep dive shipped (completion-promise word: **DENSE-MANGO**).
See FINDINGS §14 and `docs/NEXT_GPU_SESSION_v10.md` for the original plan.

## Next candidate tasks (pull from TODO.md when ready)

- **One-shot reproducibility upload (≤1 min, requires write HF token)**
    ```
    python scripts/upload_v10_to_hf.py
    ```
    The v10 NPZs (~826 MB) are not yet on
    `xrong1729/mech-interp-relativity-activations`. Until they are,
    `python scripts/fetch_from_hf.py --only v10` will hit an empty
    folder and a fresh `git pull` cannot re-run the v10 analyses
    without re-doing the GPU extraction. The token used by the rest
    of v10 was read-only.
- Paper draft (ICML MI Workshop, May 8 deadline)
- v10 follow-ups, in priority order:
  1. R²(z | x) — properly orthogonalised increment R² (project out
     prior layer's z-direction before measuring "what this layer adds")
  2. SAE place-cell vs linear at L7 (where z-encoding emerges) instead
     of L20 (where it has saturated and been compressed)
  3. Causal verification of the v10 attention head taxonomy via
     ablation (requires fresh forwards: zero out L13h2 and watch
     R²(z) drop downstream)
