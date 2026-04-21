# BUILDING.md — What to run RIGHT NOW

## Active task — v9 GPU session: Gemma 2 SAE decomposition

See `docs/NEXT_GPU_SESSION_v9.md` for full plan.

### Execution order

| # | What | GPU? | Time |
|---|------|------|------|
| 1 | Replicate 8-pair behavioral signal on `google/gemma-2-2b` | GPU | 5 min |
| 2 | Load Gemma Scope SAE (`google/gemma-scope-2b-pt-res`, layer 20, 65k width) | CPU | 1 min |
| 3 | Encode Grid B activations → sparse SAE coefficients | CPU | 2 min |
| 4 | Find z-correlated SAE features per pair, cross-pair overlap | CPU | 5 min |
| 5 | Place-cell vs linear feature analysis | CPU | 5 min |
| 6 | Decompose primal_z vs probe_z in SAE basis | CPU | 2 min |
| 7 | On-manifold steering (geodesic tangent vs fixed primal_z) | GPU | 5 min |
| 8 | Park's causal inner product steering test | GPU | 5 min |

### What's already done (CPU, no GPU needed)

- Manifold geometry analysis complete: ID ~5-D, speed has massive curvature (isomap R²=0.97 vs PCA R²=0.01), primal_z is layer-specific (mid ⊥ late)
- All v7/v8 figures regenerated
- Grid B .npz activations fetched from HF
- meta_w1 sign bug fixed in all scripts (JSON needs re-generation on GPU)

### Key scientific questions

1. Is z a single SAE feature, place-cells, or distributed?
2. Do the same SAE features fire across pairs? (shared mechanism test)
3. Does SAE-based steering cause less entropy damage than primal_z?
4. Can Park's causal metric bridge the probe-vs-steering gap?
