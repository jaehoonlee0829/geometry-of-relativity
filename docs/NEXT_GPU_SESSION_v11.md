# Next GPU Session v11 — Go Wide, Deep, and Cross-Model

**Created:** Apr 25, 2026
**Philosophy:** v10 went deep on one pair (height, 400 cells, all layers,
attention taxonomy). v9 went wide but sparse (8 pairs, 25 cells each).
v11 combines both: **dense 20x20 grid on all 8 adjective pairs on THREE
models** (Gemma 2 2B, Gemma 2 9B, Gemma 3 4B), plus follow-up analyses that were
flagged in FINDINGS §14.7 but never run because the v10 NPZs were lost
when the Vast.ai instance died.

---

## Models

| Model | HuggingFace ID | Params | VRAM (bf16) | Layers | d_model | SAE |
|---|---|---|---|---|---|---|
| Gemma 2 2B | `google/gemma-2-2b` | 2B | ~5 GB | 26 | 2304 | `google/gemma-scope-2b-pt-res` (65k, all layers) |
| Gemma 2 9B | `google/gemma-2-9b` | 9B | ~18 GB | 42 | 3584 | `google/gemma-scope-9b-pt-res` (65k, all layers) |
| Gemma 3 4B | `google/gemma-3-4b-pt` | 4B | ~9 GB | 36 | 2560 | `google/gemma-scope-2-4b-pt` (verify before session) |

All three fit on 1x H100 80GB (even simultaneously: 5+18+9 = 32 GB).
Run models sequentially to maximize activation cache memory.

**Pre-flight check:** Before renting GPU, verify Gemma 3 4B SAE exists:
```python
from huggingface_hub import list_repo_files
files = list_repo_files("google/gemma-scope-2-4b-pt")
print([f for f in files if "layer_20" in f][:5])
```
If no SAE exists for Gemma 3 4B, drop it and run only Gemma 2 2B + 9B
(both have confirmed SAEs).

---

## Figure Organization

All v11 figures go into **topic subdirectories** for easier review:

```
figures/v11/
  behavioral/          — heatmaps, marginals per pair per model
  pca/                 — 2D/3D scatter per pair, horseshoe analysis
  probing/             — increment R², orthogonalized R²
  steering/            — layer sweep, cross-pair transfer matrices
  attention/           — head heatmaps, DLA, taxonomy
  sae/                 — feature profiles, linear vs bump, L7 vs L20
  disentanglement/     — z-primal vs lexical-primal, cross-transfer
  cross_model/         — side-by-side Gemma 2 2B vs Gemma 3 4B comparisons
```

---

## Priority 1: Re-extract v10 + upload (GPU, ~2 min)

The v10 NPZs (~826 MB) were lost when the Vast.ai instance died.

```bash
python scripts/vast_remote/extract_v10_dense_height.py   # ~18 sec
python scripts/upload_v10_to_hf.py                       # ~1 min
```

**Acceptance:** `python scripts/fetch_from_hf.py --only v10` succeeds.

---

## Priority 2: Dense 20x20 grid — ALL 8 pairs × 2 models (GPU, ~40 min)

### Grid design

**Per pair per model: 20 x-values × 20 z-values × 10 seeds = 4,000 prompts**
**Total: 8 pairs × 3 models × 4,000 = 96,000 prompts**

```python
PAIRS = [
    "height",      # tall/short  — x in cm,   sigma=10
    "age",         # old/young   — x in years, sigma=10
    "weight",      # heavy/light — x in kg,   sigma=15
    "size",        # large/small — x in cm,   sigma=20
    "speed",       # fast/slow   — x in km/h, sigma=15
    "wealth",      # rich/poor   — x in $k,   sigma=30
    "experience",  # experienced/inexperienced — x in years, sigma=5
    "bmi_abs",     # obese/not obese — x in kg/m², sigma=5 (absolute adjective)
]

MODELS = [
    "google/gemma-2-2b",    # 26 layers, d=2304, ~18 sec/pair
    "google/gemma-2-9b",    # 42 layers, d=3584, ~60 sec/pair (est.)
    "google/gemma-3-4b-pt", # 36 layers, d=2560, ~30 sec/pair (est.)
]
```

### What to extract

```python
# Residuals: all layers (26 for 2B, 36 for 4B)
# Shape per pair: (4000, n_layers, d_model)

# Attention: 8 strategic layers per model
ATTN_LAYERS_2B = [0, 3, 7, 10, 13, 17, 20, 25]
ATTN_LAYERS_9B = [0, 5, 10, 16, 21, 28, 34, 41]  # scale proportionally
ATTN_LAYERS_4B = [0, 4, 9, 14, 18, 24, 30, 35]   # scale proportionally

# Behavioral: next_logit_diff, next_entropy, next_argmax per prompt
```

### Time estimate

| Step | Per pair | 8 pairs | Notes |
|---|---|---|---|
| Gemma 2 2B extraction | ~20 sec | ~3 min | Proven in v10 |
| Gemma 2 2B upload | ~30 sec | ~4 min | ~826 MB/pair |
| Gemma 2 9B extraction | ~60 sec (est.) | ~8 min | ~4.5x params, larger d |
| Gemma 2 9B upload | ~45 sec | ~6 min | ~1.3 GB/pair (d=3584) |
| Gemma 3 4B extraction | ~40 sec (est.) | ~6 min | ~2x params |
| Gemma 3 4B upload | ~30 sec | ~4 min | Similar size to 2B |
| **Subtotal** | | **~31 min** | |

### Upload strategy

Upload after EACH pair completes (not after all 8):
```bash
python scripts/upload_v11_to_hf.py --model gemma-2-2b --pair height
python scripts/upload_v11_to_hf.py --model gemma-3-4b --pair height
# ... repeat for each pair
```

**Acceptance criteria:**
- All 24 NPZ files (8 pairs × 3 models) on HuggingFace
- corr(LD, z) > 0.5 for each pair on each model
- Total disk: ~25 GB (8 × ~826 MB + 8 × ~1.3 GB + 8 × ~900 MB)

---

## Priority 3: CPU analyses (run after P2, no GPU needed)

### P3a: PCA scatter plots (2D and 3D) — CPU, ~15 min

For each pair × model, PCA the 400 cell-mean activations at L20 (2B) / L30 (4B):

```python
pca = PCA(n_components=3).fit(cell_means_400)
# 2D: PC1 vs PC2, colored by z — expect horseshoe
# 3D: PC1 vs PC2 vs PC3
# Report: R²(PC1 ~ z), R²(PC1 ~ x), R²(PC2 ~ z²), R²(PC2 ~ x)
```

**Key question:** Does every pair show the horseshoe? Is PC1 ~ z universal?

```
figures/v11/pca/{pair}_{model}_2d_L20.png    — 16 plots
figures/v11/pca/{pair}_{model}_3d_L20.png    — 16 plots
results/v11/pca_r2_summary.json
```

---

### P3b: PC1 vs primal_z cosine — CPU, ~10 min

```python
# At each layer L, for each pair:
cos_pc1_primal = cosine(pc1_direction[L], primal_z[L])
```

**Key question:** Is PC1 ≈ primal_z (cos > 0.9)? If not, what is PC1 capturing?

```
figures/v11/probing/cos_pc1_primal_per_layer_{model}.png
```

---

### P3c: Proper increment R² (orthogonalized) — CPU, ~15 min

```python
# At layer L:
# 1. w_prev = ridge_weights(h_{L-1}, z)
# 2. h_L_orth = h_L - (h_L @ w_prev) * w_prev / ||w_prev||²
# 3. R²_new(z) = cv_r2(h_L_orth, z)
```

**Prediction:** Signal concentrated at L3-L7 (encoding) and maybe L13-L17
(re-encoding for readout), near-zero elsewhere.

```
figures/v11/probing/increment_r2_naive_vs_orth_{model}.png
results/v11/orthogonalized_increment_r2.json
```

---

### P3d: z-direction vs lexical direction (KEY EXPERIMENT) — CPU, ~20 min

**"Is primal_z a z-score concept, or just a tall-minus-short vector?"**

```python
# For each pair at L20:
z_primal     = mean(h[z > 0])       - mean(h[z < 0])        # grouped by z
lexical_prim = mean(h[pred=="tall"]) - mean(h[pred=="short"]) # grouped by output

cos_zl = cosine(z_primal, lexical_prim)
# cos ~ 1.0 → z IS just the lexical direction (bad for paper)
# cos < 0.8 → z carries info beyond output word identity (good)
```

**Cross-pair transfer variant:**
```python
# Steer age prompts with height's z-primal vs height's lexical-primal
# If z-primal transfers but lexical doesn't → z encodes domain-general relativity
```
Note: the transfer variant **requires GPU** (steering = forward passes).
The cosine computation is CPU-only.

```
figures/v11/disentanglement/z_vs_lexical_cosine_per_pair.png
figures/v11/disentanglement/z_vs_lexical_cross_transfer.png   # GPU-dependent
results/v11/z_vs_lexical_disentanglement.json
```

---

### P3e: Cross-pair transfer on dense grid — **GPU, ~10 min**

**This requires GPU** (steering = injecting vectors during forward passes).

```python
# For each (pair_A, pair_B) with A != B:
# 1. Compute primal_z from pair A activations (CPU)
# 2. Steer pair B's 400 seed=0 prompts at α ∈ {-4, 0, +4} (GPU)
# 3. Measure Δlogit_diff slope

# 8×7 = 56 direction-target combinations × 3 alphas × 400 prompts
# = 67,200 forward passes — batch at 400, ~170 batches, ~5-10 min
```

v9 found ~40% transfer. Dense grid should give a definitive answer.

```
figures/v11/steering/cross_pair_transfer_8x8_{model}.png
results/v11/cross_pair_transfer_dense.json
```

---

## Priority 4: SAE analysis — L7 vs L20, cross-model — CPU, ~15 min

```python
# For each model, load SAEs at L7 and L20
# For each SAE feature, compute activation profile vs z (20 z-values)
# Compare:
#   - Monotonic vs place-cell ratio at L7 vs L20
#   - Feature overlap (Jaccard) between L7 and L20 top features
#   - Cross-pair feature overlap at L20 (improved from v9's 6% Jaccard)
```

```
figures/v11/sae/L7_vs_L20_feature_profiles_{model}.png
figures/v11/sae/L7_placecell_vs_linear_{model}.png
figures/v11/sae/cross_pair_feature_overlap_{model}.png
results/v11/sae_L7_L20_overlap.json
```

---

## Priority 5: Causal head ablation — GPU, ~5 min

Zero out specific heads and measure downstream R²(z) drop:

```python
# A: Zero L13h2 (comparator)   → predict: R²(z) drops at L14+
# B: Zero L3h0  (early writer)  → predict: drop at L4-L7, recovery by L13+
# C: Zero L0h6  (μ-aggregator)  → predict: cascading failure (no μ → no z)
```

3 ablations × 4000 prompts × 1 forward pass each ≈ ~1 min per model (×3 models = ~3 min).

```
figures/v11/attention/ablation_{head}_{model}.png
results/v11/head_ablation_causal.json
```

---

## Priority 6: Critic round (CPU, ~30 min)

After all experiments complete, spawn **3-5 critic agents** that:

1. **Methodology critic**: Check for data leakage, circular reasoning,
   selection bias in thresholds (e.g., the L≥10 z-writer filter), and
   whether CV folds properly separate seeds within cells.

2. **Alternative-explanation critic**: For each positive finding, propose
   the simplest alternative explanation that could produce the same result.
   E.g., "cross-pair transfer at 40% could just be shared token embeddings
   for number words, not a domain-general z-code."

3. **Statistical critic**: Check effect sizes, multiple-comparisons
   corrections, whether N=400 cells is enough for the dimensionality
   claims, confidence intervals on all R² values.

4. **Novelty critic**: Compare findings against existing mech-interp
   literature (Anthropic's manifold work, Goodfire/Sarfati, Park,
   Neel Nanda's IOI work). What's genuinely new vs already known?

5. **Narrative critic**: Read FINDINGS §14 + new v11 results and check
   whether the story is internally consistent. Flag contradictions
   between v9/v10/v11 claims.

Each critic produces a `results/v11/critic_{name}.md` report. Disagreements
are surfaced for human review before writing FINDINGS §15.

---

## Summary

| # | Experiment | GPU? | Est. time | Model |
|---|-----------|------|-----------|-------|
| 1 | Re-extract v10 NPZs + upload | GPU | 2 min | 2B only |
| 2 | Dense 20×20 × 8 pairs × 3 models | GPU | 31 min | All 3 |
| 3a | PCA scatter (2D/3D) | CPU | 20 min | All 3 |
| 3b | PC1 vs primal_z cosine | CPU | 10 min | All 3 |
| 3c | Orthogonalized increment R² | CPU | 15 min | All 3 |
| 3d | z vs lexical direction (cosine) | CPU | 20 min | All 3 |
| 3d' | z vs lexical cross-transfer | GPU | 15 min | All 3 |
| 3e | Cross-pair transfer (steering) | GPU | 15 min | All 3 |
| 4 | SAE L7 vs L20, cross-model | CPU | 20 min | All 3 |
| 5 | Causal head ablation | GPU | 5 min | All 3 |
| 6 | Critic agents | CPU | 30 min | — |

**Total GPU: ~70 min** (three models, all experiments)
**Total CPU: ~2.5 h** (analyses + critics)
**Estimated Vast.ai cost: ~$5-6** (1x H100 for ~2 h)

## Key questions v11 answers

1. **Is primal_z a z-score or just a lexical vector?** P3d directly tests
   this with cosine decomposition + cross-pair transfer.
2. **Does the z-code generalize across domains?** P3e with 400 cells per
   pair gives a definitive answer.
3. **Does z generalize across models?** P2 on Gemma 2 9B and Gemma 3 4B
   tests whether the encode-vs-use story, attention taxonomy, and SAE
   structure replicate across model scale (2B→9B) and family (Gemma 2→3).
4. **Does each layer add NEW z-info?** P3c orthogonalized increment R²
   distinguishes active computation from passive carry-forward.
5. **Is the head taxonomy causal?** P5 ablation tests whether L13h2 and
   L0h6 are necessary, not just correlated.
6. **How does z-encoding look at birth (L7) vs maturity (L20)?** P4.

## What v11 does NOT do

- No Fisher-information analysis (Park metric already refuted in v7)
- No new prompt templates (cross-template transfer already validated in v9)
- No tangent/manifold steering (tested in v9, 70% of primal, not worth re-running)
- No Gemma 4 models (no SAEs available for Gemma 4 family)
- No models beyond ~9B (diminishing returns for an MI workshop paper)
