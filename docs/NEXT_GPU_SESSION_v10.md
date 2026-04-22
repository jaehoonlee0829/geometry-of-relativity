# Next GPU Session v10 — Dense Single-Pair Deep Dive (Height Only)

**Created:** Apr 22, 2026
**Philosophy:** v4–v9 spread across 8 pairs with a sparse 5×5 grid. The
dimensionality and manifold claims were underpowered (25 cell-means, unreliable
ID estimates). v10 goes deep instead of wide: **one pair (height), one model
(Gemma 2 2B), dense grid, all methods.**

---

## Why height only?

- Height has the cleanest behavioral signal (R = 0.85 on both E4B and 2B)
- The "tall"/"short" tokens are unambiguous (no polysemy like "adult")
- PC1 tracks z on clean grid (R² = 0.95) — the z-encoding is real
- Cross-template transfer was validated on height (97%, 44× null)
- Simplifies attention analysis: one semantic domain, one prompt template

## Grid design

**20 x-values × 20 z-values = 400 cells × 10 seeds = 4,000 prompts**

```python
# x: raw height in cm
x_values = np.linspace(145, 190, 20)  # 145, 147.4, ..., 190 cm (step ~2.4 cm)

# z: context-relative z-score
z_values = np.linspace(-3.0, +3.0, 20)  # -3.0, -2.68, ..., +3.0 (step ~0.32)

# For each (x, z) cell: derive mu = x - z*sigma, sigma = 10 cm (fixed)
# Generate 10 prompts per cell with different random seeds for context samples

# Total: 400 cells × 10 seeds = 4,000 prompts
# Compare: v7 had 25 cells × 30 seeds = 750 prompts
```

**Why 20×20?**
- 400 cell-means is enough for PCA variance (Gurnee used 150), TWO-NN
  (reliable up to ID ≈ 8), and Gram analysis
- 20 z-values (step 0.32) is dense enough to distinguish place-cell bumps
  from monotonic features — if a bump has width ~1σ, we have ~6 points
  within it
- z ∈ [−3, +3] is wider than before (was ±2) — captures the tails where
  behavior might change

**Why 10 seeds (not 30)?**
- With 400 cells we have plenty of points for manifold analysis
- 4,000 prompts is manageable for attention extraction
- Can always add more seeds later if needed

---

## Priority 1: Behavioral confirmation + all-layer activation extraction

**GPU time:** ~15 min

```python
MODEL = "google/gemma-2-2b"
# Extract at ALL 26 layers (residual stream after each decoder block)
# Save: activations (4000, 26, 2304), ids, logit_diff, entropy, attention weights

# For attention: save attention patterns at strategic layers
# Layers: 0, 3, 7, 10, 13, 17, 20, 25 (8 layers × all heads)
# Save: attn_weights shape (4000, n_heads, seq_len, seq_len) at each layer
```

**Acceptance criteria:**
- R > 0.5 for height on the 20×20 grid (expect ~0.85)
- logit_diff heatmap shows clear z-gradient

---

## Priority 2: Dimensionality — three methods on 400 cell-means

**CPU analysis after extraction.** For each of the 26 layers:

### Method A: PCA variance explained (Gurnee-style)

```python
# 400 cell-means, each 2304-dim
pca = PCA().fit(cell_means_400)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_dims_95 = np.searchsorted(cumvar, 0.95) + 1
n_dims_99 = np.searchsorted(cumvar, 0.99) + 1
```

Report: n_dims at 80%, 90%, 95%, 99% variance, per layer.
Plot: cumulative variance curves at 5 strategic layers (L0, L7, L13, L20, L25).

### Method B: TWO-NN intrinsic dimensionality

```python
# Now with 400 points instead of 25 — reliable up to ID ≈ 8
id_estimate = two_nn_id(cell_means_400)
```

Report: ID per layer. Compare to v9's 25-point estimates. If the hunchback
survives with 400 points, it's real.

### Method C: Gram matrix of per-z-value probes

```python
# With 20 z-values: train 20 one-vs-rest logistic probes
# OR: train a single 20-class softmax probe, use weight matrix rows
# Gram G = W_normalized @ W_normalized.T  (20 × 20)
# Eigendecompose → participation ratio
```

With K=20 probes (vs K=5 before), the one-vs-rest artifact is diluted
because only 2/20 values are "extreme." The middle 16 values all need
to carve out a band, making them more comparable.

Also try: **single multiclass softmax** (Sarfati's approach) — avoids
the one-vs-rest problem entirely.

### Outputs

```
figures/v10/pca_variance_per_layer.png       — cumvar curves at 5 layers
figures/v10/id_comparison_25vs400.png        — TWO-NN: v9 (25pts) vs v10 (400pts)
figures/v10/gram_eigenvalues_20probe.png     — per-layer Gram spectrum
figures/v10/dimensionality_3methods.png      — PCA-95% vs TWO-NN vs Gram PR, per layer
```

---

## Priority 3: SAE feature analysis with dense z

**CPU analysis.** With 20 z-values (step 0.32) we can finally distinguish
place-cell bumps from monotonic features:

```python
# For each SAE feature, plot activation vs z (20 points)
# Fit two models:
#   Linear: activation = a*z + b         (r² metric)
#   Gaussian bump: activation = A * exp(-(z-z0)²/2σ²)   (r² metric)
# If bump r² >> linear r² → place cell
# If linear r² >> bump r² → monotonic feature
```

Also: with 400 cell-means, we can do PCA in SAE coefficient space
and compare to raw PCA (§12.1 showed SAE PCA was worse — does it
improve with more data?).

### Outputs

```
figures/v10/sae_placecell_vs_linear_20z.png  — activation profiles, top-10 features
figures/v10/sae_pca_vs_raw_pca_dense.png     — SAE PCA with 400 pts vs raw PCA
```

---

## Priority 4: Attention head analysis — how does the model compute z?

**The big mechanistic question:** The model sees 15 context heights + 1 target
height. To compute z = (x − μ) / σ, it needs to:
1. Aggregate context values → estimate μ
2. Compare target x to μ
3. Maybe estimate σ (or use a fixed prior)

Which attention heads do each step, and at which layers?

### Analysis approach

```python
# For each attention head at each layer:
# 1. Attention pattern from last token (the prediction position)
#    to context tokens (positions of the 15 heights)
#    → does this head "read" the context?

# 2. Correlation of head output (OV projection) with z
#    → does this head's output carry z-information?

# 3. Ablation: zero out this head's output
#    → does R²(z) at subsequent layers drop?

# Group heads into functional categories:
#   "Context aggregators" — attend broadly to context numbers
#   "Comparators" — attend to both context and target
#   "Passthrough" — don't attend to numeric tokens
```

### Specific experiments

**4a: Attention to context vs target**

```python
# For each head, compute:
attn_to_context = attn_weights[:, head, -1, context_positions].mean()
attn_to_target = attn_weights[:, head, -1, target_position].mean()
# Plot: attn_to_context vs layer, colored by head
```

**4b: Head output correlation with z**

```python
# Per head, project activations through OV matrix
head_output = attn_output[:, head, :]  # (4000, d_head)
r2_z = cv_r2(head_output, z_scores)
# Plot: r2_z per head per layer → which heads "know" z?
```

**4c: Does attention shift between phases?**

Prediction: L0–L7 heads attend to context (computing μ). L7–L13 heads
attend to target AND context (comparing x to μ). L13+ heads attend to
the comparison result (routing to output).

### Outputs

```
figures/v10/attention_to_context_per_layer.png
figures/v10/head_r2z_heatmap.png              — n_heads × n_layers, colored by R²(z)
figures/v10/attention_phase_transition.png     — context vs target attention by layer
```

---

## Priority 5: Layer sweep replication on dense grid

Redo the v9 §13 analysis with 400 cell-means instead of 25:

- CV R²(z) per layer (should match — computed on all 4000 prompts)
- PCA-95% dimensionality per layer (new — wasn't reliable before)
- TWO-NN ID per layer (now with 400 points — reliable)
- ‖primal_z‖ per layer (should match)
- cos(primal_z[L], primal_z[L-1]) per layer (should match)
- Steering at strategic layers (needs GPU forward passes)

**Key question:** Does the hunchback survive with 400 cell-means?

---

## Priority 6: Residual stream decomposition — who writes z?

Track z-information in the **layer increment** (what each layer adds)
vs the **residual** (cumulative):

```python
# residual_L = output after layer L (cumulative)
# increment_L = residual_L - residual_{L-1} (this layer's contribution)

r2_residual_L = cv_r2(residual_L, z)      # can we read z from cumulative?
r2_increment_L = cv_r2(increment_L, z)    # does THIS layer actively write z?
```

Prediction:
- r2_residual saturates at L7 (matches v9)
- r2_increment peaks at L3–L7 (layers actively computing z), drops
  to near zero at L8–L12 (z persists but nothing new written), then
  maybe rises again at L13–L17 (re-encoding z in the direction the
  late layers will read)

---

## Summary

| # | Experiment | GPU? | Time |
|---|-----------|------|------|
| 1 | Extract 4000 prompts × 26 layers + attention at 8 layers | GPU | 15 min |
| 2 | Dimensionality: PCA / TWO-NN / Gram on 400 cell-means | CPU | 10 min |
| 3 | SAE place-cell vs linear on 20 z-values | CPU | 10 min |
| 4 | Attention head analysis | CPU | 20 min |
| 5 | Layer sweep replication | CPU + GPU | 15 min |
| 6 | Residual decomposition (increment R²) | CPU | 10 min |

**Total: ~30 min GPU, ~50 min CPU**

## Key questions v10 answers

1. **Is the hunchback real?** ID on 400 points vs 25 — does it survive?
2. **Are z-features place-cells or linear?** 20 z-values can distinguish bumps from ramps
3. **How does the model compute z?** Attention analysis reveals the mechanism
4. **Which layers write z?** Increment R² shows active computation vs passthrough
5. **Does PCA dimensionality match TWO-NN?** Three methods on the same data

## What v10 does NOT do

- No cross-pair analysis (height only)
- No steering robustness (already done in v9 §11)
- No Park/Fisher metrics (already refuted)
- No cross-template transfer (already validated)
