# Next GPU Session v9 — Gemma 2 SAE Decomposition + Manifold Geometry

**Created:** Apr 22, 2026
**Context:** v7-v8 established the behavioral and geometric findings on Gemma 4 E4B
with clean Grid B. The key unexplained gap: primal_z steers 18x stronger than
probe_z, but we don't know WHY mechanistically. SAE decomposition on Gemma 2
(which has pretrained Gemma Scope SAEs) can answer this.

**New from manifold analysis (Apr 22, CPU-only on fetched .npz):**
- Intrinsic dimensionality of z-representation: ~5-D (NOT 1-D curve)
- Speed: isomap R²(z)=0.97 vs PCA R²(z)=0.01 — massive curvature, linear methods miss z entirely
- primal_z is fully layer-specific: cos(primal_z_mid, primal_z_late) ≈ 0.00 across all 8 pairs
- Both layers decode z with R² > 0.95, but using completely orthogonal directions

---

## Scientific questions

1. **Is z-score a single sparse feature or distributed?** If a single SAE feature
   activates proportionally to z, that's cleaner than primal_z. If z is tiled by
   discrete "place-cell" features (as Anthropic found for character counts), the
   model discretizes the context-relative spectrum.

2. **Do the same SAE features fire for z across pairs?** If feature #4821 fires
   for "tall relative to context" AND "heavy relative to context," that's direct
   evidence of a shared mechanism — much stronger than cos(PC1_i, PC1_j) = 0.19.

3. **Does SAE-based steering work better?** Clamping specific z-features should
   steer with less entropy damage than adding α·primal_z.

4. **Does the z-manifold have curvature?** Isomap vs PCA comparison; intrinsic
   dimensionality estimation; on-manifold vs off-manifold steering.

---

## Why Gemma 2 2B (not Gemma 4 E4B)

- **Gemma Scope** provides pretrained JumpReLU SAEs for every layer of Gemma 2 2B
  (26 layers, d=2048). No SAE training needed.
- Weights: `google/gemma-scope-2b-pt-res` on HuggingFace (residual stream, all layers)
- Available widths: 16k, 65k, 262k, 1M features
- Our v0/v1 experiments used Gemma 2 2B — behavioral signal exists there
- Model fits easily on any GPU (even T4)

Gemma Scope 2 (Dec 2025) covers Gemma 3 models (270M to 27B) with even more
SAE variants (residual, MLP, attention, transcoders). Could also use Gemma 3 4B
(`google/gemma-scope-2-4b-pt`) which is closer to E4B in scale.

---

## Priority 1: Replicate behavioral signal on Gemma 2 2B

**Estimated time:** ~5 min on any GPU

Run 8-pair Grid B prompts on `google/gemma-2-2b`. Extract:
- logit_diff per prompt (behavioral signal)
- Activations at late layer (~layer 20 of 26)

**Acceptance criteria:**
- Relativity ratio R > 0.3 for at least 5/8 pairs
- logit_diff heatmap shows z-gradient

If signal is absent on Gemma 2 2B, fall back to Gemma 3 4B + Gemma Scope 2.

### Script outline

```python
MODEL_ID = "google/gemma-2-2b"
LAYER_INDICES = {"mid": 13, "late": 20}  # 26 layers total, d=2048

# Reuse extract_v7_xz_grid.py logic with different model + layers
# Save: results/v9_gemma2/e4b_{pair}_{layer}.npz + logits.jsonl
```

---

## Priority 2: SAE feature decomposition of z

**Estimated time:** ~10 min GPU (forward pass + SAE encode), then CPU analysis

### Step 2a: Load Gemma Scope SAE

```python
from huggingface_hub import hf_hub_download
import torch

# Load 65k-width SAE for layer 20
sae_path = hf_hub_download(
    "google/gemma-scope-2b-pt-res",
    "layer_20/width_65k/average_l0_71/params.npz"
)
params = np.load(sae_path)
W_enc = params["W_enc"]   # (d_model, n_features) = (2048, 65536)
W_dec = params["W_dec"]   # (n_features, d_model)
b_enc = params["b_enc"]   # (n_features,)
b_dec = params["b_dec"]   # (d_model,)
threshold = params["threshold"]  # JumpReLU threshold
```

### Step 2b: Encode activations → sparse coefficients

```python
def sae_encode(h, W_enc, b_enc, threshold):
    """JumpReLU SAE encoding."""
    pre_acts = h @ W_enc + b_enc
    return np.where(pre_acts > threshold, pre_acts, 0.0)

# For each prompt's activation:
coeffs = sae_encode(activations, W_enc, b_enc, threshold)
# coeffs shape: (n_prompts, 65536), ~50-200 nonzero per row
```

### Step 2c: Find z-correlated features

```python
# Per pair, correlate each feature's activation with z
z_features_per_pair = {}
for pair in PAIRS:
    pair_coeffs = coeffs[pair_mask]  # (n_pair, 65536)
    pair_z = z_scores[pair_mask]
    correlations = np.array([
        np.corrcoef(pair_coeffs[:, i], pair_z)[0, 1]
        for i in range(n_features)
        if pair_coeffs[:, i].any()  # skip always-zero features
    ])
    top_z = np.argsort(np.abs(correlations))[-20:]
    z_features_per_pair[pair] = top_z
```

### Step 2d: Analysis

**Cross-pair feature overlap:**
```python
# For each pair of pairs, compute Jaccard overlap of top-20 z-features
overlap_matrix = np.zeros((8, 8))
for i, pair_a in enumerate(PAIRS):
    for j, pair_b in enumerate(PAIRS):
        a = set(z_features_per_pair[pair_a])
        b = set(z_features_per_pair[pair_b])
        overlap_matrix[i, j] = len(a & b) / len(a | b)
```

**Place-cell vs linear feature analysis:**
```python
# For each top z-feature, fit:
#   (a) linear: activation ~ a*z + b
#   (b) Gaussian bump: activation ~ exp(-(z - z0)^2 / 2σ^2)
# If (b) fits better → place cell; if (a) fits better → linear feature
```

**Encoding-vs-use decomposition:**
```python
# Project primal_z onto SAE feature directions
primal_z_in_sae = sae_encode(primal_z, W_enc, b_enc, threshold=0)  # no threshold
# Which SAE features does primal_z load onto?

# Same for probe_z
probe_z_in_sae = sae_encode(probe_z, W_enc, b_enc, threshold=0)
# If primal_z loads onto 3 features and probe_z loads onto 200,
# that explains the 18x steering gap
```

### Outputs

```
results/v9_gemma2/sae_z_features_per_pair.json
results/v9_gemma2/sae_cross_pair_overlap.json
results/v9_gemma2/sae_place_cell_vs_linear.json
figures/v9/sae_z_feature_overlap_heatmap.png
figures/v9/sae_place_cell_profiles.png         — activation vs z per top feature
figures/v9/sae_primal_vs_probe_decomposition.png
```

---

## Priority 3: Manifold geometry (CPU after .npz fetch)

### Step 3a: Isomap vs PCA

```python
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

# Per pair, on cell-mean activations:
isomap_1d = Isomap(n_components=1, n_neighbors=5).fit_transform(cell_means)
pca_1d = PCA(n_components=1).fit_transform(cell_means)

# If isomap_1d perfectly orders by z (R²≈1) but pca_1d doesn't → curved manifold
# If both order equally → manifold is approximately linear
```

### Step 3b: Intrinsic dimensionality

```python
# TWO-NN estimator (Facco et al. 2017)
# For each cell-mean, find 2 nearest neighbors, compute μ = d2/d1
# ID = 1/mean(log(μ))
# If ID ≈ 1 → 1-D curve; ID ≈ 2 → 2-D surface
```

### Step 3c: On-manifold steering (needs GPU)

```python
# 1. Fit principal curve through cell-means ordered by z
# 2. At each prompt's activation, compute local tangent to curve
# 3. Steer along tangent (on-manifold) vs primal_z (off-manifold)
# 4. Compare: logit_diff slope, entropy increase, off-manifold distance
```

### Outputs

```
figures/v9/isomap_vs_pca_8panel.png
figures/v9/intrinsic_dimensionality.png
figures/v9/on_vs_off_manifold_steering.png  (needs GPU)
```

---

## Priority 4: Park's causal inner product (CPU)

Uses the unembedding matrix W_U to compute the causal inner product from
Park et al. (ICML 2024). Tests whether transforming probe_z by the causal
metric makes it steer as well as primal_z.

```python
# W_U shape: (d_model, vocab_size)
# Causal inner product: <u, v>_causal = u^T W_U W_U^T v
# Transform probe: probe_z_causal = (W_U W_U^T)^{-1} probe_z
# Then steer with probe_z_causal — does it match primal_z's effectiveness?
```

Needs W_U.npy (on HF) + GPU for steering evaluation.

---

## Summary

| # | Experiment | GPU? | Time | Priority |
|---|-----------|------|------|----------|
| 1 | Replicate behavioral signal on Gemma 2 2B | GPU | 5 min | HIGH |
| 2 | SAE feature decomposition of z | GPU + CPU | 10 min + analysis | HIGH |
| 3 | Manifold geometry (isomap, ID, on-manifold steering) | CPU + GPU | 5 min + 5 min | MEDIUM |
| 4 | Park's causal inner product | CPU + GPU | 5 min | MEDIUM |

**Total: ~25 min GPU, ~30 min CPU analysis**

## Key references

- [Gemma Scope: Open SAEs for Gemma 2](https://arxiv.org/abs/2408.05147)
  Weights: `google/gemma-scope-2b-pt-res`
- [Gemma Scope 2 for Gemma 3](https://huggingface.co/google/gemma-scope-2)
  Covers 270M–27B with residual, MLP, attention SAEs + transcoders
- [Anthropic — When Models Manipulate Manifolds](https://arxiv.org/abs/2601.04480)
  Place-cell SAE features on curved counting manifold
- [Tegmark — Geometry of Concepts: SAE Feature Structure](https://arxiv.org/abs/2410.19750)
  Crystals, lobes, and galaxies in SAE feature dictionaries
- [Park et al. — Linear Representation Hypothesis (ICML 2024)](https://arxiv.org/abs/2311.03658)
  Causal inner product connecting probes to steering
- [SAE scaling with feature manifolds](https://arxiv.org/html/2509.02565v1)
  How SAEs tile continuous manifolds with discrete features
