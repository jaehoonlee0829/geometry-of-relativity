# FINDINGS — v4 adjective-pair experiments on Gemma 4 E4B

**Date:** 2026-04-20 / 21 (Day 4 of the 2-hour autonomous burst extended into analysis)
**Model:** `google/gemma-4-e4b` (≈8B params, residual width 2560)
**Layers probed:** `mid` = 21, `late` = 32

This document records what we actually found once the v4 pipeline finished and we started looking at the activation geometry. Companion doc to `PLANNING.md` (the pre-registration) and `STATUS.md` (the running log).

---

## 1. Setup recap

We extended the original 2-pair (tall/short, obese) design to **8 adjective pairs**, seven expected to be context-relative and one absolute control:

| Pair | Low word | High word | Unit | Expected |
|---|---|---|---|---|
| height | short | tall | cm | relative |
| age | young | old | years | relative |
| weight | light | heavy | kg | relative |
| size | small | big | cm | relative |
| speed | slow | fast | km/h | relative |
| wealth | poor | rich | USD/yr | relative (log-space) |
| experience | novice | expert | years | relative (log-space) |
| bmi_abs | underweight | obese | BMI | **absolute control** |

Per pair: 5 target values × 5 context means × 30 seeds (implicit) + 25 explicit + 5 zero-shot = **780 prompts × 8 pairs = 6240 total**. Extracted residual-stream activations at `mid` and `late` for every prompt at the final-before-label token position.

---

## 2. The "horseshoe" — PCA on cell means

When we PCA the per-cell mean activations (cells = bins of (x, μ)) at the late layer, **PC1 tracks z** and **PC2 tracks z²** — producing the classic PCA horseshoe.

Initial interpretation temptation: "the model is learning kurtosis / Gaussian surprise."
Correct interpretation: this is the well-known geometric artifact of embedding a 1-D ordered manifold into 2-D — PC2 is forced to be roughly quadratic in the PC1 coordinate to preserve ordering. See `results/v4_analysis/figures/pca_cell_means_late.png`.

**z² ≠ kurtosis.** In a Gaussian context, `z² = −2·log p(x | μ, σ) + const`, i.e. squared Mahalanobis distance / negative log-likelihood of the target under the in-context distribution. That's a meaningful interpretability quantity, but the PC2 curve itself is mostly geometric, not semantic.

---

## 3. Per-pair PCA — does every relative concept show the z pattern?

Ran PC1–PC5 per pair on late-layer cell means. (`pca_per_pair_late.png` + `pc_correlations_late.json`.)

**PC1 is z-dominant for all 7 relative pairs:** `R²(PC1 ~ z)` ranges 0.76 – 0.93, capturing ~50–60% of the cell-mean variance.

**PC2 is z²-dominant for most pairs** (R² 0.56 – 0.79) — the horseshoe generalizes. Exceptions: `size` (PC2~x dominates, R²=0.80) and `experience` (PC3~z², not PC2) — these concepts appear to mix raw-x and relativity differently, possibly because their target ranges are broader in log-space.

**The absolute control `bmi_abs` also shows the PC1~z pattern** (R²=0.85). This is suspicious — either (a) the `format_prompt_implicit` template still induces some contextual comparison even for BMI, or (b) PC1 is partly a raw-x axis rather than a pure z-axis. This motivates a no-ICL control (planned).

**PC3~x or μ, not raw x alone.** For several pairs PC3 is μ-dominant rather than x-dominant (age: R²(PC3~μ)=0.76; speed: R²(PC3~μ)=0.59). The model is not building separate axes for raw x vs μ vs z — it seems to put the z-composition front and center and treat (x, μ) as needed inputs.

---

## 4. Cross-pair PC1 cosine similarity

Stack the 8 per-pair PC1 unit vectors and compute the 8×8 |cos| matrix. (`pc_correlations_and_shared_z.png`.)

- **Mean off-diagonal |cos| = 0.32.** Not orthogonal (would be ~0.05 at random in 2560-d), not identical (would be 1.0). The z-direction is real but carries a concept-specific flavor.
- **Semantically similar pairs cluster:** `weight ↔ bmi_abs` = 0.70 (both about mass/body). `wealth ↔ experience` = 0.58 (both log-space). `age ↔ size` = 0.08 (most different — temporal vs spatial).
- Implication: the model re-uses a shared substrate for "extreme-vs-typical" judgments but specializes it per semantic domain.

---

## 5. The meta-direction (main v4 finding)

**Question:** Is there a single direction in 2560-d residual-stream space that serves as a general "how-extreme-vs-reference" axis across all 8 concepts?

**Method:**
1. For each pair, compute PC1 of cell-mean activations (late layer).
2. Sign-align each PC1 so `centered_acts @ pc1` positively correlates with z.
3. Stack into `V` ∈ ℝ^{8×2560}, take SVD: `V = U S Wᵀ`.
4. `w₁` = top right singular vector = "meta-direction" = the best single-direction approximation to all 8 per-pair z-axes simultaneously.

**Results** (`meta_z_direction.png` + `meta_z_direction.json`):

| pair | R²(own PC1 ~ z) | R²(meta w₁ ~ z) | \|cos(own, meta)\| |
|---|---|---|---|
| height | 0.857 | **0.909** | 0.75 |
| age | 0.762 | **0.800** | 0.63 |
| weight | 0.909 | 0.891 | 0.80 |
| size | 0.391 | **0.762** | 0.42 |
| speed | 0.930 | 0.914 | 0.53 |
| wealth | 0.914 | 0.919 | 0.71 |
| experience | 0.841 | **0.894** | 0.57 |
| bmi_abs | 0.848 | 0.863 | 0.66 |

**SVD scree:** `w₁` captures **41.6%** of the variance across the 8 per-pair PC1s (remaining 7 singular values drop gradually: 15%, 12%, 11%, 7%, 6%, 5%, 3%).

**Interpretation:**
- A single direction in activation space predicts z with R² ≥ 0.76 for every concept, and matches or beats each concept's *own* optimal PC1 in 5 of 8 cases. The "compromise" direction denoises concept-specific PCA residuals (most dramatic for `size`: 0.39 → 0.76).
- At 41.6% shared variance vs the 12.5% orthogonal baseline, the 8 z-axes are **~3.3× more aligned than chance** but are not identical. Real concept-specific residual exists (|cos| values 0.42–0.80).
- **Strongest interpretability claim from v4:** Gemma 4 E4B has internalized a *domain-general "extremity-vs-reference" feature* that activates across concepts, rather than instantiating separate z-machinery per domain.

---

## 6. Caveats & open questions

**(a) `bmi_abs` contamination.** The absolute control aligns with `w₁` (R²=0.863, |cos|=0.66). Either the prompt format induces unintended relativity, or `w₁` captures raw-x position on the grid rather than normalized z. Disambiguated by:
- **No-ICL control** (planned, Task #23): use `format_prompt_zero` (x only, no reference samples) and check whether PC1 still tracks z or collapses to x.

**(b) x / (μ, z) entanglement.** Since z = (x − μ)/σ, any two of {x, μ, z} determine the third. "PC3 ~ x" can always be rewritten as a mix of μ and z. The no-ICL control isolates pure-x encoding.

**(c) Correlation ≠ causation.** All current analyses are correlational — we find linear structure that predicts z. Whether the model actually *uses* `w₁` to compute the adjective distribution requires a causal intervention:
- **Meta-direction steering** (planned, Task #27): add `±α·w₁` to the residual stream at layer 32 across prompts from different concepts. If outputs shift toward high-adjectives for +α and low-adjectives for −α *across all 8 concepts*, that's causal evidence for a domain-general relativity feature. Compare to a random unit-vector baseline.

**(d) Framing claim.** A safe, defensible paper claim given current evidence:
> "Under in-context Gaussian example sets, Gemma 4 E4B represents the target primarily along a ~1-D manifold in the late residual stream. Across 8 semantic domains, the per-concept z-axes are non-orthogonal (mean |cos|=0.32) and collapse onto a single shared direction `w₁` that captures 42% of the cross-concept PC1 variance and predicts z-score with R² = 0.76–0.92 on all concepts including an absolute control. This suggests a partially-shared, partially-specialized 'relativity substrate' rather than either per-concept or fully-shared representation."

---

## 7. Artifact inventory

Scripts (in `scripts/vast_remote/`):
- `extract_v4_adjpairs.py` — 6240-prompt activation extraction (Day 4)
- `analyze_v4_adjpairs.py` — per-pair logit-diff + monotonicity checks
- `steer_v4.py`, `inlp_v4.py` — Day 4 steering & concept erasure (earlier work)

New analysis scripts (this round, in repo root or `scripts/`):
- `pca_per_pair.py` — per-pair PCA, PC1–PC5 R² table
- `pc3_x_and_shared_z.py` — PC3 vs x and cross-pair |cos| matrix
- `meta_z_direction.py` — SVD-based meta-direction analysis

Figures (`results/v4_adjpairs_analysis/figures/`):
- `pca_per_pair_late.png` — 8-panel PC1/PC2 scatter, one per pair
- `pc_correlations_and_shared_z.png` — cross-pair |cos| heatmap + per-PC R² bars
- `meta_z_direction.png` — SVD scree + R²(own vs meta~z) + |cos(own, meta)|
- `pca_cell_means_late.png` — original horseshoe figure

JSON summaries (same dir):
- `pca_per_pair_late.json`, `pc_correlations_late.json`, `meta_z_direction.json`

---

## 8. Next experiments

1. **No-ICL control** (Task #23) — run extraction with `format_prompt_zero` across same x-grid. Check whether PC1 tracks x or z when no reference samples are given. Expected: R²(PC1~z) drops sharply; R²(PC1~x) rises. If PC1 still tracks "z-like structure" without ICL, the geometry is baked into the model rather than emerging from context.
2. **Meta-direction steering** (Task #27) — causal intervention. See caveat (c) above.
3. **Transfer to Llama-3.2-3B** (replication). Does `w₁` transfer to a different model family?
