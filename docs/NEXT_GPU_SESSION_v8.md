# Next GPU Session v8 — Direct Sign Classification + Plot Refresh

**Created:** Apr 21, 2026
**Context:** v7 clean-grid confirmed R=0.47 for positive/negative math on balanced (x,z) grid. But the prompt "This number is ___" is ambiguous — the model may be completing with relative-position adjectives, not sign classification. v8 tests whether an explicit sign-comparison prompt eliminates the context effect.

---

## Priority 1: Direct Sign Classification Prompt

**Motivation:** The v6/v7 posneg_abs prompt was:

```
Number 1: 6.6
Number 2: 3.2
...
Number 16: -3. This number is
```

The model completes with any adjective. We measure logit("positive") − logit("negative"), but the model might be thinking "low/high relative to the list" rather than "positive/negative sign." The R=0.47 context effect could reflect relative-position judgment leaking into the pos/neg token competition, not genuine sign-relativity.

**Fix:** Use prompts that explicitly ask about comparison to zero.

### Experiment 1a: "above/below zero" prompt

```
Number 1: 6.6
Number 2: 3.2
...
Number 16: -3. This number is [above/below] zero.
```

Measure: `logit("above") − logit("below")`

Tokens: " above" and " below" should each be single tokens in Gemma's tokenizer. Verify with `_token_utils.py` before running.

The key word "zero" appears AFTER the blank position, so the model at the blank position doesn't see "zero" yet. **This won't work as written.** Need to restructure:

```
Number 16: -3. Compared to zero, this number is
```

Now "zero" appears before the blank. Model completes with "above"/"below"/"higher"/"lower"/etc.

### Experiment 1b: "higher/lower than zero" prompt

```
Number 1: 6.6
...
Number 16: -3. Relative to zero, this number is
```

Measure: `logit("higher") − logit("lower")`

### Experiment 1c: Forced binary classification

```
Number 1: 6.6
...
Number 16: -3. Is this number above or below zero? Answer:
```

Measure: `logit("above") − logit("below")` or `logit("Above") − logit("Below")`

This is the most direct. The model should basically always get this right. If R > 0.2 even here, that's a very strong claim about context-relativity.

### Experiment 1d: Original prompt for comparison

Keep the original "This number is ___" prompt (v6 design) running on the same prompts for direct A/B comparison.

### Grid

Use **Grid B (x, z)** to avoid the confound:
- x ∈ {−8, −3, 0, 3, 8}
- z ∈ {−2, −1, 0, 1, 2}
- μ = x − σ·z, σ = 3
- 30 seeds per cell = 750 prompts per prompt variant
- 4 variants × 750 = 3000 prompts total

### What to measure per variant

1. **Relativity ratio R** = −slope_μ / slope_x from `ld ~ b·x + c·μ`
2. **Classification accuracy**: fraction of prompts where sign(ld) = sign(x) (excluding x=0)
3. **Top-5 tokens at the blank position** for a sample of prompts — what is the model actually predicting?
4. **Confidence calibration**: does |ld| scale with |x|? (monotonicity check)

### Predictions

| Variant | Expected R if absolute | Expected R if relative |
|---------|----------------------|----------------------|
| 1a: "Compared to zero, this is" | ~0 | ~0.4 |
| 1b: "Relative to zero, this is" | ~0 | ~0.4 |
| 1c: "Is this above or below zero?" | ~0 | ~0.4 |
| 1d: "This number is" (original) | ~0.4 (known) | ~0.4 (known) |

If 1a–1c show R ≈ 0 while 1d shows R ≈ 0.4:
→ The model CAN do absolute sign classification, but the open-ended prompt elicits relative judgment instead. The R=0.4 was a prompt-design artifact, not a finding about number cognition.

If 1a–1c ALSO show R ≈ 0.3–0.4:
→ The model genuinely treats even explicit sign classification as context-dependent. This is the strong finding: **"the LLM's number representations are inherently context-relative, even when asked an absolute question."**

**Estimated time:** ~3 min on H100 (3000 prompts)

---

## Priority 2: Top-K Token Analysis

**For each prompt variant, at each (x, z) cell, log the top-10 predicted tokens and their log-probs.** This lets us see WHAT the model is actually predicting, not just the 2-token logit_diff.

Questions this answers:
- Is "positive"/"negative" even in the top 10? Or is the model thinking about different adjectives entirely?
- Do the top tokens shift from "low/small/below" → "high/large/above" as x increases?
- Are relative adjectives ("low", "small", "the lowest") more probable than absolute ones ("negative", "below zero")?

**Estimated time:** ~1 min additional (just log extra tokens during the same forward pass)

---

## Priority 3: Replot v4/v5 Figures on Grid B Data

### Which plots are FINE (behavioral, no activation geometry)

These use behavioral logit_diff data from the (x, μ) grid, which is clean by design:

| Plot | Location | Status |
|------|----------|--------|
| logit_diff_heatmap_xmu_8panel | figures/v4_adjpairs/ | ✅ Keep |
| logit_diff_heatmap_xz_8panel | figures/v4_adjpairs/ | ✅ Keep |
| logit_diff_vs_z_8panel | figures/v4_adjpairs/ | ✅ Keep |
| relativity_across_pairs | figures/v4_adjpairs/ | ✅ Keep |
| context_effect_per_pair | figures/v4_adjpairs/ | ✅ Keep |
| eight_pair_summary | figures/v4_adjpairs/ | ✅ Keep |
| g31b_vs_e4b_relativity | figures/v5_gpu_session/ | ✅ Keep |
| relativity_ratio_absolute_vs_relative | figures/v5_gpu_session/ | ✅ Keep |
| implicit_vs_explicit_scatter | figures/v5_gpu_session/ | ✅ Keep |
| zero_shot_bias_per_pair | figures/v5_gpu_session/ | ✅ Keep |

### Which plots need REDO on Grid B activations

These use activation geometry (PCA, probes, cosines, INLP) and were contaminated:

| Plot | Old Location | What to redo | v7 equivalent exists? |
|------|-------------|-------------|----------------------|
| pca_per_pair_late (horseshoe) | figures/v4_adjpairs/ | PCA on Grid B cell means | ❌ Need to generate |
| pca_cell_means_late/mid | figures/v4_dense/ | PCA on Grid B cell means | ❌ Need to generate |
| meta_z_direction (SVD scree) | figures/v4_adjpairs/ | SVD of Grid B PC1s | ❌ Need to generate |
| pc_correlations_and_shared_z | figures/v4_adjpairs/ | Cross-pair PC1 cosines from Grid B | ❌ Need to generate |
| inlp_late/mid | figures/v4_dense/ | ✅ Already in figures/v7/inlp_clean_curves.png |
| fisher_vs_euclid_cosines | figures/v5_gpu_session/ | ✅ Already in figures/v7/fisher_entropy_bins_clean.png |
| zero_shot_vs_implicit_directions | figures/v5_gpu_session/ | Probes from Grid B | ❌ Need to generate |
| park_vs_euclid_cosines | figures/v4_adjpairs/ | ✅ Covered by fisher_entropy_bins_clean.png |
| meta_w1_steering_curves | figures/v5_gpu_session/ | ✅ Already in figures/v7/seven_direction_curves_clean_8pair.png |

### New v7 plots already generated

| Plot | Location | Replaces |
|------|----------|----------|
| direction_confound_matrix_gridB | figures/v7/ | v6 confound matrix |
| seven_direction_curves_clean_8pair | figures/v7/ | v6 steering curves |
| clean_transfer_heatmap | figures/v7/ | v6 transfer (probe_z, near-noise) |
| inlp_clean_curves | figures/v7/ | v4 INLP (artifact) |
| cross_grid_direction_stability | figures/v7/ | NEW |
| clean_vs_v6_primal_probe_slopes | figures/v7/ | NEW |
| fisher_entropy_bins_clean | figures/v7/ | v5/v6 Fisher plots |

### Plots that need GPU to regenerate

These require Grid B .npz activation files (on HF, not local):

1. **PCA horseshoe on Grid B** — per-pair PCA on Grid B cell-mean activations, color by z. Does the horseshoe survive on clean grid?
2. **Meta-direction on Grid B** — SVD scree of stacked Grid B PC1s. Does the 41.6% shared variance hold?
3. **Cross-pair PC1 cosine heatmap on Grid B** — 8×8 |cos| matrix. Does mean |cos| = 0.32 hold?
4. **Zero-shot vs implicit direction comparison on Grid B** — cos(w_x_zeroshot, w_z_implicit) from clean probes.

These only need CPU once we fetch the .npz files from HF.

**Estimated time:** ~5 min CPU (fetch .npz, run PCA/SVD/cosines, plot)

---

## Priority 4: Cross-Template Transfer Test

**Motivation (red-team):** The cross-pair transfer (40% of own-pair) could be driven by shared prompt template rather than shared z-computation. All prompts use "Person/Number N: X. This __ is ___."

**Test:** For height pair, use two different prompt templates:
- Template A (standard): "Person 16: 170 cm. This person is"
- Template B (different): "Among the individuals listed, the one measuring 170 cm would be described as"

Extract activations for both templates. Compute primal_z on Template A, steer prompts from Template B. If transfer works across templates, the z-direction is about semantics, not syntax.

**Estimated time:** ~5 min on H100

---

## Summary

| # | Experiment | GPU? | Time | Priority |
|---|-----------|------|------|----------|
| 1 | Direct sign classification (4 variants) | GPU | 3 min | HIGH — resolves posneg ambiguity |
| 2 | Top-K token analysis | GPU | 1 min | HIGH — bundled with Exp 1 |
| 3 | Replot v4/v5 on Grid B data | CPU | 5 min | MEDIUM — needs HF fetch |
| 4 | Cross-template transfer test | GPU | 5 min | MEDIUM — red-team |

**Total: ~15 min GPU, ~5 min CPU**
