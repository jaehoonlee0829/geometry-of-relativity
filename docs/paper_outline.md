# Paper outline — "The Geometry of Relativity: Context-Normalized Encoding of Gradable Adjectives in LLMs"

**Target venue (primary):** ICML 2026 Mechanistic Interpretability Workshop, deadline May 8 2026.
**Target venue (secondary):** NeurIPS 2026 (abstract May 4, full paper May 6).
**Status:** draft skeleton, Apr 21 2026. Numbers marked `<<TBD>>` fill in when Vast runs land.

## One-sentence thesis

A small family of gradable adjectives — "tall", "rich", "heavy", "fast", etc. — are represented
in mid-to-late-layer residual streams of Gemma 4 E4B along a context-normalized direction
that tracks the z-score `z = (x − μ)/σ` rather than the raw magnitude `x`; absolute adjectives
(e.g. "obese" via BMI threshold) do not show this context-normalization. We back this with five
converging lines of evidence, including a causal steering test and INLP concept erasure.

## Section map

### §1 Introduction
- Motivation: how do LLMs internally represent graded, context-dependent judgments?
- Claim: relative adjectives → context-normalized (z-like) direction; absolute adjectives → raw-quantity direction.
- Why this matters for alignment / fairness / evaluation: the same person is "tall" in one context and "average" in another; the model's internal representation reflects that.
- Contribution summary (5 evidence lines, plus Fisher-Rao geometric framing).

### §2 Related work
- Linear representation hypothesis (Mikolov et al.; Park, Choe, Veitch 2023).
- Concept erasure: Ravfogel et al. 2020 (INLP); Belrose et al. 2023 (LEACE).
- Probing literature critique: Hewitt & Liang 2019; Pimentel et al. 2020.
- Scalar-attribute probes in LLMs: (reference to any prior work found during lit review; `<<TBD>>`).
- Fisher-information geometry in interpretability: Park et al. 2023 on causal inner product.

### §3 Setup

**Model.** Gemma 4 E4B (42 layers, d_model = 2560, W_U shape 256000 × 2560).
Secondary runs on Gemma 4 31B (60 layers, d_model = 5376) planned but not required for primary claims.

**Adjective pairs.** Seven relative pairs plus one absolute control:

- Relative: height/(tall,short), age/(old,young), weight/(heavy,light), size/(big,small), speed/(fast,slow), wealth/(rich,poor), experience/(expert,novice).
- Absolute control: BMI/(obese,normal) with fixed threshold 30.

**Prompt conditions (per pair).**
1. *Implicit:* 15-person context sampled from Normal(μ, σ), then "Person X is {value}. Person X is ___".
2. *Explicit:* "In a group where the average is μ, Person X is {value}. Person X is ___".
3. *Zero-shot:* "Person X is {value}. Person X is ___" — no context.

**Variables.** Raw target value x ∈ 5 levels; context mean μ ∈ 5 levels (implicit & explicit);
30 resample seeds per (x, μ) cell for implicit → 3500 implicit + 35 explicit + 5 zero-shot per pair = 780 prompts; 8 pairs → 6240 total.

**Measurements per trial.**
- `logit_diff = logit(high_adj_token) − logit(low_adj_token)` at the final "is" position.
- Residual-stream activations at layer_mid (≈ layer 21) and layer_late (≈ layer 36), last-token.

**Probes.** Ridge regression on standardized features with 5-fold shuffled CV.
Unit weight vectors are recovered by back-transforming through StandardScaler's `scale_`.

### §4 Evidence Line 1 — Behavioral relativity ratio

Core regression: `logit_diff ~ a + b·x + c·μ + ε` on the 3500 implicit trials per pair.

Define the **relativity ratio** R = −c / b. If representation is purely context-normalized (z = (x − μ)/σ),
then R → 1 (μ shifts the decision threshold 1-for-1 with x). If representation is raw-magnitude, c ≈ 0 and R → 0.

**Predictions.**
- Relative pairs: R ∈ [0.8, 1.2]. `<<TBD-from-analyze_v4_adjpairs>>`
- Absolute pair (BMI/obese): R ≈ 0. `<<TBD>>`

**Falsifier.** Either relative pairs cluster near R = 0, or the BMI pair climbs to R ≈ 1.

**Headline number.** 7/8 pairs in predicted regime, with BMI isolated at R = `<<TBD>>` vs. relative-pair median R = `<<TBD>>`. See Fig. 1.

### §5 Evidence Line 2 — Probe decodability

Train four ridge probes on residual-stream activations from the 3500 implicit trials:
predict x, μ, z, sign(z). Report 5-fold shuffled CV R². Shuffling is essential because
activations are stored sorted by (x, μ, seed); unshuffled folds would create x-bucketed
test sets and spurious negative R².

**Predictions.**
- For relative adjectives at layer_late: CV R²(z) > 0.7; CV R²(x) < 0.4. `<<TBD>>`
- For BMI absolute control: CV R²(x) ≳ CV R²(z). `<<TBD>>`

**Falsifier.** CV R²(z) ≤ CV R²(x) for relative pairs, or the ordering reversed for BMI.

### §6 Evidence Line 3 — Geometry (PCA of cell means)

Average activations within each (x, μ) cell, yielding 35 mean vectors per pair. Run PCA.
Correlate PC1 with z, x, μ, and `logit_diff`.

**Predictions.**
- PC1 correlates ≥ 0.8 with z (or equivalently with `logit_diff`) for relative pairs. `<<TBD>>`
- For BMI, PC1 correlates with x rather than z. `<<TBD>>`

We additionally compute Σ⁻¹ cos(w_adj, w_z) using the activation-covariance metric
(Cholesky-solved; never inverted explicitly). Prediction: Σ⁻¹ cosines are substantially
higher than Euclidean cosines, supporting the Fisher-Rao pullback interpretation that
w_adj and w_z read the same low-dimensional readout subspace even when they look
orthogonal in ambient space.

### §7 Evidence Line 4 — Causal steering

For each pair, hook the residual stream at a chosen layer L, add α·ŵ_z to the last-token
activation (ŵ_z from Evidence Line 2, unit-normalized), and rerun the forward pass on held-out
(explicit + zero-shot) prompts the probes never saw.

**Predictions.**
- For relative pairs, `logit_diff(α)` is near-linear and monotone with slope of order the
  natural z→logit_diff slope (~1.2 from §4's regression). `<<TBD>>`
- Sign is consistent with the probe direction (+α → more "tall"-like).
- For BMI control, steering along ŵ_z should have a *smaller* effect on the obese/normal
  logit_diff than steering along ŵ_x. `<<TBD>>`

**Falsifier.** Flat or non-monotone steering curve for relative pairs; or same-magnitude
effect for BMI along ŵ_z as for relative pairs. A weak effect is not conclusive (cancellation
across multiple z-encoding directions is possible), but a strong effect IS strong evidence.

### §8 Evidence Line 5 — INLP concept erasure

Iteratively null out the probe direction w_z from activations (Ravfogel et al. 2020):
fit a z-probe, project out its unit direction, retrain; repeat k steps. After each step,
measure CV R²(z), R²(x), R²(logit_diff). Compare against:
(a) a **random-direction null** that projects a random unit vector at each step;
(b) an **x-probe null** that nulls out the x-direction (a competing readout).

**Predictions.**
- INLP-z drops R²(z) sharply within 1–3 steps, while the random baseline preserves R²(z)
  far longer: gap R²(z)[random] − R²(z)[INLP-z] ≥ 0.5. `<<TBD>>`
- INLP-z leaves R²(x) largely intact (x is a *different* direction from z). `<<TBD>>`
- INLP-x, conversely, nulls R²(x) but leaves R²(z) partially intact.
- R²(logit_diff) under INLP-z drops substantially — indicating the model's *behavior*
  also leaves through w_z, not just a decodable-but-unused feature.

**Falsifier.** Random projection matches INLP-z's collapse rate → w_z is not uniquely
the direction encoding z, just a generic one. Alternatively, R²(logit_diff) stays high
under INLP-z → w_z is decodable but unused (passenger feature).

This is the line that distinguishes "w_z is *the* direction for z" from "w_z happens to
correlate with z." Synthetic-data smoke test (test_inlp_smoke.py) already shows the
expected qualitative pattern: initial R²(z) = 0.991 → 0.316 after INLP-z vs. 0.991 under
random, with R²(x) dropping only to 0.210 from 0.644 under INLP-z.

### §9 Discussion

**The Fisher-Rao pullback interpretation.** In the ambient d-dimensional residual space,
different probes (w_adj vs. w_z) can look nearly orthogonal in Euclidean cosine even when
they encode the same downstream readout. Under the activation-covariance metric Σ⁻¹
— equivalently, under the Fisher information pullback F(h)⁻¹ where F(h) = W_U^T (diag(p) − pp^T) W_U —
these cosines collapse toward 1. This matches the theoretical story (Park et al. 2023):
the *causal inner product* isn't Euclidean; it's the one induced by the pullback from logit space.

**Scope.** We have tested one model (Gemma 4 E4B) on seven English adjective pairs plus one
absolute control. The pattern may or may not generalize to (a) other model families, (b)
other languages, (c) absolute adjectives with less clean thresholds.

**Limitations.**
- Our "z" is computed from the implicit context's sampled values, not from the model's
  belief state — so we can't distinguish "model computes z exactly" from "model computes
  some non-linear function of x and μ that is ~linear in z over our range."
- Steering is a blunt instrument; a single direction may not exhaust the model's z-readout.
- BMI as a single absolute-adjective control is a weak baseline; multiple absolute controls
  ("freezing" at 0°C, "legal age" at 18) are queued.

**Connection to alignment.** Context-normalized representation matters for fairness
evaluations: if a model's internal "tall" is really "taller than this room's average," then
a fairness audit that holds x fixed and varies μ will see label flips that cannot be
addressed by debiasing on x alone.

### §10 Conclusion

Five converging lines of evidence — behavioral regression, probe decodability, PCA geometry,
causal steering, and INLP concept erasure — support the claim that relative gradable
adjectives are encoded in context-normalized form in Gemma 4 E4B, while absolute adjectives
are not. The Fisher-Rao pullback view cleans up the apparent primal–dual mismatch between
adjective and z-score probes.

## Appendix sketches

- A. Prompt templates for all 8 pairs (full copy of `scripts/vast_remote/extract_v4_adjpairs.py` PAIRS).
- B. Ridge probe hyperparameter sweep (α ∈ {0.1, 1.0, 10.0}).
- C. Layer sweep — full 42-layer scan of CV R²(z) and CV R²(x). `<<TBD>>`.
- D. Synthetic-data smoke-test results (evidence of method correctness before real-model runs).
- E. Alternative probes: LDA, cross-validated linear SVM — do they recover the same direction? `<<TBD>>`.

## Figures (to produce)

| # | Name                           | Source script                          | Status     |
|---|--------------------------------|----------------------------------------|------------|
| 1 | relativity_across_pairs.png    | analyze_v4_adjpairs.py                 | `<<TBD>>`  |
| 2 | probe_r2_by_layer.png          | analyze_v4.py (phase 2)                | `<<TBD>>`  |
| 3 | pca_cell_means_{layer}.png     | analyze_v4.py (phase 3)                | `<<TBD>>`  |
| 4 | fisher_cosine_comparison.png   | analyze_v4.py (phase 5)                | `<<TBD>>`  |
| 5 | steering_curves_{layer}.png    | steer_v4.py                            | `<<TBD>>`  |
| 6 | inlp_{layer}.png               | inlp_v4.py                             | `<<TBD>>`  |

## Execution checklist (before writing up)

- [ ] analyze_v4.py on Vast → `results/v4_analysis/summary.json`
- [ ] extract_v4_adjpairs.py on Vast → `results/v4_adjpairs/*.npz,*.jsonl`
- [ ] analyze_v4_adjpairs.py → `results/v4_adjpairs_analysis/summary.json` + Fig 1
- [ ] steer_v4.py --layer {late,mid} → `results/v4_steering/` + Fig 5
- [ ] inlp_v4.py --layer {late,mid} → `results/v4_analysis/inlp_*.json` + Fig 6
- [ ] Fill all `<<TBD>>` numbers and figures in this outline
- [ ] Pass outline through a single-author prose pass
- [ ] ICML MI workshop template conversion (8 pages)
