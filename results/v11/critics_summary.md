# v11 P6 critic round

## methodology

# methodology critic

# Punch list — v11 methodology critique

### 1. Orthogonalized increment R² is broken, not informative
**Concern:** `orth_r2_per_layer` goes massively **negative** at every mid/late layer for every pair (e.g. `gemma2-2b/height` reaches **−5.70** at L25; `gemma2-9b/bmi_abs` hits **−23.5** at L41; `gemma2-9b/age` −18.3 at L41). Negative out-of-sample R² means the residualization-then-refit pipeline is producing predictions worse than the mean — almost certainly because (a) residualization is done *in-sample* on z (or on h_{L−1}'s linear projection of z) and the held-out fold then sees a target whose variance has been arbitrarily inflated, or (b) the "increment" is being predicted by ẑ_{L−1} on the wrong fold. Linear residualization also cannot remove non-linear z information carried by h_{L−1}, so the "orthogonalized" claim is doubly weak.
**JSON:** `results/v11/{gemma2-2b,gemma2-9b}/*/increment_r2_orthogonalized.json`
**Fix:** Retract the orthogonalized-increment claim until the residualization is done *within each CV fold* (fit ẑ on train, project on test), and report ΔR² of (h_L predicting z) − (h_{L−1} predicting z) on the same held-out fold instead. Don't interpret negative R² as "no new info" — it's a pipeline bug.

### 2. z-vs-lexical disentanglement is non-functional
**Concern:** Every `cos_primal_lexical_zero_z` and `cos_primal_lexical_naive` entry is **NaN**, and `n_pred_high_amb = n_pred_low_amb = 0` for all 26/42 layers across all pairs in both

*(see critic_methodology.md for full)*
## alternative

# alternative critic

# Alternative-Explanation Critique of v11

## 1. Cross-pair transfer is mostly a generic "make-the-number-bigger" signal, not a shared z-code
**File:** `results/v11/gemma2-2b/cross_pair_transfer_dense.json`, `results/v11/gemma2-9b/cross_pair_transfer_dense.json`

Within-pair slopes (2B): height 0.040, weight 0.058, age 0.067, wealth 0.079, speed 0.073, size 0.100, experience 0.108. Off-diagonals are routinely 40–80% of diagonal — but look at the structure: `size→height` = 0.078 vs within-height 0.040 (transfer *exceeds* within!), and `size→weight` = 0.080 vs within-weight 0.058. That isn't "40% transfer of a domain-general z-code"; it's that the `size` direction is just a stronger generic magnitude pusher. The cheap explanation: shared **number-token-magnitude** direction in residual stream (numerals 150/170/200 etc. share embedding geometry across all numeric pairs). No z-normalization needed. Test the team didn't run: transfer with μ held constant (pure x-shift) should give similar slopes if true.

## 2. PC1≈z is just PC1≈x in disguise for high-σ/μ-ratio pairs
**File:** `results/v11/gemma2-2b/height/pca_summary.json` (PC1_vs_z=0.969, PC1_vs_x=1e-10) vs `size/pca_summary.json` (PC1_vs_z=0.075, PC1_vs_x=0.651) and `age/pca_summary.json` (PC1_vs_z=0.21, PC1_vs_x=0.42).

The pairs where PC1 "tracks z" (height 0.97, weight 0.95, bmi_abs 0.92, wealth 0.86) are exactly the pairs where x and z were most successfully **decorrelated by grid design** — so PC1 pic

*(see critic_alternative.md for full)*
## statistical

# statistical critic

# Statistical Critique of v11

## 1. No CIs anywhere in the reported JSONs
Every R² in `pca_summary.json`, every cos in `cos_pc1_primal_per_layer`, every slope in `cross_pair_transfer_dense.json`, and every `corr_z_after_ablation` in `head_ablation_causal.json` is reported as a point estimate. With N=400 cell-means (e.g. `n_groups: 400` for height, 351 for experience, 362 for size), bootstrap CIs are cheap and essential. E.g. the headline `PC1_vs_z = 0.9690` for height and `PC1_vs_z = 0.9226` for bmi_abs cannot be statistically distinguished from each other without intervals. The "size" claim of `PC1_vs_z=0.075, PC2_vs_z=0.785` (gemma2-2b) is a strong qualitative claim about axis-swapping that needs a bootstrap on the eigenvalue gap (`evr = [0.416, 0.307]` — only a 0.11 gap, likely unstable).
**Fix:** 1000-sample block bootstrap over (μ,x) cells; report 95% CI on every R² and on `λ₁−λ₂`.

## 2. Cross-pair transfer matrix: 56 off-diagonal cells, no Bonferroni
`cross_pair_transfer_dense.json` reports raw slopes ranging −0.018 to +0.080. Without a null distribution per cell, "off-diagonal ≈ 0" is unsupported. With 56 comparisons, Bonferroni α=0.05/56 ≈ 8.9e-4 — meaning each cell needs ~3.3σ. Several off-diagonals (e.g. `size→weight = 0.080`, `size→height = 0.078` for gemma2-2b) are *within 25%* of within-pair slopes (`height = 0.040`, `weight = 0.058`). The transfer story is not as clean as the diagonal-vs-off-diagonal narrative implies.
**Fix:** permutatio

*(see critic_statistical.md for full)*
## novelty

# novelty critic

# Novelty Critique of v11

## Claim 1: "Encode-vs-use as a layer-depth phenomenon" (encode by L7, use from L11+, peak L14)
- **Novelty: 2/5**
- **Closest prior:** Geva et al. 2021 ("FFN as key-value memories") + Geva 2022 ("promotes concepts in vocab space") already establish that early layers build features and late layers project to logits. nostalgebraist's logit-lens (2020) and Belrose et al.'s tuned-lens (2023) operationalize the same encode→use depth gradient. Lad et al. 2024 ("four universal stages") names the phases explicitly. The probe/primal gap (8× at L14, `cos_pc1_primal_per_layer` height L1=0.997 vs late flips) is a quantitative refinement, not a new phenomenon.
- **Recommendation:** Reframe as "graded-quantity-specific encode-use depth profile" and cite Lad/Geva/tuned-lens explicitly. The novelty is the *Z-score-vs-raw* axis, not the layer-depth observation per se.

## Claim 2: W_U-based lexical-vs-z disentanglement (P3d-fixed)
- **Novelty: 3/5**
- **Closest prior:** Park, Choe & Veitch (2024) "The Linear Representation Hypothesis" defines the causal inner product via `Cov(γ)⁻¹` on unembedding-space differences; Marks & Tegmark's "Geometry of Truth" projects probe directions onto unembedding subspaces. The v11 file `z_vs_lexical_per_layer.json` shows `cos_primal_lexical_unembed` ≈ 0 at all layers (range −0.045 to +0.058 for height) — i.e., primal_z is **orthogonal to the tall/short unembedding axis**. That's a substantive, non-trivial result Pa

*(see critic_novelty.md for full)*
## narrative

# narrative critic

# Narrative consistency critique: v11 vs v9/v10

**1. Head taxonomy: v11 9B silently disagrees with v10 2B's "canonical z-writers."**
FINDINGS §14.6 names L13h2 the standout comparator in 2B and L0h6 the μ-aggregator. v11's 2B `head_ablation_causal.json` ablates exactly these (L13h2, L0h6, L3h0) and finds **Δcorr_z ∈ [−0.0075, +0.0002]** — i.e. the "canonical" heads are causally inert (<1% of baseline 0.976). The 9B run picks entirely new candidates (L21h3 z_writer, L16h3 comparator) with similarly null effects (Δ ≤ 0.0024). This either (a) refutes the v10 §14.6 taxonomy as load-bearing, or (b) shows the taxonomy doesn't transfer 2B→9B. v11 reports the numbers but FINDINGS §14.7's claim that "L13h2 comparator, L10h0/L17h7 z-writers" are mechanistically meaningful is left standing without acknowledgment that ablating them does ~nothing. This is an unacknowledged self-refutation, not just a "structural difference in detail."

**2. Cross-pair transfer contradicts v8's 97% cross-template / 0.19 PC1 cosine framing.**
`cross_pair_transfer_dense.json` (2b, L20) shows within-pair slopes 0.040–0.108 vs off-diagonal medians ~0.025–0.05 — i.e. cross-pair transfer is **~30–60% of within-pair**, not the near-zero implied by v8's "cross-pair PC1 cosine 0.19" (STATUS bullet). 9B is similar (within 0.044–0.093 vs off-diagonal ~0.02–0.05). Whether this *strengthens* or *weakens* the "shared z-direction" story is never reconciled with v8.

**3. SAE story: v11 higher-N repli

*(see critic_narrative.md for full)*