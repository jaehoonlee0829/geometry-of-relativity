# methodology critic

# Punch list — v11 methodology critique

### 1. Orthogonalized increment R² is broken, not informative
**Concern:** `orth_r2_per_layer` goes massively **negative** at every mid/late layer for every pair (e.g. `gemma2-2b/height` reaches **−5.70** at L25; `gemma2-9b/bmi_abs` hits **−23.5** at L41; `gemma2-9b/age` −18.3 at L41). Negative out-of-sample R² means the residualization-then-refit pipeline is producing predictions worse than the mean — almost certainly because (a) residualization is done *in-sample* on z (or on h_{L−1}'s linear projection of z) and the held-out fold then sees a target whose variance has been arbitrarily inflated, or (b) the "increment" is being predicted by ẑ_{L−1} on the wrong fold. Linear residualization also cannot remove non-linear z information carried by h_{L−1}, so the "orthogonalized" claim is doubly weak.
**JSON:** `results/v11/{gemma2-2b,gemma2-9b}/*/increment_r2_orthogonalized.json`
**Fix:** Retract the orthogonalized-increment claim until the residualization is done *within each CV fold* (fit ẑ on train, project on test), and report ΔR² of (h_L predicting z) − (h_{L−1} predicting z) on the same held-out fold instead. Don't interpret negative R² as "no new info" — it's a pipeline bug.

### 2. z-vs-lexical disentanglement is non-functional
**Concern:** Every `cos_primal_lexical_zero_z` and `cos_primal_lexical_naive` entry is **NaN**, and `n_pred_high_amb = n_pred_low_amb = 0` for all 26/42 layers across all pairs in both models. The only surviving number is `cos_primal_lexical_unembed`, which is the cosine of primal_z with `W_U[tall] − W_U[short]` — a **fixed lexical axis independent of the data**. That cosine grows from ~0 early to ~0.15 late simply because primal_z increasingly aligns with the unembedding readout direction. This is exactly the W_U confound the analysis was supposed to control for: high-frequency / high-norm output tokens dominate W_U rows, so cosine with W_U[tall]−W_U[short] is confounded by token-frequency geometry, not "lexical vs z" content.
**JSON:** `results/v11/{gemma2-2b,gemma2-9b}/z_vs_lexical_per_layer.json`
**Fix:** Drop the W_U-only metric. Either (a) get the ambiguous-prompt path working so `n_pred_*_amb > 0` and `cos_primal_lexical_zero_z` is defined, or (b) retract P3d entirely. As stands the JSON proves nothing.

### 3. 9B head taxonomy thresholds + ablation are circular and null
**Concern:** `head_taxonomy.json` thresholds (`ctx=0.0456`, `tgt=0.0504`, `dla_abs=0.0400`) are top-quartile **of the 64 strategic heads being inspected** — same heads later promoted to "z_writer_top" / "comparator_top" / "mu_aggregator_top" and then ablated. Tagging is computed on all 4000 prompts; ablation effect is measured on the same prompts. More damning: the ablations are **null**: gemma2-9b L21h3 z_writer Δcorr_z = **−0.0011**; comparator L16h3 = **+0.0024** (sign flipped!); mu_aggregator L0h3 = +0.0004. gemma2-2b is the same story (Δ ≤ 0.008). The taxonomy labels are not predictive of causal importance.
**JSON:** `results/v11/{gemma2-2b,gemma2-9b}/head_taxonomy.json`, `head_ablation_causal.json`
**Fix:** Either retract the "5 z-writers / 18 comparators / 15 μ-aggregators" causal framing (re-label them as *correlational* tags), or split prompts: pick thresholds on half, ablate on held-out half, and ablate the whole tag-set jointly (not one head). With Δcorr_z < 0.01 the honest claim is "no individual head is causally necessary."

### 4. Cross-pair transfer is noise-floor + diagonal-trivial
**Concern:** `cross_pair_transfer_dense.json` reports seed-0 only, no SE/CI. Within-pair slopes range 0.040 (height) to 0.108 (experience) on gemma2-2b — a 2.7× spread that suggests per-pair noise dominates. Off-diagonal "transfer" of 0.02–0.08 is comparable to within-pair slopes for several pairs (e.g. height→size = 0.040 ≈ height→height = 0.040 on gemma2-2b). Without multi-seed variance the matrix is uninterpretable; the 8×8 figure could be all sampling noise.
**JSON:** `results/v11/{gemma2-2b,gemma2-9b}/cross_pair_transfer_dense.json`
**Fix:** Re-run with ≥5 seeds, report mean ± SE per cell, and only claim transfer where off-diagonal > diagonal − 2·SE. Otherwise downgrade to "preliminary" or drop.

### 5. PC1 sign-flip + low PC1↔primal cosine on weak pairs
**Concern:** `cos_pc1_primal_per_layer` arbitrarily flips sign across layers (e.g. gemma2-2b/height: +0.997 L1, −0.997 L17, +0.997 L18) — PCA sign is gauge, fine, but the *magnitude* drops to 0.05–0.4 on age/size/speed at late layers (e.g. gemma2-9b/age L33 cos = 0.036; size L22 = 0.42). PC1 R²(z) for those pairs is correspondingly low (age: 0.21 in 2B, 0.61 in 9B; size: 0.075 / 0.66; speed: 0.36 / 0.43). Aggregate "PC1 ≈ primal_z" claim only holds for height/weight/wealth/experience — selection-biased reporting.
**JSON:** `results/v11/{gemma2-2b,gemma2-9b}/cos_pc1_primal_summary.json`
**Fix:** Report the per-pair R²(PC1, z) table prominently and acknowledge ~half of pairs do not have a clean PC1=z mapping; do not generalize from height to "all gradable adjectives."
