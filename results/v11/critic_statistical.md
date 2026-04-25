# statistical critic

# Statistical Critique of v11

## 1. No CIs anywhere in the reported JSONs
Every R² in `pca_summary.json`, every cos in `cos_pc1_primal_per_layer`, every slope in `cross_pair_transfer_dense.json`, and every `corr_z_after_ablation` in `head_ablation_causal.json` is reported as a point estimate. With N=400 cell-means (e.g. `n_groups: 400` for height, 351 for experience, 362 for size), bootstrap CIs are cheap and essential. E.g. the headline `PC1_vs_z = 0.9690` for height and `PC1_vs_z = 0.9226` for bmi_abs cannot be statistically distinguished from each other without intervals. The "size" claim of `PC1_vs_z=0.075, PC2_vs_z=0.785` (gemma2-2b) is a strong qualitative claim about axis-swapping that needs a bootstrap on the eigenvalue gap (`evr = [0.416, 0.307]` — only a 0.11 gap, likely unstable).
**Fix:** 1000-sample block bootstrap over (μ,x) cells; report 95% CI on every R² and on `λ₁−λ₂`.

## 2. Cross-pair transfer matrix: 56 off-diagonal cells, no Bonferroni
`cross_pair_transfer_dense.json` reports raw slopes ranging −0.018 to +0.080. Without a null distribution per cell, "off-diagonal ≈ 0" is unsupported. With 56 comparisons, Bonferroni α=0.05/56 ≈ 8.9e-4 — meaning each cell needs ~3.3σ. Several off-diagonals (e.g. `size→weight = 0.080`, `size→height = 0.078` for gemma2-2b) are *within 25%* of within-pair slopes (`height = 0.040`, `weight = 0.058`). The transfer story is not as clean as the diagonal-vs-off-diagonal narrative implies.
**Fix:** permutation null (shuffle source labels) for each (source,target) cell; FDR-control (Benjamini–Hochberg) at q=0.05 across 56 tests; report which cells survive.

## 3. N=400 too small for a "7-D intrinsic manifold" claim
v10 retired the v9 hunchback as a 25-pt TWO-NN artefact; v11/v10 still uses ≤400 cell-means. TWO-NN's bias at N=400 in d=2304 ambient (gemma2-2b) is non-trivial; the literature (Facco 2017) recommends N>>2^d_int, and stability under subsampling is the standard check. No `id_bootstrap.json` is shown.
**Fix:** subsample N∈{100,200,300,400} ×50 reps; report ID vs N curve. Claim only the asymptote.

## 4. Head-taxonomy "top-quartile" thresholds have no null
`head_taxonomy.json` uses `ctx=0.0308, tgt=0.0530, dla_abs=0.0837` — these are just the 75th percentile of the empirical distribution over 64 heads, so by construction ~16 heads will be tagged on each axis. The reported "15 μ-aggregators, 18 comparators, 5 z-writers" is *mechanically near* the quartile expectation (16). The intersection counts (5 z-writers requires top-quartile on multiple axes) need a permutation null.
**Fix:** shuffle (layer,head)→metric assignments 1000×; report whether observed co-tag counts (esp. the 5 z-writers and any μ-aggregator∩comparator overlaps like L0h6) exceed the 95th percentile of the null. Without this, taxonomy is descriptive, not inferential.

## 5. Causal ablation deltas are within noise
`head_ablation_causal.json`: baseline `corr_z = 0.9757`; ablating the "comparator" L13h2 yields Δ = **−0.0032**, "early_writer" L3h0 Δ = **−0.0075**, "mu_aggregator" L0h6 Δ = **+0.0002** (helps!). On gemma2-9b, two of three ablations have Δ > 0. With N=400 and r≈0.97, the SE on Pearson r is ≈(1−r²)/√N ≈ 0.003 — i.e. *every* delta is ≤2 SE. The "concrete head taxonomy" headline is not causally supported.
**Fix:** Fisher-z transform Δr, compute SE analytically, report 95% CI; or bootstrap the per-prompt residuals. Likely none of the three reach significance — adjust the FINDINGS narrative accordingly, or ablate head-sets jointly to get larger effect sizes.
