# STATUS.md — Project status as of Apr 29, 2026

## Current phase

**v13 complete (OOD / X-TRANSFER / MIXED-POSITIVE).** Ran the minimum viable
GPU session from `docs/NEXT_GPU_SESSION_v13.md` on Gemma 2 9B. Outputs are under
`results/v13/`, `figures/v13/`, and `docs/V13_RESULTS_SUMMARY.md`.

- **Affine/OOD:** ordinary shifts and scales mostly preserve high corr(LD,
  z_eff), e.g. height stays +0.957 to +0.992 across all tested conditions.
  World-OOD weakens speed (corr +0.604) and experience (corr +0.281), so do not
  claim full affine invariance.
- **X-transfer control:** shared steering is much more `z`-specific than raw-x:
  off-diagonal transfer is +0.026 for `primal_z`, +0.006 for naive `primal_x`,
  and +0.004 for z-residualized `primal_x`; z beats x in 54/56 off-diagonal
  cells.
- **Domain/objective controls:** brightness extends cleanly (corr(LD,z)=+0.911),
  temperature is mixed (corr(LD,z)=+0.619, corr(LD,x)=+0.693), and objective
  controls track objective labels more than z. Use "robust but not fully
  affine-invariant" as the V13 headline.

**v12.2 complete (RESIDUAL-TRANSFER / MIXED).** Ran the residual-vs-lexical
cross-pair transfer follow-up from `docs/NEXT_GPU_SESSION_v12_2.md`. Outputs are
under `results/v12_2/`, `figures/v12_2/`, and
`docs/V12_2_RESULTS_SUMMARY.md`.

- **Residual vs lexical transfer:** in this single-seed Gemma 2 9B L33 test,
  residualized directions retain broad off-diagonal transfer (mean +0.024,
  56/56 positive) and outperform lexical projection off-diagonal (mean +0.011).
  Full `primal_z` remains slightly stronger off-diagonal (mean +0.026), while
  lexical projection is strongest on-diagonal.
- **Leakage caveat:** residual off-diagonal transfer is still strongly
  correlated with target lexical-subspace overlap (r≈+0.79). This is evidence
  for a residual shared component, not proof of a clean non-lexical or
  target-lexical-independent shared code.

**v12.1 complete (LEXICAL-DISENTANGLEMENT / MIXED).** Ran the narrow follow-up
from `docs/NEXT_GPU_SESSION_v12_1.md`. Outputs are under `results/v12_1/`,
`figures/v12_1/`, and `docs/V12_1_RESULTS_SUMMARY.md`.

- **Token-position capture:** literal adjective-token directions align only
  weakly with `primal_z` (mean cosine ≈ +0.10), while sentence-final states
  after the adjective align more strongly (mean cosine ≈ +0.26). This separates
  token position, not semantics: final punctuation can integrate adjective and
  sentence state.
- **Lexical subspace residualization:** the tested lexical subspace captures
  modest `primal_z` vector energy (mean norm² ≈ 0.08), but its projection is
  high-gain (mean projection/primal steering ≈ 1.25). The residualized direction
  still steers all eight pairs (mean residual/primal ≈ 0.69). Treat this as
  mixed-mechanism evidence, not proof of a clean non-lexical direction.

**v12 complete (CLAIM-HARDENED / MIXED).** Ran the planned Gemma 2 9B
claim-hardening pass from `docs/NEXT_GPU_SESSION_v12.md`. Outputs are under
`results/v12/`, `figures/v12/`, and `docs/V12_RESULTS_SUMMARY.md`.
Headline update: V12 supports early `z` decodability and later `primal_z`
steering potency, but it softens several stronger paper claims:

- **Layer sweep:** 9B `z` is linearly decodable very early, while `primal_z`
  steering peaks later (mean slope ≈ +0.097 at L25, +0.067 at L33). Keep this
  as a strategic-layer intervention result, not a fully identified circuit.
- **Lexical red-team:** simple raw-x and unembedding directions do not explain
  `primal_z`, but lexical sentence directions often steer as strongly as
  `primal_z`. Do not claim "not lexical semantics" without caveat.
- **Pure-x transfer control:** transfer persists on fixed-x/fixed-mu/matched-z
  subsets, but matched-z does not weaken cross-transfer. Treat this control as
  mixed, not as a clean scalar-magnitude refutation.
- **SAE audit:** z-correlated sparse features exist, but top features are a mix
  of pure-ish z, lexical z-like, raw numeric, and polysemantic features.
- **PC audit:** extremeness/curvature appears for some pairs, often PC2 or PC3,
  but there is no universal "PC2 = extremeness" result.

**v11.5 complete (SHARED-AMBER).** Re-ran the v4–v9 foundational research
questions on v11's enriched data. Headline results:

- **Domain-agnostic shared z-direction EXISTS.** Per-pair primal_z's
  are 55% aligned on average (cos(P_i,P_j) mean: 2B=+0.559, 9B=+0.516);
  a single Procrustes-aligned `w_shared` steers 6/8 pairs at 2B (7/8 at
  9B) at ≥50% within-pair efficiency. *Speed* and *experience* are the
  two pair-specific exceptions. FINDINGS §16.1.
- **Cross-pair transfer is statistically real.** 56/56 off-diagonal
  cells significant under BH-FDR q=0.05 on both models (5-seed
  multi-seed). FINDINGS §16.2.
- **z is available early, with most new linear information at the start.** Fold-aware
  P3c orthogonalized R² peaks at L1 (e.g. bmi_abs/2B=0.256, height/2B=0.145)
  and is near-zero at every later layer, while naive decodability is already
  high by the early layers. Read this as early availability plus carry-forward,
  not as proof that the whole computation is completed exactly at L1. FINDINGS
  §16.5.
- **Top SAE z-features pass raw-x/token controls, but V12 softens purity.**
  R²(z) ≈ 0.7–0.84 with R²(x), R²(token) ≈ 0 across all pairs/models
  for the top feature. 9B cross-pair Jaccard 0.22 (2× 2B's 0.11).
  FINDINGS §16.7. V12's lexical audit shows the top-feature population is
  mixed, so use "z-correlated sparse features" rather than "pure z features."
- **v10 §14.6 causal head taxonomy is TRIPLE-REFUTED.** Single-head
  ablations null (v11 §15.4); joint-tag-set ablations null on 2B and
  *helping* on 9B (v11.5 §16.3); permutation null on the taxonomy
  thresholds shows the tag counts are mostly chance-consistent
  (v11.5 §16.4).
- **primal_z is W_U-orthogonal but decision-aligned.** cos(primal,
  W_U[high]−W_U[low]) ≈ 0.15; cos(primal, leans-high−leans-low) ≈
  0.7–0.86 across most pairs/models. The direction that carries the
  high-vs-low semantic decision is *not* the same object as the
  unembedding readout. FINDINGS §15.3 + §16.6.

Prior phase: **v11 cross-model dense (RIPE-MANGO).** 8 pairs × 2 models
× 4,000 prompts. Behavioral 8/8 R(z) ≥ 0.92 on both models; 9B
replicates more uniformly than 2B. 5-critic post-hoc round drove
the v11.5 follow-up. FINDINGS §15.

## What's done

- v0/v1 behavioral kill-tests
- v2 prompt generator + Gemma 4 activations
- v4 dense + 8-pair extraction
- v5/v6/v7/v7b red-team and clean-grid
- v8 direct sign / cross-template / PCA horseshoe
- v9 Gemma 2 2B replication, SAE decomposition, on-manifold + Park steering, robustness, layer sweep (FINDINGS §10–§13)
- v10 dense-height deep dive (DENSE-MANGO; FINDINGS §14)
- v10 reproducibility close (re-extracted NPZs, uploaded to HF) — RIPE-MANGO step 1
- v11 cross-model dense + 5-critic post-hoc round (FINDINGS §15) — RIPE-MANGO step 2
- **v11.5 v4–v9 question replication on enriched data** (FINDINGS §16): shared z-direction, multi-seed cross-pair transfer with FDR, joint head ablation, fold-aware P3c, widened P3d, SAE token-freq control, bootstrap CIs throughout — SHARED-AMBER
- **v12 claim-hardening pass**: 9B strategic-layer sweep, direction red-team
  against raw-x/lexical probes, pure-x/fixed-mu transfer controls, SAE lexical
  audit, and PC extremeness/x audit. See `docs/V12_RESULTS_SUMMARY.md`.
- **v12.1 lexical disentanglement follow-up**: token-position lexical capture
  and lexical-subspace residualization of `primal_z`. See
  `docs/V12_1_RESULTS_SUMMARY.md`.
- **v12.2 residual-vs-lexical cross-pair transfer**: single-seed L33 transfer
  matrices for full `primal_z`, lexical projection, lexical residual, and
  target lexical-subspace leakage. See `docs/V12_2_RESULTS_SUMMARY.md`.

## Retracted (replace in any draft)

- **v10 §14.6 causal head taxonomy framing** (L13h2 / L3h0 / L0h6 as
  "causally necessary"). Refuted three independent ways in v11/v11.5
  — single-head ablations within ~2σ of zero, joint tag-set ablations
  null on 2B and helping on 9B, permutation null shows tag intersections
  largely chance-consistent. Re-frame as DLA-correlational only.
- **v11 §15.5 single-seed cross-pair transfer**. Now upgraded to multi-
  seed BH-FDR-controlled (v11.5 §16.2) — 56/56 cells significant.
- **v11 §15.7 P3c "results"** (negative out-of-sample R²). Pipeline bug
  fixed in v11.5 §16.5; the actual story is L1 = encoding, then
  carry-forward.
- **v11 §15.3 P3d ambiguous-cells NaN**. Fixed in v11.5 §16.6 via
  widened |z|<0.7 + LD-sign substitute for argmax.

## What's next

1. **Paper writing** (May 4 abstract → May 8 ICML MI Workshop).
   Headline §16.1 (shared z-direction) + §16.2 (FDR-controlled
   transfer) + §15.3 + §16.6 (W_U-orthogonal but decision-aligned
   primal_z), with V12 caveats: lexical sentence directions remain a serious
   competitor, V12.1/V12.2 show high-gain lexical projections and surviving
   residual steering/transfer, but target lexical overlap remains a major
   confound. Pure-x transfer controls are mixed, and SAE/PC interpretations
   must be framed as z-correlated/mixed rather than pure mechanisms. Three-way
   refutation of §14.6 in the Limitations section. Bootstrap CIs everywhere per
   §16.8.
2. **arXiv v2 follow-ups** (post-May-7): stronger zero-shot raw-x controls,
   bootstrap/null bands for V12 steering controls, 9B pure-ish-z feature
   asymmetry, and speed/experience pair-specific direction analysis.

## Archived session logs

`docs/archive/` contains GPU-burst session logs and PR descriptions.
