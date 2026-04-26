# STATUS.md — Project status as of Apr 25, 2026

## Current phase

**v11 complete (RIPE-MANGO).** Cross-model dense extraction across 8
pairs × 2 models (Gemma 2 2B + Gemma 2 9B), 4,000 prompts per (pair,
model) cell, 96k forward passes total + per-pair HF upload. Pre-flight
3-critic round drove three methodology fixes; post-hoc 5-critic round
flagged four issues that bound the headline claims. **What survived:
8/8-pair behavioral signal on both models (cell-mean R(z) ≥ 0.92),
9B replicates more uniformly than 2B (PCA R²(z) median 0.87 vs 0.90
but 0.43–0.94 spread vs 0.075–0.97 spread), and the methodology-fixed
P3d shows primal_z is W_U-orthogonal at every layer (cos in [−0.05,
+0.12]) — the genuinely novel piece.** What didn't: single-head
ablations of the v10 §14.6 canonical heads (L13h2/L3h0/L0h6) are null
on both models (Δcorr_z ≤ 2σ); **the "causal head taxonomy" framing
is retracted**. P3c orth-R² has a residualization-fold bug; P3d
ambiguous-cells variant returned NaN; P3e cross-pair transfer is
single-seed without Bonferroni control. FINDINGS §15.

Prior phase: **v10 complete (DENSE-MANGO).** Dense single-pair deep
dive: height only, Gemma 2 2B, 4,000 prompts, 20×20×10 grid. Confirmed
encode-vs-use as layer-depth phenomenon (cell-mean R(z)=0.972, peak
primal steering at L14 with 8× probe gap), retired the v9 mid-network
TWO-NN hunchback (low-N artefact), added an attention-head taxonomy
with explicit μ-aggregator/comparator/z-writer candidates (DLA on
dumped activations, faithfulness 0.67). FINDINGS §14. v10 NPZs
re-extracted and uploaded to HF on 2026-04-25 after the previous
Vast.ai instance died before upload.

## What's done

- v0/v1 behavioral kill-tests (Claude Opus 4.5 + Sonnet 4.6)
- v2 prompt generator + Gemma 4 activation extraction (E4B + 31B)
- v4 dense extraction (3540 prompts) + 8-pair adjective extraction (6240 prompts)
- v5 red-team follow-up: meta-direction steering, Fisher/Park metrics, random-null control, G31B scaling, critic consensus
- v6 red-team: 7-direction analysis, confound discovery (Grid A corr(x,z) = 0.58-0.86)
- v7 clean-grid rerun: Grid B (x, z) extraction, confound audit, INLP, Fisher, steering, cross-pair transfer
- v7b addendum: fixed residual confound for experience/size pairs
- v7 plot regeneration: all figures from pre-computed JSON (no GPU needed)
- v8: direct sign classification, PCA horseshoe on Grid B, SVD scree, cross-template transfer (97%), cross-pair PC1 cosine (0.19)
- Manifold geometry: ID ~5-D, isomap reveals curvature on speed (R²=0.97 vs PCA 0.01), mid ⊥ late layer primal_z
- v9 Gemma 2 2B replication (8/8 pairs R>0.3)
- v9 SAE decomposition: primal_z is 1.2–1.8× more concentrated than probe_z but both distributed across thousands of features — sparse-SAE hypothesis refuted
- v9 on-manifold tangent steering: tangent steers at 0.63–0.73× primal; NOT systematically kinder to entropy — on-manifold hypothesis refuted
- v9 Park causal steering: (W_U^T W_U)^{-1}·probe_z does NOT bridge the 18× probe/primal gap — Park hypothesis refuted
- v10 dense-height (DENSE-MANGO): 4,000-prompt 20×20×10 grid; encode-vs-use peak shifted L14 (vs v9 L20-22), 8× probe/primal gap reproduced; v9 ID hunchback was a 25-pt TWO-NN artefact; attention DLA produces explicit head taxonomy.
- **v10 reproducibility close (RIPE-MANGO step 1)**: re-extracted dense-height NPZs (corr=0.972 byte-faithful), uploaded to `xrong1729/mech-interp-relativity-activations`.
- **v11 cross-model dense (RIPE-MANGO step 2)**: 8 pairs × 2 models × 4,000 prompts; behavioral R(z) ≥ 0.92 on 8/8 pairs both models; 9B replicates more uniformly; primal_z is W_U-orthogonal (genuinely new); 5-critic post-hoc round flagged 4 issues. FINDINGS §15.

## Retracted / re-frame needed

- **v10 §14.6 "causal head taxonomy"** (L13h2 comparator, L3h0 early-writer, L0h6 μ-aggregator): single-head ablations on the v11 dense grid produce Δcorr(z) ≤ 0.008 on 2B and ≤ 0.003 on 9B — within ~2σ of zero. Re-frame these as DLA-correlational tags, not causal mechanisms.
- **v11 P3c orthogonalized increment R²**: residualization is not fold-aware → out-of-sample R² goes negative (down to −23.5). Pipeline bug; re-run with within-fold residualization or drop.
- **v11 P3d ambiguous-cells lexical variant**: returned NaN at every layer (model never argmaxes to high/low at |z|<eps). The W_U-cosine variant survives.
- **v11 P3e cross-pair transfer**: single-seed, no Bonferroni; alternative "shared numeral-magnitude" explanation untested.

## What's next

1. **Paper writing** (May 4 abstract → May 8 ICML MI Workshop deadline).
   Lead with §15.1 (8/8 behavioral), §15.2 (9B uniformity), §15.3
   (W_U-orthogonality). Retract §14.6 causal framing; describe head
   taxonomy as descriptive only.
2. **Optional v11 follow-up** (arXiv v2 only, post-May-7): fold-aware
   P3c, multi-seed P3e + Bonferroni, joint head-set ablation, ambiguous-
   cells P3d fix.

## Archived session logs

Detailed session logs and PR descriptions from GPU rental bursts are in
`docs/archive/`.
