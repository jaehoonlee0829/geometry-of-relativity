# STATUS.md — Project status as of Apr 22, 2026

## Current phase

**v10 complete (DENSE-MANGO).** Dense single-pair deep dive: height
only, Gemma 2 2B, 4,000 prompts, 20×20×10 grid. **Confirms encode-vs-use
as layer-depth phenomenon at higher density (cell-mean R(z)=0.972, peak
primal steering at L14 with 8× probe gap), retires the v9 mid-network
TWO-NN hunchback (low-N artefact), adds an attention-head taxonomy with
explicit μ-aggregator/comparator/z-writer candidates (DLA on dumped
activations, faithfulness 0.67).** FINDINGS §14.

Prior phase: **v9 complete, critic-reviewed, robustness checks incorporated, full
26-layer sweep added.** Four headline claims in §10; each red-teamed
by 3 critics and re-verified in §11. SAE-geometry + Goodfire-style
LFP in §12. Full layer sweep in §13.

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
- **v9 Gemma 2 2B replication (8/8 pairs R>0.3)**
- **v9 SAE decomposition**: primal_z is 1.2–1.8× more concentrated than probe_z but both distributed across thousands of features — sparse-SAE hypothesis refuted
- **v9 on-manifold tangent steering**: tangent steers at 0.63–0.73× primal; NOT systematically kinder to entropy — on-manifold hypothesis refuted
- **v9 Park causal steering**: (W_U^T W_U)^{-1}·probe_z does NOT bridge the 18× probe/primal gap — Park hypothesis refuted
- **v10 dense-height** (DENSE-MANGO): 4,000-prompt 20×20×10 grid; encode-vs-use peak shifted L14 (vs v9 L20-22), 8× probe/primal gap reproduced; v9 ID hunchback was a 25-pt TWO-NN artefact; attention DLA produces explicit head taxonomy.

## What's next

1. **Paper writing**: ICML MI Workshop (May 8), NeurIPS 2026 (May 4/6).
   The workshop paper should lead with behavioral R heatmap + causal
   steering + §10's retirement of three alternative explanations.
2. **Optional v10 follow-up** (arXiv only): what IS primal_z, if not
   a sparse SAE bundle / tangent / causally-rotated probe? Candidates:
   attention-weighted mean-diff, layer-aggregated direction, or a
   non-linear decoder.

## Archived session logs

Detailed session logs and PR descriptions from GPU rental bursts are in
`docs/archive/`.
