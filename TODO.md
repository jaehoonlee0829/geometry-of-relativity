# TODO.md — Rolling checklist (Apr 25 2026)

## Done

- [x] Behavioral kill-tests v0/v1
- [x] Scaffold `src/` modules: data_gen, fisher, probe, activation_extract, plots
- [x] v2 prompt generator + Gemma 4 extraction (E4B + 31B)
- [x] v4 dense extraction (GARNET-ANVIL)
- [x] v4 auto-research suite (OBSIDIAN-LATTICE): 8-pair probes, PCA, steering, INLP
- [x] v5 red-team follow-up (NEXT-ECLIPSE): meta w1 steering, Fisher/Park, random null, G31B, critic consensus
- [x] v6 red-team: 7-direction analysis, confound discovery
- [x] v7 clean-grid rerun: Grid B extraction, confound audit, INLP, Fisher, steering, transfer
- [x] v7b: fix residual confound for experience/size
- [x] v7 plot regeneration: all behavioral + geometry figures from JSON, confound matrix Grid B only
- [x] Zero-shot-corrected heatmaps: ld - zero_shot(x) isolating pure context effect
- [x] Repo cleanup: archive old session docs, formalize structure
- [x] v8: direct sign classification (R=0.31 on valid prompt), PCA horseshoe on Grid B, SVD scree, cross-template transfer (97%), cross-pair PC1 cosine (0.19)
- [x] Fix meta_w1 SVD sign convention bug (sign-align after SVD)
- [x] Manifold geometry analysis: intrinsic dimensionality (~5-D), isomap vs PCA (speed: curved manifold), mid vs late layer orthogonality
- [x] **v9 P1: Replicate behavioral signal on Gemma 2 2B** — R>0.3 on 8/8 pairs (R=0.77-1.03).
- [x] **v9 P2: SAE feature decomposition of z** — distributed-not-sparse; primal vs probe participation 10k vs 18k features; cross-pair Jaccard 0.060.
- [x] **v9 P3: On-manifold tangent steering** — tangent/primal slope ≈ 0.69; entropy not systematically cleaner.
- [x] **v9 P4: Park's causal inner product** — (W_U^T W_U)^-1·probe_z does NOT close the 18× gap; cos(probe_causal, primal) < 0.05.
- [x] **v9 robustness (critic-driven)**: SAE sensitivity, Park at layer 25 + λ sweep, behavioral bootstrap + Grid-B simulation, extended α + multi-seed null + 5-fold CV. FINDINGS §11.
- [x] **v9 SAE geometry + LFP** (Goodfire-style): SAE-basis PCA (worse than raw), per-pair LFP Gram, cross-pair 40-probe Gram. FINDINGS §12.
- [x] **v9 full 26-layer sweep**: z encoded by L7, encode-vs-use as layer-depth phenomenon. FINDINGS §13.
- [x] **v10 dense-height deep dive** (DENSE-MANGO): 4,000-prompt 20×20×10 grid; cell-mean R(z)=0.972, peak primal steering at L14 (gap to probe 8×); attention DLA taxonomy; SAE features mostly monotonic. FINDINGS §14.
- [x] **v10 reproducibility close (RIPE-MANGO step 1)**: re-extracted dense-height NPZs after the prior Vast.ai instance died, uploaded to HF dataset. corr(LD,z)=0.972 byte-faithful.
- [x] **v11 cross-model dense (RIPE-MANGO step 2)**: 8 pairs × 2 models × 4,000 prompts. 8/8 pairs cell-mean R(z)≥0.92 on both 2B and 9B; 9B replicates more uniformly than 2B; primal_z is W_U-orthogonal at every layer. FINDINGS §15.
- [x] **v11 P6 critic round**: 5 post-hoc Anthropic-API critics (methodology / alternative / statistical / novelty / narrative). Reports under `results/v11/critic_*.md`.

## Retract / re-frame (driven by v11 P6 critic round)

- [ ] **Retract v10 §14.6 "causal head taxonomy" framing.** Single-head ablations (L13h2 / L3h0 / L0h6 on 2B; top derived heads on 9B) yield Δcorr(z) ≤ 2σ on both models. Re-write FINDINGS §14.6 to describe these as DLA-correlational tags, not causal mechanisms.
- [ ] **Retract v11 P3c "orthogonalized increment R²" results.** The residualization is not fold-aware → out-of-sample R² goes negative (down to −23.5). Pipeline bug, not "no new info." Either re-implement (see v11.5-G below) or drop from FINDINGS §15.

## Queue — Paper (May 3-8)

- [ ] Complete paper draft (lead with §15.1 8/8 behavioral + §15.2 9B uniformity + §15.3 W_U-orthogonality of primal_z; retract §14.6 causal framing).
- [ ] Submit ICML 2026 MI Workshop (May 8) — primary target.
- [ ] Submit NeurIPS 2026 (May 4 abstract, May 6 full) — secondary.
- [ ] Update `docs/paper_outline.md` with v9–v11 findings.

## v11.5 follow-up backlog (post-submission, arXiv v2)

The post-hoc critic round + user feedback identified a comprehensive set
of v4–v9-style research questions that should be re-run on the enriched
v11 data. Sequencing per FINDINGS §15.9.

- [ ] **A. Domain-agnostic shared z-direction.** Build `w_shared` from per-pair primal_z directions (mean / first-PC / Procrustes); steer all 8 pairs simultaneously; measure per-pair slope. Claim "domain-general z-code" only if shared slope ≥50% of within-pair slope on 8/8 pairs.
- [ ] **B. Attention-head mapping of `w_shared`.** Re-derive head DLA jointly across all 8 pairs (not per-pair). Permutation null on top-quartile thresholds (1000 shuffles).
- [ ] **C. Multi-seed cross-pair steering with FDR control.** ≥5 seeds; block-bootstrap CIs; Bonferroni or BH-FDR at q=0.05 across the 56 off-diagonal cells per model. Pure-x control (μ held constant) to distinguish "shared z-code" from "shared numeral-magnitude direction."
- [ ] **D. Cross-feature steering effects.** When steering with `w_shared`, measure off-target pair LD changes. Distinguish transfer (correct-sign LD shift) from interference (orthogonal shift).
- [ ] **E. SAE-feature replication of v9 §12 / §14.5.** Replicate the 0.06 cross-pair Jaccard at v11's higher N. Add R²(numeral_token_id | feature) control to disambiguate from "feature tracks numeral magnitude."
- [ ] **F. Statistical hardening.** Bootstrap CIs on every reported R² and cosine; Fisher-z CIs on ablation deltas; PCA eigenvalue-gap stability under subsampling; TWO-NN ID stability across N.
- [ ] **G. P3c fold-aware orthogonalized R².** Within-fold residualization (fit ẑ on train, project on test). Re-run on all 8 pairs × 2 models.
- [ ] **H. P3d ambiguous-cells fix or drop.** At |z| < eps the model never argmaxes to the polar word — widen the band, use top-K predictions, or drop the variant.
- [ ] **I. Joint head-set ablation.** Single-head ablations are null; ablate full tag-sets jointly (all comparators / all z-writers) to test for head redundancy. Held-out threshold split (pick on half, ablate on the other half) to break circularity.

## Backlog (deferred — out of scope for v11.5)

- [ ] Context sample size saturation curve (n=5, 15, 50)
- [ ] Multilingual: Spanish "alto"/"bajo" relativity test
- [ ] Gemma 3 / Gemma 4 family replication (no SAEs yet)
- [ ] Fine-tuned model comparison (instruction-tuned vs base)
