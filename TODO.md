# TODO.md — Rolling checklist (Apr 22 2026)

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
- [x] **v9 robustness (critic-driven)**: SAE sensitivity (layer 13 + lower L0 + W_dec projection), Park at layer 25 + λ sweep, behavioral bootstrap + Grid-B simulation, extended α + multi-seed null + 5-fold CV. FINDINGS §11.
- [x] **v9 SAE geometry + LFP** (Goodfire-style): SAE-basis PCA (worse than raw), per-pair LFP Gram (ID ≈ 4.2 / 5, near-orthogonal z-probes), cross-pair 40-probe Gram (ID = 26 / 40). FINDINGS §12.

## Queue — Paper (May 3-8)

- [ ] Complete paper draft (lead with behavioral heatmap + causal steering + SAE decomposition)
- [ ] Submit ICML 2026 MI Workshop (May 8) — primary target
- [ ] Submit NeurIPS 2026 (May 4 abstract, May 6 full) — secondary
- [ ] Update `docs/paper_outline.md` with v7-v9 findings

## Backlog

- [ ] Context sample size saturation curve (n=5, 15, 50)
- [ ] Multilingual: Spanish "alto"/"bajo" relativity test
- [ ] Gemma 3 4B + Gemma Scope 2 SAEs (if Gemma 2 2B signal is weak)
