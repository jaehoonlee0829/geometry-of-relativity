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

## Queue — v9 GPU session (see `docs/NEXT_GPU_SESSION_v9.md`)

- [ ] **Priority 1: Replicate behavioral signal on Gemma 2 2B** — enables SAE analysis via Gemma Scope. ~5 min GPU.
- [ ] **Priority 2: SAE feature decomposition of z** — Gemma Scope 65k SAE on Gemma 2 2B. Find z-correlated features, cross-pair overlap, place-cell vs linear, primal_z vs probe_z in SAE basis. ~10 min GPU + CPU analysis.
- [ ] **Priority 3: On-manifold steering** — steer along geodesic tangent vs fixed primal_z. Compare entropy damage. ~5 min GPU.
- [ ] **Priority 4: Park's causal inner product** — transform probe_z by W_U metric. Test if causal-adjusted probe steers as well as primal_z. CPU + GPU.

## Queue — Paper (May 3-8)

- [ ] Complete paper draft (lead with behavioral heatmap + causal steering + SAE decomposition)
- [ ] Submit ICML 2026 MI Workshop (May 8) — primary target
- [ ] Submit NeurIPS 2026 (May 4 abstract, May 6 full) — secondary
- [ ] Update `docs/paper_outline.md` with v7-v9 findings

## Backlog

- [ ] Context sample size saturation curve (n=5, 15, 50)
- [ ] Multilingual: Spanish "alto"/"bajo" relativity test
- [ ] Gemma 3 4B + Gemma Scope 2 SAEs (if Gemma 2 2B signal is weak)
