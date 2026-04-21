# TODO.md — Rolling checklist (Apr 21 2026)

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

## Queue — v8 GPU session

- [ ] **Priority 1: Direct sign classification** — 4 prompt variants testing whether R=0.47 for posneg is a prompt artifact vs genuine context-relativity. ~3 min GPU.
- [ ] **Priority 2: Top-K token analysis** — log top-10 predicted tokens per prompt to see what the model actually predicts. Bundled with Priority 1.
- [ ] **Priority 3: PCA horseshoe + SVD scree on Grid B** — fetch .npz from HF, run PCA/SVD/cross-pair cosines on CPU. PC1 vs PC2 scatter (horseshoe) needs regeneration.
- [ ] **Priority 4: Cross-template transfer test** — does primal_z transfer across different prompt templates? Red-team for syntax-vs-semantics. ~5 min GPU.

## Queue — Paper (May 3-8)

- [ ] Complete paper draft (lead with behavioral heatmap + causal steering)
- [ ] Submit ICML 2026 MI Workshop (May 8) — primary target
- [ ] Submit NeurIPS 2026 (May 4 abstract, May 6 full) — secondary
- [ ] Update `docs/paper_outline.md` with v7 findings

## Backlog

- [ ] Replace PCA with sparse factor model on 3500+ activations
- [ ] Layer sweep (early, mid, late, final) once we know which layers matter
- [ ] Context sample size saturation curve (n=5, 15, 50)
- [ ] Multilingual: Spanish "alto"/"bajo" relativity test
