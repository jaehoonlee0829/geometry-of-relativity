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
- [x] **v11.5 §A: shared z-direction** (SHARED-AMBER). Per-pair primal_z's are 55% aligned; `w_shared` (Procrustes-aligned mean) steers 6/8 pairs at 2B (7/8 at 9B) at ≥50% within-pair efficiency. Speed/experience are pair-specific exceptions. FINDINGS §16.1.
- [x] **v11.5 §B: taxonomy permutation null** — only 9B comparator tag count is structurally non-random; combined with §I, the structural signal is not load-bearing causally. FINDINGS §16.4.
- [x] **v11.5 §C+§D: multi-seed cross-pair steering with BH-FDR**. 56/56 off-diagonal cells significant on both models. Within/off ratios reveal speed/experience asymmetry. FINDINGS §16.2.
- [x] **v11.5 §E: SAE features with token-frequency control**. Top z-features are pure-z (R²(z) ≈ 0.7–0.84, R²(x), R²(token) ≈ 0). 9B cross-pair Jaccard 0.22 vs 2B 0.11. FINDINGS §16.7.
- [x] **v11.5 §F: bootstrap CIs throughout**. Every PC1.R²(z) gets a 95% block-bootstrap CI; every head-ablation Δr gets a Fisher-z CI. FINDINGS §16.8.
- [x] **v11.5 §G: fold-aware P3c**. Bug fixed; orthogonalized R²(z) peaks at L1 (e.g. bmi_abs/2B = 0.256), then near-zero — z is encoded in one shot at L1 then carried forward. FINDINGS §16.5.
- [x] **v11.5 §H: P3d widened ambiguous cells**. Recovers signal: cos(primal, leans-high−leans-low) ≈ 0.7–0.86 across most pairs/models. primal_z is W_U-orthogonal but decision-aligned. FINDINGS §16.6.
- [x] **v11.5 §I: joint head-set ablation with held-out split**. Δcorr(z) is null on 2B and *helping* on 9B (ablating 32 heads raises corr by +0.016). v10 §14.6 causal taxonomy triple-refuted. FINDINGS §16.3.

## Retract / re-frame (concluded; bake into the paper draft)

- [x] **Retract v10 §14.6 "causal head taxonomy" framing.** Triple-refuted: single-head ablations within ~2σ of zero (v11 §15.4), joint tag-set ablations null on 2B and helping on 9B (v11.5 §16.3), permutation null shows tag intersections largely chance-consistent (v11.5 §16.4). Re-frame as DLA-correlational only.
- [x] **Retract v11 P3c "orthogonalized increment R²" results.** Pipeline bug fixed in v11.5 §16.5 (fold-aware residualization). Use the v11.5 numbers — peaks at L1, then near-zero — not the v11 broken ones.
- [x] **Retract v11 §15.5 single-seed cross-pair transfer framing.** Replaced by v11.5 §16.2 multi-seed BH-FDR (56/56 cells significant).

## Queue — Paper (May 3-8)

- [ ] Complete paper draft. Headline §16.1 (shared z-direction) + §16.2 (FDR-controlled cross-pair transfer) + §15.3 (dense geometry/behavior) + §16.5 (z encoded at L1, then carried forward) + §16.7 (top SAE features are pure-z). Treat §16.6 / W_U-orthogonal-but-decision-aligned primal_z as a supporting control, not a headline. v10 §14.6 causal framing in Limitations (triple-refuted). Bootstrap CIs everywhere per §16.8.
- [ ] Submit ICML 2026 MI Workshop (May 8) — primary target.
- [ ] Submit NeurIPS 2026 (May 4 abstract, May 6 full) — secondary.
- [ ] Update `docs/paper_outline.md` with v9–v11.5 findings.

## Queue — GPU figures for paper/README

- [ ] **Regenerate the v9 2×3 layer-sweep figure on v11 Gemma 2 9B.** Target output: `figures/v11_5/layer_sweep_9b_combined.png`, matching the explanatory structure of `figures/v9/layer_sweep_combined.png` but using the dense v11/v11.5 9B substrate. Minimum panels: cumulative `R²(z)` by layer, fold-aware new `z` information by layer, causal steering strength by layer for `primal_z` vs Ridge probe, `||primal_z||` by layer, and layer-to-layer direction stability. CPU can plot from JSON, but the missing ingredient is per-layer steering; run GPU hook sweeps at strategic/all layers and write a JSON summary first.

## arXiv v2 follow-ups (post-May-7)

- [ ] **Pure-x control on §16.2's transfer matrix.** Re-run cross-pair steering with μ held constant in target prompts to rule out the alternative critic's residual "shared numeral-magnitude direction" cheap explanation. The fact that the absolute-adjective bmi_abs aligns with the relative pairs at 0.65–0.77 ratio (§16.1) already partially refutes that hypothesis, but the explicit μ-fixed test is the gold standard.
- [ ] **Dense v11 zero-shot / in-context x-vs-z red-team.** Re-run the v8 zero-shot direction comparison on the dense v11 substrate with a clearer plot and stronger controls: compare `primal_x`, `probe_x`, and zero-shot `w_x` against `primal_z` / `probe_z`; include chance/null bands, bootstrap CIs, and in-context `w_x_context` vs `w_z_context`. The current v8 plot uses supervised zero-shot `w_x` plus zero-shot PC1, and is useful but not visually/statistically strong enough as a headline figure.
- [ ] **SAE feature interpretation audit beyond numeral controls.** For top z-SAE features, collect top-activating prompts/tokens and test lexical controls for adjective/domain words such as `tall`, `short`, `height`, `old`, `young`, etc. Current v11.5 controls rule out raw x and numeric-token magnitude, but do not yet distinguish pure relativity features from lexical/domain-semantic features.
- [ ] **Positive/negative sign control cleanup.** Keep v8 direct-sign as a measurement-warning follow-up only. If reused, rerun with top-K validation, forced-choice prompts, and clearer accuracy-vs-relativity framing before including it in paper claims.
- [ ] **9B pure-z feature count asymmetry.** 9B has 1–16 pure-z features per pair while 2B has 11–50. Investigate: smaller k or larger superposition in 9B SAEs? Different SAE training regime? Compare with width_131k SAE if available.
- [ ] **Speed and experience pair-specific direction analysis.** These two pairs are the exceptions to domain-generality (§16.1 ratios 0.27/0.44 and 0.50/0.42 across models). Vehicle vs person framing in speed; experience-domain shift. Worth a focused mini-study.
- [ ] **|LD|-quantile sensitivity for §16.6.** The widened P3d uses bottom-40% |LD| as the "ambiguous" threshold; report cos sensitivity across {20%, 40%, 60%}.

## Backlog (deferred — out of scope)

- [ ] Context sample size saturation curve (n=5, 15, 50)
- [ ] Multilingual: Spanish "alto"/"bajo" relativity test
- [ ] Gemma 3 / Gemma 4 family replication (no SAEs yet)
- [ ] Fine-tuned model comparison (instruction-tuned vs base)
