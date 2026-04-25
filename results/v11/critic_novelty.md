# novelty critic

# Novelty Critique of v11

## Claim 1: "Encode-vs-use as a layer-depth phenomenon" (encode by L7, use from L11+, peak L14)
- **Novelty: 2/5**
- **Closest prior:** Geva et al. 2021 ("FFN as key-value memories") + Geva 2022 ("promotes concepts in vocab space") already establish that early layers build features and late layers project to logits. nostalgebraist's logit-lens (2020) and Belrose et al.'s tuned-lens (2023) operationalize the same encode→use depth gradient. Lad et al. 2024 ("four universal stages") names the phases explicitly. The probe/primal gap (8× at L14, `cos_pc1_primal_per_layer` height L1=0.997 vs late flips) is a quantitative refinement, not a new phenomenon.
- **Recommendation:** Reframe as "graded-quantity-specific encode-use depth profile" and cite Lad/Geva/tuned-lens explicitly. The novelty is the *Z-score-vs-raw* axis, not the layer-depth observation per se.

## Claim 2: W_U-based lexical-vs-z disentanglement (P3d-fixed)
- **Novelty: 3/5**
- **Closest prior:** Park, Choe & Veitch (2024) "The Linear Representation Hypothesis" defines the causal inner product via `Cov(γ)⁻¹` on unembedding-space differences; Marks & Tegmark's "Geometry of Truth" projects probe directions onto unembedding subspaces. The v11 file `z_vs_lexical_per_layer.json` shows `cos_primal_lexical_unembed` ≈ 0 at all layers (range −0.045 to +0.058 for height) — i.e., primal_z is **orthogonal to the tall/short unembedding axis**. That's a substantive, non-trivial result Park's framework predicts but doesn't directly demonstrate for graded scalars.
- **Caveat:** All `cos_primal_lexical_zero_z` are NaN (n_pred_high_amb=0) — the "ambiguous-prediction" disentanglement actually failed to fire; only the unembed-cosine variant works.
- **Recommendation:** Lead with the orthogonality result, frame it as an empirical instantiation of Park's framework on a new (graded scalar) axis. Honest about the NaN failure of the ambiguous-prediction probe.

## Claim 3: Gemma-2-9B replication
- **Novelty: 1/5 (as a contribution); 4/5 (as evidence)**
- **Closest prior:** Standard practice — every Anthropic / Goodfire scaling result does this.
- **Numbers check out:** PC1_vs_z height 9B=0.928 vs 2B=0.969; within-pair steering slopes (cross_pair_transfer_dense.json) 9B=0.044 vs 2B=0.040 for height. Consistent.
- **Recommendation:** **Appendix only** for workshop. Do not headline. The 9B age PCA (PC1_vs_z=0.61, PC1_vs_x=0.05, PC2_vs_z=0.20) actually muddies the story — age looks qualitatively different at 9B.

## Claim 4: Attention-head taxonomy (μ-aggregators / comparators / z-writers) + causal ablation
- **Novelty: 2/5**
- **Closest prior:** Wang et al. IOI (name-mover, S-inhibition, duplicate-token heads); Lieberum et al. 2023 multiple-choice circuit; Hanna et al. "greater-than" circuit (2023) — *directly* analogous (numerical comparison heads in GPT-2). The taxonomy categories are renamings of well-known head-type templates.
- **Damning:** `head_ablation_causal.json`: ablating "comparator" L13h2 changes corr_z by **−0.003** (0.976→0.973); mu_aggregator ablation is **+0.0002** (no effect). The causal evidence contradicts the taxonomy. 9B is worse: comparator ablation is **+0.002**.
- **Recommendation:** Either run proper attribution patching / path patching (à la Hanna 2023) or **drop the causal-circuit claim**. Keep DLA taxonomy as descriptive only.

## Claim 5: Cross-pair transfer asymmetry (height/weight low, size/experience high)
- **Novelty: 3/5**
- **Closest prior:** Marks et al. "linear feature transferability"; Tigges et al. sentiment-direction transfer. Within-pair slopes 0.04–0.10 (`within_pair_slope`) with off-diagonal 0.005–0.08 is a real pattern not previously reported for graded scalars.
- **Recommendation:** Workshop-worthy as a secondary figure. Frame as "graded-scalar directions are pair-specific, unlike sentiment."
