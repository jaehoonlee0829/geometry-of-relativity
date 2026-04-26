# geometry-of-relativity

Mechanistic interpretability study of **how LLMs represent contextual relativity** — whether "tall" means tall-for-this-group or tall-in-absolute-terms — via activation geometry, causal steering, and SAE decomposition in Gemma.

**Target venue:** ICML 2026 MI Workshop (May 8 AOE), co-submission to NeurIPS 2026 main track.

## TL;DR

When a language model sees "Person 16: 170 cm. This person is ___", does it complete with "tall" based on the raw number (170 cm) or relative to the surrounding context (the other 15 people)?

We find:
- **A domain-agnostic shared z-direction exists** in both Gemma 2 2B and 9B. A single direction `w_shared` steers 6/8 pairs (2B) and 7/8 pairs (9B) at >=50% within-pair efficiency. Cross-pair transfer is **56/56 off-diagonal cells significant** under BH-FDR (q=0.05). Speed and experience are the two pair-specific exceptions.
- **z is encoded at L1 in one shot, then carried forward** (v11.5 fold-aware orthogonalized R^2). The earlier "by L7" plateau was cumulative carry-forward, not encoding. A simple mean-difference direction (`primal_z`) steers adjective output 10-100x stronger than the Ridge probe direction, with the probe/primal gap widening to ~8x in late layers.
- **Head taxonomy is triple-refuted**: single-head ablation null (v11), joint-head-set ablation null (v11.5), and permutation null on taxonomy thresholds. Even ablating ALL "z-circuit" heads on 9B *raises* corr(z).
- **primal_z is W_U-orthogonal but decision-aligned** (cos ~0.15 with `W_U[high] - W_U[low]`, but cos 0.47-0.86 with the data-derived "leans high vs leans low" axis). The lexical readout direction and the direction that carries the decision are different objects.
- **Top SAE z-features are pure-z, not numeral trackers** (R^2(x) and R^2(token) near zero for the top z-feature on every pair). 9B has 2x the cross-pair feature Jaccard of 2B (0.22 vs 0.11).

## Models

| Model | HuggingFace ID | Role |
|---|---|---|
| Gemma 2 2B | `google/gemma-2-2b` | Primary + SAE analysis via Gemma Scope |
| Gemma 2 9B | `google/gemma-2-9b` | Scaling comparison (42 layers, d=3584) |
| Gemma 4 E4B | `google/gemma-4-E4B` | Original extraction (42 layers, d=2560) |
| Gemma 4 31B | `google/gemma-4-31B` | Scaling comparison (60 layers, d=5376) |

## Setup

8 adjective pairs, each tested on a balanced (x, z) grid where x (raw value) and z (context-relative z-score) are independent by construction. v11 uses a dense 20x20 grid (4,000 prompts per pair per model) on all 8 pairs x 2 models (Gemma 2 2B + 9B), totalling ~64k prompts.

| Pair | Adjectives | v9 relativity ratio R | v11 cell-mean corr(LD, z) 2B | v11 cell-mean corr(LD, z) 9B |
|---|---|---|---|---|
| height | short / tall | 0.85 | 0.972 | 0.97+ |
| age | young / old | 1.03 | 0.93–0.96 | 0.93–0.97 |
| weight | light / heavy | 0.92 | 0.94–0.97 | 0.94–0.97 |
| size | small / big | 0.93 | 0.92–0.96 | 0.93–0.97 |
| speed | slow / fast | 0.77 | 0.93 | 0.94 |
| wealth | poor / rich | 0.77 | 0.95–0.97 | 0.95–0.97 |
| experience | novice / expert | 0.86 | 0.95–0.97 | 0.95–0.97 |
| bmi_abs | thin / obese | 0.83 | 0.953 | 0.95+ |

The v9 relativity ratio R = -c/b from `logit_diff ~ b*x + c*mu` measures how much the context mean shifts the model's adjective decision (R=1 means pure z-score). v11 cell-mean correlations confirm R(z) >= 0.92 on all 8 pairs on both models at dense resolution.

![behavioral heatmap](figures/v10/behavioral_logit_diff_xz.png)

## Key findings

### 1. z is encoded at L1 in one shot, then carried forward (the headline result)

Fold-aware orthogonalized increment R^2 (v11.5 FINDINGS section 16.5) reveals that the new z-information each layer adds is concentrated at **L1** (2B) / **L1-L3** (9B), then near-zero at every later layer. The earlier claim that z is "encoded by L7" was measuring cumulative carry-forward of L1's encoding, not fresh computation.

Meanwhile, primal_z steering is zero at layers 5-10. Causal potency emerges at layer 13, peaks at **layer 14** (v10 dense grid), and the probe/primal gap widens to ~8x in late layers. **The dimensions that encode z early are not the dimensions downstream layers read from.**

The full 26-layer sweep reveals a three-phase computation:

| Phase | Layers | What happens |
|---|---|---|
| **Encode** | L0-L1 | z computed from tokens in one shot (orth increment R^2 peaks at L1). |
| **Carry + Rotate** | L2-L14 | z is carried forward with minimal new info. Direction actively rotates (cos 0.3-0.5 between adjacent layers). Causal potency emerges at L13, peaks at L14. |
| **Broadcast** | L15-L25 | Direction locks (cos > 0.9). Primal_z amplified 400x from L0. Probe/primal gap widens to ~8x. |

![layer sweep](figures/v10/steering_layer_sweep.png)

### 2. Domain-agnostic shared z-direction

A single direction `w_shared` (Procrustes-aligned mean of the 8 per-pair primal_z directions) steers **6/8** pairs at >=50% within-pair efficiency on 2B and **7/8** on 9B (FINDINGS section 16.1). Pairwise mean cosine of per-pair primal_z directions is +0.56 (2B) and +0.52 (9B).

| pair       | 2B shared/within | 9B shared/within |
|---         |---:              |---:              |
| height     | **0.93**         | **0.75**         |
| weight     | **0.89**         | **0.80**         |
| size       | **0.87**         | **0.66**         |
| bmi_abs    | **0.77**         | **0.65**         |
| wealth     | **0.73**         | **0.70**         |
| age        | **0.60**         | **0.56**         |
| speed      | 0.44             | 0.42             |
| experience | 0.27             | 0.50             |

Multi-seed cross-pair transfer with BH-FDR correction at q=0.05 shows **all 56/56 off-diagonal cells significant** on both models (FINDINGS section 16.2). This is not single-seed noise.

*Speed* and *experience* are the two genuinely pair-specific exceptions. Notably, bmi_abs (the absolute-adjective control) aligns with the relative pairs at 0.65-0.77 ratio, ruling out the "shared numeral-magnitude direction" alternative.

![transfer heatmap 2B](figures/v11/steering/cross_pair_transfer_8x8_gemma2-2b.png)
![transfer heatmap 9B](figures/v11/steering/cross_pair_transfer_8x8_gemma2-9b.png)

### 3. z representation compresses monotonically

v10's 400-cell dense grid resolves the manifold geometry clearly: TWO-NN intrinsic dimensionality drops monotonically from 7.7 (L0) to 3.2 (L20). PCA-95% variance peaks at L7 (16 components) then compresses to 7. The v9 "hunchback" pattern (ID peaking mid-network at ~7) was a 25-point TWO-NN artefact that disappears with denser sampling.

Curvature evidence (v9 data, not re-tested in v10): for speed, isomap captures z with R^2=0.97 while PCA gets R^2=0.01 -- z is on a curve that linear methods miss.

![intrinsic dimensionality](figures/v10/id_per_layer_3methods.png)

### 4. primal_z is W_U-orthogonal but decision-aligned

cos(primal_z, W_U[high] - W_U[low]) falls in [-0.05, +0.12] across all layers on both models (FINDINGS section 15.3) -- primal_z is essentially orthogonal to the lexical readout direction at every depth. But it IS strongly aligned (cos 0.47-0.86) with the data-derived "leans high vs leans low" axis (FINDINGS section 16.6). primal_z carries the above-vs-below-norm semantic decision, just routed through a non-trivial projection before the final logit.

### 5. SAE z-features are pure-z, with 2x overlap at 9B

Top z-correlated SAE features have R^2(z) = 0.63-0.84 while R^2(x) and R^2(token-magnitude) are near zero (FINDINGS section 16.7). The alternative hypothesis that SAE features merely track numeral frequency is refuted at the top of the feature list.

Cross-pair top-50 Jaccard: 2B = 0.11, 9B = **0.22** -- 9B has twice the cross-pair SAE feature overlap, consistent with its stronger shared z-direction. Most z-features activate monotonically with z (r = 0.7-0.9), with rare place-cell exceptions (e.g., feature 34700: bump R^2=0.98, linear R^2=0.00).

![SAE overlap 2B](figures/v11/sae/cross_pair_feature_overlap_gemma2-2b.png)
![SAE overlap 9B](figures/v11/sae/cross_pair_feature_overlap_gemma2-9b.png)

### 6. 9B replicates more uniformly than 2B

PC1.R^2(z) on cell-means at the canonical late layer (2B L20, 9B L33):

| pair       | 2B PC1.R^2(z) [95% CI]    | 9B PC1.R^2(z) [95% CI]    |
|---         |---                        |---                        |
| height     | **0.969** [0.961, 0.975]  | **0.928** [0.907, 0.941]  |
| weight     | **0.949** [0.933, 0.960]  | **0.944** [0.930, 0.954]  |
| bmi_abs    | **0.923** [0.876, 0.956]  | **0.784** [0.750, 0.813]  |
| experience | **0.901** [0.865, 0.928]  | **0.902** [0.846, 0.930]  |
| wealth     | **0.855** [0.768, 0.908]  | **0.871** [0.838, 0.897]  |
| speed      | 0.360 [0.015, 0.627]      | **0.428** [0.271, 0.582]  |
| age        | 0.209 [0.091, 0.341]      | **0.606** [0.003, 0.843]  |
| size       | 0.075 [0.000, 0.254]      | **0.656** [0.012, 0.853]  |

2B has median 0.90 but three pairs (age, size, speed) fail (R^2 < 0.4). 9B rescues all three (R^2 0.43-0.66). Bootstrap CIs confirm: 2B size [0.000, 0.254] is not statistically distinguishable from zero; 9B size [0.012, 0.853] is wide but nonzero. **Scaling rescues the z-code on the harder pairs.**

![PCA 2B height](figures/v11/pca/height_gemma2-2b_2d_L20.png)
![PCA 9B height](figures/v11/pca/height_gemma2-9b_2d_L33.png)

## Three hypotheses tested and refuted

### On-manifold tangent steering

Tangent(z) steers at 0.63-0.73x of primal_z. At low alpha, entropy damage is similar; at high alpha (=8), tangent is kinder on 6/8 pairs but the effect is modest (0.1-0.6 nats). Not the clean win predicted.

### Park's causal inner product

(W_U^T W_U)^{-1} * probe_z does NOT rotate probe toward primal. cos(probe_causal, primal) < 0.05 across all pairs, at both layer 20 and the theoretically-favored layer 25, across a lambda sweep from 10^-5 to 10.

### Causal head taxonomy (triple-refuted)

v10's DLA-based taxonomy identified mu-aggregators, comparators, and z-writers across 38 heads. v11 and v11.5 conclusively refute the causal framing:
1. **Single-head ablation null** (v11 FINDINGS section 15.4): ablating any individual tagged head (L13h2, L3h0, L0h6 on 2B; L21h3, L16h3, L0h3 on 9B) changes corr(z) by at most 0.008 -- within 2 SE.
2. **Joint-head-set ablation null** (v11.5 FINDINGS section 16.3): ablating the *union* of all z-circuit heads (18 on 2B, 32 on 9B) produces Delta corr(z) = -0.009 on 2B and **+0.016** on 9B. Even removing all tagged heads on 9B *raises* the z-correlation.
3. **Permutation null** (v11.5 FINDINGS section 16.4): only 9B's comparator count exceeds the 95th percentile of shuffled assignments; mu-aggregator and z-writer counts are at chance on both models.

The taxonomy describes correlational patterns, not causal mechanisms. The z-code is highly redundant across heads.

## Honest negatives

- **Fisher pullback (H4) refuted.** F(h) near-isotropic at tested activations.
- **Relative/absolute dichotomy not significant** (n=7 vs 4, p=0.75).
- **PC1~z not robust for size/speed at 2B.** Bootstrap CIs include zero: size [0.000, 0.254], speed [0.015, 0.627]. 9B rescues these but 9B size CI is bimodal [0.012, 0.853].
- **Speed and experience are pair-specific exceptions** to the shared z-direction. Shared/within ratio < 0.50 on both models. Their primal_z directions are genuinely pair-specific.
- **logit_diff R requires top-K validation.** Pos/neg R=0.47 dropped to R=0.31 on the only valid prompt.
- **SAE-basis PCA is worse than raw PCA** for recovering z (catastrophic for curved-manifold pairs like speed).
- **Increment R^2 dip not observed.** The predicted encode/re-encode dip does not exist; naive increment R^2 tracks cumulative R^2 almost perfectly. The fold-aware orthogonalized version (v11.5) shows all encoding at L1.

## Repository layout

```
geometry-of-relativity/
  PLANNING.md          # Frozen project spec
  BUILDING.md          # Current active task
  FINDINGS.md          # Full experimental log (v4-v9 ss1-ss13, v10 ss14, v11 ss15, v11.5 ss16)
  STATUS.md            # Project status summary and retraction list
  TODO.md              # Rolling task checklist
  scripts/
    vast_remote/       # GPU scripts (Vast.ai)
    analyze_v9_*.py    # v9 analysis scripts (CPU)
    plot_v9_*.py       # v9 plot scripts (CPU)
    analyze_v10_*.py   # v10 analysis: dimensionality, SAE, attention, increment R^2
    plot_v10_*.py      # v10 behavioral plots
    gen_v10_*.py       # v10 prompt generation (dense height grid)
    gen_v11_dense.py   # v11 prompt generation (8 pairs x 2 models)
    analyze_v11_*.py   # v11 analysis: PCA, probing, SAE, head taxonomy, cross-pair transfer
    run_v11_*.sh       # v11 orchestration scripts
    analyze_v11_5_*.py # v11.5 analysis: shared z, multi-seed transfer, joint ablation, bootstrap CIs
    run_v11_5_all.sh   # v11.5 orchestrator
  results/             # JSON summaries (large activations on HF)
    v11/               # Per-model per-pair extraction outputs
    v11_5/             # Shared-z, transfer, ablation, bootstrap results
  figures/             # v7 (clean grid), v8 (replots), v9 (SAE + layer sweep), v10 (dense grid)
    v11/               # PCA, probing, SAE overlap, steering transfer matrices
  docs/                # Session plans, paper outline, archive
  src/                 # Core library
  tests/               # pytest suite
```

## Quick start

```bash
cp .env.example .env       # then edit .env to add HF_TOKEN at minimum
pip install -e ".[dev]"
pytest tests/ -v -m "not gpu"

# Fetch activation data from HF (private dataset; HF_TOKEN must have read access):
python scripts/fetch_from_hf.py
python scripts/fetch_from_hf.py --only v11   # v11 dense extraction (FINDINGS ss15)

# Regenerate plots (CPU only):
python scripts/plots_v7_behavioral.py
python scripts/replot_v7_from_json.py

# Re-run all v10 CPU analyses from the fetched NPZs:
python scripts/analyze_v10_dimensionality.py
python scripts/analyze_v10_increment_r2.py
python scripts/analyze_v10_sae.py
python scripts/analyze_v10_attention.py
python scripts/analyze_v10_attention_taxonomy.py
python scripts/plot_v10_behavioral.py

# Re-run all v11 CPU analyses:
python scripts/analyze_v11_pca.py
python scripts/analyze_v11_z_vs_lexical.py
python scripts/analyze_v11_cross_pair_transfer.py
python scripts/analyze_v11_sae.py
python scripts/analyze_v11_head_taxonomy_and_ablate.py

# Re-run all v11.5 analyses (shared z, transfer, ablation, bootstrap):
bash scripts/run_v11_5_all.sh
# Or individually:
python scripts/analyze_v11_5_shared_z.py
python scripts/analyze_v11_5_multiseed_transfer.py
python scripts/analyze_v11_5_joint_ablation.py
python scripts/analyze_v11_5_perm_null_taxonomy.py
python scripts/analyze_v11_5_p3c_fold_aware.py
python scripts/analyze_v11_5_p3d_widened.py
python scripts/analyze_v11_5_sae_token_freq.py
python scripts/analyze_v11_5_bootstrap_cis.py

# Re-run v10 from scratch on a GPU box (Gemma 2 2B; H100 ~2 min cached):
python scripts/gen_v10_dense_height.py
python scripts/vast_remote/extract_v10_dense_height.py
python scripts/vast_remote/exp_v10_layer_sweep_steering.py
```

## License

CC-BY-4.0 for the paper, MIT for the code.
