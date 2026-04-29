# Paper outline — "The Geometry of Relativity: Context-Normalized Encoding of Gradable Adjectives in LLMs"

**Target venue (primary):** ICML 2026 Mechanistic Interpretability Workshop, deadline May 8 2026.
**Target venue (secondary):** NeurIPS 2026 (abstract May 4, full paper May 6).
**Status:** updated Apr 29 2026 with v12/v12.1/v12.2 claim-hardening findings.

## One-sentence thesis

Relative gradable adjectives ("tall", "rich", "heavy", "fast", etc.) are represented in
Gemma 2 (2B and 9B) residual streams along a **partly shared, context-normalized direction**
that tracks the z-score `z = (x − μ)/σ` rather than the raw magnitude `x`. This direction is
available early, becomes causally potent later, and decomposes into a high-gain lexical/output
component plus a residual component that transfers broadly but still tracks target lexical
overlap.

## Section map

### §1 Introduction
- Motivation: how do LLMs internally represent graded, context-dependent judgments?
- Claim: relative adjectives → shared context-normalized (z-like) direction in residual stream.
- Why this matters for alignment / fairness / evaluation: the same person is "tall" in one
  context and "average" in another; the model's internal representation reflects that.
- Contribution summary (6 evidence lines across 2 model scales):
  1. Behavioral z-signal replicates 8/8 pairs on both models (cell-mean R(z) ≥ 0.92).
  2. Domain-agnostic shared z-direction exists (55% pairwise alignment; 6/8 pairs steered at ≥50%).
  3. Cross-pair transfer is statistically real (56/56 off-diagonal cells BH-FDR significant).
  4. z is encoded early and becomes causally potent later (V12 9B layer sweep).
  5. primal_z is W_U-orthogonal but decision-aligned (cos ~0.15 vs cos ~0.7–0.86).
  6. V12.1/V12.2 decompose the causal direction into lexical and residual components.
  7. SAE features are z-correlated and pass raw-number/token controls, but V12 finds a mixed
     population rather than pure relative-standing features.

### §2 Related work
- Linear representation hypothesis (Mikolov et al.; Park, Choe, Veitch 2024).
- Concept erasure: Ravfogel et al. 2020 (INLP); Belrose et al. 2023 (LEACE).
- Probing literature critique: Hewitt & Liang 2019; Pimentel et al. 2020.
- Encode-vs-use depth gradient: Geva et al. 2021 (FFN as key-value memories),
  nostalgebraist 2020 (logit-lens), Belrose et al. 2023 (tuned-lens), Lad et al. 2024.
- Fisher-information geometry in interpretability: Park et al. 2023 on causal inner product.
- Sparse autoencoders in mech-interp: Anthropic (Templeton et al.), Goodfire/Sarfati,
  Cunningham et al. 2023 (SAE for feature decomposition).

### §3 Setup

**Models.**
- Gemma 2 2B (`google/gemma-2-2b`): 26 layers, d_model=2304, 8 heads.
- Gemma 2 9B (`google/gemma-2-9b`): 42 layers, d_model=3584, 16 query-heads.
- SAEs: `google/gemma-scope-2b-pt-res` and `google/gemma-scope-9b-pt-res` (65k features, all layers).

**Adjective pairs.** Seven relative pairs plus one absolute control:
- Relative: height/(tall,short), age/(old,young), weight/(heavy,light), size/(large,small),
  speed/(fast,slow), wealth/(rich,poor), experience/(experienced,inexperienced).
- Absolute control: bmi_abs/(obese,not obese) with fixed clinical threshold 30.

**Prompt design (v11 dense grid).**
- 20 x-values × 20 z-values × 10 seeds = 4,000 prompts per pair per model.
- Per-pair `crc32(pair_name)` seed offset to decorrelate cross-pair nulls.
- Implicit 15-person context sampled from Normal(μ, σ) with per-pair plausibility bands
  (some cells dropped; kept counts range 3,510–4,000 per pair).

**Measurements.**
- `logit_diff = logit(high_adj_token) − logit(low_adj_token)` at final "is" position.
- Residual-stream activations at all layers, last-token.
- Attention (eager, per-head value-mix) at 8 strategic layers per model.

**Probes.** Ridge regression on cell-mean activations (400 cells per pair), 5-fold CV.

### §4 Evidence Line 1 — Behavioral z-signal (FINDINGS §15.1)

Cell-mean `corr(LD, z)` on the dense 20×20 grid:

| pair       | gemma2-2b | gemma2-9b |
|---         |---:       |---:       |
| height     | 0.972     | 0.97+     |
| age        | 0.93–0.96 | 0.93–0.97 |
| weight     | 0.94–0.97 | 0.94–0.97 |
| size       | 0.92–0.96 | 0.93–0.97 |
| speed      | 0.93      | 0.94      |
| wealth     | 0.95–0.97 | 0.95–0.97 |
| experience | 0.95–0.97 | 0.95–0.97 |
| bmi_abs    | 0.953     | 0.95+     |

8/8 pairs R(z) ≥ 0.92 on both models. This upgrades v9's "8/8 R > 0.3" to dense-grid confirmation.

### §5 Evidence Line 2 — PCA geometry (FINDINGS §15.2 + §16.8)

PCA on 400 cell-mean activations at canonical late layer (2B L20, 9B L33).
PC1 tracks z for most pairs; PC2 tracks z² (horseshoe).

| pair       | 2B PC1.R²(z) [95% CI]     | 9B PC1.R²(z) [95% CI]     |
|---         |---                        |---                        |
| height     | **0.969** [0.961, 0.975]  | **0.928** [0.907, 0.941]  |
| weight     | **0.949** [0.933, 0.960]  | **0.944** [0.930, 0.954]  |
| bmi_abs    | **0.923** [0.876, 0.956]  | **0.784** [0.750, 0.813]  |
| experience | **0.901** [0.865, 0.928]  | **0.902** [0.846, 0.930]  |
| wealth     | **0.855** [0.768, 0.908]  | **0.871** [0.838, 0.897]  |
| speed      | 0.360 [0.015, 0.627]      | **0.428** [0.271, 0.582]  |
| age        | 0.209 [0.091, 0.341]      | **0.606** [0.003, 0.843]  |
| size       | 0.075 [0.000, 0.254]      | **0.656** [0.012, 0.853]  |

2B median PC1.R²(z) = 0.901; 9B median = 0.871 but with tighter spread. Three pairs that fail
at 2B (age, size, speed) improve at 9B — **scaling rescues the z-code on harder pairs**.

Caveat (alternative critic): for size/age at 2B, PC1 tracks raw x (R²=0.65/0.42), not z.
The "PC1≈z" framing holds only where the grid successfully decorrelated x and z.

### §6 Evidence Line 3 — Shared z-direction (FINDINGS §16.1)

Construct `w_shared` via Procrustes-aligned mean of the 8 per-pair primal_z's at the canonical
late layer. Steer all 8 pairs with α · w_shared at α ∈ {−4, 0, +4}.

- Pairwise primal_z cosine mean: **+0.559** (2B), **+0.516** (9B).
- Shared/within steering ratio:

| pair       | 2B ratio | 9B ratio |
|---         |---:      |---:      |
| height     | **0.93** | **0.75** |
| weight     | **0.89** | **0.80** |
| size       | **0.87** | **0.66** |
| bmi_abs    | **0.77** | **0.65** |
| wealth     | **0.73** | **0.70** |
| age        | **0.60** | **0.56** |
| speed      | 0.44     | 0.42     |
| experience | 0.27     | 0.50     |

6/8 (2B) and 7/8 (9B) above the 0.50 threshold. Speed and experience are pair-specific exceptions.
Notably, the absolute-adjective control (bmi_abs) aligns with the relative pairs at 0.65–0.77.

### §7 Evidence Line 4 — Multi-seed cross-pair transfer with FDR (FINDINGS §16.2)

5 seeds × 400 cells per (source, target) pair. BH-FDR at q=0.05 across 56 off-diagonal cells.

- **56/56 off-diagonal cells significant on both models.**
- Off/within ratios per target pair range 0.10–0.72 (2B) and 0.25–0.54 (9B).
- Same speed/experience asymmetry: experience at 2B = 0.10, lowest of all.

### §8 Evidence Line 5 — z available early, causal use later (FINDINGS §16.5 + V12)

Fold-aware orthogonalized increment R² (fix of v11's broken pipeline):

| pair (2B) | naive peak | orth peak layer | orth peak value |
|---        |---         |---              |---:             |
| height    | L12 (0.995)| L1              | **0.145**       |
| bmi_abs   | L13 (0.993)| L1              | **0.256**       |
| experience| L17 (0.997)| L1              | **0.125**       |
| pair (9B) |            |                 |                 |
| height    | L18 (0.998)| L1              | **0.144**       |
| bmi_abs   | L19 (0.995)| L3              | **0.167**       |

New z-info is concentrated at L1 (2B) / L1–L3 (9B), then near-zero. Naive R² plateaus by L7
because the model carries L1's z-encoding forward, not because L7 encodes it.

V12 adds the causal side on Gemma 2 9B: `z` is decodable early, but `primal_z` steering peaks
later around L25 (mean slope ≈ +0.097) and remains positive at L33 (≈ +0.067). This should be
framed as encode-vs-use separation, not as a fully identified circuit.

### §9 Evidence Line 6 — W_U-orthogonal but decision-aligned (FINDINGS §15.3 + §16.6)

Two disentanglement measures at late layers:

1. **cos(primal_z, W_U[high] − W_U[low]) ≈ 0.01–0.18** across all pairs/layers/models.
   primal_z is orthogonal to the lexical readout direction.

2. **cos(primal_z, leans-high − leans-low) ≈ 0.70–0.86** for height/age/weight at late layers.
   primal_z IS aligned with the behavioral decision boundary.

Interpretation: primal_z carries the "above-vs-below-the-norm" semantic decision through
a non-trivial projection before the final logit. It is W_U-orthogonal but not decision-orthogonal.

### §10 Evidence Line 7 — Lexical/residual decomposition (V12.1 + V12.2)

V12 showed that lexical sentence directions can steer as strongly as `primal_z`, motivating
a decomposition:

```math
p_z = p_{z,\mathrm{lex}} + p_{z,\mathrm{resid}}
```

V12.1:
- mean cos(primal_z, word-token lexical direction) ≈ +0.104.
- mean cos(primal_z, sentence-final lexical direction) ≈ +0.260.
- mean norm² of `primal_z` in the lexical subspace ≈ 0.080.
- lexical projection/primal steering ≈ 1.25; residual/primal steering ≈ 0.69.

V12.2 single-seed cross-pair transfer at Gemma 2 9B L33:

| direction | diagonal mean | off-diagonal mean |
|---|---:|---:|
| full `primal_z` | +0.067 | +0.026 |
| lexical projection | +0.087 | +0.011 |
| lexical residual | +0.044 | +0.024 |

Residualized directions transfer much better off-diagonal than lexical projections and recover
most of full `primal_z` transfer. However, residual transfer still correlates with target
lexical-subspace overlap (r≈+0.79), so this is mixed-mechanism evidence rather than proof of a
clean non-lexical code.

### §11 SAE decomposition — z-correlated but mixed features (FINDINGS §16.7 + V12)

Top SAE z-features have R²(z) ≈ 0.7–0.84 with R²(x) ≈ 0 and R²(token-frequency) ≈ 0.
The alternative critic's "SAE features track numeral-magnitude" hypothesis is refuted.

- 2B: 11–50 pure-z features per pair; cross-pair Jaccard = 0.109.
- 9B: 1–16 pure-z features per pair; cross-pair Jaccard = 0.223.
- 9B has 2× 2B's cross-pair feature overlap, consistent with its higher pairwise primal_z alignment.

V12 lexical audit softens the mechanism claim: among 200 audited top 9B z-features,
43 are pure-ish z, 39 lexical z-like, 52 raw numeric, and 66 mixed/polysemantic. Use
"z-correlated sparse features" rather than "pure relative-standing features."

### §12 Discussion

**The shared z-direction.** A single Procrustes-aligned direction steers 6/8 pairs
at ≥50% of within-pair efficiency on both model scales. This is consistent with the model
having internalized a partly domain-general "above-vs-below-the-norm" feature, with pair-specific
residual for speed (vehicles vs people) and experience (domain shift). V12.2 suggests the residual
component is more transferable than the lexical projection, but target lexical overlap remains a
major confound.

**Encode-vs-use gradient.** The fold-aware orthogonalized R² isolates *encoding* from
*carry-forward*: z is written into the residual stream at L1 in essentially one operation.
This refines Geva/tuned-lens's multi-stage picture into a specific "one-shot encoding" claim
for graded scalars.

**Scope.** Tested on Gemma 2 (2B, 9B) across 8 English adjective pairs. The pattern may or
may not generalize to (a) other model families (Gemma 3/4, Llama, etc.), (b) other languages,
(c) absolute adjectives with less clean thresholds.

**Limitations.**
- The v10 §14.6 causal head taxonomy (L13h2/L3h0/L0h6) is **triple-refuted**: single-head
  ablations null (§15.4), joint tag-set ablations null on 2B and *helping* on 9B (§16.3),
  permutation null on thresholds (§16.4). Re-framed as DLA-correlational only. This belongs
  in Limitations as a cautionary result on attention-based causal claims.
- Our "z" is computed from sampled context values, not the model's belief state.
- bmi_abs behaves more like relative pairs than expected — either the prompt format
  induces unintended relativity, or the model treats BMI as graded too.
- For size/speed/age at 2B, the "PC1 ≈ z" claim has bootstrap CIs overlapping zero.
- V12 pure-x/fixed-mu/matched-z controls are mixed, not a clean scalar-magnitude refutation.
- V12.1/V12.2 prevent a clean "non-lexical shared code" claim: lexical projections are high-gain,
  residuals transfer broadly, and target lexical overlap predicts residual transfer.
- V12 SAE and PC audits are mixed: SAE features are not uniformly pure-z, and PC extremeness is
  pair-specific rather than a universal PC2 result.

### §13 Conclusion

Converging evidence — behavioral signal, PCA geometry, shared z-direction,
FDR-controlled cross-pair transfer, early decodability with later causal potency, and
z-correlated SAE features — supports the claim that relative gradable adjectives are represented
with a context-normalized geometry in Gemma 2. The strongest post-V12 framing is not a pure
domain-agnostic vector, but a mixed mechanism: a partly shared residual relativity component
coupled to high-gain lexical/output-facing adjective geometry.

## Figures (to produce)

| # | Name                               | Source data                                  | Status      |
|---|---                                 |---                                           |---          |
| 1 | behavioral_heatmaps_8pairs.png     | v11 meta JSONs (per pair)                    | has v11 fig |
| 2 | pca_scatter_horseshoe.png          | v11 pca_summary.json (per pair)              | has v11 fig |
| 3 | pc1_r2_z_with_cis.png              | v11_5 bootstrap_cis.json                     | **TODO**    |
| 4 | shared_z_steering_ratios.png       | v11_5 shared_z_analysis.json                 | **TODO**    |
| 5 | cross_pair_transfer_heatmap.png    | v11_5 multiseed_transfer.json                | **TODO**    |
| 6 | increment_r2_fold_aware.png        | v11_5 increment_r2_fold_aware.json (per pair)| **TODO**    |
| 7 | lexical_residualization.png        | v12_1 lexical_subspace_residualization       | has v12.1 fig |
| 8 | residual_vs_lexical_transfer.png   | v12_2 residual transfer JSON                 | has v12.2 fig |
| 9 | sae_mixed_features.png             | v12 SAE lexical audit                        | has v12 fig |

## Appendix sketches

- A. Prompt templates for all 8 pairs.
- B. Full 26-layer (2B) / 42-layer (9B) naive R²(z) sweep.
- C. Cross-pair transfer full 8×8 matrix with per-cell slopes and FDR q-values.
- D. Joint head-ablation results (§16.3) and permutation null (§16.4).
- E. Bootstrap CI methodology (block bootstrap over (μ, x) cells, 1000 reps).
- F. Retracted analyses: v10 §14.6 causal taxonomy, v11 P3c broken pipeline, v11 P3d NaN.
- G. V12.1 lexical-subspace residualization and V12.2 target lexical leakage metric.

## Execution checklist

- [x] v11 dense extraction (8 pairs × 2 models × 4,000 prompts)
- [x] v11 PCA, probing, cross-pair transfer, head ablation, SAE, critics
- [x] v11.5 shared z-direction (§16.1)
- [x] v11.5 multi-seed FDR transfer (§16.2)
- [x] v11.5 joint head ablation + permutation null (§16.3, §16.4)
- [x] v11.5 fold-aware P3c (§16.5)
- [x] v11.5 widened P3d (§16.6)
- [x] v11.5 SAE token-freq control (§16.7)
- [x] v11.5 bootstrap CIs (§16.8)
- [x] v12 claim-hardening: layer sweep, lexical red-team, pure-x transfer, SAE/PC audits
- [x] v12.1 lexical-subspace residualization
- [x] v12.2 residual-vs-lexical cross-pair transfer
- [ ] Produce publication-quality figures from v11.5 JSONs (items 3–8 above)
- [ ] Prose draft in ICML MI Workshop template
- [ ] Submit (May 8 deadline)
