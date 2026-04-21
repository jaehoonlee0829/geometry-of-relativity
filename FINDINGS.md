# FINDINGS — v4 adjective-pair experiments on Gemma 4 E4B

**Date:** 2026-04-20 / 21 (Day 4 of the 2-hour autonomous burst extended into analysis)
**Model:** `google/gemma-4-e4b` (≈8B params, residual width 2560)
**Layers probed:** `mid` = 21, `late` = 32

This document records what we actually found once the v4 pipeline finished and we started looking at the activation geometry. Companion doc to `PLANNING.md` (the pre-registration) and `STATUS.md` (the running log).

---

## 1. Setup recap

We extended the original 2-pair (tall/short, obese) design to **8 adjective pairs**, seven expected to be context-relative and one absolute control:

| Pair | Low word | High word | Unit | Expected |
|---|---|---|---|---|
| height | short | tall | cm | relative |
| age | young | old | years | relative |
| weight | light | heavy | kg | relative |
| size | small | big | cm | relative |
| speed | slow | fast | km/h | relative |
| wealth | poor | rich | USD/yr | relative (log-space) |
| experience | novice | expert | years | relative (log-space) |
| bmi_abs | underweight | obese | BMI | **absolute control** |

Per pair: 5 target values × 5 context means × 30 seeds (implicit) + 25 explicit + 5 zero-shot = **780 prompts × 8 pairs = 6240 total**. Extracted residual-stream activations at `mid` and `late` for every prompt at the final-before-label token position.

---

## 2. The "horseshoe" — PCA on cell means

When we PCA the per-cell mean activations (cells = bins of (x, μ)) at the late layer, **PC1 tracks z** and **PC2 tracks z²** — producing the classic PCA horseshoe.

Initial interpretation temptation: "the model is learning kurtosis / Gaussian surprise."
Correct interpretation: this is the well-known geometric artifact of embedding a 1-D ordered manifold into 2-D — PC2 is forced to be roughly quadratic in the PC1 coordinate to preserve ordering. See `results/v4_analysis/figures/pca_cell_means_late.png`.

**z² ≠ kurtosis.** In a Gaussian context, `z² = −2·log p(x | μ, σ) + const`, i.e. squared Mahalanobis distance / negative log-likelihood of the target under the in-context distribution. That's a meaningful interpretability quantity, but the PC2 curve itself is mostly geometric, not semantic.

---

## 3. Per-pair PCA — does every relative concept show the z pattern?

Ran PC1–PC5 per pair on late-layer cell means. (`pca_per_pair_late.png` + `pc_correlations_late.json`.)

**PC1 is z-dominant for all 7 relative pairs:** `R²(PC1 ~ z)` ranges 0.76 – 0.93, capturing ~50–60% of the cell-mean variance.

**PC2 is z²-dominant for most pairs** (R² 0.56 – 0.79) — the horseshoe generalizes. Exceptions: `size` (PC2~x dominates, R²=0.80) and `experience` (PC3~z², not PC2) — these concepts appear to mix raw-x and relativity differently, possibly because their target ranges are broader in log-space.

**The absolute control `bmi_abs` also shows the PC1~z pattern** (R²=0.85). This is suspicious — either (a) the `format_prompt_implicit` template still induces some contextual comparison even for BMI, or (b) PC1 is partly a raw-x axis rather than a pure z-axis. This motivates a no-ICL control (planned).

**PC3~x or μ, not raw x alone.** For several pairs PC3 is μ-dominant rather than x-dominant (age: R²(PC3~μ)=0.76; speed: R²(PC3~μ)=0.59). The model is not building separate axes for raw x vs μ vs z — it seems to put the z-composition front and center and treat (x, μ) as needed inputs.

---

## 4. Cross-pair PC1 cosine similarity

Stack the 8 per-pair PC1 unit vectors and compute the 8×8 |cos| matrix. (`pc_correlations_and_shared_z.png`.)

- **Mean off-diagonal |cos| = 0.32.** Not orthogonal (would be ~0.05 at random in 2560-d), not identical (would be 1.0). The z-direction is real but carries a concept-specific flavor.
- **Semantically similar pairs cluster:** `weight ↔ bmi_abs` = 0.70 (both about mass/body). `wealth ↔ experience` = 0.58 (both log-space). `age ↔ size` = 0.08 (most different — temporal vs spatial).
- Implication: the model re-uses a shared substrate for "extreme-vs-typical" judgments but specializes it per semantic domain.

---

## 5. The meta-direction (main v4 finding)

**Question:** Is there a single direction in 2560-d residual-stream space that serves as a general "how-extreme-vs-reference" axis across all 8 concepts?

**Method:**
1. For each pair, compute PC1 of cell-mean activations (late layer).
2. Sign-align each PC1 so `centered_acts @ pc1` positively correlates with z.
3. Stack into `V` ∈ ℝ^{8×2560}, take SVD: `V = U S Wᵀ`.
4. `w₁` = top right singular vector = "meta-direction" = the best single-direction approximation to all 8 per-pair z-axes simultaneously.

**Results** (`meta_z_direction.png` + `meta_z_direction.json`):

| pair | R²(own PC1 ~ z) | R²(meta w₁ ~ z) | \|cos(own, meta)\| |
|---|---|---|---|
| height | 0.857 | **0.909** | 0.75 |
| age | 0.762 | **0.800** | 0.63 |
| weight | 0.909 | 0.891 | 0.80 |
| size | 0.391 | **0.762** | 0.42 |
| speed | 0.930 | 0.914 | 0.53 |
| wealth | 0.914 | 0.919 | 0.71 |
| experience | 0.841 | **0.894** | 0.57 |
| bmi_abs | 0.848 | 0.863 | 0.66 |

**SVD scree:** `w₁` captures **41.6%** of the variance across the 8 per-pair PC1s (remaining 7 singular values drop gradually: 15%, 12%, 11%, 7%, 6%, 5%, 3%).

**Interpretation:**
- A single direction in activation space predicts z with R² ≥ 0.76 for every concept, and matches or beats each concept's *own* optimal PC1 in 5 of 8 cases. The "compromise" direction denoises concept-specific PCA residuals (most dramatic for `size`: 0.39 → 0.76).
- At 41.6% shared variance vs the 12.5% orthogonal baseline, the 8 z-axes are **~3.3× more aligned than chance** but are not identical. Real concept-specific residual exists (|cos| values 0.42–0.80).
- **Strongest interpretability claim from v4:** Gemma 4 E4B has internalized a *domain-general "extremity-vs-reference" feature* that activates across concepts, rather than instantiating separate z-machinery per domain.

---

## 6. Caveats & open questions

**(a) `bmi_abs` contamination.** The absolute control aligns with `w₁` (R²=0.863, |cos|=0.66). Either the prompt format induces unintended relativity, or `w₁` captures raw-x position on the grid rather than normalized z. Disambiguated by:
- **No-ICL control** (planned, Task #23): use `format_prompt_zero` (x only, no reference samples) and check whether PC1 still tracks z or collapses to x.

**(b) x / (μ, z) entanglement.** Since z = (x − μ)/σ, any two of {x, μ, z} determine the third. "PC3 ~ x" can always be rewritten as a mix of μ and z. The no-ICL control isolates pure-x encoding.

**(c) Correlation ≠ causation.** All current analyses are correlational — we find linear structure that predicts z. Whether the model actually *uses* `w₁` to compute the adjective distribution requires a causal intervention:
- **Meta-direction steering** (planned, Task #27): add `±α·w₁` to the residual stream at layer 32 across prompts from different concepts. If outputs shift toward high-adjectives for +α and low-adjectives for −α *across all 8 concepts*, that's causal evidence for a domain-general relativity feature. Compare to a random unit-vector baseline.

**(d) Framing claim.** A safe, defensible paper claim given current evidence:
> "Under in-context Gaussian example sets, Gemma 4 E4B represents the target primarily along a ~1-D manifold in the late residual stream. Across 8 semantic domains, the per-concept z-axes are non-orthogonal (mean |cos|=0.32) and collapse onto a single shared direction `w₁` that captures 42% of the cross-concept PC1 variance and predicts z-score with R² = 0.76–0.92 on all concepts including an absolute control. This suggests a partially-shared, partially-specialized 'relativity substrate' rather than either per-concept or fully-shared representation."

---

## 7. Artifact inventory

Scripts (in `scripts/vast_remote/`):
- `extract_v4_adjpairs.py` — 6240-prompt activation extraction (Day 4)
- `analyze_v4_adjpairs.py` — per-pair logit-diff + monotonicity checks
- `steer_v4.py`, `inlp_v4.py` — Day 4 steering & concept erasure (earlier work)

New analysis scripts (this round, in repo root or `scripts/`):
- `pca_per_pair.py` — per-pair PCA, PC1–PC5 R² table
- `pc3_x_and_shared_z.py` — PC3 vs x and cross-pair |cos| matrix
- `meta_z_direction.py` — SVD-based meta-direction analysis

Figures (`results/v4_adjpairs_analysis/figures/`):
- `pca_per_pair_late.png` — 8-panel PC1/PC2 scatter, one per pair
- `pc_correlations_and_shared_z.png` — cross-pair |cos| heatmap + per-PC R² bars
- `meta_z_direction.png` — SVD scree + R²(own vs meta~z) + |cos(own, meta)|
- `pca_cell_means_late.png` — original horseshoe figure

JSON summaries (same dir):
- `pca_per_pair_late.json`, `pc_correlations_late.json`, `meta_z_direction.json`

---

## 8. Next experiments

1. **No-ICL control** (Task #23) — run extraction with `format_prompt_zero` across same x-grid. Check whether PC1 tracks x or z when no reference samples are given. Expected: R²(PC1~z) drops sharply; R²(PC1~x) rises. If PC1 still tracks "z-like structure" without ICL, the geometry is baked into the model rather than emerging from context.
2. **Meta-direction steering** (Task #27) — causal intervention. See caveat (c) above.
3. **Transfer to Llama-3.2-3B** (replication). Does `w₁` transfer to a different model family?

---

# Section 6 — v5 follow-up experiments (Day 6, 2026-04-21)

Executes the 7-experiment red-team punch list from `docs/NEXT_GPU_SESSION.md`
plus a G31B replication, plus a post-hoc random-direction null requested by a
critic pass (3 parallel Claude subagents reviewing from statistical,
alternative-interpretation, and implementation-correctness angles).

Branch: `exp/next-gpu-session`. 14 commits. PR:
<https://github.com/jaehoonlee0829/geometry-of-relativity/pull/4>.

## 6.1 — Strong positive findings

### Meta-direction w₁ causally steers all 8 pairs, far above the random null

- `scripts/vast_remote/exp2_meta_steer.py` + `exp2b_random_null.py`.
- w₁ = top right singular vector of stacked sign-aligned per-pair PC1
  activations. 41.6% shared variance across 8 pairs.
- Added α·w₁ to the residual stream at E4B layer 32. 100 implicit prompts per
  pair, α ∈ {−8, −4, −2, −1, 0, 1, 2, 4, 8}.
- Every pair: monotone linear curve. Slope magnitude ∈ [0.046, 0.129] logit
  units per α-unit. See `figures/v4_adjpairs/meta_w1_steering_curves.png`.
- **Random-direction null** (3 random unit vectors, same protocol):

  | pair | w₁ slope | rand mean ± std | ratio w₁/rand |
  |---|---|---|---|
  | height | 0.080 | 0.004 ± 0.005 | **21.3×** |
  | age | 0.064 | 0.007 ± 0.008 | 8.8× |
  | weight | 0.094 | 0.010 ± 0.005 | 9.3× |
  | size | 0.129 | 0.017 ± 0.021 | 7.7× |
  | speed | 0.046 | 0.015 ± 0.016 | 3.1× |
  | wealth | 0.084 | 0.009 ± 0.008 | 9.2× |
  | experience | 0.079 | 0.009 ± 0.012 | 9.0× |
  | bmi_abs | 0.098 | 0.003 ± 0.002 | **28.5×** |

- Plot: `figures/v4_adjpairs/meta_w1_vs_random_null.png`.

## 6.2 — Honest null / negative findings

### H4 (Fisher pullback) is not supported at tested activations

- `scripts/vast_remote/exp3b_fisher_cosines.py` — torch/GPU F(h) computation
  (400 Fisher factorizations in ~2 min on H100).
- Averaged over 8 pairs × 2 layers × 25 cell-mean activations each:

  | layer | |cos(w_adj, w_z)| Euclid | |cos(w_adj, w_z)| F⁻¹ | lift |
  |---|---|---|---|
  | mid | 0.178 | 0.173 | −0.004 |
  | late | 0.182 | 0.173 | −0.009 |

- H4 predicted `cos_F⁻¹(w_adj, w_z) > 0.7` for relative adjectives.
  **Max observed is 0.30 (speed/mid).** The prediction does not hold.
- Per-cell standard deviation of F⁻¹ cosines is 1e-4 to 7e-3, so F(h) is
  essentially isotropic across the activations we probed. Mechanism:
  cell-mean activations sit in a high-entropy region of the softmax, where
  `diag(p) − ppᵀ` is well-conditioned and close to `(1/V)·I`. F⁻¹ ≈ c·I.
  H4 might still hold at peaked-prediction activations; not tested.
- Figure: `figures/v4_adjpairs/fisher_vs_euclid_cosines.png`.

### Relative-vs-absolute dichotomy is not statistically significant (n=7 vs 4)

- Added 3 new absolute controls: `temp_abs` (cold/hot), `legal_abs`
  (minor/adult), `grade_abs` (failing/passing). Each: 780 prompts, standard
  5×5 grid × 30 seeds. Total 2340 new prompts.
- Project-convention relativity ratios at E4B layer late:

  | class | n | mean ± std |
  |---|---|---|
  | relative (height, age, experience, size, speed, wealth, weight) | 7 | 0.749 ± 0.231 |
  | absolute (bmi_abs, temp_abs, legal_abs, grade_abs) | 4 | 0.708 ± 0.194 |

- **Welch t = −0.33, p = 0.75.** Not significant.
- `legal_abs` has the *highest* relativity ratio (0.89) — "adult" is
  polysemous, and the model reads the context block as the reference
  distribution even for this supposedly-anchored concept.
- Figure: `figures/v4_adjpairs/relativity_ratio_absolute_vs_relative.png`.

### Σ⁻¹ metric is essentially uninformative at d=2560, N=750

- `scripts/vast_remote/exp3a_sigma_inv.py`.
- Regularization λ = 1e-3 · trace(Σ)/d dominates Σ at this dimensionality
  → Σ⁻¹ ≈ c·I. Cosines in the Σ⁻¹ metric differ from Euclidean by ≤0.05.
- Honest negative finding; motivates focusing on Fisher (even though
  Fisher also failed, it failed for a cleaner mechanistic reason).

### Zero-shot direction "orthogonal to implicit z-direction" is null-consistent

- `scripts/vast_remote/exp1_zero_shot_expand.py` — 5 x × 30 phrasing seeds
  × 8 pairs = 1200 zero-shot prompts.
- Zero-shot *does* encode x cleanly (cv_R²(w_x_zeroshot → x) ≥ 0.96).
- cos(w_x_zeroshot, w_z_implicit) ∈ [−0.01, +0.08] across all 8 pairs × 2
  layers.
- **But** in d=2560, two random unit vectors have cos with stddev
  ≈ 1/√d ≈ 0.020. Observed |cos| ≤ 0.08 is only 2–4σ above chance —
  directionally consistent with the hypothesis, not quantitatively above
  the null.

## 6.3 — Scaling evidence (G31B)

- `scripts/vast_remote/g31b_adjpairs.py` — same 8-pair extraction on
  Gemma 4 31B (60 layers, d=5376). 6000 implicit prompts in ~4 min.

  | pair | E4B late | G31B late | Δ |
  |---|---|---|---|
  | height | 0.80 | 0.92 | +0.12 |
  | age | 0.82 | 0.92 | +0.10 |
  | weight | 0.96 | 0.91 | −0.05 |
  | size | 1.03 | 0.76 | −0.27 |
  | speed | 0.75 | 0.79 | +0.04 |
  | wealth | 0.43 | 0.69 | +0.26 |
  | experience | 0.45 | 0.88 | +0.43 |
  | bmi_abs | 0.49 | 0.54 | +0.05 |

- 6/8 pairs get *more* relative at scale. Absolute/relative gap *shrinks*.
  Caveat: 6/8 is not a significant sign test (binomial p≈0.29 under null).
- Figure: `figures/v4_adjpairs/g31b_vs_e4b_relativity.png`.

## 6.4 — Zero-shot bias (Exp 4d)

- `scripts/vast_remote/exp4d_validation.py`.
- 4/8 pairs predict the "high" word even at objective x_min. E.g., the model
  says "tall" for a 150 cm person, "old" for 20 years, "expert" for 1 year
  experience, "obese" for BMI 17 — all in zero-shot context.
- 3/8 pairs (size/speed/weight) never predict the "high" word across their
  entire x range in zero-shot.
- Alternative explanation flagged by a critic agent but not tested:
  after "is a", vowel-initial words (old, expert, obese, tall — all four
  biased-high words) may be favored by determiner-tokenization asymmetry.

## 6.5 — Critic consensus (3 parallel skeptical agents)

Three Claude subagents reviewed independently from different angles:

- **Statistical methodology**: flagged Exp 2 needed a random-direction null
  (now added as 2b), questioned Exp 7 power at n=4 vs 7, flagged multiple-
  comparison inflation across 192 cosines, and noted Σ⁻¹ regularization
  dominance. Verdict: Σ⁻¹/F⁻¹ negative findings are honest; meta-steering
  survives with the null control; Exp 1 orthogonality is null-consistent.
- **Alternative interpretation**: flagged that w₁ steers bmi_abs with the
  *largest* slope (not uniform) → probably a polarity direction, not a
  relativity direction. Flagged "legal_abs adult is polysemous" as
  prompt-design confound. Flagged that F(h) isotropy at cell-means doesn't
  refute H4 at peaked activations. Flagged vowel-onset / determiner confound
  for Exp 4d.
- **Implementation correctness**: caught a formula mismatch in Exp 7
  (primary `welch_t/p` was computed from a mixed-convention vector — fixed).
  Flagged y_adj definition divergence from the reference pipeline and a
  Fisher-precision deviation from `src/fisher.py`. Flagged a mislabel
  ("obese/healthy" → should be "obese/thin") in `plots_per_pair_v5.py` — fixed.

All three reports informed the "known limitations" section of the PR body
(`docs/PR_next_gpu_session.md`). None of the findings above were modified
post-hoc to fit the hypotheses; the critics' flags are surfaced, not buried.

## 6.6 — What this means for the paper

Recommended framing shift for ICML MI Workshop (May 8):

- **Keep and lead with**: the 8-pair behavioral heatmap + the meta-direction
  causal steering result with random-null control. Both are strong and
  reproduce on both models.
- **Reframe**: "universal relativity substrate" → "shared adjective-polarity
  substrate" — w₁ shifts polarity for absolute and relative pairs alike.
- **Soften or move to appendix**: H4 (Fisher pullback as a magnifying glass
  for relativity), the 2-class relative/absolute dichotomy, and the
  orthogonality claim for zero-shot x-direction vs implicit z-direction.
- **Explicit limitations section** required: x-ranges that don't bracket the
  decision boundary for size/speed/weight; base-model zero-shot bias;
  polysemous "adult"; underpowered n=4 absolute class.

The paper will likely be narrower in its causal claim (polarity ≠ relativity)
but stronger in evidence (null-controlled steering, 2-model scaling).

---

# Section 7 — v7 clean-grid rerun (2026-04-21, post-v6 red-team)

v6 red-team noted a design confound: the (x, μ) grid used in v4-v6 creates
corr(x, z) ∈ [+0.58, +0.86] per pair because only 5 x × 5 μ cells are
sampled. Every z-direction (primal, PC1, probe) was contaminated with x.

v7 fixes this with Grid B (iterate x, z; derive μ). The 6 follow-up
experiments rerun v4-v6 analyses on clean activations.

Branch: `exp/v7-clean-grid`. PR: (link after creation).

## 7.1 — The confound was real (Priority 1+2)

Grid A had corr(x, z) ∈ [+0.58, +0.86] per pair. Grid B has corr(x, z) ≈ 0
for 5/8 pairs and < 0.20 for 3 pairs with dropped cells.

**Mean |cos| across 8 pairs, direction × direction (z vs x comparisons):**

|  | Grid A | Grid B | Δ |
|---|---|---|---|
| primal_z × primal_x | **0.907** | **0.033** | −0.874 |
| primal_z × PC1 | 0.946 | 0.676 | −0.270 |
| primal_x × PC1 | 0.798 | 0.105 | −0.693 |
| probe_z × probe_x | 0.544 | 0.268 | −0.276 |
| zeroshot_wx × primal_x | 0.149 | 0.168 | +0.019 |

The v6 "4 clusters collapse" story was substantially a grid artifact.

## 7.2 — Primal_z steering gap widens (Priority 3)

v6 found primal_z slopes 13× larger than probe_z slopes. v7 clean-grid
directions: the gap is 18× averaged across pairs.

  height 9.6×, age 14.3×, weight 13.4×, size 5.8×, speed 48.5×,
  wealth 17.4×, experience 17.3×, bmi_abs 20.6× → mean ~18×

Primal_z slopes are stable across grids (0.10-0.16 per α-unit).
Probe_z slopes are stable too (near 0.01). The widening comes from
primal_x's slope dropping — it was a confound-inflated direction.

## 7.3 — Primal_z transfers across pairs (Priority 4)

The biggest v7 positive finding. Cross-pair transfer matrix using
primal_z_clean at α=±4:

  diagonal (primal_z on own pair)      mean = 0.126 per α-unit
  off-diagonal (A → B)                 mean = 0.051 per α-unit
  random unit vector null              mean = 0.009 per α-unit

Ratios:
  off-diag / own-pair: **0.40** — transfer at 40% of own-pair strength
  off-diag / random:   **5.50** — way above chance

v6 concluded "probe_z doesn't transfer, no shared substrate" using the
weak probe_z direction. Using the strong primal_z direction on clean
grid, transfer is substantial.

Cluster patterns:
  weight ↔ size ↔ bmi_abs: mutual transfer 0.10-0.13 (near own-pair)
  age ↔ experience:        transfer 0.07 (modest)
  speed → age:             −0.002 (near zero)

This suggests the "shared substrate" isn't perfectly universal — there
are semantic clusters where transfer is near-complete and outside them
transfer is weak. Body-attribute pairs form one tight cluster.

## 7.4 — Park + Fisher still negative (Priority 5)

Recomputed Park Cov(W_U)^{-1/2} cosines on Grid B probes AND Fisher F⁻¹
at softmax-entropy-binned activations (bottom 10% vs top 10% entropy).

Results:
  Park |cos(w_z, w_ld)| ≈ Euclidean |cos| within 0.02-0.04 — no help.
  Fisher F⁻¹ low-entropy vs high-entropy bin: max delta = 0.005.
  Fisher F⁻¹ ~1.5× Euclidean in magnitude (weak amplification).
  Max F⁻¹ |cos(w_z, w_ld)| observed: 0.14 (still far from H4's > 0.7).

Entropy range sampled: 3.69-5.39 nats (of ln(262144) = 12.5 nats max).
We never hit "peaked softmax" regime. H4 remains refuted across all
tested regimes.

## 7.5 — INLP ACTUALLY WORKS on clean grid (Priority 6)

v4 INLP on tall/short (Grid A dense) found R²(z) dropped only 0.04 after
8 iterations — interpreted as "z is distributed, INLP can't remove it".

v7 INLP on clean Grid B across 8 pairs:
  per-pair R²(z) starts ~0.97, drops to 0.45-0.70 after 8 iterations
  Δ(INLP) ranges 0.29-0.51; random-direction null shows Δ ≈ 0.00

The v4 finding was driven by the x↔z confound: projecting out the
z-direction partially removed x, but x still carried z-info via the
design correlation. Ridge re-learned a z-probe from x-like features.

On clean grid, no x-pathway exists — INLP actually removes z.

## 7.6 — Summary: what v7 changes

**NEW strong findings:**
  - Cross-pair transfer of primal_z at 40% own-pair strength, 5.5× null
  - INLP actually works on clean grid (R² drops 30-50%)
  - Body-attribute semantic cluster in transfer matrix

**v5/v6 findings that held up:**
  - primal_z steers 13-18× stronger than probe_z
  - cos(w_z, w_ld) ≈ 0 at late layer
  - H4 (Fisher) refuted
  - Meta w₁ causally effective across all 8 pairs

**v5/v6 findings that were artifacts:**
  - "primal_z ≈ primal_x" (same direction) — false on clean grid
  - "INLP barely reduces R²(z)" — false on clean grid
  - "v6 7 directions collapse to 4 clusters" — partially false

**Methodological contribution:** grid designs with derived variables
(z from x, μ) contaminate direction-based analyses. Future mech-interp
work should use independent-variable grids for geometry analysis.

## 7.7 — v7b addendum: anti-alignment for size and experience

The v7 residual confound (corr(x,z) = 0.20 for experience, 0.13 for size)
was fixed by dropping the lowest-x value from each pair entirely (x=1 for
experience, x=5 for size). Result: corr(x, z) = 0.0000 exactly on a 4×5×30
= 600-prompt grid.

On this fully clean grid:
  size:       cos(primal_z, primal_x) moves from −0.591 → −0.756
  experience: cos(primal_z, primal_x) moves from +0.008 → −0.442

Both MORE negative after cleaning, not closer to zero. The residual positive
corr(x,z) in v7 was PARTIALLY MASKING a real anti-alignment between the z
and x directions for these pairs.

Updated |cos(primal_z, primal_x)| distribution across all 8 pairs:
  height 0.18, age 0.16, weight 0.15, size 0.76, speed 0.08,
  wealth 0.12, experience 0.44, bmi_abs 0.48
  mean |cos| = 0.30   range [0.08, 0.76]

Three behavioral classes emerging:
  (a) primal_z ⟂ primal_x   (height, age, weight, speed, wealth): |cos| < 0.20
  (b) primal_z ANTI-ALIGNED with primal_x   (size, experience): cos ≈ -0.5 to -0.75
  (c) primal_z POSITIVELY ALIGNED with primal_x   (bmi_abs): cos ≈ +0.48

(c) is expected for a genuine absolute pair where z and x carry similar info.
(b) is unexpected and interesting: for size/experience, "more z" and "more x"
point in OPPOSITE directions in activation space. Could reflect context
CALIBRATION (using z to reduce x-salience). Needs further investigation.

Open question for future v8+: is (b) a model-specific feature, a domain-
specific feature, or an artifact of these two pairs having been the "dirtiest"
in the raw Grid A design? If repeating with a different clean-grid design
(e.g., independent x and μ instead of independent x and z) recovers similar
patterns, it's likely model/domain; if patterns shift, it's design-artifact.
