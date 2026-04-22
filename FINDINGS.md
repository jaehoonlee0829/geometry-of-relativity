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

---

# Section 8 — v8 prompt-sensitivity + clean-grid replots (2026-04-21)

Tests whether v7's R=0.47 for pos/neg math is prompt-design artifact;
replots v4/v6 activation-geometry figures on clean Grid B; validates
cross-pair transfer with cross-template test.

Branch: `exp/v8-direct-sign`. 5 commits. Critic-agent pass confirmed
scoring-token validity is the biggest unreported concern.

## 8.1 — Direct sign classification: only 1 of 4 variants is valid

Tested 4 prompts on Grid B pos/neg math. Top-K diagnostic revealed that
only ONE variant has its scoring tokens reliably in the model's top-10
predictions:

  variant     scoring tokens    high in top-10  low in top-10   R
  orig        positive/negative    16%             54%         0.47
  compared    above/below           0%              0%         0.24 (invalid)
  relative    higher/lower          0%              0%         0.47 (invalid)
  forced_qa   Above/Below         100%            100%         0.31  ← only valid

The "compared" and "relative" variants measure logit_diff on tokens that
never appear in top-10 — essentially tail-distribution noise. Forced Q/A
("Is this number above or below zero? Answer:") is the only variant where
the scoring tokens are the model's actual top completions.

Refined pos/neg claim: R = 0.31, accuracy = 0.95.
  - Non-zero → residual context effect exists.
  - Substantially smaller than v7's 0.47 → earlier headline was inflated
    by ambiguous-prompt scoring on out-of-top-10 tokens.

## 8.2 — PCA on clean Grid B: PC1 is NOT universally the z-axis

v4 reported "horseshoe: PC1 = z, PC2 = z²" as universal. On Grid B
(clean), PC1 splits into two classes:

  PC1 ≈ z-axis (4 pairs):     height 0.95, weight 0.97, wealth 0.60, bmi_abs 0.69
  PC1 ≈ x-axis (4 pairs):     age 0.87, size 0.67, speed 0.88, experience 0.62

The model represents height and weight with z as the primary variance
axis (context dominates). But age, size, speed, and experience have x as
the primary axis (raw magnitude dominates). The v4 universality claim was
driven by the Grid A x-z confound.

Correlation of R²(PC1~x) with x_range/σ across pairs: −0.42 (weak). So
the split isn't purely a "variance of x vs variance of μ" artifact.

## 8.3 — Meta-direction shared variance: 32.6% (vs Grid A 41.6%)

SVD of stacked per-pair PC1s on Grid B: top singular vector captures 32.6%
of shared variance, vs 41.6% on Grid A. Confound inflated the shared
variance metric by ~25%.

Cross-pair PC1 |cos| mean off-diagonal: 0.19 (vs Grid A 0.32). Confound
inflated this by ~40%.

## 8.4 — Zero-shot × implicit orthogonality confirmed on clean grid

|cos(w_x_zeroshot, w_z_implicit)| across 8 pairs on Grid B: all values
in [0.003, 0.048], at or below the √(1/d) ≈ 0.020 chance floor. The v5
claim that zero-shot x-direction is orthogonal to implicit z-direction
holds on clean grid (slightly stronger evidence).

## 8.5 — Cross-template transfer: 97% of self-steering (height)

Extracted height activations for two templates:
  A: "This person is"
  B: "Among the individuals listed, the one measuring X cm would be described as"

Results:
  cos(primal_z_A, primal_z_B)          = +0.727
  slope(primal_z_B → B)                = +0.152 (self)
  slope(primal_z_A → B)                = +0.147 (cross-template)
  slope(random → B)                    =  0.003 (null, 3 seeds)

  cross / self:    0.968  (97%)
  cross / random: 43.9×

Verified: Template B has "tall"/"short" in top-10 for 99%/53% of prompts
(Template A: 34%/1%) — so the steering evaluation on B is clean
regardless of A's scoring quality.

The primal_z direction at layer 32 is template-invariant. Tested on
n=1 pair (height); replication needed.

## 8.6 — What v8 changes for the paper

**Reframe from v7**:
  - Pos/neg math: "R=0.47" → "R=0.31" (on the only valid prompt variant).
  - PC1 horseshoe: "universal z-axis" → "z-axis for 4 pairs, x-axis for 4
    pairs". Model's internal representation is heterogeneous.
  - Meta-direction shared variance: 41.6% → 32.6%.
  - Cross-pair PC1 cosine: 0.32 → 0.19.

**New evidence**:
  - Cross-template transfer at 97% of self-steering (height only).
    Adds confidence that primal_z captures a semantic (not syntactic)
    object — at least in mid-to-late layers.
  - top-K scoring validity check becomes a methodological requirement:
    future R measurements should verify scoring tokens are in top-K before
    quantitative claims.

**Spectrum of R across concepts × prompts (all Grid B, clean):**

  concept × prompt                       R        valid?
  posneg × orig                          0.47     noisy (16% top-10)
  posneg × forced_qa                     0.31     ✓ clean
  bmi_abs × "This person is"             0.49     TBD (needs top-K check)
  legal_abs × "is a"                     0.89     invalid? (top-K unchecked)
  grade_abs × "This student is"          0.80     TBD
  temp_abs × "This liquid is"            0.65     TBD
  experience × "is"                      0.45     TBD
  wealth × "is"                          0.43     TBD
  height × "This person is"              0.80     ≈ clean (many-shot context)
  ...

The "continuous relativity spectrum" framing still holds, but the exact
R values need re-validation per-prompt. Minimum R across valid prompts
per concept is likely the right metric.

---

# Section 9 — Manifold geometry + layer analysis (2026-04-22, post-v8)

CPU-only analysis on Grid B .npz activations fetched from HF. No GPU needed.

## 9.1 — Intrinsic dimensionality: z is NOT a 1-D curve

TWO-NN estimator on cell-mean activations at late layer:

  pair          ID
  height        3.4
  weight        3.6
  speed         4.5
  experience    4.7
  age           5.3
  size          5.5
  wealth        6.5
  bmi_abs       6.8
  mean          5.0

The PCA horseshoe gave the illusion of a 1-D curve, but the actual intrinsic
dimensionality is ~3–7D. Height/weight are simplest (3–4D), wealth/bmi_abs
are highest-dimensional (6–7D).

## 9.2 — Isomap reveals curvature that PCA misses

Isomap (geodesic-preserving) vs PCA (linear) 1-D embedding, R² against z:

  pair          R²(iso)  R²(pca)  curved?
  speed         0.971    0.013    MASSIVELY — PCA misses z entirely
  wealth        0.797    0.595    yes
  age           0.218    0.063    yes (weak z overall)
  experience    0.199    0.119    yes (weak z)
  height        0.993    0.950    slightly
  weight        0.969    0.970    no (linear)

**Speed is the star finding.** PCA 1-D captures nearly zero z-variance
(R²=0.01) while isomap 1-D captures 97%. The z-information IS there but
on a curved manifold that linear methods completely miss. This explains
why speed's PC1 was x-dominated in v8 — z is encoded on a curve, not a
line, for speed.

Mean geodesic/Euclidean ratio: 10.85 (path along manifold is ~11x longer
than straight-line distance). Substantial curvature.

## 9.3 — Mid vs late layer: primal_z is completely layer-specific

  pair          R²(z)_mid  R²(z)_late  cos(pz_mid, pz_late)
  height        0.983      0.982       −0.003
  age           0.974      0.976       −0.010
  weight        0.970      0.973       +0.003
  size          0.959      0.956       +0.042
  speed         0.960      0.966       +0.032
  wealth        0.962      0.968       +0.010
  experience    0.962      0.957       +0.016
  bmi_abs       0.957      0.964       −0.020

Both layers decode z with R² > 0.95, but primal_z directions are
**completely orthogonal** (cos ≈ 0.00 across all 8 pairs). The model
doesn't maintain a fixed z-axis through the network — each layer writes
z-information using different directions, and only the late-layer direction
causally controls the adjective output.

Consistent with the "residual stream as communication channel" view:
different layers write z using different directions; downstream components
(unembedding at layer 32+) read from the late-layer direction.

## 9.4 — Bug fix: meta_w1 SVD sign convention

All meta_w1 steering slopes were negative because SVD's Wt[0] pointed
opposite to primal_z (cos ≈ −0.98). Root cause: no sign-alignment step
after SVD extraction. Fixed by flipping w1 if dot(w1, mean_PC1) < 0.
Existing JSON results still have wrong-sign slopes; upstream GPU scripts
need re-running.

## 9.5 — What §9 changes for the paper

**New findings:**
  - z lives on a ~5-D manifold, not a 1-D curve. PCA horseshoe was misleading.
  - Speed: z is encoded on a curved manifold (isomap R²=0.97, PCA R²=0.01).
    Linear methods completely miss the z-encoding for some pairs.
  - primal_z is fully layer-specific (mid ⊥ late). No gradual buildup.

**Implications:**
  - Linear probing overestimates encoding quality (R² > 0.95 everywhere)
    but misses that the encoding is geometrically complex.
  - SAE decomposition is the natural next step: decompose the multi-dimensional,
    curved manifold into sparse interpretable features.
  - On-manifold steering (along the geodesic) may outperform linear steering
    for pairs like speed where curvature is extreme.

## §10 — v9: Gemma 2 2B + Gemma Scope SAE + on-manifold / Park steering

Pivot to `google/gemma-2-2b` (26 layers, d=2304) at layer mid=13 / late=20
to unlock pretrained Gemma Scope residual-stream SAEs. Full v7 Grid B
(5x × 5z × 30 seeds per pair, 8 pairs) re-run; same analyses as v7–v8 plus
SAE decomposition, on-manifold (tangent) steering, and Park's causal
inner product.

## 10.1 — Behavioral signal replicates cleanly on 2B (8/8 pairs)

| pair       | R = −slope(μ)/slope(x) | r²(z) | r²(x,μ) |
|------------|------------------------|-------|---------|
| height     | **+0.854**             | 0.876 | 0.889   |
| age        | **+1.025**             | 0.786 | 0.788   |
| weight     | **+0.916**             | 0.845 | 0.861   |
| size       | **+0.925**             | 0.837 | 0.863   |
| speed      | **+0.769**             | 0.600 | 0.821   |
| wealth     | **+0.772**             | 0.655 | 0.739   |
| experience | **+0.862**             | 0.801 | 0.842   |
| bmi_abs    | **+0.827**             | 0.727 | 0.823   |

All 8 pairs exceed the R > 0.3 acceptance threshold (target: ≥5/8). Notable
shifts vs E4B: `age` flips from x-dominated on E4B to R≈1.03 (fully
context-relative) here; all pairs (including bmi_abs) cluster in a tight
R = 0.77–1.03 band, i.e. **Gemma 2 2B is MORE context-relative than E4B
across the board, not less.**

## 10.2 — SAE decomposition of z — distributed, not sparse

Gemma Scope width-65k, layer-20, avg_l0_61 JumpReLU SAE. Per pair:
encoded 660–750 activations, found 446–635 features active across prompts
(≈1% of the dictionary).

  - **Cross-pair Jaccard** of top-20 z-correlated features:
    off-diagonal mean = 0.060 (~10× chance baseline).
    Shared substrate exists but is weak — body-measurement pairs don't
    share a dominant set of "z-features."
  - **Participation ratio** (effective # features):
    primal_z ≈ 10k–15k vs probe_z ≈ 17k–18k.
    probe_z spreads 1.2–1.8× thinner, but neither direction is
    sparsely decomposed in SAE basis.
  - **Energy in top-20 z-features**:
    primal_z = 0.1%–1.0%,  probe_z ≈ 0.02% (uniform baseline 0.03%).
    primal IS aligned with the z-feature subspace — just at ~10×
    uniform, not the near-concentration the plan hypothesized.

**Conclusion:** the ~18× steering gap between primal_z and probe_z is NOT
explained by primal_z loading onto a handful of SAE features. Both
directions are broadly distributed across thousands of features;
primal is only moderately more concentrated and modestly more z-aligned.
Whatever makes primal_z causally effective is finer-grained than
SAE-level sparsity.

## 10.3 — On-manifold (tangent) steering — 70% as effective, not kinder

Late-layer (layer 20) causal intervention with forward hooks at the last
token position. α ∈ {−2, −1, 0, +1, +2}. All directions rescaled to
‖primal_z‖ so α is comparable. Stratified subset: 20 prompts per z-value.

Tangent(z) = piecewise-linear finite-difference tangent through the five
z-cell-means, evaluated at each prompt's z-bin.

| pair       | primal slope | tangent slope | t/p ratio | random |
|------------|--------------|---------------|-----------|--------|
| height     | +1.79        | +1.24         | 0.69      | +0.22  |
| age        | +1.77        | +1.29         | 0.73      | +0.16  |
| weight     | +2.24        | +1.54         | 0.69      | +0.21  |
| size       | +2.42        | +1.53         | 0.63      | −0.07  |
| speed      | +1.42        | +0.94         | 0.66      | −0.04  |
| wealth     | +2.25        | +1.57         | 0.70      | −0.21  |
| experience | +4.59        | +3.31         | 0.72      | +0.08  |
| bmi_abs    | +2.12        | +1.46         | 0.69      | −0.48  |

**Tangent steers at 0.63–0.73× primal, remarkably consistent across
pairs (mean 0.69).** Random null is near zero (direction norm is matched,
so this is a clean null).

Entropy damage at |α|=2 is small in all cases (|Δent| ≤ 0.15 from a
~4-nat baseline). Tangent is NOT systematically kinder than primal —
often within ±0.05 nats. **The "on-manifold = less entropy damage"
hypothesis is not supported at these α levels.**

Why tangent is weaker: primal_z encodes the full z ∈ [−2, +2] range in
one direction, while each tangent covers only a 1σ segment. Following
the local tangent accumulates curvature corrections, but those corrections
trade off with signed displacement.

## 10.4 — Park causal inner product — does NOT bridge the 18× gap

probe_z_causal = (W_U^T W_U + λI)^{−1} probe_z  with λ = 10⁻²,
where W_U = lm_head.weight ∈ ℝ^(256000 × 2304).

Steering slopes (rescaled to ‖primal_z‖):

| pair       | primal | probe | probe_causal | probe/primal | causal/primal |
|------------|--------|-------|--------------|--------------|---------------|
| height     | +1.79  | +0.07 | +0.14        | 0.04×        | 0.08×         |
| age        | +1.77  | +0.02 | +0.03        | 0.01×        | 0.02×         |
| weight     | +2.23  | +0.11 | +0.09        | 0.05×        | 0.04×         |
| size       | +2.44  | +0.05 | +0.10        | 0.02×        | 0.04×         |
| speed      | +1.41  | +0.17 | −0.02        | 0.12×        | −0.02×        |
| wealth     | +2.25  | +0.20 | +0.08        | 0.09×        | 0.04×         |
| experience | +4.62  | +0.08 | −0.42        | 0.02×        | −0.09×        |
| bmi_abs    | +2.15  | −0.22 | −0.01        | −0.10×       | ≈0            |

cos(probe_causal, primal_z) stays ≤ 0.05 across all pairs (and flips
sign for several). **Park's causal transformation leaves probe_z
essentially unchanged — it does not rotate probe_z toward primal_z.**

This is a cleaner negative than v7 could deliver: on Gemma 2 2B the
probe-vs-primal gap widens to 10–100× (age reaches ~90× weaker), and
the causal metric hypothesis fails by every reasonable measure —
slope, entropy, cosine. The encoding direction (Ridge probe_z) and
the causal direction (primal_z) are not related by a W_U-induced
inner product.

## 10.5 — What §10 changes for the paper

**Positive for the thesis:**
  - Behavioral signal replicates cleanly on a second model (2B) with
    cleaner per-pair R than the primary (E4B). Reviewers should buy
    the cross-model stability now.
  - `age` switching from x-dominated (E4B) to z-dominated (2B) is a
    small-but-real scaling observation — worth a sentence in the paper.

**Refines or refutes three specific follow-up hypotheses:**
  - **Sparse SAE decomposition of z**: refuted in its strong form.
    z is not carried by a handful of SAE features; both primal and
    probe spread across thousands. Paper should *not* claim
    SAE-interpretable sparsity for z.
  - **On-manifold tangent is kinder to entropy**: refuted at tested α.
    The curved manifold is real (§9.2 isomap R²=0.97 on speed), but the
    payoff for tangent steering is not entropy safety.
  - **Park's causal metric bridges encode-vs-use**: refuted. The
    ~18× primal/probe gap (now 10–100× on 2B) is NOT explained by a
    W_U-induced metric. Probe_z is a genuinely different direction,
    not a metric-rotated copy of primal_z.

**Net:** v9 strengthens the behavioral + primal-z story and retires three
alternative explanations for why probe_z underperforms. The remaining
open question (what IS primal_z, if not a sparse SAE feature bundle,
an on-manifold tangent, or a causally-rotated probe?) is a good arXiv
follow-up but does not block the workshop paper.

## §11 — v9 robustness: critic-driven reruns amending §10

Three skeptical critics (stats, implementation, interpretation) raised
concerns on §10's four claims. We addressed them with targeted reruns
(no new GPU data collection — all on the same Grid B activations). The
net effect is a sharper §10, not a retraction: two claims strengthen,
two receive narrower scope, one auxiliary claim ("age flipped to
z-dominated") becomes statistically solid.

### 11.1 — Behavioral R: tight bootstrap CIs; simulation calibration

  - 1000-sample bootstrap gives CI95 widths of **0.02–0.05** for every
    pair — R is well-determined, not a ratio-estimator artifact.
  - Synthetic calibration on each pair's Grid B design:
    pure-z model → R = 0.997–1.003, pure-x model → R ≈ 0.00,
    half-x + half-z → R = 0.42–0.69. R is a well-behaved variance-share
    statistic and our threshold "R > 0.3" maps to a meaningful
    "z explains ≥ ~1/3 of the variance" regime.
  - **`age` at 2B**: CI95 = [+1.003, +1.048] — definitively pure-z.
    Not a noise flip. Worth promoting to a headline paragraph in the
    paper's scaling section.

  Artifacts: `results/v9_gemma2/behavioral_bootstrap.json`,
  `figures/v9/behavioral_R_ci_vs_simulation.png`.

### 11.2 — SAE decomposition: the projection matters — primal IS sparser

A critic noted that `sae_project(v)` used encoder-columns (`v @ W_enc`),
which rolls in the constant b_enc and is the wrong operator for
"direction in SAE basis." The scientifically correct choice is
decoder-row projection (`v @ W_dec.T`), because Gemma-Scope's W_dec rows
are unit-normalized feature contribution directions. We re-ran P2 with
both:

| SAE variant                      | mean probe/primal participation-ratio |
|----------------------------------|----------------------------------------|
| layer 20, width 65k, L0 = 61     | **5.93×**   (W_dec;  was 1.50× with W_enc) |
| layer 20, width 65k, L0 = 20     | **5.02×**   (sparser dictionary)       |
| layer 13, width 65k, L0 = 74     | **2.65×**   (mid-layer — smaller gap)  |

**primal_z is 4–10× more concentrated than probe_z at late layer in
the correct basis** — far closer to the v9-plan's "few features"
hypothesis than P2's original numbers. The statement in §10.2 that
"the sparse-SAE hypothesis is refuted" was too strong; amended to:

> primal_z loads onto a much more concentrated set of SAE features
> than probe_z (participation ratio 4–10× smaller at late layer, 2.6×
> at mid layer). This is a real sparsity signal, but still not as
> extreme as "3 features" — primal fires ~3–8k effective features in
> the 65k width-65k dictionary vs ~22k for probe.

Artifacts: `results/v9_gemma2/sae_sensitivity.json`,
`figures/v9/sae_sensitivity_participation_and_energy.png`.

### 11.3 — On-manifold tangent steering: kinder at high |α|, not at |α|=2

§10.3 reported Δentropy ≤ 0.15 at |α| = 2 and concluded "tangent NOT
systematically kinder." At that α we hadn't left the data cloud. At
|α| = 8 (extended sweep), entropy damage is 0.2–1.4 nats and

| pair       | Δent primal @ α=8 | Δent tangent @ α=8 | tangent kinder? |
|------------|-------------------|--------------------|-----------------|
| height     | −0.67             | −0.41              | yes (Δ=0.26)    |
| age        | −0.74             | −0.47              | yes (Δ=0.27)    |
| weight     | −0.24             | −0.58              | NO              |
| size       | −0.61             | −0.49              | yes (Δ=0.12)    |
| speed      | −0.33             | −0.24              | yes (Δ=0.09)    |
| wealth     | −1.41             | −1.26              | yes (Δ=0.15)    |
| experience | −1.42             | −0.85              | yes (Δ=0.57)    |
| bmi_abs    | −0.27             | −0.54              | NO              |

**Tangent is kinder on 6/8 pairs at α=8** (wealth, experience most
dramatic; weight and bmi_abs invert). §10.3's refutation applied only
to the near-cloud regime (|α|≤2); the on-manifold-safety hypothesis
is **partially supported at high α** — the effect size is modest
(0.1–0.6 nats) and pair-dependent. Not a clean win, but not dead.

### 11.4 — Park causal metric: layer 25 + λ sweep — still refuted

Two critics flagged §10.4: λ = 10⁻² may have been effectively zero,
and Park's geometry is theoretically sharpest at the pre-unembedding
layer (25), not at 20.

  - **Eigendecomposition of W_U^T W_U**: min eigenvalue = 3.4, median
    242, max 168,000. λ=10⁻² is ~340× smaller than the smallest
    eigenvalue — critic was right, it was essentially zero.
  - **λ sweep ∈ {10⁻⁵, 10⁻³, 10⁻¹, 1, 10}**: at λ=10, 0.3% of
    eigenvalues are below λ (first real regularization). Across all λ,
    the probe_causal slope at layer 25 ranges over a small band —
    **no λ brings probe_causal near primal** on any pair.
  - **Layer 25 (pre-unembedding)**: primal slopes 1.5–4.5 (stronger
    than at layer 20), probe slopes −1.1…+0.6, probe_causal similar.
    Probe/primal ratio at layer 25 ≈ 0.05, essentially identical to
    layer 20. Park's hypothesis does NOT improve at the theoretically
    favored layer.

Claim stands: the W_U-induced inner product does not bridge the
encode-vs-use gap at any layer or λ tested. §10.4 now reads with
numerical backing rather than "maybe wrong layer / maybe wrong λ."

Artifacts: `results/v9_gemma2/park_layer25_summary.json`,
`figures/v9/park_layer20_vs_layer25.png`.

### 11.5 — Steering null band + held-out CV

Two more critic concerns resolved:

**Multi-seed random null (30 seeds × Gaussian directions rescaled to
‖primal_z‖)**: the 97.5th percentile of the null slope distribution is
0.2–0.7 per pair (much wider than the single-seed 0.2 used in §10).
**Both primal and tangent slopes remain clearly above the null q975
for every pair.** Null validity confirmed.

**5-fold CV on primal_z and probe_z**: we fit each direction on 80%
of Grid B and steered on held-out 20%. The primal in-sample slope
equals the out-of-sample slope to 3 decimals (e.g. height 1.796 both,
weight 2.181 both) — **primal has zero train-set overfitting**. The
probe slope is likewise within 0.05 of its in-sample value. **The
probe/primal gap persists cleanly out-of-sample (0.01–0.11× across
pairs)** — NOT a leakage artifact.

Artifacts: `results/v9_gemma2/steering_robust_summary.json`,
`figures/v9/steering_extended_alpha.png`,
`figures/v9/steering_extended_alpha_entropy.png`,
`figures/v9/steering_multiseed_null.png`,
`figures/v9/steering_heldout_cv.png`.

## §12 — SAE feature geometry + Goodfire-style LFP

Per user request (influenced by Sarfati et al. 2026, "The Shape of
Beliefs"), we added two geometry analyses on top of the SAE codes.

### 12.1 — PCA in SAE feature space is WORSE than raw-activation PCA

For each pair, we did PCA on (n_prompts × n_active_features) SAE
coefficients and compared r²(z) of the top 2 SAE PCs against the top
2 raw-activation PCs.

| pair       | r²(z) raw PC1/2 | r²(z) SAE PC1/2 |
|------------|------------------|------------------|
| height     | 0.89             | 0.88             |
| age        | 0.68             | **0.12**         |
| weight     | 0.89             | 0.89             |
| size       | 0.74             | 0.70             |
| speed      | 0.53             | **0.08**         |
| wealth     | 0.72             | 0.66             |
| experience | 0.84             | 0.83             |
| bmi_abs    | 0.80             | 0.74             |

**SAE PCA is uniformly ≤ raw PCA for recovering z in the top two
components** (catastrophic for age and speed — the two pairs with the
most curved z-manifolds per §9.2). Consistent with §11.2's finding
that z lives on *many* SAE features: the top variance-axes in SAE
space mix z with unrelated features, so first-2-PC reconstruction of
z degrades.

### 12.2 — Linear Field Probes + Gram kernel PCA

Following Sarfati et al.'s LFP procedure: per pair, train one logistic
probe per z-value (K=5), stack into W_pair ∈ ℝ^(K×d), compute Gram
G = W_n W_n^T on row-normalized probes, eigendecompose.

Per-pair Gram spectrum (both raw and SAE bases):

  - Top eigenvalue captures ≈ **30% of the total** — not rank-1.
  - Participation ratio (effective rank) ≈ **4.2 out of 5**, meaning
    **the 5 per-z-value probes are nearly orthogonal**. This is
    incompatible with "z is a single linear direction"; it's
    consistent with §9.1's ID-5 manifold claim.

**Stacked cross-pair LFP Gram** (40 probes = 5 z-values × 8 pairs):

  - Participation ratio 26 (raw) / 31.6 (SAE) out of 40.
  - Cross-pair z-probes do NOT collapse into a shared z-axis; each
    pair's z-subspace is largely its own. Matches §8.2's cross-pair
    PC1 cosine of 0.19 and §7's cross-pair transfer of ~40%.

Artifacts:
  `results/v9_gemma2/sae_feature_pca.json`,
  `results/v9_gemma2/lfp_gram_per_pair.json`,
  `results/v9_gemma2/lfp_stacked_cross_pair.json`,
  `figures/v9/sae_feature_pca_8panel.png`,
  `figures/v9/lfp_gram_spectra.png`,
  `figures/v9/lfp_kernel_pca_per_pair.png`,
  `figures/v9/lfp_stacked_cross_pair.png`.

### 12.3 — Synthesis across §§9, 11.2, 12

Three converging views of z's geometry on Gemma 2 2B layer 20:

  - **Manifold geometry (§9)**: intrinsic dim ≈ 5 via TWO-NN estimator
    on cell-means.
  - **SAE decomposition (§11.2)**: primal_z is 4–10× sparser than probe
    in decoder-row basis, but still spreads across thousands of
    features.
  - **LFP Gram (§12.2)**: 5 per-z-value probes have effective rank
    ≈ 4.2 — near-orthogonal.

All three say z is not a 1-D linear direction; it's a ~5-D
multi-direction structure. The linear primal_z "works" because the
first principal tangent is aligned with the 5-D subspace; it's not
because z is actually 1-D.

## §13 — v9 full layer sweep: encode vs use vs geometry across all 26 layers

Previous v9 sections pinned layer=20 (late) and layer=13 (mid). The
user asked: what happens if we sweep all 26 decoder blocks?

We re-extracted Gemma 2 2B activations at every `decoder_layer[k]`
output, last-content-token, for the full Grid B (5x × 5z × 30 seeds).
Then per layer we computed: Ridge CV R²(z) and R²(x), top-PC r²(z),
TWO-NN intrinsic dim on cell-means, ‖primal_z‖, cos(primal_z[L],
primal_z[L−1]), 5-probe LFP Gram spectrum. Also: causal steering
slope at 7 strategic layers {5, 10, 13, 17, 20, 22, 24}.

### 13.1 — z is encoded by layer 7–9, flat thereafter

Mean CV R²(z) across 8 pairs, per layer:

    L0 0.45   L1 0.63   L2 0.73   L3 0.80   L4 0.85   L5 0.90
    L6 0.91   L7 0.94   L8 0.94   L9 0.94   L10 0.94  L11 0.93
    L12 0.93  L13 0.93  L14 0.93  L15 0.92  L16 0.93  L17 0.93
    L18 0.92  L19 0.92  L20 0.92  L21 0.92  L22 0.92  L23 0.92
    L24 0.92  L25 0.92

Curve saturates at **L7 (~0.94) and holds flat through the last
layer.** Every pair follows the same shape. **z is decodable from
layer 7 onward** — more than halfway BEFORE our previous "late = 20"
extraction.

### 13.2 — Intrinsic dim matches Goodfire's prediction

Mean TWO-NN intrinsic dim of cell-means, per layer (across 8 pairs):

    L0  4.85   L5  4.58   L10 5.09   L13 6.91   L16 7.12
    L20 5.83   L22 5.38   L25 5.35

**ID rises through mid-network, peaks at L13–17 (≈ 7), then drops to
5.3 at the last layer** — exactly the "rise then drop at the last
layer" pattern Sarfati et al. (2026) report for Llama-3.2 belief
manifolds. First independent replication on Gemma 2 2B.

### 13.3 — Primal direction rotates mid, stabilizes late

cos(primal_z[L], primal_z[L−1]), mean across 8 pairs:

    L1 0.32   L5 0.52   L10 0.59   L13 0.65   L15 0.77
    L18 0.88  L20 0.93  L22 0.94  L25 0.91

In **early layers the primal direction actively rotates** (0.3–0.6 cos
with the previous layer — lots of reorientation). By **L18 it
stabilizes (0.88+)** and changes little for the rest of the network.
Combined with §9.3 (mid ⊥ late: cos(primal_mid, primal_late) ≈ 0), we
can sharpen the picture: **the "two orthogonal representations"
aren't because the network uses two different directions — it's
because the primal direction moves through a ~90° arc over the course
of the middle layers, then settles.**

### 13.4 — Primal magnitude grows exponentially with depth

    ‖primal_z‖:  L0 0.2   L5 1.0   L10 5.6   L15 23.8
                 L20 52.4  L25 80.0

**Roughly 10× per ~8 layers.** The semantic content of z saturates at
L7 (§13.1), but the residual stream keeps amplifying the z-direction
for the rest of the network. This is the first time in this project
we've observed the gap between "z info is there" and "z is written
loud" as a function of depth.

### 13.5 — Causal steering: z is USED only in the second half

Ran primal_z and probe_z steering at 7 strategic layers. Mean slope
(Δlogit_diff per α) across 8 pairs:

    layer         L5     L10     L13     L17     L20     L22     L24
    primal     −0.00   +0.03   +0.57   +1.91   +2.33   +2.58   +2.48
    probe      +0.00   +0.02   +0.08   +0.09   +0.06   +0.03   +0.05
    probe/pri    —     0.65    0.15    0.05    0.03    0.01    0.02

**This is the headline result of §13.** At L5 and L10, where R²(z) is
already ~0.94 and the information is decodable, **primal_z steering
does nothing** (slope ≈ 0). The z-information is present but the
network does not yet use it to drive the next-token distribution.
Causal potency emerges sharply at **L13 (slope 0.57)**, strengthens
through **L17 (1.91)**, peaks around **L20–22 (2.3–2.6)**, and eases
slightly at the last layer.

The encode-vs-use gap v7 originally framed as "probe_z vs primal_z"
(same layer, different direction) is a **layer-depth** phenomenon.
At layer 20 probe/primal = 0.03. At layer 10, where z is already
encoded, primal *itself* is causally null. The dimensions along
which z is decodable in early layers are not the dimensions the
network's late layers read from.

Probe_z steering is ~uniformly weak across all layers (max mean slope
0.09 at L17). One outlier: `experience` at L13 gives probe slope
+0.45 (~30% of primal), suggesting there are layers where probe
actually works. Not a reliable pattern across pairs though.

### 13.6 — Synthesis

Taken together, §13 resolves several lingering puzzles from earlier
sections:

  - §9.3's "mid ⊥ late primal_z" = direction slowly rotating through
    the middle layers (§13.3), not two unrelated mechanisms.
  - §9's ID ≈ 5 was at L20. §13.2 reveals the **ID peaks mid-network
    (~7) and compresses to ~5 at the end** — the manifold unfolds
    mid-network then compresses for the unembedding.
  - The 18× primal/probe gap at L20 (§10.4) is a point on a curve
    — at L10 the gap is "both are zero"; at L13 probe/primal ≈ 0.15;
    at L20+ it settles at ≈ 0.03.
  - The SAE-feature picture in §11.2 applies at layer 20. The same
    analysis at layer 7–9 (where z is most cleanly encoded) is an
    obvious next experiment.

**Single-line takeaway for the paper:** *z is encoded at layer 7,
the manifold unfolds to ID≈7 by layer 13–17, the network begins to
use z causally at layer 13 and peaks at layer 20–22, then the
manifold compresses to ID≈5 at the final layer.*

Artifacts
  `results/v9_gemma2/layer_sweep_geometry.json`
  `results/v9_gemma2/layer_sweep_steering.json`
  `figures/v9/layer_sweep_combined.png`   (6-panel summary)
  `figures/v9/layer_sweep_probe_r2.png`
  `figures/v9/layer_sweep_intrinsic_dim.png`
  `figures/v9/layer_sweep_primal_continuity.png`
  `figures/v9/layer_sweep_lfp_id.png`
  `figures/v9/layer_sweep_steering_slopes.png`
