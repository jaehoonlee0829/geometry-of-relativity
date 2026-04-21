# Next GPU Session v6 — Follow-Up Experiments

**Created:** Apr 21, 2026
**Context:** Post-v5 GPU session. Red-team review of new results (Fisher cosines, meta-steering, absolute controls, G31B replication, zero-shot expansion).

---

## What was completed in v5 GPU session

| Exp | Description | Status | Key Result |
|-----|-------------|--------|------------|
| 1 | Zero-shot expansion (1200 prompts) | ✅ Done | cos(w_x_zeroshot, w_z_implicit) ≈ 0 — null-consistent |
| 2 | Meta w₁ steering + random null | ✅ Done | 3–28× effect over random; works across all 8 pairs |
| 3a | Σ⁻¹ cosines | ✅ Done | Σ⁻¹ ≈ I (regularization-dominated) — uninformative |
| 3b | Fisher F⁻¹ cosines | ✅ Done | F⁻¹ ≈ I at cell-means — H4 refuted |
| 4d | Zero-shot bias validation | ✅ Done | 4/8 biased high, 2/8 x-insensitive |
| 5 | Re-extract per-pair logits + plots | ✅ Done | 8-panel heatmaps, scatter plots, CSVs |
| 7 | Absolute controls (temp, legal, grade) | ✅ Done | p=0.75, NOT significant |
| G31B | Cross-model replication | ✅ Done | 6/8 pairs more relative at scale |

---

## What needs to happen next (v6 session)

### Priority 1: Systematic 6-Direction Steering Comparison

**Motivation:** We steered with w₁ and per-pair w_z. But we haven't compared primal vs dual steering, and we haven't tested whether steering pushes off-manifold (the Goodfire question).

**6 directions to test (for height pair as pilot, then extend):**

| # | Name | How to compute | Type |
|---|------|---------------|------|
| 1 | Primal z | x̄(z=+2) − x̄(z=−2) | Unsupervised diff-of-means |
| 2 | Primal x | x̄(x=180) − x̄(x=150) | Unsupervised diff-of-means |
| 3 | Probe w_z | Ridge(acts, z).coef_ | Supervised probe |
| 4 | Probe w_x | Ridge(acts, x).coef_ | Supervised probe |
| 5 | Meta w₁ | SVD of stacked PC1s | Cross-concept unsupervised |
| 6 | Zero-shot w_x | Ridge(zero_shot_acts, x).coef_ | Supervised, no context |

**What to measure per direction:**
- Logit_diff shift vs α (monotone? linear?)
- Project steered activations onto PC1/PC2 — do they stay on the parabola?
- Softmax entropy at steered point — does it break?
- Cross-pair transfer — does height's w_z steer age?

**Estimated time:** ~30 min on H100

### Priority 2: On-Manifold vs Off-Manifold Test

**Motivation:** Our PCA parabola (PC1~z, PC2~z²) looks like Goodfire's curved belief manifold. If linear steering pushes off-manifold, that's evidence the curvature is real (not PCA artifact). If it stays on the parabola, our representation may be simpler/more linear.

**What to do:**
- For each of the 6 steering directions above, at each α value:
  - Project steered activation onto per-pair PC1/PC2
  - Measure distance from the nearest point on the fitted parabola
  - Plot steered trajectory overlaid on the PCA parabola

**Estimated time:** ~10 min (post-processing, no GPU needed if we save steered activations)

### Priority 3: Fisher at Peaked Activations

**Motivation:** H4 failed at cell-mean activations because softmax is near-uniform there (high entropy → F ≈ (1/V)·I). But the model DOES make peaked predictions — at extreme z values, softmax concentrates on "tall" or "short". Fisher might be non-trivial there.

**What to do:**
- Select activations where |logit_diff| > 3 (confident predictions)
- Recompute F(h) at these peaked activations
- Check if F⁻¹ is no longer isotropic
- Recompute cos_F⁻¹(w_adj, w_z) at peaked vs flat activations

**Why this matters:** If Fisher only works at peaked predictions, H4 isn't refuted — it's scope-limited. The paper could say "Fisher-pullback diagnosis works when the model is confident, but not at average activations."

**Estimated time:** ~20 min on H100

### Priority 4: PC2 Steering (Surprise/Atypicality Direction)

**Motivation:** PC2 correlates with z² across all pairs. If steering along PC2 changes hedging/uncertainty language without flipping tall/short, it's a real "surprise" feature. If nothing happens, it's a PCA artifact.

**What to do:**
- Extract per-pair PC2 direction
- Steer with α·PC2 at layer 32
- Measure: does logit_diff change? Does entropy change? Do hedging tokens ("somewhat", "relatively", "quite") appear more?

**Estimated time:** ~15 min on H100

### Priority 5: Improve Absolute Control Design

**Motivation:** legal_abs failed (R=0.89) because "adult" is polysemous. Need better-designed absolute pairs.

**Better candidates:**
- **boiling_abs:** "boiling" / "frozen" with water temperature (anchored to 100°C/0°C) — but prompt must say "This water is" not "This person is"
- **pregnant_abs:** "pregnant" / "not pregnant" — truly binary, no gradient. But weird as adjective pair.
- **alive_abs:** "alive" / "dead" — binary. Context shouldn't matter.

**Or reframe entirely:** Instead of finding "absolute" pairs, argue that ALL adjectives show contextual relativity to varying degrees. The spectrum IS the finding.

---

## Discussion Points from Red-Team (to resolve before v6)

1. **Is w₁ a "relativity axis" or a "polarity axis"?** bmi_abs steers with 28.5× effect (largest), which contradicts "absolute control." If w₁ is just "high word probability ↑", it's less interesting than "context-relative extremity ↑."

2. **Should we drop the absolute/relative framing?** With p=0.75, the dichotomy is dead. Reframe as "relativity spectrum" where all concepts show some context-dependence?

3. **What to do with the Fisher negative result?** Three options:
   - (a) Report honestly as "H4 refuted at cell-means, open at peaked activations"
   - (b) Test peaked activations (Priority 3 above) — if it works, H4 is rescued
   - (c) Drop Fisher from the paper entirely, lead with causal steering instead

4. **Goodfire comparison:** Our linear steering works (logit_diff shifts monotonically). Goodfire's doesn't (output breaks). Does this mean our manifold is less curved, or that logit_diff is too coarse to detect the breakage?
