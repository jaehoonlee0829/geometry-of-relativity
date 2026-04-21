# Next GPU Session — Follow-Up Experiments

**Created:** Apr 21, 2026
**Context:** Red-team review exposed several gaps in the v4 evidence. These experiments address the most critical ones.

---

## 1. Zero-Shot Direction + More Prompts

**Problem:** Current zero-shot has only 5 datapoints per pair (one per target x value). Too few to extract a meaningful direction or compare geometrically with implicit/explicit.

**What to do:**
- Expand zero-shot grid: 5 x values × 30 seeds = 150 prompts per pair (vary prompt phrasing, not context)
- Extract activations at mid + late layers
- Train w_z_zeroshot probe on zero-shot activations
- Compare cos(w_z_zeroshot, w_z_implicit) — are the "tallness" directions the same with and without context?
- PCA on zero-shot activations: does PC1 still track x (raw value)? It should track x, NOT z (since there's no context to define z)

**Why it matters:** Zero-shot reveals the model's prior. If zero-shot PC1 tracks x and implicit PC1 tracks z, that's direct evidence the model *constructs* z from context, not just reads x.

**Estimated time:** ~5 min extraction on H100

---

## 2. Steering with Meta Z-Direction (w₁) — Concept-Agnostic

**Problem:** We steered with per-pair w_z directions. We haven't tested whether the shared meta-direction w₁ (from SVD of stacked PC1s) works as a universal "relativity knob."

**What to do:**
- Load the meta-direction w₁ from `results/v4_adjpairs_analysis/meta_z_direction.json`
- For each of the 8 pairs, steer with α·w₁ at layer 32
- Sweep α ∈ {−8, −4, −2, −1, 0, +1, +2, +4, +8}
- Measure logit_diff shift per pair
- Compare: does w₁ steering work uniformly across all 8 pairs?

**Why it matters:** If w₁ causally shifts adjective judgment across height, age, wealth, etc., that's the strongest evidence for a domain-general "relativity substrate." This is potentially the highest-impact finding.

**Estimated time:** ~15 min on H100 (8 pairs × 9 alphas × ~100 prompts)

---

## 3. Σ⁻¹ and F⁻¹ Cosines (Properly Persisted)

**Problem:** Σ⁻¹ cosines were computed but not saved to JSON. F⁻¹ (the theoretical centerpiece, H4) was never computed on real data.

### 3a. Σ⁻¹ cosines — fix persistence
- Re-run `analyze_v4.py` Phase 2 with explicit JSON output for Σ⁻¹ cosines
- Save cos_Σ⁻¹(w_adj, w_z), cos_Σ⁻¹(w_adj, w_x), cos_Σ⁻¹(w_z, w_x) per layer per pair

### 3b. F⁻¹ cosines — new experiment
- Extract W_U (unembedding matrix) from Gemma 4 E4B: shape (256128, 2560)
- For a sample of ~50 representative activations (e.g., cell-mean activations):
  - Compute F(h) = W_U^T (diag(p) - pp^T) W_U using `src/fisher.py`
  - Solve F⁻¹·w_adj, F⁻¹·w_z, F⁻¹·w_x via Cholesky
  - Compute cos_F⁻¹(w_adj, w_z) and cos_F⁻¹(w_adj, w_x)
- Compare: does F⁻¹ make w_adj and w_z align MORE than Euclidean or Σ⁻¹?

**Why it matters:** This is hypothesis H4 — the paper's theoretical claim. Without it, the Fisher-geometric framing is unvalidated.

**Estimated time:** ~30 min on H100 (F(h) computation is O(V·d²) per activation, ~50 activations)

---

## 4. Improve Token Logits Design

**Problem:** Current approach extracts logit(exact_token_"tall") - logit(exact_token_"short"). This is:
- Biased by RLHF politeness (model avoids saying "short" about a person — social desirability)
- Biased by token frequency asymmetry ("tall" may appear more than "short" in training data)
- Missing synonym mass ("very tall", "quite tall", "taller", etc.)

**Options (ranked by effort):**

### 4a. Quick fix — aggregate synonym families (low effort)
- Define synonym sets: tall_family = {" tall", " Tall", " taller", " very"}, short_family = {" short", " Short", " shorter"}
- Sum log-probs within each family: log_prob_tall = logsumexp(logits[tall_family])
- Use log_prob_tall - log_prob_short as the signal
- **Caveat:** Requires re-extraction (new token IDs to score)

### 4b. Medium fix — probability mass ratio (medium effort)
- Extract full softmax at last position
- Define "tall-ish" and "short-ish" token sets via embedding similarity (top-50 nearest neighbors to "tall"/"short" in embedding space)
- Compute P(tall_set) / P(short_set)
- More principled but requires defining the sets carefully

### 4c. Proper fix — contrastive pair design (high effort, may need full redo)
- Use minimal-pair prompts where the ONLY difference is the target word
- Measure P("tall" | context) directly, not logit_diff
- Or use the DLI (Difference in Log-probability of Interest) metric from causal mediation analysis
- **This may require re-running all 6,240 prompts**

### 4d. Validation check — does RLHF bias matter for Gemma 4 E4B?
- Gemma 4 E4B is a BASE model, NOT instruction-tuned
- RLHF politeness bias may be minimal or absent
- **Quick check:** Compare P("short") vs P("tall") in zero-shot at x=150cm (where "short" is objectively correct). If P("short") < P("tall") even here, there's a bias. From our data: zero-shot x=150 gives logit_diff = +2.75 (tall!) — so YES, there IS a prior bias, but it may be frequency-based rather than RLHF-based since this is a base model.

**Recommendation:** Start with 4d validation. If bias is frequency-based (base model), it's less concerning — the relative shift with context is what matters, and that's clean. If we re-extract, do 4a (synonym families) simultaneously.

---

## 5. Re-Extract and Save Per-Pair Logit Data + Generate Missing Plots

**Problem:** The v4_adjpairs logit files (per-prompt logit_diff for all 8 pairs) were left on the Vast GPU instance and not downloaded. We only have aggregate stats in `summary.json`. This means we CANNOT generate:
- Per-pair (x, μ) logit_diff heatmaps (the diagonal z-tracking pattern)
- Per-pair cell-mean CSVs for implicit/explicit/zero-shot comparison
- Any new visualization that requires per-prompt data

**What to do:**
- Re-run `extract_v4_adjpairs.py` (fast: ~2 min on H100 for 6,240 prompts)
- Download ALL `.jsonl` logit files and `_trials.jsonl` locally
- Run `analyze_v4_adjpairs.py` to generate per-pair heatmaps
- Fix `pca_per_pair_late.png` — text labels are overlapping, need layout fix

**Plots to generate once data is re-extracted:**
- 8-panel logit_diff heatmap grid (x vs μ, one panel per pair) — the "hero" behavioral figure
- 8-panel (x vs z) heatmap grid — shows within-column constancy (pure relativity test)
- Per-pair implicit vs explicit comparison scatter
- Per-pair logit_diff vs z scatter (colored by x) with cell means overlaid
- Zero-shot logit_diff by x per pair (already generated from summary.json, but re-do with full data)

**NOTE:** These plots CANNOT be generated locally — the per-prompt `.jsonl` logit files were left on the Vast GPU instance. Only aggregate stats in `summary.json` are available locally. This re-extraction is BLOCKING for most visualization work.

## 6. Drop w_adj or Justify It

**Problem:** w_adj = Ridge(activations, sign(z)) is nearly tautological with w_z. Binarizing a continuous variable and showing the resulting probe correlates with the continuous probe is circular. This should be dropped from the paper or clearly justified.

**The actually informative comparisons are:**
- w_ld (what the model actually outputs) vs w_z (ground-truth z direction)
- w_z vs w_x (context-relative vs absolute)
- cos(w_z, w_ld) = 0.018 at late layer — surprisingly low, meaning z-probe and logit_diff-probe are nearly orthogonal in Euclidean space. THIS is the interesting finding.

**Action:** Remove w_adj from main results. If kept at all, relegate to appendix with clear justification (e.g., "sanity check that binary adjective classification recovers the continuous z-direction").

---

## Priority Order

1. **Experiment 5** (re-extract logits + plots) — BLOCKING, need data for everything else
2. **Experiment 2** (meta-direction steering) — highest potential impact, moderate effort
3. **Experiment 3b** (F⁻¹ cosines) — validates the paper's theoretical claim
4. **Experiment 1** (zero-shot expansion) — clean experiment, fast
5. **Experiment 3a** (Σ⁻¹ persistence) — trivial, just re-run analysis
6. **Experiment 6** (drop w_adj) — editorial, no GPU needed
7. **Experiment 4** (token design) — important but may require full re-extraction

## 7. Add More Absolute-Adjective Controls

**Problem:** We only have 1 absolute control (bmi_abs: underweight/obese). With n=1, we can't claim the absolute class behaves differently with statistical confidence. Need at least 2–3 absolute pairs.

**Candidate absolute adjective pairs:**
- **Temperature:** freezing / boiling (anchored to 0°C / 100°C) — "This water at X°C is ___"
- **Legal age:** minor / adult (anchored to 18) — "This person at age X is ___"
- **Poverty line:** poor / wealthy (anchored to specific income thresholds) — but this overlaps with our relative "wealth" pair
- **Pass/fail:** failing / passing (anchored to 50% or 60% score) — "A student with a score of X% is ___"

**What to do:**
- Add 2–3 absolute pairs to the extraction script
- Same grid design: 5 x × 5 μ × 30 seeds = 780 prompts each
- Compare relativity ratio R for absolute vs relative pairs
- With 3+ absolute controls, can do proper t-test: R_relative vs R_absolute

**Why it matters:** With only 1 absolute pair, the relative-vs-absolute distinction is anecdotal. With 3+, it becomes a statistical claim.

---

**Total estimated GPU time:** ~2 hours on H100
