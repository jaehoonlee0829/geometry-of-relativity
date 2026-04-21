# Next GPU Session v7 — Clean Rerun with (x, z) Grid

**Created:** Apr 21, 2026
**Context:** Red-team review of v6 results uncovered a fundamental confound in the activation geometry experiments: the (x, μ) grid design makes z correlated with x (Cor ≈ 0.58), which contaminates all diff-of-means directions and inflates cos(primal_z, primal_x). v7 reruns v4–v6 experiments on a properly balanced (x, z) grid where primal_z is a clean z-direction.

---

## The Confound (Why v7 Exists)

All v4–v6 activation experiments used a **(x, μ) grid**: 5 x-values × 5 μ-values × 30 seeds = 750 prompts per pair. z = (x − μ) / σ is a derived quantity, correlated with x by design.

**Concrete example (height):**

| x\μ | 145 | 155 | 165 | 175 | 185 |
|-----|-----|-----|-----|-----|-----|
| 150 | +0.5 | −0.5 | −1.5 | −2.5 | −3.5 |
| 160 | +1.5 | +0.5 | −0.5 | −1.5 | −2.5 |
| 165 | +2.0 | +1.0 | 0.0 | −1.0 | −2.0 |
| 170 | +2.5 | +1.5 | +0.5 | −0.5 | −1.5 |
| 180 | +3.5 | +2.5 | +1.5 | +0.5 | −0.5 |

- z > +1 group: mean(x) = **172.1** cm (skewed toward tall people)
- z < −1 group: mean(x) = **157.9** cm (skewed toward short people)
- Delta = 14.3 cm — almost half the 30 cm x-range

**Consequences:**
1. `primal_z = mean(acts|z>1) − mean(acts|z<-1)` is contaminated by x-variation (~48% confound ratio)
2. `cos(primal_z, primal_x) = 0.91` is partly a design artifact, not a neural finding
3. PC1 from PCA on (x, μ) cell means blends x and z effects inseparably
4. The v6 "7 directions collapse to 4 clusters" conclusion overstates the collapse
5. `meta_w1 ≈ −mean(primal_z)` (cos=0.98) inherits the same contamination through PC1

**What was NOT affected:**
- Behavioral regression `logit_diff ~ b·x + c·μ` (x ⊥ μ by design, clean)
- Ridge probes (multivariate regression, less confounded: cos(probe_z, probe_x) = 0.54 matches the design correlation)
- Meta_w1 causal steering effect (3–28× over random is a causal fact regardless of what w1 geometrically is)
- The behavioral relativity ratios R

---

## v7 Design: Dual-Grid Approach

Run **two grids per pair** on the same GPU session:

### Grid A: (x, μ) grid — for behavioral regression (unchanged from v4)

Same as before: 5 x × 5 μ × 30 seeds = 750 prompts. This gives clean `logit_diff ~ b·x + c·μ` because x ⊥ μ. Reuse v4 data if activations are still on HF; otherwise re-extract.

### Grid B: (x, z) grid — for activation geometry (NEW)

Independently vary x and z, derive μ = x − σ·z.

**Height example:**

| x\z | −2.0 | −1.0 | 0.0 | +1.0 | +2.0 |
|-----|------|------|-----|------|------|
| 150 | μ=170 | μ=160 | μ=150 | μ=140 | μ=130 |
| 160 | μ=180 | μ=170 | μ=160 | μ=150 | μ=140 |
| 165 | μ=185 | μ=175 | μ=165 | μ=155 | μ=145 |
| 170 | μ=190 | μ=180 | μ=170 | μ=160 | μ=150 |
| 180 | μ=200 | μ=190 | μ=180 | μ=170 | μ=160 |

**Properties:**
- z > +1 group has mean(x) = **165.0** = mean(all x). **No x-confound.**
- z < −1 group has mean(x) = **165.0** = mean(all x). **No x-confound.**
- primal_z from this grid is a pure z-direction
- **Trade-off:** μ is now correlated with x (for fixed z, μ = x − σ·z varies linearly with x), so this grid cannot cleanly separate x from μ. That's fine — Grid A handles behavioral regression.

**Size:** 5 x × 5 z × 30 seeds = 750 prompts per pair. Same GPU cost as v4.

**Domain sanity checks:** Some (x, z) cells produce extreme μ values (e.g., height x=180, z=−2 → μ=200). For each pair, verify that all derived μ values are within a plausible range. If not, either clip or drop those cells and note the imbalance.

---

## Experiments to Run (Replicating v4–v6 on Grid B)

### Priority 1: Extract Grid B Activations + Logits

**What to do:**
- Modify `extract_v4_adjpairs.py` to accept a `grid_mode` parameter: `"xmu"` (v4 default) or `"xz"` (new)
- When `grid_mode="xz"`: iterate over (x, z) pairs, compute μ = x − σ·z, generate prompts with that μ
- Extract activations at mid + late layers, save logit_diff per prompt
- Output: `results/v7_xz_grid/{pair}_{layer}.npz` + `_trials.jsonl` + `_logits.jsonl`

**Estimated time:** ~3 min on H100 (same prompt count as v4)

### Priority 2: Clean Primal Directions + Confound Audit

**What to do on Grid B data:**
- Compute `primal_z_clean = mean(acts|z>1) − mean(acts|z<-1)` — now x-balanced
- Compute `primal_x_clean = mean(acts|x>=x75) − mean(acts|x<=x25)` — now z-balanced
- Compute Ridge probes: `probe_z_clean = Ridge(acts, z)`, `probe_x_clean = Ridge(acts, x)`
- PCA on cell means: PC1, PC2
- Compute the full 7×7 cosine matrix (same as v6 `exp_v6_direction_confounds.py`)

**Key comparison:**
- Does cos(primal_z_clean, primal_x_clean) drop from 0.91 to something lower?
- If it drops to ~0.54 (matching the probe cosine), the v6 "collapse" was indeed an artifact
- If it stays high, the model genuinely merges x and z directions (interesting finding)

**Also compute cross-grid comparison:**
- cos(primal_z_v4, primal_z_v7) — how much did the confound distort the direction?
- cos(probe_z_v4, probe_z_v7) — probes should be more stable across grids

**Estimated time:** ~5 min CPU (no GPU needed if activations are extracted)

### Priority 3: 7-Direction Steering on Clean Directions

**Replicate v6 Block D with Grid B directions:**
- Same 7 directions but computed from Grid B activations:
  1. primal_z_clean (x-balanced diff-of-means on z)
  2. primal_x_clean (z-balanced diff-of-means on x)
  3. probe_z_clean (Ridge on z from Grid B)
  4. probe_x_clean (Ridge on x from Grid B)
  5. meta_w1_clean (SVD of stacked Grid B PC1s)
  6. zeroshot_wx (unchanged — zero-shot has no context, no confound)
  7. PC2_clean (from Grid B PCA)
- Same α-sweep, same logit_diff + entropy measurement
- Same prompts for evaluation (can use Grid A prompts for behavioral measurement)

**Key question:** Does primal_z_clean still steer 6× better than probe_z_clean? If yes, the v6 finding is real (not a confound artifact). If the gap closes, the gap was inflated by x-contamination in primal_z.

**Estimated time:** ~5 min on H100

### Priority 4: Clean Transfer Matrix

**Replicate cross-pair transfer with clean directions:**
- Use primal_z_clean (not just probe_z) for the transfer matrix
- This addresses the v6 critic flag: "you tested transfer with the weakest direction (probe_z); test with the strongest (primal_z)"
- Include random null: 3 random unit vectors per pair for comparison

**Key question:** Does primal_z from pair A steer pair B? If cross-pair transfer works with clean primal_z, that's stronger evidence for a shared substrate than meta_w1 (which may just be averaging out pair-specific directions).

**Estimated time:** ~5 min on H100

### Priority 5: Park Metric + Fisher on Clean Directions

**Replicate v6 Blocks A + C with Grid B probes:**
- Compute cos_Park(probe_z_clean, probe_ld_clean) — does Park metric reveal hidden alignment when the probes are computed on a clean grid?
- Fisher at truly low-entropy activations: instead of conditioning on |logit_diff| > p90 (which v6 showed doesn't correlate with low softmax entropy), condition on **softmax entropy < p10** directly. This requires saving softmax entropy per prompt during extraction.

**Estimated time:** ~10 min on H100

### Priority 6: INLP on Grid B

**Replicate v4 INLP with clean grid:**
- The v4 INLP finding (removing z-directions barely reduces R²(z)) may also be affected by the confound: if the z-directions being removed are contaminated with x, INLP removes x-information too, and x can reconstruct z via the design correlation.
- On Grid B: INLP removes z-directions that are x-balanced. Check if R²(z) drops faster.

**Estimated time:** ~2 min CPU

---

## Implementation Notes

### Modifying `extract_v4_adjpairs.py`

```python
# New grid mode parameter
def generate_all_prompts(grid_mode: str = "xmu") -> list[dict]:
    trials = []
    for pair in PAIRS:
        if grid_mode == "xmu":
            # Original: iterate (x, mu), derive z
            for x in pair.target_values:
                for mu in pair.mu_values:
                    z = compute_z(pair, x, mu)
                    ...
        elif grid_mode == "xz":
            # New: iterate (x, z), derive mu
            z_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
            for x in pair.target_values:
                for z in z_values:
                    if pair.name in LOG_SPACE_PAIRS:
                        mu = x / (pair.sigma ** z)
                    else:
                        mu = x - pair.sigma * z
                    # Sanity check: skip if mu is implausible
                    ...
```

### Domain-Specific μ Plausibility

For each pair, check that derived μ values are reasonable:

| Pair | x range | z range | Derived μ range | Issue? |
|------|---------|---------|-----------------|--------|
| height | 150–180 | −2 to +2 | 130–200 | μ=200 is unusual but not impossible |
| age | 20–60 | −2 to +2 | 0–80 | μ=0 is problematic (age can't be 0) |
| weight | 50–100 | −2 to +2 | 30–120 | OK |
| size | 5–60 | −2 to +2 | −45 to 110 | μ=−45 is nonsensical for room size |
| speed | 20–150 | −2 to +2 | −40 to 210 | μ=−40 is impossible |
| wealth | log-space | −2 to +2 | varies | Need log-space version |
| experience | 1–25 | −2 to +2 | −9 to 35 | μ=−9 is impossible |
| bmi_abs | 17–38 | −2 to +2 | 7–48 | borderline OK |

**Resolution:** For cells with implausible μ, either:
1. Narrow z range to [−1.5, +1.5] or [−1, +1] where μ stays plausible
2. Clip μ to domain-valid range and note the z-distortion
3. Use asymmetric z-range per x-value

This is a design decision to make before the session. Option 1 (narrower z) is safest — the confound still exists in [−2, +2] but is zero by construction in any symmetric z-range.

---

## What v7 Should Resolve

| Question | v4–v6 Answer | Confound? | v7 Should Show |
|----------|-------------|-----------|----------------|
| cos(primal_z, primal_x) ≈ 0.91 | "Same direction" | **YES** — x leaked into z-bins | True cos after removing x-leak |
| PC1 ≈ primal_z ≈ meta_w1 | "Variance axis = z axis" | **YES** — PC1 blends x and z | Whether PC1 is truly z or x or a blend |
| primal_z steers 6× better than probe_z | "Unsupervised > supervised" | **MAYBE** — primal_z included x-effect | Whether gap persists with clean primal_z |
| cos(probe_z, probe_x) = 0.54 | "Moderate correlation" | **LESS** — regression partials it out | Should be similar (validation) |
| cos(w_z, w_ld) ≈ 0 at late layer | "Mystery" | **NO** — this is a probe finding | Should be similar (validation) |
| INLP barely reduces R²(z) | "z is distributed" | **MAYBE** — removed directions had x | Whether R²(z) drops faster with clean dirs |

---

## Priority Order + Time Budget

| # | Experiment | GPU? | Time | Blocking? |
|---|-----------|------|------|-----------|
| 1 | Extract Grid B activations + logits | GPU | 3 min | YES — everything else needs this |
| 2 | Clean primal directions + confound audit | CPU | 5 min | No |
| 3 | 7-direction steering on clean directions | GPU | 5 min | No |
| 4 | Clean transfer matrix + random null | GPU | 5 min | No |
| 5 | Park metric + Fisher on clean directions | GPU | 10 min | No |
| 6 | INLP on Grid B | CPU | 2 min | No |

**Total estimated GPU time:** ~25 min on H100
**Total estimated CPU time:** ~7 min

Much lighter than v6. The main cost is the extraction; analysis reuses existing scripts with minimal modification.

---

## Discussion: Do We Need Both Grids Going Forward?

The (x, μ) grid is needed for behavioral regression (the paper's hero result). The (x, z) grid is needed for clean activation geometry. Running both doubles the prompt count from 750 to 1500 per pair (12,000 total across 8 pairs), which is still cheap (~6 min on H100).

For the paper, we should:
1. Present behavioral results from Grid A (x, μ) — unchanged
2. Present all activation geometry from Grid B (x, z) — new, clean
3. Show the confound explicitly: cos(primal_z_gridA, primal_x_gridA) = 0.91 vs cos(primal_z_gridB, primal_x_gridB) = ??? — this IS a methodological contribution (warning to other mech-interp work that conditions on derived variables)

The confound disclosure itself could be a valuable part of the paper — it's a general pitfall for any study that uses grid designs with derived variables.
