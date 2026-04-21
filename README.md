# mech-interp-relativity

Mechanistic interpretability study of **contextual relativity of gradable adjectives** in small open-weight LLMs.

**Target venue:** ICML 2026 MI Workshop (May 8 AOE), co-submission to NeurIPS 2026 main track.

## TL;DR

We test whether linear probe covectors for **relative** gradable adjectives ("tall", "short") track a **context-dependent Z-score** of a numerical attribute, while probes for **absolute** adjectives ("obese", defined by BMI cutoff) track the **raw value**. The distinction is made rigorous via the Fisher-information pullback `F(h)⁻¹·w` of the probe covector — our central formal claim is:

> An adjective `A` is *relative* iff `F_ctx(h)⁻¹·w_A` aligns with `∇_h z_C(x)` (gradient of the z-score), and *absolute* iff it aligns with `∇_h x` (gradient of the raw value).

## Models

- `google/gemma-2-2b` (primary, early experiments)
- `google/gemma-4-e4b` (primary for v4 adjective-pair sweep, ≈8B params, residual width 2560)
- `meta-llama/Llama-3.2-3B` (replication — deferred)

## Hypotheses (H1–H4)

| ID | Hypothesis | Kill-test |
|----|------------|-----------|
| H1 | "tall"/"short" probes track Z-score, not raw cm | ≥2σ shift in probe logit between narrow-low (μ=150) and narrow-high (μ=180) contexts |
| H2 | "obese" probe tracks raw BMI, not Z-score | <0.5σ shift in probe logit across contexts at fixed BMI |
| H3 | OOD inputs (10 cm, BMI=3) project to low-Fisher-metric regions of the activation manifold | Fisher determinant at OOD activations ≤ 0.1× in-distribution median |
| H4 (theoretical) | Formal gradient-alignment claim above | Empirical: `cos(F⁻¹·w_A, ∇z_C) > 0.7` for relative, `cos(F⁻¹·w_A, ∇x) > 0.7` for absolute |

## Headline v4 result (see `FINDINGS.md`)

Extending the 2-pair design to 8 adjective pairs (7 relative + `bmi_abs` control) and PCA-ing the cell-mean activations at layer 32 of Gemma 4 E4B:

- **PC1 tracks z for every relative pair** (R² = 0.76–0.93), PC2 traces the classic PCA horseshoe (quadratic in PC1 coordinate).
- **Cross-pair z-axes are non-orthogonal but not identical** (mean off-diagonal |cos(PC1_i, PC1_j)| = 0.32 — ~3.3× above chance in 2560-d).
- **A single shared "meta" z-direction `w₁`** (top right singular vector of the stacked 8×2560 PC1 matrix) captures **41.6%** of cross-concept PC1 variance and predicts z with R² ≥ 0.76 for *every* concept, matching or beating each concept's own PC1 in 5 of 8 cases.

Working interpretation: Gemma 4 E4B has internalized a *partially shared, partially domain-specific* "extremity-vs-reference" substrate rather than instantiating separate z-machinery per semantic domain. Causal confirmation (steering ±α·w₁ vs random-direction baseline) is the next planned experiment.

## Day-by-day status

See `TODO.md` for the active checklist, `PLANNING.md` for the frozen project spec, `STATUS.md` for the running log, and `FINDINGS.md` for the consolidated v4 writeup.

## Layout

```
mech-interp-relativity/
  data_gen/          # prompt templates and generators
  src/               # core experiment code
  scripts/           # runnable drivers (behavioral, extract, probe, fisher, plot)
    vast_remote/     # Vast.ai-side extraction/analysis scripts (v4 pipeline)
  results/           # raw outputs (gitignored except summary files)
    v4_adjpairs/             # extracted activations for 8-pair sweep
    v4_adjpairs_analysis/    # per-pair PCA, cross-pair cos, meta z-direction
  figures/           # publication-quality plots
  notebooks/         # exploration and sanity checks
  paper/             # LaTeX source
  tests/             # unit tests for Fisher, probe, data_gen
  FINDINGS.md        # consolidated v4 findings (meta z-direction, caveats)
  STATUS.md          # running per-day log
  PLANNING.md        # frozen project spec
  TODO.md            # active checklist
```

## License

CC-BY-4.0 for the paper, MIT for the code.
