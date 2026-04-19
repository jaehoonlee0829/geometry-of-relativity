# mech-interp-relativity

Mechanistic interpretability study of **contextual relativity of gradable adjectives** in small open-weight LLMs.

**Target venue:** ICML 2026 MI Workshop (May 8 AOE), co-submission to NeurIPS 2026 main track.

## TL;DR

We test whether linear probe covectors for **relative** gradable adjectives ("tall", "short") track a **context-dependent Z-score** of a numerical attribute, while probes for **absolute** adjectives ("obese", defined by BMI cutoff) track the **raw value**. The distinction is made rigorous via the Fisher-information pullback `F(h)⁻¹·w` of the probe covector — our central formal claim is:

> An adjective `A` is *relative* iff `F_ctx(h)⁻¹·w_A` aligns with `∇_h z_C(x)` (gradient of the z-score), and *absolute* iff it aligns with `∇_h x` (gradient of the raw value).

## Models

- `google/gemma-2-2b` (primary)
- `meta-llama/Llama-3.2-3B` (replication)

## Hypotheses (H1–H4)

| ID | Hypothesis | Kill-test |
|----|------------|-----------|
| H1 | "tall"/"short" probes track Z-score, not raw cm | ≥2σ shift in probe logit between narrow-low (μ=150) and narrow-high (μ=180) contexts |
| H2 | "obese" probe tracks raw BMI, not Z-score | <0.5σ shift in probe logit across contexts at fixed BMI |
| H3 | OOD inputs (10 cm, BMI=3) project to low-Fisher-metric regions of the activation manifold | Fisher determinant at OOD activations ≤ 0.1× in-distribution median |
| H4 (theoretical) | Formal gradient-alignment claim above | Empirical: `cos(F⁻¹·w_A, ∇z_C) > 0.7` for relative, `cos(F⁻¹·w_A, ∇x) > 0.7` for absolute |

## Day-by-day status

See `TODO.md` for the active checklist and `PLANNING.md` for the frozen project spec.

## Layout

```
mech-interp-relativity/
  data_gen/          # prompt templates and generators
  src/               # core experiment code
  scripts/           # runnable drivers (behavioral, extract, probe, fisher, plot)
  results/           # raw outputs (gitignored except summary files)
  figures/           # publication-quality plots
  notebooks/         # exploration and sanity checks
  paper/             # LaTeX source
  tests/             # unit tests for Fisher, probe, data_gen
```

## License

CC-BY-4.0 for the paper, MIT for the code.
