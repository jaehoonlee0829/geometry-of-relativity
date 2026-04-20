# TODO.md — Rolling checklist (Apr 21 2026)

## Done

- [x] Behavioral kill-tests v0 (Claude Opus 4.5, 100 completions) and v1 (Sonnet 4.6, 810 completions)
- [x] Scaffold `src/` modules: data_gen, fisher, probe, activation_extract, plots
- [x] v2 prompt generator with implicit/explicit contexts, height + wealth domains
- [x] Gemma 4 activation extraction v2: E4B + 31B, 448 prompts, 4 layers each
- [x] Probe analysis on v2 activations — found: 63 points per condition is too few, weight-vector cosines misleading, need replicates and direct logit extraction

## Active (BUILDING.md has the details)

- [ ] **v4 dense extraction** (GARNET-ANVIL): 100 seeds per (x,μ) cell, three conditions (implicit/explicit/zero-shot), activations + logit("tall")-logit("short") + top-5 tokens

## Queue — after v4 extraction

- [ ] Analyze v4 logit diffs: does mean logit_diff track z across (x,μ) grid?
- [ ] PCA on averaged activations (100 seeds per cell → 35 mean vectors) — manifold viz
- [ ] Train probes on v4 data with proper replicates (N=3500 implicit trials)
- [ ] Zero-shot vs implicit vs explicit comparison — does context actually change anything?
- [ ] Compute Σ⁻¹ and F⁻¹ cosines with enough data to be meaningful

## Queue — Paper (May 3–8)

- [ ] Complete paper draft
- [ ] Submit NeurIPS 2026 (May 4 abstract, May 6 full)
- [ ] Submit ICML 2026 MI Workshop (May 8)

## Backlog

- [ ] Wealth domain (rich/poor) — same v4 treatment
- [ ] 31B extraction with v4 dense design
- [ ] BMI / obese revisit
- [ ] Steering interventions
