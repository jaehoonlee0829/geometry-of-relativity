# geometry-of-relativity

Mechanistic interpretability study of **contextual relativity of gradable adjectives** in open-weight LLMs.

**Target venue:** ICML 2026 MI Workshop (May 8 AOE), co-submission to NeurIPS 2026 main track.

## TL;DR

We test whether linear probe covectors for **relative** gradable adjectives ("tall", "short", "rich", "poor") track a **context-dependent Z-score** of a numerical attribute, while probes for **absolute** adjectives ("obese", defined by BMI cutoff) track the **raw value**. The distinction is made rigorous via the Fisher-information pullback `F(h)^{-1} * w` of the probe covector -- our central formal claim is:

> An adjective `A` is *relative* iff `F_ctx(h)^{-1} * w_A` aligns with the gradient of the z-score, and *absolute* iff it aligns with the gradient of the raw value.

## Models

| Short name | HuggingFace ID | Role |
|---|---|---|
| Gemma 4 31B | `google/gemma-4-31B` | Primary (60 layers, d=5376) |
| Gemma 4 E4B | `google/gemma-4-E4B` | Secondary / scaling comparison (42 layers, d=2560) |

Legacy models (gemma-2-2b, Llama-3.2-3B) were used in v0/v1 behavioral experiments only; they are not part of the current activation-probing pipeline.

## Experiment history

| Version | Phase | What happened |
|---|---|---|
| **v0** | Behavioral kill-test | 20 hand-crafted prompts, Claude Opus 4.5 API. H1 (relative flip) passed; H2 (absolute stability) partially failed -- "obese" showed context sensitivity. |
| **v1** | Behavioral spectrum scan | 810 completions across 9 context means, Claude Sonnet 4.6. Confirmed "obese" is a ~1/3 absolute / ~2/3 relative hybrid. |
| **v2** | Prompt design + extraction | 448 systematic prompts (height + wealth), Gemma 4 E4B + 31B activations on Vast.ai H100. |
| **v3** | Extraction format fix | Refactored activation storage: W_U saved once per model (30x speedup). |
| **v4** | Dense extraction + auto-research | 3540 implicit trials, 8 adjective pairs, five analysis scripts (probes, PCA, steering, INLP). |
| **v5** | Red-team + critic consensus | 7 experiments + G31B replication + random-direction null + 3 skeptical-critic agents. Key finding: meta w1 steers all 8 pairs 3.1-28.5x above null. H4 (Fisher) not supported. |
| **v6** | Confound discovery | 7-direction analysis revealed Grid A (x, mu) design had corr(x,z) = 0.58-0.86. Direction-based analyses were contaminated. |
| **v7** | Clean-grid rerun | Grid B (x, z) extraction. Confound fixed. New findings: primal_z transfers at 40% across pairs (5.5x null), INLP works on clean grid, primal_z steers 18x stronger than probe_z. |

### v7 headline results (current)

**Strong findings (clean Grid B):**
- **Primal_z steering** — simple mean-difference direction steers logit_diff 13-18x more than Ridge probe direction. Encoding != use.
- **Cross-pair transfer** — primal_z from one pair steers another at 40% own-pair strength (5.5x random null). Body-attribute pairs form a tight cluster.
- **INLP works** — on clean grid, R²(z) drops 30-50% after 8 iterations (was <5% on confounded Grid A).
- **Zero-shot-corrected heatmaps** — subtracting zero-shot ld(x) isolates pure context effect: nearly flat across x, strong z-gradient.

**Known limitations:**
- H4 (Fisher pullback) refuted: F(h) near-isotropic at tested activations.
- Relative/absolute dichotomy not statistically significant (n=7 vs 4, p=0.75).
- PCA horseshoe + SVD scree need regeneration on Grid B (pending .npz fetch from HF).

## Five lines of evidence (v4 pipeline)

1. **Behavioral relativity ratio** -- regression of logit_diff on x and mu; relative pairs should show R ~ 1, absolute ~ 0.
2. **Probe decodability** -- ridge probes for x, mu, z, sign(z); CV R²(z) should dominate for relative adjectives.
3. **PCA geometry** -- PC1 of cell-mean activations should correlate with z (not x) for relative adjectives.
4. **Causal steering** -- adding alpha * w_z to residual stream should shift logit_diff monotonically.
5. **INLP concept erasure** -- iteratively projecting out w_z should collapse R²(z) while random projections do not.

## Repository layout

```
geometry-of-relativity/
  PLANNING.md          # Frozen project spec (do not modify)
  BUILDING.md          # Current active task
  TODO.md              # Rolling checklist
  STATUS.md            # High-level project status
  FINDINGS.md          # Detailed experimental findings (v4-v7)
  CLAUDE.md            # Instructions for Claude agents
  data_gen/            # Prompt JSONL files
  src/                 # Core library modules
  scripts/             # Runnable drivers
    plots_v7_behavioral.py     # Behavioral plots from v7 Grid B jsonl
    replot_v7_from_json.py     # Geometry plots from v7 pre-computed JSON
    plot_confound_matrix_gridB.py  # Grid B confound matrix (standalone)
    fetch_from_hf.py           # Fetch .npz/.jsonl from HF dataset
    vast_remote/               # GPU scripts (run on Vast.ai)
  results/             # Experimental outputs (large files gitignored)
    v7_xz_grid/        #   Grid B jsonl data (logits, trials)
    v7_analysis/        #   Pre-computed JSON (confound, INLP, Fisher)
    v7_steering/        #   Steering + transfer JSON
    csv/                #   Zero-shot expanded data
  figures/             # Publication-quality plots
    v7/                #   Geometry plots (Grid B)
    v7_behavioral/     #   Behavioral plots (Grid B)
    v5_gpu_session/    #   Legacy v5 plots
  docs/                # Design docs and paper outline
    NEXT_GPU_SESSION_v8.md  # Next GPU session plan
    paper_outline.md        # ICML MI Workshop paper skeleton
    archive/                # Historical session logs and PR descriptions
  tests/               # Unit + smoke tests (pytest)
```

## Quick start

```bash
pip install -e ".[dev]"       # CPU-only: prompts, probes, Fisher math, plots
pip install -e ".[dev,gpu]"   # Add GPU: model forward passes
pytest tests/ -v -m "not gpu"

# Regenerate all v7 plots (CPU only):
python scripts/plots_v7_behavioral.py
python scripts/replot_v7_from_json.py
```

## License

CC-BY-4.0 for the paper, MIT for the code.
