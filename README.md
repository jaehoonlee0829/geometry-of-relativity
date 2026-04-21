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
| **v1** | Behavioral spectrum scan | 810 completions across 9 context means, Claude Sonnet 4.6. Confirmed "obese" is a ~1/3 absolute / ~2/3 relative hybrid. See `FINDINGS_V1.md`. |
| **v2** | Prompt design + activation extraction | 448 systematic prompts (height + wealth domains), implicit/explicit contexts, two prompt frames. Gemma 4 E4B + 31B activations extracted on Vast.ai H100. |
| **v3** | Extraction format fix | Refactored activation storage: W_U saved once per model instead of per-layer (30x speedup). No new scientific content. |
| **v4** | Dense extraction + auto-research | 3540 implicit trials with 30 resampling seeds per (x, mu) cell. Five analysis scripts (probes, PCA, adjective-pair generalization, causal steering, INLP erasure) staged on branch `exp/v4-auto-research`. |
| **v5** | Red-team follow-up + critic consensus | 7 experiments from `docs/NEXT_GPU_SESSION.md` + G31B replication + random-direction null for meta-steering + 3 parallel skeptical-critic agents. Branch `exp/next-gpu-session`, see `docs/PR_next_gpu_session.md` and FINDINGS.md §6 for the honest summary. |

### v5 headline results (Apr 21 2026)

**What survived scrutiny** (still publishable):
- **Causal steering** — a single shared direction `w₁` (from SVD of stacked per-pair PC1s) shifts `logit_diff` monotonically for all 8 adjective pairs at layer 32. Slopes are **3.1–28.5× larger** than steering with random unit vectors in the same space. See `figures/v4_adjpairs/meta_w1_steering_curves.png` and `meta_w1_vs_random_null.png`.
- **Behavioral hero figure** — `figures/v4_adjpairs/logit_diff_heatmap_xmu_8panel.png` shows the anti-diagonal relativity pattern for tall/short, old/young, heavy/light, and the x-dominant vertical stripes for the absolute control `bmi_abs`.
- **Scaling evidence (G31B)** — the same 8-pair extraction on Gemma 4 31B replicates the effect; pairs become *more* relative at scale, not less.

**What got weaker under scrutiny** (paper framing should soften):
- **H4 (Fisher-pullback) not supported.** `cos_F⁻¹(w_adj, w_z)` ≈ `cos_Euclidean(w_adj, w_z)` within 0.01–0.04 (max cosine 0.30; H4 predicted > 0.7). F(h) is near-isotropic at the cell-mean activations we tested. See `figures/v4_adjpairs/fisher_vs_euclid_cosines.png`.
- **Relative-vs-absolute dichotomy not significant.** With n=4 absolute + n=7 relative pairs: Welch `t=-0.33, p=0.75`. Pairs like `legal_abs` (minor/adult, supposed to be anchored at 18) show strong context sensitivity (relativity_ratio = 0.89). The dichotomy may not be the right carving.
- **Zero-shot x-direction "orthogonal to z-direction" is null-consistent.** `|cos(w_x_zeroshot, w_z_implicit)| ≤ 0.08` across 8 pairs is only 2–4σ above the √(1/d) ≈ 0.02 chance floor — directional, not quantitative.
- **Meta-direction `w₁` steers bmi_abs (the absolute control) with the LARGEST slope.** So `w₁` is probably a generic "adjective polarity" knob, not specifically a relativity substrate. Framing should shift from "relativity direction" to "shared polarity direction".

Full writeup in `FINDINGS.md` §6 and `docs/PR_next_gpu_session.md`.

## Five lines of evidence (v4 pipeline)

1. **Behavioral relativity ratio** -- regression of logit_diff on x and mu; relative pairs should show R ~ 1, absolute ~ 0.
2. **Probe decodability** -- ridge probes for x, mu, z, sign(z); CV R^2(z) should dominate for relative adjectives.
3. **PCA geometry** -- PC1 of cell-mean activations should correlate with z (not x) for relative adjectives.
4. **Causal steering** -- adding alpha * w_z to residual stream should shift logit_diff monotonically.
5. **INLP concept erasure** -- iteratively projecting out w_z should collapse R^2(z) while random projections do not.

## Repository layout

```
geometry-of-relativity/
  PLANNING.md          # Frozen project spec (do not modify)
  BUILDING.md          # Current active task
  TODO.md              # Rolling checklist
  STATUS.md            # High-level project status
  FINDINGS_V1.md       # Key v1 behavioral findings
  CLAUDE.md            # Instructions for Claude agents working on this repo
  data_gen/            # Prompt JSONL files (v0, v1, v2)
  src/                 # Core library modules
    data_gen.py        #   Prompt generation (v2 implicit/explicit contexts)
    fisher.py          #   Fisher information matrix + pullback
    probe.py           #   Linear probe training (ridge, logistic)
    activation_extract.py  # HF forward-hook activation extractor
    plots.py           #   Visualization helpers
  scripts/             # Runnable drivers
    run_behavioral.py      # v0 Claude API behavioral runner
    run_behavioral_v1.py   # v1 spectrum scan runner
    gen_prompts_v1.py      # v1 prompt generator
    analyze_behavioral.py  # v0 result analysis
    analyze_v1.py          # v1 spectrum analysis + sigmoid fits
    pull_activations_from_wandb.py  # Download cached activations from W&B
    visualize_activations.py        # PCA + probe visualization (v2 data)
    vast_remote/           # Scripts designed to run on Vast.ai GPU instances
      extract_e4b_v3.py    #   Gemma 4 E4B activation extraction
      extract_g31b_v1.py   #   Gemma 4 31B activation extraction
      extract_v4_dense.py  #   v4 dense extraction (3540 trials)
      extract_v4_adjpairs.py   # 8 adjective-pair extraction (6240 prompts)
      analyze_v4.py            # Full probe/PCA/metric analysis
      analyze_v4_adjpairs.py   # Cross-pair relativity table
      steer_v4.py              # Causal steering test
      inlp_v4.py               # INLP concept erasure
  results/             # Experimental outputs (large files gitignored)
    behavioral_v1/     #   810 v1 completion JSONs
    behavioral_v0_summary.md
    behavioral_v1_summary.md
    behavioral_v1_per_mu.csv
  figures/             # Publication-quality plots
  tests/               # Unit + smoke tests (pytest)
  docs/                # Design docs and paper outline
    paper_outline.md       # ICML MI Workshop paper skeleton
    v4_research_design.md  # v4 evidence-line design document
    archive/               # Historical session logs and PR descriptions
```

## Quick start

```bash
pip install -e ".[dev]"       # CPU-only: prompts, probes, Fisher math, plots
pip install -e ".[dev,gpu]"   # Add GPU: model forward passes
pytest tests/ -v -m "not gpu"
```

## License

CC-BY-4.0 for the paper, MIT for the code.
