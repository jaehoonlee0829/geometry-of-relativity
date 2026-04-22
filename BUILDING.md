# BUILDING.md — What to run RIGHT NOW

## Active task — (empty)

v9 GPU session is complete (see FINDINGS.md §10, `results/v9_gemma2/`,
`figures/v9/`). No single task is blocking a next commit.

Next likely blocks of work, in priority order:

1. **Paper writing** — ICML MI Workshop (May 8 AOE), NeurIPS 2026
   (May 4 abstract / May 6 full). The workshop draft should lead with
   the behavioral R heatmap (v9.1), causal steering (v7 + v9.3/9.4),
   and §10.5's retirement of three alternative explanations.
2. **Optional v10 follow-up** (arXiv only) — characterize primal_z
   directly, given that it is NOT a sparse SAE bundle, NOT a local
   tangent, and NOT a W_U-rotated probe. Candidates: attention-weighted
   mean-diff, layer-aggregated direction, non-linear decoder.

Ralph loop note: there is no "completion promise word" queued for the
next iteration. When the next task is selected, add the token here so
the loop terminates correctly.
