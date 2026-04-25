# BUILDING.md — What to run RIGHT NOW

## Active task — none

v11 cross-model dense + 5-critic round shipped (completion-promise word:
**RIPE-MANGO**). See FINDINGS §15 for the result writeup, §15.7 for the
retraction list, and §15.9 for the v11.5 follow-up scope.

## Next candidate tasks (pull from TODO.md when ready)

- **Paper draft (ICML MI Workshop, May 8 deadline — 13 days from
  2026-04-25).** Headline §15.1 (8/8-pair behavioral on both models) +
  §15.2 (9B uniformity) + §15.3 (W_U-orthogonality of primal_z). Retract
  §14.6's causal head taxonomy framing per §15.4 / §15.7. Use the 5
  critic reports under `results/v11/critic_*.md` as the basis for the
  Limitations section. **This is the highest-EV task pre-deadline.**

- **v11.5 follow-up — domain-agnostic shared z-direction** (arXiv v2
  scope per FINDINGS §15.9 A→D, post-submission). Constructs `w_shared`,
  steers all 8 pairs simultaneously with multi-seed + FDR control, and
  re-derives the attention-head taxonomy on the joint cross-pair signal.
  Defer until the workshop draft is done.

- **v11.5 pipeline fixes** (TODO §G/§H): fold-aware P3c orthogonalized
  R² and P3d ambiguous-cells fix-or-drop. Smaller scope; either before
  or after the paper draft depending on author preference.

- **v11 reproducibility close**: all 16 (pair × model) NPZs are on
  `xrong1729/mech-interp-relativity-activations` under
  `v11/<model_short>/<pair>/`. Verify with
  `python scripts/fetch_from_hf.py --only v11`.
