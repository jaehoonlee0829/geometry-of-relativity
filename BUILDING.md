# BUILDING.md — What to run RIGHT NOW

## Active task — none

v11.5 v4–v9 question replication shipped (completion-promise word:
**SHARED-AMBER**). See FINDINGS §16 for the full writeup; STATUS.md
has the headline summary and retraction list.

## Next candidate tasks (pull from TODO.md when ready)

- **Paper draft (ICML MI Workshop, May 8 deadline — 12 days from
  2026-04-26).** Headline §16.1 (shared z-direction) + §16.2
  (FDR-controlled cross-pair transfer) + §15.3 + §16.6 (primal_z is
  W_U-orthogonal but decision-aligned) + §16.5 (z encoded at L1, then
  carried forward) + §16.7 (top SAE features are pure-z). Three-way
  refutation of §14.6 belongs in Limitations. Bootstrap CIs on every
  reported R² and cosine per §16.8. **Highest-EV task pre-deadline.**

- **arXiv v2 follow-ups** (post-May-7): pure-x control on §16.2's
  transfer matrix; 9B pure-z feature count asymmetry investigation;
  speed/experience pair-specific direction analysis (the two pair-
  specific exceptions to domain-generality).

- **v11 reproducibility close**: all 16 (pair × model) NPZs are on
  `xrong1729/mech-interp-relativity-activations` under
  `v11/<model_short>/<pair>/`. v11.5 outputs are JSON-only (small);
  no HF upload needed.
