# TODO.md — Rolling checklist (Apr 21 2026)

## Done

- [x] Behavioral kill-tests v0 (Claude Opus 4.5, 100 completions) and v1 (Sonnet 4.6, 810 completions)
- [x] Scaffold `src/` modules: data_gen, fisher, probe, activation_extract, plots
- [x] v2 prompt generator with implicit/explicit contexts, height + wealth domains
- [x] Gemma 4 activation extraction v2: E4B + 31B, 448 prompts, 4 layers each
- [x] Probe analysis on v2 activations — found: 63 points per condition is too few, weight-vector cosines misleading, need replicates and direct logit extraction
- [x] **v4 dense extraction** (GARNET-ANVIL): 3540 prompts, activations + logit_diff

## Active (see BUILDING.md)

- [ ] **OBSIDIAN-LATTICE — v4 auto-research suite**: analyze_v4, extract+analyze adjpairs,
  steer_v4, **inlp_v4** (5 scripts now). Branch `exp/v4-auto-research`. Ready to run;
  all 10 local smoke tests pass. Branch is NOT yet pushed to origin — do `git push
  origin exp/v4-auto-research` from your workstation before pulling on Vast.

## Queue — after OBSIDIAN-LATTICE

- [ ] Decide based on results which direction to push:
  - If relativity generalizes and is causal → start writing ICML MI Workshop paper
  - If it partially generalizes → investigate which pairs break & why
  - If steering fails → probe-only + behavioral story for the paper
- [ ] Fill the `<<TBD>>` slots in `docs/paper_outline.md` from Vast results
- [ ] Run the full suite on G31B once E4B result set is in (1 H100 can host 31B
  with activation extraction at batch-size 4)
- [x] ~~INLP concept erasure — after probe weights land, iterate project-out
  and measure downstream logit_diff degradation~~ — written as `inlp_v4.py`,
  tested, staged on branch. Just needs to run on real v4_dense data.
- [ ] Additional absolute-adj controls beyond BMI: "freezing"/0°C, "legal age"/18

## Queue — Paper (May 3–8)

- [ ] Complete paper draft
- [ ] Submit ICML 2026 MI Workshop (May 8) — primary target
- [ ] Submit NeurIPS 2026 (May 4 abstract, May 6 full) — secondary

## Backlog

- [ ] Replace PCA with a proper sparse factor model on 3500 activations
- [ ] Layer sweep (early, mid, late, final) once we know which layers matter
- [ ] Try different context sample sizes (n=5, 15, 50 people) — saturation curve
- [ ] Multilingual: does the Spanish "alto"/"bajo" show the same relativity?

## Waiting on user

- [ ] Destroy Vast instance m:56779 after OBSIDIAN-LATTICE runs complete
- [ ] Decide whether to merge `exp/v4-auto-research` into main or keep as review PR
- [ ] Write local SSH config for vastai host (minor convenience)
