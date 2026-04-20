# BUILDING.md — What to run RIGHT NOW

Only one task in this file at a time. When done, move it to TODO.md "done" section
and pull the next one in.

## Active task (Day 4, Apr 21 2026) — v4 AUTO-RESEARCH

**Four scripts are staged on branch `exp/v4-auto-research` (NOT merged to main).
Pull and run on Vast to generate the full v4 research artifact set.**

### Why

v4_dense extraction (GARNET-ANVIL) finished yesterday: 3,500 implicit trials +
35 explicit + 5 zero-shot, with activations and logit_diff. Behavioral signal
is strong (~6 logit dynamic range across z). Now we need to:

1. Properly analyze what we have (probes, PCA, variance decomposition)
2. Test if the relativity pattern generalizes beyond tall/short to 7 more
   gradable-adjective pairs — plus 1 absolute-adjective control (BMI/obese)
3. Test if the z-direction is *causal* via activation steering

Per user directive (Apr 20): "do all the research on the cloud before you
push to main or make a PR." Branch is ready; no main commits until results.

### What the four scripts do

- `scripts/vast_remote/analyze_v4.py` — full probe/PCA/metric analysis
  on existing v4_dense data. Produces `results/v4_analysis/summary.json`
  + probe .npz files + figures.
- `scripts/vast_remote/extract_v4_adjpairs.py` — extracts activations &
  logit_diff for 8 adjective pairs (7 relative + 1 absolute-control). 6,240
  prompts total, ~2 min on H100.
- `scripts/vast_remote/analyze_v4_adjpairs.py` — cross-pair relativity table.
  Core claim: relative pairs → relativity_ratio ≈ 1; absolute pair (BMI/obese)
  → relativity_ratio ≈ 0.
- `scripts/vast_remote/steer_v4.py` — causal test. Adds α·ŵ_z to the
  residual at a chosen layer, measures logit_diff response curve. Needs
  analyze_v4.py to have run first (consumes its probe .npz output).

### How to run on Vast

```bash
cd /workspace/repo
git fetch origin
git checkout exp/v4-auto-research    # or merge into main first if preferred
git pull
python scripts/vast_remote/analyze_v4.py            # ~1-2 min
python scripts/vast_remote/extract_v4_adjpairs.py   # ~2 min (model forward)
python scripts/vast_remote/analyze_v4_adjpairs.py   # seconds
python scripts/vast_remote/steer_v4.py --layer late # ~1 min
python scripts/vast_remote/steer_v4.py --layer mid  # ~1 min
```

Total ~6 min wall time.

### Definition of done

- `results/v4_analysis/summary.json` populated; 3 probe R² values look
  consistent with smoke-test synthetic ranges (R²(z) > R²(x) for late layer)
- `results/v4_adjpairs_analysis/summary.json` has per-pair relativity_ratio
- `results/v4_adjpairs_analysis/figures/relativity_across_pairs.png` rendered
- `results/v4_steering/steering_late.json` shows monotone curve
  (slope of logit_diff vs α is non-zero, same sign as positive)
- A decision made: does the relativity pattern generalize?

### Completion promise word

OBSIDIAN-LATTICE

### Scientific pre-commit (what I'm watching for)

- Strong (>0.7) Σ⁻¹ cos(w_adj, w_z) — validates Fisher-Rao framing
- Relativity ratio distribution: 7/7 relative pairs near 1.0, BMI near 0.0
- Steering slope > 0.5 per α-unit — clear causal signal

If ANY of these fail, we have an interesting finding and a harder paper
to write. Either way, we learn something.
