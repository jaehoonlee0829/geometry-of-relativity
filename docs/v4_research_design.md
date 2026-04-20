# v4 Auto-Research Design — April 20, 2026

Drafted while the user is out. This doc explains what the four new scripts do,
what they test, and what would falsify each claim. Scripts are ready to run on
the Vast H100 remote; NONE of this has been pushed to `main` yet.

## Paper claim (PLANNING.md, reiterated)

> Relative gradable adjectives ("tall", "rich", "heavy") in LLMs are encoded
> along a context-**normalized** direction (≈ a z-score), not along the raw
> physical quantity. Absolute gradable adjectives ("obese" via BMI threshold,
> "freezing" via 0 °C) should NOT show this context-normalization.

Five lines of evidence are now staged.

## Evidence 1 — Behavioral relativity ratio (extract + analyze v4_dense + adjpairs)

**Question:** does the model's preference for high-word over low-word depend on
z = (x − μ)/σ alone, or also on raw x?

**Test:** fit `logit_diff = a + b·x + c·μ + ε`. Relativity ratio is
`R = −c/b`.  If relative adjectives are truly context-normalized,
then R → 1 (μ shifts the threshold 1-for-1 with x).  If the model ignores
context, c ≈ 0 and R → 0.

**Prediction:**
- height/tall, age/old, weight/heavy, size/big, speed/fast, wealth/rich,
  experience/expert:  R ≈ 1
- bmi/obese (absolute control):  R ≈ 0

**What would falsify the paper:** relative pairs all show R ≈ 0, OR
absolute pair shows R ≈ 1.

**Scripts:** `extract_v4_adjpairs.py` → `analyze_v4_adjpairs.py`

## Evidence 2 — Probe decodability (analyze_v4.py, phase 2)

**Question:** at the representation level, which latent is linearly decodable
— x, μ, z, or sign(z)?

**Test:** train four ridge probes on 3,500 implicit activations; report 5-fold
CV R².

**Prediction:** for relative adjectives at a mid-to-late layer,
CV R²(z) ≫ CV R²(x). For absolute adjectives we have no activation data yet
(extraction will produce it), but the analogous probe run on bmi_abs
activations should show CV R²(x) ≳ CV R²(z).

**What would falsify:** CV R²(z) ≤ CV R²(x) for relative pairs.

## Evidence 3 — Causal steering (steer_v4.py)

**Question:** is w_z **used** by downstream layers, or is it a
passenger that happens to correlate with z?

**Test:** hook the residual stream at layer L, add α·ŵ_z to the last-token
activation, re-run. Measure logit_diff vs α. Do this on held-out
(explicit + zero-shot) prompts the probes never saw.

**Prediction:** if w_z is mechanistically downstream of tall/short choice,
we expect a near-linear monotone response of logit_diff vs α, with slope of
order the intrinsic z→logit_diff slope (~1.2 from the implicit regression).
If w_z is purely correlational, the curve will be flat or noisy.

**What would falsify:** flat or non-monotone steering curve.

**Caveat:** this tests the *sum* w_z contribution — if multiple directions
encode z with cancellation, a single-direction steer can be weaker than
predicted. A negative steering result is not conclusive, but a strong positive
one IS strong evidence.

## Evidence 4 — Geometric structure (analyze_v4.py, phase 3)

**Question:** does the activation manifold of cell means have an
interpretable low-dimensional structure in which z is an axis?

**Test:** PCA on the 35 cell-mean activations (5 x × 7 μ). Correlate each PC
with z, x, μ, and the behavioral logit_diff.

**Prediction:** PC1 correlates ≳ 0.8 with either z or logit_diff. The
direction in activation space we call "∂h/∂z" (coefficient of z in linear
regression of h on z, x) has a non-trivial cosine with w_z from phase 2.

**What would falsify:** PCA shows no coherent low-D structure, or PCs align
primarily with noise / μ alone.

## Evidence 5 — INLP concept erasure (inlp_v4.py)

**Question:** is w_z *the* direction that encodes z, or merely one direction
that happens to correlate with z? A linear probe on high-dimensional activations
can find a correlated readout even when many directions individually contribute
to a given feature.

**Test:** Ravfogel et al. 2020's iterative nullspace projection. At each step:
(1) fit a ridge probe for z on current activations H, obtain unit direction v,
(2) project out: H ← H(I − vvᵀ), (3) retrain; measure CV R²(z), R²(x),
R²(logit_diff) after each step. Run three schedules on the same data:
- **INLP-z** — the real thing.
- **random null** — at each step, project out a random unit vector.
- **INLP-x** — iteratively null out w_x, as an interference/competing-direction
  control.

**Prediction:**
- CV R²(z) under INLP-z collapses within 1–3 steps (≥ 0.5 drop).
- CV R²(z) under random-null stays near its initial value (trivial erasure
  along an irrelevant direction).
- CV R²(x) under INLP-z stays substantially higher than R²(z) under INLP-z
  (x is distinguishable from z, so nulling z shouldn't kill x).
- R²(logit_diff) under INLP-z drops meaningfully — the model's behavior
  also leaves through w_z, not just a passenger feature.

**What would falsify:** random-projection baseline collapses R²(z) at the
same rate as INLP-z (→ w_z isn't specifically "the" z direction), or
R²(logit_diff) survives INLP-z intact (→ the behavior routes around w_z).

**Smoke-test evidence already in hand** (`tests/test_inlp_smoke.py`):
on synthetic v4-shaped data with known true z-direction, INLP-z drops
CV R²(z) from 0.991 → 0.316 in 4 steps while random projection preserves
it at 0.991 (gap +0.896), and R²(x) under INLP-z only drops from 0.644 to
0.210 (x signal is partly preserved, as expected since x ≠ z).

## Why the "primal–dual mismatch" matters

From v2 probe analysis (commit 8bc41ed notes): w_adj and w_z predict equally
well individually, yet cos(w_adj, w_z) ≈ 0 in Euclidean geometry. Phase 5 of
analyze_v4.py computes Σ⁻¹ cosines — if the cosine rises substantially
under the activation-covariance metric, that's evidence that both probes
read the same subspace but their Euclidean direction is dominated by
isotropic noise directions that don't matter for the readout.

This distinction is central to the paper's "Fisher-Rao pullback" framing.
If Σ⁻¹ cos(w_adj, w_z) is close to 1 while Euclidean is near 0, we have
quantitative support for that framing. If Σ⁻¹ cos stays low too, then
different probes are truly picking different mechanisms.

## Execution plan (to run on Vast when user returns)

On the Vast H100 instance in /workspace/repo:

```
git pull origin exp/v4-auto-research        # pulls these new scripts
python scripts/vast_remote/analyze_v4.py    # ~1-2 min on existing v4 data
python scripts/vast_remote/extract_v4_adjpairs.py  # ~2 min, 6240 prompts
python scripts/vast_remote/analyze_v4_adjpairs.py  # seconds
python scripts/vast_remote/steer_v4.py --layer late  # ~1 min
python scripts/vast_remote/steer_v4.py --layer mid   # ~1 min
python scripts/vast_remote/inlp_v4.py --layer late --steps 8  # ~1 min
python scripts/vast_remote/inlp_v4.py --layer mid  --steps 8  # ~1 min
```

Total wall time: ~8 min. Outputs land under `results/v4_analysis/` (including
`inlp_{mid,late}.json` + figures), `results/v4_adjpairs/`,
`results/v4_adjpairs_analysis/`, `results/v4_steering/`.

## Status at time of writing

- All four scripts parse cleanly.
- `analyze_v4.py` passes a synthetic-data smoke test that checks the full
  phase-1-through-5 pipeline (`tests/test_analyze_v4_smoke.py`).
- Prompt-generation dry run for adjpairs confirms all 8 pairs render cleanly
  (6,240 prompts, including absolute-adj BMI control).
- No scripts have been executed against a real model yet — the user is away.

## Post-execution decisions (user choice)

Based on results:
- **R ≈ 1 for relative pairs, R ≈ 0 for BMI:** we have the paper's flagship
  plot. Write it up for ICML MI Workshop.
- **R varies widely:** the "relativity" framing needs refinement. Look at
  which pairs break the pattern and why (e.g., is it tokenization?
  training data overlap? cultural anchoring?).
- **Steering works:** add a § on "causal readout" to the paper, pointing to
  a steering-based definition of the linear representation.
- **Steering fails:** the paper has to rely on probe + behavioral evidence;
  acknowledge the correlation/causation gap honestly.
