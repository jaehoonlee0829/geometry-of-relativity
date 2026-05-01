# V13 Results Summary

V13 ran the minimum viable GPU session from `docs/NEXT_GPU_SESSION_v13.md` on
Gemma 2 9B. Outputs are under `results/v13/` and `figures/v13/`.

## Scope

- Model: `google/gemma-2-9b`
- Main layers: L25 and L33, with steering summaries at L33
- Primary affine/OOD pairs: height, weight, speed, experience
- Cross-pair transfer matrix: all eight V11/V12 pairs
- New domains: brightness and temperature
- Objective controls: positive/negative and even/odd

## 1. Affine/OOD Relativity

The affine-shift tests separate two different questions that are easy to
conflate:

- **Same-z shifted/scaled world:** the target and its comparison group move
  together. Example: 185 cm among 170 cm people becomes 285 cm among 270 cm
  people. This is numerically strange, but the relative standing is unchanged.
- **Target-only OOD:** the comparison group stays normal, but the target goes to
  an extreme value. Example: 300 cm among 170 cm people. This is not just an
  affine-invariance test; it is also a text-distribution/OOD plausibility test.
- **Whole-world OOD:** both target and context are implausible. Example: 300 cm
  among 285 cm people. z can be ordinary, but the entire textual world is
  off-distribution.

The most human-readable plot for this is:

```text
figures/v13/affine_shift/affine_human_readable_summary.png
```

At the behavioral LD readout, most affine-shifted conditions continue to track
`z_eff` strongly:

- height: corr(LD, z_eff) stays between +0.957 and +0.992 across all tested
  affine/OOD conditions.
- weight: mostly strong, but negative shift drops to +0.754.
- speed: mostly strong, but world-OOD drops to +0.604.
- experience: mostly strong for affine transforms, but world-OOD drops to
  +0.281.

The base-trained L33 z probe transfers well through ordinary shifts/scales, but
world-OOD weakens speed and experience:

- speed world-OOD: base-probe corr with z_eff = +0.456.
- experience world-OOD: base-probe corr with z_eff = +0.338.

Steering with the base `primal_z` direction remains positive across all tested
conditions, including world-OOD. This supports causal persistence of the
direction, but the degraded world-OOD probe/readout correlations prevent a clean
"fully affine-invariant" claim.

## 2. X-transfer Control

V13 reproduces the 8x8 cross-pair transfer comparison using `primal_z`,
naive `primal_x`, and z-residualized `primal_x` directions.

Mean transfer slopes at L33:

| direction | diagonal | off-diagonal | offdiag positive |
| --- | ---: | ---: | ---: |
| `primal_z` | +0.065 | +0.026 | 1.00 |
| `primal_x_naive` | +0.022 | +0.006 | 0.75 |
| `primal_x_resid_z` | +0.015 | +0.004 | 0.70 |

Paired off-diagonal differences:

- `z - x_naive`: mean +0.020, CI95 [+0.017, +0.023], positive in 54/56 cells.
- `z - x_resid_z`: mean +0.022, CI95 [+0.019, +0.025], positive in 54/56 cells.

This strengthens the claim that shared cross-pair steering is specific to
relative standing more than to raw magnitude.

For direct visual comparison, use:

```text
figures/v13/x_transfer/cross_pair_transfer_z_x_side_by_side_gemma2-9b.png
```

## 3. Top-logit Readout

Top-k logits were saved for affine/OOD conditions in
`results/v13/top_logits/top_logits_by_condition.jsonl`, with semantic group
scores in `top_logit_group_scores.json`.

Use these plots as diagnostic checks rather than standalone claims:

- `figures/v13/top_logits/top_token_trajectories_by_z.png`
- `figures/v13/top_logits/semantic_mass_by_z.png`
- `figures/v13/top_logits/classic_ld_vs_group_ld.png`
- `figures/v13/top_logits/ood_top_tokens_examples.png`

The current grouping is intentionally simple and top-k-limited; it is useful
for catching OOD token drift but should not be treated as a complete semantic
readout.

## 4. Domain Extension and Objective Controls

New relative domains:

- brightness: corr(LD, z) = +0.911, corr(LD, x) = +0.022.
- temperature: corr(LD, z) = +0.619, corr(LD, x) = +0.693.

Objective controls:

- positive/negative: corr(LD, objective) = +0.896 vs corr(LD, z) = +0.536.
- even/odd: corr(LD, objective) = +0.538 vs corr(LD, z) = +0.274.

New-domain cross-pair steering is positive for brightness/temperature own
directions and partially positive from existing primary domains. This is
encouraging for extension, but the temperature readout is still strongly tied
to raw `x`, so the domain-general claim remains mixed.

## Recommended Claim

V13 supports a mixed-positive update:

> Gemma 2 9B preserves a context-normalized relative-standing signal through
> many ordinary affine shifts and scales, and cross-pair steering is much more
> specific to `z` than to raw `x`. However, world-OOD prompts degrade the
> readout/probe geometry for speed and especially experience, and new-domain
> temperature behavior remains partly raw-magnitude-driven. The right claim is
> robust-but-not-fully-affine-invariant relativity, not a universal clean
> affine-invariant computation.
