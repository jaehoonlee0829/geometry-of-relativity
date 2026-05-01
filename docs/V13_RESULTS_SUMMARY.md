# V13 Results Summary

V13 ran the minimum viable GPU session from `docs/NEXT_GPU_SESSION_v13.md` on
Gemma 2 9B. Outputs are under `results/v13/` and `figures/v13/`.

## Scope

- Model: `google/gemma-2-9b`
- Main layers: L25 and L33, with steering summaries at L33
- Affine/OOD pairs: all eight V11/V12 pairs
- Cross-pair transfer matrix: all eight V11/V12 pairs
- New domains: brightness and temperature
- Objective controls: positive/negative and even/odd

## 1. Affine/OOD Relativity

The affine-shift tests separate questions that are easy to conflate:

- **Mild/moderate affine shift:** the target and its comparison group move
  together. Example: 185 cm among 170 cm people becomes 285 cm among 270 cm
  people. This is numerically strange, but relative standing is unchanged.
- **Target-only OOD:** the comparison group stays normal, but the target goes to
  an extreme value. Example: 300 cm among 170 cm people. This is not just an
  affine-invariance test; it is also a text-distribution/OOD plausibility test.
- **Extreme affine / whole-world OOD:** both target and context move together,
  but into an implausible textual regime. Example: 600 cm among 625 cm people.
  z can be negative, so a purely relative readout may still say "short", but
  this is entangled with the model seeing bizarre inputs.

The most human-readable plot for this is:

```text
figures/v13/affine_shift/affine_human_readable_summary.png
```

After rerunning affine/OOD on all eight adjective pairs, the behavioral LD
readout is mixed:

| pair | base | mild/moderate affine | target-only OOD | extreme affine / whole-world OOD |
| --- | ---: | ---: | ---: | ---: |
| height | +0.957 | +0.972 | +0.992 | +0.982 |
| age | +0.972 | +0.865 | +0.956 | +0.735 |
| weight | +0.972 | +0.907 | +0.977 | +0.968 |
| size | +0.945 | +0.875 | +0.939 | +0.330 |
| speed | +0.941 | +0.935 | +0.983 | +0.604 |
| wealth | +0.916 | +0.496 | -0.510 | +0.425 |
| experience | +0.950 | +0.916 | +0.974 | +0.281 |
| bmi_abs | +0.790 | +0.915 | +0.973 | +0.966 |

This is more critical than the original four-pair minimum run: height, weight,
and BMI are robust; age is moderately robust; size, speed, wealth, and
experience show serious degradation under at least one OOD/severity setting.
Wealth is the clearest failure under target-only OOD.

The base-trained L33 z probe transfers well through ordinary shifts/scales, but
world-OOD weakens speed and experience:

- speed world-OOD: base-probe corr with z_eff = +0.456.
- experience world-OOD: base-probe corr with z_eff = +0.338.

Steering with the base `primal_z` direction remains mostly positive across
tested conditions. This supports causal persistence of a direction, but the
degraded readout/probe correlations prevent a clean "fully affine-invariant"
claim. The strongest defensible statement is: some domains preserve relative
adjective behavior under affine shifts, but affine robustness is not universal.

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

## 4. Independent Relative Domains and Objective Controls

New independent relative adjective domains:

- brightness: corr(LD, z) = +0.911, corr(LD, x) = +0.022.
- temperature: corr(LD, z) = +0.619, corr(LD, x) = +0.693.

Objective controls are deliberately separate from continuous adjective domains.
They test whether the model preserves rule-like labels rather than converting
everything into relative standing:

- positive/negative: corr(LD, objective) = +0.896 vs corr(LD, z) = +0.536.
- even/odd: corr(LD, objective) = +0.538 vs corr(LD, z) = +0.274.

`even/odd` is categorical, not continuous. It should not be read as an
adjective-relativity test. It is a sanity/control task: if the model were
blindly using context-relative z for every numeric prompt, even/odd would show a
large z effect. Instead, the objective label correlation is larger than the z
correlation, though not perfect.

The objective-control design is:

| control | visible target values | context means | sigma | z definition | objective label | LD |
| --- | --- | --- | ---: | --- | --- | --- |
| positive/negative | signed numbers from -9 to +9 | -6, 0, +6 | 4.0 | `(x - mu) / 4` | sign(x) | `logit(positive) - logit(negative)` |
| pass/fail | scores from 40 to 100 | 50, 60, 70 | 10.0 | `(x - mu) / 10` | x >= 60 | `logit(pass) - logit(fail)` |
| even/odd | integers 11 to 29 after a +20 offset | 14, 20, 26 | 4.0 | `(x - (mu + 20)) / 4` | parity(x) | `logit(even) - logit(odd)` |

Use this plot for the intended reading:

```text
figures/v13/domain_extension/objective_control_interpretation.png
```

Because raw `corr(LD,z)` is a bad headline metric for categorical/rule tasks,
and because fever/normal and adult/minor were poor label choices, a second
objective-control pass uses cleaner single-token labels. It adds rule accuracy
and residual z-leakage after controlling for the objective label:

```text
results/v13/domain_extension/objective_control_v2_metrics.json
figures/v13/domain_extension/objective_control_v2_leakage.png
```

| control | kind | rule accuracy | corr(LD, objective) | raw corr(LD, z) | residual z leakage after objective | delta R2 from z |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| positive/negative | threshold | 1.000 | +0.907 | +0.852 | +0.779 | +0.108 |
| pass/fail | threshold | 0.692 | +0.781 | +0.699 | +0.297 | +0.034 |
| even/odd | categorical parity | 0.579 | +0.855 | +0.080 | +0.154 | +0.006 |

This makes the control story more conservative. Positive/negative and
pass/fail are threshold tasks whose objective labels are naturally coupled to
magnitude, so raw corr(LD,z) is expected to be large. The incremental z signal
after objective label is smaller by delta-R2, but not zero. Even/odd has high
objective correlation but poor zero-threshold accuracy, which means the
open-ended LD is not calibrated for parity; treat parity as an evaluation-format
warning rather than strong evidence.

New-domain cross-pair steering is positive for brightness/temperature own
directions and partially positive from existing primary domains. This is
encouraging for extension, but the temperature readout is still strongly tied
to raw `x`, so the domain-general claim remains mixed.

## Recommended Claim

V13 supports a mixed-positive update:

> Gemma 2 9B preserves a context-normalized relative-standing signal for several
> domains, and cross-pair steering is much more specific to `z` than to raw `x`.
> However, all-eight affine/OOD testing shows the robustness is domain-dependent:
> size, speed, wealth, and experience degrade under severe or target-only OOD
> settings, and temperature remains partly raw-magnitude-driven. The right claim
> is domain-dependent relative-adjective robustness, not a universal clean
> affine-invariant computation.
