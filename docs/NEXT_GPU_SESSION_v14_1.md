# Next GPU Session v14.1 - Robustness Cleanup

**Branch:** `experiment/v14-last-mile`

V14 is useful as a robustness study, but two parts need cleanup before we use
the results in the paper/readme:

1. affine/OOD low-world sampling currently rejects invalid negative values,
   which can remove or distort whole conditions;
2. the paper Figure 5 cleanup should preserve the original 2 x 3 layer-sweep
   format, not replace it with a new two-panel figure.

## What V14 Already Supports

Treat V14 as a robustness study:

- **Distribution shape robustness:** normal, uniform, skewed beta, U-shaped
  beta, and bimodal contexts mostly preserve the `LD`-vs-`z` relationship.
- **Order robustness with nuisance effects:** all orderings retain a positive
  `LD`-vs-`z` relationship, but sorted and near-target orders change the
  calibration. Random order is the cleanest default; alternating low/high is a
  strong contrast-balanced control.
- **Affine/OOD robustness in valid conflict cases:** where the data is valid,
  the model often still follows relative standing even when the absolute world
  is strange, e.g. very low absolute values can still be judged as "high" when
  they are high relative to an extremely low local context.

These are strong enough to keep as a robustness section, but the cleaned V14.1
rerun should be used for final figures.

## Experiment A - Clean Affine/OOD Rerun

### Problem In V14

V14 used normal context sampling and rejected rows when sampled values became
invalid:

```text
age around 3 years with sigma 5 years -> negative ages
speed around 5 km/h with sigma 15 km/h -> negative speeds
experience around 1 year with sigma 4 years -> negative years
```

This is not acceptable for final plots because rejection sampling changes the
intended context distribution or drops the condition entirely.

### Required Fix

Do not rely on "sample normal and reject invalid values" for low-world cases.
Use valid-support sampling by construction.

Recommended options:

```text
height, age, weight, size, speed, experience, bmi_abs:
  use low-world sigma small enough that normal draws remain positive
  OR use a truncated-positive distribution and report empirical z

wealth:
  keep log-space sampling; use log-income/log-wealth relative standing
```

For every condition, store both:

```text
population_z
empirical_z_from_rendered_context
valid_row_fraction
```

If a condition cannot be generated without changing the intended semantics,
mark it explicitly rather than silently dropping it.

### Conditions To Regenerate

Keep the V14 conditions but make low-world support valid:

```text
base
parallel_shift_high
world_extreme_high
world_extreme_low
target_extreme_high
target_extreme_low
```

The headline conflict cases are:

```text
world_extreme_low with z > 1:
  absolute value is low/weird, but target is high within the local context

world_extreme_high with z < -1:
  absolute value is high/weird, but target is low within the local context
```

### Required Plots

Do not use the old V14 affine/OOD heatmaps as interpretation figures. Keep raw
metrics if useful, but final interpretation should use:

```text
figures/v14_1/affine_ood/affine_ood_ld_by_z_lines.png
figures/v14_1/affine_ood/affine_ood_conflict_z_high.png
figures/v14_1/affine_ood/affine_ood_conflict_z_low.png
figures/v14_1/affine_ood/affine_ood_target_extremes_mean_ld.png
figures/v14_1/affine_ood/affine_ood_valid_row_fraction.png
```

Plot meanings:

- `LD by z` line plots: show whether each OOD condition still has a monotonic
  relative-standing response.
- conflict high plot: compare base/world-low/world-high for `z > 1`.
- conflict low plot: compare base/world-low/world-high for `z < -1`.
- target-extreme plot: use mean LD or LD shift, not `corr(LD,z)`, because
  target-extreme conditions have constant `z = +/-5`.
- valid-row plot: should be near 1.0 after the sampling fix.

## Experiment B - Order Local-Context Diagnostics

V14 showed order is a nuisance variable, but the first diagnostic does **not**
support a simple "the model only uses the last few examples" story.

Add these fields to each ordered row:

```text
z_full
z_first5
z_last3
z_last5
z_last10
z_last15
```

For log-space pairs, compute local z in log-space.

Required metrics:

```text
corr(LD, z_full)
corr(LD, z_first5)
corr(LD, z_last3)
corr(LD, z_last5)
corr(LD, z_last10)
corr(LD, z_last15)
corr(LD_order - LD_random, z_lastk - z_full)
```

Required plots:

```text
figures/v14_1/order/order_full_vs_local_z_corr.png
figures/v14_1/order/order_delta_ld_vs_local_z_gap.png
figures/v14_1/order/order_ld_by_z_lines.png
```

Expected interpretation to verify:

```text
full-context z should usually beat last-3/last-5 z.
last-15 may come close because it still covers much of the context.
order effects likely reflect calibration/attention weighting, not pure recency.
```

## Experiment C - Distribution Shape Story Plots

Distribution-shape results in V14 are already clean. Regenerate the same story
plots under `v14_1` for consistency:

```text
figures/v14_1/distribution/distribution_ld_by_z_lines.png
figures/v14_1/distribution/distribution_corr_range_by_pair.png
```

The main claim to preserve:

```text
The z relationship survives normal, uniform, skewed beta, U-shaped beta, and
bimodal contexts. This argues against the result being an artifact of Gaussian
prompt lists.
```

## Experiment D - Paper Figure 5 Correction

The paper Figure 5 should keep the original V12 2 x 3 layer-sweep format:

```text
figures/v12/layer_sweep_9b_combined.png
```

Do **not** replace it with a new two-panel figure for the paper.

Required change:

```text
Panel "Causal steering slope":
  replace probe_z with primal_x
  keep primal_z and random_null
```

If layerwise `primal_x` steering does not exist in current artifacts, rerun only
the missing layerwise steering artifact in the same style as V12:

```text
results/v14_1/fig5/layer_sweep_9b_steering_primal_x.json
figures/v14_1/fig5/layer_sweep_9b_combined_primal_x.png
```

Direction definitions:

```text
primal_z = E[h_L | z > 1] - E[h_L | z < -1]
primal_x = E[h_L | x >= pair 75th percentile] - E[h_L | x <= pair 25th percentile]
```

Use the same layer list, alpha, prompt set, aggregation style, and 2 x 3 layout
as the V12 figure unless there is a clear reason to change them.

## Minimal GPU Run Checklist

```bash
# 1. Clean affine/OOD rerun
python scripts/run_v14_1_gpu.py --sections affine_ood,plot \
  --pairs height age weight size speed wealth experience bmi_abs

# 2. Order local-context diagnostics
python scripts/run_v14_1_gpu.py --sections order,plot \
  --pairs height age weight size speed wealth experience bmi_abs

# 3. Figure 5 original-format correction
python scripts/run_v14_1_gpu.py --sections fig5_primal_x,plot
```

The script name is a placeholder; implement by either extending
`scripts/run_v14_gpu.py` safely or creating `scripts/run_v14_1_gpu.py`.
