# Next GPU Session v14 - Last-Mile Robustness Controls

**Branch:** `experiment/v14-last-mile`

## Goal

V14 runs last-mile controls for the paper claims. It is not meant to discover a
new central mechanism. It tests whether the main `z` story survives obvious
methodological pressure:

1. Does the result depend on the context values being sampled from a normal
   distribution?
2. Does the result depend on the order of the reference values in the prompt?
3. Do OOD/affine shifts fail because the whole comparison class is strange, or
   because the target alone is implausible?
4. Can the paper's layer/steering figure be made clearer by comparing `z`
   against raw `x` rather than using underdefined `probe_z` language?

Primary model:

```text
google/gemma-2-9b
```

Primary pairs:

```text
height, age, weight, size, speed, wealth, experience, bmi_abs
```

Primary layers:

```text
L25, L33
```

Use outputs under:

```text
results/v14/
figures/v14/
```

## Experiment A - Non-Normal Context Distributions

### Question

The original dense grids draw context examples from approximately normal
distributions. Does the model actually use a normalized relative standing, or
does it rely on heuristics that break under flat, skewed, or multimodal
comparison classes?

### Design

For each pair, sample many `(x,z)` cells. For each cell, derive the population
mean:

```math
\mu = x - z\sigma
```

Then render a long context list, default `N=31`, using different context
distribution shapes while preserving the intended population mean/spread as
closely as possible.

Distribution conditions:

```text
normal
uniform
beta_u              U-shaped beta, many low and high values
beta_low            skewed low
beta_high           skewed high
bimodal             two separated clusters
```

Primary label, using the pair-specific definition:

```text
population z = (x - mu) / sigma
wealth population z = (log x - log mu) / log(sigma_factor)
```

Also log:

```text
empirical z = (x - sample_mean(context)) / sample_std(context)
wealth empirical z = (log x - mean(log context)) / std(log context)
rank = fraction of context values below target x
```

### Prompt Examples

Normal height, target above norm:

```text
Person 1: 158 cm
Person 2: 161 cm
...
Person 31: 181 cm
Person 32: 180 cm. This person is
```

Uniform height, same intended `x`, `mu`, `sigma`:

```text
Person 1: 153 cm
Person 2: 156 cm
...
Person 31: 187 cm
Person 32: 180 cm. This person is
```

Bimodal height:

```text
Person 1: 149 cm
Person 2: 153 cm
...
Person 16: 185 cm
Person 17: 188 cm
...
Person 31: 192 cm
Person 32: 180 cm. This person is
```

### Metrics

For each pair and distribution condition:

```text
corr(LD, population_z)
corr(LD, empirical_z)
corr(LD, target_rank)
corr(LD, x)
mean LD shift relative to normal at matched cells
cos(primal_z_normal, primal_z_condition) at L33
normal-trained z probe tested on each condition
```

### Plots

```text
figures/v14/distribution/distribution_shape_examples.png
figures/v14/distribution/distribution_corr_bars.png
figures/v14/distribution/distribution_rank_vs_z.png
figures/v14/distribution/distribution_primal_alignment.png
```

## Experiment B - Reference Order Sensitivity

### Question

If the context values define a set, order should not matter much. If the model
uses recency, trends, or local pairwise contrasts, the same values in different
orders may change the readout.

### Design

Use the exact same multiset of context values and the same target. Only reorder
the reference examples. The target remains last in all conditions.

Order conditions:

```text
random
ascending
descending
alternating_low_high
near_target_first
near_target_last
```

### Prompt Examples

Ascending:

```text
Person 1: 140 cm
Person 2: 150 cm
Person 3: 160 cm
Person 4: 170 cm
Person 5: 180 cm
Person 6: 190 cm
Person 7: 175 cm. This person is
```

Descending:

```text
Person 1: 190 cm
Person 2: 180 cm
Person 3: 170 cm
Person 4: 160 cm
Person 5: 150 cm
Person 6: 140 cm
Person 7: 175 cm. This person is
```

Alternating low/high:

```text
Person 1: 140 cm
Person 2: 190 cm
Person 3: 150 cm
Person 4: 180 cm
Person 5: 160 cm
Person 6: 170 cm
Person 7: 175 cm. This person is
```

### Metrics

```text
corr(LD,z) by order condition
mean absolute LD shift from random order at matched cells
cos(primal_z_random, primal_z_order)
normal/random-order trained z probe tested on each order
```

### Plots

```text
figures/v14/order/order_corr_bars.png
figures/v14/order/order_ld_shift.png
figures/v14/order/order_primal_alignment.png
```

## Experiment C - Redesigned Affine/OOD Controls

### Question

V13 showed mixed OOD behavior, but several conditions were entangled. V14
separates:

- moving the target and context together;
- making only the target extreme;
- moving the whole world into tiny or giant regimes;
- increasing the number of context examples.

### Conditions

Run all eight pairs, with context lengths:

```text
N = 5, 15, 31
```

Conditions:

```text
base
parallel_shift_high       x and mu shifted together; z unchanged
world_extreme_high        context and target both in high/OOD regime; z unchanged
world_extreme_low         context and target both in low/tiny regime when valid; z unchanged
target_extreme_high       normal context, target pushed to very high z
target_extreme_low        normal context, target pushed to very low z when valid
```

The target-only extreme conditions have one-sided `z` values by construction.
Do not headline `corr(LD,z)` inside `target_extreme_high` or
`target_extreme_low`; those correlations are undefined. Interpret these cells
with matched LD deltas from base, LD saturation, top-token drift, and steering
slopes instead.

### Prompt Examples

Base height:

```text
Person 1: 162 cm
...
Person 15: 176 cm
Person 16: 180 cm. This person is
```

Parallel +150 cm height:

```text
Person 1: 312 cm
...
Person 15: 326 cm
Person 16: 330 cm. This person is
```

Target-only extreme:

```text
Person 1: 162 cm
...
Person 15: 176 cm
Person 16: 300 cm. This person is
```

Tiny-world height:

```text
Person 1: 12 cm
...
Person 15: 26 cm
Person 16: 30 cm. This person is
```

Tiny-world size is cleaner semantically:

```text
Object 1: 10 cm across
...
Object 15: 26 cm across
Object 16: 30 cm across. This object is
```

### Metrics

```text
corr(LD,z) by condition and N
corr(LD,x) by condition and N
matched LD shift relative to base at the same source cell
top-k token drift under OOD conditions
cos(primal_z_base, primal_z_condition)
base primal_z steering slope on each condition
condition primal_z steering slope on itself
```

### Plots

```text
figures/v14/affine_ood/affine_ood_corr_heatmap.png
figures/v14/affine_ood/affine_ood_by_context_n.png
figures/v14/affine_ood/affine_ood_primal_alignment.png
figures/v14/affine_ood/affine_ood_steering.png
figures/v14/affine_ood/affine_ood_top_tokens.png
```

## Paper Figure 5 Cleanup - CPU Side

This is not a GPU experiment. Rebuild the layer/steering figure from existing
artifacts:

```text
results/v12/layer_sweep_9b.json
results/v12/layer_sweep_9b_steering.json
```

Replace the confusing `probe_z` language with:

```text
Panel A: R^2(z) vs R^2(x) across layers
Panel B: primal_z vs primal_x steering slopes across layers
```

Output:

```text
figures/v14/paper_fig5_layer_x_z_cleanup.png
```

If existing artifacts do not include `primal_x` steering by layer, the script
must say so explicitly and render the available `R^2(z)`/`R^2(x)` panel without
pretending the missing steering data exists.

## Recommended Run

Minimum viable:

```bash
python scripts/run_v14_gpu.py --sections distribution,order,affine_ood,plot \
  --pairs height age weight size speed wealth experience bmi_abs \
  --context-n 31 --dist-cells-per-pair 96 --dist-seeds 4 \
  --order-cells-per-pair 72 --order-seeds 4 \
  --ood-cells-per-pair 72 --ood-seeds 3 --ood-context-ns 5 15 31 \
  --top-k 10
```

CPU-only Figure 5 cleanup:

```bash
python scripts/rebuild_paper_fig5_x_vs_z.py
```
