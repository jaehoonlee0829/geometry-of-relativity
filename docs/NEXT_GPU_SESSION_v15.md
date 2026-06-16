# Next GPU Session v15 - Leave-One-Out Shared Directions for z vs x

## Goal

Test the strongest version of the shared-direction claim in Section 4.2:

> Does a shared relativity direction generalize to a held-out adjective concept, and is this stronger for relative standing `z` than for raw magnitude `x`?

The current Figure 8 uses an all-concept sign-aligned mean of the eight per-concept `d_z` directions. That is useful, but it includes the target concept in the shared vector. V15 should produce both the all-concept baseline and the stronger leave-one-out version.

## Main Experiment

Use Gemma 2 9B at the canonical late layer, matching V11/V13:

- model: `google/gemma-2-9b`
- layer: `33`
- alpha: `4.0`
- adjective pairs:
  `height`, `age`, `weight`, `size`, `speed`, `wealth`, `experience`, `bmi_abs`

For each target pair `p`:

1. Compute per-pair directions for all eight pairs:
   - `d_z[p] = mean(h_L | z > 1) - mean(h_L | z < -1)`
   - `d_x[p] = mean(h_L | x >= q75_p) - mean(h_L | x <= q25_p)`
   - optional control: `d_x_resid_z[p]`, using raw `x` residualized against `z`.
2. Build all-concept shared directions using all eight pairs:
   - initialize with the mean of all eight source directions,
   - sign-flip each source direction so its dot product with the initial mean is positive,
   - average and unit-normalize.
3. Build leave-one-out shared directions from the seven non-target pairs:
   - initialize with the mean of the seven source directions,
   - sign-flip each source direction so its dot product with the initial mean is positive,
   - average and unit-normalize.
4. Also compute full source-by-target transfer matrices:
   - `full_matrix_z[target][source]`: steer target prompts with source `d_z[source]`.
   - `full_matrix_x[target][source]`: steer target prompts with source `d_x[source]`.
   - optional: `full_matrix_x_resid_z[target][source]`.
5. On the target prompts, measure steering slopes for:
   - target's own `d_z[p]`
   - all-concept shared `d_z`
   - LOO shared `d_z[-p]`
   - target's own `d_x[p]`
   - all-concept shared `d_x`
   - LOO shared `d_x[-p]`
   - optional: own and LOO `d_x_resid_z`
6. Report ratios:
   - `all_shared_z_slope / own_z_slope`
   - `loo_shared_z_slope / own_z_slope`
   - `all_shared_x_slope / own_x_slope`
   - `loo_shared_x_slope / own_x_slope`
   - optional: `loo_shared_x_resid_z_slope / own_x_resid_z_slope`

## Required Outputs

Write:

```text
results/v15/shared_direction_loo_z_vs_x.json
figures/v15/shared_direction_full_matrix_z_vs_x.png
figures/v15/shared_direction_loo_matrix_z_vs_x.png
figures/v15/shared_direction_loo_ratios_z_vs_x.png
paper/icml2026_draft/figures/fig_results_shared_direction_loo_z_vs_x_clean.png
```

The JSON must include:

- model id and model short name
- layer
- alpha
- pairs
- per-pair own slopes for `d_z` and `d_x`
- all-concept shared slopes and ratios
- leave-one-out shared slopes and ratios
- full source-by-target matrices for `d_z` and `d_x`
- leave-one-out target summaries for `d_z` and `d_x`

The main paper-facing figure should be a compact one-column plot:

- x-axis: adjective concepts
- y-axis: shared / own steering ratio
- two bars per concept: LOO `d_z` and LOO `d_x`
- horizontal reference line at `0.5`
- reader-facing labels, no "V15" in the title
- large enough font for ICML one-column display

If the plot becomes crowded, use a two-row version:

- top: `d_z`
- bottom: `d_x`
- same y-axis scale.

Also generate appendix-ready matrix plots:

1. Full source-by-target transfer matrices, side-by-side `d_z` vs `d_x`.
2. Leave-one-out shared-direction ratio matrices, side-by-side `d_z` vs `d_x`: rows are held-out target concepts and columns are `own`, `all-shared`, and `LOO-shared`, with cells shown as steering slope or shared/own ratio.

The full matrix plot should answer: "Do pair-specific source directions transfer generally?"
The LOO ratio plot should answer: "Does a shared direction built without the target concept recover held-out steering, and is this more true for `d_z` than `d_x`?"

## Acceptance Criteria

The result supports the stronger paper claim if:

- LOO shared `d_z` recovers at least half of own-`d_z` steering for most concepts, ideally at least 6/8.
- LOO shared `d_z` ratios are consistently larger than LOO shared `d_x` ratios.
- Raw `d_x` transfer is weak, diagonal, unstable, or much less positive than `d_z`.

The result weakens the claim if:

- LOO shared `d_z` fails on most concepts.
- LOO shared `d_x` transfers about as well as `d_z`.
- Effects are dominated by one source concept or one target concept.

## Implementation Hints

Reuse existing helpers rather than starting from scratch:

- `scripts/analyze_v11_5_shared_z.py`
  - computes per-pair `d_z`
  - builds the current sign-aligned shared direction
  - measures shared-vs-own steering ratios
- `scripts/run_v13_gpu.py`
  - computes `d_z`, `d_x`, and `d_x_resid_z`
  - contains the V13 `z` vs raw-`x` cross-pair transfer control

The new script can be:

```text
scripts/run_v15_shared_direction_loo.py
```

Suggested command:

```bash
python scripts/run_v15_shared_direction_loo.py \
  --model-short gemma2-9b \
  --alpha 4.0 \
  --batch-size 8 \
  --max-seq 288
```

## GPU Session Prompt

Please run V15 for `geometry-of-relativity`.

We need full and leave-one-out shared-direction experiments for the ICML paper. The question is whether a shared relativity direction generalizes to held-out adjective concepts, and whether this is stronger for `z` than raw `x`.

Use Gemma 2 9B, layer 33, alpha 4.0, and the eight standard pairs: height, age, weight, size, speed, wealth, experience, bmi_abs.

For every target pair:

1. Compute each pair's `d_z = mean(h_L | z > 1) - mean(h_L | z < -1)`.
2. Compute each pair's `d_x = mean(h_L | x >= q75) - mean(h_L | x <= q25)`.
3. Optional but useful: compute `d_x_resid_z` by residualizing `x` against `z` before taking high-minus-low activation means.
4. Build all-concept shared `d_z` and `d_x` from all eight directions by sign-aligning them to their initial mean, then averaging and normalizing.
5. Build LOO shared `d_z` and `d_x` from the seven non-target directions by the same sign-aligned averaging procedure.
6. Compute full source-by-target steering matrices for `d_z` and `d_x`.
7. Steer target prompts with own `d_z`, all-concept shared `d_z`, LOO shared `d_z`, own `d_x`, all-concept shared `d_x`, and LOO shared `d_x`.
8. Report steering slopes and ratios `shared / own`.

Save JSON to `results/v15/shared_direction_loo_z_vs_x.json`.
Save the paper-facing compact plot to `paper/icml2026_draft/figures/fig_results_shared_direction_loo_z_vs_x_clean.png`.
Save appendix-ready matrix plots under `figures/v15/`.

The plot should compare `d_z` vs `d_x` shared/own ratios side by side for each concept, with a horizontal line at 0.5 and no internal "V15" title.

Please also print a short summary:

- model, layer, alpha, prompt count per target
- how many concepts pass ratio > 0.5 for LOO shared `d_z`
- how many concepts pass ratio > 0.5 for LOO shared `d_x`
- mean ratio for `d_z`
- mean ratio for `d_x`
- mean off-diagonal full-matrix transfer for `d_z`
- mean off-diagonal full-matrix transfer for `d_x`
- any concepts where `d_x` unexpectedly transfers about as well as `d_z`
