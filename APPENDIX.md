# Technical Appendix

This appendix records the exact quantities used in the README and figures. It is
meant to make the claims reproducible without requiring readers to reverse
engineer every script.

## Prompt Variables

For each target item:

- `x`: raw target value, e.g. `170 cm`.
- `mu`: context mean, e.g. the mean height of the listed people.
- `sigma`: context spread.
- `z`: context-normalized standing.

```math
z = \frac{x - \mu}{\sigma}
```

The dense v11 grid chooses `x` and `z` independently, then derives
`mu = x - z sigma`. This is the key design choice that separates raw magnitude
from relative standing.

## Behavioral Logit Difference

For each adjective pair:

```math
\mathrm{LD} = \mathrm{logit}(\text{high adjective}) -
              \mathrm{logit}(\text{low adjective})
```

Examples:

```text
height: LD = logit(" tall") - logit(" short")
age:    LD = logit(" old")  - logit(" young")
wealth: LD = logit(" rich") - logit(" poor")
```

Positive LD means the model leans toward the high adjective. Negative LD means
the model leans toward the low adjective.

## Cell Means

Many v10/v11 analyses average over context seeds inside each `(x, z)` cell:

```math
\bar h_{x,z,L} = \mathbb{E}_{\mathrm{seed}}[h_{x,z,\mathrm{seed},L}]
```

This reduces context-sampling noise and makes the geometry easier to interpret.
The v11 grid has 20 x-values x 20 z-values x 10 seeds per pair, so the cell-mean
geometry has 400 points per pair.

## Behavioral Relativity Ratio

Earlier v4-v9 behavioral analyses fit:

```math
\mathrm{LD} \approx b x + c \mu
```

and report:

```math
R = -\frac{c}{b}
```

If the model is a pure z-score user and `sigma` is roughly fixed:

```math
\mathrm{LD} \propto x - \mu
```

so `b = k`, `c = -k`, and `R = 1`.

Interpretation:

```text
R = 0   mostly raw x, little context-mean effect
R = 1   approximately pure x - mu / z-score behavior
R > 1   context mean shifts the decision more than raw x
```

The README now uses dense v11 `corr(LD, z)` as the cleaner behavioral anchor,
but the v9 ratio is useful historical evidence.

## R² Definitions

Unless explicitly marked as cross-validated probe R², reported R² values are
ordinary squared correlations between two scalar arrays:

```math
R^2(a,b) = \mathrm{corr}(a,b)^2
```

### PCA R²

PCA is fit to activation vectors. For cell-mean activations at one layer:

```math
h_i \in \mathbb{R}^{d}
```

PCA gives a direction:

```math
v_{\mathrm{PC1}} \in \mathbb{R}^{d}
```

Each point gets a scalar PC1 score by projection:

```math
s_i = (h_i - \bar h) \cdot v_{\mathrm{PC1}}
```

Then:

```math
R^2(\mathrm{PC1}, z) = \mathrm{corr}(s, z)^2
```

This is not multivariate regression and not PCA explained variance. PCA
explained variance asks how much activation variance PC1 captures. `R²(PC1,z)`
asks whether position along PC1 orders the examples by z.

Toy example:

```text
z:         -2   -1    0    1    2
PC1 score: -2.1 -0.9  0.1  1.0  2.2
```

Here `R²(PC1,z)` is near 1.

For horseshoe geometries, PC2 can track curvature or extremeness:

```text
z:         -2  -1   0   1   2
PC1 score: -2  -1   0   1   2
PC2 score:  4   1   0   1   4
```

Then `R²(PC1,z)` is high and `R²(PC2,z²)` is high. This motivates the TODO about
testing whether PC2 sometimes represents extremeness (`|z|` or `z²`).

### SAE Feature R²

An SAE feature already gives a scalar activation per prompt or cell:

```math
a_{i,f} = \mathrm{SAEFeature}_f(h_i)
```

For a feature `f`:

```math
R^2(f,z)     = \mathrm{corr}(a_f,z)^2
R^2(f,x)     = \mathrm{corr}(a_f,x)^2
R^2(f,token) = \mathrm{corr}(a_f,\mathrm{token\ magnitude})^2
```

Toy example:

```text
x    mu   z    feature activation
170  150  +2   1.9
170  170   0   0.1
170  190  -2  -1.8
190  190   0   0.0
150  150   0  -0.1
```

This feature tracks z but not raw x: the same raw value `170` can produce high,
middle, or low activation depending on context.

### Probe R²

Probe R² uses a supervised linear model, usually Ridge regression:

```math
\hat y = h w + b
```

When the repo says `cv_R²`, it means cross-validated prediction quality on held
out folds. This is stronger than in-sample squared correlation because the probe
must generalize.

## Direction Definitions

### `primal_z`

For a pair and layer:

```math
\mathrm{primal}_z =
\mathbb{E}[h_L \mid z > 1] - \mathbb{E}[h_L \mid z < -1]
```

This is a simple mean-difference direction from below-local-norm prompts to
above-local-norm prompts.

### `primal_x`

Analogously:

```math
\mathrm{primal}_x =
\mathbb{E}[h_L \mid x \text{ high}] - \mathbb{E}[h_L \mid x \text{ low}]
```

This is useful for red-teaming whether a z direction is just a raw-number
direction.

### Probe Directions

A probe direction is the coefficient vector learned by Ridge regression:

```math
w_z = \arg\min_w \|Hw - z\|^2 + \lambda \|w\|^2
```

There are analogous `w_x`, zero-shot `w_x`, and other supervised directions.

## Steering Slopes

For a unit direction `d_hat`, layer `L`, and intervention scale `alpha`, steering
adds or subtracts the direction to the residual stream:

```math
h_L' = h_L \pm \alpha \hat d
```

The slope is:

```math
\frac{\mathbb{E}[\mathrm{LD}(h+\alpha \hat d) -
                 \mathrm{LD}(h-\alpha \hat d)]}{2\alpha}
```

It is an intervention effect on logit difference per unit alpha, not
`LD / z-score`.

The shared-direction ratio is:

```math
\frac{\mathrm{slope}(w_{\mathrm{shared}}\ \mathrm{on\ target})}
     {\mathrm{slope}(\mathrm{target\ primal}_z\ \mathrm{on\ target})}
```

Toy example:

```text
height own primal_z slope = 0.040
w_shared height slope     = 0.038
ratio = 0.038 / 0.040 = 0.95
```

## Procrustes / Shared z Direction

Each adjective pair has its own `primal_z` vector. The shared direction is a
sign-aligned mean of these per-pair vectors.

Toy example:

```text
height     = (1.0, 0.0)
weight     = (0.8, 0.6)
wealth     = (0.9, 0.2)
experience = (0.1, 1.0)
```

The normalized mean is:

```text
mean = (0.70, 0.45)
w_shared = normalize(mean) ~= (0.84, 0.54)
```

If a direction points opposite the consensus because of sign convention or
noise:

```text
size = (-0.9, -0.1)
```

it is flipped before the final average:

```text
size = (0.9, 0.1)
```

The actual scripts call this the Procrustes-aligned mean. It should be read as a
consensus above-vs-below-local-norm direction, not as proof that every domain
uses the exact same vector.

## Cross-Pair Transfer

For source pair `s` and target pair `t`, cross-pair transfer steers target
prompts with the source pair's `primal_z` direction and reads the target pair's
LD:

```text
source = height direction
target = weight prompts and weight LD
```

v11.5 repeats this across seeds and applies BH-FDR correction over the 56
off-diagonal cells. This is why the result is stronger than a single transfer
heatmap.

## SAE Jaccard Overlap

For each pair, take the top-k SAE features by `R²(z)`, usually `k=50`.

Toy top-5 example:

```text
height = {10, 11, 12, 13, 14}
weight = {12, 13, 14, 20, 21}
```

Then:

```math
\mathrm{Jaccard}(A,B) = \frac{|A \cap B|}{|A \cup B|}
```

For the toy sets:

```text
intersection = {12, 13, 14} = 3
union = {10, 11, 12, 13, 14, 20, 21} = 7
Jaccard = 3 / 7 = 0.43
```

The reported number is the mean off-diagonal Jaccard across pair-pair
comparisons. v11.5 reports about `0.11` for 2B and `0.22` for 9B. This suggests
more shared z-feature basis in 9B, but it does not mean the same features are
used everywhere.

## Fold-Aware Increment R²

The increment analysis asks where new z information appears across layers. The
fold-aware version residualizes each layer against earlier layers using training
folds and evaluates on held-out folds. This avoids the leakage bug in the older
v11 increment analysis.

Interpretation:

```text
high cumulative R²: z is decodable by this layer
high fold-aware increment R²: this layer adds new z information
near-zero later increments: later layers mostly carry forward/rotate existing z
```

## Negative and Follow-Up Results Across Experiments

The project has several results that should be treated as retractions,
measurement warnings, or follow-ups rather than headline claims.

- **v4-v5 absolute/relative split.** The early relative-vs-absolute dichotomy did
  not hold cleanly; the better framing is a continuum of context sensitivity.
- **v6-v7 confound cleanup.** Early grids partially confounded x and z. Grid B
  fixed this and reduced several inflated PC1/shared-direction claims.
- **v8 direct sign control.** Positive/negative math is prompt-sensitive.
  Open-ended logit-difference estimates were inflated when the scored tokens
  were not natural completions. Keep as a measurement warning unless rerun with
  cleaner forced-choice prompts and top-k validation.
- **v9 Park / Fisher variants.** Park-style causal inner products and Fisher
  pullback did not rotate statistical probes into causal directions in the
  tested regimes.
- **v10 attention-head taxonomy.** Mu-aggregator/comparator/z-writer labels were
  useful descriptive DLA, but v11/v11.5 ablations and permutation tests refuted
  the causal head-taxonomy framing.
- **v11 single-seed transfer.** Replaced by v11.5 multi-seed BH-FDR transfer.
- **v11 orthogonal increment R².** Replaced by v11.5 fold-aware residualization.
- **SAE interpretation.** Current SAE controls rule out raw x and numeric-token
  magnitude for top z features, but do not yet rule out lexical or domain-word
  features.
- **PC2 / z² geometry.** Some pairs show high `R²(PC2,z²)`, suggesting possible
  extremeness or curvature structure, but this needs systematic analysis before
  becoming a claim.

