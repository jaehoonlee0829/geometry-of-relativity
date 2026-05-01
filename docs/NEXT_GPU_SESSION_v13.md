# Next GPU Session v13 - OOD Relativity, X-transfer Controls, and Robust Readout

**Created:** May 1, 2026

## Goal

V13 tests whether the context-normalized relativity behavior found in V11/V12 is
robust outside the original human-like value ranges, and whether the shared
cross-pair steering story is specific to `z` rather than a generic property of
raw `x` directions.

The session has four parts:

1. OOD / affine-shift relativity: does behavior follow `z` when the whole
   numerical world is shifted or scaled?
2. Shared-direction control: reproduce the V11 cross-pair 8x8 steering matrix
   for `x`, so it can be compared directly to the existing `z` matrix.
3. Robust readout: inspect top-logit trajectories, not only fixed
   high-minus-low adjective logit differences.
4. Domain extension: replicate the most informative V12-style adjective pairs
   and add a small set of non-body relative domains plus objective controls.

Primary model:

```text
google/gemma-2-9b
```

Secondary model if cheap:

```text
google/gemma-2-2b
```

Primary steering layers:

```text
L25, L33
```

Use outputs under:

```text
results/v13/
figures/v13/
```

---

## Definitions

For each prompt cell:

```math
z = \frac{x - \mu}{\sigma}
```

where:

- `x` is the raw target value;
- `mu` is the context mean;
- `sigma` is the context standard deviation;
- `z` is target standing relative to the local context.

An affine transform changes values as:

```math
x' = a x + b,\quad \mu' = a\mu + b,\quad \sigma' = |a|\sigma
```

For `a > 0`, the z-score is invariant:

```math
z' = \frac{x' - \mu'}{\sigma'} = z
```

Example:

```text
base:          x=185 cm, mu=170 cm, sigma=10 -> z=+1.5
shift +100:    x=285 cm, mu=270 cm, sigma=10 -> z=+1.5
scale x2:      x=370 cm, mu=340 cm, sigma=20 -> z=+1.5
```

If the model implements a clean relativity computation, the shifted/scaled
worlds should preserve behavior at fixed `z`. If absolute priors dominate,
behavior should change when values become implausible or out-of-distribution.

---

## Required Inputs

Use existing V11/V12 infrastructure where possible:

```text
data_gen/v11_<pair>_trials.jsonl
results/v11/gemma2-9b/<pair>/gemma2-9b_<pair>_v11_residuals.npz
results/v11/gemma2-9b/<pair>/gemma2-9b_<pair>_v11_meta.json
figures/v11/steering/
results/v12/
results/v12_1/
results/v12_2/
```

Use Alex's PR18 phase code as reference for shot-sweep, suppression, and
top-logit style analysis:

```text
internal/kshot/phase/
internal/kshot/phase/figures/
internal/kshot/phase/results/
internal/kshot/phase/scripts/
```

Original eight V11/V12 pairs:

```text
height, age, weight, size, speed, wealth, experience, bmi_abs
```

V13 should not run every experiment on every possible pair if GPU time is tight.
Use the prioritized pair list in each section.

---

## Experiment 1 - OOD / Affine-shift Relativity

### Question

Does the model keep using `z` when the raw values move outside the familiar
training range?

Examples:

```text
normal world:       people around 170 cm, target 185 cm
tall-world shift:   people around 270 cm, target 285 cm
extreme target:     people around 170 cm, target 300 cm
scaled world:       all heights multiplied by 2
```

### Conditions

For each selected pair, generate dense grids with the same target z values but
different absolute value regimes:

```text
base             original V11-like value range
parallel_shift   x, mu shifted by +delta, sigma unchanged
negative_shift   x, mu shifted by -delta where valid
scale_up         x, mu, sigma multiplied by a > 1
scale_down       x, mu, sigma multiplied by 0 < a < 1 where valid
target_ood       target x pushed outside normal range while context remains normal
world_ood        context and target both outside normal range, same z grid
```

Recommended primary pairs:

```text
height, speed, weight, experience
```

Optional if cheap:

```text
age, wealth, size, bmi_abs
```

### Metrics

For every condition:

```text
corr(LD, z)
corr(LD, x)
corr(LD, mu)
corr(LD, z_eff) if visible context differs from intended z
slope from LD ~ z + x + mu + condition + z:condition
```

Also compute activation-space checks:

```text
cos(primal_z_base, primal_z_condition)
cos(probe_z_base, probe_z_condition)
train probe on base, test on condition
train probe on condition, test on base
```

Steering check:

```text
steer condition prompts using base primal_z
steer base prompts using condition primal_z
```

### Outputs

```text
results/v13/affine_shift/affine_shift_metrics.json
results/v13/affine_shift/affine_shift_probe_transfer.json
results/v13/affine_shift/affine_shift_steering.json
figures/v13/affine_shift/affine_shift_ld_vs_z.png
figures/v13/affine_shift/affine_shift_corr_bars.png
figures/v13/affine_shift/affine_shift_probe_transfer.png
figures/v13/affine_shift/affine_shift_steering.png
```

### Interpretation

Clean relativity evidence:

```text
corr(LD,z) stays high across affine-shifted worlds
base primal_z and shifted primal_z remain aligned
base primal_z steers shifted prompts
top logits do not collapse into OOD/weird-value tokens
```

Absolute-prior failure:

```text
corr(LD,x) rises in OOD conditions
corr(LD,z) drops at same z
top logits switch to extreme/OOD language such as impossible, giant, tiny, etc.
base primal_z no longer transfers
```

---

## Experiment 2 - X Cross-pair Transfer 8x8 Control

### Question

V11/V12 showed that `primal_z` can transfer across adjective pairs. Is that
specific to relative standing, or do raw `x` directions transfer similarly?

This experiment should reproduce the V11-style cross-pair transfer 8x8 plot,
but with `primal_x` directions.

### Directions

For each source pair `s`, build:

```text
d_z[s] = unit(primal_z[s])
d_x[s] = unit(primal_x[s])
```

Use two `x` variants if feasible:

```text
primal_x_naive:
  mean activation at high raw x - mean activation at low raw x

primal_x_resid_z:
  raw-x direction after residualizing out z, or estimated from matched-z slices
```

The residualized/matched-z version is important because in the original grid
`x` and `z` are not always fully independent.

### Steering

For every target pair `t` and source pair `s`, steer target prompts with the
source direction and measure the target adjective logit difference:

```math
\mathrm{LD}_t = \logit(\mathrm{high}_t) - \logit(\mathrm{low}_t)
```

Steering slope:

```math
\frac{\mathbb{E}[\mathrm{LD}_t(h+\alpha\hat d_s)-\mathrm{LD}_t(h-\alpha\hat d_s)]}{2\alpha}
```

Use the same alpha, target subset, prompt format, and layer conventions as the
existing V11 cross-pair steering plot.

### Outputs

```text
results/v13/x_transfer/cross_pair_transfer_x_8x8.json
results/v13/x_transfer/cross_pair_transfer_x_vs_z_summary.json
figures/v13/x_transfer/cross_pair_transfer_x_8x8_gemma2-9b.png
figures/v13/x_transfer/cross_pair_transfer_x_8x8_gemma2-2b.png
figures/v13/x_transfer/cross_pair_transfer_z_vs_x_8x8_gemma2-9b.png
figures/v13/x_transfer/cross_pair_transfer_z_vs_x_8x8_gemma2-2b.png
```

### Interpretation

Evidence for a shared relativity direction:

```text
z off-diagonal transfer >> x off-diagonal transfer
z transfer remains positive across many target/source pairs
x transfer is mostly diagonal, weak, or lexical/domain-specific
```

Evidence against specificity:

```text
x transfers about as well as z
x transfer has similar off-diagonal structure
raw magnitude directions are just as shared as relativity directions
```

Report:

```text
mean diagonal z, mean off-diagonal z
mean diagonal x, mean off-diagonal x
offdiag(z) - offdiag(x) bootstrap CI
target-wise exceptions
source-wise exceptions
```

---

## Experiment 3 - Top-logit Trajectories and Robust Readout

### Question

Fixed high-minus-low adjective LD is useful but too narrow. The model may move
probability mass into related tokens such as:

```text
tall, taller, giant, huge, high, above-average, normal, average, short, tiny
```

This is especially important for OOD prompts, where the model may stop using the
original adjective pair and switch to extreme or weird-value language.

### Procedure

For each prompt condition and z bin, save top-k logits at the target readout
position:

```text
top_k = 50 or 100
```

Store:

```text
token string
token id
logit
probability after softmax over vocabulary
rank
condition
pair
x
mu
sigma
z
z_eff
model
layer/intervention if applicable
```

Build semantic token groups for analysis:

```text
high adjective set
low adjective set
neutral / middle set
extreme high set
extreme low set
OOD / impossible / weird-value set
unit tokens
number tokens
other
```

Use logsumexp group scoring:

```math
\mathrm{score}(G) = \log \sum_{t\in G}\exp(\logit_t)
```

Then compare:

```text
classic LD = logit(high_token) - logit(low_token)
group LD = logsumexp(high_set) - logsumexp(low_set)
top-token rank trajectories across z
semantic-mass trajectories across z
```

### Outputs

```text
results/v13/top_logits/top_logits_by_condition.jsonl
results/v13/top_logits/top_logit_group_scores.json
figures/v13/top_logits/top_token_trajectories_by_z.png
figures/v13/top_logits/semantic_mass_by_z.png
figures/v13/top_logits/classic_ld_vs_group_ld.png
figures/v13/top_logits/ood_top_tokens_examples.png
```

### Interpretation

This should answer:

```text
Are classic tall-short LD results robust to synonyms?
Do OOD worlds produce normal relativity tokens or weird/OOD tokens?
Does steering shift the intended semantic mass or only one selected token?
Do middle/neutral adjectives explain cases where high-low LD looks weak?
```

---

## Experiment 4 - Domain Extension and Objective Controls

### Question

Are V11/V12 relativity results specific to human/body-ish properties, or do they
generalize to less correlated domains? Do objective classifications behave
differently from relative adjective judgments?

### Existing pairs to replicate

Use a small set of the most informative V12 pairs:

```text
height      strong canonical human/body pair
weight      strong canonical physical pair
speed       known exception / weaker transfer case
experience  lexical/residual entanglement was interesting in V12.1/V12.2
```

Optional if cheap:

```text
wealth
```

### New relative domains

Prioritize non-body domains:

```text
brightness: dim / bright, values in lumens or lux
temperature: cold / hot, values in C or F
loudness: quiet / loud, values in dB
price: cheap / expensive, values in dollars
```

Optional:

```text
duration: short / long, values in seconds or minutes
distance: near / far, values in meters or kilometers
frequency: low-frequency / high-frequency, values in Hz
```

### Objective controls

Use forced-choice prompts, not open-ended completions:

```text
positive / negative
even / odd
above zero / below zero
pass / fail at fixed threshold
fever / no fever at fixed threshold
adult / minor at fixed threshold
```

The objective-control question is:

```text
If context is provided, does the model incorrectly convert objective decisions
into relative decisions, or does it preserve absolute rule-based behavior?
```

### Outputs

```text
results/v13/domain_extension/domain_metrics.json
results/v13/domain_extension/objective_control_metrics.json
figures/v13/domain_extension/domain_corr_summary.png
figures/v13/domain_extension/domain_pca_panels.png
figures/v13/domain_extension/objective_vs_relative_summary.png
figures/v13/domain_extension/new_domain_cross_pair_transfer.png
```

### Interpretation

Evidence that relativity is general:

```text
new relative domains show high corr(LD,z)
new domains have z-structured activation geometry
new domains transfer at least partially with existing primal_z directions
objective controls do not show the same z-like behavior
```

Evidence that the story is narrower:

```text
body/people properties work, but brightness/temperature/loudness/price do not
objective controls also become z-like, suggesting generic context bias
new domains do not transfer with old domains
```

---

## Minimum Viable V13

If GPU time is limited, run this subset first:

```text
model: google/gemma-2-9b
pairs: height, weight, speed, experience
layers: L25, L33
experiments: 1, 2, 3
new domains: brightness, temperature
objective controls: positive/negative, even/odd
```

Minimum required figures:

```text
figures/v13/affine_shift/affine_shift_corr_bars.png
figures/v13/affine_shift/affine_shift_ld_vs_z.png
figures/v13/x_transfer/cross_pair_transfer_x_8x8_gemma2-9b.png
figures/v13/x_transfer/cross_pair_transfer_z_vs_x_8x8_gemma2-9b.png
figures/v13/top_logits/top_token_trajectories_by_z.png
figures/v13/top_logits/semantic_mass_by_z.png
figures/v13/domain_extension/objective_vs_relative_summary.png
```

---

## Main Claims V13 Can Support

Strong result:

```text
The model's relativity computation is approximately affine-invariant: behavior
and geometry track z even when the entire numerical world is shifted or scaled.
Shared cross-pair steering is specific to z rather than raw x. OOD readout
remains semantically stable under top-logit inspection.
```

Mixed result:

```text
Relativity is robust inside normal ranges but degrades under extreme absolute
values. Shared z remains stronger than shared x, but OOD prompts reveal lexical
or absolute-prior leakage.
```

Negative result:

```text
The behavior is not affine-invariant. Extreme raw values dominate readout, x
directions transfer about as well as z, or top logits reveal that the original
LD metric hid a different semantic behavior.
```

