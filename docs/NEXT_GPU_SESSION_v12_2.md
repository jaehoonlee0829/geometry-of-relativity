# Next GPU Session v12.2 — Residual vs Lexical Cross-pair Transfer

**Created:** Apr 29, 2026

## Goal

V12.1 decomposed each pair's context-derived `primal_z` direction into:

```math
p_z = p_{z,\mathrm{lex}} + p_{z,\mathrm{resid}}
```

where:

```math
p_{z,\mathrm{lex}} = Q_L Q_L^\top p_z
```

and:

```math
p_{z,\mathrm{resid}} = p_z - p_{z,\mathrm{lex}}
```

`p_{z,\mathrm{lex}}` is the part of `primal_z` inside the lexical subspace, and
`p_{z,\mathrm{resid}}` is the orthogonal residual.

V12.1 showed a mixed result:

- only about 8% of `primal_z` norm² lies in the lexical subspace;
- the normalized lexical projection steers strongly;
- the normalized residual still steers at about 69% of original `primal_z`.

V12.2 asks the next question:

```text
Which component is more domain-general across adjective concepts?
```

If the residual component transfers better across pairs, it is the stronger
candidate for the shared relativity / z-score direction. If the lexical
projection transfers better, prior shared-z transfer may partly reflect shared
output-facing adjective semantics.

Use outputs under:

```text
results/v12_2/
figures/v12_2/
```

Primary model:

```text
google/gemma-2-9b
```

Primary layer:

```text
L33
```

Optional if cheap:

```text
L25
```

---

## Required Inputs

Use V12.1 code and outputs as the starting point:

```text
scripts/run_v12_1_lexical_disentanglement.py
results/v12_1/token_position_lexical_capture.json
results/v12_1/lexical_subspace_residualization.json
```

If needed, recompute lexical directions directly from the V12.1 script. The GPU
session should not rely on local-only activation caches that are not committed.

Use existing v11 dense activations and trials:

```text
results/v11/gemma2-9b/<pair>/gemma2-9b_<pair>_v11_residuals.npz
results/v11/gemma2-9b/<pair>/gemma2-9b_<pair>_v11_meta.json
data_gen/v11_<pair>_trials.jsonl
```

Pairs:

```text
height, age, weight, size, speed, wealth, experience, bmi_abs
```

---

## Experiment 1 — Three Cross-pair Transfer Matrices

For each source pair `s`, construct three normalized directions at L33:

```text
d_full[s]  = unit(primal_z[s])
d_lex[s]   = unit(lexical_projection(primal_z[s]))
d_resid[s] = unit(lexical_residual(primal_z[s]))
```

For each target pair `t`, steer target prompts and measure target adjective LD:

```math
\mathrm{LD}_t = \logit(\mathrm{high}_t) - \logit(\mathrm{low}_t)
```

Steering slope:

```math
\frac{\mathbb{E}[\mathrm{LD}_t(h+\alpha\hat d)-\mathrm{LD}_t(h-\alpha\hat d)]}{2\alpha}
```

Recommended:

```text
alpha = 4
prompts per target pair = 160 or 320
target prompt subset = seed0 unique (x,z) cells, same convention as V12/V12.1
```

Write three 8x8 matrices:

```text
M_full[target, source]
M_lex[target, source]
M_resid[target, source]
```

Also include a random null matrix:

```text
M_random[target, source_or_seed]
```

At minimum, use one random direction per source-pair dimensionality. Better:
use 8 random directions and aggregate null bands.

**Outputs**

```text
results/v12_2/residual_vs_lexical_transfer.json
figures/v12_2/residual_vs_lexical_transfer_matrices.png
```

---

## Experiment 2 — Transfer Summary Metrics

For each matrix family:

```text
full
lexical_projection
lexical_residual
random_null
```

compute:

```text
mean_diagonal
mean_off_diagonal
diagonal/off_diagonal ratio
off_diagonal positive fraction
off_diagonal z-score vs random null
target-wise off-diagonal mean
source-wise off-diagonal mean
```

Run paired comparisons over the 56 off-diagonal cells:

```text
residual_offdiag - lexical_offdiag
residual_offdiag - full_offdiag
lexical_offdiag - full_offdiag
```

Use bootstrap CIs over off-diagonal cells:

```text
10,000 bootstrap samples if CPU-cheap; 2,000 is acceptable
95% CI for mean difference
```

If multi-seed steering is feasible, do BH-FDR across off-diagonal cells. If not,
clearly label this as a single-seed transfer follow-up and avoid significance
language.

**Outputs**

```text
results/v12_2/residual_vs_lexical_transfer_summary.json
figures/v12_2/residual_vs_lexical_transfer_summary.png
```

---

## Experiment 3 — Target Lexical-subspace Leakage Check

Important caveat:

```text
source residual is orthogonal to the source lexical subspace,
but not necessarily orthogonal to the target lexical subspace.
```

For every source `s` and target `t`, measure how much each source direction lies
inside the target lexical subspace:

```math
\|Q_{L,t}^\top d_{\mathrm{full},s}\|^2
```

```math
\|Q_{L,t}^\top d_{\mathrm{lex},s}\|^2
```

```math
\|Q_{L,t}^\top d_{\mathrm{resid},s}\|^2
```

This tells us whether apparent residual transfer is secretly entering the target
pair's lexical readout space.

Report:

```text
target_lexical_overlap_full[target, source]
target_lexical_overlap_lex[target, source]
target_lexical_overlap_resid[target, source]
```

Also correlate transfer slope with target lexical overlap:

```text
corr(M_resid offdiag, target_lexical_overlap_resid offdiag)
corr(M_lex offdiag, target_lexical_overlap_lex offdiag)
```

**Interpretation**

- If residual transfer is high while target lexical overlap is low, that is
  stronger evidence for a non-lexical shared relativity direction.
- If residual transfer tracks target lexical overlap, then residual transfer may
  still be using target-side lexical/readout geometry.

**Outputs**

```text
results/v12_2/target_lexical_subspace_leakage.json
figures/v12_2/target_lexical_subspace_leakage.png
```

---

## Experiment 4 — Optional Non-adjective Relative Readout Transfer

Run only if GPU time remains.

Build target prompts that avoid the original adjective pair:

```text
Compared with the group, the target is:
A. above average
B. below average
```

and:

```text
Relative to the others, this value is:
A. higher than typical
B. lower than typical
```

Steer with:

```text
d_full[source]
d_lex[source]
d_resid[source]
random_null
```

Readout:

```math
\mathrm{LD}_{relative} =
\logit(\mathrm{above/higher}) - \logit(\mathrm{below/lower})
```

This is a stronger abstraction test than adjective LD. If residual directions
transfer to non-adjective relative readouts, that supports a shared relative
standing representation beyond adjective semantics.

**Outputs**

```text
results/v12_2/non_adjective_relative_transfer.json
figures/v12_2/non_adjective_relative_transfer.png
```

---

## Main Interpretation Grid

### Case A: residual transfers better than lexical projection

```text
M_resid offdiag > M_lex offdiag
```

Interpretation:

```text
The shared cross-domain z signal is mostly in the non-lexical residual.
Lexical projection is output-potent but less domain-general.
```

This is the cleanest support for a shared relativity direction.

### Case B: lexical projection transfers better than residual

```text
M_lex offdiag > M_resid offdiag
```

Interpretation:

```text
Shared transfer may be driven by common adjective/output semantics.
The shared-z claim must be softened further.
```

### Case C: both transfer

Interpretation:

```text
Relative-standing geometry and adjective decision geometry are coupled.
Use mixed-mechanism framing.
```

### Case D: residual transfers broadly but weakly, lexical transfers strongly but clustered

This is the most plausible outcome.

Interpretation:

```text
Residual carries a weaker but more domain-general relativity signal.
Lexical projection carries high-gain output-facing semantics, strongest on
diagonal and semantically related pairs.
```

---

## Required Figures

Do not mix units in one heatmap.

Required:

```text
figures/v12_2/residual_vs_lexical_transfer_matrices.png
```

Suggested layout:

```text
1x4 panels:
full primal_z transfer
lexical-projection transfer
lexical-residual transfer
random null
```

All panels use the same color scale in `Delta logit-diff per alpha`.

Required:

```text
figures/v12_2/residual_vs_lexical_transfer_summary.png
```

Suggested:

```text
bar plot: diagonal mean vs off-diagonal mean by direction family
with bootstrap CI error bars
```

Required:

```text
figures/v12_2/target_lexical_subspace_leakage.png
```

Suggested:

```text
heatmaps of target lexical overlap for full / lex / residual
plus scatter: residual transfer vs target lexical overlap
```

---

## Research Hygiene Requirements

- Normalize every steering direction before intervention.
- Report `n_prompts` for every target pair and condition.
- Keep cosine/overlap plots separate from steering/logit-diff plots.
- Use one shared color scale across comparable transfer matrices.
- Include random nulls.
- Label whether results are single-seed or multi-seed.
- Save all generated prompt templates if Experiment 4 is run.
- Do not claim "residual is non-lexical" unless target lexical-subspace leakage is
  also low.
- Do not claim "shared z is proven" from transfer alone; phrase as evidence for
  a more domain-general component.

---

## Minimum Viable GPU Run

If time is limited, run:

1. Experiment 1 transfer matrices.
2. Experiment 2 summary metrics.
3. Experiment 3 target lexical-subspace leakage.

Skip Experiment 4 unless cheap.

Minimum outputs:

```text
results/v12_2/residual_vs_lexical_transfer.json
results/v12_2/residual_vs_lexical_transfer_summary.json
results/v12_2/target_lexical_subspace_leakage.json
figures/v12_2/residual_vs_lexical_transfer_matrices.png
figures/v12_2/residual_vs_lexical_transfer_summary.png
figures/v12_2/target_lexical_subspace_leakage.png
```

