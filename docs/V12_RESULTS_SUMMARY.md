# V12 Results Summary

Created: Apr 28, 2026

## Scope

V12 was a claim-hardening pass on the v11/v11.5 story, using the existing dense
v11 activations plus a fresh Gemma 2 9B GPU run for steering and lexical probes.
The outputs live under:

```text
results/v12/
figures/v12/
```

The key point is that V12 strengthens the narrow encode-vs-use observation, but
it also forces several paper claims to be stated more conservatively.

## Completed outputs

```text
results/v12/layer_sweep_9b.json
results/v12/layer_sweep_9b_steering.json
figures/v12/layer_sweep_9b_combined.png

results/v12/direction_redteam_x_lexical_z.json
results/v12/direction_redteam_lexical_activations.json
results/v12/direction_redteam_steering.json
figures/v12/direction_redteam_cosines.png
figures/v12/direction_redteam_steering.png

results/v12/pure_x_transfer_control.json
figures/v12/pure_x_transfer_control.png

results/v12/sae_feature_lexical_audit.json
figures/v12/sae_feature_interpretation_examples.png
docs/v12_sae_feature_notes.md

results/v12/pc_extremeness_x_z_audit.json
figures/v12/pc_extremeness_x_z_grid.png
```

## Main read

### 1. Layer sweep

Gemma 2 9B `z` is linearly decodable very early under the grouped dense-grid
split, and `primal_z` steering is strongest later. Mean `primal_z` steering
slope across the eight pairs is roughly:

```text
L0  -0.003
L17 +0.022
L25 +0.097
L33 +0.067
L41 +0.034
```

Random-null slopes are usually much smaller, but not always negligible for some
pair/layer combinations. Therefore the safe claim is:

```text
z is decodable early and the primal_z intervention is most potent at later
layers in this 9B strategic-layer sweep.
```

Avoid claiming a fully identified causal encode-then-use circuit.

### 2. Direction red-team

The late-layer `primal_z` direction is only moderately aligned with the raw
`primal_x` direction and simple high-low unembedding readout. However, the
stronger lexical controls are not cleanly passed:

```text
mean L33 steering slope, Gemma 2 9B:
primal_z_context        +0.067
primal_x_context        +0.024
lex_high_low_word       +0.052
lex_sentence_high_low   +0.086
random_null             -0.001
```

Sentence-style lexical directions are often as strong as, or stronger than, the
context `primal_z` direction. This means V12 does not support a strong "not
lexical semantics" claim. Use the weaker framing:

```text
The primal_z contrast is not simply the raw-x contrast or the high-low
unembedding vector, but lexical/adjective sentence directions remain a serious
overlap and causal competitor.
```

### 3. Pure-x / fixed-mu transfer control

Cross-pair transfer persists across the simple controlled subsets, but the
control does not selectively rescue the shared-z interpretation:

```text
mean within / cross slopes:
full_grid  +0.066 / +0.026
fixed_mu   +0.067 / +0.026
fixed_x    +0.065 / +0.026
matched_z  +0.066 / +0.027
```

Because matched-z / varying-x does not weaken transfer in this implementation,
the pure-x control should be described as mixed or inconclusive, not as a clean
refutation of shared scalar magnitude.

### 4. SAE lexical audit

The v11.5 top z-SAE features are not uniformly pure relative-standing features
once lexical/domain probes are included. Across eight pairs x top-25 features:

```text
pure-ish z          43
lexical z-like      39
raw numeric         52
mixed/polysemantic  66
```

This supports a cautious statement that z-correlated sparse features exist, but
it does not support saying that top SAE features generally implement pure
relative standing.

### 5. PC extremeness audit

Some pairs show an extremeness component (`|z|` or `z^2`), often on PC2 or PC3,
but this is not a universal PC2 result. Raw `x`, signed `z`, and pair-specific
variance remain strong competitors in multiple cases. Use:

```text
Some pairs show a low-dimensional extremeness/curvature component; the component
is pair-specific and not consistently PC2.
```

Avoid:

```text
PC2 encodes extremeness.
```

## Paper/README consequences

Promote:

- early linear decodability of `z` under the dense grouped split;
- later-layer potency of `primal_z` steering in the 9B strategic-layer sweep;
- moderate raw-x/unembedding separation for the primal contrast;
- existence of z-correlated SAE features.

Soften:

- "z is not lexical semantics";
- "cross-pair transfer is not shared scalar magnitude";
- "SAE features implement relative standing";
- "PC2 is extremeness."

## Repro command

```bash
scripts/run_v12_all.sh
```

The run loads Gemma 2 9B once for the GPU sections and then regenerates the CPU
summaries/figures after GPU JSON exists.
