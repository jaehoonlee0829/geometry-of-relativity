# Next GPU Session v12.1 — Lexical Disentanglement Follow-up

**Created:** Apr 29, 2026

## Goal

v12.1 is a narrow follow-up to V12, not a new broad experiment. V12 showed that
context-derived `primal_z` steering is strong, but lexical sentence directions
can steer as strongly or more strongly. The goal here is to separate:

1. literal adjective-token semantics;
2. sentence/template state;
3. context-derived relative-standing geometry.

Use outputs under:

```text
results/v12_1/
figures/v12_1/
```

Primary model:

```text
google/gemma-2-9b
```

Primary layer:

```text
L33
```

Optional layers if cheap:

```text
L25, L33
```

The core question is:

```text
Does the causal power of primal_z survive after we explicitly remove lexical
adjective/sentence subspaces?
```

---

## Required Existing Inputs

Use existing v11 dense activations and metadata:

```text
results/v11/gemma2-9b/<pair>/<model>_<pair>_v11_residuals.npz
results/v11/gemma2-9b/<pair>/<model>_<pair>_v11_meta.json
```

Use the same eight pairs:

```text
height, age, weight, size, speed, wealth, experience, bmi_abs
```

---

## Experiment 1 — Token-position Lexical Capture

**Problem with V12:** lexical sentence directions were captured at the final
token position. For prompts like `This person is tall.`, the final state may mix
adjective semantics, sentence completion, punctuation, and template state.

**Fix:** capture hidden states at specific token positions.

For each pair, build lexical prompts:

```text
The adjective is tall
The adjective is short
This person is tall
This person is short
A described case is tall
A described case is short
```

Also include synonyms:

```text
height: tall/high/large vs short/low/small
age: old/elderly/aged vs young/new/youthful
weight: heavy/weighty/large vs light/thin/small
size: big/large/huge vs small/little/tiny
speed: fast/quick/rapid vs slow/sluggish/unhurried
wealth: rich/wealthy/affluent vs poor/low-income/broke
experience: expert/experienced/veteran vs novice/new/inexperienced
bmi_abs: obese/heavy/large vs thin/lean/light
```

For each prompt, locate the adjective token span and capture:

```text
h_adjective_token = mean hidden state over adjective token span
h_final_token     = final hidden state
```

Construct directions:

```text
d_word_token      = mean(h_adjective_token | high adjective prompts)
                    - mean(h_adjective_token | low adjective prompts)

d_sentence_token  = same, but only sentence templates
d_sentence_final  = mean(h_final_token | high sentence prompts)
                    - mean(h_final_token | low sentence prompts)
d_synonym_token   = high synonym token mean - low synonym token mean
```

Compare cosines:

```text
cos(primal_z_context, d_word_token)
cos(primal_z_context, d_sentence_token)
cos(primal_z_context, d_sentence_final)
cos(primal_z_context, d_synonym_token)
cos(primal_x_context, d_word_token)
cos(primal_x_context, d_sentence_token)
```

**Interpretation**

- If `d_sentence_final` is much closer to `primal_z` than `d_sentence_token`,
  then V12's lexical result was partly sentence/final-state contamination.
- If `d_word_token` is already close to `primal_z`, then `primal_z` genuinely
  overlaps adjective-token semantics.

**Outputs**

```text
results/v12_1/token_position_lexical_capture.json
figures/v12_1/token_position_lexical_cosines.png
```

---

## Experiment 2 — Lexical-subspace Residualization of primal_z

This is the main experiment.

Build a lexical subspace `L` from many lexical directions:

```text
d_word_token
d_sentence_token
d_sentence_final
d_synonym_token
d_domain_token
```

where `d_domain_token` uses domain prompts such as:

```text
The property is height
The property is age
The measurement is weight
The concept is wealth
```

Use QR/SVD to make an orthonormal basis:

```math
Q_L = \operatorname{orth}(L)
```

For each pair:

```math
p_z = \operatorname{unit}(\mathrm{primal\_z})
```

Split it into lexical and non-lexical pieces:

```math
p_{z,\mathrm{lex}} = Q_L Q_L^\top p_z
```

```math
p_{z,\mathrm{resid}} = p_z - p_{z,\mathrm{lex}}
```

Normalize both nonzero vectors:

```math
\hat p_{z,\mathrm{lex}} = \operatorname{unit}(p_{z,\mathrm{lex}})
```

```math
\hat p_{z,\mathrm{resid}} = \operatorname{unit}(p_{z,\mathrm{resid}})
```

Steer dense-context prompts with:

```text
primal_z_context
lexical_projection(primal_z)
lexical_residual(primal_z)
primal_x_context
d_word_token
d_sentence_token
d_sentence_final
random_null
```

Measure:

```math
\frac{\mathbb{E}[\mathrm{LD}(h+\alpha\hat d)-\mathrm{LD}(h-\alpha\hat d)]}{2\alpha}
```

Recommended:

```text
alpha = 4
prompts per pair = 160 or 320
```

**Interpretation**

- If `lexical_projection(primal_z)` explains most steering and
  `lexical_residual(primal_z)` is weak, then the current causal effect is mostly
  lexical/adjective geometry.
- If `lexical_residual(primal_z)` still steers strongly, then there is a
  non-lexical context-relative component.
- If both steer, the representation is mixed: relative standing and lexical
  adjective semantics share causal coordinates.

**Outputs**

```text
results/v12_1/lexical_subspace_residualization.json
figures/v12_1/lexical_subspace_residualization_steering.png
figures/v12_1/lexical_subspace_fraction_removed.png
```

Required scalar summaries:

```text
fraction_of_primal_z_norm_in_lexical_subspace
cos(primal_z, lexical_projection)
cos(primal_z, lexical_residual)
steering_slope(primal_z)
steering_slope(lexical_projection)
steering_slope(lexical_residual)
residual_over_primal_steering_ratio
```

---

## Experiment 3 — Same Adjective Readout, Different Context z

Use prompts where the scored adjective pair is identical, but the context makes
the same raw target high-z or low-z.

Example:

```text
short-context group: target 170 cm -> high z
tall-context group:  target 170 cm -> low z
```

The output readout is still:

```text
logit(tall) - logit(short)
```

but the raw target value and lexical readout words are held fixed.

Measure whether the pre-output activation contrast aligns more with:

```text
primal_z_context
lexical token directions
lexical residual direction from Experiment 2
```

**Outputs**

```text
results/v12_1/same_adjective_different_context_z.json
figures/v12_1/same_adjective_different_context_z.png
```

**Interpretation**

If the contrast tracks context-z while the adjective readout is fixed, that is
direct evidence for a context-relative component beyond the literal adjective
tokens.

---

## Experiment 4 — Non-adjective Relative Readout Transfer

Avoid direct adjective labels entirely.

Prompt style:

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

Test whether the original adjective-derived `primal_z` direction transfers to
these non-adjective relative-choice readouts.

Directions to steer:

```text
primal_z_context from adjective prompts
lexical_residual(primal_z)
d_word_token
d_sentence_token
random_null
```

Readout:

```text
LD_relative = logit(above/higher) - logit(below/lower)
```

**Outputs**

```text
results/v12_1/non_adjective_relative_readout_transfer.json
figures/v12_1/non_adjective_relative_readout_transfer.png
```

**Interpretation**

- Strong transfer from `lexical_residual(primal_z)` supports a genuine
  non-adjective relative-standing direction.
- Transfer only from adjective lexical directions would weaken the abstraction
  claim.

---

## Minimum Viable GPU Run

If time is limited, run only:

1. Experiment 1 token-position lexical capture.
2. Experiment 2 lexical-subspace residualization.

These two are enough to decide whether V12's lexical steering result is mostly
adjective-token semantics, sentence-final template state, or a genuine overlap
with `primal_z`.

Minimum outputs:

```text
results/v12_1/token_position_lexical_capture.json
results/v12_1/lexical_subspace_residualization.json
figures/v12_1/token_position_lexical_cosines.png
figures/v12_1/lexical_subspace_residualization_steering.png
```

---

## Research Hygiene Requirements

- Do not mix cosine similarities and steering/logit effects in the same heatmap.
- Do not print `nan` cells in final figures; omit missing cells or use a separate
  availability mask.
- Report sample counts for every steering condition.
- Report the exact token-position capture rule.
- Save prompt templates and token spans in JSON for reproducibility.
- Include random-direction nulls for every steering plot.
- Keep claims conservative: this is a follow-up red-team, not a new headline
  unless the residual direction survives strongly.

