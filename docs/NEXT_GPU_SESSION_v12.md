# Next GPU Session v12 — Claim Hardening and Red-Team Controls

**Created:** Apr 28, 2026

## Goal

v12 is not a new broad sweep. It is a claim-hardening session for the current
README/paper story:

1. z is available early, while steering works later.
2. z geometry is not just raw number magnitude.
3. z geometry is also not just raw lexical/adjective semantics such as
   `tall`, `short`, `height`, or synonyms.
4. cross-pair steering is not just shared raw scalar magnitude.
5. SAE z-features are not merely raw numerals, token-frequency artifacts, or
   lexical/domain-word features.
6. PC2 / curved horseshoe structure may encode extremeness (`|z|`, `z²`) rather
   than signed relativity alone.

The desired output is a small number of decisive JSON summaries and paper-grade
figures under `results/v12/` and `figures/v12/`.

---

## Models and Data

Primary model:

| Model | Layer targets | Why |
|---|---:|---|
| Gemma 2 9B | all layers for layer sweep; L33 for late geometry/SAE | strongest current model and README focus |

Secondary model, only where cheap:

| Model | Layer targets | Why |
|---|---:|---|
| Gemma 2 2B | L20 / existing v11 layers | cross-scale check |

Use existing v11 dense activations when possible:

```text
results/v11/gemma2-2b/<pair>/*.npz
results/v11/gemma2-9b/<pair>/*.npz
```

Generate fresh GPU data only for:

- per-layer steering hooks;
- zero-shot / lexical prompt activations if missing;
- SAE activations if current caches are insufficient;
- pure-x / fixed-mu steering controls.

---

## Priority 1 — Dense v11-style Layer Sweep on Gemma 2 9B

**Question:** Is the canonical encode-vs-use story true on dense v11 9B?

Claim to test:

```text
z is available early across adjective domains, but causal steering works best
later after the representation is carried/rotated.
```

For all 8 adjective pairs and all 42 layers:

1. Compute cumulative `R²(z)` by layer.
2. Compute fold-aware increment `R²(z)` by layer.
3. Compute `||primal_z||` by layer.
4. Compute layer-to-layer `cos(primal_z[L], primal_z[L-1])`.
5. Run steering for selected/all layers:
   - `primal_z`
   - Ridge `probe_z`
   - random null directions
6. Measure LD steering slope:

```math
\frac{\mathbb{E}[\mathrm{LD}(h+\alpha\hat d)-\mathrm{LD}(h-\alpha\hat d)]}{2\alpha}
```

Recommended alphas:

```text
alpha ∈ {-4, 0, +4}
```

If full all-layer steering is too expensive, run strategically:

```text
L = [0, 1, 3, 5, 7, 10, 13, 14, 17, 21, 25, 29, 33, 37, 41]
```

**Outputs**

```text
results/v12/layer_sweep_9b.json
figures/v12/layer_sweep_9b_combined.png
```

**Acceptance**

- Figure should be a v11/v12 replacement for `figures/v9/layer_sweep_combined.png`.
- It should clearly show early decodability and later steering potency.

---

## Priority 2 — Direction Red-Team: z vs Raw x vs Lexical/Adjective Concepts

**Question:** Is the z direction geometrically distinct from:

1. raw number magnitude directions;
2. raw adjective/lexical/domain directions;
3. zero-shot no-context directions?

This is broader than the old v8 zero-shot raw-x plot. We want to know whether
`primal_z` is distinct from both raw scalar magnitude and direct lexical
semantics like `tall`, `short`, `height`, `old`, `young`, synonyms, and short
sentence probes.

### Direction families

For each pair/model/layer:

**Context z directions**

```text
primal_z_context = mean(h | z > +1) - mean(h | z < -1)
probe_z_context  = Ridge(h -> z).coef_
```

**Context raw-x directions**

```text
primal_x_context = mean(h | x high) - mean(h | x low)
probe_x_context  = Ridge(h -> x).coef_
```

**Zero-shot raw-x directions**

No context list; only target value.

```text
w_x_zero = Ridge(h_zero_shot -> x).coef_
```

**Lexical/adjective/domain directions**

Construct prompts or token sets for direct lexical semantics. Examples:

```text
height high-word direction:      h("tall") - h("short")
height domain-word direction:    h("height") - h("baseline/control")
age high-word direction:         h("old") - h("young")
wealth high-word direction:      h("rich") - h("poor")
synonym direction:               mean(h(["tall", "large", "high"])) - mean(h(["short", "small", "low"]))
sentence direction:              h("This person is tall.") - h("This person is short.")
```

Use multiple lexical templates so the direction is not one-token idiosyncrasy:

```text
"The word is: tall"
"This adjective is tall"
"A description: tall"
"This person is tall."
"The property is height."
```

For each pair, compute several lexical directions:

```text
lex_high_low_word
lex_synonym_mean
lex_sentence_high_low
lex_domain_word
```

### Comparisons

Report cosine matrices:

```text
cos(primal_z_context, primal_x_context)
cos(probe_z_context, probe_x_context)
cos(primal_z_context, w_x_zero)
cos(probe_z_context, w_x_zero)
cos(primal_z_context, lex_high_low_word)
cos(primal_z_context, lex_synonym_mean)
cos(primal_z_context, lex_sentence_high_low)
cos(primal_z_context, lex_domain_word)
```

Add:

- bootstrap CIs across cells/seeds;
- random-direction null bands;
- pair-level and model-level aggregate plots.

### Steering subtest

If GPU budget allows, steer with:

```text
primal_z_context
primal_x_context
lex_high_low_word
lex_sentence_high_low
random null
```

Measure LD slope on dense context prompts. This distinguishes:

```text
linearly decodable direction
```

from:

```text
causally output-relevant direction
```

**Outputs**

```text
results/v12/direction_redteam_x_lexical_z.json
figures/v12/direction_redteam_cosines.png
figures/v12/direction_redteam_steering.png
```

**Acceptance**

- If `primal_z` is near-orthogonal to raw-x and lexical directions, that supports
  a distinct relative-standing representation.
- If lexical directions align strongly, we must soften the claim: z geometry may
  partially overlap with adjective/domain semantics.

---

## Priority 3 — Pure-x / Fixed-mu Cross-Pair Transfer Control

**Question:** Is cross-pair transfer really shared z, or just shared scalar
magnitude?

Current result:

```text
v11.5 multi-seed cross-pair transfer: 56/56 off-diagonal cells significant.
```

Remaining critic:

```text
Maybe a height direction transfers to weight because both encode generic
"bigger number / higher scalar value", not relative standing.
```

### Design

Rerun cross-pair steering on controlled target subsets:

1. **Fixed-mu / narrow-mu slice**
   - hold context mean approximately fixed;
   - vary x and z according to the grid.
2. **Fixed-x / varying-mu slice**
   - hold raw target value fixed;
   - vary context mean, hence z.
3. **Matched-z / varying-x slice**
   - hold z approximately fixed;
   - vary raw x.

For each source-target pair:

```text
source direction = primal_z[source]
target prompts   = controlled target subset
readout          = target LD
```

Compare slopes across:

```text
full grid
fixed-mu
fixed-x
matched-z
```

**Outputs**

```text
results/v12/pure_x_transfer_control.json
figures/v12/pure_x_transfer_control.png
```

**Acceptance**

- Strongest support for shared z: transfer remains on fixed-x / varying-mu
  slices and weakens on matched-z / varying-x slices.
- If transfer mostly follows raw x, shared-z claim must be softened.

---

## Priority 4 — SAE Feature Interpretation Audit Beyond Numeral Controls

**Question:** Are top z-SAE features abstract relative-standing features, or are
they lexical/domain features?

Current v11.5 result:

```text
Top z-features have high R²(z), near-zero R²(x), and near-zero R²(token magnitude).
```

Unresolved:

```text
Do they also fire on words like "tall", "short", "height", "old", "young"?
```

### For each top z-SAE feature

Use top features from:

```text
results/v11_5/<model>/sae_features_with_token_freq_control.json
```

Compute:

1. `R²(feature, z)`
2. `R²(feature, x)`
3. `R²(feature, token magnitude)`
4. `R²(feature, lexical high/low adjective indicator)`
5. `R²(feature, domain-word indicator)`
6. top-activating prompts/cells
7. top-activating lexical probes

Lexical probes:

```text
tall / short / height / high / low
old / young / age
heavy / light / weight
big / small / size
fast / slow / speed
rich / poor / wealth
expert / novice / experience
obese / thin / BMI
```

### Classification of features

Assign each feature a provisional label:

```text
pure-ish z:        high R²(z), low lexical/domain controls
lexical z-like:    high R²(z), also fires on adjective/domain words
raw numeric:       high R²(x) or token magnitude
mixed/polysemantic: no clean single interpretation
```

**Outputs**

```text
results/v12/sae_feature_lexical_audit.json
figures/v12/sae_feature_interpretation_examples.png
docs/v12_sae_feature_notes.md
```

**Acceptance**

- We need at least a few concrete feature examples before claiming SAE features
  implement relative standing.
- If most features are lexical/domain mixed, README should say SAE supports
  z-correlated sparse features but not pure relativity features.

---

## Priority 5 — PC2 / z² / Extremeness / x Audit

**Question:** Is the horseshoe geometry decomposing into signed relativity and
extremeness?

Hypothesis:

```text
PC1: signed relativity, z
PC2: extremeness, |z| or z²
```

But we must also include raw x:

```text
PCk may track z, z², |z|, x, x², or mixed geometry.
```

### Analysis

For each pair/model/canonical layer:

```text
R²(PC1, z)
R²(PC1, x)
R²(PC1, z²)
R²(PC1, x²)
R²(PC1, |z|)
R²(PC1, |x|)

R²(PC2, z)
R²(PC2, x)
R²(PC2, z²)
R²(PC2, x²)
R²(PC2, |z|)
R²(PC2, |x|)

same for PC3
```

Also compare direction vectors:

```text
cos(PC1, primal_z)
cos(PC1, primal_x)
cos(PC2, primal_z)
cos(PC2, primal_x)
```

This connects the scalar score analysis to the vector analysis.

**Outputs**

```text
results/v12/pc_extremeness_x_z_audit.json
figures/v12/pc_extremeness_x_z_grid.png
```

**Acceptance**

- If PC2 consistently tracks `z²` or `|z|`, we can describe a
  signed-relativity plus extremeness geometry.
- If PC2 tracks raw `x²` or domain-specific variance, treat it as a weaker
  pair-specific geometry note.

This is CPU-only if existing v11 activations are present.

---

## Priority 6 — Speed / Experience Exception Audit

**Question:** Why are speed and experience less shared?

Current observation:

```text
speed and experience are pair-specific exceptions to shared z steering.
```

Do not overclaim a human/non-human distinction yet. Instead test semantic
heterogeneity.

### Design

Inspect and optionally split templates:

```text
speed:
  person running speed
  vehicle speed
  process/computation speed

experience:
  worker years of experience
  doctor/programmer/teacher domain experience
  novice/expert skill framing
```

Run a small grid or reuse v11 where possible:

```text
10 x-values x 10 z-values x 5 seeds
```

Compare:

```text
PC1/PC2 geometry
primal_z direction cosine to shared direction
shared/within steering ratio
SAE top-feature overlap
```

**Outputs**

```text
results/v12/speed_experience_exception_audit.json
figures/v12/speed_experience_exception_audit.png
```

**Acceptance**

- If subdomains become cleaner, current exception is due to semantic mixture.
- If still pair-specific, speed/experience may genuinely use different relative
  representations.

---

## Priority 7 — Direct-Sign Cleanup, Only If Time

This is low priority. Keep it as measurement hygiene, not a main relativity
claim.

If rerun:

1. forced-choice prompts only;
2. top-k validation required;
3. report accuracy and relativity ratio together;
4. no open-ended `"This number is ___"` logit scoring.

**Outputs**

```text
results/v12/direct_sign_cleanup.json
figures/v12/direct_sign_cleanup.png
```

---

## Recommended Run Order

1. **Layer sweep 9B** — most important README/paper figure.
2. **Direction red-team: z vs x vs lexical semantics** — decides whether the
   zero-shot/lexical orthogonality story can be promoted.
3. **Pure-x / fixed-mu transfer control** — directly defends shared-z transfer.
4. **SAE lexical audit** — defends SAE interpretation.
5. **PC2 / z² / x audit** — CPU-friendly and potentially interesting.
6. **Speed / experience audit** — useful if time.
7. **Direct-sign cleanup** — only if spare time.

---

## Final Deliverables

Minimum success:

```text
figures/v12/layer_sweep_9b_combined.png
figures/v12/direction_redteam_cosines.png
figures/v12/pure_x_transfer_control.png
figures/v12/sae_feature_interpretation_examples.png
figures/v12/pc_extremeness_x_z_grid.png

results/v12/layer_sweep_9b.json
results/v12/direction_redteam_x_lexical_z.json
results/v12/pure_x_transfer_control.json
results/v12/sae_feature_lexical_audit.json
results/v12/pc_extremeness_x_z_audit.json
```

Paper/README decision after v12:

```text
Promote only claims that survive:
- x and lexical direction red-team
- pure-x transfer control
- SAE lexical audit
```

