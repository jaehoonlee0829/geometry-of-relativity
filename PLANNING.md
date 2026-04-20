# PLANNING.md — Spec v2 (revised Apr 20 2026, evening)

This file is the north star. If BUILDING.md or TODO.md drifts from this, STOP and reconcile.

## Scientific claim (the one sentence to defend)

> In open-weight transformer LMs, linear probe covectors for **relative** gradable adjectives ("tall", "short") align with the gradient of a *context-normalized Z-score*, while probes for adjectives closer to the absolute end of the spectrum ("obese") partially align with the gradient of the *raw attribute value*. We quantify this via a decomposition `w_adj ≈ α·w_z + β·w_x` and compare three geometric frameworks — Euclidean cosine, covariance-adjusted (Σ⁻¹·w), and Fisher-pullback (F⁻¹·w) — showing that the Fisher metric most cleanly separates relative from absolute adjectives.

## Primary artifacts

### Hero figure 1: α/β decomposition across the relativity spectrum
- X-axis: Z-score z = (x − μ) / σ
- Y-axis: probe logit (w^T h + b)
- Separate panels/curves for different raw values x (to decorrelate x from z)
- Two adjective pairs: tall/short (height), rich/poor (wealth)

### Hero figure 2: metric comparison
- Three columns: Euclidean, Σ⁻¹, F⁻¹
- Y-axis: cosine alignment of (adjusted) probe covector with w_z vs w_x
- Shows whether Fisher is necessary or Euclidean suffices

## Models

1. `google/gemma-4-31B` — primary (31B dense, fits on 1× H100 80GB in bf16)
2. `google/gemma-2-9b` — secondary / replication (fits on 1× A100 40GB)

Compute: rent H100 (or 2× H100 for headroom) via Vast.ai or similar.

## Domains and adjective pairs

### Domain 1: Height (tall/short)
- **Target values x**: 150, 155, 160, 165, 170, 175, 180 cm (7 values)
- **Context means μ**: 145, 150, 155, 160, 165, 170, 175, 180, 185 cm (9 values)
- **σ**: 10 cm (fixed)
- **Z-scores**: range from −3.5 to +3.5, decorrelated from x

### Domain 2: Wealth (rich/poor)
- **Target values x**: $20K, $50K, $100K, $250K, $500K, $1M, $5M (7 values, log-spaced)
- **Context means μ**: $15K, $30K, $75K, $150K, $300K, $750K, $2M (7 values, log-spaced)
- **σ**: factor of 2 (i.e., in a group with μ=$100K, most people earn $50K–$200K)
- Z-scores computed in log-space: z = (log(x) − log(μ)) / log(σ_factor)

### Follow-up domains (only if primary is clean)
- BMI / obese (hybrid adjective — revisit v1 behavioral findings)
- Age / young-old
- Temperature / hot-cold

## Prompt design

### Two context types (both run, compared)

**Implicit context** (primary — ecologically valid):
List 15 people's attribute values sampled from N(μ, σ), then present the target:
```
Person 1: 148 cm
Person 2: 153 cm
...
Person 15: 146 cm
Person 16: 165 cm is [considered]
```
Fixed random seed per (μ, σ) condition so the sample is deterministic across trials.

**Explicit context** (control):
```
In a group where heights cluster around 150 cm (σ ≈ 10 cm), a person who is 165 cm is [considered]
```

### Two prompt frames (both run, compared)

| Frame | Last tokens before target | Hypothesis |
|-------|--------------------------|------------|
| **"is ___"** | `... 165 cm is` | More objective/factual completion |
| **"is considered ___"** | `... 165 cm is considered` | More subjective/context-dependent completion |

Prediction: "is considered" elicits stronger context sensitivity (higher α relative to β).

### Activation extraction point

Extract hidden state at the **last input token** before the adjective prediction:
- "is ___" frame → activation at "is"
- "is considered ___" frame → activation at "considered"

## Experiment matrix — Axis 1 (HERO)

Per domain (height):
- 7 target values × 9 context means × 2 context types × 2 prompt frames = **252 prompts**
- × 4 layers × 2 models = **2016 activation vectors** per domain
- × 3 metrics (Euclidean, Σ⁻¹, F⁻¹) × 2 reference directions (w_x, w_z) = **12,096 cosine measurements** per domain

Per domain (wealth): similar scale.

Total: ~500 prompts, ~4000 activations, ~24,000 measurements. Very manageable.

## Three geometric comparison methods (run all in parallel)

| Method | Formula | What it captures |
|--------|---------|-----------------|
| **Euclidean** | cos(w_adj, w_z) | Raw direction alignment |
| **Covariance-adjusted** | cos(Σ⁻¹·w_adj, w_z) under Σ inner product | Accounts for activation variance structure |
| **Fisher-pullback** | cos(F⁻¹·w_adj, w_z) under F inner product | Accounts for output-distribution sensitivity |

If Euclidean already separates relative from absolute → simpler result, Fisher is bonus.
If only Fisher separates them → methodological contribution, Fisher is necessary.

## Probe training

Three probes trained on the same activations:
1. **w_adj**: logistic regression, h → "tall" vs "short" (or "rich" vs "poor")
2. **w_x**: linear regression, h → raw target value x
3. **w_z**: linear regression, h → Z-score z = (x − μ) / σ

The α/β decomposition: regress w_adj onto [w_x, w_z] to get mixing coefficients.

## Layers

Early, mid, late, final — exact indices per model in `src/activation_extract.py::LAYER_INDICES`.

## What is OUT of scope

- Fine-tuning any model
- Multi-token adjectives ("extremely tall")
- Reasoning models (o1-style)
- RL / RLHF analysis
- Steering interventions (deferred to follow-up)

## Compute budget

- Local CPU: prompt generation, probe training, metric computation, plotting
- GPU (H100 80GB, rented in 2–4h bursts): model forward passes, activation caching
- Total GPU hours budget: **≤30 hours** across the full project

### GPU session plan (single session, ~1h wall clock)

Run both models in one H100 session:
1. Setup + pip install: ~15 min
2. Download Gemma 4 31B weights (~62GB): ~10 min
3. Extract activations for both domains (~500 prompts × 4 layers): ~20–30 min
4. Download Gemma 2 9B weights (~18GB): ~3 min
5. Extract activations on Gemma 2 9B (replication): ~10 min
6. Download .npz results to local (<1 min, ~30MB per model)

Total: **~1 hour** for both models, well within a single Vast.ai burst.

## Deadlines

- NeurIPS 2026 abstract: **May 4** (Mon)
- NeurIPS 2026 full paper: **May 6** (Wed)
- arXiv preprint (self-imposed): **May 7** (Thu)
- ICML 2026 MI Workshop: **May 8** (Fri, AOE)

## Hard pivot trigger

**Day 5 (Apr 24):** if zero clean signal (probe shift < 1σ between low-context and high-context at mid layer of Gemma 4 31B), scope down to a 4-page short paper or defer to ICLR 2027.

## Success criteria for workshop paper

1. Tall/short probe α >> β (relative), demonstrated across at least 3 context means
2. Comparison of Euclidean vs Σ⁻¹ vs F⁻¹ shows meaningful differences
3. Second domain (wealth: rich/poor) replicates the pattern
4. "is" vs "is considered" prompt frame shows predicted difference in context sensitivity
