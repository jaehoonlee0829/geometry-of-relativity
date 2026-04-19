# PLANNING.md — Frozen spec (v0, Apr 20 2026)

This file is the north star. If BUILDING.md or TODO.md drifts from this, STOP and reconcile.

## Scientific claim (the one sentence to defend)

> In small open-weight transformer LMs, linear probe covectors for **relative** gradable adjectives ("tall", "short") align with the Fisher-pullback gradient of a *context-normalized* Z-score, while probes for **absolute** adjectives ("obese", defined by BMI cutoff) align with the Fisher-pullback gradient of the *raw attribute value*. This asymmetry explains why the same input ("165 cm") produces flipped adjective completions across contexts, and predicts that OOD inputs (10 cm, BMI=3) project to low-Fisher-curvature regions of the activation manifold.

## Primary artifact

One hero figure:
- X-axis: context mean μ (150 cm narrow-low → 180 cm narrow-high)
- Y-axis: Fisher-normalized cosine between probe covector and one of {∇z_C, ∇x}
- Two curves: `tall`/`short` (relative, should track ∇z_C) vs `obese` (absolute, should track ∇x)
- Layers: early / mid / late / final (faceted)

## Models (final list, no expansion without explicit re-spec)

1. `google/gemma-2-2b` — primary
2. `meta-llama/Llama-3.2-3B` — replication

Reasons: both are small enough for single-GPU (A100 40GB or 2× RTX 4090), have clean HuggingFace loaders, have published activation-extraction recipes, and differ in architecture (Gemma2 uses hybrid attention, Llama3 uses standard GQA) so a convergent finding generalizes.

## Experiment matrix — Axis 1 (HERO, do first)

2 models × 2 adjective classes (tall/short vs obese) × 4 contexts × 4 layers ≈ **64 cells**

### Contexts

| ID | Description | μ (cm or BMI) | σ |
|----|-------------|---------------|---|
| narrow_low | tight, low-centered | 150 cm / BMI 19 | 3 / 1 |
| narrow_high | tight, high-centered | 180 cm / BMI 33 | 3 / 1 |
| wide_symmetric | general adult pop | 165 cm / BMI 27 | 10 / 5 |
| ood_contaminated | includes impossible values | mixed | n/a |

### Layers

Early, mid, late, final — exact indices per model written in `src/activation_extract.py::LAYER_INDICES`.

## Axes 2–4 (ADDITIVE, only if Axis 1 is clean by Day 10)

- **Axis 2**: additional adjective pairs (heavy/light for weight, young/old for age — relative; "underweight" — absolute)
- **Axis 3**: steering interventions (Dual Steering via `F⁻¹·w` vs naive `w` vs `Σ⁻¹·w`)
- **Axis 4**: cross-model stitch (does Llama-3.2-3B's `w_tall` transfer to Gemma-2-2b via a learned affine?)

## What is OUT of scope (do not touch unless paper needs it)

- Non-human domains (animals, buildings)
- Fine-tuning any model
- Multi-token adjectives ("extremely tall" — requires different probe design)
- Reasoning models (o1-style)
- RL / RLHF analysis

## Compute budget

- Local CPU: prompt generation, probe training, Fisher computation (matrix ops in float64), plotting
- Vast.ai GPU (rented in 2-4h bursts): HuggingFace model forward passes, activation caching
- Total GPU hours budget: **≤20 hours** across the full project (1h/day × 20 days)

## Deadlines

- NeurIPS 2026 abstract: **May 4** (Tue)
- NeurIPS 2026 full paper: **May 6** (Thu)
- arXiv preprint (self-imposed): **May 7** (Fri)
- ICML 2026 MI Workshop: **May 8** (Fri, AOE)

## Hard pivot trigger

**Day 5 (Apr 24):** if zero clean signal (H1 probe shift < 1σ between narrow-low and narrow-high at mid layer of Gemma-2-2b), scope down to a 4-page short paper or defer to ICLR 2027.

## Success criteria for workshop paper

1. H1 holds for at least one model at at least one layer, p < 0.05 with n=50+ (x,ctx) pairs
2. H2 holds as a paired contrast (obese probe shift is ≥5× smaller than tall probe shift across contexts)
3. The Fisher-pullback framing is demonstrated (at least one plot showing the alignment)
4. OOD prediction (H3) checked as a bonus if time permits
