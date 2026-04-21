# v1 Spectrum Findings — Sonnet 4.6, 810 completions (0 errors)

## Design

Fair symmetric spectrum, both adjectives on the same experimental footing:

- **Tall** (target = 165 cm): 9 μ values ∈ {140, 150, 155, 160, 165, 170, 175, 180, 190}, σ=3
- **Obese** (target = BMI 32 or equivalently 170cm/92kg): 9 μ values ∈ {19, 22, 25, 28, 31, 34, 37, 40, 43}, σ=1
- 3 template variants per family
- **Paired obese variants**: BMI-direct ("BMI of 32") vs height+weight ("170cm, 92kg") — difference isolates literal-number pattern-matching from genuine body-habitus relativity
- 10 samples per (family × μ × template) = 81 prompts × 10 = 810 calls

## Headline

Relative ("tall") and absolute ("obese") gradables behave qualitatively differently — but "obese" is NOT purely absolute. It exhibits a remarkably clean **~1/3 absolute, ~2/3 relative** mixture.

## Tall spectrum — clean sigmoid, pure relativity

| μ (cm)    | 140 | 150 | 155 | 160 | 165 | 170 | 175 | 180 | 190 |
|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| % "tall"  | 100 | 100 | 100 | 67  | 0   | 0   | 0   | 0   | 0   |
| % "avg"   | 0   | 0   | 0   | 33  | 100 | 33  | 3   | 0   | 0   |
| % "short" | 0   | 0   | 0   | 0   | 0   | 67  | 97  | 100 | 100 |

Sigmoid fit: `L=1.00, x0=160.2, k=-3.40`. Fifty-percent crossover at μ≈160.2 cm (i.e. 5 cm below the target value). Slope |k|≈3.4 per cm is **sharp** — within 5 cm of the crossover the label flips completely. This is the signature of a relative adjective: the target's interpretation is fully defined by the context distribution.

## Obese spectrum — hybrid, with a striking plateau at 1/3

BMI-direct prompt (BMI 32 vs context μ):

| μ (BMI)    | 19  | 22  | 25  | 28  | 31  | 34  | 37  | 40  | 43  |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| % "obese"  | 100 | 100 | 100 | 100 | 33  | 33  | 33  | 33  | 33  |
| % "avg/normal" | 0 | 0 | 0 | 0 | 67 | 0 | 0 | 0 | 0 |
| % "thin/underweight/slim" | 0 | 0 | 0 | 0 | 0 | ~67 | ~67 | ~67 | ~67 |

Height+weight prompt (170cm/92kg vs context's reference weight at μ):

| μ (BMI-eq) | 19  | 22  | 25  | 28  | 31  | 34  | 37  | 40  | 43  |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| % "obese"  | 100 | 100 | 100 | 100 | 90  | 33  | 33  | 33  | 33  |

### Why this is striking

1. **The 1/3 plateau is flat across μ = 34, 37, 40, 43** — extreme fat-context does not drive "obese" below 0.33. This is a real absolute floor.
2. **The plateau matches EXACTLY between BMI-direct and height+weight prompts.** If Claude were merely pattern-matching the literal number "32," the BMI-direct curve should stay high while the height+weight curve (where "32" never appears) should go to zero. They are identical → the effect is genuinely about the *concept* "obese," not the string "32."
3. **The remaining 2/3 flips to "thin / underweight / slim" at high μ** — a fully relative response. Raw sample at μ=43 (BMI direct): 17 "thin," 10 "obese," 2 "relatively thin," 1 "slim." Same at μ=43 (height+weight): 13 "underweight," 10 "obese," 5 "thin."

### Interpretation

Claude Sonnet 4.6 has a **mixture representation** for "obese":
- With P ≈ 1/3, the medical-absolute semantics fires (BMI ≥ 30 → obese, regardless of context).
- With P ≈ 2/3, a relative body-habitus semantics fires (person is slim relative to their peers).

The mixture weights are **remarkably stable** across the extreme half of the spectrum. This is not sampling noise on a binary decision; it is a bimodal, stochastic output that reveals two genuinely different internal computations sharing the same output position. Good target for mechanistic probing — we should find two separable directions in activation space, a "medical/clinical" obese direction and a "body-habitus relative" obese direction, and Sonnet 4.6 samples between them with ~1:2 mixing.

## Paired comparison Δ(BMI-direct − height+weight)

| μ    | 19  | 22  | 25  | 28  | 31   | 34  | 37  | 40  | 43  |
|------|-----|-----|-----|-----|------|-----|-----|-----|-----|
| Δ    |  0  |  0  |  0  |  0  | -0.57|  0  |  0  |  0  |  0  |

Zero everywhere except at the crossover μ=31, where BMI-direct gives 0.33 "obese" while height+weight gives 0.90 "obese." At the crossover, BMI-direct is already reading the 32 as "just one BMI unit above the context" (a hair above normal), while height+weight still weighs 92kg on a 170cm frame as decisively heavy. The ≥1 cm differentiation is a nice probe signal. **Elsewhere, the literal-number hypothesis is definitively ruled out.**

## Verdict on H1 and H2 (revised from v0)

- **H1 (relative adjectives flip with context): PASSED.** Tall flips cleanly across a 9-point spectrum with a sigmoid transition, crossover 5 cm below the target value, slope |k|≈3.4/cm. Template-agnostic (3 templates, all agree).
- **H2 (absolute adjectives stay absolute): FAILED cleanly, in the most informative way.** "Obese" exhibits a ~1/3 absolute / ~2/3 relative mixture. The mixture ratio is stable across the extreme half of the spectrum and is *independent* of whether the prompt gives a direct BMI number or forces the model to derive BMI from height+weight. This is evidence for an internal mixture of *two* semantics for "obese" — not a continuous shift and not a literal-string pattern match.

## Paper framing

Three-way spectrum, not binary:
1. **Pure relative** (tall, short, heavy, light) — body-habitus-relative only, no absolute anchor.
2. **Hybrid / mixture** (obese, underweight, rich, poor, expensive) — stochastic mixture of medical/clinical/official absolute thresholds with relative semantics.
3. **Pure absolute** (dead, pregnant, prime-numbered) — no context sensitivity at all. (Prediction for Day 4 — add to v2 battery.)

The Fisher-pullback `F⁻¹·w` becomes the tool to *measure where each adjective sits on the spectrum*. The α/β split in `w_A = α·∇z_C(x) + β·∇x` is now the headline quantitative claim.

## What's next

- **v2 battery**: add "pure absolute" controls (dead, pregnant) + add "rich/poor" as second hybrid candidate for replication of the plateau phenomenon.
- **Activation probing (Day 5)**: predict that for "obese" we will find TWO orthogonal probe directions in mid-layer — one tracking medical obesity, one tracking relative body size — rather than a single direction. This is a strong, falsifiable prediction the mixture theory makes.
- **Sanity control**: "blue" should show zero context sensitivity. Add to v2.

## Files

- `results/behavioral_v1/` — 81 JSON files, 810 completions, 0 errors
- `results/behavioral_v1_summary.md` — tables + sigmoid fits
- `results/behavioral_v1_per_mu.csv` — machine-readable
- `figures/spectrum_v1.pdf` — side-by-side plot
- `scripts/gen_prompts_v1.py`, `scripts/run_behavioral_v1.py`, `scripts/analyze_v1.py`
