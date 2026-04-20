# v1 Spectrum Analysis — Sonnet 4.6, 81 prompts × 10 samples

## Key question

Do relative ('tall') and absolute ('obese') gradable adjectives respond differently
to a spectrum of context means μ? Fair symmetric design: both adjectives tested across
9 μ values symmetric around the target, at narrow σ.


## TALL / target = 165 cm

| μ | fraction `tall` | fraction average/normal | fraction other | n |
|---|---:|---:|---:|---:|
| 140 | 1.00 | 0.00 | 0.00 | 30 |
| 150 | 1.00 | 0.00 | 0.00 | 30 |
| 155 | 1.00 | 0.00 | 0.00 | 30 |
| 160 | 0.67 | 0.33 | 0.00 | 30 |
| 165 | 0.00 | 1.00 | 0.00 | 30 |
| 170 | 0.00 | 0.33 | 0.00 | 30 |
| 175 | 0.00 | 0.03 | 0.00 | 30 |
| 180 | 0.00 | 0.00 | 0.00 | 30 |
| 190 | 0.00 | 0.00 | 0.00 | 30 |

Sigmoid fit: L=1.000, x0=160.20 (transition center), k=-3.395 (slope), b=-0.000
Interpretation: 50% transition at μ≈160.2. Slope magnitude |k|=3.39.

## OBESE (BMI-direct) / target = BMI 32

| μ | fraction `obese` | fraction average/normal | fraction other | n |
|---|---:|---:|---:|---:|
| 19 | 1.00 | 0.00 | 0.00 | 30 |
| 22 | 1.00 | 0.00 | 0.00 | 30 |
| 25 | 1.00 | 0.00 | 0.00 | 30 |
| 28 | 1.00 | 0.00 | 0.00 | 30 |
| 31 | 0.33 | 0.67 | 0.00 | 30 |
| 34 | 0.33 | 0.00 | 0.30 | 30 |
| 37 | 0.33 | 0.00 | 0.23 | 30 |
| 40 | 0.33 | 0.00 | 0.13 | 30 |
| 43 | 0.33 | 0.00 | 0.03 | 30 |

Sigmoid fit: L=0.667, x0=29.50 (transition center), k=-15.167 (slope), b=0.333
Interpretation: 50% transition at μ≈29.5. Slope magnitude |k|=15.17.

## OBESE (height+weight, BMI derivation required) / target = 170cm/92kg

| μ | fraction `obese` | fraction average/normal | fraction other | n |
|---|---:|---:|---:|---:|
| 19 | 1.00 | 0.00 | 0.00 | 30 |
| 22 | 1.00 | 0.00 | 0.00 | 30 |
| 25 | 1.00 | 0.00 | 0.00 | 30 |
| 28 | 1.00 | 0.00 | 0.00 | 30 |
| 31 | 0.90 | 0.10 | 0.00 | 30 |
| 34 | 0.33 | 0.00 | 0.37 | 30 |
| 37 | 0.33 | 0.00 | 0.00 | 30 |
| 40 | 0.33 | 0.00 | 0.00 | 30 |
| 43 | 0.33 | 0.00 | 0.03 | 30 |

## Paired comparison: BMI-direct vs height+weight (obese)

If 'obese' is anchored to the literal number 32 (pattern-match), the BMI-direct curve stays high.
If 'obese' is anchored to body habitus relative to context, both curves track each other.
The difference between the two curves at each μ measures the contribution of literal-numeric-anchoring.

| μ | obese% BMI-direct | obese% height+weight | Δ (bmi − hw) |
|---|---:|---:|---:|
| 19 | 1.00 | 1.00 | +0.00 |
| 22 | 1.00 | 1.00 | +0.00 |
| 25 | 1.00 | 1.00 | +0.00 |
| 28 | 1.00 | 1.00 | +0.00 |
| 31 | 0.33 | 0.90 | -0.57 |
| 34 | 0.33 | 0.33 | +0.00 |
| 37 | 0.33 | 0.33 | +0.00 |
| 40 | 0.33 | 0.33 | +0.00 |
| 43 | 0.33 | 0.33 | +0.00 |

## Verdict on H2 (revised, fair-design test)

- fraction 'obese' at μ=40 (BMI-direct): 0.33
- fraction 'obese' at μ=43 (BMI-direct): 0.33
If obese were fully absolute, both should be ≈1.00 (BMI=32 is always medically obese).
If obese is fully relative, both should be ≈0.00 (BMI=32 is slim relative to μ=43).