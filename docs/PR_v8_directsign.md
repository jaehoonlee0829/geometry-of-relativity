# PR: `exp/v8-direct-sign` — test posneg prompt-sensitivity + replot on clean grid + cross-template

**Motivation** (per `docs/NEXT_GPU_SESSION_v8.md`): v7 claimed pos/neg math shows R=0.47 context-relativity. Red-team worry: the prompt "This number is ___" is ambiguous — model may complete with relative-position adjectives ("low", "smallest") rather than sign. v8 tests whether explicit sign-classification prompts eliminate the context effect. Also: replot v4/v6 activation-geometry figures on Grid B (clean) data; test whether cross-pair transfer is template-dependent.

## Results (after 3-critic consensus)

### Priority 1+2 — direct sign classification, 4 variants

Critical finding from self-triggered top-K token audit:

| variant | scoring tokens | high in top-10 | low in top-10 | R | valid? |
|---|---|---|---|---|---|
| orig | positive/negative | 16% | 54% | 0.47 | noisy (most "positive" scoring is tail) |
| compared | above/below | **0%** | **0%** | 0.24 | **invalid** (scoring tokens never in top-10) |
| relative | higher/lower | **0%** | **0%** | 0.47 | **invalid** |
| **forced_qa** | Above/Below | 100% | 100% | **0.31** | **valid — only clean measurement** |

Most-common top-1 tokens per variant:
- `orig`: " the" (397), " not" (224), " negative" (69)
- `compared`: " " bare space (342), " negative" (332), " positive" (37)
- `relative`: " " bare space (706), " the" (36)
- `forced_qa`: " Below" (361), " Above" (361)

**Refined posneg claim**: with a forced classification prompt, pos/neg math shows **R = 0.31, accuracy = 0.95** — real but smaller residual context effect than v6/v7's R=0.47. The earlier headline was inflated because "positive"/"negative" aren't the model's natural completions after "This number is".

### Priority 3 — replot activation geometry on Grid B

PCA horseshoe: **PC1 does NOT universally track z**. On clean grid, 8 pairs split:
- **z-dominant** (PC1 ≈ z-axis): height (R²=0.95), weight (0.97), wealth (0.60), bmi_abs (0.69)
- **x-dominant** (PC1 ≈ x-axis): age (0.87), size (0.67), speed (0.88), experience (0.62)

The v4 "horseshoe proves PC1 = z-axis" claim was driven by the Grid A confound. On clean grid, the model represents different pairs with different principal variance directions.

Other Grid B replots:
- Meta-direction SVD top shared variance: **32.6%** (Grid A: 41.6% — confound inflated ~25%)
- Cross-pair PC1 cosine: mean |cos| off-diag **0.19** (Grid A: 0.32 — confound inflated ~40%)
- Zero-shot × implicit cosine: all |cos| ≤ 0.05 (≈ chance floor 0.020 — v5 orthogonality claim confirmed on clean grid)

### Priority 4 — cross-template transfer (height)

Extracted height activations under two prompt templates:
- A: "This person is"  (standard)
- B: "Among the individuals listed, the one measuring X cm would be described as"

Measured:
```
cos(primal_z_A, primal_z_B)          = +0.727
slope(primal_z_B → B)                = +0.152   (self-transfer)
slope(primal_z_A → B)                = +0.147   (cross-template transfer)
slope(random dirs → B)               =  0.003   (null, mean of 3)

cross / self     = 0.968  (97%)
cross / random   = 43.9×
```

**97% cross-template transfer, 44× above null.** The primal_z direction is template-invariant at layer 32. Tested on 1 pair (height); replicating on other pairs is future work.

Caveat: Template A scoring ("tall"/"short" in top-10 for 34%/1% of prompts) is noisy; Template B scoring is clean (99%/53%). The **cross-transfer was measured with steering EVALUATED on Template B's clean scoring** — so the result is valid regardless of A's scoring quality.

## What 3 critic agents flagged + how this PR addresses each

**Statistical critic**:
- SE on R ≈ 0.005-0.011; R=0.31 for forced_qa is 28σ from zero (real effect)
- PC1 sample-size concern: 25 cell means in d=2560 → rank-24 estimate. Flagged in PR body.
- Multiple-comparisons: 500+ statistics across v4-v8 with no FWER. Headline effects survive any BH correction; mid-tier claims in [0.10, 0.22] range don't.
- **Biggest flag**: top-K tokens check. **Addressed in `top10_diagnostic.json` + commit `a425da4`** — only forced_qa is a valid R measurement.

**Alt-interp critic**:
- PC1 orientation might just track x-variance vs z-variance ratio, not semantic encoding. Correlation check: `corr(R²(PC1~x), x_range/σ) = -0.42` — weak correlation, some support but not overwhelming.
- Cross-template 97% at layer 32 (76% depth) could be late-integration artifact, not semantic. Future: repeat at earlier layers.
- v8 overall: R varies 0.24-0.47 across 4 prompts for pos/neg — suggests R measures (concept, prompt) jointly, not pure concept-relativity.

**Implementation critic**:
- P1 "compared"/"relative" variants have scoring tokens NOT in top-10 → R values are noise. **Confirmed & addressed** above.
- P4 Template B might also have noisy scoring. **Checked: "tall" in 99/100, "short" in 53/100 for Template B — valid.**
- PC1 sign-alignment for x-dominant pairs is noise-driven (but R² is symmetric, so per-pair R² is unaffected; only cross-pair signed cos is noisy).
- Fisher/Park/cross-template computations verified correct. No critical bugs.

## Refined paper claim after v8

**Previously** (v7): "mathematical pos/neg shows 47% context-relativity; all concepts partially relative."

**Now** (v8, after forced_qa): "mathematical pos/neg shows **31% context-relativity** with 95% accuracy under explicit sign classification. Ambiguous prompts inflate R measurement by ~50% due to scoring on low-probability tokens. The existence of context-dependence for sign classification holds, but at a smaller magnitude than earlier prompts suggested."

**Added**: "PC1 in activation space splits pairs into z-dominant vs x-dominant classes — the model's internal representation is heterogeneous. Cross-template transfer of primal_z at 97% (height, n=1 pair) suggests the direction is semantic, not syntactic."

## Data

- Git: 7 scripts + 7 JSONs + 7 figures. +5 MB.
- HF: v8 activations (cross-template height) to be uploaded; v8_direct_sign logits.jsonls uploaded.
- Local Vast: all intermediates.

## Reproducibility

```bash
git checkout exp/v8-direct-sign
cd /workspace/repo2
HF_TOKEN=... python scripts/fetch_from_hf.py  # Grid B activations
python scripts/vast_remote/exp_v8_direct_sign.py      # ~3 min GPU
python scripts/vast_remote/exp_v8_replot_gridB.py     # <1 min CPU
python scripts/vast_remote/exp_v8_cross_template.py   # ~5 min GPU
```
