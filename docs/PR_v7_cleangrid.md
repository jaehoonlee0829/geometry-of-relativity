# PR: `exp/v7-clean-grid` — fix the (x, μ) → z confound + rerun v4-v6 cleanly

**Motivation** (per `docs/NEXT_GPU_SESSION_v7.md`): v4-v6 used an (x, μ) grid
where z = (x − μ)/σ was derived. Because we only sampled 5 x × 5 μ cells,
the z-bins |z|>1 contained biased x means (corr(x, z) ∈ [+0.58, +0.86]).
Every "z-direction" computed via diff-of-means or PCA was contaminated.

**Fix**: Grid B iterates (x, z) directly and derives μ = x − σ·z. Clipping
implausible μ drops 0-3 cells per pair. Grid B achieves corr(x, z) = 0.00
for 5/8 pairs and 0.09-0.20 for 3 pairs with dropped cells (size, speed,
experience). 5 pairs fully clean; 3 pairs ~90-99% clean by R².

## What v7 rules INTO and OUT of the v5/v6 story

### CONFIRMED robust (same finding on clean grid)

| v5/v6 claim | clean-grid result |
|---|---|
| **primal_z steers much stronger than probe_z** | **18× on clean grid** (v6 was 13×). Primal_z slopes ~0.10-0.16; probe_z ~0.01. Cleaner story: primal_z effect ≈ 32σ above null; probe_z effect ≈ 2σ above null. |
| Meta w₁ causally steers all 8 pairs | YES — same sign, similar magnitudes (−0.03 to −0.13). |
| cos(w_z, w_ld) ≈ 0 at late layer | YES — Euclidean avg 0.06 on clean probes. |
| F⁻¹ ≈ Euclidean (H4 refuted in tested regime) | YES — now also at softmax-entropy-p10 activations. Fisher gives ~1.5× amplification at most; absolute values 0.02-0.14. |
| **posneg (pos/neg math) shows ~0.4 relativity ratio** | YES — **R=0.47 on Grid B** (v6: 0.42). The "all concepts show partial relativity" claim holds on clean grid. |

### OVERTURNED (v5/v6 finding was an artifact of the confound)

| v5/v6 claim | clean-grid fact |
|---|---|
| "primal_z ≈ primal_x are the same direction" | **Mean signed cos = 0.03; mean \|cos\| = 0.22.** Not zero, but vastly lower than 0.91. Per-pair: size=−0.59 (large negative), bmi_abs=+0.48 (still substantial) — not every pair fully cleans up. |
| "INLP barely reduces R²(z) (v4: Δ=0.04)" | **R²(z) drops 0.29-0.51 across 8 pairs on clean grid.** v4 finding was driven by x↔z pathway leakage. Random-direction null confirms direction-specific removal (flat baseline). |
| "PC1 ≈ primal_z ≈ primal_x" | PC1 aligned 0.68 with primal_z, 0.11 with primal_x. PC1 is mostly but not purely the z-direction. Per-pair PC1 sign-stability across grids varies: cos(PC1_A, PC1_B) ranges from **−0.35 (age, sign flip!)** to +0.98 (height). |

### NEW findings in v7

| Finding | Magnitude |
|---|---|
| **Cross-pair transfer with primal_z_clean** | Diagonal slopes avg 0.126, off-diag 0.051 (40% of diagonal), null 0.009. 24σ separation from null. Real. |
| **BUT transfer is body-cluster-dominated** | Within {height, weight, size, bmi_abs}: off-diag 0.080 (63% of own-pair). Outside: 0.045 (36%). Cross-cluster: 0.042 (33%). So "universal substrate" is too strong — it's a **semantic-cluster substrate with body-attributes as the strongest link**. |
| **PC2 is genuinely causal for some pairs** | size +0.139, experience +0.140, speed −0.104, wealth +0.096. Not purely geometric. |

## What the 3 critic agents flagged (and what was done)

All 3 agents reviewed v7 independently.

### Stats critic
- **"experience has residual corr(x,z) = 0.203"** — asymmetric cell-dropping (3 cells at high z) introduced a new confound. Size has +6/55 (11%) residual, experience +3.8/24 (16%). FLAGGED in the commits.
- **"3 Fisher regimes are really 1 regime"** — cell-means, |ld|-p90, entropy-p10 all sample the same ~3.5-5.4 nat entropy range. H4 refutation is scope-limited to moderate-entropy activations. ADOPTED in the FINDINGS framing.
- **"INLP drop is direction-specific, not rank artifact"** — random-null trajectory stays flat, so the drop is real. This was actually a validation of the v7 result.
- **"18× ratio is arithmetically right but misleading"** — probe_z slopes are near-noise (~0.01); dividing a strong signal by near-noise inflates ratios. Reframed as "primal_z ≈ 32σ above null; probe_z ≈ 2σ above null".

### Alt-interp critic
- **"cos=0.033 is mean-of-signed-values"** — the mean |cos| is 0.221. CORRECTED in this PR body.
- **"posneg wasn't rerun on Grid B"** — the strongest "v6 is all confound" critique for absolute controls was untested. ADDRESSED: `exp_v7_posneg_clean.py` run as addendum; posneg R=0.47 on clean grid (slight increase from v6's 0.42).
- **"Cell-dropping creates new mini-confound"** — verified for size (+11% x residual at z>+1 vs z<-1) and experience (+16%). Acknowledged in this PR body.
- **"Transfer matrix has semantic-cluster structure"** — validated. Body-attribute cluster transfers at ~60% of own-pair; across clusters ~35%. The "universal substrate" framing is too strong.
- **"PC1 cross-grid stability is heterogeneous"** — age PC1 sign flips between grids (cos = −0.35). FLAGGED.

### Implementation critic
- No critical bugs found.
- Minor: PC2 sign-flip propagates from PC1 (arbitrary convention, not wrong). Dead code line in confound_audit.py (variable computed but unused). Non-uniform cell counts across pairs mean mean-cos stats mix 22-cell and 25-cell pairs without reweighting — minor effect.
- **Fisher dtype fix verified correct.** INLP concern about near-zero Ridge directions is not realized empirically.

## Data

- **Git**: 7 new scripts + 7 analysis JSONs + 5 figures. +4 MB.
- **HF Dataset**: Grid B activations uploaded as `v7_xz_grid/` (~80 MB, 5820 prompts).

## The refined paper claim

**Before v7**: "A shared polarity direction steers all 8 pairs; 7 candidate
directions collapse to a shared axis; probe directions = diff-of-means."

**After v7**: "A shared z-polarity direction exists and causally steers
adjective judgment. It is NOT identical to the raw-value (x) direction
on a clean (x, z) grid — the v6 'same direction' claim was a grid artifact.
The direction transfers across semantically-related pairs (body-attribute
cluster: height ↔ weight ↔ size ↔ bmi_abs transfer at ~60% of own-pair),
with weaker transfer across clusters. Supervised Ridge probes learn
statistically-correlated but causally-weak directions (near-orthogonal to
the primal z-axis). Fisher-pullback (H4) is refuted in the moderate-entropy
regime where these activations operate; peaked-softmax regimes remain
untested. All probed pairs (including mathematical pos/neg) show partial
context-dependence (R ∈ [0.42, 0.95]), supporting a continuous-relativity
framing over a relative-vs-absolute dichotomy."

**Methodological contribution**: conditioning on derived variables
(like z from (x, μ)) contaminates activation-geometry analyses. We
demonstrate this for INLP (v4 result was artifact) and direction-
cosine analysis (v6 "direction collapse" was artifact), while the
causal-steering claim survives the correction.

## Reproducibility

```bash
git checkout exp/v7-clean-grid
cd /workspace/repo2
HF_TOKEN=... python scripts/fetch_from_hf.py  # if data not present
python scripts/vast_remote/export_W_U.py e4b   # for Park + Fisher
python scripts/vast_remote/extract_v7_xz_grid.py
python scripts/vast_remote/exp_v7_confound_audit.py
python scripts/vast_remote/exp_v7_clean_steering.py
python scripts/vast_remote/exp_v7_transfer_matrix.py
python scripts/vast_remote/exp_v7_park_fisher_clean.py
python scripts/vast_remote/exp_v7_inlp_clean.py
python scripts/vast_remote/exp_v7_posneg_clean.py  # addendum
```

Total GPU time ~25 min; total CPU time ~10 min.
