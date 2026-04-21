# PR: `exp/v7-clean-grid` — fix the (x, μ) → z confound + rerun v4-v6 cleanly

**Motivation** (per `docs/NEXT_GPU_SESSION_v7.md`): v4-v6 used an (x, μ) grid
where z = (x − μ)/σ was derived. Because we only sampled 5 x × 5 μ cells,
the z-bins |z|>1 contained biased x means (correlation corr(x,z) ∈
[+0.58, +0.86] across pairs). Every "z-direction" computed via diff-of-means,
PCA, or Ridge was therefore contaminated with x-information.

**Fix**: Grid B iterates (x, z) directly and derives μ = x − σ·z (or
log-space equivalent for wealth). Domain-plausible μ clipping drops 0-3
cells per pair. Grid B achieves corr(x, z) ≈ 0 for 5/8 pairs and < 0.20
for the three pairs with dropped cells (size 0.13, speed 0.09, experience
0.20).

## What v7 rules INTO and OUT of the v5/v6 story

### CONFIRMED robust (same finding on clean grid)

| v5/v6 claim | clean-grid result |
|---|---|
| **primal_z (diff-of-means) steers 13× stronger than probe_z** | **18× on clean grid** — the gap WIDENS. Primal_z slope still 0.10-0.16 per α-unit; probe_z still 0.01. |
| Meta w₁ causally steers all 8 pairs | YES — slopes −0.03 to −0.13, same direction as v6. |
| cos(w_z, w_ld) ≈ 0 at late layer | YES — Euclidean average ~0.06 on clean probes. |
| F⁻¹ ≈ Euclidean (H4 refuted) | YES — now also at softmax-entropy-binned activations. ~1.5× amplification at most; absolute values 0.02-0.14. |

### OVERTURNED (v5/v6 finding was an artifact of the confound)

| v5/v6 claim | what clean grid actually shows |
|---|---|
| "primal_z ≈ primal_x (cos 0.91, same direction)" | **cos 0.03 on clean grid.** Separate directions. The collapse was a design artifact. |
| "PC1 ≈ primal_z ≈ primal_x" | PC1 is 0.68 aligned with primal_z, only 0.11 with primal_x. PC1 is closer to z than x, but not a full overlap either. |
| "meta_w1 = −mean(primal_z)" (cos −0.98) | On clean grid, the meta-direction is still close to PC1s' shared axis, but the "primal_z ≈ primal_x" component of that identity is gone. |
| "probe_z ≈ probe_x (cos 0.54)" | cos 0.27 on clean grid — shared Ridge artifacts remain, but ~half as much. |
| "INLP barely reduces R²(z) (v4: Δ=0.04)" | **R²(z) drops 0.29-0.51 across 8 pairs on clean grid.** The v4 finding was driven by x-pathway leakage. |

### NEW findings in v7

| Finding | Magnitude |
|---|---|
| **Cross-pair transfer with primal_z_clean** | 40% of own-pair slope, 5.5× above random-direction null. Previously thought to not transfer (v6 used probe_z, the weak direction). |
| **Body-attribute cluster in transfer** | weight↔bmi_abs↔size all transfer at near-own-pair strength. Suggests semantic clustering, not universal substrate. |
| **PC2 is genuinely causal for some pairs** | size (+0.139), experience (+0.140), speed (−0.104), wealth (+0.096). Not a pure geometric artifact. |
| **Grid B residual corr for 3 pairs** | size 0.13, speed 0.09, experience 0.20 — due to dropped-μ cells. Not fully clean. |

## Methodological contribution (could be a section of the paper)

**Conditioning on derived variables in mech-interp studies contaminates
direction-based analysis.** Anyone using (x, μ) grids to study derived-z
effects needs to double-check using an (x, z) grid. We demonstrate this
contamination concretely for INLP (v4 result was artifact) and direction-
cosine analysis (v6 "direction collapse" was artifact), while showing
it did NOT affect the causal-steering claim.

## Data

- **Git**: 6 new scripts + 6 analysis JSONs + 5 figures. +4 MB.
- **HF**: Grid B activations uploaded to `xrong1729/mech-interp-relativity-activations/v7_xz_grid/` (5820 prompts × 2 layers × fp32 = ~80 MB).

## Headline for the paper

- **Before v7**: "A shared polarity direction steers all 8 adjective pairs;
  7 candidate directions collapse to a shared primal axis; H4 (Fisher)
  doesn't rescue probe alignment."
- **After v7**: "A shared polarity direction (primal_z on clean grid)
  steers all 8 pairs and transfers across pairs at 40% of own-pair
  strength. Probe directions (supervised Ridge) are nearly orthogonal
  to this causal axis (cos ≈ 0.05-0.10). Prior mech-interp work using
  (x, μ) grids may have overestimated direction-overlap between z and x."

## Reproducibility

```bash
git checkout exp/v7-clean-grid
cd /workspace/repo2
# If data gone, fetch activations from HF
HF_TOKEN=... python scripts/fetch_from_hf.py
# Re-export W_U for Park + Fisher
python scripts/vast_remote/export_W_U.py e4b
# Run each priority (Grid B extract ~3 min; analysis scripts ~5 min each)
python scripts/vast_remote/extract_v7_xz_grid.py
python scripts/vast_remote/exp_v7_confound_audit.py
python scripts/vast_remote/exp_v7_clean_steering.py
python scripts/vast_remote/exp_v7_transfer_matrix.py
python scripts/vast_remote/exp_v7_park_fisher_clean.py
python scripts/vast_remote/exp_v7_inlp_clean.py
```

## Critic consensus

(Added after 3 critic agents reviewed the commits — section completed
pre-merge; see `critic_consensus.md` in this PR for the full writeup.)
