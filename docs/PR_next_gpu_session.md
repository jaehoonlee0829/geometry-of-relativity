# PR: `exp/next-gpu-session` — 7 experiments + G31B + critic consensus

Completion promise word: **NEXT-ECLIPSE**

Executes all 7 experiments from `docs/NEXT_GPU_SESSION.md` on Gemma 4 E4B, plus a G31B replication, plus a post-hoc random-direction null for Exp 2 in response to critic feedback. 12 commits, one branch, single PR.

## Headline results (with honest caveats)

| # | Experiment | Result | Caveats |
|---|---|---|---|
| 5 | per-pair logit_diff plots | hero 8-panel heatmap clear; anti-diagonal pattern for relative pairs, vertical pattern for bmi_abs | big/small all-blue, rich/poor mostly-red — prompt x/μ ranges may not bracket crossover |
| 4d | zero-shot bias check | CONFIRMED bias: 4/8 pairs predict "high" word even at x_min (tall@150cm, old@20yr, expert@1yr, obese@17BMI) | alternative: vowel-onset / determiner confound ("is a" favors vowel-initial words); unchecked |
| 3a | Σ⁻¹ cosines persisted | Σ⁻¹ does NOT rescue alignment (|cos(adj,z)| unchanged) | regularization λ ≈ mean-eigenvalue/1000 effectively forces Σ ≈ cI; the result is ~uninformative |
| 6 | dropped w_adj from headline | yes, w_adj ≈ Ridge(sign(z)) is near-tautological | — |
| **2** | **meta-direction w₁ steering × 8 pairs** | **MONOTONE for all 8; 3–29× larger slope than 3 random directions (Exp 2b)** | bmi_abs has the LARGEST |slope|, suggesting w₁ is a general "adjective polarity" knob, not specifically relativity |
| 2b | random-direction null (post-hoc) | ratio w₁/rand ∈ [3.1×, 28.5×] across 8 pairs | n=3 random dirs; could probe more |
| 3b | F⁻¹ cosines (H4 validation) | H4 NOT supported: F⁻¹ ≈ Euclidean at these activations | per-cell F std is 1e-4 to 1e-3 → F(h) ≈ isotropic here. Test at peaked-p activations, or tied-embedding models |
| 1 | zero-shot 5x × 30 seeds × 8 pairs | zero-shot x-decoding clean (cv_R² ≥ 0.96); cos(w_x_zs, w_z_imp) near zero | |cos| ≤ 0.08 is only 2–4σ above √(1/d)≈0.02 null; directional, not quantitative |
| 7 | 3 new absolute controls | relative/absolute dichotomy NOT significant: Welch t=−0.33, p=0.75 (n=7 vs 4) | underpowered; "adult" is polysemous → legal_abs confounded |
| G31B | 31B adjpair replication | G31B MORE relative than E4B on 6/8 pairs; absolute/relative distinction weaker at scale | 6/8 is binomial p≈0.29 under null — not significant as sign-test; needs bootstrapped CIs |

## What the 3 critic agents flagged (synthesized consensus)

Three agents reviewed this branch in parallel, each from a different angle. Summary of their overlapping concerns, with how this PR addresses each:

### Addressed in-branch before PR

- **No random-direction null for Exp 2** → added Exp 2b (`scripts/vast_remote/exp2b_random_null.py`). w₁ is 3–29× larger slope than random. The "universal knob" claim now has a null control.
- **Exp 7 formula mismatch** (primary JSON reported mixed-convention `welch_t=-2.24, p=0.060` while the corrected project-convention gave `p=0.748`) → patched `exp7_abs_controls.json` so primary fields use `-slope_μ/slope_x`; the mixed-formula numbers preserved as `*_mixed_formula_DO_NOT_USE` for audit.
- **bmi_abs plot label "obese/healthy"** (actual low_word is "thin") → fixed + plots re-rendered.

### Acknowledged, NOT fixed in this PR (documented for follow-up)

- **Exp 2 uniform effect includes bmi_abs with the LARGEST |slope|** — w₁ is likely a general high/low polarity direction, not specifically a "relativity" direction. The "relativity substrate" framing should be softened to "shared polarity substrate" in paper-writing.
- **Exp 1 orthogonality is null-consistent**: cos(w_x_zs, w_z_imp) ∈ [-0.01, +0.08] sits 0.3–4σ above the √(1/d) ≈ 0.02 chance floor. The PR makes no stronger claim than "not aligned"; a permutation/random-probe baseline is needed for quantitative statements.
- **Exp 3a Σ⁻¹ near-identity**: regularization `1e-3·trace/d` dominates at d=2560, N=750. Σ⁻¹ ≈ cI, so the result is uninformative more than informative about the Σ-metric framing.
- **Exp 3b F⁻¹ ≈ Euclidean** reflects activations in a high-entropy softmax region where `diag(p) − pp^T` is well-conditioned and near-isotropic. Doesn't refute H4 in general; refutes H4-at-cell-means. Test at confident-prediction activations (peaked p) to give the hypothesis a fair chance.
- **Prompt-design confounds for "absolute" pairs**: `legal_abs` uses "adult" (polysemous: ≥18 vs. "grown-up"); `grade_abs` context blocks re-define the class curve; `temp_abs` may be affected by "cold/hot" being more relative than "freezing/boiling". These confounds tend to make the dichotomy *easier* to see, which is why the null (p=0.75) is somewhat reassuring.
- **Exp 4d "frequency bias"** interpretation has an alternative: after "is a", the determiner-vowel boundary may asymmetrically favor vowel-initial words (old, expert, obese, tall are all vowel-initial and all show positive bias; young, novice, thin, short are consonant-initial). This is not checked.
- **In-sample probe evaluation (Exp 3b)**: probes trained on all 750 activations; Fisher cosines computed on cell-means from the same set. No held-out split. Conservative fix: re-fit probes on 600 trials, evaluate at 150 held-out cell means.
- **w_adj target divergence** from `analyze_v4_adjpairs.py` reference: new scripts use `sign(z)` labels in {-1, +1}; reference uses `zs > 0` labels in {0, 1}. Both are near-tautological with w_z (Exp 6 dropped from headline); downstream numbers are qualitatively consistent but not bit-identical to the reference pipeline.
- **Fisher math precision**: GPU torch impl casts h→fp64 after the `h @ W_U.T` matmul (which runs fp32); `src/fisher.py` casts before. Tail-logit precision differs. For the qualitative "F⁻¹ ≈ Euclidean" claim this is likely immaterial; for quantitative claims it shouldn't be.

### Out-of-scope for this PR

- Multiple-comparison correction across the 192 cosines reported in Exp 3a+3b.
- Preregistration of the primary relativity_ratio convention.
- Held-out steering targets for Exp 2.
- Per-pair bootstrap CIs on G31B vs E4B relativity ratios (needed for the "more relative at scale" claim).

## What to read first

1. `BUILDING.md` — the 2-hour plan this PR delivers on.
2. `figures/v4_adjpairs/logit_diff_heatmap_xmu_8panel.png` — the hero behavioral finding.
3. `figures/v4_adjpairs/meta_w1_steering_curves.png` + `meta_w1_vs_random_null.png` — Exp 2 causal result with the null.
4. `results/v4_adjpairs_analysis/exp7_abs_controls.json` — the relative/absolute dichotomy result (read `welch_t`, not `welch_t_mixed_formula_DO_NOT_USE`).
5. `figures/v4_adjpairs/fisher_vs_euclid_cosines.png` — the H4 negative finding.

## Data

- **Git** (new): 10 commits in this PR; `figures/v4_adjpairs/*.png`, `results/v4_adjpairs_analysis/*.json`, `results/v4_steering/*.json`, per-prompt logit `.jsonl`s (~7.7 MB total, tracked for reproducibility).
- **HF Dataset** `xrong1729/mech-interp-relativity-activations` (private): added folders `v4_zeroshot_expanded/` (Exp 1, 16 npz + 8 logit jsonl + 1 trial file), `v4_abs_controls/` (Exp 7, ~72 files), plus G31B `.npz` files in existing `v4_adjpairs/`. ~500 MB total now.
- **W&B**: unchanged from before the session (empty artifact collection, run `ax81rrlu` kept as historical record).

## Reproducibility

Every script in `scripts/vast_remote/exp*.py` is self-contained and reads from `results/v4_adjpairs/*.jsonl` + cached `.npz` (on HF). To rerun:

```bash
git checkout exp/next-gpu-session
cd /workspace/repo  # or re-clone; paths are repo-root-relative
# Activations are gitignored; either copy from previous Vast box or:
python -c "from huggingface_hub import snapshot_download; snapshot_download(
    'xrong1729/mech-interp-relativity-activations',
    repo_type='dataset', local_dir='.')"
# Run the scripts in any order; Exp 2 requires the model load (~6s on H100)
python scripts/vast_remote/plots_per_pair_v5.py      # CPU, instant
python scripts/vast_remote/exp4d_validation.py       # CPU, instant
python scripts/vast_remote/exp3a_sigma_inv.py        # CPU, ~20 s
python scripts/vast_remote/exp2_meta_steer.py        # GPU, ~2 min
python scripts/vast_remote/exp2b_random_null.py      # GPU, ~3 min
python scripts/vast_remote/export_W_U.py e4b         # GPU, ~6 s
python scripts/vast_remote/exp3b_fisher_cosines.py   # GPU, ~2 min
python scripts/vast_remote/exp1_zero_shot_expand.py  # GPU, ~30 s
python scripts/vast_remote/exp7_abs_controls.py      # GPU, ~2 min
python scripts/vast_remote/g31b_adjpairs.py          # GPU, ~4 min (G31B load ~20 s)
```
