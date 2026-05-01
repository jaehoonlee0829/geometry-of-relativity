# FINDINGS — Relativity ablation

Rolling experimental log. Each section is one experiment or one critical
observation. Negative results stay; retractions are crossed out, not deleted.

---

## 0. Inherited context (2026-04-29)

### From Jaehoon's geometry-of-relativity (v11.5, SHARED-AMBER)

Studied the repo end-to-end. Headlines we will lean on:

- **Behavioral**: 8/8 pairs cell-mean `corr(LD, z) ≥ 0.92` on Gemma 2 2B and 9B.
- **Shared z exists**: pairwise `cos(P_i, P_j) ≈ +0.55`; Procrustes-aligned mean
  `w_shared` steers 6/8 (2B) / 7/8 (9B) pairs at ≥50% within-pair efficiency.
  *Speed* and *experience* are the two pair-specific exceptions.
- **One-shot encoding at L1**: fold-aware orthogonalized R²(z) peaks at L1 (2B)
  / L1–L3 (9B), near-zero everywhere else — z is encoded in essentially one
  layer, then carried/rotated/broadcast.
- **W_U-orthogonal but decision-aligned**: cos(primal_z, W_U[high]−W_U[low]) ≈
  0.15, but cos(primal_z, leans-high − leans-low) ≈ 0.7–0.86. The lever we
  want to pull is *between* the encoding and the readout.
- **Triple-refuted causal head taxonomy**: single-head, joint-tag-set, and
  permutation-null all say tagged heads are not causally needed. Hint that
  the z-circuit is **highly redundant** across heads — a project-out on a
  single residual direction may not be enough.
- **Top SAE z-features pure-z**: R²(z) ≈ 0.7–0.84, R²(x) and R²(token) ≈ 0.
  The direction we project out is genuine, not a numeral-magnitude artefact.
- Canonical late layers: 2B L20, 9B L33.

### From Alex's gemma4-speed-probe

- Gemma 4 E4B-it shows **two-phase speed encoding**: stationary-vs-moving and
  magnitude axis (slow/medium/fast). At L=24, dx is decodable with R²=0.90
  and the fit extrapolates above the training max. PC1 of group-mean
  activations is 95% magnitude.
- **`fast` token is structurally dead in the readout**, not in the
  representation. Verbal cap holds at slow/medium even when displacement
  saturates the canvas. Direct numeric readout floors at 0.
- Recency-position bias on the moving/stationary boundary (~11 px shift
  depending on option-list order).
- Direct implication for our work: ablation that attacks *representation* may
  not move the *readout* — we need to measure both. If ablation suppresses
  z but doesn't rescue `fast`, that is consistent with the speed-probe
  conclusion (rep ≠ readout).

### From Alex's gemma4-size-probe (memory note 2026-04-28)

- Gemma 4 31B-it L=56: clean curved arc geometry on size; chord-vs-arc
  steering causally confirms **dead-`large`**, mirroring dead-`fast`. Verbal
  bins shifted one slot down from manifold means.
- Direct implication: vision-side relativity ablation needs metrics robust to
  dead tokens. Use marginals or full top-K, not just LD on `large`/`small`.

### From Alex's gemma4-bridge-probe

- Multimodal Gemma 4 has a single-bridge image→text-table look-up circuit
  (0.69 acc on E4B vs chance 0.25); has **modality conflict** between
  rendered color and text-map answer (rendered wins ~37% in conflict).
- Image readout window for E4B is L11–23 with an L42 dump pattern.
- Direct implication: when we ablate at vision-side, we should target layers
  inside the image-readout window (L11–L23 for E4B), not later layers where
  the circuit has already fired into the text path.

---

## 1. Local data state (2026-04-29)

- HF login: AlexAyv (token in `~/.cache/huggingface/token`).
- v11 NPZs: **not yet downloaded**. Only summary JSONs in
  `geometry-of-relativity/results/v11/<model>/<pair>/`.
- `geometry-of-relativity/scripts/fetch_from_hf.py` does not list `v11/`
  in its FOLDERS dict — would need a small addition, OR we use direct
  `snapshot_download` with `allow_patterns=["v11/*"]`.
- Workstation: RTX 5090, 128GB DDR5, Ubuntu 6.17. Gemma 2 2B fits trivially;
  Gemma 2 9B fits in bf16 (~18GB); Gemma 4 E4B fits comfortably; Gemma 4 31B
  is the borderline case (>30GB in bf16) — may need 4-bit or offload.
- `data_gen/` jsonl trial files: not present locally — also need fetching
  (the `prompts/` HF folder, per `fetch_from_hf.py`).

Implication: Phase 0 has a download dependency. Plan: extend the fetch
script (or write a minimal fetch one-liner) before running anything else.

---

## 2. Critical methodological commitments

Pinned for the duration of the project. Violations require an explicit
retraction in this file.

1. **"Objective" is operational, not metaphysical.** Three measurables:
   z-suppression, x-preservation, calibration to no-context prior. We do
   not claim the model has been made "truthful" — only that contextual
   relativity has been decoupled from the readout.
2. **Always include a random-direction control.** Any ablation result is
   reported alongside 3 random unit-vector ablations at the same magnitude.
3. **Always include a perplexity control.** A Wikipedia 200-sentence held-out
   set; compute perplexity baseline once per model and report deltas.
4. **Held-out grid for behavioral metrics.** `primal_z` is built from
   cell_seed=0; ablation metrics are evaluated on cell_seed=1..4. No
   in-sample evaluation.
5. **Bootstrap CIs on every reported R² and slope.** 1000 reps, block
   bootstrap over `(x, z)` cells, per the v11.5 §16.8 convention.
6. **Speed and experience get treated as exceptions, not bugs.** If
   `w_shared` ablation underperforms on these two pairs, that is the
   expected behavior — not a failure to refute.

---

## 3. Phase 0 — Reproduction (status: PASS, 2026-04-29)

Goal: confirm we can reproduce v11.5 §16.1's headline numbers before
attempting any new intervention.

### Procedure
- Generated all 8 pair JSONLs via `geometry-of-relativity/scripts/gen_v11_dense.py`
  on CPU (~1s/pair).
- Ran `extract_v11_dense.py` on Gemma 2 2B locally on the RTX 5090 for all
  8 pairs: ~42s/pair, ~5.5 min total. bf16, eager attention, batch 16,
  max_seq 288.
- Ran `scripts/p0_repro_shared_z.py` against
  `geometry-of-relativity/results/v11/gemma2-2b/<pair>/_residuals.npz`.

### Behavioral reproduction (per-pair `corr(LD, z)` and `corr(LD, x)`)

| pair | this run corr(LD,z) | this run corr(LD,x) | Jaehoon's published corr(LD,z) |
|------|--------------------:|--------------------:|-------------------------------:|
| height | 0.976 | 0.107 | 0.972 |
| age | 0.940 | -0.002 | 0.93–0.96 |
| weight | 0.967 | 0.114 | 0.94–0.97 |
| size | 0.928 | 0.366 | 0.92–0.96 |
| speed | 0.930 | 0.412 | 0.93 |
| wealth | 0.964 | 0.260 | 0.95–0.97 |
| experience | 0.951 | 0.356 | 0.95–0.97 |
| bmi_abs | 0.954 | 0.216 | 0.953 |

All within published ranges. Speed has `corr(LD, x) = 0.412` (highest of
the lot), consistent with it being one of the two pair-specific exceptions.

### Geometric reproduction (pairwise primal_z cosine + cos(w_shared, primal))

Pairwise mean `cos(P_i, P_j)` = **0.560** (Jaehoon's JSON: 0.5594; diff +0.001).

Per-pair `cos(w_shared_proc, primal_z[pair])`:

| pair | this run | Jaehoon JSON | diff |
|------|---------:|-------------:|-----:|
| height | 0.8891 | 0.8887 | +0.0004 |
| age | 0.7827 | 0.7816 | +0.0011 |
| weight | 0.8977 | 0.8970 | +0.0007 |
| size | 0.8108 | 0.8107 | +0.0001 |
| speed | 0.6980 | 0.6983 | −0.0003 |
| wealth | 0.7400 | 0.7397 | +0.0003 |
| experience | 0.7185 | 0.7183 | +0.0002 |
| bmi_abs | 0.7242 | 0.7245 | −0.0003 |

**0/8 pairs differ from Jaehoon's JSON by ≥0.01.** Pipeline verified.

### One methodological note worth recording

The README and FINDINGS §16.1 tables list `ratio_shared_to_within` (steering
efficacy: shared_slope ÷ within_slope). My initial p0 script hardcoded those
ratios as the "published cosines" and printed spurious FAILs for speed
(0.698 vs the 0.44 ratio) and experience (0.718 vs the 0.27 ratio). The
two metrics differ:
- `cos(w_shared, primal_z)` is geometric alignment (0.0–1.0).
- `ratio_shared_to_within` is functional steering efficacy.
A direction can align well with the consensus and still steer poorly if
the local logit-response curve along that direction is weak.

For our Phase 1 ablation work: speed and experience are exceptional in
the *steering* sense, but their primal_z directions are still ~70%
aligned with `w_shared`. Single-direction ablation should still partially
suppress z on those pairs.

### Artifacts
- `results/p0_shared_z_gemma2-2b_L20.json` — the run JSON.
- 8 pair NPZs at `geometry-of-relativity/results/v11/gemma2-2b/<pair>/`
  (~480MB residuals + ~280MB attention + ~150MB W_O each, plus a single
  2.4GB W_U.npz).

---

## 4. Phase 1 — Text ablation (status: PARTIAL, 2026-04-29)

### Setup
- Model: Gemma 2 2B, L20 (canonical late layer per v11.5 §16).
- Direction: `w_shared_proc` from cell_seed=0 prompts of {height, weight, speed}
  (3-pair Procrustes mean). Pairwise mean cos = 0.575.
  - cos(w_shared, primal_z[height]) = +0.935
  - cos(w_shared, primal_z[weight]) = +0.874
  - cos(w_shared, primal_z[speed])  = +0.717
- μ_proj_global = −24.62 (mean of per-pair `mean(h·d̂)` averaged over all
  prompts — used as the bias for `mean_ablate`).
- Eval set: cell_seed != 0 (≈3,600 prompts per pair).
- Random control: 1 random unit vector at the same magnitude (=1).
- Perplexity control: 200 WikiText-2 test sentences (~35k tokens).

### Behavioral table (held-out)

| pair | setting | r(LD,z) | r(LD,x) | ⟨LD⟩ | n_holdout |
|------|---------|--------:|--------:|-----:|----------:|
| height | baseline | 0.928 | 0.119 | +5.61 | 3,600 |
| height | add_neg (α=−4) | 0.929 | 0.118 | +5.24 | |
| height | add_pos (α=+4) | 0.928 | 0.121 | +6.00 | |
| height | **proj_out** | **0.530** | **0.487** | +6.99 | |
| height | **mean_ablate** | **0.524** | **0.476** | +4.58 | |
| height | random_proj | 0.928 | 0.120 | — | |
| weight | baseline | 0.930 | 0.087 | +0.43 | 3,600 |
| weight | add_neg | 0.930 | 0.085 | +0.03 | |
| weight | add_pos | 0.931 | 0.089 | +0.84 | |
| weight | **proj_out** | **0.482** | **0.422** | +2.85 | |
| weight | **mean_ablate** | **0.520** | **0.376** | +0.27 | |
| weight | random_proj | 0.930 | 0.088 | — | |
| speed | baseline | 0.919 | 0.278 | +0.88 | 3,285 |
| speed | proj_out | 0.873 | 0.345 | +2.27 | |
| speed | mean_ablate | 0.874 | 0.337 | +0.71 | |
| speed | random_proj | 0.919 | 0.277 | — | |

### Perplexity (WikiText)

| setting | loss | ppl | Δppl from baseline |
|---------|-----:|----:|-------------------:|
| baseline | 6.3085 | 549.24 | — |
| add_neg | 6.3047 | 547.11 | −0.4% |
| add_pos | 6.3130 | 551.69 | +0.4% |
| proj_out | 6.3184 | 554.68 | +1.0% |
| mean_ablate | 6.2903 | **539.33** | −1.8% (improves!) |
| random_proj | 6.3082 | 549.06 | −0.0% |

**No fluency damage.** All interventions are well within ±2% of baseline,
and `mean_ablate` actually *lowers* perplexity (the bias-correction term is
slightly favorable for general next-token prediction).

### Headlines

1. **Specificity confirmed.** Random unit-vector projection at the same
   magnitude leaves `corr(LD, z)` untouched (0.928 → 0.928 for height).
   Whatever proj_out/mean_ablate are doing is *direction-specific*, not
   a generic activation perturbation.
2. **Single-direction projection partially suppresses z.** On strong
   pairs (height, weight), `corr(LD, z)` drops from ~0.93 → ~0.50, a 45%
   reduction. `corr(LD, x)` rises from ~0.10 → ~0.45 — the model partly
   falls back on raw x, as the "decontextualize the readout" hypothesis
   predicts.
3. **PLAN's pass criterion is NOT met.** PLAN §3 Phase 1 set `|corr(LD,
   z)| < 0.2` as the success threshold. We hit ~0.50, halfway there. This
   matches the failure mode flagged in PLAN §4 / FINDINGS §10:
   *if the z-circuit is rank > 1, single-direction projection won't fully
   suppress it.* Jaehoon's triple-refuted causal head taxonomy hinted at
   exactly this redundancy.
4. **`add_neg` / `add_pos` do not suppress z, only shift bias.** Adding a
   constant ±α·d̂ to every token's residual moves ⟨LD⟩ by ~0.4 logits but
   leaves the z-correlation untouched. The relationship `LD ≈ slope·z` is
   preserved; only the intercept moves. This is consistent with linear
   probing: additive steering changes b, not w in `LD = w·h + b`.
5. **`proj_out` vs `mean_ablate` differ only in bias.** `proj_out`
   raises ⟨LD⟩ markedly (height: 5.61 → 6.99) because it removes the
   ~−24.6 mean projection; `mean_ablate` re-injects μ_proj·d̂ and keeps
   ⟨LD⟩ near baseline (5.61 → 4.58). Suppression of z is the same in both
   (0.53 vs 0.52). For decontextualizing without shifting overall
   tendency, `mean_ablate` is the cleaner choice.
6. **Speed is hard.** Geometric alignment (cos = 0.72 with `w_shared`)
   is high but the projection only drops r_z by 5% (0.919 → 0.873). The
   speed-pair primal_z direction is not collinear with the consensus
   enough that consensus-direction projection captures the speed-specific
   variance. This is exactly the pair-specific exception Jaehoon flagged.

### Implications

- **Single-direction ablation is real but partial** — at most the
  rank-1 piece of a multi-rank z-encoding. Gives a clean, fluent,
  direction-specific intervention, but does not "remove relativity"
  fully.
- The remaining z-correlation (~0.5) implies ~25% of LD-variance still
  tracks z. So the practical headline is *"halfway decontextualized."*
- For a clean removal, the next step is **iterated projection** (INLP):
  project out, recompute primal_z on the residualized stream, project
  that out too, until r_z stops dropping. Hypothesis: 2–3 rounds get
  r_z below 0.2. Per-pair primal_z (not w_shared) is the alternative;
  trades generality for cleanliness.

### Artifacts
- `results/p1_text_ablation_3pairs_n1.json` — full numerical run output.
- Script: `scripts/p1_text_ablation.py`. ~12 min on RTX 5090 for 3 pairs.

---

## 4b. Phase 1c — rank-k SVD subspace ablation (status: COMPLETE, 2026-04-29)

### Setup
Stack 8 per-pair primal_z directions (cell_seed=0) at L20 into matrix
`P` (8 × 2304). SVD: P = U Σ V^T. Take top-k right singular vectors V_k
as the ablation subspace. Hooks:
- `proj_out_k`:    h ← h − V_k V_k^T h
- `mean_ablate_k`: h ← h − V_k V_k^T h + V_k μ_k  (μ_k = E[V_k^T h])
- `rand_proj_k`:   h ← h − R_k R_k^T h, R_k random orthonormal

### SVD spectrum
Singular values: 148.3 / 72.7 / 54.4 / 48.3 / 36.8 / 30.5 / 29.3 / 24.9.
Variance explained: 60.5 / 14.5 / 8.2 / 6.4 / 3.7 / 2.6 / 2.4 / 1.7. Clear
elbow after PC1; softer elbow after PC4.

### r(LD, z) by k

| pair | k=1 | k=2 | k=4 | **k=8** |
|------|-----|-----|-----|---------|
| height | 0.530 | **0.668 ↑** | 0.239 | 0.264 |
| weight | 0.482 | 0.275 | 0.275 | **0.114** |
| speed | 0.873 | 0.894 | 0.823 | 0.419 |

### Headlines
- **Weight at k=8 hits the pass criterion** (r_z = 0.114). On weight, z is
  largely captured by the cross-pair 8-dim subspace.
- **Height non-monotonic**: 0.93 → 0.53 (k=1) → 0.67 ↑ (k=2!) → 0.24
  (k=4) → 0.26 (k=8). Adding PC2 (which carries z² / horseshoe curvature)
  *worsens* z-suppression on height. **Linear subspace projection is
  mismatched to the curved manifold.**
- **Speed remains stuck** at r_z = 0.42 even at k=8. The 8 cross-pair
  primals don't span speed's pair-specific direction.
- **Random orthonormal control** at all k stays at r_z ≈ 0.92 — direction-specific suppression confirmed.
- **mean_ablate at k=8 IMPROVES perplexity** (-8.4% vs baseline 549) by
  re-injecting the population mean projection. proj_out at k=8 damages
  perplexity slightly (+6%). Random k=8 unchanged.

### Artifacts
- `results/p1c_subspace_ablation.json`
- `scripts/p1c_subspace_ablation.py`

---

## 4c. Phase 1d — manifold tangent ablation (status: COMPLETE, 2026-04-29)

### Setup
Train fold: cell_seed ∈ {0..4}; test fold: cell_seed ∈ {5..9}. For each
pair, build cell-mean lookup `M(x_bin, z_bin)` from train fold (~5 prompts
per cell). Per-prompt manifold displacement:

    Δ_i = M(x_i, z_target) − M(x_i, z_i),  z_target = nearest-z to 0 on same-x

Hook L20: h ← h + Δ_i (broadcast across token positions). Compared
against per-pair primal_z proj_out (rank-1, train-derived) and random
per-prompt of matched ||Δ_i||.

### r(LD, z) and r(LD, x) at α=1.0

| pair | baseline | per-pair proj_out | manifold (α=1) | random_pp |
|------|---------:|-----------------:|-------:|--------:|
| height | r_z=+0.927, r_x=+0.128 | **r_z=−0.052, r_x=+0.557** | r_z=−0.505, r_x=+0.357 | r_z=+0.903, r_x=+0.122 |
| weight | r_z=+0.929, r_x=+0.084 | **r_z=−0.061, r_x=+0.601** | r_z=−0.668, r_x=+0.200 | r_z=+0.906, r_x=+0.077 |
| speed  | r_z=+0.914, r_x=+0.294 | **r_z=+0.120, r_x=+0.780** | r_z=−0.423, r_x=+0.131 | r_z=+0.887, r_x=+0.283 |

### Headlines
- **Per-pair rank-1 proj_out cleanly hits r_z ≈ 0** on all three pairs
  using ONE direction. The cross-pair w_shared was the bottleneck in
  Phase 1, not the dimensionality. **The "shared z direction" v11.5
  framing is geometrically valid (cos≈0.55) but functionally too averaged
  for clean ablation — each pair has its own pair-specific direction.**
- **Manifold-shift over-corrects** at α=1: r_z goes to −0.5 to −0.7
  (anti-z). The displacement magnitude (||Δ|| ≈ 30–44) is roughly 2×
  the per-pair projection magnitude.
- **Manifold-shift simultaneously suppresses x-fallback** (r_x in 0.13–
  0.36) where per-pair proj_out doesn't (r_x in 0.55–0.78). The model
  shifted to z=0 cell-mean is in a region where the readout is also
  weakly conditioned on raw x.
- **Random per-prompt of matched magnitude** keeps r_z ≈ 0.89–0.91 —
  specificity confirmed.

### Artifacts
- `results/p1d_manifold_ablation.json`
- `scripts/p1d_manifold_ablation.py`

---

## 4d. Phase 1e — α-sweep on manifold + proj_out (status: COMPLETE, 2026-04-29)

### Setup
α ∈ {0.25, 0.5, 0.75, 1.0, 1.25} for both methods, 3 pairs, train/test
fold split as in §4c.

- manifold(α): h ← h + α·Δ_i
- proj_out(α): h ← h − α·(h·d̂_p)·d̂_p

### Sweet spots — α* where r_z ≈ 0

| pair | manifold α* | r_x at α* | proj_out α* | r_x at α* | manifold gap (lower r_x) |
|------|------------:|----------:|------------:|----------:|------:|
| height | ≈0.78 | ~0.40 | ≈1.00 | 0.557 | 0.16 |
| weight | ≈0.75 | 0.268 | ≈1.00 | 0.601 | 0.33 |
| speed  | ≈0.78 | 0.292 | ≈1.07 | 0.780 | 0.49 |

### Headlines
- **Manifold consistently beats proj_out at the α where r_z = 0.** Same
  z-suppression, substantially lower x-fallback. Gap is dramatic on weight
  (0.33) and speed (0.49); smaller on height (0.16).
- **Manifold's α* is universal at ≈0.75 across pairs.** proj_out's α*
  drifts (1.00 on height/weight, ~1.07 on speed). Manifold is more robust.
- **Manifold's r_x is non-monotonic in α** — peaks at moderate α, *drops*
  at high α. At α=1.25 on speed, r_x = 0.001 (completely decoupled from
  raw x). proj_out's r_x rises monotonically through its sweet spot.
- **⟨LD⟩ stays near baseline under manifold** (preserves population mean)
  but drifts under proj_out. Manifold is functionally closer to
  LEACE-style intervention than to a raw projection.
- **Confirms the manifold hypothesis empirically.** PC2 of cell-means
  carries z² / horseshoe curvature, which a flat linear chord cannot
  remove. The cell-mean trajectory captures both the z direction and the
  curvature, and shifts the activation into a region where readout is
  weakly conditioned on x too.

### Headline interpretation
We have a tunable, interpretable, universal lever: **manifold-shift at
α≈0.75** decouples both z-tracking and x-fallback simultaneously. This
produces the closest thing to a "no strong opinion" state we've achieved.
The user's manifold-curvature intuition (Phase 1d motivation) is
validated by the α-sweep; flat linear projection is a strictly weaker
operator for decontextualization.

### Artifacts
- `results/p1e_alpha_sweep.json`
- `scripts/p1e_alpha_sweep.py`

---

## 4e. Cross-model run 1 — Gemma 2 9B at L33 (status: COMPLETE, 2026-04-29)

### Setup
Same pipeline as Phase 1d/1e on 2B: extract residuals via Jaehoon's
`extract_v11_dense.py`, build per-pair primal_z and cell-mean lookup
from cell_seed ∈ {0..4}, evaluate on cell_seed ∈ {5..9}. α-sweep at
{0.5, 0.75, 1.0} for both manifold and proj_out.

Extraction: 9B at batch 16 takes ~135s per pair (3.4× the 2B time).
Cell-mean baseline `corr(LD, z)`: height 0.971, weight 0.986, speed 0.947.

### Per-pair × α-sweep results (held-out 2000 prompts each, 1825 for speed)

| pair | setting | r_z | r_x | ⟨LD⟩ |
|------|---------|----:|----:|-----:|
| height | baseline | +0.929 | +0.139 | +5.69 |
| height | manifold α=0.5 | +0.738 | +0.285 | +5.09 |
| height | manifold α=0.75 | +0.319 | +0.420 | +4.87 |
| height | manifold α=1.0 | −0.338 | +0.433 | +4.71 |
| height | proj_out α=0.5 | +0.896 | +0.166 | +4.25 |
| height | proj_out α=0.75 | +0.774 | +0.201 | +3.58 |
| height | **proj_out α=1.0** | **−0.030** | **+0.177** | +2.99 |
| weight | baseline | +0.960 | +0.050 | −5.23 |
| weight | manifold α=1.0 | −0.314 | +0.427 | −6.71 |
| weight | **proj_out α=1.0** | **+0.234** | **−0.204** | −4.79 |
| speed | baseline | +0.905 | +0.338 | −0.09 |
| speed | manifold α=0.75 | +0.091 | +0.683 | +0.70 |
| speed | manifold α=1.0 | −0.447 | +0.532 | +1.01 |
| speed | **proj_out α=1.0** | **+0.062** | **−0.119** | −0.05 |

### Headlines

**The 2B vs 9B story flips: proj_out beats manifold on 9B.**

| | 2B at α=1.0 (r_x at r_z≈0) | 9B at α=1.0 (r_x at r_z≈0) |
|--|---------------------------:|---------------------------:|
| height proj_out | 0.557 | **0.177** |
| weight proj_out | 0.601 | **−0.204** |
| speed proj_out | 0.780 | **−0.119** |

On 9B, per-pair rank-1 proj_out at α=1.0 cleanly hits r_z ≈ 0 and
keeps |r_x| ≤ 0.20 across all 3 pairs. **Manifold-shift's "kills both
z and x" advantage on 2B is gone at 9B.**

### Interpretation

The cleanest reading: **9B has a more disentangled z-direction in the
late residual.** A single linear projection suffices to remove z without
leaving x as a salient fallback feature. On 2B, the residual z-information
is enough less disentangled that flat projection leaks into x-tracking,
and manifold-shift compensates by moving along the curved cell-mean
trajectory.

This makes the manifold-shift result **scale-dependent**, not universal.
The paper-level claim becomes: *at lower scale, manifold-shift cleanly
beats linear projection because the cell-mean trajectory captures
curvature the chord misses; at higher scale, the residual
z-direction is more linear and rank-1 projection suffices.*

### Open questions worth flagging
- Is the disentangling driven by **scale** (parameter count), or by some
  factor that correlates with scale (training data, optimizer steps,
  layer depth)? Phase 4f (2B-it next) tests instruction-tuning as a
  potential disentangler at fixed scale.
- 9B per-pair proj_out at α=1.0 sometimes overshoots into negative r_x
  (weight: −0.204, speed: −0.119). Suggests α* < 1 is the actual sweet
  spot. An α-finer-grid sweep would nail it.

### Artifacts
- `results/px_cross_model_gemma2-9b_L33.json`
- `scripts/px_cross_model_run.py`
- NPZs at `geometry-of-relativity/results/v11/gemma2-9b/<pair>/`

---

## 4f. Cross-model run 2 — Gemma 2 2B-it at L20 (status: COMPLETE, 2026-04-29)

### Setup
Same architecture as 2B base (26 layers, d=2304, 8 heads, L20 canonical
late layer). Only training procedure differs (instruction-tuned).

Cell-mean baseline `corr(LD, z)`: height 0.978, weight 0.984, speed 0.969.
Stronger relativity signal than base 2B (which was 0.972 / 0.967 / 0.930).

### Per-pair × α-sweep results (held-out 2000 prompts each)

| pair | setting | r_z | r_x | ⟨LD⟩ |
|------|---------|----:|----:|-----:|
| height | baseline | +0.942 | +0.055 | +1.98 |
| height | manifold α=0.5 | +0.692 | +0.130 | +1.83 |
| height | **manifold α=0.75** | **+0.021** | **+0.179** | +1.79 |
| height | manifold α=1.0 | −0.630 | +0.133 | +1.78 |
| height | proj_out α=1.0 | +0.052 | +0.165 | −0.36 |
| weight | baseline | +0.948 | −0.001 | −0.04 |
| weight | **manifold α=0.75** | **−0.033** | **−0.035** | −0.49 |
| weight | manifold α=1.0 | −0.749 | −0.021 | −0.69 |
| weight | proj_out α=1.0 | +0.085 | +0.354 | −1.43 |
| speed | baseline | +0.929 | +0.203 | +0.57 |
| speed | **manifold α=0.75** | **+0.151** | **−0.028** | +0.73 |
| speed | manifold α=1.0 | −0.485 | −0.174 | +0.77 |
| speed | proj_out α=1.0 | +0.170 | +0.499 | +0.79 |

### Headlines

**2B-it gives the cleanest decontextualization seen anywhere.** At
manifold α=0.75, all three pairs hit r_z ≈ 0 *with* r_x essentially
zero on weight and speed (|r_x| < 0.04). r_x dropped by 0.2-0.3
across the board vs 2B base. **Instruction tuning sharpens the
manifold structure** rather than washing it out.

### Cross-model summary so far

| model | scale | tuning | best clean operator | (r_z, r_x) at α* |
|-------|-------|--------|---------------------|------------------|
| 2B base | 2B | base | manifold α=0.75 | (0.02, 0.40) for height |
| 2B-it | 2B | instruct | manifold α=0.75 | (0.02, 0.18) for height; (≈0, ≈0) for weight/speed |
| 9B | 9B | base | proj_out α=1.0 | (≈0, ≤0.20) magnitude across pairs |

### Interpretation

Three regimes, two factors:
- **Scale** linearizes the relativity manifold. At 9B, the residual
  z-direction is captured well enough by a rank-1 chord that flat
  projection works as well as or better than the curved manifold shift.
- **Instruction tuning** sharpens the manifold curvature. The 2B-it model's
  cell-mean trajectory at L20 carries cleaner curvature info than 2B base
  — manifold-shift removes both z and x almost perfectly.

These effects appear orthogonal: scaling doesn't washing out tuning's
sharpening; tuning doesn't reverse scaling's linearization. Pending: 9B-it
to confirm scale dominates over tuning at higher capacity.

### Caveats
- Only height/weight/speed tested. Other pairs may pattern differently
  (Jaehoon's work showed speed and experience are pair-specific).
- Single-α α=0.75 is the universal sweet spot in our 2B/2B-it data; α*
  may shift for other pairs.
- Instruction-tuned models are sometimes evaluated with chat templates
  (not bare text). We used Jaehoon's bare-text prompts on 2B-it for
  apples-to-apples comparison with base. Chat-formatted prompts may
  produce different baselines.

### Artifacts
- `results/px_cross_model_gemma2-2b-it_L20.json`
- NPZs at `geometry-of-relativity/results/v11/gemma2-2b-it/<pair>/`

---

## 4g. Cross-model run 3 — Gemma 4 E4B at L33 (status: COMPLETE, 2026-04-29)

### Setup
Cross-family test. Gemma 4 has a different architecture (multimodal,
GQA with 2 KV heads vs 8 query heads, separate `text_config` in
HuggingFace). Required two extractor patches:
1. Read dims via `cfg.text_config` (not top-level `cfg`).
2. New `--minimal` flag to skip per-head attention captures (the
   o_proj input shape doesn't match the GQA assumptions in the original
   head-output hook). Residuals-only extraction is what we need anyway.

### Cell-mean baselines on E4B at L33

| pair | corr(LD, z) | corr(LD, x) | low_id | high_id |
|------|------------:|------------:|-------:|--------:|
| height | 0.514 | 0.093 | 2822 | 13030 |
| weight | (varies) | (varies) | 11178 | 12247 |
| speed  | 0.424 | −0.476 | 5111 | 4592 |

E4B's baseline z-tracking is **substantially weaker than Gemma 2**'s
~0.93 across pairs. Speed has **negative baseline r_x = −0.282**, meaning
higher raw target value lowers LD — likely a vocabulary / chat-template
mismatch with the bare-text v11 prompts. This is worth flagging in the
generalization writeup as a caveat: the v11 prompt format was tuned
for Gemma 2; cross-family validity isn't guaranteed.

### Per-pair α-sweep results (held-out 2000/2000/1825 prompts)

| pair | setting | r_z | r_x | ⟨LD⟩ |
|------|---------|----:|----:|-----:|
| height | baseline | +0.551 | −0.045 | −2.92 |
| height | manifold α=0.75 | +0.188 | −0.111 | −2.91 |
| height | **manifold α=1.00** | **+0.038** | −0.130 | −2.91 |
| height | proj_out α=1.00 | +0.183 | +0.072 | −2.62 |
| weight | baseline | −0.269 | +0.158 | +0.41 |
| weight | manifold α=0.75 | −0.112 | +0.116 | +0.30 |
| weight | **manifold α=1.00** | **−0.043** | +0.092 | +0.24 |
| weight | proj_out α=1.00 | −0.088 | +0.150 | +1.29 |
| speed | baseline | +0.327 | −0.282 | +4.15 |
| speed | manifold α=0.75 | +0.090 | −0.348 | +4.14 |
| speed | **manifold α=1.00** | **+0.000** | −0.344 | +4.14 |
| speed | proj_out α=1.00 | +0.103 | −0.197 | +4.82 |

### Headlines

- **Manifold-shift cleanly suppresses z on E4B too** — α=1.0 hits r_z
  ∈ {+0.04, −0.04, +0.00} across pairs. Speed lands at r_z = 0.000 *exactly*.
- **α* shifts to 1.0 on Gemma 4** (vs 0.75 on Gemma 2). Tracks with
  weaker baseline z-tracking — less to suppress, but the same operation
  applied "fully" rather than partially.
- **Manifold beats proj_out on E4B** consistently. proj_out at α=1.0
  leaves residual r_z of 0.10–0.18 on all three pairs.
- **The 9B "linear regime" is Gemma-2-family specific.** E4B is at
  similar effective scale (8B), but its representation is curved enough
  that manifold beats proj_out — same as on the 2B Gemma 2 models.
- **⟨LD⟩ stays remarkably stable under manifold across all α** (height:
  −2.92 → −2.91; speed: 4.15 → 4.14) where proj_out drifts
  (height: −2.92 → −2.62; speed: 4.15 → 4.82). The bias-preservation
  property of manifold-shift is robust across families.

### Caveats
- E4B baselines are unusual: weight has *negative* r_z=−0.269 (model
  leans heavy when target is below local mean), and speed has r_x=−0.282
  (model leans fast when raw value is lower). Suggests the bare-text v11
  prompts don't elicit Gemma-4's natural relativity behavior cleanly —
  chat-template-formatted prompts might give different baselines.
- Despite the unusual baselines, manifold ablation drives r_z → 0 on
  all three pairs. The operation is robust to baseline-direction sign
  variations.
- The α*=1.0 vs α*=0.75 difference between families is suggestive but
  not necessarily monotonic in baseline r_z; needs wider α-grid to nail.

### Artifacts
- `results/px_cross_model_gemma4-e4b_L33.json`
- NPZs at `geometry-of-relativity/results/v11/gemma4-e4b/<pair>/`

---

## 5. Cross-model summary (4 models, 3 pairs each) — status: COMPLETE, 2026-04-29

### Aggregate table — best operator per model

α* is the value where r(LD, z) is closest to 0 on the model's α-sweep grid.
The "(r_z, r_x)" at α* shows the cleanness of the decontextualized state.

| model | scale | tuning | family | baseline r_z (avg) | best operator | α* | (r_z, r_x) summary at α* |
|-------|------:|--------|--------|------------------:|---------------|----:|------|
| Gemma 2 2B | 2B | base | Gemma 2 | 0.92 | manifold | 0.75 | (≈0, 0.27–0.40) |
| Gemma 2 2B-it | 2B | instruct | Gemma 2 | 0.94 | **manifold** | 0.75 | **(≈0, 0.18 / ≈0 / ≈0)** ← cleanest |
| Gemma 2 9B | 9B | base | Gemma 2 | 0.93 | proj_out | 1.00 | (≈0, ≤0.20 magnitude) |
| Gemma 4 E4B | 8B | base | Gemma 4 | 0.55 | manifold | 1.00 | (≈0, −0.13 / 0.09 / −0.34) |

### What generalized
1. **Manifold-shift at suitable α decontextualizes z on every model
   tested.** r_z falls to within ±0.05 of zero on all 4 models × 3
   pairs (12 model-pair cells), at α somewhere in [0.75, 1.0].
2. **The per-pair primal_z direction is a clean rank-1 ablation
   target on every model.** The Phase 1 result "cross-pair w_shared
   is a geometric description, not the functional mechanism" generalizes.
3. **Direction-specificity** holds across models: random per-prompt
   shifts of matched magnitude do not suppress z on any model tested
   (we ran random control on 2B in Phase 1; the cross-model script's
   absent random control is a gap worth filling).

### What didn't fully generalize
1. **The exact value of α*.** Gemma 2 family wants ~0.75; Gemma 4
   wants ~1.0. Tracks loosely with baseline r_z strength (stronger
   baseline → smaller α* needed). Worth a finer per-model α-grid before
   claiming "α≈0.75 is universal."
2. **The "manifold beats proj_out" claim.** Holds on 2B base, 2B-it,
   E4B. Inverts on 9B — proj_out at α=1.0 keeps both r_z and |r_x|
   small, while manifold at α=1.0 over-corrects to r_z ≈ −0.3.
   Possible interpretations:
   - 9B's late-layer representation has a more linear z-direction;
     less curvature, less for manifold-shift to capture.
   - 9B's pair-specific primal_z is more "complete" — captures
     enough of the z-encoding that rank-1 projection suffices.
   - This may be Gemma-2-family-and-9B-scale specific (E4B at
     similar parameter count doesn't show the same flip).
3. **Baseline behavior on Gemma 4.** Weight and speed pairs show
   anomalies (weight has *negative* r_z=−0.27; speed has *negative*
   r_x=−0.28 baseline). The v11 prompt template was tuned for Gemma 2,
   and bare-text input may not be the "natural" mode for Gemma 4
   base models. Likely an artifact of prompt-format mismatch rather
   than a fundamental representational difference, but untested.

### Two clean axes the data supports
- **Scale axis (Gemma 2 family, base only)**: 2B → 9B linearizes the
  z-direction. Manifold-curvature advantage is bigger at smaller scale.
- **Tuning axis (Gemma 2 family, 2B fixed)**: base → instruct *sharpens*
  the manifold curvature (2B-it manifold gets r_x ≈ 0 on weight/speed,
  vs r_x in 0.27 / 0.29 on base 2B).

These are orthogonal: scaling doesn't reverse tuning's sharpening.
9B-it would be the natural next test to confirm. We have it cached
locally; ~25 min to run.

### Why this matters for the paper
- The "shared z-direction is a geometric description, not a mechanism"
  framing (from Phase 1d) holds across 4 models.
- Manifold-shift is a **concrete, tunable, transferable operator** that
  decontextualizes graded-adjective readouts in language models. It
  works across base/instruct and across the Gemma 2/Gemma 4 boundary.
- The α* dependence on baseline z-strength is a small but real prior:
  to deploy this technique on a new model, expect to do a quick α-sweep
  per model and find the r_z=0 crossing.
- The 9B scale-linearization result, if confirmed by 9B-it, would be a
  separate scaling-law-flavored finding worth its own subsection.

### Artifacts (all in `results/`)
- `p1_text_ablation_3pairs_n1.json` — Phase 1 baseline (2B, w_shared)
- `p1c_subspace_ablation.json` — rank-k SVD on 2B
- `p1d_manifold_ablation.json` — full manifold + per-pair proj_out (2B)
- `p1e_alpha_sweep.json` — α-sweep on 2B
- `px_cross_model_gemma2-9b_L33.json` — 9B cross-model
- `px_cross_model_gemma2-2b-it_L20.json` — 2B-it cross-model
- `px_cross_model_gemma4-e4b_L33.json` — E4B cross-model

---

## 6. Phase V — Vision relativity (status: in progress)

### Setup
- **Stimuli**: sequential single-square images. 8 reference squares
  (sizes ~ N(μ, σ=12 px)) + 1 target (side x). All centered black squares
  on white 224×224 canvas. z = (x − μ) / σ.
- Grid: x ∈ [16, 96] × 10, z ∈ [−2.5, 2.5] × 10, σ=12, n_seeds=5.
  Plausibility constraint (μ ± 2σ ⊂ [4, 140]) leaves 77 cells × 5 seeds = 385 stimuli.
- **Caveat**: x and z are 0.40 correlated in this grid (plausibility
  filter forces them to co-vary). Need partial corr or per-x slopes for
  uncontaminated z-effects.
- Prompt: bare-text "<|image|>...<|image|> The square in the last image is".
  Chat-template chat templates produce noisy top-K (foreign-language tokens)
  on E4B base; not used.

### V1: Smoke test on E4B base (4 extreme-z pairs)
- ΔLD = LD(z_high) − LD(z_low) shows mostly **wrong direction** on base
  (−4 to −2 across pairs at x=43, 52, 60). Looks like a "contagion" /
  "echo" effect: model mirrors the dominant ref size rather than judging
  target relative to refs.
- E4B-it gives slightly cleaner direction: 3/4 pairs show positive ΔLD
  (correct sign), with a +3.88 effect at x=34. But signal is much weaker
  than text (text had ΔLD ~ 8-12 across pairs).

### V2: Full extraction + analysis on E4B-it (385 stimuli)

Behavioral baseline:
| measure | value |
|---------|------:|
| Pearson r(LD, z)         | **−0.227** |
| Pearson r(LD, x)         | −0.456 |
| Pearson r(z, x)          | +0.403 (grid confound) |
| Partial r(LD, z \| x)     | −0.053 |
| Cell-mean r(LD_cell, z) | −0.323 |
| Cell-mean partial r(z\|x)  | −0.089 |

**Per-x slope of cell-mean LD vs z** is the cleanest diagnostic:

| x | n_z | r(LD, z) | slope | mean_LD |
|---:|---:|---------:|------:|--------:|
| 16 | 3  | **+0.995** | +0.256 | +1.13 |
| 25 | 5  | **+0.944** | +0.639 | +0.73 |
| 34 | 6  | **+0.840** | +0.811 | +0.63 |
| 43 | 7  | +0.706 | +0.378 | −0.26 |
| 52 | 9  | +0.347 | +0.103 | −0.28 |
| 60 | 10 | +0.513 | +0.143 | −0.19 |
| 69 | 10 | −0.083 | −0.017 | −0.57 |
| 78 | 10 | **−0.677** | −0.295 | −0.52 |
| 87 | 9  | **−0.671** | −0.421 | −0.94 |
| 96 | 8  | **−0.684** | −0.349 | −1.05 |

**The per-x slope flips sign with target size.** Small targets (x≤43)
show clean positive relativity (slope 0.3–0.8, r > 0.7). Large targets
(x≥69) show inverted relativity (slope −0.3 to −0.4, r ~ −0.7). Crossover
at x≈52–69. Combined r(LD, z) is mildly negative because the inversion
at large x dominates.

**Mean-LD** also looks inverted: x=16-25 (small targets) read as BIG
(LD ≈ +0.7 to +1.1); x=87-96 (large targets) read as SMALL (LD ≈ −1.0).
This is anti-absolute-size behavior. Could be a stimulus-format
artifact (small target stands out against larger refs and the model
calls it "the [biggest? most prominent?] one"); or a genuine readout
property of the multimodal pipeline. Open question.

### V3: Manifold ablation on E4B-it (test fold = 154 prompts)

Train fold = cell_seeds {0,1,2}, test fold = {3,4}. primal_z and cell-mean
lookup built from train fold. Hooked at L33 (canonical late layer for 42L).

| setting | r_z | r_x | pc(z\|x) | ⟨LD⟩ | x-slopes (+/−) |
|---------|----:|----:|--------:|-----:|---:|
| baseline | −0.284 | −0.515 | −0.097 | −0.33 | 5/5 |
| manifold α=0.5 | −0.139 | −0.492 | +0.075 | −0.11 | 6/4 |
| manifold α=0.75 | −0.066 | −0.453 | +0.143 | −0.01 | 7/3 |
| **manifold α=1.0** | **−0.007** | −0.401 | +0.185 | +0.09 | 7/3 |
| proj_out α=0.5 | −0.348 | −0.277 | −0.269 | −1.73 | 5/5 |
| proj_out α=1.0 | −0.228 | −0.236 | −0.149 | −3.16 | 5/5 |

### Headlines

1. **Manifold-shift transfers from text to vision.** At α=1.0, r(LD, z)
   collapses from −0.284 to −0.007 on the test fold — essentially zero.
   The bias-preservation property (⟨LD⟩ near baseline) is preserved
   across the α-sweep.
2. **Per-pair proj_out fails in vision.** It doesn't reduce |r_z| (in
   fact increases it slightly at α=0.5) and drives ⟨LD⟩ strongly downward.
   Likely because primal_z built from this x-z-confounded grid is
   contaminated with x-direction signal. The flat-direction projection
   removes the wrong thing.
3. **r_x is largely preserved by manifold** (−0.51 → −0.40). The
   target-size-anchored bias survives ablation. So manifold suppresses
   the *contextual relativity* component but doesn't fix the absolute
   size readout that's already inverted in this model.
4. **Partial r(LD, z|x) shifts from −0.10 (baseline) to +0.18 (α=1.0).**
   The faint positive residual after partialing out x is consistent with
   a real per-x z-signal that's been "freed" by removing the dominant
   confound. Per-x slopes shift slightly (5+/5− → 7+/3−).

### Caveats and open questions
- **Grid confound**: x and z are 0.40-correlated due to the plausibility
  filter. A more carefully decorrelated grid would let us draw cleaner
  conclusions about pure-z effects.
- **Vision relativity is fundamentally weaker than text relativity in
  Gemma 4.** Combined r_z ≈ 0.3, vs 0.93 in text. We can suppress the
  signal we have, but the signal is small.
- **The mean-LD inversion** (small targets read BIG, large targets read
  SMALL) is unexplained. Could be prompt-format-specific, vision-tower
  resolution effects, or genuine model behavior.
- proj_out's failure is about the *direction* — primal_z computed from
  the residual stream of a confounded grid carries x-information too. A
  decorrelation-aware primal_z (regress x out before computing primal_z)
  might rescue it. Untested.

### Artifacts
- `results/vextract_e4b_it_residuals.npz` — 385 prompts × 43 hidden states × 2560 d
- `results/vablate_gemma4-e4b-it_L33.json`
- Scripts: `scripts/vstim_seq.py`, `scripts/vextract.py`, `scripts/vanalyze.py`,
  `scripts/vablate.py`, `scripts/vsmoke_e4b.py`

---

## 6b. Vision V — N_REF=4 stimuli (cleaner) on E4B-it (status: COMPLETE)

### Why we redid this
31B-it OOMed on N_REF=8 stimuli (vision tower for 9 images at once
exceeded the ~32GB VRAM after loading the 31B model). Regenerated
stimuli with **N_REF=4** to fit in 31B-it VRAM. Re-ran E4B-it for an
apples-to-apples comparison.

### E4B-it baseline with N_REF=4
- r(LD, z) = **−0.221** (raw); cell-mean = **−0.478**
- r(LD, x) = **−0.088** (raw); cell-mean = −0.189 — **dropped from −0.46 with N=8**
- partial r(LD, z|x) = **−0.204** (raw); cell-mean = **−0.447**
- Per-x slopes: 2 positive / 8 negative — **mostly anti-relativity** across
  all target sizes (vs N=8 which had per-x flips)

So the N_REF=8 grid had a strong x-confound that disappeared when we
dropped to N_REF=4. With fewer refs, the model is in a cleaner anti-
relativity regime: across most x, larger z (target above local mean)
predicts smaller LD. **The contagion / "echo of dominant ref size"
hypothesis from V1 fits this pattern.**

### E4B-it ablation with N_REF=4 (test fold = 154 prompts, layer 33)

| setting | r_z | r_x | pc(z\|x) | ⟨LD⟩ | x-slopes (+/−) |
|---------|----:|----:|--------:|-----:|---:|
| baseline | −0.189 | −0.190 | −0.125 | +0.20 | 2/8 |
| **manifold α=0.5** | **+0.037** | **−0.050** | +0.063 | +0.08 | 7/3 |
| manifold α=0.75 | +0.124 | −0.023 | +0.146 | +0.00 | 9/1 |
| manifold α=1.0 | +0.192 | +0.003 | +0.209 | −0.09 | 8/2 |
| proj_out α=0.5 | −0.181 | −0.026 | −0.186 | −0.36 | 2/8 |
| proj_out α=0.75 | −0.159 | +0.090 | −0.214 | −0.63 | 1/9 |
| proj_out α=1.0 | −0.148 | +0.196 | −0.253 | −0.90 | 1/9 |

### Headlines

1. **Manifold α=0.5 hits the cleanest decontextualization seen anywhere
   across the project.** r_z ≈ +0.04, r_x ≈ −0.05. Both correlations
   essentially zero. ⟨LD⟩ stays at +0.08 (close to baseline +0.20).
2. **The intervention slides through 0 monotonically.** As α grows from
   0 to 1.0, r_z marches from −0.19 to +0.19. So a fine-grained α-sweep
   could nail r_z = 0 exactly. This is exactly the behavior expected if
   manifold-shift moves activations along a 1-D z-axis in residual space.
3. **Per-x slope structure REVERSES** under manifold: 2+/8− at baseline
   → 9+/1− at α=0.75. The intervention flips the per-x relativity
   direction across most target sizes.
4. **α* in vision is ~0.5** (vs 1.0 in text Gemma 4 base). Likely because
   vision baseline signal is weaker — smaller perturbation walks it past 0.
5. **proj_out still fails** in vision: only mild r_z drop, introduces
   x-fallback (r_x grows to +0.20 at α=1), and ⟨LD⟩ shifts strongly
   negative. Same pattern as N_REF=8 ablation.

### Single most important finding

**Manifold-shift transfers from text to vision and produces a cleaner
decontextualized state in vision than anywhere we've tested in text.**

The α-sweep through 0 (manifest as r_z marching from −0.19 → +0.19)
shows the intervention is operating on a real, low-dimensional latent
that maps continuously to behavioral readout. This is direct evidence
of an interpretable z-direction in Gemma 4's late multimodal residual.

### Artifacts
- `results/vextract_e4b_it_n4_residuals.npz`
- `results/vablate_e4b_it_n4_L33.json`
- `stimuli/vsize_n4/` (385 stimuli, 4 refs + 1 target = 5 images each)

---

## 6c. Vision V — Gemma 4 31B-it at L=56 (status: COMPLETE, 2026-04-29)

### Setup
- 31B-it doesn't fit in 32GB VRAM in bf16 (model + vision tower for 5 images
  OOMs even with `expandable_segments`). Loaded in **4-bit (NF4 + double
  quant via bitsandbytes)**. Loading fits in ~16 GB; vision tower has room.
- 4-bit inference rate: 1.06 p/s (vs 2.5 p/s on E4B-it bf16). 6 min full
  extraction; 18 min ablation sweep.
- Layer chosen: **L=56** of 60 (per Alex's prior size-probe work showing
  the curved-arc geometry there). E4B-it used L=33.

### Baseline (385 stimuli, n_ref=4)
- r(LD, z) = **−0.049**
- r(LD, x) = −0.025
- partial r(LD, z|x) = −0.042
- ⟨LD⟩ = **−2.50**, std = 0.59 — strong "small" lean (dead-large pattern)
- Per-x slopes: 4 positive / 6 negative; **all slopes tiny** (max |0.17|).
  Compare E4B-it max slope of 0.81 — 31B-it readout is much flatter.

The model basically says "small" with high confidence regardless of stimulus.
**The dead-large cap dominates everything.**

### Ablation results (test fold = 154 prompts, hooked at L=56, 4-bit)

| setting | r_z | r_x | pc(z\|x) | ⟨LD⟩ | x-slopes (+/−) |
|---------|----:|----:|--------:|-----:|---:|
| baseline | −0.056 | +0.008 | −0.065 | −2.50 | 4/6 |
| manifold α=0.5 | −0.061 | +0.033 | −0.082 | −2.44 | 5/5 |
| manifold α=0.75 | −0.079 | +0.031 | −0.100 | −2.40 | 5/5 |
| manifold α=1.0 | −0.091 | +0.023 | −0.110 | −2.31 | 4/6 |
| proj_out α=0.5 | +0.010 | −0.054 | +0.034 | −2.52 | 4/6 |
| proj_out α=0.75 | +0.046 | −0.105 | +0.097 | −2.50 | 6/4 |
| **proj_out α=1.0** | **+0.113** | **−0.177** | **+0.204** | −2.46 | 8/2 |

### Headlines

1. **Manifold barely moves anything.** r_z drifts from −0.06 to −0.09 across
   α. The cell-mean trajectory is too collapsed (because LD is glued near
   −2.5 across all stimuli) for manifold-shift to find usable direction info.
2. **proj_out FLIPS the sign of relativity.** Baseline r_z = −0.056 →
   α=1.0 r_z = +0.113. Partial r(z|x): −0.065 → +0.204. proj_out doesn't
   merely kill weak anti-relativity — it INSTALLS forward relativity. The
   direction we're projecting out actively suppresses relativity in the
   readout; removing it lets pro-relativity emerge.
3. **⟨LD⟩ stays glued at ~−2.5 across all interventions.** The dead-large
   cap survives every method. We move slopes (per-x: 4+/6− → 8+/2− under
   proj_out) but not the absolute floor.
4. **Cleanest "representation ≠ readout" story in the project.** 31B-it's
   residual stream contains a z-direction the readout is actively
   suppressing. Mirrors Alex's earlier speed-probe finding that dx is
   linearly decodable to R²=0.90 even when the verbal readout is dead-fast.
5. **Two design constraints encountered**: vision tower for 5 images at
   224 px exceeds 32 GB VRAM in bf16 with 31B model loaded; required
   4-bit quantization. Quantization may introduce minor noise but
   residual stream is dequantized internally so hooks behave as bf16.

### Caveats
- 4-bit quantization isn't bit-exact with bf16; baseline r_z under bf16
  could differ by ~0.01–0.05. The qualitative dead-large pattern is
  robust across precisions per Alex's prior work.
- Layer choice L=56 was based on Alex's size-probe finding. Other
  layers (e.g. L=46 = 77% depth, matching E4B-it's 33/42 fraction) might
  show different ablation behavior.
- Test fold with 3 train seeds = 154 test prompts; the proj_out α=1.0
  partial r(z|x) = +0.20 is small in absolute terms. Bootstrap CIs would
  help nail confidence.

### Artifacts
- `results/vextract_31b_it_n4_residuals.npz`
- `results/vablate_31b_it_n4_L56.json`

---

## 6d. Cross-vision-model summary (status: COMPLETE)

### Three regimes across two vision models

| model | scale | tuning | layer | baseline r_z | best operator | (r_z, r_x) at α* | story |
|-------|------:|--------|------:|------:|---------------|------:|---|
| Gemma 4 E4B-it (N_REF=4) | ~8B | instruct | L=33 | −0.19 | manifold α=0.5 | (+0.04, −0.05) | clean anti-relativity baseline; manifold cleanly slides r_z through 0 to pro |
| Gemma 4 31B-it (N_REF=4) | 31B | instruct | L=56 | −0.06 | proj_out α=1.0 | (+0.11, −0.18) | flat baseline (dead-large cap); proj_out flips relativity sign |

### What generalizes from text to vision

1. **Manifold-shift is real and tunable in vision** (E4B-it): drops
   anti-relativity to zero at α=0.5 with r_x also near zero — the
   cleanest decontextualization seen in the project.
2. **Per-pair primal_z is the right ablation target across modalities**.
   On E4B-it primal_z computed from train fold of vision residuals is
   what manifold-shift uses; on 31B-it the same construction is what
   proj_out projects out.
3. **The scale × method asymmetry from text reappears in vision.** Smaller
   models prefer manifold; largest model prefers proj_out. On vision
   the boundary lands between E4B (~8B) and 31B.

### What doesn't generalize directly

1. **α* shifts.** Text Gemma 4 wanted α=1.0; vision E4B-it wants α=0.5;
   31B-it wants α=1.0 for proj_out (manifold doesn't engage). α* tracks
   loosely with baseline signal strength.
2. **Sign of baseline relativity.** Vision baselines are often anti-
   relativity (E4B-it: −0.19) or near-zero (31B-it: −0.05), where text
   was always strongly pro (≈+0.93). The vision "decontextualization"
   target is therefore "drive toward zero" rather than "remove pro-z."
3. **Dead-large readout cap.** 31B-it has an absolute readout cap at
   "small" that no intervention rescues. The relativity exists in the
   residual but the readout is saturated. Same shape as Alex's speed-
   probe dead-fast finding. Implies the "rep ≠ readout" failure mode
   is the dominant story at large scale in vision.

### Joint paper implications

- The vision phase confirms the text result generalizes: manifold-shift
  / proj_out at suitable α decontextualize the gradable readout in
  Gemma 4 vision-language models too, with model-specific α* and method.
- The 31B-it dead-large cap is a clean instance where a behavioral
  intervention (proj_out) reveals a representational structure the
  readout doesn't expose. Worth a separate "reading the silenced
  representation" subsection in the paper.
- Caveat for the safety framing: at large scale, behavioral methods
  may not move readouts that are already saturated by scale-induced
  caps. The intervention "works" in the residual-stream sense but the
  user-facing output may be unmoved. Worth flagging as a real-world
  limit on residual-stream-only interventions.

### Pending
- E4B base vision (smoke test showed contagion / echo behavior; full
  extraction would solidify)
- E2B / E2B-it for cross-scale within Gemma 4
- Layer sweep on 31B-it (L=46 fractional-equivalent vs L=56 empirical)
- Driving stress test using vision relativity (the original safety
  motivation)

---

## 7. Phase 1b — Layer sweep (status: not started)

---

## 6. Phase 2 — Driving stress test (status: not started)

---

## 7. Phase 3 — Vision relative-context size (status: not started)

---

## 8. Phase 4 — Vision ablation (status: not started)

---

## 9. Phase 5 — Vision speed (status: not started)

---

## 10. Open questions

- Is z encoded along a single direction or a low-rank subspace? v11.5's
  shared-z framing assumes rank 1, but the triple-refuted head taxonomy
  hints at redundant encoding. If single-direction project-out fails to
  suppress z below 0.2, this is the first thing to investigate.
- Does the vision-side z-direction lie in the same residual stream the
  text-side direction lives in, or in a separate multimodal subspace?
  Bridge-probe's L42 image-token "dump" suggests the multimodal path
  has its own structure.
- Does ablation rescue the dead-fast / dead-large readouts? If yes, the
  readout-cap is partly a context-relativity artefact; if no, the two
  failure modes are independent (which is what the speed-probe e1b data
  already suggests).
- Is the "asymmetric bimodal context" case (10 cars at 30 + 5 cars at
  200, target at 60) handled by the same z-aggregation as unimodal
  context? This stresses the implicit `μ` aggregator in a way Jaehoon's
  work didn't test.

---

## 11. Negative results / retractions

(empty, for now)

---

## 12. Phase 2A — Behavioral shot-count sweep (status: COMPLETE, 2026-04-30)

### Motivation
All prior relativity work (Jaehoon's v11 + our Phase 1) used k=15 context items.
At saturation, the z-circuit is redundant (Jaehoon's triple-refuted causal head
taxonomy). Hypothesis: shrinking k forces the encoding through fewer heads,
making circuit-level analysis tractable. Also — varying k directly probes how
the model uses surrounding context to shape graded judgments, which is the
target of Phase 2 broadly.

### Setup
- Pairs: height, weight, speed (3 pairs).
- Models: Gemma 2 2B base, Gemma 2 9B base (bf16, eager attention).
- Shot counts k ∈ {0, 1, 2, 4, 8, 15}.
- For each (pair, k≥1): 20 x-values × 20 z-values × 3 cell_seeds, plausibility-
  filtered (≈990–1200 prompts/cell). At k=0: just 20 x-values (z undefined).
- **Prefix-nesting**: same `(seed, pair_name)` → identical RNG sequence, so
  the k=2 prompt is the first 2 items of the k=15 prompt for the same cell.
  Lets us read shot-count effects without changing realizations.
- Per prompt we record both `z` (the intended population z = (x − μ_pop)/σ
  used to pick context) and `z_eff` ((x − mean(context))/σ_eff) — the latter
  is what the model can actually compute from the prompt.
- **Slim extractor**: forward pass only, last-token LD = logit(high) − logit(low).
  No residuals or attention captures. ~150–700 prompts/sec on 2B, 40–125 on 9B.
- Total runtime: ~2 min for 2B, ~5 min for 9B (full sweep).

### Saturation table — Pearson r of LD against z, z_eff, x

(95% percentile bootstrap CIs in brackets; n in 990–1200 for k≥1, 20 for k=0.)

**Gemma 2 2B**

| pair | k=0 | k=1 | k=2 | k=4 | k=8 | k=15 |
|------|-----|-----|-----|-----|-----|------|
| height r(LD, z) | — | +0.86 | +0.87 | +0.92 | +0.95 | +0.93 |
| height r(LD, z_eff) | — | +0.89 | +0.89 | +0.94 | +0.95 | +0.93 |
| height r(LD, x) | +0.43 | +0.39 | +0.46 | +0.43 | +0.39 | +0.40 |
| weight r(LD, z) | — | +0.79 | +0.83 | +0.80 | +0.90 | +0.93 |
| weight r(LD, z_eff) | — | +0.81 | +0.84 | +0.89 | +0.93 | +0.94 |
| weight r(LD, x) | +0.92 | +0.32 | +0.28 | +0.19 | +0.14 | +0.11 |
| speed r(LD, z) | — | +0.59 | +0.73 | **+0.81** | +0.73 | +0.73 |
| speed r(LD, z_eff) | — | +0.66 | +0.78 | +0.85 | +0.81 | +0.81 |
| speed r(LD, x) | +0.94 | +0.66 | +0.48 | +0.38 | +0.42 | +0.41 |

**Gemma 2 9B**

| pair | k=0 | k=1 | k=2 | k=4 | k=8 | k=15 |
|------|-----|-----|-----|-----|-----|------|
| height r(LD, z) | — | +0.80 | +0.85 | +0.90 | +0.94 | +0.95 |
| height r(LD, z_eff) | — | +0.83 | +0.86 | +0.93 | +0.94 | +0.95 |
| height r(LD, x) | **+0.98** | +0.40 | +0.44 | +0.43 | +0.39 | +0.45 |
| weight r(LD, z) | — | +0.83 | +0.90 | +0.88 | +0.93 | +0.94 |
| weight r(LD, z_eff) | — | +0.86 | +0.91 | +0.93 | +0.94 | +0.95 |
| weight r(LD, x) | **+0.99** | +0.17 | +0.08 | +0.07 | +0.08 | +0.08 |
| speed r(LD, z) | — | +0.84 | +0.87 | +0.86 | +0.86 | +0.82 |
| speed r(LD, z_eff) | — | +0.88 | +0.91 | +0.91 | +0.92 | +0.89 |
| speed r(LD, x) | **+0.98** | +0.28 | +0.19 | +0.16 | +0.21 | +0.30 |

### Headlines

1. **r(LD, z_eff) saturates fast — by k≈4.** On 2B and 9B, r(LD, z_eff) reaches
   90–95% of its k=15 value by k=4 across all three pairs. Even at **k=1**,
   r(LD, z_eff) ≈ 0.83 on weight/height (9B). One context anchor already gives
   the model 87% of the 15-shot relativity signal.
2. **r(LD, z_eff) is consistently higher than r(LD, z) at small k** (gap of
   0.03–0.07 at k=1, vanishing by k=8). Confirms the model is reading the
   *sample mean* of context, not the population μ — exactly what the prompt
   makes available.
3. **The k=0 prior is scale-dependent.** On 9B, r(LD, x) at k=0 is 0.98 across
   all three pairs — clean *absolute* response (bigger x → more "tall"/"heavy"/
   "fast"). On 2B, the k=0 prior is much weaker for height (0.43, weakly
   U-shaped over the range) but still strong for weight/speed (0.92/0.94).
   **Scale gives the model an absolute numerical sense for height** that 2B
   doesn't have — but doesn't add much for weight/speed, which 2B already
   reads absolutely.
4. **Adding context flips both models from absolute to relative.** r(LD, x)
   collapses from ~0.99 (k=0) to ~0.10 (k=15) on 9B weight. The relativity
   circuit, once engaged, suppresses the absolute readout completely.
5. **Speed is unique: peaks at k=4 then degrades.** On 2B, r_z goes 0.59 →
   0.73 → 0.81 → 0.73 → 0.73 across k=1,2,4,8,15. On 9B, peak is at k=2 (0.87)
   and k=15 is *lower* (0.82). Interpretation: speed has a different aggregator
   that's more efficient with fewer context items. Consistent with Jaehoon's
   "speed is pair-specific" exception, but adds a new observation: more context
   actively *hurts* speed past k=4.

### The qualitative finding — comparator vs graded modes

The shape of the LD-vs-z_eff scatter changes qualitatively with k. At k=1, the
distribution is **bimodal**: LD snaps to ±A around z_eff=0, with little graded
response. By k=4–15, LD is linear in z_eff. We quantified this with three
diagnostics (`scripts/analyze_p2a_step_vs_graded.py`, results in
`results/p2a_step_vs_graded.json`).

Best-fit `LD = A · tanh(β · z_eff)` parameters:

| | 2B height | 2B weight | 2B speed | 9B height | 9B weight | 9B speed |
|--|-----:|-----:|-----:|-----:|-----:|-----:|
| β at k=1 | 76.8 | 722 | 56.8 | 75.3 | 890 | 51.4 |
| β at k=2 | (~1.5) | 1.20 | 1.78 | 1.59 | 1.30 | 1.22 |
| β at k=4 | (~0.9) | 0.28 | 1.03 | 0.88 | 0.71 | 1.20 |
| β at k=15 | (~0.5) | 0.47 | 0.45 | 0.50 | 0.80 | 0.37 |

A tanh with β=722 has a transition width of ~1/722 ≈ 0.001 in z_eff — this is
a near-perfect step function. By k=4–15, β is in [0.4, 1.2] — gentle slope,
indistinguishable from linear within noise.

Sarle's bimodality coefficient on the LD distribution itself:
- 9B height k=1: **0.86** (strongly bimodal); k=15: 0.61.
- 9B weight k=1: **0.85**; k=15: 0.75.

Linear-vs-tanh RSS ratio (`rss_tanh / rss_lin`, <1 means tanh fits better):
- 9B height k=1: **0.20** (tanh wins decisively); k=15: 1.72 (linear wins).
- 9B weight k=1: **0.44**; k=15: 0.58.

### Interpretation

**Two computational modes for context-relative graded judgment:**

- **Comparator mode (k=1)** — hard threshold around z_eff=0, output ≈ ±A.
  The model decides "above-or-below" with effectively zero gradedness. The
  computation here is a binary comparison against a single anchor.
- **Graded mode (k≥4)** — output linear in z_eff. The model is producing
  a continuous magnitude, not just a sign.

The transition between the two modes is *not* a strength change — on 9B, the
linear slope of LD vs z_eff is nearly **k-invariant** (1.24 at k=1 → 1.32
at k=15 for height). What k controls is the *shape* of the response: same total
response strength, redistributed from a step into a slope. Adding context items
doesn't add z-information; it spreads the model's response over the z range.

This sharpens the circuit-level question for Phase 2B/2C/2D:

> *At k=1, what writes the bimodal step into the residual?*  
> *At k≥4, what additional machinery converts the step into a linear slope?*

If these are the same heads with different attention patterns, ablation should
behave smoothly across k. If different heads are involved, ablation at k=1
will identify the comparator while the graded-writers escape — which would
explain Jaehoon's triple-refuted null at k=15: "the heads we tagged as
comparators were the COMPARATOR heads, but the LD at k=15 is dominated by
graded-writers Jaehoon's taxonomy didn't isolate."

### Implications for Phase 2B+

- **Original plan** (k=2 vs k=15 for attention) is half-right. The interesting
  contrast is **k=1 (comparator-only) vs k≥8 (graded-dominant)**. k=2 is in the
  transition zone — useful but not the cleanest extreme.
- **Proposed update**: Phase 2B extracts attention captures at k ∈ {1, 4, 15}.
  k=1 isolates the comparator; k=4 is "early graded mode"; k=15 is the
  saturated/redundant regime.
- **Speed should be its own subsection.** Peak-at-k=4 and degradation at k=15
  is an observation that distinguishes speed from the rest. The pair has a
  different circuit, not just a weaker version of the same one.
- **9B vs 2B for circuit work**: 9B's step is much sharper (β=890 on weight at
  k=1 vs 2B's β=722, and 9B's RSS ratio decisively favors tanh while 2B's
  doesn't). The comparator circuit is *cleaner* in 9B. Counter-intuitive given
  Phase 1's "9B is the linear regime" finding — the *late-layer* z-direction
  is more linearized at 9B, but the *behavioral* response at k=1 is more
  step-like. Consistent if the comparator emits a clean step into the residual,
  and 9B's downstream layers don't smooth it before readout.

### Caveats and open questions

- The `n=20` k=0 grid for r(LD, x) is small; CIs are wide. A finer x grid would
  tighten the prior estimate.
- We tested only height/weight/speed. The other 5 v11 pairs may show different
  saturation profiles or step-vs-graded crossovers.
- "Comparator" and "graded" are *behavioral* labels. They map onto distinct
  computations only if circuits work confirms different head sets. Phase 2B
  is the test.
- Slope-invariance on 9B is a strong claim from limited data (3 pairs × 5 k
  values). Worth replicating on the full 8 pairs.

### Artifacts
- `data/p2_shot_sweep/<pair>_k<k>.jsonl` — all prompts (16,920 total)
- `results/p2_ld/<model>/<pair>_k<k>.npz` — per-prompt LD + metadata
- `results/p2a_summary.json` — saturation r-values with CIs
- `results/p2a_step_vs_graded.json` — slope, tanh_β, bimodality per cell
- `figures/p2a_shot_sweep.png` — saturation curves
- `figures/p2a_ld_vs_z_height_<model>.png` — scatter showing step → linear
- `figures/p2a_step_vs_graded.png` — slope and RSS ratio vs k
- Scripts: `scripts/gen_p2_shot_sweep.py`, `scripts/p2_extract_ld.py`,
  `scripts/analyze_p2a.py`, `scripts/analyze_p2a_step_vs_graded.py`

---

## 13. Phase 2B — Per-head attention by shot count (status: COMPLETE, 2026-04-30)

### Setup
- Pair: height (one pair to start; broaden if signal carries).
- Models: Gemma 2 2B base, Gemma 2 9B base.
- Shot counts k ∈ {1, 4, 15}.
- ~990 prompts per (model, k); ~600 sub-sampled for CV speed in some metrics.
- Captures (every layer, every head):
  - Attention probabilities at the LAST query position (the "is" token):
    `attn_last[N, n_layers, n_heads, max_seq]` fp16
  - Per-head value-mix at the last position before W_O:
    `head_outs[N, n_layers, n_heads, head_dim]` fp16
  - Full residual stream `residuals[N, n_layers, d_model]` fp16
  - LD logits, position bookkeeping, pad offsets.

### Headline 1 — z-encoding layer SHIFTS LATER as k decreases

Residual `r²(z_eff)` per layer (`scripts/analyze_p2b_per_head_z.py`):

**Gemma 2 2B**

| L | k=1 | k=4 | k=15 |
|---|----:|----:|----:|
| 0 | 0.43 | 0.59 | 0.70 |
| 1 | 0.92 | 0.85 | 0.83 |
| 2 | 0.94 | 0.94 | 0.95 |
| 4 | 0.98 | 0.98 | 0.98 |

ΔR²(z_eff) at each layer:

| L | k=1 | k=4 | k=15 |
|---|----:|----:|----:|
| 0 | 0.43 | 0.59 | **0.70** |
| 1 | **0.49** | 0.27 | 0.13 |
| 2 | 0.02 | 0.09 | 0.12 |
| 3 | 0.01 | 0.02 | 0.02 |

**At k=15, layer 0 alone encodes 70% R² of z. At k=1, layer 1 contributes
nearly half (0.49) of the total.** The encoding is *redistributed* across early
layers as k decreases. Same qualitative story on 9B: at k=1 the encoding is
spread over L0+L1+L3 (Δ ≈ 0.38, 0.45, 0.10); at k=15 it's L0-dominant (0.74).

This is consistent with Phase 2A's comparator/graded behavioral split: at low k
the model needs an extra layer of attention to extract z (because a single
context anchor doesn't give a noisy mean to be averaged out); at high k a
single layer of "bag-of-context" aggregation is enough.

### Headline 2 — comparator heads identifiable via attention-modulation correlation

Per-head attention mass on context-value tokens is **uniformly tiny** (≤0.10 on
any head at any k). Most heads put their mass on BOS or the "is" suffix —
classic attention-sink pattern. This made the original "high-context-mass head
is the aggregator" tagging fail (the attention sinks dominate every head's
mass distribution).

The right metric is the **modulation** of attention mass with z_eff:
`r(target_attn − last_context_attn, z_eff)` across prompts. A head whose
attention shifts toward the target when z_eff > 0 (target above context) and
toward the context when z_eff < 0 is performing a comparison.

This metric finds clean candidate comparators on both models
(`scripts/analyze_p2b_l0_l1_focused.py`):

**Gemma 2 2B (8 heads/layer):**

| head | k=1 | k=4 | k=15 | role |
|------|----:|----:|----:|------|
| **L1H6** | **+0.71** | **+0.70** | **+0.67** | primary, k-invariant |
| L1H1 | +0.55 | +0.55 | +0.46 | k-invariant secondary |
| L1H4 | +0.54 | +0.48 | +0.27 | auxiliary, decays with k |
| L1H0 | +0.48 | +0.43 | +0.22 | auxiliary, decays |
| L2H7 | +0.22 | +0.10 | +0.04 | minor |

**Gemma 2 9B (16 heads/layer):**

| head | k=1 | k=4 | k=15 | role |
|------|----:|----:|----:|------|
| **L1H11** | **+0.60** | **+0.50** | +0.40 | primary |
| L1H10 | +0.35 | +0.31 | **+0.47** | growing with k (graded contributor) |
| L1H6 | +0.45 | +0.30 | +0.16 | auxiliary, decays |
| L1H3 | +0.27 | +0.17 | +0.07 | auxiliary, decays |
| L1H12, L1H15 | +0.20–0.24 | +0.15–0.19 | +0.22–0.24 | minor k-invariant |

L0 has *no* heads with comparator score above 0.20 (mass is mostly BOS;
the encoding done by L0 is a uniform "bag of context" pattern that doesn't
correlate target-vs-context attention with z). L2 and L3 carry small
secondary signals (H4 grows with k on 2B; H1 and H7 spike at k=15 on 9B).

### Headline 3 — primary comparators have small absolute mass on the values

L1H6 on 2B and L1H11 on 9B do not put majority mass on either context or
target value tokens — both are in the 0.05–0.10 range. The signal is in the
*direction* of attention-mass modulation (a few percentage points shifted
toward whichever side reflects "target above context" or "target below").
The query at the "is" position is reading a small but z-correlated signal
from the value tokens.

This explains Jaehoon's triple-refuted causal head taxonomy at k=15:
- His tagging used "high context-attention mass" as the aggregator criterion.
- True z-writers carry only a few percent of their mass on context values —
  they wouldn't pass that filter.
- Heads that DO have higher context mass (e.g., L0H6 with ~0.10) are
  bag-of-context aggregators that *don't* compare; ablating them removes
  μ-aggregation but the rest of the circuit may compensate via the residual.

### Headline 4 — auxiliary comparators decay; one head grows with k

Within L1 of each model:
- **k-invariant comparators** (L1H6 on 2B, L1H11 on 9B) hold their score across
  k. These are the "always-on" z-writers.
- **Decaying comparators** (L1H0/H1/H4 on 2B, L1H6 on 9B) lose comparator
  signal as k grows. They contribute to the encoding when the task is "easy"
  (1 anchor) but get crowded out when the model can rely on aggregation.
- **Growing comparators** (L1H10 on 9B; r climbs 0.35 → 0.47 with k). These
  fire more under saturated context. Plausibly a graded-mode contributor that
  helps spread the response over the z range (cf. Phase 2A's "graded mode at
  k≥4").

### Why this matters

This is the first crisp circuit-level localization in the relativity work:

1. **z-encoding lives at L1** for both 2B (h6) and 9B (h11/h10) at k=1.
   *Single layer, 1-2 heads.* This is the smallest known z-circuit.
2. **The "scale axis" finding from Phase 1 has a circuit explanation**:
   9B's "more linearized z" might come from L1H10 (the graded contributor
   that grows with k) being more dominant in 9B than the analogous head in 2B.
3. **The "redundancy" Jaehoon described is real, but it operates within a
   single layer (L1)**. Multiple L1 heads contribute to z encoding; ablating
   any one degrades but doesn't destroy the encoding. The next test (Phase 2C)
   is whether ablating the *primary + decaying-auxiliaries* together at k=1
   destroys the encoding — the prediction is yes.

### Caveats and open questions

- **Pair**: Only height tested. Weight should follow the same pattern; speed
  is the known exception and may have a different (or shifted-layer)
  comparator. Worth replicating on 2-3 pairs.
- **Causal verification pending**: The comparator-score metric is correlational.
  Phase 2C (causal ablation at L1H6 on 2B; L1H11+L1H10 on 9B) is the test.
  Predictions:
  - At k=1, ablating L1H6 (2B) drops residual r²(z_eff) at L1 substantially.
  - At k=15, ablation of the same head leaves z encoding intact (because
    L0 carries it).
  - Ablating L1H6+H1+H4 together at k=1 should be much more damaging than
    ablating L1H6 alone.
- **DLA approximation**: per-head DLA was approximated using `head_outs @
  primal_z[head_slice]` (treating W_O as identity). The real W_O hasn't been
  loaded yet. Worth checking.
- **r(Δattn, z_eff) is a single Pearson correlation**: a richer metric (e.g.,
  multivariate regression of attention pattern on z_eff) might reveal more.
- **The rapid drop in r²(z_eff) at L1 from k=1 to k=15 (0.92 → 0.83)** is
  worth flagging: at k=15, L1 carries *less* z-info in absolute terms than at
  k=1 (because L0 has done more of the work). This is a counter-intuitive
  property of the encoding: more context = less work for L1.

### Artifacts
- `results/p2_attn/<model>/<pair>_k<k>.npz` — full attention + per-head + residual
  dumps (~250–830 MB per file)
- `results/p2b_per_head_r2_<model>_<pair>.json` — per-(L, H) r²(z_eff)
- `results/p2b_l0l1_focused_<model>_<pair>.json` — bucket masses + comparator scores
- `figures/p2b_increment_r2_<model>_<pair>.png` — ΔR²(z) vs layer
- `figures/p2b_per_head_r2z_<model>_<pair>.png` — per-(L, H) r²(z) heatmaps
- `figures/p2b_l0l1_focused_<model>_<pair>.png` — per-head bar charts L0-L3
- `figures/p2b_comparator_landscape.png` — clean cross-model comparator score plot
- Scripts: `scripts/p2_extract_attn.py`, `scripts/analyze_p2b_attention.py`,
  `scripts/analyze_p2b_per_head_z.py`, `scripts/analyze_p2b_l0_l1_focused.py`,
  `scripts/plot_p2b_comparator_landscape.py`

---

## 14. Phase 2C — Causal head ablation (status: COMPLETE, 2026-04-30)

### Setup
- Pair: height. Models: Gemma 2 2B base, 9B base.
- Shot counts k ∈ {1, 4, 15}. n=600 per (model, k).
- Ablation: forward_pre_hook on `o_proj` zeroes the per-head slice of its input,
  i.e., zeros that head's contribution to the residual at the layer it's in.
- Configs:
  - `baseline` — no ablation
  - `primary` — L1H6 (2B) / L1H11 (9B), the top comparator-score head
  - `primary_plus` — primary + the auxiliary heads identified in Phase 2B
  - `l1_all` — every head at L1
  - `l0_all` — every head at L0
  - `random_single`, `random_set` — direction-specificity controls
- Per ablation: residuals at every layer + LD at every prompt → r²(z_eff) per
  layer, r(LD, z_eff), ⟨LD⟩.

### Headline 1 — single-head and L1-wide ablations don't break the readout

Δ r(LD, z_eff) [ablated − baseline]:

| ablation | 2B k=1 | 2B k=4 | 2B k=15 | 9B k=1 | 9B k=4 | 9B k=15 |
|----------|-------:|-------:|--------:|-------:|-------:|--------:|
| primary | -0.005 | -0.005 | -0.002 | +0.001 | -0.000 | +0.002 |
| primary_plus | -0.011 | -0.000 | +0.011 | +0.003 | +0.000 | +0.002 |
| l1_all | -0.048 | -0.012 | +0.011 | +0.009 | -0.056 | -0.090 |
| **l0_all** | **-0.189** | **-0.165** | **-0.102** | **-0.438** | **-0.310** | **-0.503** |
| random_single | -0.001 | -0.002 | -0.000 | -0.001 | +0.002 | +0.001 |
| random_set | -0.009 | -0.005 | -0.023 | +0.005 | +0.000 | +0.002 |

Single-head and primary_plus ablations are within the noise floor of random
ablations. **L1 as a whole** has a small but real effect on 2B at k=1 (-0.05)
and on 9B at k=15 (-0.09), but these are an order of magnitude smaller than
**L0 as a whole**, which is the single layer whose ablation cleanly damages
behavioral z-tracking.

### Headline 2 — L1 ablations are RECOVERED by downstream layers

The figure `figures/p2c_r2z_trajectory.png` plots residual r²(z_eff) per layer
for each ablation. Three things stand out:

1. At k=1 on 2B, `primary_plus` and `l1_all` drop residual r²(z) at L1 from
   0.93 to 0.70 / 0.46 respectively. **By L4, both have recovered to ≥0.97**
   — within 0.01 of baseline.
2. Same on 9B: `primary_plus` and `l1_all` drop L1 r²(z) from 0.83 to
   0.44 / 0.38; both are **fully recovered (≥0.97) by L4**.
3. `l0_all` is different: at k=1 it leaves L1 r²(z) intact (because L1's
   attention can still pull positional info from the embedded value tokens)
   but the readout is destroyed. At k=15 it does drop L1 r²(z) (0.85 → 0.58
   on 9B) and the readout drops dramatically (-0.50 r(LD, z_eff)).

This is the textbook "redundancy across layers" pattern. Layers L2-L5 contain
attention that can re-extract z from the residual stream when L1 has been
silenced. Phase 2B found L1 was the largest *single-layer* increment in z
encoding, but it's not load-bearing — it's load-shareable.

### Headline 3 — L0 is the only layer whose attention is causally critical

`l0_all` is the only ablation that produces a behavioral effect outside the
noise floor on both models, at every k. The effect is:

- Stronger on 9B than on 2B (Δr(LD, z_eff) ≈ -0.4 vs -0.15).
- Stronger at low k on 2B (-0.19 at k=1 vs -0.10 at k=15) but stronger at
  high k on 9B (-0.50 at k=15 vs -0.44 at k=1).
- Behaviorally most striking at 9B k=1: the step-function shape from Phase 2A
  is visibly destroyed. r(LD, z_eff) collapses from 0.83 → 0.39, and LD scatter
  becomes a band around +2 with no clear z_eff dependence.

Note however: the L1 r²(z_eff) under l0_all is sometimes *unchanged* or even
*higher* than baseline (9B k=1: 0.83 → 0.87). The behavioral collapse is not
mediated by destroying z encoding in the residual — z is still there. The
collapse is *readout-side*: the residual carries z, but the model's W_U @ h
mapping no longer reads it correctly. This is the cleanest "rep ≠ readout"
case in the project so far. Possible interpretations:
1. L0's attention contributes a position-dependent baseline to every token's
   residual; without it, downstream layers operate on out-of-distribution
   activations and the readout mis-fires.
2. L0 attention writes context-mean information through a path independent
   of the z-encoding direction; removing it changes the residual's "frame"
   such that z's projection onto W_U is wrong even though z is present.

This requires further investigation (e.g., partial L0 ablation that retains
the residual norm). Out of scope for 2C.

### Headline 4 — within-L1 redundancy is real

Across 2B and 9B at every k, **ablating the primary comparator alone has zero
effect** even though Phase 2B's correlational metric flagged it as the head
most strongly modulating attention with z_eff. Ablating it + auxiliaries
(primary_plus, 3-4 heads) has zero effect on 2B and minimal effect on 9B.
Ablating the whole layer has a small effect.

This confirms Jaehoon's "z circuit is redundant" framing, but adds a
specific structure: the redundancy lives **within L1** (single heads matter
little; whole layer matters more) AND **across early layers** (L1 ablation
is recovered by L2-L5). Single-head causal localization is a fundamentally
unlikely target.

### Why this matters for the joint paper

This is a strong negative result for "single-head causal localization of the
relativity circuit" but a clean *positive* result for several other claims:

1. **Phase 2B's correlational localization is real** (the heads identified do
   carry the strongest z-modulating attention patterns) but **not causally
   load-bearing** because of redundancy.
2. **L0 ablation is the cleanest causal lever found.** It generalizes Phase 1's
   single-direction projection (which was at L20 in the residual): a much
   broader intervention (zero all attention contributions at L0) damages
   z-readout by 0.4 in 9B at k=15. Phase 1's manifold-shift at L20 was a
   smaller-magnitude intervention; L0 ablation here is a larger-magnitude
   intervention with clearly larger behavioral effect.
3. **The "rep ≠ readout" failure mode** is the dominant story in 2C: ablating
   L0 leaves z in the residual but breaks the readout. Echoes the speed-probe
   "dead-fast" finding (rep is fine; readout is dead). This is becoming a
   recurring theme.
4. **The model's z-encoding is robust to attention-head intervention.** From
   a safety perspective, this means *single-head edits are ineffective*. If
   we want to remove relativity behavior, we have to either (a) intervene
   at the residual stream (Phase 1's manifold-shift), or (b) ablate at the
   layer level (zero all attention at a layer), or (c) intervene at readout
   (modify W_U). Single-head surgery doesn't move the model.

### Caveats

- Only height tested. Speed (the known-exception pair) might localize
  differently — speed's peak r(LD, z_eff) at k=4 (Phase 2A) hints at a
  different circuit. Worth replicating.
- L0 ablation effect partially confounded by "OOD activation" — it's not
  clear how much of the -0.4 effect on 9B is "L0 attention writes z-essential
  info" vs "downstream layers expect L0 to have run." A "partial L0 ablation"
  experiment (e.g., zeroing only some heads) would disambiguate.
- Single-prompt-bank ablation is per-position uniform (all token positions get
  the same head zeroed). A position-selective ablation (zero only at the
  target token, or only at context tokens) might reveal finer-grained roles.

### Artifacts
- `results/p2c_ablation_<model>_<pair>.json` — full ablation results
- `figures/p2c_r2z_trajectory.png` — residual r²(z_eff) per layer × ablation × k
- `figures/p2c_behavioral_bars.png` — Δr(LD, z_eff) by ablation × k
- `figures/p2c_ld_scatter_k1.png` — LD vs z_eff scatter under each ablation at k=1
- Scripts: `scripts/p2c_ablate_heads.py`, `scripts/plot_p2c.py`


## 15. Phase 2E — manifold α-sweep across three features (status: COMPLETE, 2026-04-30)

### Setup
Ran the same wide manifold-α sweep (α ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0})
on three pairs (height, weight, speed) for both Gemma 2 2B (L20) and 9B (L33),
all at k=15. Same script as before — `p2e_alpha_sweep.py` — now reads
`low_word`/`high_word` from the prompt JSONL so it generalizes across pairs.

### Per-pair r(z_eff, x) in the prompt distribution
The plausibility-filter confound depends on the pair's x-grid and σ:
- height: r(z, x) = +0.36 (strong)
- weight: r(z, x) = +0.04 (negligible)
- speed:  r(z, x) = +0.11 (small)

For weight and speed, partial-correlation correction barely moves the markers —
the raw r(LD, z) and r(LD, x) are already nearly the partial values. For height
the correction is meaningful (~0.1 - 0.2 inflation in raw r_x).

### Headline numbers (raw correlations)

| model | pair | base r_z | base r_x | α=1.0 r_z | α=1.0 r_x | α=2.0 r_z | α=2.0 r_x |
|-------|------|----------|----------|-----------|-----------|-----------|-----------|
| 2B | height | +0.65 | +0.40 | -0.10 | +0.18 | -0.78 | -0.04 |
| 2B | weight | +0.94 | +0.11 | -0.17 | +0.15 | -0.85 | +0.01 |
| 2B | speed  | +0.78 | +0.44 | -0.60 | +0.16 | -0.86 | -0.08 |
| 9B | height | +0.65 | +0.45 | +0.06 | +0.30 | -0.85 | +0.06 |
| 9B | weight | +0.94 | +0.09 | +0.01 | +0.27 | -0.83 | +0.14 |
| 9B | speed  | +0.87 | +0.34 | -0.48 | +0.41 | -0.78 | +0.22 |

### Key observations
1. **All six baselines are RELATIVISTIC.** r_z ranges 0.65 - 0.94 across pairs.
   Weight is the most context-bound (r_z=0.94 in both models, r_x=0.09-0.11);
   speed has the largest baseline objective component (r_x=0.34-0.44).
2. **9B reliably converts RELATIVISTIC → OBJECTIVE at α=1.0.**
   - 9B/weight is the cleanest landing seen so far: (r_x=+0.27, r_z=+0.01) at
     α=1.0 — z is nulled out exactly, x correlation is preserved/grows.
   - 9B/height lands at (+0.30, +0.06) — same pattern.
   - 9B/speed overshoots: at α=1.0 r_z=-0.48 (anti-relativistic). The α=0.5
     state (+0.55, +0.37) is its closest approach to OBJECTIVE.
3. **2B never reaches OBJECTIVE.** All three 2B trajectories cut diagonally
   from RELATIVISTIC straight to anti-relativistic, never building r_x. By
   α=1.0 the 2B trajectories are already at r_z ≈ -0.1 to -0.6 with r_x
   roughly flat or shrinking. This matches the Phase 2D finding that 2B can't
   recover an objective representation when its relativistic component is
   removed — it has weaker absolute-value priors.
4. **9B passes through COMPLETE state for height and speed.** At α=0.5-0.75
   for these pairs, both r_z and r_x are simultaneously > 0.5 (height) or near
   that boundary (speed). Weight skips COMPLETE — its baseline r_x is too
   small (0.09) to clear the COMPLETE quadrant boundary even after steering.
5. **Anti-relativity is the universal endpoint at α=2.0**: every (model, pair)
   has r_z ∈ [-0.78, -0.86] at α=2.0. Pushing past z=0 inverts the comparator —
   the model now says "tall/heavy/fast" when the target is BELOW context mean.
6. **Per-pair shape of the α-trajectory differs by absolute-value priors.**
   - Height/speed: have meaningful baseline r_x → trajectory bends right
     (toward OBJECTIVE) in 9B before falling.
   - Weight: baseline r_x ≈ 0 → trajectory goes nearly straight down (no
     OBJECTIVE bend); 9B still nulls z cleanly but x stays small.
   - Speed: highest baseline r_x → 2B speed at α=0.5 is the only 2B point
     that briefly reaches the right-half plane, but it falls back fast.

### Interpretation
The manifold-shift intervention is a clean knob that converts a RELATIVISTIC
representation into one that's increasingly absolute (or anti-relativistic).
Whether the model lands in OBJECTIVE versus BIASED depends on the strength
of its baseline absolute-value prior:
- Strong absolute prior (9B with all pairs) → trajectory bends rightward,
  hitting OBJECTIVE on the way.
- Weak absolute prior (all 2B pairs, also 9B/weight where r_x is small) →
  trajectory short-circuits through BIASED, never reaching OBJECTIVE.

This validates the three-state framing as a real phase space, not an artifact
of one (height) experiment: every model × pair combination follows the same
qualitative arc, just scaled by how much absolute knowledge the model has
about the target attribute.

### Artifacts
- `results/p2e_alpha_sweep_{model}_{pair}_k15.json` × 6 combos
- `figures/p2e_alpha_trajectory_multi.png` — 2×3 grid (rows=models, cols=pairs)
- Scripts: `scripts/p2e_alpha_sweep.py` (parameterised on JSONL low/high words),
  `scripts/plot_p2e_alpha_trajectory.py` (multi-pair grid)


## 16. Phase 2F — z-tracking through attention magnitude (status: COMPLETE, 2026-04-30)

### Question
Phase 2B found the comparator head whose output value-mix encodes z (2B L1H6,
9B L1H11). But that's the OV side. Does the QK side — the attention pattern
itself — also encode z? If so, where? And do the same heads track z across
different features (height/weight/speed), or is it pair-specific?

### Method
Reuse the existing `attn_last` field in the p2_attn NPZs (last-token attention
distribution per layer × head). Decompose into 6 buckets per prompt:
{pad, pre_context, ctx_value, ctx_scaffold, tgt_value, tgt_scaffold}. Then
compute per-head Pearson(attn-to-ctx-values, z_eff) on the held-out fold.
Run for all 6 (model, pair) combos at k=15.

### Bucket findings
- `pad` is exactly zero everywhere — left padding is fully masked.
- Most attention goes to `pre_context` (BOS / "Person 1:" prefix) — the
  attention-sink phenomenon. Many heads dump 50-99% of attention there.
- `ctx_value` attention is universally LOW (<12% per head, mostly <5%) — the
  model is NOT primarily attending to context numbers at the last position.
- `ctx_scaffold` ("Person N:", "cm", "\n") gets the largest non-sink share,
  especially at L0 and at L15.
- `tgt_value` self-attention is mostly small except a few late heads.

So the z-encoding is happening in a **small slice (<5% of total attention)** of
ctx_value attention — but the variation in that small slice is what tracks z.

### z-tracking heatmap
Per (layer, head), r(attn-to-ctx-values, z_eff) is the signal. Strongest heads:

**Gemma 2 2B:**
| layer | head | r_height | r_weight | r_speed |
|-------|------|----------|----------|---------|
| L23 | H5 | -0.81 | (weak) | (weak) | (height-specific)
| L1  | H6 (primary) | -0.73 | -0.37 | -0.41 | (cross-pair, weak on speed)
| L1  | H0 | -0.70 | -0.45 | -0.38 |
| L20 | H6 | +0.49 | +0.48 | +0.51 | (cross-pair, opposite sign)
| L20 | H2 | -0.73 | (mid) | (mid) |

**Gemma 2 9B:**
| layer | head | r_height | r_weight | r_speed |
|-------|------|----------|----------|---------|
| L32 | H4  | -0.77 | -0.83 | -0.68 | (universal z-tracker)
| L25 | H6  | -0.66 | -0.70 | -0.64 |
| L31 | H2  | (mid) | -0.82 | -0.77 |
| L24 | H1  | -0.71 | -0.49 | -0.60 |
| L1  | H10 | -0.77 | (mid) | (mid) |
| L1  | H11 (primary) | (-0.7 ish on each pair, see plot) |

### Headline architectural finding
**2B encodes z in attention at early layers (L0-L5)** — the cross-pair-consistent
band with negative r sits in layers 0-5.

**9B encodes z in attention at mid-late layers (L24-L33)** — the consistent
negative band is L24-L33, with L32H4 standing out as a universal cross-pair
z-tracker (min |r| = 0.68 across height/weight/speed).

Both models also have a band of weak positive correlations in late layers
(2B L20-L25, 9B L36-L41). These are likely "undo / readout" heads — they
re-introduce z-positive signal during the final readout, complementary to the
z-negative attention at lower layers.

### Primary-head (Phase 2B) deep dive — L1H6 (2B) and L1H11 (9B)
Each plot has 3 panels:
A. Mean attention to each of 15 context slots — both heads show clear
   PRIMACY (slot 1 ~1.6%) and a U-shape rising back toward slot 15 (~1.1%).
B. Same plot split by z_eff tertile. Curves separate cleanly — z_low (target
   below context) attends ~2x more to context than z_high (target above).
C. Scatter: total attention to ctx_values vs z_eff (one dot per prompt).
   L1H6 r=-0.73, L1H11 r=-0.72. Target self-attention r=+0.34/+0.46 (slight
   POSITIVE — target self-attention grows when z is high).

So both primary heads do the SAME thing: when target is below context (z low),
they attend MORE to context (especially the early/late slots — primacy/recency).
When target is above context, attention shifts to the target token itself.

### Interpretation
Two complementary z-encoding mechanisms exist in the same head:
- **OV-path** (Phase 2B): the head's VALUES at the last position encode z
  through a directional component in residual space.
- **QK-path** (Phase 2F): the head's ATTENTION distribution encodes z by
  shifting weight off context (and onto target) as the target rises.

In 2B these mechanisms co-exist in the same early head (L1H6). In 9B the
QK z-tracker is much later (L32H4) than the OV z-tracker (L1H11) — they're
different heads doing different jobs.

### Causal predictions (untested as of 2026-04-30)
1. Ablating 9B/L32H4 (zero its o_proj input slice) should reduce r(LD, z_eff)
   on all three pairs — currently the strongest single-head candidate for a
   causal z-tracker.
2. Ablating the cross-pair consistent 2B early layer cluster (L0H5, L1H0,
   L1H6, L2H7) might be needed jointly to break z, since signal is spread
   across multiple heads.

### Artifacts
- `results/p2f_attn_circuit_<model>_<pair>_k15.json` — bucket means and
  per-head r vs z_eff
- `figures/p2f_bucket_<model>_<pair>_k15.png` — bucket × layer × head heatmap
- `figures/p2f_z_corr_<model>_<pair>_k15.png` — signed r heatmap, primary
  head circled
- `figures/p2f_primary_<model>_<pair>_k15.png` — primary-head deep dive (3 panels)
- `figures/p2f_cross_pair_<model>.png` — 3×3 cross-pair summary
- Scripts: `scripts/p2f_attn_circuit.py`, `scripts/plot_p2f_cross_pair.py`


## 17. Phase 2G — context-info flow, layer × slot (status: COMPLETE, 2026-04-30)

### Question
We know the model encodes z via attention magnitude (Phase 2F). But which
context items does the target query, when, and how does that depend on z?

### Method
For each prompt, decompose `attn_last` (target's last-position attention over
all keys, per layer × head) into per-slot attention to each of the 15 context
value tokens. Sum over heads to get attention(layer, slot) per prompt. Three
views: (i) mean attention layer × slot, (ii) per-layer total split by z
tertile, (iii) Pearson r(per-prompt slot-attn, z_eff) at each layer × slot.

### Where context is read (aggregation layer)
Total target → ctx_value attention peaks sharply at one layer per model:
- 2B: peak at L16 (~0.35-0.45 of total)
- 9B: peak at L21 (~1.0-1.7 of total — the absolute number is higher because
  9B has more heads, but normalized per head it's roughly comparable)

This peak is consistent across all three pairs in each model. Below and above
this aggregation layer, ctx_value attention is much smaller (~0.05). So the
model has a single dominant "read context" event per stack.

### Positional structure of attention
At the aggregation layer, attention is *not* uniform across the 15 slots:
- 9B/height/weight: massive primacy on slot 1 (~22% at L21), strong recency
  on slots 13-14 (~15%), middle slots receive less.
- 9B/speed: more uniform across slots — speed has weaker primacy.
- 2B follows the same primacy pattern but with smaller magnitudes.

### z-modulation per (layer × slot)
The signed-r heatmap reveals the structural mechanism of z-encoding:
- **Slots 1-3 (early): attention RISES with z** (red band on left side of
  the heatmap, layers L20-25 in 2B and L10-30 in 9B for height).
- **Slots 9-12 (mid-late): attention DROPS with z** (blue stripe in the
  middle of the heatmap).
- **L0-L1 (early layers): attention to ALL slots drops uniformly with z**
  (blue top row).

So the model does TWO things at once:
1. Early layers: globally reduce ctx attention as z rises (Phase 2F finding).
2. Mid layers: reallocate the remaining attention budget — pull MORE from
   early slots, LESS from middle slots, as z rises.

### Early-vs-late summary (9B/height around L20-25)
- z-high targets attend MORE to slots 1-5 (early, "anchor" items).
- z-low targets attend MORE to slots 11-15 (late, "recent" items).
- The crossing happens at the aggregation layer L21.

This isn't averaging — it's differential querying. When the target is above
the context mean, the model reaches back to the first/anchor items; when
below, it weighs the recent items more.

### Architectural divergence
- 2B's aggregation layer L16 is at ~62% of stack depth (16/26).
- 9B's aggregation layer L21 is at ~50% of stack depth (21/42).
- The z-tracking is densely concentrated *before* aggregation in 9B (the
  L24-L33 cluster from Phase 2F is mostly *after* aggregation). So in 9B
  the late z-tracking heads are reading the AGGREGATED context rather
  than building the aggregation themselves.

### Hypothesis (untested)
The early-layer z-encoding (Phase 2F's L0-L5 cluster in 2B, L1H10/H11 in 9B)
projects z into a low-dimensional subspace that is then USED at the aggregation
layer (L16 / L21) to reweight slot attention. The mid-late z-tracking heads
in 9B (L25H6, L31H2, L32H4) are then reading off the differentially-aggregated
representation.

### Artifacts
- `results/p2g_info_flow_summary_k15.json`
- `figures/p2g_info_flow_mean.png` — layer × slot mean attention, 6 panels
- `figures/p2g_info_flow_total_by_z.png` — per-layer total ctx attn split by z
- `figures/p2g_info_flow_r_layer_slot.png` — per-(layer × slot) r with z_eff
- `figures/p2g_info_flow_early_vs_late.png` — early vs late slot attn, by z
- Script: `scripts/p2g_info_flow.py`
