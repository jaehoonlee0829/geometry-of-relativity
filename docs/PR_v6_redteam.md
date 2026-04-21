# PR: `exp/v6-redteam` — 4 blocks + self-red-team + direction-confound audit

Executes the experiments in `docs/NEXT_GPU_SESSION_v6.md` (post-v5 red-team
punch list) end-to-end, including 3 parallel skeptical-critic agents run on
the results before the PR opens.

## Blocks

- **A. Park causal inner product** (`Cov(W_U)^{-1/2}`) — replaces v5's
  activation-covariance Σ⁻¹. Well-conditioned (N/d=102, condition number
  1.66e3, no regularizer needed).
- **B. positive / negative math** pair — the cleanest possible absolute
  control (threshold at exactly 0).
- **C. Fisher at peaked activations** — addresses red-team concern that v5
  Exp 3b F ≈ Euclidean was scope-limited to cell means.
- **D. 7-direction steering** × 8 pairs — primal_z, primal_x, probe_z,
  probe_x, meta_w1, zeroshot_wx, PC2 — with α-sweep, softmax entropy,
  on-manifold distance, cross-pair transfer matrix.

Then triggered by critics:

- **Direction-confound audit** — pairwise cosines of all 7 "directions"
  across 8 pairs.

## Headline findings (with honest caveats)

### 1. Park's metric gives the same answer as Euclidean on this model.
`cos(w_z, w_ld)` at late layer shifts by only 0.01–0.03 under
`Cov(W_U)^{-1/2}` vs Euclidean. The v5 "Σ⁻¹ was uninformative" result was
NOT due to the wrong covariance — it survives the correct Park
construction. Scope: this is about where the probes live, not about
Park's method in general. Cov(W_U) has condition number 1.66e3 — not
isotropic globally, just effectively-isotropic in the subspace probes occupy.

### 2. Even positive/negative math shows partial relativity (R = 0.42).
The cleanest possible absolute control (mathematical threshold at 0, no
polysemy) still has slope_μ ≈ 40% of slope_x. **BUT alternative-interpretation
critic flagged** that the prompt "This number is" may condition for
lexical-evaluative completions, not signed-magnitude judgment:
- zero-shot ld(x=0) = +0.5 (biased toward "positive" at the threshold);
- zero-shot ld(x=3) = +1.31 > ld(x=8) = +1.25 (non-monotonicity within the
  5-point sweep suggests a token-competition artifact).
So posneg_abs R=0.42 is *compatible with* "pure lexical prior + small
x-effect"; we cannot claim it's measuring signedness. Welch t (n=5 vs 7):
t=−0.80, p=0.45 — underpowered (power~0.12 for Cohen's d=0.5); the
relative/absolute distinction remains statistically indistinguishable.

### 3. Fisher at "peaked" activations is not peaked.
Block C's premise was that |logit_diff| > p90 samples would have low-entropy
softmax, making F(h) anisotropic. The diagnostic within the same script
falsifies this: at late layer, entropy(peaked) = 2.60 > entropy(flat) = 2.48.
|logit_diff| measures only the gap between 2 of 262k tokens — not softmax
concentration overall. F⁻¹ ≈ Euclidean at both bins. H4 is robust-negative,
but NOT for the reason we tested: we never hit the low-entropy regime where
the theoretical pullback should make a difference.

### 4. The "13×" gap between primal and probe steering is a direction-space
    artifact, not a methodological divide.

Block D measured per-α-unit slope of logit_diff for 7 candidate directions,
all unit-normed. primal_z: ~0.13; probe_z: ~0.01; ratio ~13×.

The direction-confound audit (run after the critic pass) shows:
- primal_z ≈ primal_x ≈ PC1 ≈ −meta_w1   (|cos| 0.80–0.98)
- probe_z ≈ probe_x                        (|cos| 0.54) — orthogonal to the above
- cos(probe_z, primal_z) ≈ 0.08; cos(meta_w1, mean primal_z) = −0.98.

**Reinterpretation**: the 7 candidate directions collapse to ~4 clusters.
"primal" is just steering along the activation-data's principal-variance
axis (PC1), which happens to encode adjective polarity. Ridge probes,
because of shrinkage, pick a low-variance direction that statistically
correlates with z but is nearly-orthogonal to the direction the model
actually uses to write the logit_diff. This RESOLVES v5's "cos(w_z, w_ld) ≈ 0"
mystery: w_z and w_ld are both weak in the PC1 direction that matters.

For the paper: the causal axis to steer along is primal_z / PC1 (or
equivalently meta_w1 with a sign flip). Supervised Ridge probes are
correlates, NOT causal directions. That's the story.

### 5. "On-manifold" claim is circular.
Block D projected steered activations onto PC1/PC2 and computed distance
to a fitted parabola. But: PC1 is ≈ primal_z (the strongest steering
direction), so α·primal_z shifts the PC1 coordinate by α — the activation
MUST leave PC1's 2D slice by construction. The parabola is a 2D projection
of a 2560-D space; any direction outside span(PC1, PC2) is trivially
"off-manifold". This analysis does NOT measure real manifold curvature and
is excluded from the headline.

### 6. Cross-pair transfer is near-noise for probe_z.
Transfer matrix (probe_z learned on A, evaluated on B) shows values |·|<0.05,
often with off-diagonal > diagonal. Stats-critic estimates SE on transfer
≈0.03 — most entries are within 1σ of zero. No random-direction null was
run (another critic flag). The "probe_z does not transfer across pairs"
conclusion would be more convincing with primal_z transfer + random null;
both are future work.

## What the 3 critic agents flagged

All three converged on:

**Statistical (agent A)**: underpower everywhere (Block B n=5, Block D
transfer noise floor ≈0.03), Block A scope-limited (not a global metric
claim), Block C premise self-falsified, no multiple-comparison correction.

**Alternative interpretation (agent B)**: Block B's "This number is" is
lexical-evaluative bias; Block D primal_z and primal_x are effectively the
same direction (confirmed by audit: |cos|=0.91); meta_w1 is −mean primal_z
(confirmed: |cos|=0.98); on-manifold claim is circular.

**Implementation (agent C)**: all directions were correctly unit-normed;
meta_w1 sign is arbitrary but reproducible; empty z-mask for primal_z is
a landmine (didn't trip for these 8 pairs); hook captures POST-steering
activation which is consistent with the analysis if projection basis is
the same (it is — both in fp64).

Implementation critic found **no outright bugs**. Stats critic and
alt-interpretation critic reframe the story without overturning the numbers.

## What this changes for the paper

- **Keep**: the behavioral hero (v5 Exp 5), the v5 meta-direction causal
  steering with random-null control (v5 Exp 2b, 3–29× over random), the
  G31B scaling evidence.
- **Reframe**: the causal direction is **PC1 / primal_z / meta_w1**
  (all ≈ the same thing). Ridge probes (probe_z) are statistical correlates
  that look near-zero in the output direction (w_ld). The "shared polarity
  substrate" story is correct; the mechanism is "the dominant variance
  direction in last-layer activations encodes adjective polarity".
- **Drop or soften**: H4 (Fisher-pullback) — negative across cell-mean,
  flat, and peaked activations. Could be scope-limited to peaked
  SOFTMAX (not |logit_diff|) which wasn't tested.
- **Spectrum framing** for relative/absolute — posneg_abs still shows 42%
  context-dependence, but confounded by prompt design. With 11 pairs
  total (7 relative + 5 absolute-attempts) the dichotomy is p=0.45,
  not significant.
- **Scientific honesty section**: flag the direction-confound story
  explicitly; the "probe_z" and "primal_z" distinction shouldn't be
  overstated as two separate methods.

## Data

- **Git**: new scripts, new analysis JSONs, 6 new figures. ~2 MB total.
- **HF Dataset** `xrong1729/mech-interp-relativity-activations`:
  added `v4_abs_controls/e4b_posneg_abs_*` activations + jsonls.
- **Local Vast disk**: all intermediate files.

## Reproducibility

```bash
git checkout exp/v6-redteam
cd /workspace/repo2
# if .npz gone, fetch from HF
HF_TOKEN=... python scripts/fetch_from_hf.py
# need W_U for Block A + C
python scripts/vast_remote/export_W_U.py e4b
# run experiments
python scripts/vast_remote/exp_v6_park_metric.py            # CPU+GPU ~10 s
python scripts/vast_remote/exp_v6_posneg.py                  # GPU ~1 min
python scripts/vast_remote/exp_v6_fisher_peaked.py           # GPU ~3 min
python scripts/vast_remote/exp_v6_7dir_steering.py           # GPU ~5 min
python scripts/vast_remote/exp_v6_direction_confounds.py     # CPU ~5 s
```
