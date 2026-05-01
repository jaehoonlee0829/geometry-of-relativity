# PLAN — Relativity ablation: text and vision

**Date created:** 2026-04-29
**Companion repo:** `../geometry-of-relativity` (Jaehoon Lee's v11.5 substrate)
**Companion infra:** `../gemma4-size-probe`, `../gemma4-speed-probe`, `../gemma4-bridge-probe`
**Working title:** *Decontextualizing graded judgments: causal removal of context-relativity in language and vision-language models*

---

## 1. Motivation

A model exposed to many examples of "things at speed X" can shift its sense of
what counts as fast. In safety-critical settings (driving, medical triage,
risk classification), this matters: whether something is fast or slow should be
mostly **absolute** within a domain, not anchored to whatever distribution the
context happens to present. Jaehoon's v11.5 work shows Gemma 2 has a clean
domain-agnostic "above-vs-below-the-local-norm" direction (`primal_z`) that
56/56 off-diagonal cells transfer through under FDR and that 6–7/8 pairs are
steered by a single shared direction. That direction is the lever.

**Thesis:** if relativity lives on a single low-dimensional subspace, we should
be able to **ablate** it and recover whatever the model would say *without
context-relative computation*. Then we ask whether that "decontextualized"
output is closer to a stable absolute judgment.

---

## 2. Sharpening "objective"

There is no model-internal ground-truth absolute scale. The model was trained
on human language where "fast" is always relative. The repo's own zero-shot
control finds Gemma has strong logit priors with no context (some pairs prefer
the high adjective at the *lowest* raw `x`). So we cannot validate ablation
against truth.

The right operational definitions:

- **Z-suppression**: after intervention, `corr(LD, z) ≈ 0` on a held-out
  `(x, z)` grid. Measure on cells the intervention was not derived from.
- **X-preservation**: after intervention, `corr(LD, x)` is preserved or rises.
  This is the *literal* objective signal — the raw value the prompt names.
- **Calibration to zero-shot prior**: ablated LD per-x should match (within
  noise) the zero-shot LD per-x measured with no context list.
- **No fluency damage**: perplexity of the ablated model on a held-out text
  corpus (e.g., 200 WikiText sentences) should not move more than the noise
  floor of a random-direction control.

The headline claim becomes: *we can suppress z-tracking without breaking
anything else, and the residual behavior aligns with the model's no-context
prior.* That's defensible. "We made it objective" is not.

---

## 3. Phases

Each phase produces a short writeup in `FINDINGS.md` (success or failure) and
a JSON of metrics. Hard pivot triggers are listed where relevant.

### Phase 0 — Reproduce shared-z direction (smoke test)  [CPU + small GPU]

- Pull the 16 v11 NPZs from HF (`xrong1729/mech-interp-relativity-activations`).
  The `fetch_from_hf.py` in `geometry-of-relativity/` currently lists only up
  to v10; needs a small addition for `v11/<model>/<pair>/` to work locally.
- Reproduce v11.5 §16.1 numbers for height + weight (2B, L20):
  pairwise `cos(P_i, P_j)`, shared/within steering ratio.
- **Pass criterion**: cosines reproduce within 0.02 of the published values.
- **Fail signal**: data drift from HF, or `primal_z` definition mismatch —
  flag and reconcile.

Scripts: `scripts/p0_repro_shared_z.py` (CPU).

### Phase 1 — Text ablation: three intervention modes  [GPU on Gemma 2 2B]

For each of three interventions, hook the residual at L20 of `gemma-2-2b`,
modify `h`, and measure LD on the dense v11 grid.

| Mode | Hook |
|------|------|
| `add_neg` | `h ← h − α·d̂` (additive steering opposite to high-adj) |
| `proj_out` | `h ← h − (h·d̂) d̂` (project out the direction entirely) |
| `mean_ablate` | `h ← h − (h·d̂) d̂ + (μ·d̂) d̂` (LEACE-flavored: replace projection with population mean projection) |

`d̂` = unit `w_shared` from Phase 0 (Procrustes mean of the 8 per-pair
`primal_z`s at L20). Hyperparameter `α` is swept; mode comparison is at the
`α` that maximizes `corr(LD, x)` while keeping perplexity within 5% of
baseline.

For each pair × mode × `α`:
- corr(LD, z), corr(LD, x), |LD| at z=±2, |LD| at x_max vs x_min.
- Perplexity on 200 WikiText sentences (control for fluency damage).
- Random-direction control (3 random unit vectors at the same magnitude).

**Pass criterion**: at least one mode achieves `|corr(LD, z)| < 0.2` while
keeping perplexity within 5% of baseline AND keeping `|corr(LD, x)| > 0.5` of
its no-context value, on at least 4/8 pairs.

**Critical caveat**: project-out has an obvious failure mode — if z is encoded
along multiple non-orthogonal directions (per Jaehoon's "many redundant
heads" finding), single-direction projection won't fully suppress. Iterated
projection (INLP-style) is the next step if single fails.

**Hard pivot**: if no mode suppresses z without trashing perplexity, the
shared-z direction is not the right ablation target — pivot to per-pair
ablation or to subspace ablation (rank > 1).

Scripts: `scripts/p1_text_ablation.py`.

### Phase 1b — Layer sweep  [GPU]

The encode-vs-use phase structure (L1 encode, L2–L14 carry, L15+ broadcast)
predicts that ablation efficacy and damage profile differ by layer.

Hypotheses:
- **L1 ablation**: removes the original encoding. Most damaging to z, may
  also damage other capabilities (anything else encoded at L1).
- **L7–L14 ablation**: catches the carry/rotate phase. Should suppress z
  in the late residual.
- **L20+ ablation**: too late; primal_z has already been "broadcast" and a
  single late-layer ablation may not undo it.

Run mean-ablation at L1, L7, L13, L20, L25 on height + speed. Plot z-suppression
vs perplexity damage by layer.

Scripts: `scripts/p1b_layer_sweep.py`.

### Phase 2 — Driving stress test (text)  [GPU]

Build a prompt that mimics the "context bombardment" scenario:

```
You are observing traffic on a highway.
Car 1: 95 km/h
Car 2: 102 km/h
Car 3: 88 km/h
... (15 cars near 95)
Your car: 60 km/h. Your car is moving
```

Compare:
- Baseline: model says "slowly" (or equivalent low-speed token).
- After Phase 1's mean-ablation at L20: does the readout shift? In the
  zero-shot prior (no context), 60 km/h alone might read as "moderately"
  or "slowly" depending on prior; we measure the prior first.

Variations:
- High-context: 15 cars at 200 km/h, target at 60 km/h. Without ablation,
  60 km/h reads as "slow"; with ablation, should approach the no-context
  baseline.
- Low-context: 15 cars at 30 km/h, target at 60 km/h. Without ablation,
  60 km/h reads as "fast"; with ablation, should approach the same baseline.
- Two-context-cluster (asymmetric): 10 cars at 30, 5 cars at 200, target
  at 60. Stress test for whether `μ`-aggregation breaks under bimodal context.

**Note**: speed is one of Jaehoon's two pair-specific exceptions. Shared-z
direction works less well on speed. So Phase 2 may need a per-pair `primal_z`
direction rather than `w_shared`. If so, that's a finding.

**Bonus connection**: Alex's speed-probe shows `dead-fast` is a *readout*
problem, not a representation problem (e1b R²=0.90 at L=24). If we ablate
shared-z and the readout still caps at "slow/medium", that confirms the
two are independent failure modes.

Scripts: `scripts/p2_driving_text.py`.

### Phase 3 — Vision: relative-context size  [GPU on Gemma 4]

**Design (revised 2026-04-29)**: sequential images, not a single composite
canvas. Each trial = N_REF reference images each showing one square at a
known size, then one target image showing the target square. The model sees
the sequence and is asked to judge the target. This better mirrors the
"context bombardment" framing — you see many things, then a new thing, and
you ask whether your perception of the new thing is anchored to the prior
distribution.

Sample from the same `(x, z)` grid as v11:

- `x`: target square side in pixels.
- `μ`: mean reference-square side.
- `σ`: spread.
- `z = (x − μ) / σ`.

Reference frames: each reference image shows one square of side
`sampled from N(μ, σ)`, centered, on a fixed canvas (e.g., 224×224).
Target frame: same canvas, square of side `x`, centered.

Prompt template (chat-formatted, with N_REF + 1 image content blocks):

```
[ref_img_1] [ref_img_2] ... [ref_img_N] [target_img]
You have just seen <N> images of squares of various sizes, followed by one
final image. The square in the final image is
```

Decode logits for `big`, `small`, `large`, `tiny`, `medium`, `huge`.
Read top-K rather than just LD = logit(big) − logit(small), because of
dead-large.

**First question**: does Gemma 4 multimodal exhibit the same z-pattern in
vision? Cell-mean `corr(LD, z)` (or `corr(top-K-projected score, z)`) on a
10×10×3 grid (`x`, `z`, seed). Hypothesis: yes, `corr ≥ 0.7` on Gemma 4
31B-it given the size-probe arc geometry. But this is the open question —
sequential context bombardment may or may not produce the same z-anchor as
the text setting.

**Sequence-length variants worth testing**:
- N_REF = 5, 10, 15. Does the relativity effect saturate, or grow with
  more context?
- Reference order: ascending vs descending vs shuffled. Does order matter?

**Critical caveat**: Gemma 4 31B-it has *dead-large* per Alex's
2026-04-28 size-probe results. The behavioral cap will make raw LD a poor
metric — use marginals or top-K projections. Also: the speed-probe found
recency-position bias on the moving/stationary boundary (~11 px shift)
when option order changed. Order effects are real on this model family.

**Stimulus generator**: per-frame single-square renderer (reuse
`gemma4-size-probe/make_size_frame.py`); compose sequence at the prompt
level via the chat-template image content blocks (mirrors the speed-probe
multi-frame setup).

Scripts: `scripts/p3_render_relative_size_seq.py` (stimulus gen, CPU),
`scripts/p3b_extract_vision.py` (forward-pass extraction, GPU).

**Note**: an earlier composite-canvas version exists at
`scripts/p3_render_relative_size.py`. Kept around for an alt design but
not on the critical path.

### Phase 4 — Vision ablation  [GPU on Gemma 4]

Repeat the Phase 1 ablation modes on Gemma 4 multimodal. Direction comes from
the multimodal residual at the last token before the answer (analogous to
`primal_z` but on multimodal embeddings of relative-context images).

Measure same z-suppression / x-preservation / fluency triplet. Fluency
control: perplexity on a held-out caption set or on plain text continuation
of an unrelated image.

**Pass criterion**: at least mean-ablate suppresses z below 0.3 on 31B-it
without driving caption perplexity outside 1.5× baseline.

**Critical caveat**: multimodal models are sensitive — ablating residuals can
collapse fluent generation. We may need to apply the projection only at the
text-generation positions, not at the image-token positions.

Scripts: `scripts/p4_vision_ablation.py`.

### Phase 5 — Vision speed (stretch goal)  [GPU on Gemma 4]

Same sequential-image structure as Phase 3, but each "image" is itself a
short multi-frame trajectory. Reference trajectories at multiple `dx_ref`
values, then a target trajectory at `dx_target`. Build z from
`(dx_target − μ_dx) / σ_dx`.

Each "frame slot" in the prompt becomes a multi-frame trajectory
(e.g., 3 frames per trajectory, like `gemma4-speed-probe` p1–p3). Total
images per prompt: N_REF · 3 + 3 (target). For N_REF=5 that's 18 image
content blocks per prompt — Gemma 4 multimodal handles this in principle
but cost scales linearly.

This tests whether the relativity-ablation generalizes from a static cue
(sizes, single-frame target) to a dynamic across-frame cue (speeds,
multi-frame target). The two are plausibly different mechanisms.

Scripts: `scripts/p5_vision_speed.py`.

---

## 4. Risks and pivot triggers

| Risk | Symptom | Pivot |
|------|---------|-------|
| `w_shared` is rank-1 — multidim z subspace | Phase 1 project-out fails to suppress z | Move to rank-k subspace; INLP iterated projection. |
| Ablation breaks fluency | Perplexity > 5% damage | Layer-shift (try earlier or later); scale α down. |
| Vision z-direction not analogous | Phase 3 shows no z-pattern | Backtrack — may be that vision relativity goes through a different mechanism than text. |
| Gemma 4 dead-large makes LD uninformative | Logit-diff ranges are tiny across the whole grid | Use marginal `P(target_pixels_within_K)` or open-ended-then-classify two-stage probe. |
| Driving prompt is too unnatural | Both baseline and ablated outputs are noise | Reframe with chat templates, instruction-tuned model, more naturalistic context. |

---

## 5. What success looks like

A two-figure result that goes in the joint MATS paper:

1. **Text panel**: across 8 v11 pairs, mean-ablation at L20 brings
   `corr(LD, z)` from ~0.95 to <0.2 while preserving `corr(LD, x)` and
   keeping perplexity within 5% of baseline.
2. **Vision panel**: on relative-size and relative-speed multimodal
   stimuli, the same ablation strategy at the analogous Gemma 4 layer
   recovers a behavior pattern dependent on absolute pixel size /
   absolute frame-displacement, not on the within-image reference set.

Plus a clean negative result if vision relativity does not factor through
the same kind of direction — that's still a paper finding.

---

## 6. Out of scope for now

- Fine-tuning to remove relativity at training time (interesting but a
  different paper).
- Multilingual.
- Reasoning models (Gemma 4 dense / E4B-it is enough; CoT was already
  characterized by the bridge-probe and speed-probe work).
- Gemma 2 9B vision (Gemma 2 is text-only).
- Driving policy claims — we are language-model authors, not robotics
  authors. Frame as "this miscalibration mechanism could plug into any
  downstream system" not "this would crash a car."

---

## 7. Sequencing and timing

Phases 0–1 are the load-bearing pieces. Phases 2–5 are independent given
1's interventions land. Suggested order:

```
Phase 0 (1 day, CPU + a few minutes GPU)
  → Phase 1 (2 days, GPU)
  → Phase 1b (1 day, GPU)
  → Phase 2 (1 day, GPU) -- in parallel
  → Phase 3 (2 days, mostly GPU + stimulus design)
  → Phase 4 (2 days, GPU)
  → Phase 5 (stretch; if time before MATS app)
```

Total ~9–10 days of focused work on the workstation.
