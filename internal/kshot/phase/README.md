# relativity_ablation

Extension of Jaehoon Lee's `geometry-of-relativity` — using activation
ablation to suppress context-relative readouts in language and
vision-language models.

See:
- [PLAN.md](PLAN.md) — phases, methods, pivot triggers
- [FINDINGS.md](FINDINGS.md) — rolling experimental log
- `scripts/p0_repro_shared_z.py` — reproduce v11.5 §16.1 shared-z
- `scripts/fetch_v11_subset.py` — fetch v11 NPZs (currently blocked by
  private HF dataset access; we will regenerate locally instead)
- `scripts/p3_render_relative_size.py` — generate composite-canvas
  relative-context vision stimuli

## Quick orientation

We are *not* trying to "make the model objective." We are trying to
**decouple context-relativity from the readout**, then check whether the
residual behavior aligns with the model's no-context prior. The
distinction is in PLAN §2 and FINDINGS §2 — please read those before
running anything.
