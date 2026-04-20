# BUILDING.md — What I am doing RIGHT NOW

Only one task in this file at a time. When done, move it to TODO.md "done" section and pull the next one in.

## Active task (Day 4, Apr 21 2026)

**Train the three probes (w_adj, w_x, w_z) on Gemma 4 E4B mid-layer activations and compute the Fisher-pullback + α/β decomposition — kill-test for the whole paper**

Context: Day 3 (INDIGO-COMPASS) complete. E4B + 31B activations live on W&B
Artifact `gemma4-activations` (alias `day4`, run `ax81rrlu`) and on the Vast
instance at `/workspace/repo/results/activations/`. 448 v2 prompts: 252 height
(linear-z, σ=7 cm) + 196 wealth (log-z, σ=log-USD). 4 layers per model: E4B
{10, 21, 32, 41}, 31B {14, 30, 45, 59}. The last-layer 31B activations appear
post-normed (std≈0.06), so probes there will be uninformative — use layer 45
("late") as the near-final slice.

### Concrete steps

1. Download E4B `.npz` files locally (pull via W&B Artifact or Jupyter Contents
   API GET — they're tiny, ~19 MB total). Keep `e4b_W_U.npy` on the remote;
   don't pull 2.7 GB locally unless analysis demands it.
2. For each (domain, layer) — start with E4B mid (layer 21) as the sanity
   check — train three ridge probes on the 252 height trials:
   - `w_adj`: target = sign(last-token logit for "tall" − "short") scaled so
     we regress directly onto the adjective-pair contrast (use the adjective
     prompt, not the "is considered" frame, for this probe)
   - `w_x`: target = raw height value (cm)
   - `w_z`: target = (x − μ_ctx) / σ_ctx (the context-Z-score from the implicit
     15-sample window)
3. α/β decomposition: regress w_adj onto [w_x, w_z] in the Euclidean metric,
   then repeat in the Σ⁻¹ metric (activation covariance inverse) and the F(h)⁻¹
   metric (Fisher pullback). Report β/(α+β) — this is the "Z-vs-raw" signal.
4. Kill gate (CLAUDE.md PLANNING.md): **if β/(α+β) < 0.5 at mid-layer for
   height on E4B, stop and scope down the paper.** Otherwise proceed to 31B.
5. Wire up `scripts/run_probe_day4.py` as the driver; log a summary row to
   `results/day4_probe_summary.json` + append to STATUS.md.

### Definition of done

- `results/day4_probe_summary.json` with α, β, β/(α+β), cond(F), cond(Σ) for
  E4B × {height, wealth} × mid-layer in all three metrics.
- Kill-gate decision written to BUILDING.md (either "PROCEED — SAPPHIRE-BEARING"
  or "SCOPE DOWN — see notes").
- One paragraph in STATUS.md summarising the first quantitative geometry
  result from this paper.

### Completion promise word

SAPPHIRE-BEARING
