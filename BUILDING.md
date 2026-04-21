# BUILDING.md — What to run RIGHT NOW

## Active task (Day 5→6, Apr 21 2026) — NEXT-GPU-SESSION

Execute all 7 experiments from `docs/NEXT_GPU_SESSION.md` + G31B secondary run, single branch `exp/next-gpu-session`, one commit per experiment, one PR at the end. Critic-agent consensus pass before PR opens.

### Time budget

Not hard-capped. Goal: all 7 done thoroughly, skip rationalization.

### Execution order (unblock first, then priority)

| # | Experiment | Est GPU | What it unblocks |
|---|---|---|---|
| A | Setup: branch, data reuse (copy from `/workspace/repo`), BUILDING.md, port `export_W_U.py` | 0 | Everything |
| 5 | Per-pair plots from cached logits (8-panel heatmaps, scatter, etc.) | 0 (reuse) | Visual hero figures for paper |
| 4d | P(short) vs P(tall) zero-shot bias validation | 0 (reuse) | Whether Exp 4a is needed |
| 3a | Σ⁻¹ cosine JSON persistence | 0 (CPU) | Red-team #3 |
| 6 | Drop w_adj from main results (editorial) | 0 | Paper cleanup |
| 2 | Meta-direction w₁ steering × 8 pairs × 9 α | ~15 min | Causal claim |
| 3b | F⁻¹ cosines (H4 validation) — sample ~50 cell-mean activations | ~30 min | Paper's theoretical anchor |
| 1 | Zero-shot expansion (5x × 30 seeds × 8 pairs) | ~5 min | Zero-shot direction analysis |
| 7 | 3 new absolute-adjective controls (freezing, minor/adult, pass/fail) | ~30 min | n=4 absolute → statistical comparison |
| 4a | Synonym-family re-extraction (conditional on 4d) | ~5 min | Token-design fix |
| G31B | v4_adjpairs extraction + core analyses on Gemma 4 31B | ~10 min | Paper scaling evidence |
| C | Spawn 3 critic agents, synthesize consensus | 0 | Skepticism pass |
| D | Upload new data to HF, commit per-exp, open PR | 0 | Ship |

### Data provenance

- v4_adjpairs / v4_dense / activations: copied from `/workspace/repo/results/` (same Vast box, same files uploaded to HF at `xrong1729/mech-interp-relativity-activations`).
- W_U: re-generated on demand via `scripts/vast_remote/export_W_U.py e4b`.
- Model weights: cached at `/workspace/.hf_home/hub/` (E4B + G31B).

### Working principles (per user directive Apr 21)

1. Separate branch → one PR (`exp/next-gpu-session`).
2. Upload data to HF for anything GPU-expensive; results/plots in git.
3. Surface red flags; never rationalize away a hypothesis-killing result.
4. Save every relevant plot.

### Completion promise word

NEXT-ECLIPSE
