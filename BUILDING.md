# BUILDING.md — What I am doing RIGHT NOW

Only one task in this file at a time. When done, move it to TODO.md "done" section and pull the next one in.

## Active task (Day 4, Apr 21 2026)

**v4 dense extraction: 100 seeds per cell, three conditions, activations + logits**

### Why

The v2 extraction had only 63 data points per condition (1 per (x, μ) cell). This
is too few to see manifold structure, compute error bars, or separate signal from
template noise. We also never checked what the model actually outputs.

### What v4 fixes

1. **100 random seeds** per (x, μ) cell for implicit context — 3,500 implicit trials
2. **Three conditions**: implicit (15-person list), explicit (stated μ/σ), zero-shot (no context — control)
3. **Extract logit("tall") - logit("short")** at every prompt — direct behavioral signal
4. **Extract top-5 predicted tokens** — see what the model actually wants to say
5. **Fewer grid points, more replicates**: 5 x × 7 μ × 100 seeds (was 7 × 9 × 1)

### Prompt budget

| Condition | Count | Notes |
|-----------|-------|-------|
| Implicit | 3,500 | 5 x × 7 μ × 100 seeds |
| Explicit | 35 | 5 x × 7 μ × 1 (deterministic) |
| Zero-shot | 5 | 5 x × 1 (no context) |
| **Total** | **3,540** | ~70s on E4B at 50 p/s |

### How to run on Vast

```bash
cd /workspace/repo
git pull origin main
python scripts/vast_remote/extract_v4_dense.py
```

Script is self-contained. Outputs to `results/v4_dense/`.

### Definition of done

- `results/v4_dense/e4b_implicit_mid.npz` + `_late.npz` with shape (3500, 2560)
- `results/v4_dense/e4b_implicit_logits.jsonl` with logit diffs and top-5 tokens
- Same for explicit and zero_shot conditions
- Quick analysis table printed: mean logit_diff per (x, μ) cell
- Upload results to W&B or download locally

### Completion promise word

GARNET-ANVIL
