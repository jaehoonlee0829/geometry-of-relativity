## Summary

- Extracts last-token hidden states from **Gemma 4 E4B (42L, d=2560)** and **Gemma 4 31B (60L, d=5376)** at four depths each — early / mid / late / final — across all 448 v2 prompts.
- Uploads the full activation set (18 `.npz` + 2 `W_U.npy`, ~16 GB total) to W&B as a single artifact `gemma4-activations` with aliases `day4` + `v2-prompts`. Run: [ax81rrlu](https://wandb.ai/xrong-optiver/geometry-of-relativity/runs/ax81rrlu).
- Checks in the two extractor scripts that ran on the Vast box so the pipeline is reproducible from a fresh rental.
- Rolls BUILDING.md forward to the Day-4 probe task (SAPPHIRE-BEARING kill-test).

## Depends on

`feat/v2-prompt-generator` — needs `data_gen/prompts_v2.jsonl` to be present. Merge that first.

## Schema

```
<model>_W_U.npy                        # once per model, uncompressed fp32 .npy (fast)
<model>_<domain>_<layer>.npz           # activations (n_prompts, d) fp32 + ids + metadata
```

Pulling `W_U` into every per-layer `.npz` (the v2 mistake) cost ~30× in zlib time. The v3 layout fixed this — per-layer files are now ~2.5–5.5 MB each and save sub-second.

## Layer indices

| model   | early | mid | late | final |
| ------- | ----- | --- | ---- | ----- |
| E4B     | 10    | 21  | 32   | 41    |
| 31B     | 14    | 30  | 45   | 59    |

## Observations worth flagging in the paper

- **31B layer 59 sits after an internal RMSNorm.** Activation std ≈ 0.064 vs ~3.6 at layer 45 — consistent with Gemma 2/3's dual-norm pattern. Use **layer 45 ("late")** as the near-final slice for un-renormalized residual geometry. Layer 59 is still correct for logit-lens work.
- Through-stack std climb is healthy and monotone on the other three depths (mean ≈ 0, growing std).
- Gemma 4 31B weights were pre-cached on the Parsed Vast host, so the 62 GB download was skipped and the whole extraction came in under $4 of GPU spend.

## Test plan

- [x] Both extractors run on a single 2× H100 PCIE instance without OOM
- [x] All 18 `.npz` load cleanly on CPU; shapes, IDs, and `layer_index` metadata round-trip
- [x] W&B artifact download → `np.load` → shape check passes on a fresh machine
- [ ] Day-4 probe training (SAPPHIRE-BEARING) exercises these activations end-to-end — kill-gate at β/(α+β) < 0.5 on E4B mid-layer height

Completion token: **INDIGO-COMPASS** (commit `6f29b0b`)
