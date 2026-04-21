# STATUS.md — Project status as of Apr 21, 2026

## Current phase

**v7 clean-grid analysis complete.** All behavioral and activation-geometry plots
regenerated on Grid B (x, z) data. Confound matrix now shows Grid B only.
See `docs/NEXT_GPU_SESSION_v8.md` for the next GPU session plan.

## What's done

- v0/v1 behavioral kill-tests (Claude Opus 4.5 + Sonnet 4.6)
- v2 prompt generator + Gemma 4 activation extraction (E4B + 31B)
- v4 dense extraction (3540 prompts) + 8-pair adjective extraction (6240 prompts)
- v5 red-team follow-up: meta-direction steering, Fisher/Park metrics, random-null control, G31B scaling, critic consensus
- v6 red-team: 7-direction analysis, confound discovery (Grid A corr(x,z) = 0.58-0.86)
- **v7 clean-grid rerun**: Grid B (x, z) extraction, confound audit, INLP, Fisher, steering, cross-pair transfer
- v7b addendum: fixed residual confound for experience/size pairs
- **v7 plot regeneration**: all figures regenerated from pre-computed JSON (no GPU needed)
  - Behavioral: 7 plots in `figures/v7_behavioral/` (including zero-shot-corrected heatmaps)
  - Geometry: 8 plots in `figures/v7/` (confound matrix Grid B only, INLP, steering, transfer, Fisher)

## What's next

1. **v8 GPU session** (~15 min GPU): direct sign classification (4 prompt variants), top-K token analysis, cross-template transfer test
2. **PCA horseshoe on Grid B**: needs .npz fetch from HF (`python scripts/fetch_from_hf.py`) then CPU scripts
3. **Paper writing**: ICML MI Workshop (May 8), NeurIPS 2026 (May 4/6)

## Archived session logs

Detailed session logs and PR descriptions from GPU rental bursts are in `docs/archive/`.
