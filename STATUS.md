# STATUS.md — Project status as of Apr 21, 2026

## Current phase

**v4 auto-research suite** (branch `exp/v4-auto-research`) is staged and ready
to run on Vast.ai. See `BUILDING.md` for the active task (OBSIDIAN-LATTICE).

## What's done

- v0 behavioral kill-tests (Claude Opus 4.5, 100 completions) -- H1 passed, H2 partially failed
- v1 spectrum scan (Claude Sonnet 4.6, 810 completions) -- confirmed hybrid nature of "obese"
- v2 prompt generator (448 prompts, height + wealth domains)
- Gemma 4 activation extraction (E4B + 31B, v2 prompts, 4 layers each) -- artifacts on W&B
- v4 dense extraction (3540 prompts, activations + logit_diff) -- strong behavioral signal
- Five v4 analysis scripts written and smoke-tested locally

## What's next

Run the v4 auto-research suite on Vast (see `BUILDING.md` for exact commands).
Then decide paper direction based on results (see `TODO.md` queue).

## Archived session logs

Detailed session logs from GPU rental bursts are in `docs/archive/`.
