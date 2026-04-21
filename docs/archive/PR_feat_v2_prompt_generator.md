## Summary

- Adds the v2 prompt generator: implicit-context trials with decorrelated (x, μ, σ) triples across two domains (height / wealth), plus explicit-context and two-frame variants for later ablations.
- Ships `data_gen/prompts_v2.jsonl` (448 trials: 252 height + 196 wealth) ready to feed into Gemma 4 activation extraction.
- Brings `src/data_gen.py` in line with PLANNING.md v2: `DomainSpec`, `HEIGHT_SPEC`, `WEALTH_SPEC`, `TrialV2`, deterministic context sampler, implicit/explicit renderers, JSONL writer.
- 17 new unit tests covering determinism, z-score construction, frame rendering, and JSONL round-trip. Full suite 31/31 green.

## Design notes

- Height uses linear Z (σ = 7 cm); wealth uses log-Z in USD space, so "considered rich" tracks log-income, not raw-income.
- Context window size = 15 samples — wide enough to estimate σ on the fly but small enough for the model's context window.
- The `"is"` vs `"is considered"` frame split lives in the renderer so a single trial record can serve both prompt frames without re-sampling.
- No activation code changed in this PR — that lands in `exp/gemma4-activations-day4`.

## Test plan

- [x] `pytest tests/test_data_gen.py` — 17 new + 14 existing = 31/31 green
- [x] `python -c "from src import data_gen; data_gen.build_v2_trials()"` round-trips cleanly
- [x] Eyeballed 20 random prompts for grammatical sanity across both domains and both frames
- [ ] Downstream: verified in `exp/gemma4-activations-day4` — both Gemma 4 models ingest the JSONL and produce well-shaped activations

Completion token: **COPPER-LANTERN** (commit `dc41935`)
