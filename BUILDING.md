# BUILDING.md — What I am doing RIGHT NOW

Only one task in this file at a time. When done, move it to TODO.md "done" section and pull the next one in.

## Active task (Day 3, Apr 20 2026, evening)

**Implement v2 prompt generator + update activation extraction for new design**

Specifically:

1. Rewrite `src/data_gen.py` to support the v2 prompt design:
   - Implicit context: generate 15 sampled heights/incomes, then present target with "is ___" or "is considered ___"
   - Explicit context: stated μ and σ, then target
   - Both height (tall/short) and wealth (rich/poor) domains
   - Parameterized by (target_value, context_mu, sigma, context_type, prompt_frame)
   - Fixed random seed per (μ, σ) condition for deterministic context samples

2. Update `src/__init__.py` with new model IDs:
   - `google/gemma-4-31B` (primary)
   - `google/gemma-2-9b` (secondary)
   - Layer indices for both models

3. Update `src/activation_extract.py`:
   - Support extracting activation at "is" token vs "considered" token (not just last token)
   - Handle Gemma 4 architecture

4. Generate `data_gen/prompts_v2.jsonl` — full experiment matrix

5. Update tests to cover new prompt design

### Definition of done

- `python -m src.data_gen` generates v2 prompts for both domains
- `pytest tests/` all green
- `data_gen/prompts_v2.jsonl` committed with correct trial count (~500 prompts)

### Completion promise word

COPPER-LANTERN
