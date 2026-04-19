# TODO.md — Rolling checklist (Day 3 of 20, Apr 20 2026, evening)

## Done

- [x] Compress 7 seed papers (Notion: 7 Papers Primer)
- [x] Commit primary research angle (Notion: Primary Angle subpage)
- [x] Supersede A/B/C placeholder angles on AI Research Factory hub
- [x] Design prompt battery v0 (`data_gen/prompts_v0.jsonl`)
- [x] Write behavioral runner `scripts/run_behavioral.py`
- [x] **Day-1 behavioral kill-test complete (100 completions, 20 prompts × 5 samples, Claude Opus 4.5):**
  - H1 PASSED decisively (100% flip tall→short across narrow-low/narrow-high)
  - H2 partially failed in an informative way — "obese" exhibits context sensitivity
  - Paper framing updated: relativity spectrum, not relative/absolute binary
- [x] Scaffold 5 core `src/` modules — data_gen (352 LOC), fisher (241 LOC), probe (205 LOC), activation_extract (245 LOC), plots (6.9 KB)
- [x] All src imports work; all __main__ smoke tests pass (except activation_extract which is guarded for missing torch)
- [x] Write `scripts/analyze_behavioral.py` + generate `results/behavioral_v0_summary.md`

## Active (BUILDING.md has the details)

- [ ] git init, push to public GitHub, add CI, write 3 unit tests (test_fisher, test_probe, test_data_gen)

## Queue — Day 4 (Apr 21)

- [ ] Update Notion Primary Angle subpage with revised H1/H2/H3/H4 (relativity spectrum framing) — DONE today, but double-check on Day 4 before the GPU run
- [ ] Rent Vast.ai GPU (A100 40GB, 4h burst) — verify account status first
- [ ] Upload Gemma-2-2b weights via HF auth on the GPU instance
- [ ] Run activation extraction for Gemma-2-2b on all 20 prompts × 4 layers
- [ ] Save activations to `.npz`, download back to local

## Queue — Day 5 (Apr 22) — HARD PIVOT CHECK

- [ ] Train probes for `{tall, short, obese}` on Gemma-2-2b mid layer
- [ ] Compute F(h) at each activation, F⁻¹·w via Cholesky
- [ ] Compute ρ_rel and ρ_abs per cell
- [ ] IF H1 probe shift < 1σ on Gemma-2-2b mid layer: STOP, switch to 4-page short paper or defer to ICLR 2027
- [ ] ELSE: proceed

## Queue — Day 6–10 (Apr 23–27) — Exploration

- [ ] Replicate on Llama-3.2-3B
- [ ] Sweep all 4 layers × 2 models × 3 adjectives × 4 contexts
- [ ] Generate hero figure v0 — use `src/plots.plot_hero`
- [ ] Sanity check: scramble adjective labels, does probe still "work"? It should NOT.
- [ ] Sanity check: control adjective ("blue") — should NOT show context sensitivity

## Queue — Day 11–15 (Apr 28 – May 2) — Understanding

- [ ] Red-team all positive findings within 24h
- [ ] Ablation: remove Fisher-pullback (use Σ⁻¹·w instead), does ρ collapse?
- [ ] Add Axis-2 adjectives (heavy/light for weight) if Axis-1 is clean by Day 10
- [ ] Draft results section with real numbers

## Queue — Day 16–20 (May 3–8) — Distillation

- [ ] Complete paper draft
- [ ] Internal review pass
- [ ] Upload to arXiv (Day 19)
- [ ] Submit to NeurIPS 2026 (May 4 abstract, May 6 full)
- [ ] Submit to ICML 2026 MI Workshop (May 8)

## Backlog

- [ ] Axis 2-4 (only if Axis 1 clean by Day 10)
- [ ] Dual Steering intervention benchmark
- [ ] Cross-model stitch via learned affine
