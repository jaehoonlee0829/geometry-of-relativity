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
- [x] Scaffold 5 core `src/` modules — data_gen, fisher, probe, activation_extract, plots
- [x] All src imports work; all __main__ smoke tests pass
- [x] Write `scripts/analyze_behavioral.py` + generate `results/behavioral_v0_summary.md`
- [x] git init, push to public GitHub, add CI, write 3 unit tests
- [x] **v1 spectrum experiment (810 completions, Sonnet 4.6):**
  - Tall: clean sigmoid flip, crossover at μ≈160.2, slope |k|≈3.4/cm
  - Obese: ~1/3 absolute + ~2/3 relative mixture, stable plateau
  - Literal-number hypothesis ruled out (BMI-direct ≡ height+weight)
- [x] Migrate requirements.txt → pyproject.toml
- [x] **PLANNING.md v2**: redesigned experiment — implicit context, varying target values, two domains (height + wealth), three metric comparisons (Euclidean / Σ⁻¹ / F⁻¹), Gemma 4 31B primary
- [x] **v2 prompt generator** (`feat/v2-prompt-generator`, commit `dc41935`, COPPER-LANTERN):
  - `src/data_gen.py`: DomainSpec + HEIGHT_SPEC + WEALTH_SPEC + TrialV2 + deterministic context sampler + implicit/explicit renderers + JSONL writer
  - `tests/test_data_gen.py`: 17 new v2 tests, all 31 green
  - `data_gen/prompts_v2.jsonl`: 448 trials (252 height + 196 wealth)
  - Branch staged locally; user pushes + opens PR when back
- [x] **Gemma 4 activation extraction, both models** (`exp/gemma4-activations-day4`, INDIGO-COMPASS):
  - Vast 2× H100 PCIE box, Jupyter Contents API upload for scripts (heredoc paste was unreliable)
  - `scripts/vast_remote/extract_e4b_v3.py`: E4B (42L, d=2560) — layers 10/21/32/41 — 252 height + 196 wealth prompts
  - `scripts/vast_remote/extract_g31b_v1.py`: 31B (60L, d=5376) — layers 14/30/45/59 — same prompts
  - Schema: `<model>_W_U.npy` once + `<model>_<domain>_<layer>.npz` (activations-only, ~2.5-5.5 MB each)
  - All 18 files + 2 W_U uploaded to W&B Artifact `gemma4-activations` (alias `day4`, run `ax81rrlu`)
  - Verified load: shapes, layer_idx, IDs all correct. Note: 31B layer 59 has small std (~0.064) — post-norm output, use `late` (layer 45) for un-renormalized residual geometry
  - Vast instance left running per authorization C

## Active (BUILDING.md has the details)

- [ ] Day-4: probe training + Fisher analysis (SAPPHIRE-BEARING)

## Queue — Day 4 (Apr 21)

- [ ] Run v2 behavioral sanity check: ~20 implicit-context prompts through Claude API to verify implicit design elicits clean flips

## Queue — Day 5 (Apr 22) — HARD PIVOT CHECK

- [ ] Train three probes on mid-layer activations: w_adj (tall/short), w_x (raw cm), w_z (Z-score)
- [ ] Compute all three metrics in parallel: Euclidean cos, Σ⁻¹ cos, F⁻¹ cos
- [ ] Compute α/β decomposition: regress w_adj onto [w_x, w_z]
- [ ] IF probe shift < 1σ on Gemma 4 31B mid layer: STOP, scope down
- [ ] ELSE: proceed to full sweep

## Queue — Day 6–10 (Apr 23–27) — Full sweep + second domain

- [ ] Sweep all 4 layers × 2 context types × 2 prompt frames for height
- [ ] Extract activations for wealth domain (rich/poor), train probes
- [ ] Compare "is ___" vs "is considered ___" — does prompt frame affect α/β?
- [ ] Generate hero figures (α/β decomposition + metric comparison)
- [ ] Compare Gemma 4 31B vs E4B — does signal scale with model size?
- [ ] Probe-artifact controls (R4): scrambled-label, MLP-vs-linear gap, causal steering validation
- [ ] Sanity check: scramble adjective labels → probe should fail

## Queue — Day 11–15 (Apr 28 – May 2) — Understanding + writing

- [ ] Red-team all positive findings within 24h
- [ ] Analyze: does Euclidean baseline already work? If so, what does Fisher add? (R3)
- [ ] Report cond(F(h)) and angle between w and F⁻¹w
- [ ] α/(α+β) cross-adjective correlation with behavioral sigmoid (R2)
- [ ] SVD on stacked probe directions — shared subspace analysis (R1)
- [ ] Draft results section with real numbers
- [ ] Draft intro + methods

## Queue — Day 16–20 (May 3–8) — Paper

- [ ] Complete paper draft
- [ ] Internal review pass
- [ ] Upload to arXiv (Day 19)
- [ ] Submit to NeurIPS 2026 (May 4 abstract, May 6 full)
- [ ] Submit to ICML 2026 MI Workshop (May 8)

## Backlog

- [ ] BMI / obese domain (revisit v1 mixture finding with activation probing)
- [ ] Additional adjective pairs: young/old, hot/cold
- [ ] Steering interventions (F⁻¹·w vs naive w)
- [ ] Cross-model stitch via learned affine
- [ ] "Pure absolute" controls (dead, pregnant, prime-numbered)
