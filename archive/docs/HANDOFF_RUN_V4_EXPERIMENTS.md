# Handoff prompt — run v4 auto-research experiments on Vast GPU

Copy-paste everything below this line into a fresh Claude conversation that has Chrome MCP working. It has full context to pick up where I left off.

---

## Who you are / what this is

You are a Claude agent continuing a mech-interp research project with Jaehoon. The repo is **Geometry of Relativity: Context-Normalized Encoding of Gradable Adjectives in LLMs** at `/sessions/elegant-fervent-edison/mnt/geometry-of-relativity` (this is the user's synced workspace folder — it maps to his real filesystem). The user is on Day 3-4 of a 20-day plan targeting:
- **ICML 2026 MI Workshop** (May 8, 2026)
- **NeurIPS 2026** (May 4/6, 2026)

Today is **2026-04-20**, about 18 days to the first deadline. The user's style: direct, impatient, swears when frustrated, wants fast execution and "notes every interesting result as it comes." Don't ask clarifying questions before acting on things that are already clear; don't belabor explanations; don't over-apologize.

## Read these first (in order, don't skip)

1. `/sessions/elegant-fervent-edison/mnt/geometry-of-relativity/CLAUDE.md` — project rules (frozen spec, one-task-one-commit, completion promise word).
2. `/sessions/elegant-fervent-edison/mnt/geometry-of-relativity/PLANNING.md` — the frozen research spec. Do not edit.
3. `/sessions/elegant-fervent-edison/mnt/geometry-of-relativity/BUILDING.md` — single active task. If non-empty, that's the priority.
4. `/sessions/elegant-fervent-edison/mnt/geometry-of-relativity/STATUS.md` — latest progress log.
5. `/sessions/elegant-fervent-edison/mnt/geometry-of-relativity/docs/v4_research_design.md` — the v4 experiment plan.
6. `/sessions/elegant-fervent-edison/mnt/geometry-of-relativity/docs/paper_outline.md` — target paper skeleton.

## Scientific context (the claim we're testing)

Key hypothesis: linear probes for **relative** gradable adjectives ("tall", "short", "young", "old", "heavy", "light", "big", "small", "fast", "slow", "rich", "poor", "experienced", "inexperienced") should track a **context-dependent Z-score** `z = (x − μ) / σ`, while probes for **absolute** adjectives ("obese") should track the raw value `x`. We make the distinction rigorous via the **Fisher-Information pullback** `F(h)⁻¹ · w`, where `F(h) = W_U^T (diag(p) − p p^T) W_U` is the Fisher-Rao metric on the residual stream under the LM head.

Key operational quantity: the **relativity ratio** `R = −c/b` from the OLS regression `logit_diff ~ a + b·x + c·μ`. For relative adjectives `R ≈ 1` (context fully cancels raw value); for absolute `R ≈ 0`.

Wealth is the one **log-space** pair: `z = (log(x) − log(μ)) / log(σ_factor)` with `σ_factor = 2`. Implemented as `compute_z()` in `scripts/vast_remote/extract_v4_adjpairs.py`, gated by `LOG_SPACE_PAIRS = {"wealth"}`.

## Current git state

Branch: `exp/v4-auto-research` (already pushed to `origin`). Latest commit:

```
d9bb99d exp(v4): fix 4 research-hygiene bugs in auto-research pipeline
cc6fb51 docs: log INLP + paper-outline work in STATUS.md, TODO.md
43f91b4 exp(v4): add INLP erasure (5th line of evidence) + paper outline skeleton
c194cf1 exp(v4): use trial's actual z value — minor fix, OBSIDIAN-LATTICE-FIX
4f4c694 exp(v4): auto-research pipeline — analysis + adjective-pair extension + causal steering
```

The four fixes in d9bb99d:
1. **CV data leakage** — `StandardScaler` was fit on full data before `KFold`. Replaced with `sklearn.pipeline.Pipeline([StandardScaler, Ridge])` so the scaler refits per fold. Files: `scripts/vast_remote/analyze_v4.py`, `scripts/vast_remote/analyze_v4_adjpairs.py`, `scripts/vast_remote/inlp_v4.py`.
2. **Underweight multi-token bias** — the BMI pair's `low_word` was "underweight" (multi-token in SentencePiece). Changed to "thin" (single-token). See `extract_v4_adjpairs.py`.
3. **Unified tokenization helper** — new file `scripts/vast_remote/_token_utils.py` with `first_token_id()` and `tokens_of_word()` that prefer the leading-space variant (" tall" beats "tall" when the prompt ends with "is"). Used by `extract_v4_dense.py`, `extract_v4_adjpairs.py`, `steer_v4.py`. Script `main()` raises `SystemExit(2)` if any target word is multi-token at load time.
4. **Wealth log-space z** — new `compute_z(pair, x, mu)` helper, `LOG_SPACE_PAIRS = {"wealth"}`, `sample_context()` gains `log_space=` kwarg using `rng.gauss(log(μ), log(σ))`. Smoke test `test_implicit_sampling_mean_near_mu` now checks geometric mean for log-space (arithmetic mean of log-normal is biased up by `exp(σ_log²/2)`).

All 10/10 smoke tests pass locally (`tests/test_adjpairs_smoke.py` + `tests/test_inlp_smoke.py`).

## Vast GPU access

The user has an H100 80GB instance running on Vast.ai with a Jupyter Hub served over a **cloudflare tunnel** (URL format: `https://<random-hyphenated-subdomain>.trycloudflare.com/lab` or `/terminals/N`). Tunnel hostnames change per session — you'll need to get the current one.

**Two ways to get the URL:**
- Easiest: ask Jaehoon for it directly.
- Via Vast: `https://cloud.vast.ai/instances/` shows his running instances; the tunnel URL is usually in the instance's notes or the Vast "Open" button. He's already logged into vast.ai in Chrome.

**Critical**: the previous Claude (me) had the Chrome MCP load successfully and could see tab context, but every `navigate` / `screenshot` / `read_page` call returned "Permission denied by user" — his Claude-in-Chrome extension seems to auto-deny or the per-domain dialog never surfaces. **Tell him to click the Claude-in-Chrome extension icon and check that `cloud.vast.ai` and `trycloudflare.com` are set to Ask or Allow before you start.** If Chrome still doesn't work, fall back to asking for the URL text directly.

## The experiments to run, in order

All scripts live on the Vast box at `~/geometry-of-relativity/scripts/vast_remote/` (or wherever he cloned). Start with a `git pull` on `exp/v4-auto-research` to get commit `d9bb99d`.

```bash
cd ~/geometry-of-relativity
git fetch origin
git checkout exp/v4-auto-research
git pull
python -m pytest tests/ -x -q        # sanity: 10/10 should pass
```

Then run (order matters — analyze depends on extract):

```bash
# 1. Probes + geometry on v4 dense (30 trials × ~100 prompts, already extracted as results/v4_dense/e4b_acts_*.npz)
python scripts/vast_remote/analyze_v4.py         # writes results/v4_analysis/probes/*.npz + json

# 2. 8-pair extraction (7 relative + 1 absolute bmi). 6240 prompts total. ~15-25 min on H100.
python scripts/vast_remote/extract_v4_adjpairs.py

# 3. Per-pair probes + relativity ratio R = -c/b per pair
python scripts/vast_remote/analyze_v4_adjpairs.py

# 4. Causal steering — adds alpha*w_z to residual at layer L, checks monotone logit_diff shift
python scripts/vast_remote/steer_v4.py --layer late
python scripts/vast_remote/steer_v4.py --layer mid

# 5. INLP concept erasure — iteratively null out w_z, check R^2 collapse
python scripts/vast_remote/inlp_v4.py --layer late
python scripts/vast_remote/inlp_v4.py --layer mid
```

## Markers to call out as results come in

Per Jaehoon's directive "note every interesting result". Flag each of these explicitly when you see them:

- **R²(z) vs R²(x) gap**: at mid/late layers (idx 21, 32 for E4B), implicit-condition R² for predicting z from activations should be >> R² for predicting x. Expected gap ≥ 0.3.
- **Σ⁻¹ vs Euclidean cosine of w_adj**: Fisher/covariance-metric cosine similarity between the adjective probe and the z probe should be near 1.0 under Σ⁻¹ pullback even when Euclidean cos is mediocre. This is the central "geometry matters" result.
- **Per-pair relativity ratio R = −c/b**: 7 relative pairs should cluster near 1.0, bmi_abs near 0.0. If wealth is outlier you may need more seeds or the log-space z isn't flowing through — double-check.
- **Steering monotonicity (steer_v4.py)**: `mean_logit_diff` should be monotone in α, slope ≥ 0.5 per α-unit for w_z at late layer. Flat curve for a random null direction.
- **INLP collapse (inlp_v4.py)**: R²(z) under INLP-z should drop sharply by step 3 (≥ 0.5 gap vs random-null). R²(logit_diff) should also drop, linking the probe direction to actual adjective prediction.
- **Participation ratio** `(Σλ_i)² / Σλ_i²` of cell-mean covariance: if < 3 for relative pairs, strong evidence for low-dim (μ, σ)-parameterized manifold à la Shape of Beliefs (arxiv 2602.02315). Not in analyze_v4.py yet — slip it in if time allows.

## Model + layer reference

Primary model: `google/gemma-4-E4B` (42 layers, d=2560). `LAYER_TO_IDX = {"mid": 21, "late": 32}` in `steer_v4.py` and `inlp_v4.py`. Secondary (for paper): `google/gemma-4-31B` (60 layers, d=5376) — probably don't run this in this session unless the user asks.

Tall/short tokenization check: `first_token_id(tok, "tall")` and `first_token_id(tok, "short")` must both be single-token after spacing-variant search. The scripts will abort at startup if not.

## If something breaks

- **Git index.lock "Operation not permitted"**: this is a known FUSE-mount quirk. Fix: `mv .git/index.lock .git/_dead_indexlock_$(date +%s%N).lock` before retrying the git op.
- **`pytest` tmpdir cleanup error**: pre-existing, ignore as long as assertions pass.
- **OOM on extract_v4_adjpairs.py**: reduce `BATCH_SIZE` in the script (grep for it).
- **HF model gated**: run `huggingface-cli login` on the box with his token (already in `.env` but sometimes needs re-login).

## Conventions

- One task, one commit. Commit message last line must contain a completion promise word (e.g., `SAPPHIRE-MEADOW`). Invent one per task.
- Black (line length 100), ruff, python 3.10, type hints.
- Never commit `.env`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, or Vast creds.

## Open todos (from prior session)

- `#31` analyze_v4.py on Vast — **run it**
- `#32` extract_v4_adjpairs.py on Vast — **run it**
- `#33` steer_v4.py — **run it**
- `#30` auto-research on v4 — in-progress umbrella task; gets closed when the above 3 + INLP all produce results and you've appended markers to STATUS.md
- `#26` destroy Vast instance m:56779 — don't touch unless user confirms

---

**Start by**: reading PLANNING.md + STATUS.md, asking Jaehoon for the current tunnel URL (or unlock Chrome permissions), then `git pull` + `pytest` + run experiments in the order above, calling out markers as they print.
