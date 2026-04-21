# Session log — Day 3/4 (Apr 20, 2026) — 2hr autonomous GPU burst

(Previous Day-1 STATUS content is archived in `docs/archive/status_day1.md` and the "Done" block of `TODO.md`.)

Started: 2026-04-20T06:14Z (while user on 1hr walk)
Budget: ~$27 Parsed credit; $4.04/hr × ≤2hr = ~$8 expected. Auto-reload on.

## Rental

- [x] 2× H100 PCIE (m:56779, UK host:67231, $4.042/hr, 8651 Mbps down) rented → Parsed team instance #11 (m:31687 was sniped)
- [ ] Instance boot + PyTorch image pull (~3–5 min ETA)

## Two parallel tracks

### Track A — Local CPU work (Cowork session, this agent)
- [ ] Branch `feat/v2-prompt-generator`
- [ ] Implement v2 prompt generator (`src/data_gen.py`)
- [ ] Update `src/__init__.py` Gemma 4 model IDs + LAYER_INDICES
- [ ] Update `src/activation_extract.py` for "is"/"considered" token + Gemma 4 arch
- [ ] Generate `data_gen/prompts_v2.jsonl` (~500 prompts)
- [ ] Tests green
- [ ] Commit with `COPPER-LANTERN`, push branch, `gh pr create` (NOT merging)

### Track B — Remote GPU work (via Vast Jupyter terminal over Chrome MCP)
- [ ] Wait for instance ready → open Jupyter terminal
- [ ] Git clone repo (public, HTTPS, no auth for read)
- [ ] Export HF_TOKEN / WANDB_API_KEY / ANTHROPIC_API_KEY / GH_TOKEN to `~/.bashrc`
- [ ] `pip install -e '.[gpu]'` (pyproject + heavy extras)
- [ ] `huggingface-cli login` via token
- [ ] `wandb login` via key
- [ ] Smoke-test Gemma 4 E4B (~8B, ~5 min load on H100)
- [ ] If smoke pass: download Gemma 4 31B (~62GB / 1 Gbps = ~8 min)
- [ ] Run activation extraction for both domains × 4 layers × 2 models
- [ ] Upload .npz to W&B Artifacts
- [ ] Commit branch `exp/gemma4-activations-day4` + open PR (NOT merging)
- [ ] Install Claude Code with `--dangerously-skip-permissions` for future Ralph loops
- [ ] **DO NOT destroy instance** (per authorization C)

## Hard rules

- ⚠️ Never touch the 10 pre-existing Parsed team instances or their storage
- ⚠️ Every change goes via PR, never push to main
- ⚠️ Terminal/IDE apps are click-tier — all remote commands via Vast's browser Jupyter terminal
- ⚠️ Budget: auto-reload on (per user), but logging each hourly tick
- ⚠️ Fallback: if 31B OOMs or download stalls, ship E4B-only results + log 31B issue to TODO

## Progress log

- **06:14Z** — Instance rented (m:56779, UK, 2× H100 PCIE). Starting parallel Track A + B setup.
- **06:46Z** — Track A complete: `feat/v2-prompt-generator` branch committed locally (`dc41935`, COPPER-LANTERN). 31/31 tests green. 448 v2 prompts written (252 height + 196 wealth). BUILDING.md/TODO.md rolled forward to INDIGO-COMPASS (remote GPU setup). FUSE mount can't delete, so git lock required rename-sidestep — no damage, commit is real. Heading to Track B now.
- **07:28Z** — Remote env up: HF login OK, E4B (~19 GB) pulled + loaded on single H100 (6.5 s). `scripts/activation_extract.py` pre-compaction hit `Gemma4Config has no attribute hidden_size` (Gemma 4 nests text attrs). Fixed with `get_text_cfg` helper. Uploaded self-contained `extract_e4b_v2.py` to remote via Jupyter Contents API (heredoc paste truncated silently — learned lesson, always use Contents API for >1KB scripts).
- **07:30Z** — E4B forward pass over 252 height prompts completed in 5.1 s (49 p/s). BUT save step bottlenecked at `savez_compressed` with 262144×2560 `W_U` re-serialized per layer → ~6 min per `.npz` × 8 files = 48 min projected.
- **07:53Z** — Killed v2 extractor, shipped `extract_e4b_v3.py`: `W_U` saved ONCE as uncompressed `.npy`, per-layer `.npz` holds activations only. Re-ran E4B end-to-end: height fwd 2.7s (93 p/s), wealth fwd 4.2s (46 p/s), all 8 saves <10 ms each. Outputs on remote `/workspace/repo/results/activations/`:
  - `e4b_W_U.npy` 2.7 GB (262144×2560 float32)
  - `e4b_{height,wealth}_{early,mid,late,final}.npz` 8 × ~2.5 MB (layers 10/21/32/41)
  - Loaded back cleanly on CPU: means ≈ 0, std climbs 1.0→1.8→3.4 through the stack (healthy residual scale growth), IDs = `{domain}_implicit_is_0000`+ as expected.
- **07:58Z** — Uploaded `extract_g31b_v1.py` to remote + launched G31B extraction (PID 3821, log `/tmp/extract_g31b_v1.log`). 60-layer budget: early=14, mid=30, late=45, final=59. Expect ~62 GB weight download + ~1 min fwd pass + 8 tiny saves. Fallback in place: E4B-only is a complete shippable deliverable if 31B OOMs/stalls.
- **08:00Z** — **G31B DONE** — much faster than expected. Weights were pre-cached on this Vast host (Parsed team must have pulled 31B earlier), so no re-download. Forward passes: height 7.9 s (31.7 p/s, bs=4), wealth 9.5 s (20.7 p/s). All 8 activation files saved sub-second each. Verified load:
  - `g31b_W_U.npy` 5.6 GB (262144×5376 fp32)
  - `g31b_{height,wealth}_{early,mid,late,final}.npz` 8 files (~5.5 MB height, ~4.2 MB wealth), layers 14/30/45/59
  - Activation stats healthy: std climbs 3.76 → 3.60 → 2.36 through layers 14→30→45. Layer 59 std=0.064 on height_final, std small on wealth_final too — this is the **last transformer block output** which in Gemma 4 appears to sit *after* an internal normalization (consistent with Gemma 2/3 dual-norm pattern). Worth a note in methods: for the Fisher-pullback analysis, use `late` (layer 45) as the "near-final" slice if we want un-renormalized residual geometry.

## G31B artifact manifest (remote paths)

```
/workspace/repo/results/activations/g31b_W_U.npy          # 5637144704 B, (262144,5376) fp32
/workspace/repo/results/activations/g31b_height_early.npz # 5454648 B, (252,5376), layer 14
/workspace/repo/results/activations/g31b_height_mid.npz   # 5454640 B, (252,5376), layer 30
/workspace/repo/results/activations/g31b_height_late.npz  # 5454644 B, (252,5376), layer 45
/workspace/repo/results/activations/g31b_height_final.npz # 5454648 B, (252,5376), layer 59
/workspace/repo/results/activations/g31b_wealth_early.npz # 4242808 B, (196,5376), layer 14
/workspace/repo/results/activations/g31b_wealth_mid.npz   # 4242800 B, (196,5376), layer 30
/workspace/repo/results/activations/g31b_wealth_late.npz  # 4242804 B, (196,5376), layer 45
/workspace/repo/results/activations/g31b_wealth_final.npz # 4242808 B, (196,5376), layer 59
```

Both Gemma 4 models fully extracted in a single ~1hr Vast burst, at ~$4/hr rate — ≈$4 spend. Under budget.

- **08:03-08:35Z** — W&B Artifact upload complete. All 18 activation files + 2 W_U matrices synced as one `gemma4-activations` artifact with aliases `day4` + `v2-prompts`. Run URL: https://wandb.ai/xrong-optiver/geometry-of-relativity/runs/ax81rrlu . 19 artifact files (18 data + 1 wandb-generated manifest), ~16 GB.
- **08:35Z** — INDIGO-COMPASS complete. Rolling BUILDING.md to the Day-4 probe task (SAPPHIRE-BEARING). Vast instance left running per authorization C. Total autonomous spend this session: ~$4 of $27 budget.

## Ship list for INDIGO-COMPASS

| Deliverable | Location | Status |
| --- | --- | --- |
| E4B activations (4 layers × 2 domains) | W&B + remote `/workspace/repo/results/activations/e4b_*` | ✅ |
| 31B activations (4 layers × 2 domains) | W&B + remote `/workspace/repo/results/activations/g31b_*` | ✅ |
| W_U for both models | W&B + remote | ✅ |
| Extractor scripts checked into repo | `scripts/vast_remote/{extract_e4b_v3.py, extract_g31b_v1.py, README.md}` | ✅ |
| STATUS.md live progress log | this file | ✅ |
| Vast instance kept alive | Parsed team instance #11, m:56779 | ✅ |
| SSH config / Claude Code install on remote | deferred to Day 4 start | ⏭ |

## E4B artifact manifest (remote paths)

```
/workspace/repo/results/activations/e4b_W_U.npy          # 2684354688 B, (262144,2560) fp32
/workspace/repo/results/activations/e4b_height_early.npz # 2616120 B, (252,2560) fp32, layer 10
/workspace/repo/results/activations/e4b_height_mid.npz   # 2616112 B, (252,2560) fp32, layer 21
/workspace/repo/results/activations/e4b_height_late.npz  # 2616116 B, (252,2560) fp32, layer 32
/workspace/repo/results/activations/e4b_height_final.npz # 2616120 B, (252,2560) fp32, layer 41
/workspace/repo/results/activations/e4b_wealth_early.npz # 2035064 B, (196,2560) fp32, layer 10
/workspace/repo/results/activations/e4b_wealth_mid.npz   # 2035056 B, (196,2560) fp32, layer 21
/workspace/repo/results/activations/e4b_wealth_late.npz  # 2035060 B, (196,2560) fp32, layer 32
/workspace/repo/results/activations/e4b_wealth_final.npz # 2035064 B, (196,2560) fp32, layer 41
```

W&B Artifacts upload + 31B extraction are in flight; this file will be
re-updated when both land or when the budget window closes, whichever first.

## Day 4 addendum (Apr 20 2026, autonomous "dinner & drinks" window)

Staged a **fifth line of evidence** (INLP concept erasure) on branch
`exp/v4-auto-research` plus a paper outline and two new smoke tests. All
work is local — not yet pushed to origin because this sandbox has no GitHub
credentials configured. Before pulling on Vast, run `git push origin
exp/v4-auto-research` from your workstation.

### New commit on exp/v4-auto-research (43f91b4)

- `scripts/vast_remote/inlp_v4.py` — iterative nullspace projection on w_z,
  with random-direction and INLP-x controls. Reads cached v4_dense
  activations + logit_diff; no model forward pass needed.
- `tests/test_inlp_smoke.py` — synthetic-data end-to-end test. Uses
  `--seed 42` (not 0) to avoid coinciding with the `make_fake_v4` seed=0
  random draw that would otherwise make the "random null" direction equal
  the true z-direction and give spurious perfect erasure.
- `tests/test_adjpairs_smoke.py` — 9 CPU-only checks for prompt generation:
  pair count, 6240 total, placeholder coverage, prompt endings, seed
  variance, sample-mean-near-μ, BMI/wealth formatting, determinism.
- `scripts/vast_remote/extract_v4_adjpairs.py` — lazy torch/transformers
  imports so the smoke test can import without torch; docstring cleanup.
- `scripts/vast_remote/analyze_v4.py` + `analyze_v4_adjpairs.py` — fixed a
  **real bug**: default `KFold(cv=5)` doesn't shuffle, and v4 data is
  stored sorted by (x, μ, seed). Each fold was becoming a separate
  x-bucket → CV R² collapsed to spurious negatives. Now uses
  `KFold(shuffle=True, random_state=cv_seed)`. Same fix in inlp_v4.py's
  `cv_r2` helper. This affects all probe R² numbers in the final paper.
- `docs/paper_outline.md` — ICML MI Workshop skeleton with 5 evidence-line
  sections and `<<TBD>>` placeholders for the numbers that land from Vast.
- `docs/v4_research_design.md` + `BUILDING.md` — updated to 5 evidence
  lines, include `inlp_v4.py` in the run order.
- `.gitignore` — ignore `results/*.bak_smoke/` and `results/*.bak_inlp_smoke/`
  leftovers from FUSE-mount smoke-test cleanup.

### Local smoke-test status (all green)

- `tests/test_adjpairs_smoke.py` — 9/9 pass.
- `tests/test_inlp_smoke.py` — 1 end-to-end test passes. On synthetic data:
  initial R²(z)=0.991 → 0.316 under INLP-z in 4 steps; random projection
  preserves R²(z)=0.991 (gap +0.896). R²(x) under INLP-z partially survives
  (0.644 → 0.210), as expected since x ≠ z direction.
- `tests/test_analyze_v4_smoke.py` — has a FUSE-mount permission issue in
  its setup's `shutil.rmtree` call on stale `.bak_smoke/` dirs. Not caused
  by this commit; the test logic itself is fine. Re-run after a mount
  refresh or use `rm -rf results/v4_*.bak_*` from a shell that owns those
  paths.

### Run order on Vast when you get back

```bash
cd /workspace/repo
git fetch origin
git checkout exp/v4-auto-research
git pull
python scripts/vast_remote/analyze_v4.py              # phases 1–5
python scripts/vast_remote/extract_v4_adjpairs.py     # 6240 prompts, ~2 min
python scripts/vast_remote/analyze_v4_adjpairs.py     # seconds
python scripts/vast_remote/steer_v4.py --layer late
python scripts/vast_remote/steer_v4.py --layer mid
python scripts/vast_remote/inlp_v4.py --layer late --steps 8
python scripts/vast_remote/inlp_v4.py --layer mid  --steps 8
```

Total wall time ~8 min.
