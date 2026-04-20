# STATUS.md — Live progress during 2hr autonomous burst

(Previous Day-1 STATUS content is archived in `docs/status_day1.md` and the "Done" block of `TODO.md`.)

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
