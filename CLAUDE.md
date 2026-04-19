# CLAUDE.md — Instructions for any Claude instance working on this repo

## How to work on this repo

1. **Read PLANNING.md first.** It is the frozen spec. Everything in BUILDING.md and TODO.md must stay consistent with PLANNING.md. If you find an inconsistency, STOP and surface it to the human — do NOT silently reconcile.

2. **Read BUILDING.md next.** It tells you the single active task. Do NOT start work on anything in TODO.md until BUILDING.md is empty/done.

3. **One task, one commit.** When BUILDING.md's active task is done, the last line of your commit message must include the task's "completion promise word" (e.g., `RIPE-MANGO`). That token is how the Ralph loop knows the iteration completed successfully.

4. **After committing, update TODO.md.** Move the active task to the "done" section. Pull the next queued task into BUILDING.md with concrete acceptance criteria and a fresh completion promise word. Then exit.

5. **Don't change PLANNING.md unless explicitly told to.** It's the frozen spec for a reason.

## Scientific context

This is a mech-interp study of gradable adjectives. The key idea: linear probes for relative adjectives ("tall", "short") should track a *context-dependent Z-score*, not a raw numerical value, while probes for absolute adjectives ("obese") should track the raw value. The distinction is made rigorous via Fisher-information pullback `F(h)⁻¹·w`.

When implementing:
- Fisher matrix `F(h) = W_U^T (diag(p) − p p^T) W_U` — never construct `p p^T` explicitly for vocab_size=256k; use low-rank factor tricks.
- Probe inverse `F⁻¹·w` via Cholesky (`scipy.linalg.cho_solve`), NEVER `np.linalg.inv`.
- Default dtype: float64 for Fisher math, float32/bfloat16 for model forward passes.

## Style / conventions

- Python 3.10, type hints everywhere
- Black formatting (line length 100)
- `ruff` for linting (rules in `pyproject.toml`)
- Tests live in `tests/`, run with `pytest`
- `scripts/` is for drivers with `argparse`, `src/` is for importable modules

## Model IDs (canonical)

```python
MODELS = {
    "gemma-2-2b": "google/gemma-2-2b",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}
```

## Compute

- CPU is fine for: prompt generation, probe training, Fisher math, plotting
- GPU is needed for: HuggingFace model forward passes (Gemma-2-2b, Llama-3.2-3B)
- Rent Vast.ai in 2–4h bursts; cache activations locally between bursts

## Secrets policy

- Never commit `ANTHROPIC_API_KEY`, `HF_TOKEN`, or any Vast.ai credentials
- Use `.env` for local dev, which is gitignored
- CI reads from GitHub Secrets
