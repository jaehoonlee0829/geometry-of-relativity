# BUILDING.md — What I am doing RIGHT NOW

Only one task in this file at a time. When done, move it to TODO.md "done" section and pull the next one in.

## Active task (Day 3, Apr 20 2026)

**Initialize git, push scaffold to public GitHub, add CI**

Specifically:

1. `cd /path/to/mech-interp-relativity && git init && git add . && git commit -m "Initial scaffold (Day 3): PLANNING/BUILDING/TODO/CLAUDE docs + src stubs + behavioral battery v0 + Day-1 kill-test results"`
2. Create public GitHub repo named `mech-interp-relativity`. Use `gh repo create mech-interp-relativity --public --source=. --push` if `gh` CLI is authenticated, otherwise give the human the exact CLI they need to run.
3. Write `.github/workflows/ci.yml` that runs `pytest tests/` on push to main. Only Python 3.10, install from requirements.txt, skip GPU-dependent tests via `pytest -m "not gpu"`.
4. Write `tests/test_fisher.py` — at minimum, the finite-difference sanity check: F(h) times a random v should approximate the Jacobian of (∂p/∂h) · something. Concretely: verify that `(p_{h+εv} - p_{h-εv}) / (2ε) ≈ J @ v` where `J = (diag(p) - p p^T) W_U` (the Jacobian of p w.r.t. h).
5. Write `tests/test_probe.py` — verify the synthetic recovery test from `src/probe.py::__main__` runs as a pytest.
6. Write `tests/test_data_gen.py` — verify that `generate_trials` with `canonical_v0_contexts()` and `adjectives=["tall","obese","short"]`, `target_values=[155,165,175,32,3,10,300]` produces at least 40 trials and all have valid adjective_class.
7. Push, verify CI turns green on the first run.

### Definition of done

- Public GitHub repo exists, scaffold pushed
- First CI run is green
- BUILDING.md updated to pull next task: "Rent Vast.ai GPU, extract Gemma-2-2b activations on prompts_v0.jsonl"

### Completion promise word

When this task is done, the agent writes `PURPLE-HAMMER` on the last line of the final commit message.

### Notes for the agent running this (Ralph or Claude Code with --dangerously-skip-permissions)

- The repo currently lives at `/sessions/elegant-fervent-edison/mnt/outputs/mech-interp-relativity` (sandbox workspace). For the human, this is their Cowork workspace folder. If the human has their own git/GitHub credentials available, they should run the push commands themselves — the sandbox does not have persistent GitHub auth.
- If `gh` CLI is not authenticated, print the commands for the human and exit with `PURPLE-HAMMER-PENDING` instead — do NOT attempt to hardcode any credentials.
