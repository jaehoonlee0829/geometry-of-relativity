"""
Day-1 behavioral kill-test runner.

Consumes prompts_v0.jsonl, calls Claude API, writes one JSON response per prompt to
results/behavioral_v0/<prompt_id>.json. The Day-1 kill-test question:

  Does the completion of "a person who is 165cm is considered ___" flip between
  "tall" and "short" across narrow-low (mu=150) vs narrow-high (mu=180) contexts?

H1 passes if >= 80% of narrow-low completions include "tall" and >= 80% of
narrow-high completions include "short" (or adjacent terms).
H2 (absolute-adjective control): completions for "BMI=32" should be "obese"
regardless of context, with > 80% agreement across narrow-low / narrow-high /
wide-symmetric.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/run_behavioral.py --model claude-opus-4-6 --n_samples 5

Each prompt is sampled N times at temperature=1.0 to observe the full
completion distribution.
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import sys
import time

try:
    import anthropic
except ImportError:
    print("pip install anthropic", file=sys.stderr)
    sys.exit(1)


REPO = pathlib.Path(__file__).resolve().parent.parent
PROMPTS_PATH = REPO / "data_gen" / "prompts_v0.jsonl"
OUT_DIR = REPO / "results" / "behavioral_v0"


def load_prompts() -> list[dict]:
    with PROMPTS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(client: anthropic.Anthropic, model: str, prompt: str, n: int) -> list[str]:
    completions = []
    for i in range(n):
        # We want a base-model-style next-token continuation, so we frame the
        # Claude chat as: "Complete the following fragment with the single
        # most natural next word or phrase. Do not add commentary."
        resp = client.messages.create(
            model=model,
            max_tokens=20,
            temperature=1.0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Complete the following sentence fragment naturally, "
                        "with only the word or short phrase that should come next. "
                        "Do not add any commentary, explanation, or punctuation beyond the "
                        "word itself.\n\n"
                        f"Fragment: \"{prompt}\"\n\n"
                        "Completion:"
                    ),
                }
            ],
        )
        completions.append(resp.content[0].text.strip())
    return completions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="Run only first N prompts (smoke test)")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    for p in prompts:
        out_path = OUT_DIR / f"{p['id']}.json"
        if out_path.exists():
            print(f"skip {p['id']} (already done)")
            continue
        print(f"running {p['id']} ...", end="", flush=True)
        t0 = time.time()
        try:
            completions = run_one(client, args.model, p["prompt"], args.n_samples)
        except anthropic.APIError as e:
            print(f"  API error: {e}")
            continue
        record = {
            **p,
            "model": args.model,
            "n_samples": args.n_samples,
            "completions": completions,
        }
        out_path.write_text(json.dumps(record, indent=2))
        print(f" ok ({time.time() - t0:.1f}s)")

    print(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
