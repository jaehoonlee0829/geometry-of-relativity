"""v1 behavioral runner — async concurrency, reads prompts_v1.jsonl, writes per-prompt JSON.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt  # if inside sandbox
  python scripts/run_behavioral_v1.py --model claude-sonnet-4-6 --n_samples 10 --concurrency 5
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

try:
    from anthropic import AsyncAnthropic, APIError
except ImportError:
    print("pip install anthropic", file=sys.stderr)
    sys.exit(1)

REPO = pathlib.Path(__file__).resolve().parent.parent
PROMPTS_PATH = REPO / "data_gen" / "prompts_v1.jsonl"
OUT_DIR = REPO / "results" / "behavioral_v1"


def load_prompts():
    with PROMPTS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def format_user_message(prompt: str) -> str:
    return (
        "Complete the following sentence fragment naturally, with only the word or "
        "short phrase that should come next. Do not add any commentary, explanation, "
        "or punctuation beyond the word itself.\n\n"
        f"Fragment: \"{prompt}\"\n\n"
        "Completion:"
    )


_RATE_LIMITER_LOCK = asyncio.Lock()
_LAST_CALL_TIME = [0.0]
_MIN_INTERVAL = 60.0 / 40.0  # 40 calls per minute = 1.5s between starts


async def _throttle():
    async with _RATE_LIMITER_LOCK:
        now = asyncio.get_event_loop().time()
        wait = _LAST_CALL_TIME[0] + _MIN_INTERVAL - now
        if wait > 0:
            await asyncio.sleep(wait)
        _LAST_CALL_TIME[0] = asyncio.get_event_loop().time()


async def sample_one(client: AsyncAnthropic, model: str, prompt: str, sem: asyncio.Semaphore) -> str:
    """Retry on 429 rate limit with exponential backoff."""
    async with sem:
        for attempt in range(8):
            await _throttle()
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=20,
                    temperature=1.0,
                    messages=[{"role": "user", "content": format_user_message(prompt)}],
                )
                return resp.content[0].text.strip()
            except APIError as e:
                msg = str(e)
                is_429 = "429" in msg or "rate_limit" in msg
                if is_429 and attempt < 7:
                    wait = min(2 ** attempt, 30) + 0.5 * attempt
                    await asyncio.sleep(wait)
                    continue
                return f"__ERROR__ {e}"
        return "__ERROR__ exhausted retries"


async def run_prompt(client, model, row, n_samples, sem) -> dict:
    tasks = [sample_one(client, model, row["prompt"], sem) for _ in range(n_samples)]
    completions = await asyncio.gather(*tasks)
    return {**row, "model": model, "n_samples": n_samples, "completions": completions}


async def main_async(model: str, n_samples: int, concurrency: int, limit: int | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_prompts()
    if limit:
        rows = rows[:limit]
    def _needs_rerun(r):
        p = OUT_DIR / f"{r['id']}.json"
        if not p.exists():
            return True
        try:
            data = json.loads(p.read_text())
            return any(c.startswith("__ERROR__") for c in data.get("completions", []))
        except Exception:
            return True
    remaining = [r for r in rows if _needs_rerun(r)]
    print(f"{len(rows)} total prompts, {len(remaining)} to run, {len(rows) - len(remaining)} cached")
    if not remaining:
        return
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    done = 0
    # Kick off all prompts as a single gather; the semaphore throttles to `concurrency`.
    tasks = [run_prompt(client, model, r, n_samples, sem) for r in remaining]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        out_path = OUT_DIR / f"{result['id']}.json"
        out_path.write_text(json.dumps(result, indent=2))
        done += 1
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(remaining) - done) / rate if rate > 0 else float("inf")
        print(f"[{done}/{len(remaining)}] {result['id']}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    print(f"\nDone in {time.time() - t0:.0f}s. Results in {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: only first N prompts")
    args = ap.parse_args()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    asyncio.run(main_async(args.model, args.n_samples, args.concurrency, args.limit))


if __name__ == "__main__":
    main()
