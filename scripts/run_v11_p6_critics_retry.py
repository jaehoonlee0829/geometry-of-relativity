"""Retry the 4 P6 critics that 429'd on the first run.

Fixes vs run_v11_p6_critics.py:
  - 65-second sleep between calls (Anthropic limit is 30k input tokens / minute
    for opus-4-7 on this org; one big call eats most of a minute)
  - context blob trimmed to 50k chars (down from 120k)
  - exponential backoff on 429 (up to 3 retries)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from run_v11_p6_critics import CRITIC_PROMPTS  # noqa: E402

CONTEXT_CAP = 50000
SLEEP_BETWEEN = 65


def load_results_summary() -> str:
    out_dir = REPO / "results" / "v11"
    bits: list[str] = []
    for model in ("gemma2-2b", "gemma2-9b"):
        mdir = out_dir / model
        if not mdir.exists(): continue
        for p in sorted(mdir.glob("*.json")):
            content = p.read_text()
            if len(content) > 4000:
                content = content[:4000] + "\n... [truncated]"
            bits.append(f"--- {p.relative_to(REPO)} ---\n{content}")
        for pair_dir in sorted(mdir.iterdir()):
            if not pair_dir.is_dir(): continue
            for p in sorted(pair_dir.glob("*.json")):
                if "meta" in p.name: continue
                content = p.read_text()
                if len(content) > 1500:
                    content = content[:1500] + "\n... [truncated]"
                bits.append(f"--- {p.relative_to(REPO)} ---\n{content}")
    return "\n\n".join(bits)[:CONTEXT_CAP]


def list_figures() -> str:
    fig_dir = REPO / "figures" / "v11"
    if not fig_dir.exists(): return "<no figures>"
    return "\n".join(sorted(str(p.relative_to(REPO)) for p in fig_dir.rglob("*.png")))


def call_critic(name: str, prompt: str, results_blob: str, figs: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    project_context = ""
    for f in ("PLANNING.md", "STATUS.md"):
        p = REPO / f
        if p.exists():
            project_context += f"\n\n=== {f} ===\n" + p.read_text()[:3500]
    findings = REPO / "FINDINGS.md"
    findings_tail = ""
    if findings.exists():
        text = findings.read_text()
        findings_tail = "\n\n=== FINDINGS.md (last 6000 chars) ===\n" + text[-6000:]
    sys_prompt = (
        "You are an independent research critic. Be specific, terse, and concrete. "
        "Cite file paths, layer numbers, and metric values from the supplied JSONs. "
        "Cap your reply at ~500 words.\n\n"
        f"{prompt}"
    )
    user = (
        f"PROJECT CONTEXT:\n{project_context}{findings_tail}\n\n"
        f"V11 RESULT JSONS (truncated):\n{results_blob}\n\n"
        f"V11 FIGURES:\n{figs}\n\n"
        f"Write your critique."
    )
    last_err = None
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=3000,
                system=sys_prompt,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        except Exception as e:
            last_err = e
            if "rate_limit" in str(e) or "429" in str(e):
                wait = 70 * (attempt + 1)
                print(f"   429 on attempt {attempt + 1}, sleeping {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"3 attempts failed: {last_err}")


def main():
    out_dir = REPO / "results" / "v11"
    failed = ("alternative", "statistical", "novelty", "narrative")
    blob = load_results_summary()
    figs = list_figures()
    print(f"[P6-retry] blob {len(blob)} chars, figures {figs.count(chr(10)) + 1}, "
          f"sleep {SLEEP_BETWEEN}s between critics")

    for i, name in enumerate(failed):
        if i > 0:
            print(f"[P6-retry] sleeping {SLEEP_BETWEEN}s for rate limit...", flush=True)
            time.sleep(SLEEP_BETWEEN)
        print(f"[P6-retry] running critic: {name}...", flush=True)
        try:
            text = call_critic(name, CRITIC_PROMPTS[name], blob, figs)
        except Exception as e:
            text = f"<critic {name} failed after retries: {e}>"
            print(f"[P6-retry]   FAILED: {e}", file=sys.stderr)
        out_path = out_dir / f"critic_{name}.md"
        out_path.write_text(f"# {name} critic\n\n{text}\n")
        print(f"[P6-retry]   wrote {out_path.relative_to(REPO)} ({len(text)} chars)")

    # Rebuild summary
    summary_lines = ["# v11 P6 critic round\n"]
    for name in ("methodology",) + failed:
        p = out_dir / f"critic_{name}.md"
        if p.exists():
            text = p.read_text()
            summary_lines.append(f"## {name}\n\n{text[:1500]}\n\n*(see {p.name} for full)*")
    (out_dir / "critics_summary.md").write_text("\n".join(summary_lines))
    print(f"[P6-retry] rewrote {out_dir / 'critics_summary.md'}")


if __name__ == "__main__":
    main()
