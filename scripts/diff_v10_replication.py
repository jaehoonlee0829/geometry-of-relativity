"""Diff replication run vs original v10 numbers.

Compares JSON files in results/v10/ against backup at /tmp/v10_orig/ and
reports max absolute deltas per scalar key. Emits a markdown table suitable
for posting as a PR comment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OLD = Path("/tmp/v10_orig")
NEW = Path("results/v10")


def flatten(prefix: str, obj: Any, out: dict[str, float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(f"{prefix}[{i}]", v, out)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)


def diff_json(name: str) -> tuple[int, float, str | None, list[tuple[str, float, float, float]]]:
    a = OLD / name
    b = NEW / name
    if not a.exists():
        return 0, 0.0, f"old missing: {a}", []
    if not b.exists():
        return 0, 0.0, f"new missing: {b}", []
    A = json.loads(a.read_text())
    B = json.loads(b.read_text())
    fa, fb = {}, {}
    flatten("", A, fa)
    flatten("", B, fb)
    common = sorted(set(fa) & set(fb))
    deltas = []
    for k in common:
        if abs(fb[k]) > 1e-12 or abs(fa[k]) > 1e-12:
            d = fb[k] - fa[k]
            deltas.append((k, fa[k], fb[k], d))
    n = len(deltas)
    max_abs = max((abs(d) for *_, d in deltas), default=0.0)
    return n, max_abs, None, deltas


def main() -> None:
    files = ["behavioral_summary.json",
             "dimensionality_per_layer.json",
             "increment_r2_per_layer.json",
             "sae_feature_fits_L20.json",
             "attention_per_head.json",
             "attention_per_head_taxonomy.json"]

    print("# v10 replication diff")
    print()
    print("| File | n_keys compared | max |Δ| | top-5 deltas |")
    print("|---|---:|---:|---|")

    for f in files:
        n, mx, err, deltas = diff_json(f)
        if err:
            print(f"| `{f}` | — | — | {err} |")
            continue
        # Top 5 by absolute delta
        top = sorted(deltas, key=lambda r: -abs(r[3]))[:5]
        top_str = "; ".join(f"`{k}` Δ={d:+.4g}" for k, _, _, d in top) or "—"
        print(f"| `{f}` | {n} | {mx:.4g} | {top_str} |")

    # Always also print full delta blob to a file
    print()
    print("## Headline scalar comparison (behavioral_summary.json)")
    a = json.loads((OLD / "behavioral_summary.json").read_text())
    b = json.loads((NEW / "behavioral_summary.json").read_text())
    print()
    print("| key | original | replicate | Δ |")
    print("|---|---:|---:|---:|")
    for k in sorted(set(a) & set(b)):
        if isinstance(a[k], (int, float)):
            d = b[k] - a[k]
            print(f"| `{k}` | {a[k]:.6g} | {b[k]:.6g} | {d:+.4g} |")


if __name__ == "__main__":
    main()
