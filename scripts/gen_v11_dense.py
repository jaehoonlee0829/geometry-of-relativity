"""Generate v11 dense grid trials for any pair.

Per pair: 20 x-values × 20 z-values × 10 seeds = 4,000 implicit prompts.

Reuses the per-pair templates / context-sampling logic from
``scripts/vast_remote/extract_v4_adjpairs.py`` (the canonical PAIRS list)
so the prompt strings stay byte-identical across v9 / v10 / v11.

Writes: ``data_gen/v11_<pair>_trials.jsonl``

Usage:
    python scripts/gen_v11_dense.py --pair height
    python scripts/gen_v11_dense.py --pair all      # generate all 8 pairs

Per-pair seed offset prevents correlated nulls across pairs (methodology critic
flagged seed reuse). Each pair's seed = pair_seed_offset + cell_seed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import (  # noqa: E402
    PAIRS,
    LOG_SPACE_PAIRS,
    Pair,
    build_implicit_items,
    fmt_num,
    compute_z,
)

OUT = REPO / "data_gen"
OUT.mkdir(parents=True, exist_ok=True)

N_X = 20
N_Z = 20
N_SEEDS = 10
Z_MIN, Z_MAX = -3.0, 3.0


def x_grid(pair: Pair) -> list[float]:
    """20 x-values spanning the pair's target range, plus a buffer either side."""
    lo = float(min(pair.target_values))
    hi = float(max(pair.target_values))
    if pair.name in LOG_SPACE_PAIRS:
        # Log-space wealth: log-spaced grid over [lo*0.7, hi*1.5].
        log_lo = math.log(lo * 0.7)
        log_hi = math.log(hi * 1.5)
        log_xs = np.linspace(log_lo, log_hi, N_X)
        xs = np.exp(log_xs)
        # Round to whole dollars/units to keep tokenization stable.
        return [round(x, 0) if x >= 100 else round(x, 1) for x in xs.tolist()]
    span = hi - lo
    # Never go below ~0 — negative x makes no physical sense for any v11 pair
    # and the extractor regex can't parse negative values.
    x_lo = max(0.5, lo - 0.1 * span)
    return np.linspace(x_lo, hi + 0.1 * span, N_X).round(1).tolist()


def z_grid() -> list[float]:
    return np.linspace(Z_MIN, Z_MAX, N_Z).round(2).tolist()


def derive_mu(pair: Pair, x: float, z: float) -> float:
    """Inverse of compute_z: mu given x and target z."""
    if pair.name in LOG_SPACE_PAIRS:
        # z = (log x - log mu) / log sigma   →   mu = x * sigma^(-z)
        return x * (pair.sigma ** (-z))
    return x - pair.sigma * z


def is_plausible_mu(pair: Pair, mu: float) -> bool:
    """Reject mu values that would make the context distribution implausible."""
    # Per-pair plausibility band — derived from the per-pair target range with
    # a generous multiplicative buffer. Tightening these is fine; the dropped
    # cells just shrink the grid.
    if pair.name in LOG_SPACE_PAIRS:
        lo = float(min(pair.target_values)) / 10.0
        hi = float(max(pair.target_values)) * 10.0
        return lo <= mu <= hi
    lo = float(min(pair.target_values)) * 0.4
    hi = float(max(pair.target_values)) * 2.5
    return lo <= mu <= hi


def make_implicit_prompt(pair: Pair, x: float, mu: float, seed: int) -> str:
    """Format an implicit prompt for (x, mu, seed). Uses the canonical
    per-pair items builder + format string."""
    items = build_implicit_items(pair, mu, seed)
    items_block = "\n".join(items)
    if pair.name == "wealth":
        # Wealth uses "earns ${x_str}/year" and the items use "earns ${v}/year"
        # — keep formatter consistent with extract_v4_adjpairs.format_prompt_implicit.
        x_str = fmt_num(x)
    else:
        x_str = fmt_num(x)
    return pair.format_prompt_implicit.format(
        items=items_block,
        n_last=len(items) + 1,
        x_str=x_str,
    )


def pair_seed_offset(pair_name: str) -> int:
    """Stable per-pair offset so two pairs don't share the same RNG draws."""
    return zlib.crc32(pair_name.encode()) & 0xFFFF  # 16-bit, deterministic


def gen_pair(pair: Pair) -> tuple[list[dict], list[dict]]:
    trials: list[dict] = []
    dropped: list[dict] = []
    xs = x_grid(pair)
    zs = z_grid()
    seed_off = pair_seed_offset(pair.name)
    idx = 0
    for x in xs:
        for z in zs:
            mu = derive_mu(pair, x, z)
            if not is_plausible_mu(pair, mu):
                dropped.append({"x": x, "z": z, "mu": round(mu, 2)})
                continue
            for cell_seed in range(N_SEEDS):
                seed = seed_off + cell_seed
                prompt = make_implicit_prompt(pair, x, mu, seed)
                trials.append({
                    "id": f"v11_{pair.name}_{idx:06d}",
                    "pair": pair.name,
                    "condition": "implicit",
                    "x": float(x),
                    "z": float(z),
                    "mu": round(float(mu), 4),
                    "sigma": float(pair.sigma),
                    "seed": int(seed),
                    "cell_seed": int(cell_seed),
                    "low_word": pair.low_word,
                    "high_word": pair.high_word,
                    "prompt": prompt,
                })
                idx += 1
    return trials, dropped


def write_pair(pair: Pair) -> None:
    trials, dropped = gen_pair(pair)
    trials_path = OUT / f"v11_{pair.name}_trials.jsonl"
    with trials_path.open("w") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")

    drop_path = OUT / f"v11_{pair.name}_dropped.json"
    drop_path.write_text(json.dumps({
        "pair": pair.name,
        "n_dropped_cells": len(dropped),
        "dropped": dropped,
    }, indent=2))

    n_cells_total = N_X * N_Z
    n_cells_kept = n_cells_total - len(dropped)
    print(f"[{pair.name:11s}]  cells={n_cells_kept}/{n_cells_total}  "
          f"trials={len(trials)}  →  {trials_path.relative_to(REPO)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True,
                    help="pair name (e.g. height) or 'all' for every pair")
    args = ap.parse_args()

    pairs_to_run: list[Pair]
    if args.pair == "all":
        pairs_to_run = list(PAIRS)
    else:
        match = [p for p in PAIRS if p.name == args.pair]
        if not match:
            raise SystemExit(
                f"unknown pair {args.pair!r}; valid: {[p.name for p in PAIRS]}"
            )
        pairs_to_run = match

    for p in pairs_to_run:
        write_pair(p)


if __name__ == "__main__":
    main()
