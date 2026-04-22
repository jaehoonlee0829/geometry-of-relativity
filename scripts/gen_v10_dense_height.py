"""Generate v10 dense grid trials for height only.

20 x-values × 20 z-values × 10 seeds = 4,000 implicit prompts.
Height pair: short/tall, sigma=10 cm.

Writes: data_gen/v10_dense_height_trials.jsonl

Run: python scripts/gen_v10_dense_height.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import sample_context, fmt_num  # noqa: E402

OUT = REPO / "data_gen"
OUT.mkdir(parents=True, exist_ok=True)

# Height pair config
SIGMA = 10.0
LOW_WORD = "short"
HIGH_WORD = "tall"
PROMPT_TEMPLATE = "{items}\nPerson {n_last}: {x_str} cm. This person is"

# Dense grid: 20 x-values × 20 z-values
X_VALUES = np.linspace(145, 190, 20).round(1).tolist()   # 145.0, 147.4, ..., 190.0
Z_VALUES = np.linspace(-3.0, 3.0, 20).round(2).tolist()  # -3.0, -2.68, ..., +3.0
N_SEEDS = 10


def derive_mu(x: float, z: float) -> float:
    return x - SIGMA * z


def is_plausible_mu(mu: float) -> bool:
    """Reject mu values that would make the context distribution implausible."""
    return 100.0 <= mu <= 230.0  # reasonable human height range for context mean


def build_items(mu: float, seed: int, n: int = 15) -> list[str]:
    """Sample 15 context heights and format."""
    low = 100.0
    high = 240.0
    sample = sample_context(mu, SIGMA, seed, n, low, high, log_space=False)
    return [f"Person {i+1}: {int(v)} cm" for i, v in enumerate(sample)]


def make_prompt(x: float, mu: float, seed: int) -> str:
    items = build_items(mu, seed)
    items_block = "\n".join(items)
    return PROMPT_TEMPLATE.format(
        items=items_block,
        n_last=len(items) + 1,
        x_str=fmt_num(x),
    )


def main():
    trials = []
    dropped = []
    idx = 0

    for x in X_VALUES:
        for z in Z_VALUES:
            mu = derive_mu(x, z)
            if not is_plausible_mu(mu):
                dropped.append({"x": x, "z": z, "mu": round(mu, 1)})
                continue
            for seed in range(N_SEEDS):
                prompt = make_prompt(x, mu, seed)
                trial = {
                    "id": f"v10_height_{idx:06d}",
                    "pair": "height",
                    "condition": "implicit",
                    "x": x,
                    "z": z,
                    "mu": round(mu, 2),
                    "sigma": SIGMA,
                    "seed": seed,
                    "low_word": LOW_WORD,
                    "high_word": HIGH_WORD,
                    "prompt": prompt,
                }
                trials.append(trial)
                idx += 1

    out_path = OUT / "v10_dense_height_trials.jsonl"
    with out_path.open("w") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")

    # Summary
    n_cells_total = len(X_VALUES) * len(Z_VALUES)
    n_cells_kept = (n_cells_total - len(dropped))
    print(f"Grid: {len(X_VALUES)} x-values × {len(Z_VALUES)} z-values = {n_cells_total} cells")
    print(f"Dropped {len(dropped)} cells (implausible mu), kept {n_cells_kept}")
    print(f"Seeds per cell: {N_SEEDS}")
    print(f"Total prompts: {len(trials)}")
    print(f"Wrote: {out_path}")

    if dropped:
        print(f"\nDropped cells (mu out of [100, 230]):")
        for d in dropped[:10]:
            print(f"  x={d['x']}, z={d['z']}, mu={d['mu']}")
        if len(dropped) > 10:
            print(f"  ... and {len(dropped) - 10} more")

    # Also save dropped cells for reference
    drop_path = OUT / "v10_dense_height_dropped.json"
    drop_path.write_text(json.dumps({"dropped": dropped, "n_dropped": len(dropped)}, indent=2))
    print(f"Wrote: {drop_path}")


if __name__ == "__main__":
    main()
