#!/usr/bin/env python3
"""Plot direction-confound cosine matrix — Grid B (clean) only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
JSON_PATH = REPO / "results" / "v7_analysis" / "direction_confound_audit_clean.json"
OUT_PATH = REPO / "figures" / "v7" / "direction_confound_matrix_gridB.png"

DIR_NAMES = ["primal_z", "primal_x", "probe_z", "probe_x", "pc1", "pc2", "zeroshot_wx"]


def main() -> None:
    with open(JSON_PATH) as f:
        data = json.load(f)

    # Compute mean |cos| across pairs from per_pair_gridB
    pairs = data["per_pair_gridB"]
    matrices = [np.array(pairs[k]) for k in pairs]
    mean_B = np.mean(np.abs(np.stack(matrices)), axis=0)  # (7, 7)

    n = len(DIR_NAMES)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mean_B, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = mean_B[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(DIR_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(DIR_NAMES, fontsize=8)

    ax.set_title("Direction cosines — Grid B (x, z) — v7 clean", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="|cos θ|")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
