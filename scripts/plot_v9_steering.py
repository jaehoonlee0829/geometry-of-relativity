"""v9 P3 + P4 plots: consolidate steering experiments into figures.

Reads:
    results/v9_gemma2/steering_manifold_summary.json  (P3)
    results/v9_gemma2/park_causal_summary.json        (P4)

Writes:
    figures/v9/steering_manifold_slopes.png
    figures/v9/steering_manifold_entropy.png
    figures/v9/park_causal_slopes.png
    figures/v9/steering_all_comparison.png   (combined primal/tangent/probe/probe_causal)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import PAIRS  # noqa: E402

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def bar_compare(pair_names, series: dict[str, list[float]], ylabel: str,
                title: str, out: Path, hline0=True):
    n = len(pair_names)
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, vals) in enumerate(series.items()):
        offs = (i - (len(series) - 1) / 2) * width
        ax.bar(np.arange(n) + offs, vals, width=width, label=name)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(pair_names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend()
    if hline0:
        ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")


def main():
    pair_names = [p.name for p in PAIRS]
    p3 = json.loads((RES_DIR / "steering_manifold_summary.json").read_text())
    pp3 = p3["per_pair"]
    slopes_primal = [pp3[n]["primal"]["slope"] for n in pair_names]
    slopes_tangent = [pp3[n]["tangent"]["slope"] for n in pair_names]
    slopes_random = [pp3[n]["random"]["slope"] for n in pair_names]
    ent_primal = [pp3[n]["primal"]["entropy_shift_2"] for n in pair_names]
    ent_tangent = [pp3[n]["tangent"]["entropy_shift_2"] for n in pair_names]
    ent_random = [pp3[n]["random"]["entropy_shift_2"] for n in pair_names]

    bar_compare(
        pair_names,
        {"primal_z": slopes_primal, "tangent(z) — on-manifold": slopes_tangent,
         "random (null)": slopes_random},
        ylabel="Δlogit_diff per α (slope)",
        title="v9 P3 — steering slope: primal vs on-manifold tangent vs random",
        out=FIG_DIR / "steering_manifold_slopes.png",
    )
    bar_compare(
        pair_names,
        {"primal_z": ent_primal, "tangent(z)": ent_tangent, "random": ent_random},
        ylabel="Δentropy at |α|=2 (vs α=0)",
        title="v9 P3 — entropy damage at |α|=2 (lower magnitude = less damage)",
        out=FIG_DIR / "steering_manifold_entropy.png",
    )

    # P4 — Park causal
    p4_path = RES_DIR / "park_causal_summary.json"
    if p4_path.exists():
        p4 = json.loads(p4_path.read_text())
        pp4 = p4["per_pair"]
        slopes_pr = [pp4[n]["primal"]["slope"] for n in pair_names]
        slopes_probe = [pp4[n]["probe"]["slope"] for n in pair_names]
        slopes_probe_c = [pp4[n]["probe_causal"]["slope"] for n in pair_names]
        bar_compare(
            pair_names,
            {"primal_z": slopes_pr,
             "probe_z (Ridge)": slopes_probe,
             "probe_z causal-adjusted": slopes_probe_c},
            ylabel="Δlogit_diff per α (slope)",
            title="v9 P4 — Park causal metric: does (W_U^T W_U)^{-1}·probe_z close the gap?",
            out=FIG_DIR / "park_causal_slopes.png",
        )
        # Combined plot: all four directions, per pair
        fig, ax = plt.subplots(figsize=(13, 5.5))
        w = 0.18
        n = len(pair_names)
        xs = np.arange(n)
        series = [
            ("primal_z (P3)",          slopes_primal,   "C0"),
            ("tangent(z) (P3)",        slopes_tangent,  "C1"),
            ("probe_z Ridge (P4)",     slopes_probe,    "C2"),
            ("probe_z causal (P4)",    slopes_probe_c,  "C3"),
        ]
        for i, (lab, vals, col) in enumerate(series):
            off = (i - 1.5) * w
            ax.bar(xs + off, vals, width=w, label=lab, color=col)
        ax.set_xticks(xs); ax.set_xticklabels(pair_names, rotation=30, ha="right")
        ax.set_ylabel("Δlogit_diff per α (slope)")
        ax.set_title("v9 P3+P4 — steering slope across all directions, per pair",
                     fontsize=11)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "steering_all_comparison.png", dpi=140)
        print(f"  wrote {FIG_DIR}/steering_all_comparison.png")
    else:
        print(f"  (P4 summary not found yet at {p4_path})")


if __name__ == "__main__":
    main()
