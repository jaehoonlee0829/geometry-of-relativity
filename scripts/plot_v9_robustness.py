"""v9 robustness plots — consolidates the fix scripts into figures:

- Extended α curves + entropy (from steering_robust_summary.json)
- Multi-seed random null band (per pair) vs primal/tangent slopes
- Held-out CV: in-sample vs out-of-sample slopes for primal and probe
- Park-at-layer-25 vs Park-at-layer-20 comparison (λ sweep)
- Bootstrap CI bars for behavioral R (already saved, plot again for consistency)

Writes
  figures/v9/steering_extended_alpha.png
  figures/v9/steering_multiseed_null.png
  figures/v9/steering_heldout_cv.png
  figures/v9/park_layer20_vs_layer25.png
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


def maybe_load(path):
    if not path.exists():
        print(f"  (skipping — missing {path})")
        return None
    return json.loads(path.read_text())


def plot_steering_extended_alpha():
    rob = maybe_load(RES_DIR / "steering_robust_summary.json")
    if rob is None:
        return
    alphas = rob["extended_alphas"]
    rows = [json.loads(l) for l in (RES_DIR / "steering_robust_rows.jsonl").open()
            if l.strip()]
    # Group rows by (pair, direction, alpha)
    by = {}
    for r in rows:
        if r.get("experiment") != "extended_alpha":
            continue
        by.setdefault((r["pair"], r["direction"]), {}).setdefault(r["alpha"], []) \
          .append((r["logit_diff"], r["entropy"]))

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, p in zip(axes.flat, PAIRS):
        for dname, color in [("primal", "C0"), ("tangent", "C1")]:
            if (p.name, dname) not in by: continue
            d = by[(p.name, dname)]
            xs = alphas
            ys = [np.mean([ld for ld, _ in d.get(a, [(0, 0)])]) for a in xs]
            es = [np.mean([ent for _, ent in d.get(a, [(0, 0)])]) for a in xs]
            ax.plot(xs, ys, "o-", color=color, label=f"{dname}  (ld)")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_title(p.name, fontsize=9)
        ax.set_xlabel("α"); ax.set_ylabel("logit_diff")
        ax.legend(fontsize=7)
    fig.suptitle("v9 robustness — logit_diff vs α (extended range)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "steering_extended_alpha.png", dpi=140)
    print(f"  wrote {FIG_DIR}/steering_extended_alpha.png")

    # Entropy-vs-α plot
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, p in zip(axes.flat, PAIRS):
        for dname, color in [("primal", "C0"), ("tangent", "C1")]:
            if (p.name, dname) not in by: continue
            d = by[(p.name, dname)]
            xs = alphas
            es = [np.mean([ent for _, ent in d.get(a, [(0, 0)])]) for a in xs]
            ax.plot(xs, es, "o-", color=color, label=dname)
        ax.set_title(p.name, fontsize=9)
        ax.set_xlabel("α"); ax.set_ylabel("entropy (nats)")
        ax.legend(fontsize=7)
    fig.suptitle("v9 robustness — entropy vs α (tangent kinder at high |α|?)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "steering_extended_alpha_entropy.png",
                                    dpi=140)
    print(f"  wrote {FIG_DIR}/steering_extended_alpha_entropy.png")


def plot_multiseed_null():
    rob = maybe_load(RES_DIR / "steering_robust_summary.json")
    orig = maybe_load(RES_DIR / "steering_manifold_summary.json")
    if rob is None or orig is None:
        return
    pairs = [p.name for p in PAIRS]
    xs_b = np.arange(len(pairs))
    primal_slope = [orig["per_pair"][n]["primal"]["slope"] for n in pairs]
    tangent_slope = [orig["per_pair"][n]["tangent"]["slope"] for n in pairs]
    # Null band per pair from 30 seeds
    null_seeds = [rob["per_pair"][n]["null_slopes_seeds"] for n in pairs]
    null_q025 = [rob["per_pair"][n]["null_slope_q025"] for n in pairs]
    null_q975 = [rob["per_pair"][n]["null_slope_q975"] for n in pairs]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    # 95% null band
    ax.fill_between(xs_b, null_q025, null_q975, alpha=0.2, color="gray",
                    label="null 95% band (30 seeds)")
    # Individual null slopes as strip plot
    for i, slopes in enumerate(null_seeds):
        ax.scatter([i] * len(slopes), slopes, s=6, color="gray", alpha=0.4)
    ax.plot(xs_b, primal_slope, "o", color="C0", markersize=10,
            label="primal_z slope", zorder=5)
    ax.plot(xs_b, tangent_slope, "s", color="C1", markersize=10,
            label="tangent(z) slope", zorder=5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(xs_b); ax.set_xticklabels(pairs, rotation=30, ha="right")
    ax.set_ylabel("Δlogit_diff per α (slope)")
    ax.set_title("v9 robustness — 30-seed random null band vs primal/tangent slopes\n"
                 "primal far above null; tangent also clearly above null for all pairs",
                 fontsize=10)
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "steering_multiseed_null.png", dpi=140)
    print(f"  wrote {FIG_DIR}/steering_multiseed_null.png")


def plot_cv():
    rob = maybe_load(RES_DIR / "steering_robust_summary.json")
    if rob is None:
        return
    pairs = [p.name for p in PAIRS]
    xs_b = np.arange(len(pairs))
    def get(name, key):
        return [rob["per_pair"][n]["cv_slopes"][key]["mean"] for n in pairs]
    def err(name, key):
        return [rob["per_pair"][n]["cv_slopes"][key]["std"] for n in pairs]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    w = 0.18
    for i, (k, col, lbl) in enumerate([
        ("primal_in",  "C0", "primal (in-sample)"),
        ("primal_out", "C2", "primal (5-fold OUT)"),
        ("probe_in",   "C1", "probe (in-sample)"),
        ("probe_out",  "C3", "probe (5-fold OUT)"),
    ]):
        mean = get(pairs, k); sd = err(pairs, k)
        ax.bar(xs_b + (i - 1.5) * w, mean, width=w, yerr=sd,
               label=lbl, color=col, alpha=0.85, capsize=3)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_xticks(xs_b); ax.set_xticklabels(pairs, rotation=30, ha="right")
    ax.set_ylabel("Δlogit_diff per α (slope)")
    ax.set_title("v9 robustness — held-out CV: in-sample vs out-of-sample slopes\n"
                 "(if primal's advantage collapses on OUT, it was a fit-set artifact)",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(FIG_DIR / "steering_heldout_cv.png", dpi=140)
    print(f"  wrote {FIG_DIR}/steering_heldout_cv.png")


def plot_park_layer_comparison():
    p20 = maybe_load(RES_DIR / "park_causal_summary.json")
    p25 = maybe_load(RES_DIR / "park_layer25_summary.json")
    if p20 is None or p25 is None:
        return
    pairs = [p.name for p in PAIRS]
    xs_b = np.arange(len(pairs))
    l20_primal = [p20["per_pair"][n]["primal"]["slope"] for n in pairs]
    l20_probe  = [p20["per_pair"][n]["probe"]["slope"]  for n in pairs]
    l20_causal = [p20["per_pair"][n]["probe_causal"]["slope"] for n in pairs]
    l25_primal = [p25["per_pair"][n]["slopes"]["primal"] for n in pairs]
    l25_probe  = [p25["per_pair"][n]["slopes"]["probe"]  for n in pairs]
    # best-λ probe_causal at layer 25
    lam_keys = [k for k in p25["per_pair"][pairs[0]]["slopes"]
                if k.startswith("probe_causal_lam")]
    l25_causal_best_lam = []
    for n in pairs:
        slopes = [(k, p25["per_pair"][n]["slopes"][k]) for k in lam_keys]
        best = max(slopes, key=lambda kv: kv[1])  # most positive slope
        l25_causal_best_lam.append(best[1])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    w = 0.25
    a1.bar(xs_b - w, l20_primal, width=w, label="primal", color="C0")
    a1.bar(xs_b,     l20_probe,  width=w, label="probe",  color="C1")
    a1.bar(xs_b + w, l20_causal, width=w, label="probe_causal (λ=1e-2)",
           color="C2")
    a1.set_xticks(xs_b); a1.set_xticklabels(pairs, rotation=30, ha="right")
    a1.set_ylabel("slope"); a1.set_title("Layer 20 (v9 P4)", fontsize=10)
    a1.axhline(0, color="k", lw=0.3); a1.legend(fontsize=8)

    a2.bar(xs_b - w, l25_primal, width=w, label="primal", color="C0")
    a2.bar(xs_b,     l25_probe,  width=w, label="probe",  color="C1")
    a2.bar(xs_b + w, l25_causal_best_lam, width=w,
           label="probe_causal (best-λ over sweep)", color="C2")
    a2.set_xticks(xs_b); a2.set_xticklabels(pairs, rotation=30, ha="right")
    a2.set_ylabel("slope"); a2.set_title("Layer 25 — pre-unembedding", fontsize=10)
    a2.axhline(0, color="k", lw=0.3); a2.legend(fontsize=8)

    fig.suptitle("v9 robustness — Park causal steering at two layers\n"
                 "probe_causal does NOT reach primal at either layer", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "park_layer20_vs_layer25.png", dpi=140)
    print(f"  wrote {FIG_DIR}/park_layer20_vs_layer25.png")


def main():
    print("=== Generating robustness plots ===")
    plot_steering_extended_alpha()
    plot_multiseed_null()
    plot_cv()
    plot_park_layer_comparison()


if __name__ == "__main__":
    main()
