"""Exp 5 plots from cached adjpair logits + trials.

Reads:
  results/v4_adjpairs/e4b_trials.jsonl
  results/v4_adjpairs/e4b_{pair}_{condition}_logits.jsonl

Writes into figures/v4_adjpairs/:
  logit_diff_heatmap_xmu_8panel.png    — 8-panel (x,μ) heatmaps (hero behavioral figure)
  logit_diff_heatmap_xz_8panel.png     — 8-panel (x,z) heatmaps (pure relativity)
  implicit_vs_explicit_scatter.png     — per-pair, per-cell
  logit_diff_vs_z_8panel.png           — colored by x, cell means overlaid
  zero_shot_bias_per_pair.png          — zero-shot logit_diff by x
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ADJPAIRS = Path("results/v4_adjpairs")
OUT = Path("figures/v4_adjpairs")
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = ["height", "age", "experience", "size", "speed", "wealth", "weight", "bmi_abs"]


def load_trials():
    by_id = {}
    for line in (ADJPAIRS / "e4b_trials.jsonl").open():
        t = json.loads(line)
        by_id[t["id"]] = t
    return by_id


def load_logits(pair: str, cond: str):
    path = ADJPAIRS / f"e4b_{pair}_{cond}_logits.jsonl"
    return [json.loads(line) for line in path.open()]


def merge(pair: str, cond: str, trials_by_id):
    for rec in load_logits(pair, cond):
        t = trials_by_id[rec["id"]]
        yield {**t, "logit_diff": rec["logit_diff"]}


def pair_title(p: str) -> str:
    return {
        "height": "tall / short (cm)",
        "age": "old / young (yrs)",
        "experience": "experienced / novice (yrs)",
        "size": "big / small (cm)",
        "speed": "fast / slow (km/h)",
        "wealth": "rich / poor ($k)",
        "weight": "heavy / light (kg)",
        "bmi_abs": "obese / thin (BMI)",  # extract_v4_adjpairs.py uses low_word='thin'
    }.get(p, p)


def heatmap_panel(ax, rows, x_key: str, y_key: str, title: str):
    xs = sorted({r[x_key] for r in rows})
    ys = sorted({r[y_key] for r in rows})
    M = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        i = ys.index(r[y_key])
        j = xs.index(r[x_key])
        if np.isnan(M[i, j]):
            M[i, j] = 0.0
            cnt = np.zeros_like(M)
        # we accumulate mean below via second pass
    counts = np.zeros_like(M)
    sums = np.zeros_like(M)
    for r in rows:
        i = ys.index(r[y_key])
        j = xs.index(r[x_key])
        if not np.isnan(r["logit_diff"]):
            sums[i, j] += r["logit_diff"]
            counts[i, j] += 1
    M = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    vmax = max(abs(np.nanmin(M)), abs(np.nanmax(M)))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:g}" for y in ys], fontsize=7)
    ax.set_xlabel(x_key, fontsize=8)
    ax.set_ylabel(y_key, fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def eight_panel_heatmap(trials_by_id, y_key: str, fname: str):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pair in zip(axes.ravel(), PAIRS):
        rows = list(merge(pair, "implicit", trials_by_id))
        im = heatmap_panel(ax, rows, "x", y_key, pair_title(pair))
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.suptitle(
        f"logit_diff(high−low) over ({y_key}, x) — implicit prompts, E4B layer late",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT/fname}")


def implicit_vs_explicit_scatter(trials_by_id):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, pair in zip(axes.ravel(), PAIRS):
        imp = list(merge(pair, "implicit", trials_by_id))
        exp = list(merge(pair, "explicit", trials_by_id))
        imp_cells = defaultdict(list)
        for r in imp:
            imp_cells[(r["x"], r["mu"])].append(r["logit_diff"])
        exp_cells = defaultdict(list)
        for r in exp:
            exp_cells[(r["x"], r["mu"])].append(r["logit_diff"])
        xs, ys = [], []
        for k in sorted(set(imp_cells) & set(exp_cells)):
            xs.append(np.mean(imp_cells[k]))
            ys.append(np.mean(exp_cells[k]))
        ax.scatter(xs, ys, alpha=0.7, s=30)
        lo = min(min(xs, default=0), min(ys, default=0))
        hi = max(max(xs, default=0), max(ys, default=0))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        if xs:
            r = np.corrcoef(xs, ys)[0, 1]
            ax.set_title(f"{pair_title(pair)}\n Pearson r={r:.3f}", fontsize=9)
        else:
            ax.set_title(pair_title(pair))
        ax.set_xlabel("implicit cell mean", fontsize=8)
        ax.set_ylabel("explicit cell mean", fontsize=8)
    fig.suptitle("Implicit vs explicit cell means — 25 cells per pair", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "implicit_vs_explicit_scatter.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT/'implicit_vs_explicit_scatter.png'}")


def logit_diff_vs_z_colored_by_x(trials_by_id):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pair in zip(axes.ravel(), PAIRS):
        rows = list(merge(pair, "implicit", trials_by_id))
        xs = np.array([r["x"] for r in rows])
        zs = np.array([r["z"] for r in rows])
        ld = np.array([r["logit_diff"] for r in rows])
        sc = ax.scatter(zs, ld, c=xs, cmap="viridis", s=8, alpha=0.35)
        # Cell-mean overlay
        cells = defaultdict(list)
        for r in rows:
            cells[(r["x"], r["mu"])].append((r["z"], r["logit_diff"]))
        for (x, mu), pts in cells.items():
            z_bar = np.mean([p[0] for p in pts])
            ld_bar = np.mean([p[1] for p in pts])
            ax.scatter([z_bar], [ld_bar], c="red", s=30, edgecolors="black", linewidths=0.5, zorder=3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("z", fontsize=8)
        ax.set_ylabel("logit_diff", fontsize=8)
        ax.set_title(pair_title(pair), fontsize=9)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03, label="x")
    fig.suptitle("logit_diff vs z, colored by x — implicit prompts (red = cell means)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "logit_diff_vs_z_8panel.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT/'logit_diff_vs_z_8panel.png'}")


def zero_shot_per_pair(trials_by_id):
    fig, ax = plt.subplots(figsize=(10, 6))
    for pair in PAIRS:
        rows = list(merge(pair, "zero_shot", trials_by_id))
        if not rows:
            continue
        rows.sort(key=lambda r: r["x"])
        xs = [r["x"] for r in rows]
        # normalize x to unit scale for overlay
        x_min, x_max = min(xs), max(xs)
        x_norm = [(x - x_min) / (x_max - x_min) if x_max > x_min else 0.5 for x in xs]
        ld = [r["logit_diff"] for r in rows]
        ax.plot(x_norm, ld, marker="o", label=f"{pair_title(pair)}  (x∈[{x_min:g},{x_max:g}])")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("x (normalized to [0,1])")
    ax.set_ylabel("zero-shot logit_diff(high − low)")
    ax.set_title("Zero-shot prior: logit_diff by x for each pair (5 points each)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "zero_shot_bias_per_pair.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT/'zero_shot_bias_per_pair.png'}")


def main():
    trials = load_trials()
    print(f"{len(trials)} trials loaded")
    eight_panel_heatmap(trials, y_key="mu", fname="logit_diff_heatmap_xmu_8panel.png")
    eight_panel_heatmap(trials, y_key="z", fname="logit_diff_heatmap_xz_8panel.png")
    implicit_vs_explicit_scatter(trials)
    logit_diff_vs_z_colored_by_x(trials)
    zero_shot_per_pair(trials)
    print("done")


if __name__ == "__main__":
    main()
