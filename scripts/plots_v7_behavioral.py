"""Regenerate all behavioral plots from v7 (x, z) grid data.

Replaces the v4/v5 behavioral plots that used the confounded (x, μ) grid.

Reads:
  results/v7_xz_grid/e4b_trials.jsonl
  results/v7_xz_grid/e4b_{pair}_logits.jsonl

Writes into figures/v7_behavioral/:
  logit_diff_heatmap_xz_8panel.png     — 8-panel (x, z) heatmaps (hero figure)
  logit_diff_heatmap_xmu_8panel.png    — 8-panel (x, derived-μ) heatmaps
  logit_diff_vs_z_8panel.png           — logit_diff vs z, colored by x
  context_effect_per_pair.png          — zero-shot bias vs context effect per x
  relativity_summary.png               — relativity ratio R per pair + probe R²
  zero_shot_bias_per_pair.png          — zero-shot logit_diff by x (unchanged)

Run:
  python scripts/plots_v7_behavioral.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
V7 = ROOT / "results" / "v7_xz_grid"
# Also check for v4 zero-shot data (zero-shot is context-free, grid doesn't matter)
V4 = ROOT / "results" / "v4_adjpairs"
OUT = ROOT / "figures" / "v7_behavioral"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = ["height", "age", "experience", "size", "speed", "wealth", "weight", "bmi_abs"]

PAIR_TITLES = {
    "height": "tall / short (cm)",
    "age": "old / young (yrs)",
    "experience": "experienced / novice (yrs)",
    "size": "big / small (cm)",
    "speed": "fast / slow (km/h)",
    "wealth": "rich / poor ($k)",
    "weight": "heavy / light (kg)",
    "bmi_abs": "obese / thin (BMI)",
}


def load_v7_trials() -> dict[str, dict]:
    """Load v7 trials, keyed by id."""
    by_id = {}
    for line in (V7 / "e4b_trials.jsonl").open():
        t = json.loads(line)
        by_id[t["id"]] = t
    return by_id


def load_v7_logits(pair: str) -> list[dict]:
    path = V7 / f"e4b_{pair}_logits.jsonl"
    return [json.loads(line) for line in path.open()]


def merge_v7(pair: str, trials_by_id: dict) -> list[dict]:
    """Merge trial metadata with logit_diff."""
    rows = []
    for rec in load_v7_logits(pair):
        t = trials_by_id[rec["id"]]
        rows.append({**t, "logit_diff": rec["logit_diff"], "entropy": rec["entropy"]})
    return rows


# ---------------------------------------------------------------------------
# Plot 1: (x, z) heatmap — the hero figure
# ---------------------------------------------------------------------------
def heatmap_panel(ax, rows: list[dict], x_key: str, y_key: str, title: str):
    xs = sorted({r[x_key] for r in rows})
    ys = sorted({r[y_key] for r in rows})
    counts = np.zeros((len(ys), len(xs)))
    sums = np.zeros((len(ys), len(xs)))
    for r in rows:
        i = ys.index(r[y_key])
        j = xs.index(r[x_key])
        if not np.isnan(r["logit_diff"]):
            sums[i, j] += r["logit_diff"]
            counts[i, j] += 1
    M = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    vmax = max(abs(np.nanmin(M)), abs(np.nanmax(M)), 1e-6)
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Annotate cells with values
    for i in range(len(ys)):
        for j in range(len(xs)):
            if not np.isnan(M[i, j]):
                color = "white" if abs(M[i, j]) > 0.6 * vmax else "black"
                ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:g}" for y in ys], fontsize=7)
    ax.set_xlabel(x_key, fontsize=8)
    ax.set_ylabel(y_key, fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def plot_heatmap_8panel(trials_by_id: dict, y_key: str, fname: str, suptitle: str):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pair in zip(axes.ravel(), PAIRS):
        rows = merge_v7(pair, trials_by_id)
        im = heatmap_panel(ax, rows, "x", y_key, PAIR_TITLES[pair])
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / fname}")


# ---------------------------------------------------------------------------
# Plot 2: logit_diff vs z, colored by x
# ---------------------------------------------------------------------------
def plot_logit_diff_vs_z(trials_by_id: dict):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pair in zip(axes.ravel(), PAIRS):
        rows = merge_v7(pair, trials_by_id)
        xs_arr = np.array([r["x"] for r in rows])
        zs_arr = np.array([r["z"] for r in rows])
        ld_arr = np.array([r["logit_diff"] for r in rows])

        sc = ax.scatter(zs_arr, ld_arr, c=xs_arr, cmap="viridis", s=8, alpha=0.35)

        # Cell-mean overlay
        cells: dict[tuple[float, float], list[float]] = defaultdict(list)
        for r in rows:
            cells[(r["x"], r["z"])].append(r["logit_diff"])
        for (x, z), lds in sorted(cells.items()):
            ax.scatter([z], [np.mean(lds)], c="red", s=30,
                       edgecolors="black", linewidths=0.5, zorder=3)

        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("z", fontsize=8)
        ax.set_ylabel("logit_diff", fontsize=8)
        ax.set_title(PAIR_TITLES[pair], fontsize=9)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03, label="x")

    fig.suptitle(
        "logit_diff vs z, colored by x — v7 Grid B (red = cell means)", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "logit_diff_vs_z_8panel.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'logit_diff_vs_z_8panel.png'}")


# ---------------------------------------------------------------------------
# Plot 3: context effect per pair — R from regression ld ~ b*x + c*mu
# ---------------------------------------------------------------------------
def compute_relativity_ratio(rows: list[dict]) -> tuple[float, float, float]:
    """Fit ld ~ b*x + c*mu, return R = -c/b, slope_x, slope_mu."""
    X = np.array([[r["x"], r["mu"]] for r in rows])
    y = np.array([r["logit_diff"] for r in rows])
    # Standardize for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std
    X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    # Convert back to original scale
    slope_x = beta[0] / X_std[0]
    slope_mu = beta[1] / X_std[1]
    R = -slope_mu / slope_x if abs(slope_x) > 1e-12 else np.nan
    return R, slope_x, slope_mu


def plot_context_effect(trials_by_id: dict):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, pair in zip(axes.ravel(), PAIRS):
        rows = merge_v7(pair, trials_by_id)
        R, slope_x, slope_mu = compute_relativity_ratio(rows)

        # Per-x: mean logit_diff
        x_vals = sorted({r["x"] for r in rows})
        by_x = defaultdict(list)
        for r in rows:
            by_x[r["x"]].append(r["logit_diff"])
        overall_mean = np.mean([r["logit_diff"] for r in rows])
        zero_shot_ld = [np.mean(by_x[x]) - overall_mean for x in x_vals]  # x effect
        context_ld = [overall_mean - np.mean(by_x[x]) for x in x_vals]  # rough context

        # Better: for each x, compute spread across z
        context_effects = []
        for x in x_vals:
            z_means = defaultdict(list)
            for r in rows:
                if r["x"] == x:
                    z_means[r["z"]].append(r["logit_diff"])
            z_avgs = [np.mean(v) for v in z_means.values()]
            context_effects.append(max(z_avgs) - min(z_avgs) if len(z_avgs) > 1 else 0)

        # Just plot mean ld per x as "x effect" and spread across z as "context effect"
        x_means = [np.mean(by_x[x]) for x in x_vals]
        bar_w = 0.35
        idx = np.arange(len(x_vals))
        ax.bar(idx - bar_w / 2, x_means, bar_w, label="mean ld", color="tab:orange", alpha=0.8)
        ax.bar(idx + bar_w / 2, context_effects, bar_w, label="z spread", color="tab:blue", alpha=0.8)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{x:g}" for x in x_vals], fontsize=7)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3, ls="--")
        ax.set_title(f"{PAIR_TITLES[pair]} (R={R:.2f})", fontsize=9)
        ax.set_ylabel("logit_diff", fontsize=8)
        if pair == PAIRS[0]:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Mean logit_diff per x vs context spread across z — v7 Grid B", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "context_effect_per_pair.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'context_effect_per_pair.png'}")


# ---------------------------------------------------------------------------
# Plot 4: relativity ratio summary (replaces eight_pair_summary + relativity_across_pairs)
# ---------------------------------------------------------------------------
def plot_relativity_summary(trials_by_id: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Rs = []
    for pair in PAIRS:
        rows = merge_v7(pair, trials_by_id)
        R, _, _ = compute_relativity_ratio(rows)
        Rs.append(R)

    # Left: relativity ratio
    colors = ["tab:blue"] * len(PAIRS)
    colors[PAIRS.index("bmi_abs")] = "tab:red"  # absolute control
    ax1.bar(range(len(PAIRS)), Rs, color=colors)
    ax1.set_xticks(range(len(PAIRS)))
    ax1.set_xticklabels(PAIRS, rotation=30, ha="right", fontsize=9)
    ax1.axhline(1.0, color="red", ls="--", alpha=0.5, label="pure relativity")
    ax1.axhline(0.0, color="gray", ls="--", alpha=0.5, label="pure absolute")
    ax1.set_ylabel("Relativity ratio R = −slope_μ / slope_x", fontsize=10)
    ax1.set_title("Behavioral relativity — v7 Grid B", fontsize=11)
    ax1.legend(fontsize=8)
    for i, r in enumerate(Rs):
        ax1.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8, fontweight="bold")

    # Right: INLP R² from v7 analysis (if available)
    inlp_path = ROOT / "results" / "v7_analysis" / "inlp_clean.json"
    if inlp_path.exists():
        inlp = json.loads(inlp_path.read_text())
        r2_init = [inlp[p]["r2_init"] for p in PAIRS]
        r2_final = [inlp[p]["r2_final_inlp"] for p in PAIRS]
        r2_rand = [inlp[p]["r2_final_rand"] for p in PAIRS]
        idx = np.arange(len(PAIRS))
        w = 0.25
        ax2.bar(idx - w, r2_init, w, label="R²(z) initial", color="tab:blue")
        ax2.bar(idx, r2_final, w, label="R²(z) after 8 INLP", color="tab:cyan")
        ax2.bar(idx + w, r2_rand, w, label="R²(z) random null", color="tab:orange")
        ax2.set_xticks(idx)
        ax2.set_xticklabels(PAIRS, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("CV R²(z) from activations", fontsize=10)
        ax2.set_title("INLP: z is genuinely erasable — v7 Grid B", fontsize=11)
        ax2.legend(fontsize=7, loc="lower left")
        ax2.set_ylim(0.4, 1.02)
    else:
        ax2.text(0.5, 0.5, "INLP data not available\n(run exp_v7_inlp_clean.py)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT / "relativity_summary.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'relativity_summary.png'}")


# ---------------------------------------------------------------------------
# Plot 5: zero-shot bias per pair (context-free, can use v4 data)
# ---------------------------------------------------------------------------
def plot_zero_shot(trials_by_id: dict):
    """Zero-shot uses v4 data since it's context-free (no grid dependence)."""
    # Try v4 first (has zero_shot condition)
    v4_trials = {}
    v4_path = V4 / "e4b_trials.jsonl"
    if v4_path.exists():
        for line in v4_path.open():
            t = json.loads(line)
            v4_trials[t["id"]] = t
    else:
        print("  SKIP zero_shot_bias_per_pair.png (v4 data not available)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for pair in PAIRS:
        logit_path = V4 / f"e4b_{pair}_zero_shot_logits.jsonl"
        if not logit_path.exists():
            continue
        rows = []
        for line in logit_path.open():
            rec = json.loads(line)
            t = v4_trials.get(rec["id"])
            if t:
                rows.append({**t, "logit_diff": rec["logit_diff"]})
        if not rows:
            continue
        rows.sort(key=lambda r: r["x"])
        xs = [r["x"] for r in rows]
        x_min, x_max = min(xs), max(xs)
        x_norm = [(x - x_min) / (x_max - x_min) if x_max > x_min else 0.5 for x in xs]
        ld = [r["logit_diff"] for r in rows]
        ax.plot(x_norm, ld, marker="o",
                label=f"{PAIR_TITLES[pair]}  (x∈[{x_min:g},{x_max:g}])")

    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("x (normalized to [0,1])")
    ax.set_ylabel("zero-shot logit_diff(high − low)")
    ax.set_title("Zero-shot prior: logit_diff by x (5 points each, no context)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "zero_shot_bias_per_pair.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'zero_shot_bias_per_pair.png'}")


# ---------------------------------------------------------------------------
# Plot 6: Zero-shot-corrected heatmaps — isolate context effect
# ---------------------------------------------------------------------------
def load_zeroshot_baseline() -> dict[str, dict[float, float]]:
    """Load mean zero-shot ld per (pair, x) from CSV."""
    import csv
    csv_path = ROOT / "results" / "csv" / "v4_zeroshot_expanded_e4b.csv"
    if not csv_path.exists():
        return {}
    baseline: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            baseline[row["pair"]][float(row["x"])].append(float(row["logit_diff"]))
    return {p: {x: np.mean(vs) for x, vs in xd.items()} for p, xd in baseline.items()}


def plot_zeroshot_corrected_heatmaps(trials_by_id: dict):
    zs_baseline = load_zeroshot_baseline()
    if not zs_baseline:
        print("  SKIP zeroshot-corrected heatmaps (CSV not available)")
        return

    for y_key, fname, suptitle in [
        ("z", "logit_diff_corrected_xz_8panel.png",
         "Context effect: ld − zero-shot(x) over (z, x) — v7 Grid B"),
        ("mu", "logit_diff_corrected_xmu_8panel.png",
         "Context effect: ld − zero-shot(x) over (μ, x) — v7 Grid B"),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(17, 8))
        for ax, pair in zip(axes.ravel(), PAIRS):
            rows = merge_v7(pair, trials_by_id)
            bl = zs_baseline.get(pair, {})
            corrected = []
            for r in rows:
                zs_ld = bl.get(r["x"], 0.0)
                corrected.append({**r, "logit_diff": r["logit_diff"] - zs_ld})
            im = heatmap_panel(ax, corrected, "x", y_key, PAIR_TITLES[pair])
            plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / fname, dpi=140)
        plt.close(fig)
        print(f"  wrote {OUT / fname}")


def main():
    # Check v7 data exists
    trials_path = V7 / "e4b_trials.jsonl"
    if not trials_path.exists():
        print(f"ERROR: v7 trials not found at {trials_path}")
        print("Run:  python scripts/fetch_from_hf.py --only v7_xz_grid --data-kind jsonl")
        return

    trials = load_v7_trials()
    print(f"Loaded {len(trials)} v7 trials")

    # Check per-pair logit files exist
    missing = [p for p in PAIRS if not (V7 / f"e4b_{p}_logits.jsonl").exists()]
    if missing:
        print(f"ERROR: missing logit files for {missing}")
        print("Run:  python scripts/fetch_from_hf.py --only v7_xz_grid --data-kind jsonl")
        return

    # Generate all plots
    plot_heatmap_8panel(
        trials, y_key="z",
        fname="logit_diff_heatmap_xz_8panel.png",
        suptitle="logit_diff(high−low) over (z, x) — v7 Grid B, E4B layer late",
    )
    plot_heatmap_8panel(
        trials, y_key="mu",
        fname="logit_diff_heatmap_xmu_8panel.png",
        suptitle="logit_diff(high−low) over (μ, x) — v7 Grid B (μ derived), E4B layer late",
    )
    plot_logit_diff_vs_z(trials)
    plot_context_effect(trials)
    plot_relativity_summary(trials)
    plot_zero_shot(trials)
    plot_zeroshot_corrected_heatmaps(trials)
    print("done")


if __name__ == "__main__":
    main()
