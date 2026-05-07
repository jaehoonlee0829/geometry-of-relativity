#!/usr/bin/env python3
"""Human-readable V14 story plots from generated GPU rows.

These plots are interpretation aids. They do not rerun the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "v14"
FIGS = REPO / "figures" / "v14"
PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def sem(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.std(xs, ddof=1) / np.sqrt(len(xs)))


def binned_line(rows: list[dict], condition_key: str, conditions: list[str], title: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True)
    axes = axes.ravel()
    for ax, pair in zip(axes, PAIRS):
        pair_rows = [r for r in rows if r["pair"] == pair and r.get("n_context", 31) == 31]
        for cond in conditions:
            cond_rows = [r for r in pair_rows if r.get(condition_key) == cond and -3.1 <= float(r["z"]) <= 3.1]
            zs = sorted({round(float(r["z"]), 2) for r in cond_rows})
            xs, ys = [], []
            for z in zs:
                vals = [float(r["ld"]) for r in cond_rows if round(float(r["z"]), 2) == z]
                if len(vals) >= 2:
                    xs.append(z)
                    ys.append(mean(vals))
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, lw=1.4, label=cond)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(pair)
        ax.set_xlabel("population z")
        ax.set_ylabel("LD(high-low)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_affine_conflict(rows: list[dict]) -> None:
    out_dir = FIGS / "affine_ood"
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = ["base", "parallel_shift_high", "world_extreme_high", "world_extreme_low"]

    binned_line(
        rows,
        "ood_condition",
        conditions,
        "Affine/OOD: LD still tracks z in shifted worlds",
        out_dir / "affine_ood_ld_by_z_lines.png",
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(PAIRS))
    width = 0.22
    bars = [
        ("base", -width),
        ("world_extreme_low", 0.0),
        ("world_extreme_high", width),
    ]
    for cond, offset in bars:
        vals = []
        errs = []
        ns = []
        for pair in PAIRS:
            subset = [
                float(r["ld"])
                for r in rows
                if r["pair"] == pair
                and r.get("n_context") == 31
                and r.get("ood_condition") == cond
                and float(r["z"]) > 1.0
            ]
            vals.append(mean(subset))
            errs.append(sem(subset))
            ns.append(len(subset))
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=2, label=cond)
        for xi, val, n in zip(x + offset, vals, ns):
            if n == 0:
                ax.text(xi, 0.05, "missing", rotation=90, ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(x, PAIRS, rotation=30, ha="right")
    ax.set_ylabel("mean LD(high-low), z > 1")
    ax.set_title("Conflict test: absolutely strange worlds, relatively high targets")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "affine_ood_conflict_z_high.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = [
        ("base", -0.27),
        ("target_extreme_low", 0.0),
        ("target_extreme_high", 0.27),
    ]
    for cond, offset in bars:
        vals = []
        errs = []
        for pair in PAIRS:
            if cond == "base":
                subset = [
                    float(r["ld"])
                    for r in rows
                    if r["pair"] == pair
                    and r.get("n_context") == 31
                    and r.get("ood_condition") == cond
                    and abs(float(r["z"])) < 0.25
                ]
            else:
                subset = [
                    float(r["ld"])
                    for r in rows
                    if r["pair"] == pair and r.get("n_context") == 31 and r.get("ood_condition") == cond
                ]
            vals.append(mean(subset))
            errs.append(sem(subset))
        ax.bar(x + offset, vals, width=0.27, yerr=errs, capsize=2, label=cond)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(x, PAIRS, rotation=30, ha="right")
    ax.set_ylabel("mean LD(high-low)")
    ax.set_title("Target-only extremes: constant-z cases should be read as LD shifts")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "affine_ood_target_extremes_mean_ld.png", dpi=170)
    plt.close(fig)

    counts = np.zeros((len(PAIRS), len(conditions)))
    for i, pair in enumerate(PAIRS):
        expected = max(
            len(
                [
                    r
                    for r in rows
                    if r["pair"] == pair and r.get("n_context") == 31 and r.get("ood_condition") == cond
                ]
            )
            for cond in conditions
        )
        for j, cond in enumerate(conditions):
            n = len(
                [
                    r
                    for r in rows
                    if r["pair"] == pair and r.get("n_context") == 31 and r.get("ood_condition") == cond
                ]
            )
            counts[i, j] = n / expected if expected else np.nan
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(counts, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(conditions)), conditions, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(PAIRS)), PAIRS)
    ax.set_title("Affine/OOD valid-row fraction at N=31")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if np.isfinite(counts[i, j]):
                ax.text(j, i, f"{counts[i, j]:.2f}", ha="center", va="center", color="white" if counts[i, j] < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "affine_ood_valid_row_fraction.png", dpi=170)
    plt.close(fig)


def plot_distribution_story(rows: list[dict]) -> None:
    out_dir = FIGS / "distribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = ["normal", "uniform", "beta_u", "beta_low", "beta_high", "bimodal"]
    binned_line(
        rows,
        "dist_kind",
        conditions,
        "Distribution controls: LD-z relationship across context shapes",
        out_dir / "distribution_ld_by_z_lines.png",
    )

    metrics = json.loads((RESULTS / "distribution" / "distribution_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    mins, maxs = [], []
    for pair in PAIRS:
        vals = [
            v["corr_ld_z"]
            for v in metrics["metrics_by_pair_distribution"][pair].values()
            if isinstance(v.get("corr_ld_z"), (int, float))
        ]
        mins.append(min(vals))
        maxs.append(max(vals))
    ax.vlines(np.arange(len(PAIRS)), mins, maxs, color="#4C78A8", lw=5)
    ax.scatter(np.arange(len(PAIRS)), mins, color="#1F4E79", s=28, label="min across shapes")
    ax.scatter(np.arange(len(PAIRS)), maxs, color="#F58518", s=28, label="max across shapes")
    ax.set_xticks(np.arange(len(PAIRS)), PAIRS, rotation=30, ha="right")
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("corr(LD,z)")
    ax.set_title("Distribution robustness: range across normal/uniform/beta/bimodal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "distribution_corr_range_by_pair.png", dpi=170)
    plt.close(fig)


def plot_order_story(rows: list[dict]) -> None:
    out_dir = FIGS / "order"
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = ["random", "ascending", "descending", "alternating_low_high", "near_target_first", "near_target_last"]
    binned_line(
        rows,
        "order_kind",
        conditions,
        "Order controls: same values, different sequence order",
        out_dir / "order_ld_by_z_lines.png",
    )

    metrics = json.loads((RESULTS / "order" / "order_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    random_vals, worst_vals = [], []
    for pair in PAIRS:
        by_cond = metrics["metrics_by_pair_order"][pair]
        random_vals.append(by_cond["random"]["corr_ld_z"])
        worst_vals.append(min(v["corr_ld_z"] for v in by_cond.values() if isinstance(v.get("corr_ld_z"), (int, float))))
    x = np.arange(len(PAIRS))
    ax.bar(x - 0.18, random_vals, width=0.36, label="random order")
    ax.bar(x + 0.18, worst_vals, width=0.36, label="worst ordered condition")
    ax.set_xticks(x, PAIRS, rotation=30, ha="right")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("corr(LD,z)")
    ax.set_title("Order sensitivity: robust but not invariant")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "order_random_vs_worst_corr.png", dpi=170)
    plt.close(fig)


def main() -> None:
    affine = load_jsonl(RESULTS / "affine_ood" / "affine_ood_rows.jsonl")
    distribution = load_jsonl(RESULTS / "distribution" / "distribution_rows.jsonl")
    order = load_jsonl(RESULTS / "order" / "order_rows.jsonl")
    plot_affine_conflict(affine)
    plot_distribution_story(distribution)
    plot_order_story(order)
    print("Wrote V14 story plots under figures/v14/")


if __name__ == "__main__":
    main()
