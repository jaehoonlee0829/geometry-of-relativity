"""v10 behavioral plots — cell-mean logit_diff(tall - short) on the dense grid.

Inputs:  results/v10/gemma2_height_v10_residuals.npz
Outputs: figures/v10/behavioral_logit_diff_xz.png
         figures/v10/behavioral_x_vs_z_marginals.png
"""
from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "v10"
FIG = REPO / "figures" / "v10"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    res = np.load(RES / "gemma2_height_v10_residuals.npz")
    x = res["x"].astype(float)
    z = res["z"].astype(float)
    ld = res["next_logit_diff"].astype(float)
    ent = res["next_entropy"].astype(float)

    # Cell-means
    keys = sorted({(round(float(x[i]), 4), round(float(z[i]), 4))
                   for i in range(len(x))})
    cell_mean = {k: [] for k in keys}
    for i in range(len(x)):
        cell_mean[(round(float(x[i]), 4), round(float(z[i]), 4))].append(ld[i])
    means = {k: float(np.mean(v)) for k, v in cell_mean.items()}

    x_vals = sorted({k[0] for k in keys})
    z_vals = sorted({k[1] for k in keys})
    M = np.full((len(z_vals), len(x_vals)), np.nan)
    for (xv, zv), m in means.items():
        i = z_vals.index(zv)
        j = x_vals.index(xv)
        M[i, j] = m

    cell_zs = np.array([k[1] for k in keys])
    cell_xs = np.array([k[0] for k in keys])
    cell_means_arr = np.array([means[k] for k in keys])

    r_z = float(np.corrcoef(cell_means_arr, cell_zs)[0, 1])
    r_x = float(np.corrcoef(cell_means_arr, cell_xs)[0, 1])
    r_per_prompt = float(np.corrcoef(ld, z)[0, 1])

    # Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
                   extent=[min(x_vals), max(x_vals),
                           min(z_vals), max(z_vals)],
                   vmin=-max(abs(M.min()), abs(M.max())),
                   vmax=max(abs(M.min()), abs(M.max())))
    ax.set_xlabel("x (height in cm)")
    ax.set_ylabel("z = (x - μ) / σ")
    ax.set_title(f"v10 dense grid (height): cell-mean logit_diff(tall − short)\n"
                 f"corr(LD, z)={r_z:.3f}   corr(LD, x)={r_x:.3f}   "
                 f"per-prompt R(z)={r_per_prompt:.3f}")
    plt.colorbar(im, ax=ax, label="mean logit_diff")
    plt.tight_layout()
    plt.savefig(FIG / "behavioral_logit_diff_xz.png", dpi=120)
    plt.close()

    # Marginals: mean(ld) vs z and vs x
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    z_marg = {zv: [] for zv in z_vals}
    x_marg = {xv: [] for xv in x_vals}
    for (xv, zv), m in means.items():
        z_marg[zv].append(m)
        x_marg[xv].append(m)
    z_means = [float(np.mean(z_marg[zv])) for zv in z_vals]
    z_stds = [float(np.std(z_marg[zv])) for zv in z_vals]
    x_means = [float(np.mean(x_marg[xv])) for xv in x_vals]
    x_stds = [float(np.std(x_marg[xv])) for xv in x_vals]
    axes[0].errorbar(z_vals, z_means, yerr=z_stds, fmt="o-", capsize=3)
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_xlabel("z-score")
    axes[0].set_ylabel("mean logit_diff (over x)")
    axes[0].set_title(f"z-marginal — corr={r_z:.3f}")
    axes[0].grid(alpha=0.3)
    axes[1].errorbar(x_vals, x_means, yerr=x_stds, fmt="s-", color="C3", capsize=3)
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("mean logit_diff (over z)")
    axes[1].set_title(f"x-marginal — corr={r_x:.3f}")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "behavioral_x_vs_z_marginals.png", dpi=120)
    plt.close()

    summary = {
        "r_z_cellmean": r_z,
        "r_x_cellmean": r_x,
        "r_z_per_prompt": r_per_prompt,
        "n_prompts": int(len(x)),
        "n_cells": int(len(keys)),
        "entropy_mean": float(ent.mean()),
        "entropy_std": float(ent.std()),
        "logit_diff_min": float(ld.min()),
        "logit_diff_max": float(ld.max()),
    }
    (RES / "behavioral_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote 2 figures + behavioral_summary.json")


if __name__ == "__main__":
    main()
