#!/usr/bin/env python3
"""Per-pair PCA on cell means. For each of the 8 pairs:
   - Load implicit_late.npz activations
   - Attach (x, mu, z) per trial via generate_all_prompts()
   - Bin into (x, mu) cells, compute cell means
   - PCA(2), scatter coloured by z
   - Grid of 8 subplots (2 rows x 4 cols): height, age, weight, size, speed, wealth, experience, bmi_abs

Output: results/v4_adjpairs_analysis/figures/pca_per_pair_late.png
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

REPO = Path("/workspace/repo")
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import PAIRS, generate_all_prompts  # noqa: E402

trials = generate_all_prompts()
id2trial = {t["id"]: t for t in trials}

PAIRS_ORDER = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
name2pair = {p.name: p for p in PAIRS}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("PC1 vs PC2 of implicit/late cell means — per adjective pair (colour = z)",
             fontsize=14, y=0.995)

for ax, pname in zip(axes.ravel(), PAIRS_ORDER):
    pair = name2pair[pname]
    npz_path = REPO / "results" / "v4_adjpairs" / f"e4b_{pname}_implicit_late.npz"
    if not npz_path.exists():
        ax.set_title(f"{pname}: missing")
        ax.axis("off")
        continue
    d = np.load(npz_path)
    acts, ids = d["activations"], d["ids"]

    xs = np.array([id2trial[i]["x"]  for i in ids])
    mus = np.array([id2trial[i]["mu"] for i in ids])
    zs = np.array([id2trial[i]["z"]  for i in ids])

    # Cell means over (x, mu)
    x_vals = sorted(set(xs.tolist()))
    mu_vals = sorted(set(mus.tolist()))
    cell_acts, cell_x, cell_mu, cell_z = [], [], [], []
    for x in x_vals:
        for mu in mu_vals:
            m = (xs == x) & (mus == mu)
            if m.sum() == 0:
                continue
            cell_acts.append(acts[m].mean(axis=0))
            cell_x.append(x); cell_mu.append(mu); cell_z.append(zs[m].mean())
    cell_acts = np.stack(cell_acts)
    cell_x = np.array(cell_x); cell_mu = np.array(cell_mu); cell_z = np.array(cell_z)

    # PCA(2) on the centred cell means
    pca = PCA(n_components=3)
    proj = pca.fit_transform(cell_acts - cell_acts.mean(0))
    pc1, pc2 = proj[:, 0], proj[:, 1]
    ev = pca.explained_variance_ratio_ * 100

    # Fit quadratic PC2 ≈ a * z^2 + b * z + c  and report R²
    A = np.column_stack([cell_z ** 2, cell_z, np.ones_like(cell_z)])
    coef, _, _, _ = np.linalg.lstsq(A, pc2, rcond=None)
    pred = A @ coef
    ss_res = ((pc2 - pred) ** 2).sum()
    ss_tot = ((pc2 - pc2.mean()) ** 2).sum()
    r2_q = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    sc = ax.scatter(pc1, pc2, c=cell_z, cmap="RdBu_r", edgecolor="k",
                    linewidth=0.3, s=60, vmin=-3.3, vmax=3.3)
    ax.set_title(f"{pname}   PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  R²(PC2~z²)={r2_q:.2f}",
                 fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.axhline(0, color="gray", lw=0.3); ax.axvline(0, color="gray", lw=0.3)

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
cbar.set_label("z = (x − μ)/σ")

out_dir = REPO / "results" / "v4_adjpairs_analysis" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "pca_per_pair_late.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Saved {out_path}")

# Also save a tiny JSON summary
summary = {}
for pname in PAIRS_ORDER:
    pair = name2pair[pname]
    npz_path = REPO / "results" / "v4_adjpairs" / f"e4b_{pname}_implicit_late.npz"
    if not npz_path.exists():
        summary[pname] = None; continue
    d = np.load(npz_path)
    acts, ids = d["activations"], d["ids"]
    xs  = np.array([id2trial[i]["x"]  for i in ids])
    mus = np.array([id2trial[i]["mu"] for i in ids])
    zs  = np.array([id2trial[i]["z"]  for i in ids])
    x_vals = sorted(set(xs.tolist())); mu_vals = sorted(set(mus.tolist()))
    cell_acts, cell_z = [], []
    for x in x_vals:
        for mu in mu_vals:
            m = (xs == x) & (mus == mu)
            if m.sum() == 0: continue
            cell_acts.append(acts[m].mean(0)); cell_z.append(zs[m].mean())
    cell_acts = np.stack(cell_acts); cell_z = np.array(cell_z)
    pca = PCA(n_components=3)
    proj = pca.fit_transform(cell_acts - cell_acts.mean(0))
    ev = (pca.explained_variance_ratio_ * 100).tolist()
    A = np.column_stack([cell_z**2, cell_z, np.ones_like(cell_z)])
    coef, _, _, _ = np.linalg.lstsq(A, proj[:, 1], rcond=None)
    pred = A @ coef
    ss_res = ((proj[:, 1] - pred) ** 2).sum()
    ss_tot = ((proj[:, 1] - proj[:, 1].mean()) ** 2).sum()
    r2_q = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    summary[pname] = {"var_PC1_pct": ev[0], "var_PC2_pct": ev[1], "var_PC3_pct": ev[2],
                      "R2_PC2_vs_z2": r2_q, "quad_coef_z2": float(coef[0]),
                      "quad_coef_z":  float(coef[1]), "n_cells": int(len(cell_z))}

with open(out_dir / "pca_per_pair_late.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {out_dir/'pca_per_pair_late.json'}")
for k, v in summary.items():
    if v: print(f"  {k}: PC1={v['var_PC1_pct']:.1f}%  PC2={v['var_PC2_pct']:.1f}%  "
                f"R²(PC2~z²)={v['R2_PC2_vs_z2']:.3f}  n={v['n_cells']}")
