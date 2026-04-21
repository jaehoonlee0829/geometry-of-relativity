#!/usr/bin/env python3
"""PC3 vs raw x + cross-pair PC1 cosine similarity (late layer).

For each of the 8 pairs:
  - Load implicit_late.npz
  - Bin into (x, mu) cells, compute cell means
  - PCA(5) on cell means
  - R² of each PC regressed on x, mu, z, z², (x, mu)

Cross-pair: stack PC1 directions, compute 8x8 |cos| similarity.

Outputs:
  - results/v4_adjpairs_analysis/figures/pc_correlations_and_shared_z.png
  - results/v4_adjpairs_analysis/pc_correlations_late.json
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

def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else None

def lstsq_r2(X, y):
    Xi = np.column_stack([X, np.ones(len(y))])
    coef, _, _, _ = np.linalg.lstsq(Xi, y, rcond=None)
    return r2(y, Xi @ coef)

records = {}
pc1_vecs = {}

for pname in PAIRS_ORDER:
    npz_path = REPO / "results" / "v4_adjpairs" / f"e4b_{pname}_implicit_late.npz"
    if not npz_path.exists():
        records[pname] = None
        continue
    d = np.load(npz_path)
    acts, ids = d["activations"], d["ids"]
    xs  = np.array([id2trial[i]["x"]  for i in ids])
    mus = np.array([id2trial[i]["mu"] for i in ids])
    zs  = np.array([id2trial[i]["z"]  for i in ids])

    x_vals  = sorted(set(xs.tolist()))
    mu_vals = sorted(set(mus.tolist()))
    cell_acts, cx, cm, cz = [], [], [], []
    for x in x_vals:
        for mu in mu_vals:
            m = (xs == x) & (mus == mu)
            if m.sum() == 0:
                continue
            cell_acts.append(acts[m].mean(0))
            cx.append(x); cm.append(mu); cz.append(zs[m].mean())
    cell_acts = np.stack(cell_acts)
    cx = np.array(cx, float); cm = np.array(cm, float); cz = np.array(cz, float)

    pca = PCA(n_components=5)
    proj = pca.fit_transform(cell_acts - cell_acts.mean(0))
    ev = pca.explained_variance_ratio_ * 100
    pc1_vecs[pname] = pca.components_[0]

    pc_r2 = {}
    for k in range(5):
        pck = proj[:, k]
        pc_r2[f"PC{k+1}"] = {
            "var_pct": float(ev[k]),
            "R2_vs_x":    lstsq_r2(cx.reshape(-1, 1), pck),
            "R2_vs_mu":   lstsq_r2(cm.reshape(-1, 1), pck),
            "R2_vs_z":    lstsq_r2(cz.reshape(-1, 1), pck),
            "R2_vs_z2":   lstsq_r2((cz ** 2).reshape(-1, 1), pck),
            "R2_vs_x_mu": lstsq_r2(np.column_stack([cx, cm]), pck),
        }
    records[pname] = {"n_cells": int(len(cz)), "pcs": pc_r2}

# Cross-pair PC1 cosine similarity
names = [n for n in PAIRS_ORDER if pc1_vecs.get(n) is not None]
V = np.stack([pc1_vecs[n] for n in names])
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
cos = Vn @ Vn.T

# --------------------- Figure ---------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.5))

im = ax1.imshow(np.abs(cos), cmap="viridis", vmin=0, vmax=1)
ax1.set_xticks(range(len(names))); ax1.set_yticks(range(len(names)))
ax1.set_xticklabels(names, rotation=45, ha="right"); ax1.set_yticklabels(names)
ax1.set_title("|cos(PC1_i, PC1_j)|  —  cross-pair similarity of z-direction\n"
              "(high = shared z-subspace; low = concept-specific)")
for i in range(len(names)):
    for j in range(len(names)):
        c = "w" if abs(cos[i, j]) < 0.6 else "k"
        ax1.text(j, i, f"{abs(cos[i, j]):.2f}", ha="center", va="center", color=c, fontsize=8)
fig.colorbar(im, ax=ax1, shrink=0.8)

bars_z, bars_z2, bars_x3, bars_mu3 = [], [], [], []
for pname in names:
    r = records[pname]["pcs"]
    bars_z.append(r["PC1"]["R2_vs_z"])
    bars_z2.append(r["PC2"]["R2_vs_z2"])
    bars_x3.append(r["PC3"]["R2_vs_x"])
    bars_mu3.append(r["PC3"]["R2_vs_mu"])
xs = np.arange(len(names)); w = 0.2
ax2.bar(xs - 1.5 * w, bars_z,   w, label="R²(PC1 ~ z)",  color="#1f77b4")
ax2.bar(xs - 0.5 * w, bars_z2,  w, label="R²(PC2 ~ z²)", color="#ff7f0e")
ax2.bar(xs + 0.5 * w, bars_x3,  w, label="R²(PC3 ~ x)",  color="#2ca02c")
ax2.bar(xs + 1.5 * w, bars_mu3, w, label="R²(PC3 ~ μ)",  color="#d62728")
ax2.set_xticks(xs); ax2.set_xticklabels(names, rotation=45, ha="right")
ax2.set_ylabel("R²"); ax2.set_ylim(0, 1.05); ax2.legend(loc="upper right", fontsize=9)
ax2.set_title("What does each PC encode?  (late layer, cell means)")
ax2.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
out_dir = REPO / "results" / "v4_adjpairs_analysis" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "pc_correlations_and_shared_z.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Saved {out_path}")

mask = ~np.eye(len(names), dtype=bool)
summary = {
    "cosine_matrix": {
        "rows": names,
        "matrix": cos.tolist(),
        "abs_mean_offdiag": float(np.abs(cos)[mask].mean()),
    },
    "per_pair": records,
}
with open(out_dir / "pc_correlations_late.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {out_dir/'pc_correlations_late.json'}")

print("\n=== Per-pair: R² of each PC vs target (late layer) ===")
print(f"{'pair':<12} {'PC':<4} {'var%':>6} {'~z':>7} {'~z²':>7} {'~x':>7} {'~μ':>7} {'~x+μ':>7}")
for pname in names:
    for k in range(5):
        r = records[pname]["pcs"][f"PC{k+1}"]
        def fmt(v): return f"{v:.3f}" if v is not None else "  -  "
        print(f"{pname:<12} PC{k+1:<3} {r['var_pct']:>5.1f}% {fmt(r['R2_vs_z']):>7} {fmt(r['R2_vs_z2']):>7} "
              f"{fmt(r['R2_vs_x']):>7} {fmt(r['R2_vs_mu']):>7} {fmt(r['R2_vs_x_mu']):>7}")

print(f"\nMean |cos(PC1_i, PC1_j)| off-diagonal = {summary['cosine_matrix']['abs_mean_offdiag']:.3f}")
