"""v11 P3a/P3b — PCA scatter and PC1 vs primal_z cosine across layers.

For each pair × model:
  - cell-mean activation over (x, z) cells at each layer
  - PCA(n=3) at the canonical late layer (L20 for 2B, L33 for 9B)
  - PC1/PC2/PC3 R² vs (z, x, z²)
  - cos(PC1, primal_z) at every layer

Outputs:
  results/v11/<model_short>/<pair>/pca_summary.json
  figures/v11/pca/{pair}_{model_short}_2d_L<late>.png
  figures/v11/pca/{pair}_{model_short}_3d_L<late>.png
  figures/v11/probing/cos_pc1_primal_per_layer_{pair}_{model_short}.png
  results/v11/<model_short>/cos_pc1_primal_summary.json   (one per model, all pairs)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]


def cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def cell_mean(acts: np.ndarray, x: np.ndarray, z: np.ndarray):
    """Mean over (x, z) cells. Returns (n_cells, n_layers, d_model), cell_z, cell_x."""
    keys = np.array([f"{round(float(xi), 4)}_{round(float(zi), 4)}" for xi, zi in zip(x, z)])
    uniq, inv = np.unique(keys, return_inverse=True)
    n_cells = len(uniq)
    out = np.zeros((n_cells, acts.shape[1], acts.shape[2]), dtype=np.float64)
    cnt = np.zeros(n_cells, dtype=np.int32)
    for i in range(len(acts)):
        out[inv[i]] += acts[i].astype(np.float64)
        cnt[inv[i]] += 1
    out /= np.maximum(cnt, 1)[:, None, None]
    cz = np.array([float(k.split("_")[1]) for k in uniq])
    cx = np.array([float(k.split("_")[0]) for k in uniq])
    return out, cz, cx


def analyze_pair(model_short: str, pair: str) -> dict | None:
    res_path = (REPO / "results" / "v11" / model_short / pair /
                f"{model_short}_{pair}_v11_residuals.npz")
    if not res_path.exists():
        return None
    d = np.load(res_path)
    acts = d["activations"]
    z = d["z"]; x = d["x"]
    n_layers = acts.shape[1]
    late = 20 if model_short == "gemma2-2b" else 33

    cm, cz, cx = cell_mean(acts, x, z)

    # PCA at late layer
    pca = PCA(n_components=3).fit(cm[:, late, :])
    proj = pca.transform(cm[:, late, :])
    pc1, pc2, pc3 = proj[:, 0], proj[:, 1], proj[:, 2]

    def r2(u, v):
        if v.std() < 1e-12 or u.std() < 1e-12: return 0.0
        return float(np.corrcoef(u, v)[0, 1] ** 2)

    pca_r2 = {
        "PC1_vs_z": r2(pc1, cz), "PC1_vs_x": r2(pc1, cx),
        "PC2_vs_z": r2(pc2, cz), "PC2_vs_x": r2(pc2, cx),
        "PC2_vs_z2": r2(pc2, cz ** 2),
        "PC3_vs_z": r2(pc3, cz), "PC3_vs_x": r2(pc3, cx),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    # 2D scatter
    fig_dir = REPO / "figures" / "v11" / "pca"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(pc1, pc2, c=cz, cmap="RdBu_r", s=20)
    plt.xlabel(f"PC1 (R²(z)={pca_r2['PC1_vs_z']:.2f})")
    plt.ylabel(f"PC2 (R²(z²)={pca_r2['PC2_vs_z2']:.2f})")
    plt.title(f"{model_short} / {pair}  L{late}  cell-mean PCA")
    plt.colorbar(label="z")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{pair}_{model_short}_2d_L{late}.png", dpi=110)
    plt.close()

    # 3D scatter
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc1, pc2, pc3, c=cz, cmap="RdBu_r", s=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"{model_short}/{pair}  L{late}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{pair}_{model_short}_3d_L{late}.png", dpi=110)
    plt.close()

    # PC1 vs primal_z cosine across layers
    high_z = z > +1.0
    low_z = z < -1.0
    cos_per_layer = []
    for L in range(n_layers):
        h = acts[:, L, :].astype(np.float64)
        primal = h[high_z].mean(0) - h[low_z].mean(0)
        # PC1 of cell-means at layer L
        p = PCA(n_components=1).fit(cm[:, L, :])
        pc1_dir = p.components_[0]
        cos_per_layer.append(cos(pc1_dir, primal))

    fig_dir2 = REPO / "figures" / "v11" / "probing"
    fig_dir2.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(n_layers), cos_per_layer, marker="o")
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("layer")
    plt.ylabel("cos(PC1, primal_z)")
    plt.title(f"{model_short}/{pair}  PC1 vs primal_z")
    plt.tight_layout()
    plt.savefig(fig_dir2 / f"cos_pc1_primal_per_layer_{pair}_{model_short}.png", dpi=110)
    plt.close()

    return {
        "model_short": model_short,
        "pair": pair,
        "late_layer": late,
        "pca_r2": pca_r2,
        "cos_pc1_primal_per_layer": [float(c) for c in cos_per_layer],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--pair", required=True, help="pair or 'all'")
    args = ap.parse_args()

    pairs = ALL_PAIRS if args.pair == "all" else [args.pair]
    summary = {}
    for p in pairs:
        info = analyze_pair(args.model_short, p)
        if info is None:
            print(f"[{args.model_short}/{p}] residuals missing — skip")
            continue
        out_path = (REPO / "results" / "v11" / args.model_short / p / "pca_summary.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(info, indent=2))
        summary[p] = info
        late = info["late_layer"]
        r2z = info["pca_r2"]["PC1_vs_z"]
        print(f"[{args.model_short}/{p}] L{late}  PC1.R²(z)={r2z:.3f}  cos_PC1_primal[L{late}]="
              f"{info['cos_pc1_primal_per_layer'][late]:+.3f}", flush=True)

    out2 = REPO / "results" / "v11" / args.model_short / "cos_pc1_primal_summary.json"
    out2.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out2.relative_to(REPO)}")


if __name__ == "__main__":
    main()
