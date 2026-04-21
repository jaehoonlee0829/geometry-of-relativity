"""v8 Priority 3: replot 4 activation-geometry figures on Grid B (clean).

Replaces v4/v6 confounded-grid versions of:
  - PCA horseshoe (PC1 vs PC2, color by z) per pair
  - SVD scree of stacked per-pair PC1s (meta-direction analysis)
  - Cross-pair PC1 cosine heatmap
  - Zero-shot vs implicit direction comparison

All CPU-only from cached Grid B activations.

Writes:
  figures/v8/pca_horseshoe_gridB_8panel.png
  figures/v8/meta_w1_svd_scree_gridB.png
  figures/v8/cross_pair_pc1_cosine_gridB.png
  figures/v8/zeroshot_vs_implicit_gridB.png
  results/v8_replots/{pca, meta_w1, cross_pair, zeroshot_compare}.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_v4_adjpairs import PAIRS  # noqa: E402

V7 = REPO / "results" / "v7_xz_grid"
ZS_EXPANDED = REPO / "results" / "v4_zeroshot_expanded"
OUT = REPO / "results" / "v8_replots"
OUT_FIG = REPO / "figures" / "v8"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def load_pair(pair_name: str):
    trials_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in (V7 / "e4b_trials.jsonl").open()}
    npz = np.load(V7 / f"e4b_{pair_name}_late.npz", allow_pickle=True)
    acts = npz["activations"].astype(np.float64)
    ids = [str(s) for s in npz["ids"]]
    xs = np.array([trials_by_id[i]["x"] for i in ids])
    zs = np.array([trials_by_id[i]["z"] for i in ids])
    mus = np.array([trials_by_id[i]["mu"] for i in ids])
    return acts, xs, zs, mus


def cell_means(acts, xs, zs):
    by_cell = defaultdict(list)
    for a, x, z in zip(acts, xs, zs):
        by_cell[(x, z)].append(a)
    keys = sorted(by_cell)
    cmeans = np.stack([np.mean(by_cell[k], axis=0) for k in keys])
    cxs = np.array([k[0] for k in keys])
    czs = np.array([k[1] for k in keys])
    return cmeans, cxs, czs


def plot_pca_horseshoe_gridB():
    result = {}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, p_obj in zip(axes.ravel(), PAIRS):
        pn = p_obj.name
        acts, xs, zs, mus = load_pair(pn)
        cm, cxs, czs = cell_means(acts, xs, zs)
        centered = cm - cm.mean(0)
        pca = PCA(n_components=3).fit(centered)
        proj = centered @ pca.components_.T
        pc1_proj, pc2_proj = proj[:, 0], proj[:, 1]
        # sign-align PC1 with z
        if np.corrcoef(pc1_proj, czs)[0, 1] < 0:
            pc1_proj = -pc1_proj
        r2_pc1_z = float(np.corrcoef(pc1_proj, czs)[0, 1] ** 2)
        r2_pc1_x = float(np.corrcoef(pc1_proj, cxs)[0, 1] ** 2)
        r2_pc2_z2 = float(np.corrcoef(pc2_proj, czs**2)[0, 1] ** 2)
        sc = ax.scatter(pc1_proj, pc2_proj, c=czs, cmap="coolwarm", s=60, edgecolors="black", lw=0.4)
        ax.set_title(f"{pn}\nR²(PC1~z)={r2_pc1_z:.2f}  R²(PC2~z²)={r2_pc2_z2:.2f}",
                      fontsize=9)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8); ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, fraction=0.04, label="z")
        result[pn] = {"r2_pc1_z": r2_pc1_z, "r2_pc1_x": r2_pc1_x,
                      "r2_pc2_z2": r2_pc2_z2,
                      "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
    fig.suptitle("PCA horseshoe on Grid B cell-means (color = z)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "pca_horseshoe_gridB_8panel.png", dpi=140)
    plt.close(fig)
    return result


def meta_w1_svd_scree_gridB():
    pc1s, names = [], []
    for p_obj in PAIRS:
        acts, xs, zs, mus = load_pair(p_obj.name)
        cm, cxs, czs = cell_means(acts, xs, zs)
        centered = cm - cm.mean(0)
        pc1 = PCA(n_components=1).fit(centered).components_[0]
        if np.corrcoef(centered @ pc1, czs)[0, 1] < 0:
            pc1 = -pc1
        pc1s.append(pc1 / np.linalg.norm(pc1))
        names.append(p_obj.name)
    V = np.stack(pc1s)
    U, S, Wt = np.linalg.svd(V, full_matrices=False)
    var_ratio = (S**2) / (S**2).sum()
    w1 = Wt[0] / np.linalg.norm(Wt[0])

    # R² of each pair's PC1 projection onto w1 vs z
    r2_meta = {}
    for i, (pn, pc) in enumerate(zip(names, pc1s)):
        acts, xs, zs, mus = load_pair(pn)
        cm, cxs, czs = cell_means(acts, xs, zs)
        centered = cm - cm.mean(0)
        proj_w1 = centered @ w1
        r2_meta[pn] = float(np.corrcoef(proj_w1, czs)[0, 1] ** 2)

    result = {
        "pairs": names,
        "singular_values": S.tolist(),
        "variance_ratio": var_ratio.tolist(),
        "top_shared_variance_pct": float(var_ratio[0] * 100),
        "r2_meta_w1_vs_z_per_pair": r2_meta,
    }
    (OUT / "meta_w1.json").write_text(json.dumps(result, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(S)+1), var_ratio*100)
    axes[0].set_xlabel("singular value index"); axes[0].set_ylabel("% variance")
    axes[0].set_title(f"SVD scree (Grid B)\ntop singular vector captures {var_ratio[0]*100:.1f}% of cross-pair PC1 variance")
    axes[0].grid(alpha=0.3)
    names_sorted = names
    axes[1].bar(range(len(names_sorted)), [r2_meta[n] for n in names_sorted])
    axes[1].set_xticks(range(len(names_sorted))); axes[1].set_xticklabels(names_sorted, rotation=30, ha="right")
    axes[1].set_ylabel("R²(proj onto meta_w1 ~ z)")
    axes[1].set_title("Per-pair R² of meta_w1 projection against z")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "meta_w1_svd_scree_gridB.png", dpi=140)
    plt.close(fig)
    return result, pc1s, names


def cross_pair_cosine_gridB(pc1s, names):
    n = len(pc1s)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = float(np.dot(pc1s[i], pc1s[j]))
    offdiag_abs = np.mean(np.abs(M[~np.eye(n, dtype=bool)]))
    result = {"pairs": names, "cos_matrix": M.tolist(),
              "mean_offdiag_abs": float(offdiag_abs)}
    (OUT / "cross_pair.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.abs(M), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{abs(M[i][j]):.2f}", ha="center", va="center",
                    color="white" if abs(M[i][j]) > 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"Cross-pair |cos(PC1_i, PC1_j)| (Grid B, cell-means)\nmean off-diag = {offdiag_abs:.3f}")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "cross_pair_pc1_cosine_gridB.png", dpi=140)
    plt.close(fig)
    return result


def zeroshot_vs_implicit_gridB():
    zs_trials = {json.loads(l)["id"]: json.loads(l)
                 for l in (ZS_EXPANDED / "e4b_trials.jsonl").open()}
    result = {}
    names = [p.name for p in PAIRS]
    for pn in names:
        # Implicit probe (from Grid B cell means -- use all trials for Ridge)
        acts, xs, zs, mus = load_pair(pn)
        w_z_imp = Ridge(alpha=1.0).fit(acts, zs).coef_

        # Zero-shot probe
        zs_npz = np.load(ZS_EXPANDED / f"e4b_{pn}_late.npz", allow_pickle=True)
        zs_acts = zs_npz["activations"].astype(np.float64)
        zs_ids = [str(s) for s in zs_npz["ids"]]
        zs_xs = np.array([zs_trials[i]["x"] for i in zs_ids])
        w_x_zs = Ridge(alpha=1.0).fit(zs_acts, zs_xs).coef_
        pc1_zs = PCA(n_components=1).fit(zs_acts - zs_acts.mean(0)).components_[0]

        def c(u, v): return float(np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-12))
        result[pn] = {
            "cos_wx_zs_vs_wz_imp":  c(w_x_zs, w_z_imp),
            "cos_pc1_zs_vs_wz_imp": c(pc1_zs, w_z_imp),
        }
    (OUT / "zeroshot_compare.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(11, 5))
    xpos = np.arange(len(names))
    cos_w = [abs(result[n]["cos_wx_zs_vs_wz_imp"]) for n in names]
    cos_p = [abs(result[n]["cos_pc1_zs_vs_wz_imp"]) for n in names]
    ax.bar(xpos - 0.2, cos_w, 0.4, label="|cos(w_x_zeroshot, w_z_implicit)|")
    ax.bar(xpos + 0.2, cos_p, 0.4, label="|cos(PC1_zeroshot, w_z_implicit)|")
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.axhline(1/np.sqrt(2560), color="gray", ls="--",
               label="chance (√(1/d) = 0.020)", alpha=0.6)
    ax.set_title("Zero-shot x-direction vs implicit (Grid B) z-direction")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "zeroshot_vs_implicit_gridB.png", dpi=140)
    plt.close(fig)
    return result


def main():
    print("1. PCA horseshoe on Grid B cell-means")
    pca_res = plot_pca_horseshoe_gridB()
    (OUT / "pca.json").write_text(json.dumps(pca_res, indent=2))
    for p, r in pca_res.items():
        print(f"   {p:12s}  R²(PC1~z)={r['r2_pc1_z']:.3f}  R²(PC1~x)={r['r2_pc1_x']:.3f}  R²(PC2~z²)={r['r2_pc2_z2']:.3f}")

    print("\n2. SVD scree + meta_w1 per-pair R²")
    meta_res, pc1s, names = meta_w1_svd_scree_gridB()
    print(f"   top shared variance: {meta_res['top_shared_variance_pct']:.1f}%")
    for p, r in meta_res['r2_meta_w1_vs_z_per_pair'].items():
        print(f"   {p:12s}  R²(meta_w1 proj ~ z) = {r:.3f}")

    print("\n3. Cross-pair PC1 cosine heatmap")
    cross_res = cross_pair_cosine_gridB(pc1s, names)
    print(f"   mean |cos| off-diagonal = {cross_res['mean_offdiag_abs']:.3f}")

    print("\n4. Zero-shot vs implicit direction comparison")
    zs_res = zeroshot_vs_implicit_gridB()
    print(f"   per-pair |cos(w_x_zs, w_z_imp)|:")
    for p, r in zs_res.items():
        print(f"     {p:12s}  w_x: {abs(r['cos_wx_zs_vs_wz_imp']):.3f}  PC1: {abs(r['cos_pc1_zs_vs_wz_imp']):.3f}")
    print(f"\nwrote 4 plots in figures/v8/")


if __name__ == "__main__":
    main()
