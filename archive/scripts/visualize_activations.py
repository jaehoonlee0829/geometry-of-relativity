"""Visualize activation geometry: PCA projections colored by x, z, and mu.

Creates 2D and 3D PCA plots showing how activations vary across
the (x, z) grid, and checks whether the adjective probe's predictions
correlate with z even though the weight vectors don't align.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
ACT_DIR = REPO / "results" / "activations"
PROMPTS = REPO / "data_gen" / "prompts_v2.jsonl"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_trials(domain: str) -> list[dict]:
    trials = []
    with PROMPTS.open() as f:
        for line in f:
            t = json.loads(line)
            if t["domain"] == domain:
                trials.append(t)
    return trials


def load_activations(model: str, domain: str, layer: str):
    path = ACT_DIR / f"{model}_{domain}_{layer}.npz"
    with np.load(path, allow_pickle=True) as z:
        return z["activations"], z["ids"].tolist()


def analyze_and_plot(model: str, domain: str, layer: str):
    trials = load_trials(domain)
    acts, ids = load_activations(model, domain, layer)
    trial_by_id = {t["id"]: t for t in trials}

    xs = np.array([trial_by_id[tid]["x"] for tid in ids])
    zs = np.array([trial_by_id[tid]["z"] for tid in ids])
    mus = np.array([trial_by_id[tid]["mu"] for tid in ids])
    ctx_types = np.array([trial_by_id[tid]["context_type"] for tid in ids])
    frames = np.array([trial_by_id[tid]["prompt_frame"] for tid in ids])

    # ---- PCA on activations ----
    scaler = StandardScaler()
    acts_s = scaler.fit_transform(acts)
    pca = PCA(n_components=10)
    acts_pca = pca.fit_transform(acts_s)

    print(f"\n{'='*60}")
    print(f"  {model} / {domain} / {layer}")
    print(f"  Explained variance (top 10 PCs): {pca.explained_variance_ratio_[:10].round(3)}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_[:10]).round(3)}")

    # ---- Train probes and get predictions ----
    w_z_model = Ridge(alpha=1.0).fit(acts_s, zs)
    w_x_model = Ridge(alpha=1.0).fit(acts_s, xs)
    y_adj = (zs > 0).astype(float)
    w_adj_model = Ridge(alpha=1.0).fit(acts_s, y_adj)

    z_pred = w_z_model.predict(acts_s)
    x_pred = w_x_model.predict(acts_s)
    adj_pred = w_adj_model.predict(acts_s)

    # ---- Key question: does adj_pred correlate with z empirically? ----
    corr_adj_z = np.corrcoef(adj_pred, zs)[0, 1]
    corr_adj_x = np.corrcoef(adj_pred, xs)[0, 1]
    corr_adj_mu = np.corrcoef(adj_pred, mus)[0, 1]
    print(f"  Probe prediction correlations:")
    print(f"    corr(adj_pred, z)  = {corr_adj_z:+.3f}")
    print(f"    corr(adj_pred, x)  = {corr_adj_x:+.3f}")
    print(f"    corr(adj_pred, mu) = {corr_adj_mu:+.3f}")

    # ---- Also check: does the ACTUAL z-probe output track the adj probe? ----
    corr_zpred_adjpred = np.corrcoef(z_pred, adj_pred)[0, 1]
    print(f"    corr(z_pred, adj_pred) = {corr_zpred_adjpred:+.3f}")

    # ---- Filter to implicit_is only for cleaner viz ----
    mask_implicit_is = (ctx_types == "implicit") & (frames == "is")
    n_sub = mask_implicit_is.sum()

    # ---- Figure 1: PCA colored by z, x, mu ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, vals, label, cmap in [
        (axes[0], zs, "z-score", "RdBu_r"),
        (axes[1], xs, "raw x", "viridis"),
        (axes[2], mus, "context μ", "plasma"),
    ]:
        sc = ax.scatter(
            acts_pca[mask_implicit_is, 0],
            acts_pca[mask_implicit_is, 1],
            c=vals[mask_implicit_is],
            cmap=cmap,
            s=30,
            alpha=0.7,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"Colored by {label}")
        plt.colorbar(sc, ax=ax)

    plt.suptitle(f"{model} / {domain} / {layer} — PCA of activations (implicit_is only, n={n_sub})", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / f"pca_{model}_{domain}_{layer}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    # ---- Figure 2: adj_pred vs z, colored by x ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sc = ax.scatter(zs[mask_implicit_is], adj_pred[mask_implicit_is],
                    c=xs[mask_implicit_is], cmap="viridis", s=30, alpha=0.7)
    ax.set_xlabel("z-score (ground truth)")
    ax.set_ylabel("Adjective probe prediction")
    ax.set_title(f"adj_pred vs z (colored by x)\ncorr={corr_adj_z:+.3f}")
    plt.colorbar(sc, ax=ax, label="raw x")

    ax = axes[1]
    sc = ax.scatter(xs[mask_implicit_is], adj_pred[mask_implicit_is],
                    c=zs[mask_implicit_is], cmap="RdBu_r", s=30, alpha=0.7)
    ax.set_xlabel("raw x (ground truth)")
    ax.set_ylabel("Adjective probe prediction")
    ax.set_title(f"adj_pred vs x (colored by z)\ncorr={corr_adj_x:+.3f}")
    plt.colorbar(sc, ax=ax, label="z-score")

    plt.suptitle(f"{model} / {domain} / {layer}", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / f"adj_vs_zx_{model}_{domain}_{layer}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    # ---- Figure 3: the (x, mu) grid — mean activation PC1 per cell ----
    unique_x = sorted(set(xs[mask_implicit_is]))
    unique_mu = sorted(set(mus[mask_implicit_is]))

    grid_pc1 = np.full((len(unique_x), len(unique_mu)), np.nan)
    grid_pc2 = np.full((len(unique_x), len(unique_mu)), np.nan)
    grid_adj = np.full((len(unique_x), len(unique_mu)), np.nan)

    for i, x_val in enumerate(unique_x):
        for j, mu_val in enumerate(unique_mu):
            mask_cell = mask_implicit_is & (xs == x_val) & (mus == mu_val)
            if mask_cell.sum() > 0:
                grid_pc1[i, j] = acts_pca[mask_cell, 0].mean()
                grid_pc2[i, j] = acts_pca[mask_cell, 1].mean()
                grid_adj[i, j] = adj_pred[mask_cell].mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, grid, title, cmap in [
        (axes[0], grid_pc1, "Mean PC1", "RdBu_r"),
        (axes[1], grid_pc2, "Mean PC2", "RdBu_r"),
        (axes[2], grid_adj, "Mean adj_pred", "RdBu_r"),
    ]:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(unique_mu)))
        ax.set_xticklabels([f"{m:.0f}" for m in unique_mu], fontsize=7, rotation=45)
        ax.set_yticks(range(len(unique_x)))
        ax.set_yticklabels([f"{x:.0f}" for x in unique_x], fontsize=7)
        ax.set_xlabel("Context mean μ")
        ax.set_ylabel("Target value x")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"{model} / {domain} / {layer} — (x, μ) grid (implicit_is)", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / f"grid_{model}_{domain}_{layer}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    return {
        "model": model, "domain": domain, "layer": layer,
        "corr_adj_z": corr_adj_z, "corr_adj_x": corr_adj_x,
        "corr_adj_mu": corr_adj_mu,
        "pca_var_explained_3": float(np.sum(pca.explained_variance_ratio_[:3])),
    }


def main():
    results = []
    for model in ["e4b", "g31b"]:
        for domain in ["height", "wealth"]:
            for layer in ["mid", "late"]:
                if model == "g31b" and layer == "final":
                    continue
                results.append(analyze_and_plot(model, domain, layer))

    # Summary
    print(f"\n{'='*70}")
    print("CORRELATION SUMMARY: corr(adj_pred, z) vs corr(adj_pred, x)")
    print(f"{'='*70}")
    print(f"{'model':<6} {'domain':<8} {'layer':<6} {'corr(a,z)':<12} {'corr(a,x)':<12} {'corr(a,mu)':<12}")
    for r in results:
        print(f"{r['model']:<6} {r['domain']:<8} {r['layer']:<6} "
              f"{r['corr_adj_z']:<+12.3f} {r['corr_adj_x']:<+12.3f} {r['corr_adj_mu']:<+12.3f}")


if __name__ == "__main__":
    main()
