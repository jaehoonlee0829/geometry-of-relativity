"""v10 P2 — dimensionality of the activation manifold per layer.

Operates on the 400 cell-means (mean over 10 seeds per (x, z) cell).
Three methods, all per layer:

  A. PCA cumulative variance      → n_dims at 80/90/95/99%
  B. TWO-NN intrinsic dimensionality  (Facco et al. 2017)
  C. Gram matrix of one-vs-rest z-probes  → participation ratio

Inputs : results/v10/gemma2_height_v10_residuals.npz
Outputs: results/v10/dimensionality_per_layer.json
         figures/v10/pca_cumvar_5layers.png
         figures/v10/id_per_layer_3methods.png

CPU-only.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="lbfgs failed to converge")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, StandardScaler

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "v10"
FIG = REPO / "figures" / "v10"
FIG.mkdir(parents=True, exist_ok=True)


def cell_means(activations: np.ndarray, x: np.ndarray, z: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average activations across the 10 seeds per (x, z) cell.

    Returns (cell_acts (M, n_layers, d), cell_x (M,), cell_z (M,)).
    """
    keys = sorted({(round(float(x[i]), 4), round(float(z[i]), 4))
                   for i in range(len(x))})
    M = len(keys)
    n_layers, d = activations.shape[1], activations.shape[2]
    out = np.zeros((M, n_layers, d), dtype=np.float32)
    cx = np.zeros(M, dtype=np.float32)
    cz = np.zeros(M, dtype=np.float32)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    counts = np.zeros(M, dtype=np.int32)
    for i in range(len(x)):
        k = (round(float(x[i]), 4), round(float(z[i]), 4))
        j = key_to_idx[k]
        out[j] += activations[i]
        cx[j] = k[0]
        cz[j] = k[1]
        counts[j] += 1
    out /= counts[:, None, None]
    return out, cx, cz


def two_nn_id(X: np.ndarray, frac: float = 0.9) -> float:
    """Facco et al. 2017 TWO-NN intrinsic-dim estimator.

    For each point, ratio = dist(2nd NN) / dist(1st NN). The CDF of log(ratio)
    is linear in -d, so we fit a line on the lower `frac` of points (drops the
    long tail).
    """
    from scipy.spatial.distance import cdist
    n = X.shape[0]
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    near = np.partition(D, 2, axis=1)[:, :2]   # 1st and 2nd NN distances
    near.sort(axis=1)
    mu = near[:, 1] / np.clip(near[:, 0], 1e-12, None)
    mu = np.sort(mu)
    # Drop the top (1-frac) outliers
    keep = int(frac * n)
    mu = mu[:keep]
    Femp = (np.arange(1, keep + 1) - 0.5) / n
    y = -np.log1p(-Femp)        # cumulative empirical
    x = np.log(mu)
    # least squares slope through origin: y = d * x
    d_est = float((x * y).sum() / (x * x).sum())
    return d_est


def gram_participation_ratio(activations: np.ndarray, z: np.ndarray,
                             n_bins: int = 20) -> dict:
    """Compute Gram-matrix participation ratio of z-bin direction set.

    OvR: closed-form ridge regression of each bin-indicator → fast and gives
         the same separating-direction we'd get from logistic regression up
         to a constant scale, which doesn't change the normalized Gram.
    MC:  multinomial softmax LR (Sarfati's recipe) on standardized data.

    n_bins=20 matches the 20 z-values in the dense grid.
    """
    Xs = StandardScaler().fit_transform(activations).astype(np.float32)

    # Bin z by 21 quantile edges → up to n_bins classes
    edges = np.unique(np.quantile(z, np.linspace(0, 1, n_bins + 1)))
    if len(edges) - 1 < n_bins:
        n_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(z, edges[1:-1]), 0, n_bins - 1)

    # OvR via closed-form ridge:  W = Y^T X (X^T X + λI)^{-1}
    Y = np.zeros((Xs.shape[0], n_bins), dtype=np.float32)
    Y[np.arange(Xs.shape[0]), bin_idx] = 1.0
    Y -= Y.mean(0, keepdims=True)
    lam = 1.0
    XTX = Xs.T @ Xs
    np.fill_diagonal(XTX, XTX.diagonal() + lam)
    XTY = Xs.T @ Y
    W_ovr = np.linalg.solve(XTX, XTY).T   # (n_bins, d)
    W_ovr_n = normalize(W_ovr, axis=1)
    G_ovr = W_ovr_n @ W_ovr_n.T
    eig_ovr = np.clip(np.linalg.eigvalsh(G_ovr), 0, None)
    pr_ovr = (eig_ovr.sum() ** 2) / (eig_ovr ** 2).sum()

    # Multinomial softmax (Sarfati). LBFGS converges slowly on raw 2304-dim;
    # pre-reduce via PCA to the first 64 components (captures >95% var at every
    # layer per our own analysis) so the fit is <5 s/layer.
    pc = PCA(n_components=min(64, Xs.shape[1], Xs.shape[0] - 1))
    Xp = pc.fit_transform(Xs)
    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    clf.fit(Xp, bin_idx)
    # Project the PCA-space weight rows back to the original d_model space so
    # the Gram is comparable to OvR.  W_full = W_pca @ pc.components_  (K, d)
    W_mc = clf.coef_ @ pc.components_            # (n_bins, d)
    W_mc_n = normalize(W_mc, axis=1)
    G_mc = W_mc_n @ W_mc_n.T
    eig_mc = np.linalg.eigvalsh(G_mc)
    eig_mc = np.clip(eig_mc, 0, None)
    pr_mc = (eig_mc.sum() ** 2) / (eig_mc ** 2).sum()

    return {
        "pr_ovr": float(pr_ovr),
        "pr_multinomial": float(pr_mc),
        "n_bins": int(n_bins),
    }


def main() -> None:
    print("[P2] loading residuals...", flush=True)
    res = np.load(RES / "gemma2_height_v10_residuals.npz")
    acts = res["activations"].astype(np.float32)
    x = res["x"]
    z = res["z"]
    n_prompts, n_layers, d = acts.shape
    print(f"[P2]   acts {acts.shape}, n_prompts={n_prompts}, n_layers={n_layers}",
          flush=True)

    print("[P2] computing cell-means (400 cells × 10 seeds)...", flush=True)
    cell_acts, cell_x, cell_z = cell_means(acts, x, z)
    print(f"[P2]   cell_acts {cell_acts.shape}", flush=True)

    out: dict = {"layers": list(range(n_layers)), "per_layer": []}

    for L in range(n_layers):
        X = cell_acts[:, L, :]
        # PCA cumvar
        pca = PCA().fit(X)
        cv = np.cumsum(pca.explained_variance_ratio_)
        n80 = int(np.searchsorted(cv, 0.80) + 1)
        n90 = int(np.searchsorted(cv, 0.90) + 1)
        n95 = int(np.searchsorted(cv, 0.95) + 1)
        n99 = int(np.searchsorted(cv, 0.99) + 1)
        # TWO-NN
        id_2nn = two_nn_id(X)
        # Gram PR — fit on cell-mean activations (400 samples, sufficient for
        # K=20 binary directions and an order of magnitude faster than fitting
        # on all 4000 prompts).
        gram = gram_participation_ratio(X, cell_z)

        rec = {
            "layer": L,
            "pca_n80": n80, "pca_n90": n90, "pca_n95": n95, "pca_n99": n99,
            "two_nn_id": float(id_2nn),
            "gram_pr_ovr": gram["pr_ovr"],
            "gram_pr_multinomial": gram["pr_multinomial"],
        }
        out["per_layer"].append(rec)
        print(f"[P2] L{L:2d}  PCA-95%={n95:3d}  TWO-NN={id_2nn:.2f}  "
              f"Gram-PR(OvR)={gram['pr_ovr']:.2f}  "
              f"Gram-PR(MC)={gram['pr_multinomial']:.2f}", flush=True)

    json_path = RES / "dimensionality_per_layer.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"[P2] wrote {json_path}", flush=True)

    # ----- figures
    layers = np.arange(n_layers)
    pca95 = np.array([r["pca_n95"] for r in out["per_layer"]])
    pca99 = np.array([r["pca_n99"] for r in out["per_layer"]])
    pca80 = np.array([r["pca_n80"] for r in out["per_layer"]])
    twonn = np.array([r["two_nn_id"] for r in out["per_layer"]])
    pr_ovr = np.array([r["gram_pr_ovr"] for r in out["per_layer"]])
    pr_mc = np.array([r["gram_pr_multinomial"] for r in out["per_layer"]])

    # PCA cumvar curves at 5 strategic layers
    fig, ax = plt.subplots(figsize=(7, 5))
    for L in [0, 7, 13, 20, 25]:
        X = cell_acts[:, L, :]
        pca = PCA(n_components=min(60, X.shape[0])).fit(X)
        cv = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(np.arange(1, len(cv) + 1), cv, label=f"layer {L}")
    ax.axhline(0.95, ls="--", color="gray", lw=0.8)
    ax.axhline(0.99, ls=":", color="gray", lw=0.8)
    ax.set_xlabel("PCA components")
    ax.set_ylabel("cumulative variance explained")
    ax.set_xlim(1, 50)
    ax.set_title("PCA cumulative variance (400 cell-means, height)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "pca_cumvar_5layers.png", dpi=120)
    plt.close()

    # 3-method ID per layer
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, pca80, "o-", label="PCA n@80%", alpha=0.7)
    ax.plot(layers, pca95, "o-", label="PCA n@95%", alpha=0.7)
    ax.plot(layers, twonn, "s-", label="TWO-NN ID", color="C2", lw=2)
    ax.plot(layers, pr_mc, "^-", label="Gram PR (multinomial)", color="C3")
    ax.plot(layers, pr_ovr, "^--", label="Gram PR (OvR)", color="C4", alpha=0.7)
    ax.set_xlabel("layer")
    ax.set_ylabel("dimensionality estimate")
    ax.set_title("Manifold dimensionality per layer (400 cell-means)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "id_per_layer_3methods.png", dpi=120)
    plt.close()
    print(f"[P2] wrote {FIG}/pca_cumvar_5layers.png  and id_per_layer_3methods.png",
          flush=True)


if __name__ == "__main__":
    main()
