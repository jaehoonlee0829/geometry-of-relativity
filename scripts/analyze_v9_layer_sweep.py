"""v9 §13: per-layer geometry sweep across all 26 decoder layers.

For each pair × each layer k ∈ 0..25, reads the (n, 26, d) activations
produced by `extract_v9_gemma2_all_layers.py` and computes:

  (1) Ridge probe R²(z)    — encoding quality
  (2) Ridge probe R²(x)    — raw-value encoding quality
  (3) Top-PC r²(z)         — how much of PC1 is z
  (4) Intrinsic dimension  — TWO-NN estimator on cell-means (5 cells × x)
  (5) Norm of primal_z, cos with previous layer's primal_z (continuity)
  (6) LFP 5×5 Gram eigenvalue spectrum + participation ratio

Matches Goodfire §3.2: "intrinsic dimensionality of the manifold
increases over layers, but drops at the last layer."

Outputs
  results/v9_gemma2/layer_sweep_geometry.json
  figures/v9/layer_sweep_probe_r2.png
  figures/v9/layer_sweep_intrinsic_dim.png
  figures/v9/layer_sweep_primal_continuity.png
  figures/v9/layer_sweep_lfp_id.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import PAIRS  # noqa: E402
from extract_v7_xz_grid import Z_VALUES  # noqa: E402

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_pair_all_layers(pair_name: str):
    """Return (acts shape (n, L, d), zs (n,), xs (n,))."""
    with np.load(RES_DIR / f"gemma2_{pair_name}_alllayers.npz",
                 allow_pickle=True) as z_:
        acts = z_["activations"].astype(np.float32)  # (n, L, d)
        ids = z_["ids"].tolist()
    trials = {}
    with (RES_DIR / "gemma2_trials.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["pair"] == pair_name:
                trials[t["id"]] = t
    zs = np.array([trials[i]["z"] for i in ids], dtype=np.float64)
    xs = np.array([trials[i]["x"] for i in ids], dtype=np.float64)
    return acts, zs, xs


def r2_against(z, coord):
    A = np.column_stack([np.ones_like(coord), coord])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def ridge_cv_r2(X, y, alpha=1.0, k=5):
    """Simple k-fold CV R² with Ridge."""
    n = len(y)
    fold = np.arange(n) % k
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    fold = fold[perm]
    preds = np.zeros(n)
    for kk in range(k):
        tr = perm[fold != kk]
        te = perm[fold == kk]
        model = Ridge(alpha=alpha, fit_intercept=True).fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def two_nn_id(X: np.ndarray) -> float:
    """Facco et al. 2017 TWO-NN intrinsic-dim estimator.

    For each point find the 1st and 2nd nearest neighbors with distances
    d1 < d2. Then μ = d2/d1, and ID = 1 / mean(log μ).
    Robust to curvature; good when n >> ID.
    """
    from scipy.spatial.distance import cdist
    D = cdist(X, X)
    D_sorted = np.sort(D, axis=1)  # first col is 0 (self)
    d1 = D_sorted[:, 1]
    d2 = D_sorted[:, 2]
    mu = d2 / np.maximum(d1, 1e-12)
    logmu = np.log(np.maximum(mu, 1.0 + 1e-9))
    m = float(logmu.mean())
    return 1.0 / m if m > 1e-9 else float("nan")


def lfp_gram_id(acts_layer: np.ndarray, zs: np.ndarray, z_vals):
    """Train one probe per z-value; return (Gram eigvals, participation_ratio)."""
    probes = []
    for z_val in z_vals:
        y = (np.isclose(zs, z_val, atol=1e-6)).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            probes.append(np.zeros(acts_layer.shape[1], dtype=np.float32))
            continue
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=300, solver="lbfgs").fit(acts_layer, y)
        probes.append(clf.coef_.ravel().astype(np.float32))
    W = np.stack(probes)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    G = W @ W.T
    G = 0.5 * (G + G.T)
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.sort(eigvals)[::-1]
    eigvals_pos = np.maximum(eigvals, 0)
    pr = float(eigvals_pos.sum() ** 2 / (eigvals_pos ** 2).sum())
    return eigvals.tolist(), pr


def analyze_pair_all_layers(pair_name: str):
    acts, zs, xs = load_pair_all_layers(pair_name)
    n, L, d = acts.shape
    prev_primal = None
    layer_records = []
    for k in range(L):
        A = acts[:, k, :]
        # Cell means for TWO-NN ID: one mean per (x, z) cell
        x_vals = sorted(set(xs))
        z_vals = sorted(set(zs))
        cell_means = []
        for xv in x_vals:
            for zv in z_vals:
                mask = (np.isclose(xs, xv) & np.isclose(zs, zv))
                if mask.any():
                    cell_means.append(A[mask].mean(axis=0))
        cell_means = np.stack(cell_means) if cell_means else A[:0]

        # (1, 2) Probe R²
        r2_z = ridge_cv_r2(A, zs)
        r2_x = ridge_cv_r2(A, xs)

        # (3) Top-PC r²(z) using best of top-2 PCs
        pca = PCA(n_components=2).fit_transform(A)
        r2_pc1 = r2_against(zs, pca[:, 0])
        r2_pc2 = r2_against(zs, pca[:, 1])
        r2_top12 = float(max(r2_pc1, r2_pc2))

        # (4) Intrinsic dim on cell-means
        try:
            id_cell = two_nn_id(cell_means) if len(cell_means) >= 5 else float("nan")
        except Exception:
            id_cell = float("nan")

        # (5) primal direction + continuity
        hi = zs > 0
        lo = zs < 0
        primal = (A[hi].mean(0) - A[lo].mean(0)).astype(np.float32)
        norm_p = float(np.linalg.norm(primal))
        cos_prev = (
            float(primal @ prev_primal /
                  (norm_p * np.linalg.norm(prev_primal) + 1e-9))
            if prev_primal is not None else float("nan")
        )
        prev_primal = primal

        # (6) LFP Gram ID on raw activations
        eig, pr = lfp_gram_id(A, zs, list(Z_VALUES))

        layer_records.append({
            "layer": k,
            "r2_cv_z": r2_z,
            "r2_cv_x": r2_x,
            "r2_pc1_z": r2_pc1,
            "r2_pc2_z": r2_pc2,
            "r2_top12_z": r2_top12,
            "id_cell_means_TWONN": id_cell,
            "primal_norm": norm_p,
            "primal_cos_prev_layer": cos_prev,
            "lfp_gram_eigvals": eig,
            "lfp_participation_ratio": pr,
        })
    return {"pair": pair_name, "n_layers": L, "layer_records": layer_records}


def main():
    all_results = []
    for p in PAIRS:
        print(f"\n=== {p.name} ===", flush=True)
        r = analyze_pair_all_layers(p.name)
        print(f"  {p.name}: layers={r['n_layers']}", flush=True)
        # Brief per-pair summary printout
        rec = r["layer_records"]
        peak_z = max(rec, key=lambda x: x["r2_cv_z"])
        print(f"  peak R²(z)    at layer {peak_z['layer']:2d}: {peak_z['r2_cv_z']:.3f}")
        peak_id = max(rec, key=lambda x: x["id_cell_means_TWONN"]
                      if np.isfinite(x["id_cell_means_TWONN"]) else -1)
        print(f"  peak ID(TWO-NN) at layer {peak_id['layer']:2d}: "
              f"{peak_id['id_cell_means_TWONN']:.2f}")
        all_results.append(r)
    (RES_DIR / "layer_sweep_geometry.json").write_text(
        json.dumps({"pairs": all_results}, indent=2)
    )
    print(f"\nWrote {RES_DIR}/layer_sweep_geometry.json")

    # ---- PLOTS ----
    pairs = [p.name for p in PAIRS]
    L = all_results[0]["n_layers"]
    layers = np.arange(L)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for r in all_results:
        rec = r["layer_records"]
        a1.plot(layers, [x["r2_cv_z"] for x in rec], "-o", label=r["pair"], ms=3)
    a1.set_title("CV R²(z) per layer (Ridge, 5-fold)", fontsize=11)
    a1.set_xlabel("layer"); a1.set_ylabel("R²(z)")
    a1.legend(fontsize=7, ncol=2); a1.grid(alpha=0.3)
    for r in all_results:
        rec = r["layer_records"]
        a2.plot(layers, [x["r2_cv_x"] for x in rec], "-o", label=r["pair"], ms=3)
    a2.set_title("CV R²(x) per layer", fontsize=11)
    a2.set_xlabel("layer"); a2.set_ylabel("R²(x)")
    a2.legend(fontsize=7, ncol=2); a2.grid(alpha=0.3)
    fig.suptitle("v9 §13 — encoding quality by depth", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / "layer_sweep_probe_r2.png", dpi=140)
    print(f"  wrote {FIG_DIR}/layer_sweep_probe_r2.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in all_results:
        rec = r["layer_records"]
        ax.plot(layers, [x["id_cell_means_TWONN"] for x in rec], "-o",
                label=r["pair"], ms=3)
    ax.set_title("v9 §13 — intrinsic dimension (TWO-NN on cell-means) by layer\n"
                 "Goodfire prediction: rises with depth, drops at the last layer",
                 fontsize=10)
    ax.set_xlabel("layer"); ax.set_ylabel("ID (TWO-NN)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "layer_sweep_intrinsic_dim.png", dpi=140)
    print(f"  wrote {FIG_DIR}/layer_sweep_intrinsic_dim.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in all_results:
        rec = r["layer_records"]
        a1.plot(layers, [x["primal_norm"] for x in rec], "-o", label=r["pair"], ms=3)
        a2.plot(layers[1:], [x["primal_cos_prev_layer"] for x in rec[1:]],
                "-o", label=r["pair"], ms=3)
    a1.set_title("‖primal_z‖ per layer", fontsize=11)
    a1.set_xlabel("layer"); a1.set_ylabel("‖primal_z‖")
    a1.legend(fontsize=7, ncol=2); a1.grid(alpha=0.3)
    a2.set_title("cos(primal_z[layer], primal_z[layer-1])", fontsize=11)
    a2.set_xlabel("layer"); a2.set_ylabel("cosine")
    a2.axhline(0, color="k", lw=0.3); a2.axhline(1, color="k", lw=0.3, alpha=0.5)
    a2.legend(fontsize=7, ncol=2); a2.grid(alpha=0.3)
    fig.suptitle("v9 §13 — primal direction: magnitude and continuity across layers",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "layer_sweep_primal_continuity.png", dpi=140)
    print(f"  wrote {FIG_DIR}/layer_sweep_primal_continuity.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in all_results:
        rec = r["layer_records"]
        ax.plot(layers, [x["lfp_participation_ratio"] for x in rec], "-o",
                label=r["pair"], ms=3)
    ax.axhline(5, color="k", ls=":", lw=0.6, label="max (K=5)")
    ax.axhline(1, color="k", ls=":", lw=0.6, label="min (all-same-direction)")
    ax.set_title("v9 §13 — LFP 5-probe Gram participation ratio per layer\n"
                 "near 5 = z-probes orthogonal; near 1 = single z-axis",
                 fontsize=10)
    ax.set_xlabel("layer"); ax.set_ylabel("participation ratio")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "layer_sweep_lfp_id.png", dpi=140)
    print(f"  wrote {FIG_DIR}/layer_sweep_lfp_id.png")


if __name__ == "__main__":
    main()
