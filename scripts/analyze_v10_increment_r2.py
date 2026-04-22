"""v10 P6 — residual stream increment R² for z.

Compares per-layer:
  R²_residual_L  = R² of z on residual_L (cumulative through layer L)
  R²_increment_L = R² of z on (residual_L - residual_{L-1})  (this layer's add)

Prediction (per docs/NEXT_GPU_SESSION_v10.md §6):
  R²_residual saturates at L7
  R²_increment peaks at L3-L7 (active z-writers), drops near zero L8-L12,
  may rise again L13-L17 (re-encoding for late readout).

Inputs:  results/v10/gemma2_height_v10_residuals.npz
Outputs: results/v10/increment_r2_per_layer.json
         figures/v10/increment_r2_per_layer.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "v10"
FIG = REPO / "figures" / "v10"
FIG.mkdir(parents=True, exist_ok=True)


def _ridge_solve(Xtr: np.ndarray, ytr: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge: w = (X^T X + λI)^{-1} X^T y."""
    Xc = Xtr - Xtr.mean(0, keepdims=True)
    yc = ytr - ytr.mean()
    A = Xc.T @ Xc
    A.flat[::A.shape[0] + 1] += lam      # add λ to diagonal in-place
    w = np.linalg.solve(A, Xc.T @ yc)
    b = ytr.mean() - Xtr.mean(0) @ w
    return w, b


def cv_r2(X: np.ndarray, y: np.ndarray, k: int = 5, lam: float = 100.0) -> float:
    """5-fold CV R² using closed-form ridge — ~0.5 s for 4000×2304."""
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    preds = np.zeros_like(y, dtype=np.float64)
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    for tr, te in kf.split(X):
        w, b = _ridge_solve(X[tr], y[tr], lam)
        preds[te] = X[te] @ w + b
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main() -> None:
    print("[P6] loading residuals...", flush=True)
    res = np.load(RES / "gemma2_height_v10_residuals.npz")
    acts = res["activations"].astype(np.float32)
    z = res["z"].astype(np.float32)
    x = res["x"].astype(np.float32)
    n_prompts, n_layers, d = acts.shape
    print(f"[P6]   acts {acts.shape}", flush=True)

    # The hooks captured POST-block residuals (layer outputs). For increments we
    # need the pre-block input as well. Approximation: increment_L = h_L - h_{L-1}.
    # Layer 0 increment = h_0 (treats embedding+layer0 as the first add).

    r2_resid_z, r2_resid_x = [], []
    r2_incr_z, r2_incr_x = [], []
    for L in range(n_layers):
        h_L = acts[:, L, :]
        r2_resid_z.append(cv_r2(h_L, z))
        r2_resid_x.append(cv_r2(h_L, x))
        if L == 0:
            inc = h_L
        else:
            inc = h_L - acts[:, L - 1, :]
        r2_incr_z.append(cv_r2(inc, z))
        r2_incr_x.append(cv_r2(inc, x))
        print(f"[P6] L{L:2d}  resid R²(z)={r2_resid_z[-1]:.3f}  "
              f"incr R²(z)={r2_incr_z[-1]:.3f}  "
              f"resid R²(x)={r2_resid_x[-1]:.3f}  "
              f"incr R²(x)={r2_incr_x[-1]:.3f}", flush=True)

    out = {
        "layers": list(range(n_layers)),
        "r2_residual_z": list(map(float, r2_resid_z)),
        "r2_residual_x": list(map(float, r2_resid_x)),
        "r2_increment_z": list(map(float, r2_incr_z)),
        "r2_increment_x": list(map(float, r2_incr_x)),
    }
    json_path = RES / "increment_r2_per_layer.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"[P6] wrote {json_path}", flush=True)

    layers = np.arange(n_layers)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ax[0].plot(layers, r2_resid_z, "o-", label="z (cumulative)", color="C0", lw=2)
    ax[0].plot(layers, r2_incr_z, "s-", label="z (per-layer increment)", color="C1", lw=2)
    ax[0].axhline(0, color="gray", lw=0.6)
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("CV R²")
    ax[0].set_title("z-encoding by layer")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)
    ax[1].plot(layers, r2_resid_x, "o-", label="x (cumulative)", color="C0", lw=2)
    ax[1].plot(layers, r2_incr_x, "s-", label="x (per-layer increment)", color="C1", lw=2)
    ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].set_xlabel("layer")
    ax[1].set_title("x-encoding by layer (control)")
    ax[1].legend(loc="upper left")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "increment_r2_per_layer.png", dpi=120)
    plt.close()
    print(f"[P6] wrote {FIG}/increment_r2_per_layer.png", flush=True)


if __name__ == "__main__":
    main()
