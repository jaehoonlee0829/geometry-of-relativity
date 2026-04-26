"""v11 P3c — orthogonalized increment R² across layers (CPU).

Methodology-critic fix vs the original v11 plan: the doc said
``h_L_orth = h_L - (h_L @ w_prev) * w_prev / ||w_prev||²`` where ``w_prev``
is a *probe weight* (z-output space) — projecting an activation onto a
probe-weight has no clean geometric meaning.

Correct formulation (used here): residualize ``h_L`` against the **previous
layer's z-prediction** ``ẑ_{L-1}``, not against the probe weight.

  1. Fit ridge probe at L-1:  w_{L-1} = ridge(h_{L-1}, z)
  2. Predict at L-1:           ẑ_{L-1}[i] = h_{L-1}[i] @ w_{L-1}      scalar per prompt
  3. Per feature j of h_L, regress out the contribution explained by ẑ_{L-1}:
        β_j = cov(h_L[:, j], ẑ_{L-1}) / var(ẑ_{L-1})
        h_L_resid[:, j] = h_L[:, j] - β_j · ẑ_{L-1}
  4. Fit new ridge probe on h_L_resid → CV-R²(z) = "new info added at L"

Cross-validation: 5-fold by `cell_seed` to avoid leakage from same-cell prompts.

Outputs:
  results/v11/<model_short>/<pair>/increment_r2_orthogonalized.json

Reports both the naive layer R²(z) curve and the orthogonalized "new info"
curve. Prediction (per FINDINGS §14): orthogonalized signal concentrates at
L3-L7 (encoding) with a small bump at L13-L17 (re-encoding for readout) and
near-zero elsewhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent

ALPHA_RIDGE = 1.0
N_FOLDS = 5


def cv_r2_groupwise(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """5-fold CV R² with sklearn GroupKFold over `groups`."""
    n_groups = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_groups)
    if n_splits < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(y, dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = Ridge(alpha=ALPHA_RIDGE)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_layer_predictions(h: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold ridge predictions of y from h."""
    n_groups = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_groups)
    preds = np.zeros_like(y, dtype=np.float64)
    if n_splits < 2:
        m = Ridge(alpha=ALPHA_RIDGE)
        m.fit(h, y)
        return m.predict(h)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(h, y, groups):
        m = Ridge(alpha=ALPHA_RIDGE)
        m.fit(h[tr], y[tr])
        preds[te] = m.predict(h[te])
    return preds


def residualize_on_scalar(h: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-feature OLS residualization of h on the scalar s.

    For each feature j of h:
      β_j = cov(h[:, j], s) / var(s)
      h_resid[:, j] = h[:, j] - β_j * s
    """
    s_centered = s - s.mean()
    var_s = float((s_centered ** 2).sum())
    if var_s < 1e-12:
        return h.astype(np.float64).copy()
    h64 = h.astype(np.float64)
    h_centered = h64 - h64.mean(axis=0, keepdims=True)
    betas = (h_centered * s_centered[:, None]).sum(axis=0) / var_s   # (d_model,)
    h_resid = h64 - betas[None, :] * s[:, None]
    return h_resid


def analyze_pair(model_short: str, pair: str) -> dict | None:
    res_path = (REPO / "results" / "v11" / model_short / pair /
                f"{model_short}_{pair}_v11_residuals.npz")
    if not res_path.exists():
        print(f"[{model_short}/{pair}] residuals missing — skip")
        return None
    d = np.load(res_path)
    acts = d["activations"]   # (N, n_layers, d_model)
    z = d["z"].astype(np.float64)
    # group by (x, z) cell so seeds within a cell stay in the same fold
    xs = d["x"]
    zs = d["z"]
    cell_id = np.array([
        f"{round(float(x), 4)}_{round(float(zv), 4)}" for x, zv in zip(xs, zs)
    ])
    _, group_idx = np.unique(cell_id, return_inverse=True)

    n_layers = acts.shape[1]

    naive_r2 = np.zeros(n_layers)
    orth_r2 = np.zeros(n_layers)
    print(f"[{model_short}/{pair}] running {n_layers}-layer increment R²...", flush=True)
    prev_pred: np.ndarray | None = None
    for L in range(n_layers):
        h = acts[:, L, :].astype(np.float64)

        naive_r2[L] = cv_r2_groupwise(h, z, group_idx)

        if prev_pred is None:
            # L=0: no previous prediction; orth_r2 == naive_r2 by construction
            orth_r2[L] = naive_r2[L]
        else:
            h_resid = residualize_on_scalar(h, prev_pred)
            orth_r2[L] = cv_r2_groupwise(h_resid, z, group_idx)

        # Update prev_pred from THIS layer's CV fits, for the next layer.
        prev_pred = fit_layer_predictions(h, z, group_idx)

        print(f"  L{L:2d}  naive_r2={naive_r2[L]:+.3f}  "
              f"orth_r2={orth_r2[L]:+.3f}", flush=True)

    return {
        "model_short": model_short,
        "pair": pair,
        "n_layers": n_layers,
        "naive_r2_per_layer": naive_r2.tolist(),
        "orth_r2_per_layer": orth_r2.tolist(),
        "alpha_ridge": ALPHA_RIDGE,
        "n_folds": N_FOLDS,
        "n_groups": int(group_idx.max() + 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True,
                    choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--pair", required=True,
                    help="pair name or 'all'")
    args = ap.parse_args()

    pairs = (["height", "age", "weight", "size", "speed",
              "wealth", "experience", "bmi_abs"]
             if args.pair == "all" else [args.pair])

    for p in pairs:
        result = analyze_pair(args.model_short, p)
        if result is None:
            continue
        out_path = (REPO / "results" / "v11" / args.model_short / p /
                    "increment_r2_orthogonalized.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  wrote {out_path.relative_to(REPO)}\n", flush=True)


if __name__ == "__main__":
    main()
