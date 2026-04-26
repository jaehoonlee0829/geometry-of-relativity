"""v11.5 §G — fold-aware orthogonalized increment R².

The original P3c fitted ẑ_{L-1} on ALL prompts then residualized h_L's features
against that scalar — leaking train info into the held-out fold and producing
spuriously negative out-of-sample R² (down to −23.5 on bmi_abs L41).

Correct: within each CV fold, fit ẑ_{L-1} on the train fold ONLY, project on
the test fold, then orthogonalize h_L on that test-fold ẑ_{L-1} and refit
ẑ_L. Report R²(z) on the residualized test activations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
ALPHA = 1.0
N_FOLDS = 5


def fold_aware_orth_r2_per_layer(acts: np.ndarray, z: np.ndarray, groups: np.ndarray):
    """Returns (naive_r2, orth_r2) arrays of length n_layers."""
    n_layers = acts.shape[1]
    n_groups = int(groups.max() + 1)
    naive = np.zeros(n_layers); orth = np.zeros(n_layers)
    n_splits = min(N_FOLDS, n_groups)
    if n_splits < 2:
        return naive, orth
    gkf = GroupKFold(n_splits=n_splits)

    # Pre-compute per-fold ẑ_L for every layer (out-of-fold predictions only)
    splits = list(gkf.split(acts, z, groups))
    pred_z = np.zeros((n_layers, len(z)), dtype=np.float64)
    for L in range(n_layers):
        h = acts[:, L, :].astype(np.float64)
        for tr, te in splits:
            m = Ridge(alpha=ALPHA).fit(h[tr], z[tr])
            pred_z[L, te] = m.predict(h[te])
        ss = ((z - pred_z[L]) ** 2).sum(); ss0 = ((z - z.mean()) ** 2).sum()
        naive[L] = 1 - ss / max(ss0, 1e-12)

    # Orthogonalized: residualize h_L test-fold features against ẑ_{L-1}'s
    # *test-fold* predictions (each test fold uses ẑ_{L-1} that was fit on
    # the corresponding train fold), then refit ẑ_L on the residualized
    # test activations.
    for L in range(n_layers):
        if L == 0:
            orth[L] = naive[L]
            continue
        h = acts[:, L, :].astype(np.float64)
        prev_pred = pred_z[L - 1]      # out-of-fold ẑ_{L-1}
        pred_l_resid = np.zeros_like(z, dtype=np.float64)
        for tr, te in splits:
            # Within the train fold: regress out the LINEAR contribution of
            # prev_pred[tr] (a scalar per prompt) from each feature of h[tr].
            s_tr = prev_pred[tr]
            s_tr_centered = s_tr - s_tr.mean()
            var = float((s_tr_centered ** 2).sum())
            if var < 1e-9:
                resid_tr = h[tr]; resid_te = h[te]
            else:
                h_tr_centered = h[tr] - h[tr].mean(axis=0, keepdims=True)
                betas = (h_tr_centered * s_tr_centered[:, None]).sum(axis=0) / var
                resid_tr = h[tr] - betas[None, :] * s_tr[:, None]
                resid_te = h[te] - betas[None, :] * prev_pred[te][:, None]
            m = Ridge(alpha=ALPHA).fit(resid_tr, z[tr])
            pred_l_resid[te] = m.predict(resid_te)
        ss = ((z - pred_l_resid) ** 2).sum(); ss0 = ((z - z.mean()) ** 2).sum()
        orth[L] = 1 - ss / max(ss0, 1e-12)

    return naive, orth


def cell_groups(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    keys = np.array([f"{round(float(xi), 4)}_{round(float(zi), 4)}" for xi, zi in zip(x, z)])
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--pair", default="all")
    args = ap.parse_args()
    pairs = ALL_PAIRS if args.pair == "all" else [args.pair]

    for pair in pairs:
        rp = (REPO / "results" / "v11" / args.model_short / pair /
              f"{args.model_short}_{pair}_v11_residuals.npz")
        if not rp.exists(): continue
        d = np.load(rp)
        acts = d["activations"]
        z = d["z"].astype(np.float64)
        gid = cell_groups(d["x"], d["z"])
        print(f"[p3c-fix] {args.model_short}/{pair}  ({acts.shape[1]}L, {acts.shape[0]} prompts)",
              flush=True)
        naive, orth = fold_aware_orth_r2_per_layer(acts, z, gid)
        # Per-layer print (compact)
        n_layers = len(naive)
        print(f"[p3c-fix]   naive R²(z): " +
              f"L0={naive[0]:.3f}  L{n_layers // 2}={naive[n_layers // 2]:.3f}  "
              f"L{n_layers - 1}={naive[-1]:.3f}")
        print(f"[p3c-fix]   ORTH  R²(z): " +
              f"L0={orth[0]:.3f}  L{n_layers // 2}={orth[n_layers // 2]:.3f}  "
              f"L{n_layers - 1}={orth[-1]:.3f}  "
              f"max={orth.max():.3f}@L{int(orth.argmax())}")
        out = {
            "model_short": args.model_short,
            "pair": pair,
            "n_layers": int(n_layers),
            "n_prompts": int(acts.shape[0]),
            "n_groups": int(gid.max() + 1),
            "naive_r2_per_layer": naive.tolist(),
            "orth_r2_per_layer_FOLD_AWARE": orth.tolist(),
            "fold_aware": True,
            "alpha_ridge": ALPHA,
            "n_folds": N_FOLDS,
        }
        out_dir = REPO / "results" / "v11_5" / args.model_short / pair
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "increment_r2_fold_aware.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
