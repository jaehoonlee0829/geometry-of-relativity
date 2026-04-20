"""INLP concept erasure: iteratively null out w_z from activations and watch the
z-regression R² collapse. Compared against a random-direction null.

This is the 5th line of evidence for the paper's "w_z is the readout direction"
claim:

  - Line 1 (behavioral):  logit_diff tracks z (relativity ratio ≈ 1).
  - Line 2 (probe):       CV R²(z) >> CV R²(x).
  - Line 3 (geometry):    PCA of cell means aligns PC1 with z.
  - Line 4 (causal):      adding α·ŵ_z to the residual bends logit_diff.
  - Line 5 (erasure):     projecting ŵ_z *out* of activations kills R²(z)
                          faster than projecting a random direction. This is
                          what distinguishes "w_z is THE direction for z" from
                          "w_z happens to correlate with z".

Algorithm (Ravfogel et al. 2020):
  1. Train ridge probe for z on activations H. Get unit direction v_1.
  2. Project: H' = H (I - v_1 v_1^T).
  3. Retrain ridge probe on H'. Get v_2.
  4. Repeat k times. After each step, also measure CV R²(z), R²(x),
     R²(logit_diff) to see what information survives the erasure.
  5. Random-direction null: same schedule but with random unit vectors.
     If w_z is special, z-R² falls faster under INLP than under random projections.

Usage:
  python scripts/vast_remote/inlp_v4.py [--layer mid|late] [--steps 8]

Inputs:
  results/v4_dense/e4b_implicit_{layer}.npz       — activations
  results/v4_dense/e4b_implicit_logits.jsonl      — logit_diff
  results/v4_dense/e4b_trials.jsonl               — x, mu, z metadata

Outputs:
  results/v4_analysis/inlp_{layer}.json            — R² curves per step
  results/v4_analysis/figures/inlp_{layer}.png     — R² vs projection step
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"[fatal] sklearn missing: {e}", file=sys.stderr)
    sys.exit(2)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

REPO = Path(__file__).resolve().parent.parent.parent
V4_DIR = REPO / "results" / "v4_dense"
OUT_DIR = REPO / "results" / "v4_analysis"
FIG_DIR = OUT_DIR / "figures"


def load_data(layer: str):
    """Return (acts, xs, mus, zs, logit_diffs), aligned by id."""
    trials_path = V4_DIR / "e4b_trials.jsonl"
    logits_path = V4_DIR / "e4b_implicit_logits.jsonl"
    acts_path = V4_DIR / f"e4b_implicit_{layer}.npz"

    trials = {}
    with trials_path.open() as f:
        for line in f:
            t = json.loads(line)
            trials[t["id"]] = t

    logits = {}
    with logits_path.open() as f:
        for line in f:
            r = json.loads(line)
            logits[r["id"]] = r

    with np.load(acts_path, allow_pickle=True) as z:
        acts = z["activations"].astype(np.float64)
        ids = z["ids"].tolist()

    xs = np.array([trials[i]["x"] for i in ids])
    mus = np.array([trials[i]["mu"] for i in ids])
    zs = np.array([trials[i]["z"] for i in ids])
    lds = np.array([logits[i]["logit_diff"] for i in ids])
    return acts, xs, mus, zs, lds, ids


def fit_unit_direction_for(y: np.ndarray, X: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Fit Ridge y ~ X and return the UNIT direction (in ORIGINAL space).

    We fit in scaled-X space so regularization is meaningful across dimensions,
    then back-transform the weight vector to unit-scale. Finally unit-normalize.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X_s, y)
    w_scaled = m.coef_
    # Back-transform. StandardScaler divides by scale_, so unscaled w = w_scaled / scale_
    w = w_scaled / scaler.scale_
    n = np.linalg.norm(w)
    if n < 1e-12:
        return np.zeros_like(w)
    return w / n


def cv_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
          cv: int = 5, seed: int = 0) -> float:
    """5-fold CV R² of Ridge(y ~ X). Uses *shuffled* folds so the test set
    distribution matches train (critical for data that comes pre-sorted by
    x/mu/seed — otherwise each fold gets a distinct x bucket and R² collapses
    to spurious negatives).
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(model, X_s, y, cv=kf, scoring="r2")))


def project_out(H: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return H (I - v v^T), assuming v is a UNIT vector."""
    # Efficient: subtract the component along v.
    # proj = (H @ v) outer v   → shape (N, d)
    proj_coeffs = H @ v  # (N,)
    return H - np.outer(proj_coeffs, v)


def run_inlp(acts: np.ndarray, xs, mus, zs, lds, steps: int,
             mode: str, seed: int = 0) -> dict:
    """One INLP schedule.

    mode: "z"       — each step uses the new z-probe direction (true INLP).
          "random"  — each step projects out a random unit direction (null).
          "x"       — each step projects out the x-probe direction (interference).
    """
    H = acts.copy()
    rng = np.random.default_rng(seed)
    record = {"step": [], "r2_z": [], "r2_x": [], "r2_ld": [],
              "cos_v_z_current": []}  # cosine between projected-out direction and current z-probe

    # Initial metrics
    record["step"].append(0)
    record["r2_z"].append(cv_r2(H, zs))
    record["r2_x"].append(cv_r2(H, xs))
    record["r2_ld"].append(cv_r2(H, lds))
    record["cos_v_z_current"].append(1.0)  # identity

    for s in range(1, steps + 1):
        # Pick the direction to project out.
        if mode == "z":
            v = fit_unit_direction_for(zs, H)
        elif mode == "x":
            v = fit_unit_direction_for(xs, H)
        elif mode == "random":
            v = rng.standard_normal(H.shape[1])
            v /= np.linalg.norm(v) + 1e-12
        else:
            raise ValueError(mode)

        # Record cos(v, current w_z) BEFORE projecting — diagnostic.
        w_z_cur = fit_unit_direction_for(zs, H)
        if np.linalg.norm(w_z_cur) < 1e-12:
            cos_cur = 0.0
        else:
            cos_cur = float(np.abs(np.dot(v, w_z_cur)))
        record["cos_v_z_current"].append(cos_cur)

        # Project.
        H = project_out(H, v)

        # Metrics post-projection.
        record["step"].append(s)
        record["r2_z"].append(cv_r2(H, zs))
        record["r2_x"].append(cv_r2(H, xs))
        record["r2_ld"].append(cv_r2(H, lds))

    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", default="late", choices=["mid", "late"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading v4_dense/implicit {args.layer} ...")
    acts, xs, mus, zs, lds, ids = load_data(args.layer)
    print(f"  N={len(ids)}  d={acts.shape[1]}")

    print(f"\n=== INLP mode=z  (iteratively project out the z-probe direction) ===")
    rec_z = run_inlp(acts, xs, mus, zs, lds, args.steps, mode="z", seed=args.seed)

    print(f"\n=== Null mode=random  (project out random unit directions) ===")
    rec_rand = run_inlp(acts, xs, mus, zs, lds, args.steps, mode="random", seed=args.seed)

    print(f"\n=== Interference mode=x  (project out the x-probe direction) ===")
    rec_x = run_inlp(acts, xs, mus, zs, lds, args.steps, mode="x", seed=args.seed)

    def fmt_row(rec, label):
        return (f"{label:10s}  "
                + "  ".join(f"{v:+.3f}" for v in rec["r2_z"]))

    print(f"\n{'='*70}")
    print(f"R²(z) vs projection step  ({args.steps} steps + initial)")
    print(f"{'='*70}")
    header = "mode      " + "  ".join(f"s={s:>2}" for s in rec_z["step"])
    print(header)
    print(fmt_row(rec_z, "INLP-z"))
    print(fmt_row(rec_rand, "random"))
    print(fmt_row(rec_x, "INLP-x"))

    # Key claim: R²(z) under INLP-z drops toward 0 far faster than under random
    final_z = rec_z["r2_z"][-1]
    final_rand = rec_rand["r2_z"][-1]
    print(f"\nFinal R²(z) after {args.steps} steps:")
    print(f"  INLP-z = {final_z:+.3f}  (expect ≪ random if w_z captures the signal)")
    print(f"  random = {final_rand:+.3f}")
    print(f"  gap    = {final_rand - final_z:+.3f}  (positive = INLP outperforms random)")

    # Logit-diff decodability: how much does it suffer under INLP-z?
    print(f"\nR²(logit_diff) after {args.steps} steps:")
    print(f"  INLP-z = {rec_z['r2_ld'][-1]:+.3f}")
    print(f"  random = {rec_rand['r2_ld'][-1]:+.3f}")

    # x decodability under INLP-z: should NOT collapse (x and z are distinguishable)
    print(f"\nR²(x) after {args.steps} steps:")
    print(f"  INLP-z = {rec_z['r2_x'][-1]:+.3f}   (should stay high — x is not z)")
    print(f"  INLP-x = {rec_x['r2_x'][-1]:+.3f}   (should fall — x direction nulled)")

    out = {
        "layer": args.layer,
        "n_steps": args.steps,
        "N": len(ids),
        "d": int(acts.shape[1]),
        "inlp_z":  rec_z,
        "null_random": rec_rand,
        "inlp_x":  rec_x,
    }
    out_path = OUT_DIR / f"inlp_{args.layer}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Plot
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        steps = rec_z["step"]
        for ax, key, title in [
            (axes[0], "r2_z",  "R²(z) by projection step"),
            (axes[1], "r2_x",  "R²(x)"),
            (axes[2], "r2_ld", "R²(logit_diff)"),
        ]:
            ax.plot(steps, rec_z[key],    "-o", label="INLP-z")
            ax.plot(steps, rec_rand[key], "-o", label="random")
            ax.plot(steps, rec_x[key],    "-o", label="INLP-x")
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel("projection step")
            ax.set_ylabel("CV R²")
            ax.set_title(title)
            ax.legend()
        fig.suptitle(f"INLP concept erasure (layer={args.layer})", fontsize=12)
        fig.tight_layout()
        fig_path = FIG_DIR / f"inlp_{args.layer}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"Saved {fig_path}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
