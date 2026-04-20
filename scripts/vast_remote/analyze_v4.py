"""Comprehensive analysis of v4 dense extraction outputs.

Reads from results/v4_dense/:
  e4b_{implicit,explicit,zero_shot}_{mid,late}.npz  (activations)
  e4b_{implicit,explicit,zero_shot}_logits.jsonl    (logit_diff + top-5)
  e4b_trials.jsonl                                   (x, mu, z, seed, condition)

Writes to results/v4_analysis/:
  summary.json                 — every headline number in one file
  figures/*.png                — PCA, heatmaps, probe-weight scatters
  probes/*.npz                 — trained probe weight vectors
  tables/*.csv                 — cell-means, per-condition regressions

Phases:
  1. Behavioral — variance decomposition + OLS of logit_diff on (x, mu, z)
  2. Probes     — ridge w_x, w_z, w_adj, w_logit_diff with 5-fold CV R²
  3. Geometry   — PCA of 35 cell-mean activations; z-axis vs x-axis directions
  4. Controls   — scrambled-label probe, implicit/explicit/zero_shot comparison
  5. Metrics    — Euclidean vs Sigma^{-1} cosines (the "primal-dual" discrepancy)

Usage (from /workspace/repo on Vast):
  python scripts/vast_remote/analyze_v4.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Lazy imports — fail loud if sklearn/matplotlib missing on the host
try:
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError as e:
    print(f"[fatal] scikit-learn not installed: {e}", file=sys.stderr)
    sys.exit(2)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # figures skipped

REPO = Path(__file__).resolve().parent.parent.parent
V4_DIR = REPO / "results" / "v4_dense"
OUT_DIR = REPO / "results" / "v4_analysis"
FIG_DIR = OUT_DIR / "figures"
PROBE_DIR = OUT_DIR / "probes"
TABLE_DIR = OUT_DIR / "tables"

MODEL = "e4b"
LAYERS = ["mid", "late"]
CONDITIONS = ["implicit", "explicit", "zero_shot"]


# -------------------- Data loading --------------------

def load_trials() -> dict[str, dict]:
    """Load trial metadata keyed by id."""
    path = V4_DIR / f"{MODEL}_trials.jsonl"
    trials = {}
    with path.open() as f:
        for line in f:
            t = json.loads(line)
            trials[t["id"]] = t
    return trials


def load_logits(condition: str) -> dict[str, dict]:
    """Load per-prompt logit info keyed by id."""
    path = V4_DIR / f"{MODEL}_{condition}_logits.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def load_activations(condition: str, layer: str) -> tuple[np.ndarray, list[str]]:
    """Load activation matrix and ordered ids."""
    path = V4_DIR / f"{MODEL}_{condition}_{layer}.npz"
    with np.load(path, allow_pickle=True) as z:
        acts = z["activations"]
        ids = z["ids"].tolist()
    return acts, ids


# -------------------- Phase 1: Behavioral --------------------

def phase1_behavioral(trials, logits_all, condition_log=None) -> dict:
    """Variance decomposition + OLS regressions of logit_diff on (z, x, mu)."""
    results = {}
    print("\n" + "=" * 70)
    print("PHASE 1: BEHAVIORAL (logit_diff regressions)")
    print("=" * 70)

    for cond in CONDITIONS:
        if cond not in logits_all:
            continue
        logits = logits_all[cond]
        rows = []
        for tid, info in logits.items():
            t = trials[tid]
            rows.append({
                "id": tid,
                "x": t["x"], "mu": t["mu"], "z": t["z"],
                "seed": t["seed"],
                "logit_diff": info["logit_diff"],
                "logit_tall": info["logit_tall"],
                "logit_short": info["logit_short"],
            })
        rows_z = [r for r in rows if r is not None]
        if not rows_z:
            continue

        ys = np.array([r["logit_diff"] for r in rows_z])
        xs = np.array([r["x"] for r in rows_z])
        mus = np.array([r["mu"] for r in rows_z])
        zs = np.array([r["z"] for r in rows_z])

        # Cell means for variance decomposition (only meaningful for implicit)
        cell_mean = defaultdict(list)
        cell_z = {}  # remember z for each (x, mu) cell
        for r in rows_z:
            key = (r["x"], r["mu"])
            cell_mean[key].append(r["logit_diff"])
            cell_z[key] = r["z"]

        cell_means = {k: float(np.mean(v)) for k, v in cell_mean.items()}
        cell_stds = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in cell_mean.items()}

        # Total variance = between-cell + within-cell
        grand_mean = float(np.mean(ys))
        total_var = float(np.var(ys, ddof=1))
        within_var = float(np.mean([np.var(v, ddof=1) if len(v) > 1 else 0.0
                                    for v in cell_mean.values()]))
        between_var = max(total_var - within_var, 0.0)
        frac_between = between_var / total_var if total_var > 0 else 0.0

        # OLS: logit_diff ~ z
        if len(set(zs)) >= 2:
            A = np.column_stack([np.ones_like(zs), zs])
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            pred = A @ coef
            r2_z = 1.0 - np.sum((ys - pred) ** 2) / np.sum((ys - grand_mean) ** 2)
            slope_z = float(coef[1])
        else:
            r2_z, slope_z = 0.0, 0.0

        # OLS: logit_diff ~ x + mu
        if len(set(xs)) >= 2 and len(set(mus)) >= 2:
            A = np.column_stack([np.ones_like(xs), xs, mus])
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            pred = A @ coef
            r2_xmu = 1.0 - np.sum((ys - pred) ** 2) / np.sum((ys - grand_mean) ** 2)
            slope_x = float(coef[1])
            slope_mu = float(coef[2])
            # Relativity test: is slope_x ≈ -slope_mu? (pure z-dependence)
            relativity_ratio = -slope_mu / slope_x if abs(slope_x) > 1e-9 else None
        else:
            r2_xmu, slope_x, slope_mu, relativity_ratio = 0.0, 0.0, 0.0, None

        # OLS: logit_diff ~ z + x  (does absolute x add info beyond z?)
        if len(set(xs)) >= 2 and len(set(zs)) >= 2:
            A = np.column_stack([np.ones_like(xs), zs, xs])
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            pred = A @ coef
            r2_zx = 1.0 - np.sum((ys - pred) ** 2) / np.sum((ys - grand_mean) ** 2)
            slope_z_joint = float(coef[1])
            slope_x_joint = float(coef[2])
        else:
            r2_zx, slope_z_joint, slope_x_joint = 0.0, 0.0, 0.0

        out = {
            "n": len(ys),
            "logit_diff_mean": grand_mean,
            "logit_diff_std": float(np.std(ys, ddof=1)),
            "total_var": total_var,
            "within_cell_var": within_var,
            "between_cell_var": between_var,
            "frac_between_cell": frac_between,
            "r2_on_z": r2_z,
            "slope_z": slope_z,
            "r2_on_x_plus_mu": r2_xmu,
            "slope_x_alone": slope_x,
            "slope_mu_alone": slope_mu,
            "relativity_ratio_-mu/x": relativity_ratio,
            "r2_on_z_plus_x": r2_zx,
            "slope_z_joint": slope_z_joint,
            "slope_x_joint": slope_x_joint,
        }
        results[cond] = out

        print(f"\n  [{cond}]  N={out['n']}  mean={out['logit_diff_mean']:+.3f}  "
              f"sd={out['logit_diff_std']:.3f}")
        print(f"    variance:  within-cell={within_var:.3f}  between-cell={between_var:.3f}  "
              f"frac_between={frac_between*100:.1f}%")
        print(f"    OLS on z            :  R2={r2_z:.3f}  slope={slope_z:+.3f}")
        print(f"    OLS on x+mu          :  R2={r2_xmu:.3f}  "
              f"slope_x={slope_x:+.4f}  slope_mu={slope_mu:+.4f}  "
              f"(-mu/x ratio = {relativity_ratio})")
        print(f"    OLS on z+x (joint)   :  R2={r2_zx:.3f}  "
              f"slope_z={slope_z_joint:+.3f}  slope_x={slope_x_joint:+.4f}")

        # Save cell means table
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        with (TABLE_DIR / f"cell_means_{cond}.csv").open("w") as f:
            f.write("x,mu,z,n_seeds,mean_logit_diff,std_logit_diff\n")
            for (x, mu), vals in sorted(cell_mean.items()):
                z = cell_z.get((x, mu), 0.0)
                f.write(f"{x},{mu},{z:.2f},{len(vals)},"
                        f"{np.mean(vals):.4f},{np.std(vals, ddof=1) if len(vals)>1 else 0:.4f}\n")

    return results


# -------------------- Phase 2: Probes --------------------

def train_ridge_probe(X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
                      cv_seed: int = 0):
    """Train ridge probe. Returns weight in ORIGINAL (pre-scale) space + CV R2.

    CV folds are SHUFFLED because extract_v4_dense stores rows sorted by
    (x, mu, seed). Without shuffling, each fold becomes a distinct x-bucket
    and CV R² collapses (generalization to unseen x is not what we want to
    measure — we want within-distribution generalization to new seeds).
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    # Fit on full data to get final w
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_s, y)
    w_scaled = model.coef_
    # Back-transform to original space
    w = w_scaled / scaler.scale_
    # 5-fold shuffled CV
    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    cv_r2 = float(np.mean(cross_val_score(Ridge(alpha=alpha), X_s, y,
                                          cv=kf, scoring="r2")))
    return w, w_scaled, cv_r2, scaler


def metric_cosine(u, v, Sigma_inv=None):
    """Cosine under metric G: <u, Gv> / sqrt(<u,Gu><v,Gv>). G=I is Euclidean."""
    if Sigma_inv is None:
        gu, gv = u, v
    else:
        gu = Sigma_inv @ u
        gv = Sigma_inv @ v
    num = float(np.dot(u, gv))
    du = float(np.sqrt(max(np.dot(u, gu), 1e-24)))
    dv = float(np.sqrt(max(np.dot(v, gv), 1e-24)))
    return num / (du * dv)


def phase2_probes(trials, logits_all, condition="implicit") -> dict:
    """Train ridge probes for x, z, adj (sign z), logit_diff. Report CV R2."""
    print("\n" + "=" * 70)
    print(f"PHASE 2: PROBES on activations ({condition})")
    print("=" * 70)

    logits = logits_all[condition]
    results = {}

    for layer in LAYERS:
        try:
            acts, ids = load_activations(condition, layer)
        except FileNotFoundError:
            print(f"  [skip] {condition}/{layer}: file missing")
            continue

        N, d = acts.shape
        # Align with trial metadata
        xs = np.array([trials[tid]["x"] for tid in ids])
        mus = np.array([trials[tid]["mu"] for tid in ids])
        zs = np.array([trials[tid]["z"] for tid in ids])
        logit_diffs = np.array([logits[tid]["logit_diff"] for tid in ids])
        y_adj = (zs > 0).astype(np.float64)

        print(f"\n  [{condition}/{layer}]  N={N}  d={d}")

        w_x, w_x_s, r2_x, _ = train_ridge_probe(acts, xs)
        w_mu, _, r2_mu, _ = train_ridge_probe(acts, mus)
        w_z, w_z_s, r2_z, _ = train_ridge_probe(acts, zs)
        w_adj, w_adj_s, r2_adj, _ = train_ridge_probe(acts, y_adj)
        w_ld, _, r2_ld, _ = train_ridge_probe(acts, logit_diffs)

        print(f"    CV R²:  x={r2_x:+.3f}  mu={r2_mu:+.3f}  z={r2_z:+.3f}  "
              f"adj={r2_adj:+.3f}  logit_diff={r2_ld:+.3f}")

        # Scrambled control: shuffle y, re-probe
        rng = np.random.default_rng(0)
        y_adj_perm = rng.permutation(y_adj)
        _, _, r2_adj_scr, _ = train_ridge_probe(acts, y_adj_perm)
        zs_perm = rng.permutation(zs)
        _, _, r2_z_scr, _ = train_ridge_probe(acts, zs_perm)
        print(f"    Scrambled-y CV R² (should be ≤0):  z_perm={r2_z_scr:+.3f}  "
              f"adj_perm={r2_adj_scr:+.3f}")

        # Euclidean cosines between directions
        cos = {
            "cos_adj_z": metric_cosine(w_adj, w_z),
            "cos_adj_x": metric_cosine(w_adj, w_x),
            "cos_z_x":   metric_cosine(w_z, w_x),
            "cos_adj_mu": metric_cosine(w_adj, w_mu),
            "cos_z_mu":   metric_cosine(w_z, w_mu),
            "cos_x_mu":   metric_cosine(w_x, w_mu),
            "cos_z_ld":   metric_cosine(w_z, w_ld),
            "cos_adj_ld": metric_cosine(w_adj, w_ld),
            "cos_x_ld":   metric_cosine(w_x, w_ld),
        }

        print(f"    Euclid cos(adj,z)={cos['cos_adj_z']:+.3f}  cos(adj,x)={cos['cos_adj_x']:+.3f}  "
              f"cos(z,x)={cos['cos_z_x']:+.3f}  cos(x,mu)={cos['cos_x_mu']:+.3f}")
        print(f"    Euclid cos(z,logit_diff)={cos['cos_z_ld']:+.3f}  "
              f"cos(adj,logit_diff)={cos['cos_adj_ld']:+.3f}")

        # α/β decomposition: w_adj ≈ α·w_z_hat + β·w_x_hat (unit vectors)
        w_z_hat = w_z / (np.linalg.norm(w_z) + 1e-12)
        w_x_hat = w_x / (np.linalg.norm(w_x) + 1e-12)
        A = np.column_stack([w_z_hat, w_x_hat])
        w_adj_hat = w_adj / (np.linalg.norm(w_adj) + 1e-12)
        coefs, *_ = np.linalg.lstsq(A, w_adj_hat, rcond=None)
        alpha, beta = float(coefs[0]), float(coefs[1])
        recon = alpha * w_z_hat + beta * w_x_hat
        r2_recon = 1.0 - np.sum((w_adj_hat - recon) ** 2) / np.sum(w_adj_hat ** 2)
        alpha_frac = abs(alpha) / (abs(alpha) + abs(beta) + 1e-12)
        print(f"    w_adj ≈ {alpha:+.3f}·ẑ + {beta:+.3f}·x̂  "
              f"(α_frac={alpha_frac:.3f}, R²_recon={r2_recon:.3f})")

        # Sigma^{-1}-metric cosine: use activation covariance
        Sigma = np.cov(acts, rowvar=False)  # (d, d)
        # Tikhonov-regularized inverse for stability
        reg = 1e-3 * np.trace(Sigma) / d
        Sigma_reg = Sigma + reg * np.eye(d)
        # Use Cholesky to solve rather than explicit inv
        from scipy.linalg import cho_factor, cho_solve
        try:
            L = cho_factor(Sigma_reg, lower=True)
            def Sinv_apply(v): return cho_solve(L, v)
            cos_sigma_adj_z = float(np.dot(w_adj, Sinv_apply(w_z)) / (
                np.sqrt(np.dot(w_adj, Sinv_apply(w_adj)) * np.dot(w_z, Sinv_apply(w_z))) + 1e-24))
            cos_sigma_adj_x = float(np.dot(w_adj, Sinv_apply(w_x)) / (
                np.sqrt(np.dot(w_adj, Sinv_apply(w_adj)) * np.dot(w_x, Sinv_apply(w_x))) + 1e-24))
            cos_sigma_z_x = float(np.dot(w_z, Sinv_apply(w_x)) / (
                np.sqrt(np.dot(w_z, Sinv_apply(w_z)) * np.dot(w_x, Sinv_apply(w_x))) + 1e-24))
            cos["cos_sigma_adj_z"] = cos_sigma_adj_z
            cos["cos_sigma_adj_x"] = cos_sigma_adj_x
            cos["cos_sigma_z_x"]   = cos_sigma_z_x
            print(f"    Σ⁻¹ cos(adj,z)={cos_sigma_adj_z:+.3f}  "
                  f"cos(adj,x)={cos_sigma_adj_x:+.3f}  cos(z,x)={cos_sigma_z_x:+.3f}")
        except Exception as e:
            print(f"    [warn] Sigma^-1 cosine failed: {e}")

        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(PROBE_DIR / f"{condition}_{layer}_probes.npz",
                 w_x=w_x, w_z=w_z, w_adj=w_adj, w_ld=w_ld, w_mu=w_mu,
                 ids=np.array(ids))

        results[f"{condition}_{layer}"] = {
            "N": N, "d": d,
            "cv_r2_x": r2_x, "cv_r2_z": r2_z, "cv_r2_adj": r2_adj,
            "cv_r2_mu": r2_mu, "cv_r2_logit_diff": r2_ld,
            "cv_r2_z_scrambled": r2_z_scr, "cv_r2_adj_scrambled": r2_adj_scr,
            "alpha_zx_decomp": alpha, "beta_zx_decomp": beta,
            "alpha_frac": alpha_frac, "r2_zx_reconstruction": r2_recon,
            **cos,
        }

    return results


# -------------------- Phase 3: Geometry / PCA --------------------

def phase3_geometry(trials, logits_all) -> dict:
    """PCA of cell-mean activations. Check z-direction vs x-direction vs mu-direction."""
    print("\n" + "=" * 70)
    print("PHASE 3: GEOMETRY — PCA on cell-mean activations")
    print("=" * 70)
    results = {}

    for layer in LAYERS:
        try:
            acts, ids = load_activations("implicit", layer)
        except FileNotFoundError:
            continue

        # Compute mean per (x, mu) cell
        by_cell = defaultdict(list)
        for tid, row in zip(ids, acts):
            t = trials[tid]
            by_cell[(t["x"], t["mu"])].append(row)
        cells = sorted(by_cell.keys())
        means = np.stack([np.mean(by_cell[k], axis=0) for k in cells])  # (35, d)
        xs = np.array([k[0] for k in cells])
        mus = np.array([k[1] for k in cells])
        zs = (xs - mus) / 10.0

        # Also get logit-diff cell means
        log_by_cell = defaultdict(list)
        for tid in ids:
            t = trials[tid]
            ld = logits_all["implicit"][tid]["logit_diff"]
            log_by_cell[(t["x"], t["mu"])].append(ld)
        ld_means = np.array([np.mean(log_by_cell[k]) for k in cells])

        # Mean-center cells before PCA
        means_c = means - means.mean(axis=0, keepdims=True)
        pca = PCA(n_components=min(10, len(cells) - 1))
        pca.fit(means_c)
        proj = pca.transform(means_c)
        evr = pca.explained_variance_ratio_
        print(f"\n  [{layer}]  PCA of {len(cells)} cell means  "
              f"evr top-5 = {[f'{v*100:.1f}%' for v in evr[:5]]}")

        # Correlate each PC with z, x, mu, logit_diff
        corrs = {}
        for k in range(min(5, proj.shape[1])):
            corrs[f"PC{k+1}_vs_z"]  = float(np.corrcoef(proj[:, k], zs)[0, 1])
            corrs[f"PC{k+1}_vs_x"]  = float(np.corrcoef(proj[:, k], xs)[0, 1])
            corrs[f"PC{k+1}_vs_mu"] = float(np.corrcoef(proj[:, k], mus)[0, 1])
            corrs[f"PC{k+1}_vs_ld"] = float(np.corrcoef(proj[:, k], ld_means)[0, 1])
            print(f"    PC{k+1} ({evr[k]*100:.1f}%):  "
                  f"corr(z)={corrs[f'PC{k+1}_vs_z']:+.2f}  "
                  f"corr(x)={corrs[f'PC{k+1}_vs_x']:+.2f}  "
                  f"corr(mu)={corrs[f'PC{k+1}_vs_mu']:+.2f}  "
                  f"corr(ld)={corrs[f'PC{k+1}_vs_ld']:+.2f}")

        # Also compute direct "population" directions: ∂h/∂z (z-gradient), ∂h/∂x at fixed z,
        # ∂h/∂mu at fixed x. Done via linear regression in activation space.
        # means ≈ β_z · z + β_x_abs · x + intercept
        A = np.column_stack([np.ones_like(zs), zs])
        dh_dz = np.linalg.lstsq(A, means, rcond=None)[0][1]  # (d,)
        A = np.column_stack([np.ones_like(xs), xs])
        dh_dx = np.linalg.lstsq(A, means, rcond=None)[0][1]
        A = np.column_stack([np.ones_like(mus), mus])
        dh_dmu = np.linalg.lstsq(A, means, rcond=None)[0][1]
        A = np.column_stack([np.ones_like(zs), zs, xs])
        coef = np.linalg.lstsq(A, means, rcond=None)[0]
        dh_dz_partial = coef[1]  # z effect controlling for x
        dh_dx_partial = coef[2]

        dir_cos = {
            "cos(dh_dz, dh_dx)":          metric_cosine(dh_dz, dh_dx),
            "cos(dh_dz, dh_dmu)":         metric_cosine(dh_dz, dh_dmu),
            "cos(dh_dx, dh_dmu)":         metric_cosine(dh_dx, dh_dmu),
            "cos(dh_dz_partial, dh_dx)":  metric_cosine(dh_dz_partial, dh_dx),
            "cos(dh_dz_partial, dh_dx_partial)": metric_cosine(dh_dz_partial, dh_dx_partial),
        }
        print(f"    Cosines between gradient directions:")
        for name, v in dir_cos.items():
            print(f"      {name}: {v:+.3f}")

        results[layer] = {
            "n_cells": len(cells),
            "evr_top5": [float(v) for v in evr[:5]],
            "pc_correlations": corrs,
            "direction_cosines": dir_cos,
            "norm_dh_dz": float(np.linalg.norm(dh_dz)),
            "norm_dh_dx": float(np.linalg.norm(dh_dx)),
            "norm_dh_dmu": float(np.linalg.norm(dh_dmu)),
        }

        # Make figure
        if plt is not None:
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, (color_vals, label) in zip(axes, [(zs, "z"), (xs, "x (cm)"), (mus, "μ (cm)")]):
                sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_vals,
                                cmap="RdBu_r", s=80, edgecolors="k", linewidths=0.3)
                ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
                ax.set_title(f"implicit/{layer}: cell means by {label}")
                plt.colorbar(sc, ax=ax, label=label)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"pca_cell_means_{layer}.png", dpi=120)
            plt.close(fig)

            # Heatmap of logit_diff
            fig, ax = plt.subplots(figsize=(6, 4))
            x_vals = sorted(set(xs))
            mu_vals = sorted(set(mus))
            M = np.full((len(x_vals), len(mu_vals)), np.nan)
            for (x, mu), vals in log_by_cell.items():
                i = x_vals.index(x)
                j = mu_vals.index(mu)
                M[i, j] = np.mean(vals)
            im = ax.imshow(M, cmap="RdBu_r", aspect="auto", vmin=-np.nanmax(np.abs(M)),
                           vmax=np.nanmax(np.abs(M)))
            ax.set_xticks(range(len(mu_vals)))
            ax.set_xticklabels([f"{int(m)}" for m in mu_vals])
            ax.set_yticks(range(len(x_vals)))
            ax.set_yticklabels([f"{int(x)}" for x in x_vals])
            ax.set_xlabel("context μ (cm)")
            ax.set_ylabel("target x (cm)")
            ax.set_title("Mean logit(tall)-logit(short) per (x, μ) cell — implicit")
            for i in range(len(x_vals)):
                for j in range(len(mu_vals)):
                    v = M[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                                color="white" if abs(v) > 1.5 else "black", fontsize=8)
            plt.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"logit_diff_heatmap_implicit.png", dpi=120)
            plt.close(fig)

    return results


# -------------------- Phase 4: Controls --------------------

def phase4_controls(trials, logits_all) -> dict:
    """Compare implicit vs explicit vs zero_shot on matched (x, μ)."""
    print("\n" + "=" * 70)
    print("PHASE 4: CONTROLS — implicit vs explicit vs zero_shot")
    print("=" * 70)
    results = {}

    # For each (x, mu) point where both implicit and explicit exist, compare cell-mean logit_diff
    imp_cell = defaultdict(list)
    exp_cell = defaultdict(list)
    for tid, info in logits_all["implicit"].items():
        t = trials[tid]
        imp_cell[(t["x"], t["mu"])].append(info["logit_diff"])
    for tid, info in logits_all["explicit"].items():
        t = trials[tid]
        exp_cell[(t["x"], t["mu"])].append(info["logit_diff"])

    diffs_imp = []
    diffs_exp = []
    for k in imp_cell:
        if k in exp_cell:
            diffs_imp.append(np.mean(imp_cell[k]))
            diffs_exp.append(np.mean(exp_cell[k]))
    diffs_imp = np.array(diffs_imp)
    diffs_exp = np.array(diffs_exp)

    if len(diffs_imp) > 2:
        r = float(np.corrcoef(diffs_imp, diffs_exp)[0, 1])
        mean_imp = float(np.mean(diffs_imp))
        mean_exp = float(np.mean(diffs_exp))
        range_imp = float(np.max(diffs_imp) - np.min(diffs_imp))
        range_exp = float(np.max(diffs_exp) - np.min(diffs_exp))
        print(f"\n  Implicit vs Explicit (matched {len(diffs_imp)} cells):")
        print(f"    corr(implicit, explicit) = {r:+.3f}")
        print(f"    mean implicit={mean_imp:+.3f}  explicit={mean_exp:+.3f}")
        print(f"    range implicit={range_imp:.2f}  explicit={range_exp:.2f}")
        results["implicit_vs_explicit"] = {
            "n_cells": len(diffs_imp),
            "corr": r,
            "mean_implicit": mean_imp, "mean_explicit": mean_exp,
            "range_implicit": range_imp, "range_explicit": range_exp,
        }

    # Zero-shot: same x across all μ. Compare to implicit at each μ.
    zs_cell = {}
    for tid, info in logits_all["zero_shot"].items():
        t = trials[tid]
        zs_cell[t["x"]] = info["logit_diff"]
    print(f"\n  Zero-shot (no context) logit_diff by x:")
    for x in sorted(zs_cell):
        print(f"    x={int(x)}: {zs_cell[x]:+.3f}")
    results["zero_shot_by_x"] = {float(k): float(v) for k, v in zs_cell.items()}

    # Does context add signal beyond x alone? Compare implicit within-x variance (across μ)
    # to zero-shot point.
    for x in sorted(zs_cell):
        vals_across_mu = [np.mean(imp_cell[(x, mu)]) for mu in sorted(set(m for xx, m in imp_cell if xx == x))]
        if vals_across_mu:
            within_x_range = float(np.max(vals_across_mu) - np.min(vals_across_mu))
            print(f"    x={int(x)}: zero_shot={zs_cell[x]:+.3f}  "
                  f"implicit_range_over_mu={within_x_range:.2f}  "
                  f"(= context effect at this x)")

    return results


# -------------------- Phase 5: Σ⁻¹ metric deep-dive --------------------

def phase5_metrics(trials, logits_all) -> dict:
    """Compare Euclidean and Σ⁻¹ cosines. Look for primal-dual mismatch."""
    print("\n" + "=" * 70)
    print("PHASE 5: METRIC COMPARISON — Euclidean vs Σ⁻¹")
    print("=" * 70)
    # This is covered inside phase2 already; summarize here for clarity.
    # We expect that under Σ⁻¹ metric, cos(w_adj, w_z) goes UP from ~0
    # because the ~0 Euclid cosine is dominated by isotropic "junk" directions
    # that don't carry probe signal but dominate vector norm.
    return {"note": "Sigma^-1 cosines are in phase2 results under keys cos_sigma_*"}


# -------------------- Main --------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading trials from {V4_DIR} ...")
    trials = load_trials()
    logits_all = {c: load_logits(c) for c in CONDITIONS if (V4_DIR / f"{MODEL}_{c}_logits.jsonl").exists()}
    print(f"  {len(trials)} trials total, "
          f"{sum(len(v) for v in logits_all.values())} logit rows")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL,
        "v4_dir": str(V4_DIR),
        "n_trials": len(trials),
    }

    summary["phase1_behavioral"] = phase1_behavioral(trials, logits_all)
    summary["phase2_probes"] = phase2_probes(trials, logits_all, condition="implicit")
    # Also run phase-2 for explicit, since we have 35 points — much smaller but useful
    summary["phase2_probes_explicit"] = phase2_probes(trials, logits_all, condition="explicit")
    summary["phase3_geometry"] = phase3_geometry(trials, logits_all)
    summary["phase4_controls"] = phase4_controls(trials, logits_all)
    summary["phase5_metrics"] = phase5_metrics(trials, logits_all)

    out_path = OUT_DIR / "summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    dt = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {dt:.1f}s. Summary at {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
