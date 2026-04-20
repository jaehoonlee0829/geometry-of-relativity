"""Day-4 probe training + α/β decomposition analysis.

Trains three ridge probes on activation vectors:
  w_adj: predicts z-score sign (positive = "tall"/"rich", negative = "short"/"poor")
  w_x:   predicts raw attribute value x
  w_z:   predicts context-normalized z-score

Then decomposes w_adj ≈ α·w_z + β·w_x via OLS regression, and reports
Euclidean cosine similarities between all probe directions.

Usage:
    python scripts/run_probe_day4.py [--model e4b|g31b] [--domain height|wealth]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
ACT_DIR = REPO / "results" / "activations"
PROMPTS = REPO / "data_gen" / "prompts_v2.jsonl"


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def load_trials(domain: str) -> list[dict]:
    """Load v2 trials for a given domain from prompts_v2.jsonl."""
    trials = []
    with PROMPTS.open() as f:
        for line in f:
            t = json.loads(line)
            if t["domain"] == domain:
                trials.append(t)
    return trials


def load_activations(model: str, domain: str, layer: str) -> tuple[np.ndarray, list[str]]:
    """Load activation matrix and trial IDs for a (model, domain, layer) cell."""
    path = ACT_DIR / f"{model}_{domain}_{layer}.npz"
    with np.load(path, allow_pickle=True) as z:
        acts = z["activations"]  # (N, d)
        ids = z["ids"].tolist()
    return acts, ids


def train_ridge_probe(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Train a ridge regression probe, return the weight vector w (d,)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_s, y)
    # Transform w back to original space: w_orig = w_scaled / scale
    w = model.coef_ / scaler.scale_
    return w


def analyze_cell(
    model: str, domain: str, layer: str, trials: list[dict]
) -> dict:
    """Run full probe analysis for one (model, domain, layer) cell."""
    acts, ids = load_activations(model, domain, layer)
    N, d = acts.shape

    # Build trial lookup
    trial_by_id = {t["id"]: t for t in trials}

    # Align trials with activation order
    xs = np.array([trial_by_id[tid]["x"] for tid in ids], dtype=np.float64)
    zs = np.array([trial_by_id[tid]["z"] for tid in ids], dtype=np.float64)
    mus = np.array([trial_by_id[tid]["mu"] for tid in ids], dtype=np.float64)

    # y_adj: sign of z-score (1 = tall/rich, 0 = short/poor)
    y_adj = (zs > 0).astype(np.float64)

    # Train three probes
    w_x = train_ridge_probe(acts, xs)
    w_z = train_ridge_probe(acts, zs)
    w_adj = train_ridge_probe(acts, y_adj)

    # Normalize for cosine comparisons
    w_x_n = w_x / np.linalg.norm(w_x)
    w_z_n = w_z / np.linalg.norm(w_z)
    w_adj_n = w_adj / np.linalg.norm(w_adj)

    # Euclidean cosines
    cos_adj_z = cosine(w_adj, w_z)
    cos_adj_x = cosine(w_adj, w_x)
    cos_z_x = cosine(w_z, w_x)

    # α/β decomposition: regress w_adj onto [w_z, w_x]
    # w_adj ≈ α·w_z + β·w_x
    # This is a 2-variable OLS in d-dimensional space
    A = np.column_stack([w_z_n, w_x_n])  # (d, 2)
    coeffs, residuals, _, _ = np.linalg.lstsq(A, w_adj_n, rcond=None)
    alpha_coeff = float(coeffs[0])
    beta_coeff = float(coeffs[1])

    # Normalized fractions
    abs_sum = abs(alpha_coeff) + abs(beta_coeff)
    alpha_frac = abs(alpha_coeff) / abs_sum if abs_sum > 1e-12 else 0.5
    beta_frac = abs(beta_coeff) / abs_sum if abs_sum > 1e-12 else 0.5

    # Reconstruction quality: how much of w_adj is explained by the 2D subspace?
    w_adj_reconstructed = alpha_coeff * w_z_n + beta_coeff * w_x_n
    r2 = 1.0 - np.sum((w_adj_n - w_adj_reconstructed) ** 2) / np.sum(w_adj_n ** 2)

    # Probe quality: cross-validated R² for each probe
    scaler = StandardScaler()
    X_s = scaler.fit_transform(acts)
    cv_r2_x = float(np.mean(cross_val_score(Ridge(alpha=1.0), X_s, xs, cv=5, scoring="r2")))
    cv_r2_z = float(np.mean(cross_val_score(Ridge(alpha=1.0), X_s, zs, cv=5, scoring="r2")))
    cv_r2_adj = float(np.mean(cross_val_score(Ridge(alpha=1.0), X_s, y_adj, cv=5, scoring="r2")))

    return {
        "model": model,
        "domain": domain,
        "layer": layer,
        "n_trials": N,
        "hidden_dim": d,
        # Cosines
        "cos_adj_z": cos_adj_z,
        "cos_adj_x": cos_adj_x,
        "cos_z_x": cos_z_x,
        # α/β decomposition
        "alpha": alpha_coeff,
        "beta": beta_coeff,
        "alpha_frac": alpha_frac,
        "beta_frac": beta_frac,
        "decomposition_r2": r2,
        # Probe CV R²
        "cv_r2_x": cv_r2_x,
        "cv_r2_z": cv_r2_z,
        "cv_r2_adj": cv_r2_adj,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="Model prefix (e4b or g31b). Default: both.")
    ap.add_argument("--domain", default=None, help="Domain (height or wealth). Default: both.")
    ap.add_argument("--layer", default=None, help="Layer (early/mid/late/final). Default: all.")
    args = ap.parse_args()

    models = [args.model] if args.model else ["e4b", "g31b"]
    domains = [args.domain] if args.domain else ["height", "wealth"]
    layers = [args.layer] if args.layer else ["early", "mid", "late", "final"]

    # Skip g31b final layer (post-norm, std≈0.06, uninformative)
    skip = {("g31b", "final")}

    all_results = []

    for model in models:
        for domain in domains:
            trials = load_trials(domain)
            for layer in layers:
                if (model, layer) in skip:
                    print(f"\n--- SKIP {model}/{domain}/{layer} (post-norm) ---")
                    continue

                act_path = ACT_DIR / f"{model}_{domain}_{layer}.npz"
                if not act_path.exists():
                    print(f"\n--- SKIP {model}/{domain}/{layer} (file missing) ---")
                    continue

                print(f"\n{'='*60}")
                print(f"  {model} / {domain} / {layer}")
                print(f"{'='*60}")

                result = analyze_cell(model, domain, layer, trials)
                all_results.append(result)

                print(f"  N={result['n_trials']}, d={result['hidden_dim']}")
                print(f"  Probe CV R²:  x={result['cv_r2_x']:.3f}  "
                      f"z={result['cv_r2_z']:.3f}  adj={result['cv_r2_adj']:.3f}")
                print(f"  Cosines:  cos(adj,z)={result['cos_adj_z']:+.3f}  "
                      f"cos(adj,x)={result['cos_adj_x']:+.3f}  "
                      f"cos(z,x)={result['cos_z_x']:+.3f}")
                print(f"  α/β decomposition:  α={result['alpha']:+.4f}  "
                      f"β={result['beta']:+.4f}")
                print(f"  α fraction: {result['alpha_frac']:.3f}  "
                      f"β fraction: {result['beta_frac']:.3f}")
                print(f"  Decomposition R²: {result['decomposition_r2']:.3f}")

    # Save results
    out_path = REPO / "results" / "day4_probe_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'model':<6} {'domain':<8} {'layer':<6} "
          f"{'R²(z)':<8} {'R²(x)':<8} {'R²(adj)':<8} "
          f"{'cos(a,z)':<10} {'cos(a,x)':<10} "
          f"{'α_frac':<8} {'β_frac':<8} {'dec_R²':<8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<6} {r['domain']:<8} {r['layer']:<6} "
              f"{r['cv_r2_z']:<8.3f} {r['cv_r2_x']:<8.3f} {r['cv_r2_adj']:<8.3f} "
              f"{r['cos_adj_z']:<+10.3f} {r['cos_adj_x']:<+10.3f} "
              f"{r['alpha_frac']:<8.3f} {r['beta_frac']:<8.3f} {r['decomposition_r2']:<8.3f}")


if __name__ == "__main__":
    main()
