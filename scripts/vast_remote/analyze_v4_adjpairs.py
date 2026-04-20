"""Cross-pair relativity analysis for v4_adjpairs extraction.

Reads results/v4_adjpairs/ and computes, per adjective pair:
  - Behavioral: relativity ratio (-slope_mu / slope_x), R² on z vs R² on x alone
  - Probe: CV R²(z) vs CV R²(x) — does the activation space carry z more than x?
  - α/β decomposition w_adj ≈ α·ẑ + β·x̂
  - Euclidean vs Σ⁻¹ cos(w_adj, w_z)

Then produces a summary table that compares all pairs side-by-side. The
core scientific claim: if relativity is a general property of gradable
adjectives, then for ALL pairs we should see:
  - relativity_ratio ≈ 1.0
  - CV R²(z) >> CV R²(x)
  - |α| > |β|  (i.e. adj direction is more z-aligned than x-aligned)

Absolute adjectives (if we ever add them) would show the opposite pattern.

Usage:
  python scripts/vast_remote/analyze_v4_adjpairs.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
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
ADJ_DIR = REPO / "results" / "v4_adjpairs"
OUT_DIR = REPO / "results" / "v4_adjpairs_analysis"
FIG_DIR = OUT_DIR / "figures"
MODEL = "e4b"
LAYERS = ["mid", "late"]

PAIR_NAMES = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
RELATIVE_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience"]
ABSOLUTE_PAIRS = ["bmi_abs"]


def load_trials() -> dict[str, dict]:
    path = ADJ_DIR / f"{MODEL}_trials.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            t = json.loads(line)
            out[t["id"]] = t
    return out


def train_ridge_probe(X, y, alpha=1.0):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X_s, y)
    w = m.coef_ / scaler.scale_
    cv_r2 = float(np.mean(cross_val_score(Ridge(alpha=alpha), X_s, y, cv=5, scoring="r2")))
    return w, cv_r2


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def analyze_pair(pair: str, trials, layer: str = "late") -> dict:
    """Full analysis for one pair at one layer."""
    # Load logits
    log_path = ADJ_DIR / f"{MODEL}_{pair}_implicit_logits.jsonl"
    if not log_path.exists():
        return {"pair": pair, "layer": layer, "error": "no implicit logits"}
    log_rows = [json.loads(l) for l in log_path.open()]

    # Load activations
    act_path = ADJ_DIR / f"{MODEL}_{pair}_implicit_{layer}.npz"
    if not act_path.exists():
        return {"pair": pair, "layer": layer, "error": f"no acts for {layer}"}
    with np.load(act_path, allow_pickle=True) as z:
        acts = z["activations"]
        ids = z["ids"].tolist()

    # Align logits with acts by id
    log_by_id = {r["id"]: r for r in log_rows}
    ordered_logits = [log_by_id[i] for i in ids]
    trial_by_id = {k: trials[k] for k in ids}

    xs = np.array([trial_by_id[i]["x"] for i in ids])
    mus = np.array([trial_by_id[i]["mu"] for i in ids])
    zs = np.array([trial_by_id[i]["z"] for i in ids])
    lds = np.array([r["logit_diff"] for r in ordered_logits])

    # Behavioral regressions
    # logit_diff ~ a + b*z
    if len(set(zs)) >= 2:
        A = np.column_stack([np.ones_like(zs), zs])
        coef, *_ = np.linalg.lstsq(A, lds, rcond=None)
        r2_z = 1.0 - np.sum((lds - A @ coef) ** 2) / np.sum((lds - lds.mean()) ** 2)
        slope_z = float(coef[1])
    else:
        r2_z, slope_z = 0.0, 0.0

    # logit_diff ~ a + b*x
    A = np.column_stack([np.ones_like(xs), xs])
    coef, *_ = np.linalg.lstsq(A, lds, rcond=None)
    r2_x_alone = 1.0 - np.sum((lds - A @ coef) ** 2) / np.sum((lds - lds.mean()) ** 2)
    slope_x_alone = float(coef[1])

    # logit_diff ~ a + b*x + c*mu
    A = np.column_stack([np.ones_like(xs), xs, mus])
    coef, *_ = np.linalg.lstsq(A, lds, rcond=None)
    r2_xmu = 1.0 - np.sum((lds - A @ coef) ** 2) / np.sum((lds - lds.mean()) ** 2)
    slope_x = float(coef[1])
    slope_mu = float(coef[2])
    relativity_ratio = -slope_mu / slope_x if abs(slope_x) > 1e-9 else None

    # Probes
    y_adj = (zs > 0).astype(np.float64)
    w_x, cv_r2_x = train_ridge_probe(acts, xs)
    w_z, cv_r2_z = train_ridge_probe(acts, zs)
    w_adj, cv_r2_adj = train_ridge_probe(acts, y_adj)
    w_ld, cv_r2_ld = train_ridge_probe(acts, lds)

    # α/β decomp
    w_z_hat = w_z / (np.linalg.norm(w_z) + 1e-12)
    w_x_hat = w_x / (np.linalg.norm(w_x) + 1e-12)
    w_adj_hat = w_adj / (np.linalg.norm(w_adj) + 1e-12)
    A_ab = np.column_stack([w_z_hat, w_x_hat])
    coefs_ab, *_ = np.linalg.lstsq(A_ab, w_adj_hat, rcond=None)
    alpha_c, beta_c = float(coefs_ab[0]), float(coefs_ab[1])
    recon = alpha_c * w_z_hat + beta_c * w_x_hat
    r2_recon = 1.0 - np.sum((w_adj_hat - recon) ** 2) / np.sum(w_adj_hat ** 2)
    alpha_frac = abs(alpha_c) / (abs(alpha_c) + abs(beta_c) + 1e-12)

    # Compare zero-shot vs implicit cell means (does context matter at all?)
    zs_log_path = ADJ_DIR / f"{MODEL}_{pair}_zero_shot_logits.jsonl"
    zero_by_x = {}
    if zs_log_path.exists():
        for l in zs_log_path.open():
            r = json.loads(l)
            zero_by_x[trials[r["id"]]["x"]] = r["logit_diff"]
    # For each x, range of implicit mean logit_diff across mu
    imp_by_xmu = defaultdict(list)
    for tid, ld in zip(ids, lds):
        t = trial_by_id[tid]
        imp_by_xmu[(t["x"], t["mu"])].append(ld)
    context_effect = {}
    for x in sorted(set(xs.tolist())):
        xkey = float(x)
        means_over_mu = [np.mean(imp_by_xmu[(xkey, m)])
                         for m in sorted(set(mus[xs == xkey].tolist()))]
        if means_over_mu:
            context_effect[xkey] = float(np.max(means_over_mu) - np.min(means_over_mu))

    return {
        "pair": pair,
        "layer": layer,
        "N_implicit": int(len(ids)),
        "logit_diff_mean": float(lds.mean()),
        "logit_diff_std": float(lds.std(ddof=1)),
        # Behavioral
        "slope_z": slope_z,
        "r2_on_z": float(r2_z),
        "r2_on_x_alone": float(r2_x_alone),
        "slope_x_alone": slope_x_alone,
        "r2_on_x_plus_mu": float(r2_xmu),
        "slope_x": slope_x,
        "slope_mu": slope_mu,
        "relativity_ratio": relativity_ratio,
        # Probes
        "cv_r2_x": cv_r2_x,
        "cv_r2_z": cv_r2_z,
        "cv_r2_adj": cv_r2_adj,
        "cv_r2_logit_diff": cv_r2_ld,
        # Decomp
        "cos_adj_z": cos(w_adj, w_z),
        "cos_adj_x": cos(w_adj, w_x),
        "cos_z_x":   cos(w_z, w_x),
        "cos_z_ld":  cos(w_z, w_ld),
        "alpha_zx_decomp": alpha_c,
        "beta_zx_decomp": beta_c,
        "alpha_frac": alpha_frac,
        "r2_decomp": r2_recon,
        # Context effect
        "zero_shot_by_x": {k: float(v) for k, v in zero_by_x.items()},
        "context_effect_by_x": context_effect,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    trials = load_trials()

    all_results = []
    for pair in PAIR_NAMES:
        for layer in LAYERS:
            print(f"\n--- {pair}/{layer} ---")
            res = analyze_pair(pair, trials, layer=layer)
            if "error" in res:
                print(f"  {res['error']}")
                continue
            print(f"  N={res['N_implicit']}  ld_mean={res['logit_diff_mean']:+.2f} "
                  f"sd={res['logit_diff_std']:.2f}")
            print(f"  Behavioral: R²(z)={res['r2_on_z']:.3f}  R²(x_alone)={res['r2_on_x_alone']:.3f}  "
                  f"R²(x+μ)={res['r2_on_x_plus_mu']:.3f}  relativity_ratio={res['relativity_ratio']}")
            print(f"  Probes:    R²(z)={res['cv_r2_z']:.3f}  R²(x)={res['cv_r2_x']:.3f}  "
                  f"R²(adj)={res['cv_r2_adj']:.3f}  R²(logit_diff)={res['cv_r2_logit_diff']:.3f}")
            print(f"  Decomp:    α={res['alpha_zx_decomp']:+.3f}  β={res['beta_zx_decomp']:+.3f}  "
                  f"α_frac={res['alpha_frac']:.3f}  R²_recon={res['r2_decomp']:.3f}")
            all_results.append(res)

    # Summary table
    print(f"\n{'='*100}")
    print("CROSS-PAIR SUMMARY (layer=late)")
    print(f"{'='*100}")
    print(f"{'pair':<12}{'kind':<6}{'R²(z)':>9}{'R²(x+μ)':>10}{'rel_ratio':>10} "
          f" {'CVR²(z)':>9}{'CVR²(x)':>9}{'α_frac':>9}{'cos(adj,z)':>12}")
    print("-" * 100)
    for r in all_results:
        if r["layer"] != "late":
            continue
        rr = r["relativity_ratio"]
        rr_str = f"{rr:+.2f}" if rr is not None else " N/A "
        kind = "REL" if r["pair"] in RELATIVE_PAIRS else "ABS"
        print(f"{r['pair']:<12}{kind:<6}{r['r2_on_z']:>9.3f}{r['r2_on_x_plus_mu']:>10.3f}{rr_str:>10} "
              f" {r['cv_r2_z']:>9.3f}{r['cv_r2_x']:>9.3f}{r['alpha_frac']:>9.3f}{r['cos_adj_z']:>12.3f}")

    # Core paper claim: relative pairs show relativity, absolute pairs don't
    rel_ratios_rel = [r["relativity_ratio"] for r in all_results
                      if r["layer"] == "late" and r["pair"] in RELATIVE_PAIRS
                      and r["relativity_ratio"] is not None]
    rel_ratios_abs = [r["relativity_ratio"] for r in all_results
                      if r["layer"] == "late" and r["pair"] in ABSOLUTE_PAIRS
                      and r["relativity_ratio"] is not None]
    if rel_ratios_rel and rel_ratios_abs:
        print(f"\nCORE CLAIM TEST:")
        print(f"  mean relativity_ratio, RELATIVE pairs ({len(rel_ratios_rel)}): "
              f"{np.mean(rel_ratios_rel):+.3f}")
        print(f"  mean relativity_ratio, ABSOLUTE pairs ({len(rel_ratios_abs)}): "
              f"{np.mean(rel_ratios_abs):+.3f}")
        print(f"  (expect REL → 1.0, ABS → 0.0 for clean relative/absolute distinction)")

    # Save JSON
    out_path = OUT_DIR / "summary.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # Make a bar chart for relativity_ratio across pairs
    if plt is not None:
        late_results = [r for r in all_results if r["layer"] == "late"]
        pairs = [r["pair"] for r in late_results]
        rr = [r["relativity_ratio"] if r["relativity_ratio"] is not None else 0
              for r in late_results]
        cv_z = [r["cv_r2_z"] for r in late_results]
        cv_x = [r["cv_r2_x"] for r in late_results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.bar(pairs, rr, color="steelblue")
        ax.axhline(1.0, color="red", linestyle="--", label="pure relativity (=1.0)")
        ax.set_ylabel("relativity ratio  −slope(μ) / slope(x)")
        ax.set_title("Relativity of gradable adjectives: behavioral regression")
        ax.legend()
        ax.set_ylim(-0.2, 2.0)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)

        ax = axes[1]
        w = 0.35
        x_pos = np.arange(len(pairs))
        ax.bar(x_pos - w/2, cv_z, w, label="probe R²(z)", color="steelblue")
        ax.bar(x_pos + w/2, cv_x, w, label="probe R²(x)", color="salmon")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pairs, rotation=30)
        ax.set_ylabel("cross-validated R²")
        ax.set_title("Probe decodability: z (relative) vs x (raw) by pair")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "relativity_across_pairs.png", dpi=120)
        plt.close(fig)
        print(f"Saved {FIG_DIR/'relativity_across_pairs.png'}")


if __name__ == "__main__":
    main()
