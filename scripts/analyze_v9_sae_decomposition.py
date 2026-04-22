"""v9 Priority 2: SAE feature decomposition of z via Gemma Scope.

Pipeline (all 8 pairs, late layer 20 on Gemma 2 2B):
  1. Load Gemma Scope SAE (width 65k, layer 20, avg_l0_61) — JumpReLU.
  2. Encode Grid B activations → sparse SAE coefficients.
  3. Per pair: find top-K features by |Pearson corr(feature, z)|.
  4. Cross-pair Jaccard overlap of top-K feature sets.
  5. Place-cell vs linear fit per top feature (Gaussian-bump vs z).
  6. Decompose primal_z (mean-diff) and probe_z (Ridge) in SAE basis.

The "primal steers 18× stronger than probe" gap motivates this — we expect
primal_z to load on a small number of features vs probe_z spread across many.

Outputs
  results/v9_gemma2/sae_z_features_per_pair.json
  results/v9_gemma2/sae_cross_pair_overlap.json
  results/v9_gemma2/sae_place_cell_vs_linear.json
  results/v9_gemma2/sae_primal_vs_probe_decomp.json
  figures/v9/sae_z_feature_overlap_heatmap.png
  figures/v9/sae_place_cell_profiles.png
  figures/v9/sae_primal_vs_probe_decomposition.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import PAIRS  # noqa: E402

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_PATH = "layer_20/width_65k/average_l0_61/params.npz"
TOP_K = 20
PLACE_GRID_POINTS = 40


def sae_encode(h: np.ndarray, W_enc, b_enc, b_dec, threshold) -> np.ndarray:
    """Gemma Scope JumpReLU encoder.

    h: (n, d_model)
    Returns: (n, n_features) — zero where pre-activations <= per-feature threshold.
    """
    x = h - b_dec  # center on decoder bias, as per Gemma Scope recipe
    pre = x @ W_enc + b_enc
    return np.where(pre > threshold, pre, 0.0).astype(np.float32)


def sae_project(v: np.ndarray, W_enc) -> np.ndarray:
    """Project a DIRECTION (not an activation) onto all 65k encoder columns.

    Pure linear response per feature, response_i = v @ W_enc[:, i].

    We deliberately drop b_dec and b_enc: those define the input offset and
    per-feature intercept of the encoder, which are appropriate for encoding
    an ACTIVATION h but not a direction v. Including b_enc would make the
    response magnitude dominated by the constant bias, obscuring which
    features v actually loads on.
    """
    return (v @ W_enc).astype(np.float32)


def load_pair_data(pair_name: str):
    """Return (acts, z, x, ids) for one pair at the late layer."""
    with np.load(RES_DIR / f"gemma2_{pair_name}_late.npz", allow_pickle=True) as z_:
        acts = z_["activations"].astype(np.float32)
        ids = z_["ids"].tolist()
    # Align trials
    trials = {}
    with (RES_DIR / "gemma2_trials.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["pair"] == pair_name:
                trials[t["id"]] = t
    zs = np.array([trials[i]["z"] for i in ids], dtype=np.float64)
    xs = np.array([trials[i]["x"] for i in ids], dtype=np.float64)
    return acts, zs, xs, ids


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def fit_gaussian_bump(zs: np.ndarray, fs: np.ndarray):
    """Fit f(z) ~ A * exp(-(z-z0)^2 / (2 σ^2)) + c via coarse grid + 1 refine.

    Returns dict with fit params and R^2. If feature is nearly flat or never
    fires, returns R^2 = 0 and flat fit.
    """
    if fs.std() < 1e-6 or (fs > 0).sum() < 5:
        return {"z0": 0.0, "sigma": 1.0, "A": 0.0, "c": float(fs.mean()),
                "r2": 0.0, "active_frac": float((fs > 0).mean())}
    z_grid = np.linspace(zs.min(), zs.max(), PLACE_GRID_POINTS)
    sig_grid = np.geomspace(0.2, 3.0, 20)
    best = {"r2": -np.inf, "z0": 0.0, "sigma": 1.0, "A": 0.0, "c": 0.0}
    fs_var = float(np.sum((fs - fs.mean()) ** 2))
    if fs_var < 1e-12:
        return {"z0": 0.0, "sigma": 1.0, "A": 0.0, "c": float(fs.mean()),
                "r2": 0.0, "active_frac": float((fs > 0).mean())}
    for z0 in z_grid:
        for sig in sig_grid:
            bump = np.exp(-0.5 * ((zs - z0) / sig) ** 2)
            # Solve f = A*bump + c in closed form (2 params)
            X = np.column_stack([bump, np.ones_like(bump)])
            coef, *_ = np.linalg.lstsq(X, fs, rcond=None)
            pred = X @ coef
            r2 = 1.0 - float(np.sum((fs - pred) ** 2)) / fs_var
            if r2 > best["r2"]:
                best = {"z0": float(z0), "sigma": float(sig),
                        "A": float(coef[0]), "c": float(coef[1]), "r2": r2}
    best["active_frac"] = float((fs > 0).mean())
    return best


def fit_linear(zs: np.ndarray, fs: np.ndarray):
    if fs.std() < 1e-6:
        return {"slope": 0.0, "intercept": float(fs.mean()), "r2": 0.0}
    X = np.column_stack([zs, np.ones_like(zs)])
    coef, *_ = np.linalg.lstsq(X, fs, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((fs - pred) ** 2))
    ss_tot = float(np.sum((fs - fs.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"slope": float(coef[0]), "intercept": float(coef[1]), "r2": r2}


def main():
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    print("Loading SAE:", SAE_REPO, SAE_PATH)
    sae_p = hf_hub_download(SAE_REPO, SAE_PATH)
    P = np.load(sae_p)
    W_enc = P["W_enc"].astype(np.float32)   # (d_model, n_features)
    W_dec = P["W_dec"].astype(np.float32)   # (n_features, d_model)
    b_enc = P["b_enc"].astype(np.float32)
    b_dec = P["b_dec"].astype(np.float32)
    thr   = P["threshold"].astype(np.float32)
    d_model, n_feat = W_enc.shape
    print(f"  d_model={d_model}  n_features={n_feat}")

    # Per-pair encoding + top features
    top_z_feats = {}     # pair -> list[int] (feature indices)
    top_z_corrs = {}     # pair -> list[float] matching
    place_vs_linear = {} # pair -> list[{"feat", "corr_z", "place_r2", "linear_r2"}]
    primal_decomp = {}   # pair -> dict with primal/probe feature responses
    for p in PAIRS:
        print(f"\n=== {p.name} ===")
        acts, zs, xs, ids = load_pair_data(p.name)

        coeffs = sae_encode(acts, W_enc, b_enc, b_dec, thr)   # (n, n_feat)
        active_mask = coeffs.any(axis=0)
        n_active = int(active_mask.sum())
        print(f"  n_prompts={acts.shape[0]}  active_features={n_active}/{n_feat}")

        # Correlations with z (only over active features)
        active_idx = np.where(active_mask)[0]
        corrs = np.zeros(n_feat, dtype=np.float32)
        for i in active_idx:
            corrs[i] = corrcoef_safe(coeffs[:, i], zs)
        order = np.argsort(-np.abs(corrs))
        top = order[:TOP_K]
        top_z_feats[p.name] = [int(i) for i in top]
        top_z_corrs[p.name] = [float(corrs[i]) for i in top]

        # Place-cell vs linear for top K
        pvl = []
        for i in top[:10]:   # analyze top 10 to keep plot manageable
            fs = coeffs[:, i]
            g = fit_gaussian_bump(zs, fs)
            ln = fit_linear(zs, fs)
            pvl.append({
                "feat": int(i),
                "corr_z": float(corrs[i]),
                "place_r2": g["r2"],
                "place_z0": g["z0"],
                "place_sigma": g["sigma"],
                "place_A": g["A"],
                "linear_r2": ln["r2"],
                "linear_slope": ln["slope"],
                "active_frac": g["active_frac"],
            })
        place_vs_linear[p.name] = pvl

        # Ridge probe_z (uncentered activations; match repo convention).
        rid = Ridge(alpha=1.0, fit_intercept=True).fit(acts, zs)
        probe_z = rid.coef_.astype(np.float32)     # (d_model,)

        # primal_z = mean(hi) - mean(lo)
        hi_mask = zs > 0
        lo_mask = zs < 0
        if hi_mask.sum() == 0 or lo_mask.sum() == 0:
            primal_z = np.zeros(d_model, dtype=np.float32)
        else:
            primal_z = (acts[hi_mask].mean(0) - acts[lo_mask].mean(0)).astype(np.float32)

        primal_resp = sae_project(primal_z, W_enc)
        probe_resp  = sae_project(probe_z,  W_enc)

        def participation_ratio(v: np.ndarray) -> float:
            """Effective number of features: (Σ p_i)^2 / Σ p_i^2 where p_i = proj^2.

            Ranges from 1 (one feature carries everything) to len(v) (uniform).
            Scale-invariant in v. Robust to the overcomplete-basis 'spread'.
            """
            p = v.astype(np.float64) ** 2
            s = p.sum()
            if s < 1e-18:
                return float(len(v))
            return float(s * s / (p * p).sum())

        def energy_frac_in_topK(v: np.ndarray, top_idx: list[int]) -> float:
            """Fraction of L2 energy of v's projection in given feature indices."""
            p = v.astype(np.float64) ** 2
            total = p.sum()
            if total < 1e-18:
                return 0.0
            return float(p[top_idx].sum() / total)

        top20 = top_z_feats[p.name]
        primal_decomp[p.name] = {
            "primal_norm":              float(np.linalg.norm(primal_z)),
            "probe_norm":               float(np.linalg.norm(probe_z)),
            "primal_top10_feats":       [int(i) for i in np.argsort(-np.abs(primal_resp))[:10]],
            "probe_top10_feats":        [int(i) for i in np.argsort(-np.abs(probe_resp))[:10]],
            # Participation ratio: effective # features carrying the response
            "primal_participation":     participation_ratio(primal_resp),
            "probe_participation":      participation_ratio(probe_resp),
            "probe_over_primal_PR":     participation_ratio(probe_resp) / max(participation_ratio(primal_resp), 1e-9),
            # Energy in the top-20 z-correlated features
            "primal_energy_in_top20z":  energy_frac_in_topK(primal_resp, top20),
            "probe_energy_in_top20z":   energy_frac_in_topK(probe_resp, top20),
            # Overlap between primal-dominant features and z-correlated features
            "primal_vs_z_jaccard_top10":
                float(len(set(int(i) for i in np.argsort(-np.abs(primal_resp))[:10])
                          & set(top_z_feats[p.name][:10])) / 10.0),
        }
        pd_ = primal_decomp[p.name]
        print(f"  participation   primal={pd_['primal_participation']:8.0f}   "
              f"probe={pd_['probe_participation']:8.0f}   "
              f"(probe/primal = {pd_['probe_over_primal_PR']:.2f}x)")
        print(f"  energy in top-20 z-feats   primal={pd_['primal_energy_in_top20z']:.3f}   "
              f"probe={pd_['probe_energy_in_top20z']:.3f}")
        print(f"  primal top-10 ∩ z top-10 = {int(pd_['primal_vs_z_jaccard_top10']*10)}/10")

    # Cross-pair Jaccard overlap
    names = [p.name for p in PAIRS]
    n = len(names)
    jacc = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            A = set(top_z_feats[a])
            B = set(top_z_feats[b])
            jacc[i, j] = len(A & B) / len(A | B) if A | B else 0.0
    print("\nCross-pair Jaccard of top-20 z-features (off-diag mean "
          f"= {jacc[~np.eye(n, dtype=bool)].mean():.3f})")

    # ----- write JSONs -----
    (RES_DIR / "sae_z_features_per_pair.json").write_text(json.dumps({
        "sae": f"{SAE_REPO}/{SAE_PATH}",
        "top_k": TOP_K,
        "features": top_z_feats,
        "corrs":    top_z_corrs,
    }, indent=2))
    (RES_DIR / "sae_cross_pair_overlap.json").write_text(json.dumps({
        "pair_order": names,
        "jaccard":    jacc.tolist(),
        "off_diag_mean": float(jacc[~np.eye(n, dtype=bool)].mean()),
    }, indent=2))
    (RES_DIR / "sae_place_cell_vs_linear.json").write_text(
        json.dumps(place_vs_linear, indent=2))
    (RES_DIR / "sae_primal_vs_probe_decomp.json").write_text(
        json.dumps(primal_decomp, indent=2))
    print("\nWrote 4 JSONs to results/v9_gemma2/")

    # ----- PLOTS -----
    # 1. Jaccard heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(jacc, cmap="viridis", vmin=0, vmax=jacc.max() if jacc.max() > 0 else 0.1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{jacc[i,j]:.2f}", ha="center", va="center",
                    color="white" if jacc[i, j] < 0.5 else "black", fontsize=7)
    ax.set_title(f"v9 P2 — cross-pair Jaccard, top-{TOP_K} z-correlated SAE features\n"
                 f"layer 20, width 65k, avg_l0_61  "
                 f"(off-diag mean = {jacc[~np.eye(n, dtype=bool)].mean():.3f})",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sae_z_feature_overlap_heatmap.png", dpi=140)
    print(f"  wrote {FIG_DIR}/sae_z_feature_overlap_heatmap.png")

    # 2. Place-cell profiles: top-3 z-features per pair, activation vs z
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    for ax, p in zip(axes.flat, PAIRS):
        acts, zs, xs, _ = load_pair_data(p.name)
        coeffs = sae_encode(acts, W_enc, b_enc, b_dec, thr)
        for rank, fi in enumerate(top_z_feats[p.name][:3]):
            ax.scatter(zs + 0.05 * rank, coeffs[:, fi], s=5, alpha=0.5,
                       label=f"#{fi} (r={top_z_corrs[p.name][rank]:+.2f})")
        ax.set_title(p.name, fontsize=10)
        ax.set_xlabel("z")
        ax.set_ylabel("SAE activation")
        ax.legend(fontsize=6)
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle("v9 P2 — top-3 z-correlated SAE features per pair (Gemma 2 2B, layer 20)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sae_place_cell_profiles.png", dpi=140)
    print(f"  wrote {FIG_DIR}/sae_place_cell_profiles.png")

    # 3. Primal-vs-probe: participation ratio + energy-in-z-features
    pair_names = [p.name for p in PAIRS]
    primal_pr = [primal_decomp[n]["primal_participation"] for n in pair_names]
    probe_pr  = [primal_decomp[n]["probe_participation"]  for n in pair_names]
    primal_ez = [primal_decomp[n]["primal_energy_in_top20z"] for n in pair_names]
    probe_ez  = [primal_decomp[n]["probe_energy_in_top20z"]  for n in pair_names]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    xs_b = np.arange(len(pair_names))
    a1.bar(xs_b - 0.2, primal_pr, width=0.4, label="primal_z")
    a1.bar(xs_b + 0.2, probe_pr,  width=0.4, label="probe_z (Ridge)")
    a1.set_xticks(xs_b); a1.set_xticklabels(pair_names, rotation=30)
    a1.set_ylabel("effective # features (participation ratio)")
    a1.set_title("Participation ratio (lower = concentrated)", fontsize=10)
    a1.legend()
    a2.bar(xs_b - 0.2, primal_ez, width=0.4, label="primal_z")
    a2.bar(xs_b + 0.2, probe_ez,  width=0.4, label="probe_z (Ridge)")
    a2.set_xticks(xs_b); a2.set_xticklabels(pair_names, rotation=30)
    a2.set_ylabel(f"L² energy fraction in top-{TOP_K} z-correlated features")
    a2.set_title("Energy concentrated in z-features (higher = more specific)", fontsize=10)
    a2.axhline(TOP_K / n_feat, color="k", ls=":", lw=0.8,
               label=f"uniform baseline ({TOP_K}/{n_feat})")
    a2.legend()
    fig.suptitle("v9 P2 — does primal_z concentrate on z-correlated SAE features more than probe_z?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sae_primal_vs_probe_decomposition.png", dpi=140)
    print(f"  wrote {FIG_DIR}/sae_primal_vs_probe_decomposition.png")


if __name__ == "__main__":
    main()
