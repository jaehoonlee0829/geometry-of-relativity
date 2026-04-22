"""v10 P3 — SAE place-cell vs linear analysis (Gemma Scope, layer 20, 65k width).

For each SAE feature, plot its activation vs z over the 20 z-values (using
cell-mean activations) and fit:
  linear:   f(z) = a*z + b
  bump:     f(z) = A * exp(-(z-z0)²/(2σ²)) + c

Place-cell  ⇔  bump R² >> linear R²
Monotonic   ⇔  linear R² >> bump R²

Inputs : results/v10/gemma2_height_v10_residuals.npz
Outputs: results/v10/sae_feature_fits_L20.json
         figures/v10/sae_top10_z_profiles.png
         figures/v10/sae_linear_vs_bump_scatter.png

CPU-only. Downloads SAE on first run (~1.3 GB).
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "v10"
FIG = REPO / "figures" / "v10"
FIG.mkdir(parents=True, exist_ok=True)

SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_FILE = "layer_20/width_65k/average_l0_61/params.npz"
SAE_LAYER = 20


def jumprelu_encode(acts: np.ndarray, params: dict) -> np.ndarray:
    """JumpReLU SAE encode: relu(acts @ W_enc + b_enc) gated by threshold.

    Gemma Scope SAEs use JumpReLU with per-feature threshold:
      pre = acts @ W_enc + b_enc
      mask = pre > threshold
      feats = pre * mask
    """
    pre = acts @ params["W_enc"] + params["b_enc"]
    mask = pre > params["threshold"]
    return pre * mask


def gaussian_bump(z, A, z0, sigma, c):
    return A * np.exp(-((z - z0) ** 2) / (2 * sigma ** 2 + 1e-6)) + c


def fit_linear_r2(z, y) -> tuple[float, float, float]:
    a, b = np.polyfit(z, y, 1)
    yp = a * z + b
    ss_res = ((y - yp) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot < 1e-10:
        return float(a), float(b), float("nan")    # dead feature
    return float(a), float(b), float(1 - ss_res / ss_tot)


def fit_bump_r2(z, y) -> tuple[float, ...]:
    z0_init = z[np.argmax(np.abs(y - y.mean()))]
    A_init = y.max() - y.min()
    sigma_init = (z.max() - z.min()) / 4
    c_init = y.min()
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot < 1e-10:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    try:
        popt, _ = curve_fit(
            gaussian_bump, z, y,
            p0=[A_init, z0_init, sigma_init, c_init],
            bounds=([-1e3, z.min() - 0.5, 0.3, -1e3],   # σ_min raised 0.1→0.3
                    [1e3, z.max() + 0.5, 5.0, 1e3]),    # so σ floor isn't a
            maxfev=3000,                                  # single-z spike
        )
        yp = gaussian_bump(z, *popt)
        ss_res = ((y - yp) ** 2).sum()
        return *popt.tolist(), float(1 - ss_res / ss_tot)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")


def main() -> None:
    print("[P3] downloading SAE...", flush=True)
    sae_path = hf_hub_download(SAE_REPO, SAE_FILE,
                               token=os.environ.get("HF_TOKEN"))
    print(f"[P3]   {sae_path}", flush=True)
    sae_raw = np.load(sae_path)
    sae = {k: sae_raw[k] for k in sae_raw.files}
    for k, v in sae.items():
        print(f"[P3]   {k}: {v.shape} {v.dtype}", flush=True)

    # Load residuals at layer 20
    res = np.load(RES / "gemma2_height_v10_residuals.npz")
    acts_full = res["activations"]                      # (N, 26, 2304) fp16
    acts_L = acts_full[:, SAE_LAYER, :].astype(np.float32)
    z = res["z"].astype(np.float32)
    x = res["x"].astype(np.float32)
    print(f"[P3] acts at L{SAE_LAYER}: {acts_L.shape}", flush=True)

    # Encode (chunked to save memory; 4000 × 65k = 1 GB at fp32)
    print("[P3] encoding through SAE (chunked)...", flush=True)
    chunk = 256
    n_feats = sae["W_enc"].shape[1]
    feats = np.zeros((acts_L.shape[0], n_feats), dtype=np.float16)
    for i in range(0, acts_L.shape[0], chunk):
        f = jumprelu_encode(acts_L[i:i + chunk], sae)
        feats[i:i + chunk] = f.astype(np.float16)
    print(f"[P3] feats: {feats.shape}  active rate (mean nonzero)="
          f"{(feats != 0).mean():.4f}", flush=True)

    # Cell-means: 400 cells × n_feats
    print("[P3] computing cell-means in feature space...", flush=True)
    keys = sorted({(round(float(x[i]), 4), round(float(z[i]), 4))
                   for i in range(len(x))})
    M = len(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    cell_feats = np.zeros((M, n_feats), dtype=np.float32)
    cell_z = np.zeros(M, dtype=np.float32)
    counts = np.zeros(M, dtype=np.int32)
    for i in range(acts_L.shape[0]):
        k = (round(float(x[i]), 4), round(float(z[i]), 4))
        j = key_to_idx[k]
        cell_feats[j] += feats[i].astype(np.float32)
        cell_z[j] = k[1]
        counts[j] += 1
    cell_feats /= counts[:, None]

    # For each (x, z) cell we already have a single mean. To plot "feature vs z"
    # we average across the 20 x-values for each z, giving 20 points per feature.
    z_values = sorted(set(np.round(cell_z, 4).tolist()))
    print(f"[P3] {len(z_values)} unique z-values", flush=True)
    z_arr = np.array(z_values)
    by_z = np.zeros((len(z_values), n_feats), dtype=np.float32)
    counts_z = np.zeros(len(z_values), dtype=np.int32)
    z_to_idx = {round(zv, 4): k for k, zv in enumerate(z_values)}
    for j in range(M):
        zk = round(float(cell_z[j]), 4)
        by_z[z_to_idx[zk]] += cell_feats[j]
        counts_z[z_to_idx[zk]] += 1
    by_z /= counts_z[:, None]                # (20, n_feats)

    # Variance of by_z per feature → which features actually move with z?
    var_z = by_z.var(0)
    # Filter dead/near-dead features then take top by variance
    active_mask = var_z > 1e-3
    n_active = int(active_mask.sum())
    print(f"[P3] active features (var(z)>1e-3): {n_active} of {n_feats}",
          flush=True)
    n_top = min(200, n_active)
    top_idx = np.argsort(-var_z)[:n_top]
    print(f"[P3] fitting linear+bump on top {n_top} z-varying features",
          flush=True)

    fits = []
    for fid in top_idx:
        y = by_z[:, fid]
        a, b, lin_r2 = fit_linear_r2(z_arr, y)
        A, z0, sig, c, bump_r2 = fit_bump_r2(z_arr, y)
        if not (np.isfinite(lin_r2) and np.isfinite(bump_r2)):
            continue
        fits.append({
            "feature_id": int(fid),
            "var_over_z": float(var_z[fid]),
            "linear_a": a, "linear_b": b, "linear_r2": lin_r2,
            "bump_A": A, "bump_z0": z0, "bump_sigma": sig, "bump_c": c,
            "bump_r2": float(bump_r2),
            "place_cell_score": float(bump_r2 - lin_r2),
        })

    # Sort by linear R² descending → top-10 monotonic
    by_lin = sorted(fits, key=lambda r: -r["linear_r2"])
    by_bump = sorted(fits, key=lambda r: -(r["bump_r2"] - r["linear_r2"]))

    print(f"[P3] top-5 by linear R²:", flush=True)
    for r in by_lin[:5]:
        print(f"   feat {r['feature_id']:5d}  lin R²={r['linear_r2']:.3f}  "
              f"bump R²={r['bump_r2']:.3f}  Δ={r['bump_r2'] - r['linear_r2']:+.3f}",
              flush=True)
    print(f"[P3] top-5 by 'place-cell score' (bump - linear):", flush=True)
    for r in by_bump[:5]:
        print(f"   feat {r['feature_id']:5d}  lin R²={r['linear_r2']:.3f}  "
              f"bump R²={r['bump_r2']:.3f}  Δ={r['bump_r2'] - r['linear_r2']:+.3f}",
              flush=True)

    out = {
        "sae_repo": SAE_REPO, "sae_file": SAE_FILE, "sae_layer": SAE_LAYER,
        "n_features": int(n_feats),
        "n_top_fitted": int(n_top),
        "top_by_linear_r2": by_lin[:20],
        "top_by_place_cell_score": by_bump[:20],
        "fraction_features_active_anywhere": float((feats != 0).any(0).mean()),
    }
    json_path = RES / "sae_feature_fits_L20.json"
    json_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[P3] wrote {json_path}", flush=True)

    # ---- figure: top-10 features by linear R² (z profiles)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    for i, r in enumerate(by_lin[:10]):
        ax = axes[i // 5, i % 5]
        fid = r["feature_id"]
        y = by_z[:, fid]
        ax.plot(z_arr, y, "o-", color="C0")
        zfit = np.linspace(z_arr.min(), z_arr.max(), 50)
        ax.plot(zfit, r["linear_a"] * zfit + r["linear_b"], "--",
                color="C1", alpha=0.7, label=f"linear R²={r['linear_r2']:.2f}")
        ax.plot(zfit, gaussian_bump(zfit, r["bump_A"], r["bump_z0"],
                                    r["bump_sigma"], r["bump_c"]),
                ":", color="C2", alpha=0.7, label=f"bump R²={r['bump_r2']:.2f}")
        ax.set_title(f"feat {fid}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        if i % 5 == 0:
            ax.set_ylabel("SAE activation")
        if i // 5 == 1:
            ax.set_xlabel("z-score")
    fig.suptitle("Top 10 SAE features by linear R²(z) — L20, 65k width", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG / "sae_top10_z_profiles.png", dpi=120)
    plt.close()

    # ---- figure: linear vs bump R² scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    lin_r2s = np.array([r["linear_r2"] for r in fits])
    bump_r2s = np.array([r["bump_r2"] for r in fits])
    ax.scatter(lin_r2s, bump_r2s, s=20, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("linear R²(z)")
    ax.set_ylabel("Gaussian-bump R²(z)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"top-{n_top} z-varying SAE features at L20\n"
                 "above diagonal = place-cell-like, below = monotonic")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "sae_linear_vs_bump_scatter.png", dpi=120)
    plt.close()
    print(f"[P3] wrote 2 figures to {FIG}", flush=True)


if __name__ == "__main__":
    main()
