"""v7 Priority 5: Park metric + Fisher at LOW-ENTROPY activations (clean grid).

v6 Block C tried to condition on |logit_diff| > p90 as a proxy for
"peaked softmax" but the script's own diagnostic showed |ld| is not a
good proxy. v7 fixes this by using SOFTMAX ENTROPY directly (saved per
prompt during Grid B extraction). Also recomputes Park metric with
Grid B probes.

Writes:
  results/v7_analysis/park_fisher_clean.json
  figures/v7/fisher_entropy_bins_clean.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_v4_adjpairs import PAIRS  # noqa: E402

V7 = REPO / "results" / "v7_xz_grid"
OUT = REPO / "results" / "v7_analysis"
OUT_FIG = REPO / "figures" / "v7"
W_U_PATH = REPO / "results" / "activations" / "e4b_W_U.npy"

K = 10
LOW_PCT = 10
HIGH_PCT = 90


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def fisher_matrix_gpu(h, W_U, jitter=1e-6):
    # W_U is fp64 here; cast h to fp64 for matmul compatibility
    h64 = h.double()
    logits = h64 @ W_U.T
    p = torch.softmax(logits, dim=0)
    scaled = p.unsqueeze(1) * W_U
    term1 = W_U.T @ scaled
    w_t_p = W_U.T @ p
    F = term1 - torch.outer(w_t_p, w_t_p)
    d = F.shape[0]
    F = F + jitter * torch.eye(d, dtype=F.dtype, device=F.device) * F.diagonal().mean().clamp_min(1e-12)
    return F


def cos_via_L(L, u, v):
    Fu = torch.cholesky_solve(u.unsqueeze(1), L).squeeze(1)
    Fv = torch.cholesky_solve(v.unsqueeze(1), L).squeeze(1)
    num = float((u * Fv).sum())
    du = float((u * Fu).sum()); dv = float((v * Fv).sum())
    return num / ((du**0.5)*(dv**0.5) + 1e-12)


def main():
    device = torch.device("cuda")
    print(f"Loading W_U → {device}", flush=True)
    W_U = torch.from_numpy(np.load(W_U_PATH)).to(device).double()
    V, d = W_U.shape
    print(f"  W_U {V}×{d}", flush=True)

    # Park metric
    gamma_mean = W_U.mean(0)
    Cov = ((W_U - gamma_mean).T @ (W_U - gamma_mean)) / V
    eigvals, eigvecs = torch.linalg.eigh(Cov)
    eigvals = torch.clamp(eigvals, min=eigvals.max().item() * 1e-8)
    inv_sqrt = (eigvecs * (1.0 / torch.sqrt(eigvals))) @ eigvecs.T
    inv_sqrt_np = inv_sqrt.cpu().numpy()

    trials_by_id = {json.loads(l)["id"]: json.loads(l) for l in (V7 / "e4b_trials.jsonl").open()}

    result = {}
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for p_obj in PAIRS:
        pn = p_obj.name
        print(f"\n[{pn}]", flush=True)
        npz = np.load(V7 / f"e4b_{pn}_late.npz", allow_pickle=True)
        acts = npz["activations"].astype(np.float64)
        ids = [str(s) for s in npz["ids"]]
        xs = np.array([trials_by_id[i]["x"] for i in ids])
        zs = np.array([trials_by_id[i]["z"] for i in ids])
        # Load entropy from logits jsonl
        logit_rec = {json.loads(l)["id"]: json.loads(l)
                     for l in (V7 / f"e4b_{pn}_logits.jsonl").open()}
        entropy = np.array([logit_rec[i]["entropy"] for i in ids])
        ld = np.array([logit_rec[i]["logit_diff"] for i in ids])

        w_z = Ridge(alpha=1.0).fit(acts, zs).coef_.astype(np.float64)
        w_x = Ridge(alpha=1.0).fit(acts, xs).coef_.astype(np.float64)
        w_ld = Ridge(alpha=1.0).fit(acts, ld).coef_.astype(np.float64)

        # Park cosines
        def ecos(u, v):
            return float(np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-12))

        def pcos(u, v):
            u2 = inv_sqrt_np @ u; v2 = inv_sqrt_np @ v
            return float(np.dot(u2, v2) / (np.linalg.norm(u2)*np.linalg.norm(v2) + 1e-12))

        # Fisher at entropy bins
        low_thr = np.percentile(entropy, LOW_PCT)
        high_thr = np.percentile(entropy, HIGH_PCT)
        low_idx = np.where(entropy <= low_thr)[0]
        high_idx = np.where(entropy >= high_thr)[0]
        K_each = min(K, len(low_idx), len(high_idx))
        low_sel = rng.choice(low_idx, K_each, replace=False)
        high_sel = rng.choice(high_idx, K_each, replace=False)

        w_z_t = torch.from_numpy(w_z).to(device).double()
        w_ld_t = torch.from_numpy(w_ld).to(device).double()

        out_bins = {}
        for bin_name, sel in [("low_entropy", low_sel), ("high_entropy", high_sel)]:
            cos_vals = []
            for i in sel:
                h = torch.from_numpy(acts[i].astype(np.float32)).to(device)
                try:
                    F = fisher_matrix_gpu(h, W_U)
                    L = torch.linalg.cholesky(F)
                    cos_vals.append(cos_via_L(L, w_z_t, w_ld_t))
                except Exception as e:
                    # Retry with larger jitter
                    try:
                        F = fisher_matrix_gpu(h, W_U, jitter=1e-3)
                        L = torch.linalg.cholesky(F)
                        cos_vals.append(cos_via_L(L, w_z_t, w_ld_t))
                    except Exception as e2:
                        print(f"    skip cell {i}: {e2}", flush=True)
            out_bins[bin_name] = {
                "n": len(cos_vals),
                "entropy_mean": float(entropy[sel].mean()),
                "F_cos_z_ld_mean": float(np.mean(cos_vals)),
                "F_cos_z_ld_std":  float(np.std(cos_vals)),
            }

        result[pn] = {
            "euclid_cos_z_ld": ecos(w_z, w_ld),
            "park_cos_z_ld":   pcos(w_z, w_ld),
            "euclid_cos_adj_z": ecos(Ridge(1.0).fit(acts, np.sign(zs)).coef_, w_z),
            "park_cos_adj_z":   pcos(Ridge(1.0).fit(acts, np.sign(zs)).coef_, w_z),
            "entropy_low_thr":  float(low_thr),
            "entropy_high_thr": float(high_thr),
            "bin_low":  out_bins["low_entropy"],
            "bin_high": out_bins["high_entropy"],
        }
        print(f"  entropy bins: low={low_thr:.3f}  high={high_thr:.3f}")
        print(f"  E cos(z,ld)={result[pn]['euclid_cos_z_ld']:+.3f}  "
              f"Park={result[pn]['park_cos_z_ld']:+.3f}  "
              f"F⁻¹@low={result[pn]['bin_low']['F_cos_z_ld_mean']:+.3f}  "
              f"F⁻¹@high={result[pn]['bin_high']['F_cos_z_ld_mean']:+.3f}")

    (OUT / "park_fisher_clean.json").write_text(json.dumps(result, indent=2))

    # Figure
    fig, ax = plt.subplots(figsize=(13, 5))
    names = [p.name for p in PAIRS]
    xpos = np.arange(len(names))
    euc = [abs(result[n]["euclid_cos_z_ld"]) for n in names]
    park = [abs(result[n]["park_cos_z_ld"]) for n in names]
    f_low = [abs(result[n]["bin_low"]["F_cos_z_ld_mean"]) for n in names]
    f_high = [abs(result[n]["bin_high"]["F_cos_z_ld_mean"]) for n in names]
    ax.bar(xpos-0.3, euc, 0.2, label="Euclidean")
    ax.bar(xpos-0.1, park, 0.2, label="Park Cov(W_U)^{-1/2}")
    ax.bar(xpos+0.1, f_low, 0.2, label="F⁻¹ at low-entropy acts")
    ax.bar(xpos+0.3, f_high, 0.2, label="F⁻¹ at high-entropy acts")
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("|cos(w_z, w_ld)|")
    ax.set_title("v7 clean-grid: does Fisher at truly low-entropy activations show alignment?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fisher_entropy_bins_clean.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {OUT/'park_fisher_clean.json'} + figure")


if __name__ == "__main__":
    main()
