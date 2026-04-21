"""v6 Block A: Park et al.'s causal inner product via Cov(W_U)⁻¹.

v5 Exp 3a used Σ⁻¹ from activation covariance (N=750, d=2560 → rank-deficient,
regularizer dominated, so Σ⁻¹ ≈ c·I and the metric was uninformative). That's
NOT what Park et al. did.

Park's method: Σ = Cov(W_U), where W_U is the unembedding matrix (V=262144,
d=2560 for E4B). N/d = 102 → well-conditioned, full rank, no regularizer
needed. Define inv_sqrt_Σ = Σ^{-1/2} via eigendecomposition. The causal cosine
between two probe directions u, v is then
    cos_Park(u, v) = (u_W · v_W) / (||u_W|| · ||v_W||)
where u_W = inv_sqrt_Σ · u.

Writes:
  results/v4_adjpairs_analysis/park_cov_wu_cosines.json
  figures/v4_adjpairs/park_vs_euclid_cosines.png
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent

ADJPAIRS = REPO / "results" / "v4_adjpairs"
OUT_JSON = REPO / "results" / "v4_adjpairs_analysis"
OUT_FIG = REPO / "figures" / "v4_adjpairs"
OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

PAIRS = ["height", "age", "experience", "size", "speed", "wealth", "weight", "bmi_abs"]
LAYERS = ["mid", "late"]
W_U_PATH = REPO / "results" / "activations" / "e4b_W_U.npy"


def load_pair_layer(pair: str, layer: str):
    trials_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in (ADJPAIRS / "e4b_trials.jsonl").open()}
    ld_by_id = {r["id"]: r["logit_diff"]
                for r in map(json.loads, (ADJPAIRS / f"e4b_{pair}_implicit_logits.jsonl").open())}
    npz = np.load(ADJPAIRS / f"e4b_{pair}_implicit_{layer}.npz", allow_pickle=True)
    acts = npz["activations"].astype(np.float64)
    ids = [str(s) for s in npz["ids"]]
    xs, zs, ld = [], [], []
    for i in ids:
        t = trials_by_id[i]
        xs.append(t["x"])
        zs.append(t["z"])
        ld.append(ld_by_id[i])
    return acts, np.array(xs), np.array(zs), np.array(ld)


def train_probe(X, y, alpha=1.0):
    return Ridge(alpha=alpha).fit(X, y).coef_.astype(np.float64)


def eucl_cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def metric_cos(u, v, inv_sqrt):
    """cos under the Park metric: apply inv_sqrt to both sides, then standard cos."""
    u2 = inv_sqrt @ u
    v2 = inv_sqrt @ v
    return float(np.dot(u2, v2) / (np.linalg.norm(u2) * np.linalg.norm(v2) + 1e-12))


def main() -> None:
    assert W_U_PATH.exists(), f"missing {W_U_PATH} — run export_W_U.py e4b first"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading W_U → {device}", flush=True)
    W_U = torch.from_numpy(np.load(W_U_PATH)).to(device).double()   # (V, d) fp64
    V, d = W_U.shape
    print(f"  W_U shape={V}×{d}  N/d={V/d:.1f}", flush=True)

    # Park's Cov(W_U): center by mean row, compute d×d covariance
    t0 = time.time()
    gamma_mean = W_U.mean(dim=0)
    centered = W_U - gamma_mean
    Cov_wu = (centered.T @ centered) / V    # (d, d)
    print(f"  Cov(W_U) computed in {time.time()-t0:.1f}s  cond #={torch.linalg.cond(Cov_wu).item():.3e}",
          flush=True)

    # Eigendecomposition (symmetric → eigh)
    t0 = time.time()
    eigvals, eigvecs = torch.linalg.eigh(Cov_wu)
    # Clamp small eigvalues for numerical safety — but WITHOUT regularization
    # pushing toward identity. With V/d=102 this clamp barely moves anything.
    eigvals = torch.clamp(eigvals, min=eigvals.max().item() * 1e-8)
    inv_sqrt_Sigma = (eigvecs * (1.0 / torch.sqrt(eigvals))) @ eigvecs.T    # (d, d)
    print(f"  eigendecomp in {time.time()-t0:.1f}s  "
          f"min/max eigval: {eigvals.min().item():.3e} / {eigvals.max().item():.3e}",
          flush=True)

    inv_sqrt = inv_sqrt_Sigma.cpu().numpy()
    # Save the metric itself (2560×2560 fp64 = 50 MB — don't, too big). Just save
    # eigenvalue spectrum for diagnostics.
    spectrum = eigvals.cpu().numpy()

    # Train probes per pair × layer; measure Euclid + Park cosines + cross-pair
    per_pair: dict = {}
    w_z_store: dict = {}
    w_ld_store: dict = {}
    for pair in PAIRS:
        per_pair[pair] = {}
        for layer in LAYERS:
            acts, xs, zs, ld = load_pair_layer(pair, layer)
            y_adj = np.sign(zs)
            y_adj[y_adj == 0] = 1.0
            w_x = train_probe(acts, xs)
            w_z = train_probe(acts, zs)
            w_adj = train_probe(acts, y_adj)
            w_ld = train_probe(acts, ld)
            w_z_store[(pair, layer)] = w_z
            w_ld_store[(pair, layer)] = w_ld

            per_pair[pair][layer] = {
                "euclid": {
                    "cos_adj_z": eucl_cos(w_adj, w_z),
                    "cos_adj_x": eucl_cos(w_adj, w_x),
                    "cos_z_x":   eucl_cos(w_z, w_x),
                    "cos_z_ld":  eucl_cos(w_z, w_ld),
                },
                "park_cov_wu": {
                    "cos_adj_z": metric_cos(w_adj, w_z, inv_sqrt),
                    "cos_adj_x": metric_cos(w_adj, w_x, inv_sqrt),
                    "cos_z_x":   metric_cos(w_z, w_x, inv_sqrt),
                    "cos_z_ld":  metric_cos(w_z, w_ld, inv_sqrt),
                },
                "n": int(acts.shape[0]),
            }
            print(f"  [{pair}/{layer}]  "
                  f"cos(z,ld): E={per_pair[pair][layer]['euclid']['cos_z_ld']:+.3f}  "
                  f"Park={per_pair[pair][layer]['park_cov_wu']['cos_z_ld']:+.3f}",
                  flush=True)

    # Cross-pair: how aligned are per-pair w_z's under the Park metric vs Euclidean?
    cross_layer = "late"
    pnames = PAIRS
    n = len(pnames)
    cos_eucl = np.zeros((n, n))
    cos_park = np.zeros((n, n))
    for i, p1 in enumerate(pnames):
        for j, p2 in enumerate(pnames):
            u = w_z_store[(p1, cross_layer)]
            v = w_z_store[(p2, cross_layer)]
            cos_eucl[i, j] = eucl_cos(u, v)
            cos_park[i, j] = metric_cos(u, v, inv_sqrt)

    # Summary stats
    offdiag_mask = ~np.eye(n, dtype=bool)
    mean_eucl = float(np.mean(np.abs(cos_eucl[offdiag_mask])))
    mean_park = float(np.mean(np.abs(cos_park[offdiag_mask])))
    print(f"\nCross-pair |cos(w_z_i, w_z_j)|  late layer:")
    print(f"  Euclidean: {mean_eucl:.3f}")
    print(f"  Park:      {mean_park:.3f}")

    result = {
        "method": "Park et al. causal inner product = Cov(W_U)^{-1/2}",
        "wu_shape": [int(V), int(d)],
        "wu_nd_ratio": V / d,
        "eigvalue_min": float(eigvals.min().item()),
        "eigvalue_max": float(eigvals.max().item()),
        "eigvalue_spectrum_top10": spectrum[-10:].tolist(),
        "eigvalue_spectrum_bot10": spectrum[:10].tolist(),
        "condition_number": float(torch.linalg.cond(Cov_wu).item()),
        "per_pair": per_pair,
        "cross_pair_wz_cosine_late": {
            "pairs": pnames,
            "cos_euclidean": cos_eucl.tolist(),
            "cos_park": cos_park.tolist(),
            "mean_offdiag_abs_euclid": mean_eucl,
            "mean_offdiag_abs_park": mean_park,
        },
    }
    (OUT_JSON / "park_cov_wu_cosines.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON/'park_cov_wu_cosines.json'}", flush=True)

    # Plot: per-pair |cos(w_z, w_ld)| — Euclid vs Park
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, layer in zip(axes, LAYERS):
        xpos = np.arange(len(PAIRS))
        eu = [abs(per_pair[p][layer]["euclid"]["cos_z_ld"]) for p in PAIRS]
        pa = [abs(per_pair[p][layer]["park_cov_wu"]["cos_z_ld"]) for p in PAIRS]
        ax.bar(xpos - 0.2, eu, 0.4, label="Euclidean")
        ax.bar(xpos + 0.2, pa, 0.4, label="Park (Cov(W_U)^{-1/2})")
        ax.set_xticks(xpos); ax.set_xticklabels(PAIRS, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"|cos(w_z, w_ld)| at layer={layer}")
        ax.set_ylabel("|cos|"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Does Park's causal inner product resolve the cos(w_z, w_ld) mystery?")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "park_vs_euclid_cosines.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'park_vs_euclid_cosines.png'}", flush=True)


if __name__ == "__main__":
    main()
