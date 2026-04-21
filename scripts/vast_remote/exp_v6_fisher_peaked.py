"""v6 Block C: Fisher at peaked activations.

v5 Exp 3b showed F⁻¹ ≈ Euclidean at CELL-MEAN activations. Red-team flagged:
cell means sit in high-entropy softmax region where F(h) ≈ (1/V)·I. Test F at
PEAKED activations (confident predictions, |logit_diff| > 3) where p is
concentrated and F should be highly anisotropic.

For each pair × layer, split implicit activations by |logit_diff|:
  peaked: |ld| > peaked_threshold
  flat:   |ld| < flat_threshold

Sample K activations from each bin, compute F(h), F⁻¹·w, Fisher cosines.
Compare cos_F⁻¹(w_z, w_ld) at peaked vs flat.

Writes:
  results/v4_adjpairs_analysis/fisher_peaked_vs_flat.json
  figures/v4_adjpairs/fisher_peaked_vs_flat.png
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

ADJPAIRS = REPO / "results" / "v4_adjpairs"
OUT_JSON = REPO / "results" / "v4_adjpairs_analysis"
OUT_FIG = REPO / "figures" / "v4_adjpairs"

PAIRS = ["height", "age", "experience", "size", "speed", "wealth", "weight", "bmi_abs"]
LAYERS = ["mid", "late"]
K_SAMPLES = 10         # activations per bin per pair × layer
PEAKED_PCT = 90        # top 10% |ld|
FLAT_PCT = 30          # bottom 30% |ld|
W_U_PATH = REPO / "results" / "activations" / "e4b_W_U.npy"


def fisher_matrix_gpu(h, W_U, jitter=1e-6):
    logits = h @ W_U.T
    p = torch.softmax(logits.double(), dim=0)
    W_U64 = W_U.double()
    scaled = p.unsqueeze(1) * W_U64
    term1 = W_U64.T @ scaled
    w_t_p = W_U64.T @ p
    F = term1 - torch.outer(w_t_p, w_t_p)
    d = F.shape[0]
    F = F + jitter * torch.eye(d, dtype=F.dtype, device=F.device) * F.diagonal().mean().clamp_min(1e-12)
    return F


def fisher_cos_gpu(L, u, v):
    Finv_u = torch.cholesky_solve(u.unsqueeze(1), L).squeeze(1)
    Finv_v = torch.cholesky_solve(v.unsqueeze(1), L).squeeze(1)
    num = float((u * Finv_v).sum())
    du = float((u * Finv_u).sum())
    dv = float((v * Finv_v).sum())
    if du <= 0 or dv <= 0:
        return float("nan")
    return num / ((du ** 0.5) * (dv ** 0.5))


def softmax_entropy_gpu(h, W_U):
    logits = h @ W_U.T
    p = torch.softmax(logits.double(), dim=0)
    return float(-(p * torch.log(p.clamp_min(1e-30))).sum())


def load_pair_layer(pair, layer):
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


def main():
    assert W_U_PATH.exists()
    device = torch.device("cuda")
    W_U = torch.from_numpy(np.load(W_U_PATH).astype(np.float32)).to(device)
    print(f"W_U shape={tuple(W_U.shape)}  device={W_U.device}", flush=True)

    rng = np.random.default_rng(0)
    result: dict = {"K_samples": K_SAMPLES, "peaked_pct": PEAKED_PCT, "flat_pct": FLAT_PCT,
                    "per_pair": {}}

    for pair in PAIRS:
        result["per_pair"][pair] = {}
        for layer in LAYERS:
            acts, xs, zs, ld = load_pair_layer(pair, layer)
            y_adj = np.sign(zs); y_adj[y_adj == 0] = 1.0
            w_x = train_probe(acts, xs)
            w_z = train_probe(acts, zs)
            w_ld = train_probe(acts, ld)

            abs_ld = np.abs(ld)
            peaked_thr = np.percentile(abs_ld, PEAKED_PCT)
            flat_thr   = np.percentile(abs_ld, FLAT_PCT)
            peaked_idx = np.where(abs_ld >= peaked_thr)[0]
            flat_idx = np.where(abs_ld <= flat_thr)[0]
            if len(peaked_idx) < K_SAMPLES or len(flat_idx) < K_SAMPLES:
                print(f"  [{pair}/{layer}] insufficient samples, skipping")
                continue
            peaked_sel = rng.choice(peaked_idx, K_SAMPLES, replace=False)
            flat_sel = rng.choice(flat_idx, K_SAMPLES, replace=False)

            w_z_t = torch.from_numpy(w_z).to(device).double()
            w_ld_t = torch.from_numpy(w_ld).to(device).double()
            w_x_t = torch.from_numpy(w_x).to(device).double()

            out_bins = {}
            for bin_name, sel in [("peaked", peaked_sel), ("flat", flat_sel)]:
                cell_cos_zld, cell_cos_zx = [], []
                entropies = []
                for i in sel:
                    h = torch.from_numpy(acts[i].astype(np.float32)).to(device)
                    entropies.append(softmax_entropy_gpu(h, W_U))
                    try:
                        F = fisher_matrix_gpu(h, W_U)
                        L = torch.linalg.cholesky(F)
                        cell_cos_zld.append(fisher_cos_gpu(L, w_z_t, w_ld_t))
                        cell_cos_zx.append(fisher_cos_gpu(L, w_z_t, w_x_t))
                    except Exception as e:
                        pass
                out_bins[bin_name] = {
                    "n": len(cell_cos_zld),
                    "F_cos_z_ld_mean": float(np.mean(cell_cos_zld)),
                    "F_cos_z_ld_std":  float(np.std(cell_cos_zld)),
                    "F_cos_z_x_mean":  float(np.mean(cell_cos_zx)),
                    "F_cos_z_x_std":   float(np.std(cell_cos_zx)),
                    "entropy_mean":    float(np.mean(entropies)),
                    "entropy_std":     float(np.std(entropies)),
                }
            result["per_pair"][pair][layer] = {
                "euclid_cos_z_ld": eucl_cos(w_z, w_ld),
                "euclid_cos_z_x":  eucl_cos(w_z, w_x),
                "peaked_threshold_|ld|": float(peaked_thr),
                "flat_threshold_|ld|":   float(flat_thr),
                **{f"bin_{k}": v for k, v in out_bins.items()},
            }
            print(f"  [{pair}/{layer}]  "
                  f"|ld|>{peaked_thr:.2f} (n={len(peaked_sel)})  "
                  f"entropy peaked={out_bins['peaked']['entropy_mean']:.3f} vs flat={out_bins['flat']['entropy_mean']:.3f}  "
                  f"F_cos(z,ld): peaked={out_bins['peaked']['F_cos_z_ld_mean']:+.3f}  flat={out_bins['flat']['F_cos_z_ld_mean']:+.3f}  "
                  f"E={eucl_cos(w_z, w_ld):+.3f}",
                  flush=True)

    (OUT_JSON / "fisher_peaked_vs_flat.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON/'fisher_peaked_vs_flat.json'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, layer in zip(axes, LAYERS):
        names = PAIRS
        xpos = np.arange(len(names))
        euc = [result["per_pair"][p][layer]["euclid_cos_z_ld"] for p in names]
        peaked = [result["per_pair"][p][layer]["bin_peaked"]["F_cos_z_ld_mean"] for p in names]
        flat   = [result["per_pair"][p][layer]["bin_flat"]["F_cos_z_ld_mean"] for p in names]
        ax.bar(xpos - 0.25, np.abs(euc), 0.25, label="Euclidean")
        ax.bar(xpos,        np.abs(flat), 0.25, label="F⁻¹ at flat-softmax acts")
        ax.bar(xpos + 0.25, np.abs(peaked), 0.25, label="F⁻¹ at peaked acts (|ld|>p90)")
        ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"|cos(w_z, w_ld)|  layer={layer}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Does F⁻¹ reveal z↔ld alignment when softmax is peaked (low-entropy)?")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fisher_peaked_vs_flat.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'fisher_peaked_vs_flat.png'}")


if __name__ == "__main__":
    main()
