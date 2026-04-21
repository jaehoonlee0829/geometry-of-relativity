"""Exp 3b: F⁻¹ (Fisher) cosines for w_adj, w_z, w_x — the H4 validation.

For each pair × layer, computes probe weights (ridge on cached activations),
then picks up to N_CELL cell-mean activations, computes F(h) at each via
src/fisher.py, solves F⁻¹·w via Cholesky, and reports cos_F⁻¹(w_adj, w_z),
cos_F⁻¹(w_adj, w_x), cos_F⁻¹(w_z, w_x).

Compared to cached Σ⁻¹ cosines (exp3a_sigma_inv.py) and Euclidean.

Writes: results/v4_adjpairs_analysis/fisher_cosines.json
        figures/v4_adjpairs/fisher_vs_euclid_cosines.png
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO))
import torch  # noqa: E402


def fisher_matrix_gpu(h: torch.Tensor, W_U: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """GPU Fisher: F(h) = W_U^T diag(p) W_U - (W_U^T p)(W_U^T p)^T + jitter·I.

    h: (d,) float32 on device
    W_U: (V, d) float32 on device
    """
    logits = h @ W_U.T                                    # (V,)
    p = torch.softmax(logits.double(), dim=0)            # (V,) fp64 for stability
    W_U64 = W_U.double()
    scaled = p.unsqueeze(1) * W_U64                      # (V, d)
    term1 = W_U64.T @ scaled                             # (d, d)
    w_t_p = W_U64.T @ p                                  # (d,)
    F = term1 - torch.outer(w_t_p, w_t_p)
    d = F.shape[0]
    F = F + jitter * torch.eye(d, dtype=F.dtype, device=F.device) * F.diagonal().mean().clamp_min(1e-12)
    return F

ADJPAIRS = REPO / "results" / "v4_adjpairs"
OUT_JSON = REPO / "results" / "v4_adjpairs_analysis"
OUT_FIG = REPO / "figures" / "v4_adjpairs"
OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

PAIRS = ["height", "age", "experience", "size", "speed", "wealth", "weight", "bmi_abs"]
LAYERS = ["mid", "late"]
N_CELL = 25          # per pair: 25 cells = 5 x × 5 μ
W_U_PATH = REPO / "results" / "activations" / "e4b_W_U.npy"


def eucl_cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def load_pair_layer(pair: str, layer: str):
    trials_by_id = {json.loads(l)["id"]: json.loads(l) for l in (ADJPAIRS / "e4b_trials.jsonl").open()}
    ld_by_id = {r["id"]: r["logit_diff"] for r in map(json.loads, (ADJPAIRS / f"e4b_{pair}_implicit_logits.jsonl").open())}
    npz = np.load(ADJPAIRS / f"e4b_{pair}_implicit_{layer}.npz", allow_pickle=True)
    acts = npz["activations"].astype(np.float64)
    ids = [str(s) for s in npz["ids"]]
    xs, zs, ld = [], [], []
    keys = []
    for i in ids:
        t = trials_by_id[i]
        xs.append(t["x"])
        zs.append(t["z"])
        ld.append(ld_by_id[i])
        keys.append((t["x"], t["mu"]))
    return acts, np.array(xs), np.array(zs), np.array(ld), keys


def train_probe(X, y, alpha=1.0):
    w = Ridge(alpha=alpha).fit(X, y).coef_
    return w.astype(np.float64)


def cell_mean_activations(acts: np.ndarray, keys: list[tuple]) -> np.ndarray:
    by_cell = defaultdict(list)
    for a, k in zip(acts, keys):
        by_cell[k].append(a)
    means = np.stack([np.mean(by_cell[k], axis=0) for k in sorted(by_cell)])
    return means


def fisher_cos_gpu(L: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> float:
    # L is Cholesky lower; solve F·x = u by back-sub: x = L^{-T} L^{-1} u
    Finv_u = torch.cholesky_solve(u.unsqueeze(1), L).squeeze(1)
    Finv_v = torch.cholesky_solve(v.unsqueeze(1), L).squeeze(1)
    num = float((u * Finv_v).sum())
    du = float((u * Finv_u).sum())
    dv = float((v * Finv_v).sum())
    if du <= 0 or dv <= 0:
        return float("nan")
    return num / ((du ** 0.5) * (dv ** 0.5))


def main() -> None:
    assert W_U_PATH.exists(), f"missing {W_U_PATH} — run export_W_U.py e4b first"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading W_U from {W_U_PATH} → {device} …", flush=True)
    W_U_np = np.load(W_U_PATH).astype(np.float32)
    W_U = torch.from_numpy(W_U_np).to(device)
    print(f"  W_U shape={tuple(W_U.shape)}  dtype={W_U.dtype}  device={W_U.device}", flush=True)

    result: dict = {"n_cells_per_pair": N_CELL, "per_pair": {}}
    for pair in PAIRS:
        result["per_pair"][pair] = {}
        for layer in LAYERS:
            print(f"\n[{pair}/{layer}]", flush=True)
            acts, xs, zs, ld, keys = load_pair_layer(pair, layer)
            y_adj = np.sign(zs).astype(np.float64)
            y_adj[y_adj == 0] = 1.0
            w_x = train_probe(acts, xs)
            w_z = train_probe(acts, zs)
            w_adj = train_probe(acts, y_adj)
            w_ld = train_probe(acts, ld)

            # Cell-mean activations for Fisher points
            H = cell_mean_activations(acts, keys)
            if H.shape[0] > N_CELL:
                H = H[:N_CELL]
            per_cell = []
            w_adj_t = torch.from_numpy(w_adj).to(device).double()
            w_z_t   = torch.from_numpy(w_z).to(device).double()
            w_x_t   = torch.from_numpy(w_x).to(device).double()
            w_ld_t  = torch.from_numpy(w_ld).to(device).double()
            t0 = time.time()
            for i, h_np in enumerate(H):
                h = torch.from_numpy(h_np.astype(np.float32)).to(device)
                try:
                    F = fisher_matrix_gpu(h, W_U)
                    L = torch.linalg.cholesky(F)
                    c = {
                        "F_cos_adj_z": fisher_cos_gpu(L, w_adj_t, w_z_t),
                        "F_cos_adj_x": fisher_cos_gpu(L, w_adj_t, w_x_t),
                        "F_cos_z_x":   fisher_cos_gpu(L, w_z_t, w_x_t),
                        "F_cos_z_ld":  fisher_cos_gpu(L, w_z_t, w_ld_t),
                    }
                except Exception as e:
                    print(f"    cell {i} failed: {e}", flush=True)
                    continue
                per_cell.append(c)
            elapsed = time.time() - t0
            means = {k: float(np.mean([c[k] for c in per_cell])) for k in per_cell[0]} if per_cell else {}
            stds = {k: float(np.std([c[k] for c in per_cell])) for k in per_cell[0]} if per_cell else {}
            eucl = {
                "E_cos_adj_z": eucl_cos(w_adj, w_z),
                "E_cos_adj_x": eucl_cos(w_adj, w_x),
                "E_cos_z_x":   eucl_cos(w_z, w_x),
                "E_cos_z_ld":  eucl_cos(w_z, w_ld),
            }
            print(f"  {len(per_cell)} cells, {elapsed:.1f}s", flush=True)
            print(f"  cos(adj,z):  E={eucl['E_cos_adj_z']:+.3f}   F⁻¹={means.get('F_cos_adj_z', 0):+.3f}±{stds.get('F_cos_adj_z',0):.3f}", flush=True)
            print(f"  cos(z,ld):   E={eucl['E_cos_z_ld']:+.3f}   F⁻¹={means.get('F_cos_z_ld', 0):+.3f}±{stds.get('F_cos_z_ld',0):.3f}", flush=True)
            result["per_pair"][pair][layer] = {
                "n_cells": len(per_cell),
                "euclidean": eucl,
                "fisher_mean": means,
                "fisher_std": stds,
                "per_cell": per_cell,
            }

    (OUT_JSON / "fisher_cosines.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON/'fisher_cosines.json'}")

    # Figure: per-pair |cos(adj,z)| — Euclidean vs Fisher (bar pairs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric_keys = [("cos_adj_z", "|cos(w_adj, w_z)|"), ("cos_z_ld", "|cos(w_z, w_ld)|")]
    for ax, (mkey, title) in zip(axes, metric_keys):
        xs_bar = np.arange(len(PAIRS))
        w = 0.2
        for li, layer in enumerate(LAYERS):
            eu = [abs(result["per_pair"][p][layer]["euclidean"][f"E_{mkey}"]) for p in PAIRS]
            fi = [abs(result["per_pair"][p][layer]["fisher_mean"].get(f"F_{mkey}", 0)) for p in PAIRS]
            ax.bar(xs_bar + (li*2)*w, eu, w, label=f"E / {layer}")
            ax.bar(xs_bar + (li*2+1)*w, fi, w, label=f"F⁻¹ / {layer}")
        ax.set_xticks(xs_bar + 1.5*w)
        ax.set_xticklabels(PAIRS, rotation=30, ha="right", fontsize=9)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Euclidean vs F⁻¹ cosines — does Fisher pullback reveal alignment?")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fisher_vs_euclid_cosines.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'fisher_vs_euclid_cosines.png'}")


if __name__ == "__main__":
    main()
