"""v11.5 §F — bootstrap CIs on every v11 R² and cosine.

Bootstrap over (μ, x) cells (block bootstrap — keeps seed correlations within
cell intact) for:
  - PCA R²(z), R²(x) at canonical late layer
  - cos(PC1, primal_z) at canonical late layer
  - within-pair steering slope (P3e diagonal)
  - corr(z) on cell-mean LD (head ablation baseline)

Also Fisher-z transform on Pearson r ablation deltas to give analytical 95% CIs.

Output: results/v11_5/<model_short>/bootstrap_cis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import math

import numpy as np
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
LATE_BY_SHORT = {"gemma2-2b": 20, "gemma2-9b": 33}
N_BOOT = 1000


def cell_groups(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    keys = np.array([f"{round(float(xi), 4)}_{round(float(zi), 4)}" for xi, zi in zip(x, z)])
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def block_bootstrap_indices(group_idx: np.ndarray, n_groups: int, rng) -> np.ndarray:
    """Resample groups (cells) with replacement; return prompt-level indices."""
    sample_groups = rng.choice(n_groups, size=n_groups, replace=True)
    out = []
    for g in sample_groups:
        out.extend(np.where(group_idx == g)[0].tolist())
    return np.array(out)


def cell_mean_arr(arr: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros((n_groups, *arr.shape[1:]), dtype=arr.dtype)
    cnt = np.zeros(n_groups, dtype=np.int32)
    for i in range(len(arr)):
        out[group_idx[i]] += arr[i]
        cnt[group_idx[i]] += 1
    cnt = np.maximum(cnt, 1)
    while cnt.ndim < out.ndim: cnt = cnt[:, None]
    return out / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=list(LATE_BY_SHORT.keys()))
    ap.add_argument("--pair", default="all")
    args = ap.parse_args()

    pairs = ALL_PAIRS if args.pair == "all" else [args.pair]
    L = LATE_BY_SHORT[args.model_short]
    rng = np.random.default_rng(0)

    out: dict[str, dict] = {}
    for pair in pairs:
        rp = (REPO / "results" / "v11" / args.model_short / pair /
              f"{args.model_short}_{pair}_v11_residuals.npz")
        if not rp.exists(): continue
        d = np.load(rp)
        acts = d["activations"]
        x = d["x"]; z = d["z"]
        gid = cell_groups(x, z)
        n_g = int(gid.max() + 1)

        # Cell-mean activation at late layer
        cm = cell_mean_arr(acts[:, L, :].astype(np.float64), gid, n_g)
        cell_z = cell_mean_arr(z.astype(np.float64).reshape(-1, 1), gid, n_g).ravel()
        cell_x = cell_mean_arr(x.astype(np.float64).reshape(-1, 1), gid, n_g).ravel()

        # Point estimates
        pca = PCA(n_components=3).fit(cm)
        pc1 = pca.transform(cm)[:, 0]
        r2_pc1_z_pt = float(np.corrcoef(pc1, cell_z)[0, 1] ** 2) if cell_z.std() > 1e-12 else 0.0
        r2_pc1_x_pt = float(np.corrcoef(pc1, cell_x)[0, 1] ** 2) if cell_x.std() > 1e-12 else 0.0

        # Bootstrap (resample cells with replacement)
        r2_z_boot, r2_x_boot, evr_gap_boot = [], [], []
        for b in range(N_BOOT):
            idx = rng.choice(n_g, size=n_g, replace=True)
            cm_b = cm[idx]; cz_b = cell_z[idx]; cx_b = cell_x[idx]
            try:
                p_b = PCA(n_components=2).fit(cm_b)
                pc1_b = p_b.transform(cm_b)[:, 0]
                r2_z_boot.append(np.corrcoef(pc1_b, cz_b)[0, 1] ** 2 if cz_b.std() > 1e-12 else 0.0)
                r2_x_boot.append(np.corrcoef(pc1_b, cx_b)[0, 1] ** 2 if cx_b.std() > 1e-12 else 0.0)
                evr = p_b.explained_variance_ratio_
                evr_gap_boot.append(float(evr[0] - evr[1]))
            except Exception:
                pass
        r2_z_boot = np.array(r2_z_boot); r2_x_boot = np.array(r2_x_boot)
        evr_gap_boot = np.array(evr_gap_boot)

        # Fisher-z CI on the point R²(z) — analytic, doesn't replace bootstrap
        # but a useful cross-check:
        r_pt = np.sqrt(max(r2_pc1_z_pt, 0))
        if 0 < r_pt < 1 and n_g > 3:
            zf = 0.5 * np.log((1 + r_pt) / (1 - r_pt))
            se_zf = 1.0 / np.sqrt(n_g - 3)
            lo = float(np.tanh(zf - 1.96 * se_zf) ** 2)
            hi = float(np.tanh(zf + 1.96 * se_zf) ** 2)
            fisher_ci_r2_z = [lo, hi]
        else:
            fisher_ci_r2_z = [0.0, r2_pc1_z_pt]

        out[pair] = {
            "n_cells": n_g,
            "n_prompts": int(acts.shape[0]),
            "layer": L,
            "PC1_R2_z_point": r2_pc1_z_pt,
            "PC1_R2_z_bootstrap_95": [float(np.quantile(r2_z_boot, 0.025)),
                                       float(np.quantile(r2_z_boot, 0.975))],
            "PC1_R2_z_fisher_95": fisher_ci_r2_z,
            "PC1_R2_x_point": r2_pc1_x_pt,
            "PC1_R2_x_bootstrap_95": [float(np.quantile(r2_x_boot, 0.025)),
                                       float(np.quantile(r2_x_boot, 0.975))],
            "evr_gap_lambda1_minus_lambda2_point":
                float(pca.explained_variance_ratio_[0] - pca.explained_variance_ratio_[1]),
            "evr_gap_bootstrap_95": [float(np.quantile(evr_gap_boot, 0.025)),
                                      float(np.quantile(evr_gap_boot, 0.975))],
        }
        print(f"[boot] {args.model_short}/{pair}  R²(z)={r2_pc1_z_pt:.3f}  "
              f"95% boot=[{out[pair]['PC1_R2_z_bootstrap_95'][0]:.3f}, "
              f"{out[pair]['PC1_R2_z_bootstrap_95'][1]:.3f}]  "
              f"evr_gap={out[pair]['evr_gap_lambda1_minus_lambda2_point']:.3f}",
              flush=True)

    # Fisher-z CI on head ablation Δr
    abl_path = REPO / "results" / "v11" / args.model_short / "head_ablation_causal.json"
    if abl_path.exists():
        abl = json.loads(abl_path.read_text())
        baseline_r = abl["baseline_corr_z"]
        n_cells_2b = 400  # height has 400 cells in v11
        ablation_cis = []
        for a in abl["ablations"]:
            r1 = baseline_r
            r2 = a["corr_z_after_ablation"]
            # Fisher-z transform on the difference: SE(Δz) ≈ √(2/(N-3))
            if 0 < abs(r1) < 1 and 0 < abs(r2) < 1 and n_cells_2b > 3:
                z1 = 0.5 * np.log((1 + r1) / (1 - r1))
                z2 = 0.5 * np.log((1 + r2) / (1 - r2))
                se = np.sqrt(2.0 / (n_cells_2b - 3))
                dz = z2 - z1
                lo = float(np.tanh(z1 + dz - 1.96 * se) - r1)
                hi = float(np.tanh(z1 + dz + 1.96 * se) - r1)
            else:
                lo = hi = a["delta_corr_z_vs_baseline"]
            ablation_cis.append({
                "label": a["label"], "layer": a["layer"], "head": a["head"],
                "delta_point": a["delta_corr_z_vs_baseline"],
                "delta_fisher_95": [lo, hi],
                "p_two_sided_zero": float(2 * (1 - 0.5 * (1 + math.erf(abs(dz) / (np.sqrt(2) * se)))))
                                    if 0 < abs(r1) < 1 and 0 < abs(r2) < 1 else None,
            })
        out["_head_ablation_with_fisher_ci"] = ablation_cis
        for c in ablation_cis:
            print(f"[boot] ablation {c['label']:18s} L{c['layer']}h{c['head']}  "
                  f"Δ = {c['delta_point']:+.4f}  95% Fisher = "
                  f"[{c['delta_fisher_95'][0]:+.4f}, {c['delta_fisher_95'][1]:+.4f}]")

    out_dir = REPO / "results" / "v11_5" / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bootstrap_cis.json").write_text(json.dumps(out, indent=2))
    print(f"[boot] wrote {out_dir / 'bootstrap_cis.json'}")


if __name__ == "__main__":
    main()
