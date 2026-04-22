"""v9 P5: SAE-feature-space geometry + Goodfire-style Linear Field Probing.

Two analyses addressing the user's follow-up question: "did we see some SAE
feature geometry? like the PCA post feature extraction on SAEs."

──────────────────────────────────────────────────────────────────────────
Part A — SAE-feature PCA (what the user literally asked for)
──────────────────────────────────────────────────────────────────────────
  For each pair:
    1. Encode Grid B activations through Gemma-Scope JumpReLU SAE
       (layer 20, width 65k, avg_l0_61) → coeffs (n_prompts × 65536).
    2. Restrict to active-in-data features (columns that fire at least
       once). Typically ~450-650 such columns.
    3. PCA on these coefficients (center per feature).
    4. Visualize PC1 vs PC2 colored by z, and compare the R²(z) of the
       top SAE-PC against the R²(z) of the top raw-activation PC.

  The scientific question: does moving to SAE basis give a CLEANER 2D
  manifold of z than raw PCA? (v8.2 found PC1 is NOT universally z on
  Grid B for raw activations — is it z in SAE basis?)

──────────────────────────────────────────────────────────────────────────
Part B — Linear Field Probing + Gram kernel PCA (Goodfire-style)
──────────────────────────────────────────────────────────────────────────
  For each pair:
    1. Train ONE logistic probe per z-value (5 probes, one per
       Z_VALUES = [-2,-1,0,+1,+2]): probe_k distinguishes "activations
       at z=z_k" from "activations NOT at z=z_k" using Ridge.
    2. Stack probes as W_pair (5 × d). Compute Gram G = W W^T (5 × 5).
    3. Eigendecompose G — the eigenvalue spectrum reveals how many
       independent directions tile the z-axis.
       - Rank-1 G → single shared z-axis (horseshoe).
       - Spread eigenvalues → per-z-value independent directions.
    4. Kernel-PCA coordinates: K probes as points in 2D (λ₁u₁, λ₂u₂).
       Shape tells you whether probes lie on a line (linear trajectory)
       or a curve (curved manifold, matches §9.2's isomap finding).

  Also runs the SAME LFP analysis on SAE-encoded coefficients
  (probes trained on sparse features instead of raw activations) to see
  whether the sparse basis reveals a cleaner probe geometry.

──────────────────────────────────────────────────────────────────────────
Part C — Stacked cross-pair LFP Gram (40 probes total)
──────────────────────────────────────────────────────────────────────────
  Stack 5 probes × 8 pairs = 40 probes. Kernel-PCA the 40 × 40 Gram.
  Ask: do the 40 probes cluster by PAIR (each pair its own submanifold)
  or by Z-VALUE (a shared z-axis across pairs)?

Outputs
  results/v9_gemma2/sae_feature_pca.json
  results/v9_gemma2/lfp_gram_per_pair.json
  results/v9_gemma2/lfp_stacked_cross_pair.json
  figures/v9/sae_feature_pca_8panel.png
  figures/v9/lfp_gram_spectra.png
  figures/v9/lfp_kernel_pca_per_pair.png
  figures/v9/lfp_stacked_cross_pair.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import PAIRS  # noqa: E402
from extract_v7_xz_grid import Z_VALUES  # noqa: E402

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_PATH = "layer_20/width_65k/average_l0_61/params.npz"


def load_pair(pair_name: str, layer_name="late"):
    with np.load(RES_DIR / f"gemma2_{pair_name}_{layer_name}.npz",
                 allow_pickle=True) as z_:
        acts = z_["activations"].astype(np.float32)
        ids = z_["ids"].tolist()
    trials = {}
    with (RES_DIR / "gemma2_trials.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["pair"] == pair_name:
                trials[t["id"]] = t
    zs = np.array([trials[i]["z"] for i in ids], dtype=np.float64)
    return acts, zs


def load_sae():
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    p = hf_hub_download(SAE_REPO, SAE_PATH)
    P = np.load(p)
    return {k: P[k].astype(np.float32) for k in P.files}


def sae_encode(h, W_enc, b_enc, b_dec, threshold):
    x = h - b_dec
    pre = x @ W_enc + b_enc
    return np.where(pre > threshold, pre, 0.0).astype(np.float32)


def r2_against(z, coord):
    A = np.column_stack([np.ones_like(coord), coord])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def fit_onevsall_probes(X, zs, z_vals, alpha=1.0):
    """Return W of shape (len(z_vals), d_features): one probe per z-value.

    Uses Logistic Regression with L2 (C=1/alpha). Balanced class weights
    because each z-value holds only ~20% of the trials.
    """
    probes = []
    for z_val in z_vals:
        y = (np.isclose(zs, z_val, atol=1e-6)).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            probes.append(np.zeros(X.shape[1], dtype=np.float32))
            continue
        # Simple L2 logistic; avoid convergence warnings with max_iter=500
        clf = LogisticRegression(
            C=1.0 / alpha, class_weight="balanced", max_iter=500, solver="lbfgs",
        ).fit(X, y)
        w = clf.coef_.ravel().astype(np.float32)
        probes.append(w)
    return np.stack(probes)


def gram_kernel_pca(G: np.ndarray):
    """Eigendecompose symmetric Gram G and return (eigvals descending,
    KPCA coords, where coord[i] = eigvecs[:, i] * sqrt(lam_i))."""
    # Make it numerically symmetric
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)   # ascending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals_clip = np.maximum(eigvals, 0.0)
    coords = eigvecs * np.sqrt(eigvals_clip + 1e-12)[None, :]
    return eigvals, coords


# ─── Part A: SAE-feature PCA ──────────────────────────────────────────────

def part_a_sae_pca(sae):
    results = {}
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, p in zip(axes.flat, PAIRS):
        acts, zs = load_pair(p.name)
        coeffs = sae_encode(acts, sae["W_enc"], sae["b_enc"],
                            sae["b_dec"], sae["threshold"])
        active = coeffs.any(axis=0)
        X = coeffs[:, active]  # n × n_active

        # Raw PCA (reference, matches v8.2)
        raw_pca = PCA(n_components=2).fit_transform(acts)
        # SAE PCA
        sae_pca = PCA(n_components=2).fit_transform(X)

        r2_raw = r2_against(zs, raw_pca[:, 0])
        r2_sae = r2_against(zs, sae_pca[:, 0])
        r2_raw_pc2 = r2_against(zs, raw_pca[:, 1])
        r2_sae_pc2 = r2_against(zs, sae_pca[:, 1])

        # Pick best-of (PC1, PC2) per method
        r2_raw_best = max(r2_raw, r2_raw_pc2)
        r2_sae_best = max(r2_sae, r2_sae_pc2)

        results[p.name] = {
            "n_active_features":  int(active.sum()),
            "r2_raw_pc1":         r2_raw,
            "r2_raw_pc2":         r2_raw_pc2,
            "r2_sae_pc1":         r2_sae,
            "r2_sae_pc2":         r2_sae_pc2,
            "r2_raw_best12":      float(r2_raw_best),
            "r2_sae_best12":      float(r2_sae_best),
        }

        # Plot SAE PC1 vs SAE PC2 colored by z
        sc = ax.scatter(sae_pca[:, 0], sae_pca[:, 1], c=zs, cmap="coolwarm",
                        s=8, alpha=0.8)
        ax.set_title(
            f"{p.name}  r²(PC1)={r2_sae:.2f}  r²(PC2)={r2_sae_pc2:.2f}\n"
            f"[vs raw-act: r²(PC1)={r2_raw:.2f}  r²(PC2)={r2_raw_pc2:.2f}]",
            fontsize=9,
        )
        ax.set_xlabel("SAE PC1"); ax.set_ylabel("SAE PC2")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, label="z")
    fig.suptitle(
        "v9 P5-A — PCA in SAE feature space (colored by z) — "
        "layer 20 width 65k avg_l0_61",
        fontsize=11)
    fig.savefig(FIG_DIR / "sae_feature_pca_8panel.png", dpi=140,
                bbox_inches="tight")
    print(f"  wrote {FIG_DIR}/sae_feature_pca_8panel.png")
    return results


# ─── Part B: LFP Gram kernel PCA per pair ─────────────────────────────────

def part_b_lfp_per_pair(sae):
    z_vals = list(Z_VALUES)
    per_pair = {}
    fig_spec, axes_spec = plt.subplots(2, 4, figsize=(16, 8))
    fig_kpca, axes_kpca = plt.subplots(2, 4, figsize=(16, 8))
    for ax_spec, ax_kpca, p in zip(axes_spec.flat, axes_kpca.flat, PAIRS):
        acts, zs = load_pair(p.name)
        coeffs = sae_encode(acts, sae["W_enc"], sae["b_enc"],
                            sae["b_dec"], sae["threshold"])
        active = coeffs.any(axis=0)
        X_sae = coeffs[:, active]

        # Raw-act probes + Gram
        W_raw = fit_onevsall_probes(acts, zs, z_vals)
        W_raw_n = W_raw / (np.linalg.norm(W_raw, axis=1, keepdims=True) + 1e-9)
        G_raw = W_raw_n @ W_raw_n.T
        eig_raw, kpca_raw = gram_kernel_pca(G_raw)

        # SAE probes + Gram
        W_sae = fit_onevsall_probes(X_sae, zs, z_vals)
        W_sae_n = W_sae / (np.linalg.norm(W_sae, axis=1, keepdims=True) + 1e-9)
        G_sae = W_sae_n @ W_sae_n.T
        eig_sae, kpca_sae = gram_kernel_pca(G_sae)

        per_pair[p.name] = {
            "z_values": z_vals,
            "eigvals_raw": eig_raw.tolist(),
            "eigvals_sae": eig_sae.tolist(),
            "G_raw": G_raw.tolist(),
            "G_sae": G_sae.tolist(),
            "kpca_raw": kpca_raw[:, :2].tolist(),
            "kpca_sae": kpca_sae[:, :2].tolist(),
            "ID_raw": float(eig_raw.sum() ** 2 / (eig_raw ** 2).sum())
                       if (eig_raw ** 2).sum() > 1e-12 else 0.0,
            "ID_sae": float(eig_sae.sum() ** 2 / (eig_sae ** 2).sum())
                       if (eig_sae ** 2).sum() > 1e-12 else 0.0,
        }

        # Plot: spectrum
        ax_spec.plot(range(1, len(eig_raw) + 1), eig_raw, "o-", label="raw")
        ax_spec.plot(range(1, len(eig_sae) + 1), eig_sae, "s-", label="SAE")
        ax_spec.set_title(p.name, fontsize=9)
        ax_spec.set_xlabel("eigenvalue rank")
        ax_spec.set_ylabel("eigenvalue")
        ax_spec.legend(fontsize=7)
        ax_spec.axhline(0, color="k", lw=0.5)

        # Plot: KPCA 2D, colored by z-value, on SAE basis
        cols = plt.cm.coolwarm(np.linspace(0, 1, len(z_vals)))
        for k, zv in enumerate(z_vals):
            ax_kpca.scatter(kpca_sae[k, 0], kpca_sae[k, 1],
                            c=[cols[k]], s=120, edgecolor="k",
                            label=f"z={zv:+.0f}")
        # Connect points in z-order to see trajectory
        ax_kpca.plot(kpca_sae[:, 0], kpca_sae[:, 1], "k-", lw=0.8, alpha=0.5)
        ax_kpca.set_title(
            f"{p.name} — SAE-basis LFP kPCA\n"
            f"ID(raw)={per_pair[p.name]['ID_raw']:.2f}  "
            f"ID(SAE)={per_pair[p.name]['ID_sae']:.2f}",
            fontsize=9)
        ax_kpca.set_xlabel("LFP KPC 1"); ax_kpca.set_ylabel("LFP KPC 2")
        if p.name == "height":
            ax_kpca.legend(fontsize=6)

    fig_spec.suptitle(
        "v9 P5-B — Eigenvalue spectrum of 5×5 LFP-probe Gram matrix "
        "(one probe per z-value)", fontsize=11)
    fig_spec.tight_layout(); fig_spec.savefig(FIG_DIR / "lfp_gram_spectra.png", dpi=140)
    fig_kpca.suptitle(
        "v9 P5-B — Kernel-PCA of LFP-probe Gram (SAE basis) — "
        "curve shape = z-manifold shape", fontsize=11)
    fig_kpca.tight_layout(); fig_kpca.savefig(FIG_DIR / "lfp_kernel_pca_per_pair.png",
                                              dpi=140)
    print(f"  wrote {FIG_DIR}/lfp_gram_spectra.png")
    print(f"  wrote {FIG_DIR}/lfp_kernel_pca_per_pair.png")
    return per_pair


# ─── Part C: stacked cross-pair LFP Gram ──────────────────────────────────

def part_c_stacked(sae):
    z_vals = list(Z_VALUES)
    all_probes_raw = []
    all_probes_sae = []
    labels = []
    for p in PAIRS:
        acts, zs = load_pair(p.name)
        coeffs = sae_encode(acts, sae["W_enc"], sae["b_enc"],
                            sae["b_dec"], sae["threshold"])
        active = coeffs.any(axis=0)
        W_raw = fit_onevsall_probes(acts, zs, z_vals)
        W_sae = fit_onevsall_probes(coeffs[:, active], zs, z_vals)
        for k, zv in enumerate(z_vals):
            all_probes_raw.append(W_raw[k])
            all_probes_sae.append(np.zeros(coeffs.shape[1], dtype=np.float32))
            # put SAE probe back into the full 65k space (zeros elsewhere)
            all_probes_sae[-1][active] = W_sae[k]
            labels.append({"pair": p.name, "z": float(zv)})

    W_R = np.stack(all_probes_raw)
    W_S = np.stack(all_probes_sae)
    W_R = W_R / (np.linalg.norm(W_R, axis=1, keepdims=True) + 1e-9)
    W_S = W_S / (np.linalg.norm(W_S, axis=1, keepdims=True) + 1e-9)
    G_R = W_R @ W_R.T
    G_S = W_S @ W_S.T
    eig_R, kpca_R = gram_kernel_pca(G_R)
    eig_S, kpca_S = gram_kernel_pca(G_S)

    # Plot 2×2: Gram heatmaps + KPCA embeddings
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    n = len(labels)
    for ax, G, name in zip(axes[0], [G_R, G_S], ["raw activations", "SAE basis"]):
        vmax = np.abs(G).max()
        im = ax.imshow(G, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        # Tick labels: "pair, z"
        ticks = [f"{labels[i]['pair'][:4]}/{labels[i]['z']:+.0f}" for i in range(n)]
        ax.set_xticks(range(n)); ax.set_xticklabels(ticks, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(ticks, fontsize=6)
        ax.set_title(f"40×40 probe-cosine Gram ({name})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
        # overlay pair-block grid
        for i in range(1, 8):
            ax.axhline(5 * i - 0.5, color="k", lw=0.5, alpha=0.3)
            ax.axvline(5 * i - 0.5, color="k", lw=0.5, alpha=0.3)

    # KPCA embeddings colored by z-value with pair-labelled markers
    colors = plt.cm.coolwarm(np.linspace(0, 1, 5))
    markers = ["o", "s", "D", "v", "^", "P", "X", "*"]
    for ax, kpca, name in zip(axes[1], [kpca_R, kpca_S], ["raw", "SAE"]):
        for k, lbl in enumerate(labels):
            pair_idx = [p.name for p in PAIRS].index(lbl["pair"])
            z_idx = z_vals.index(lbl["z"])
            ax.scatter(kpca[k, 0], kpca[k, 1],
                       marker=markers[pair_idx % len(markers)],
                       color=colors[z_idx], s=80, edgecolor="k",
                       linewidth=0.6)
        ax.set_xlabel("KPC 1"); ax.set_ylabel("KPC 2")
        ax.set_title(f"40-probe kernel PCA ({name}) — color=z, marker=pair",
                     fontsize=10)
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    fig.suptitle(
        "v9 P5-C — Stacked cross-pair LFP Gram: do probes cluster by pair "
        "or by z-value?", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "lfp_stacked_cross_pair.png", dpi=140)
    print(f"  wrote {FIG_DIR}/lfp_stacked_cross_pair.png")

    return {
        "labels": labels,
        "eigvals_raw": eig_R.tolist(),
        "eigvals_sae": eig_S.tolist(),
        "ID_raw": float(eig_R.sum() ** 2 / (eig_R ** 2).sum())
                  if (eig_R ** 2).sum() > 1e-12 else 0.0,
        "ID_sae": float(eig_S.sum() ** 2 / (eig_S ** 2).sum())
                  if (eig_S ** 2).sum() > 1e-12 else 0.0,
        "G_raw": G_R.tolist(),
        "G_sae": G_S.tolist(),
        "kpca_raw_top3": kpca_R[:, :3].tolist(),
        "kpca_sae_top3": kpca_S[:, :3].tolist(),
    }


def main():
    print("Loading SAE…")
    sae = load_sae()
    print("\n=== Part A: SAE-feature PCA per pair ===")
    a = part_a_sae_pca(sae)
    (RES_DIR / "sae_feature_pca.json").write_text(json.dumps(a, indent=2))
    for name, r in a.items():
        print(f"  {name:12s}  raw best r²(z)={r['r2_raw_best12']:.3f}  "
              f"SAE best r²(z)={r['r2_sae_best12']:.3f}  "
              f"{'↑ SAE' if r['r2_sae_best12'] > r['r2_raw_best12'] else '↓ SAE'}")

    print("\n=== Part B: LFP Gram per pair ===")
    b = part_b_lfp_per_pair(sae)
    (RES_DIR / "lfp_gram_per_pair.json").write_text(json.dumps(b, indent=2))
    for name, r in b.items():
        print(f"  {name:12s}  eig_sum ratio: "
              f"raw {r['eigvals_raw'][0] / max(sum(r['eigvals_raw']),1e-6):.2f}  "
              f"SAE {r['eigvals_sae'][0] / max(sum(r['eigvals_sae']),1e-6):.2f}   "
              f"ID: raw={r['ID_raw']:.2f}  SAE={r['ID_sae']:.2f}")

    print("\n=== Part C: stacked cross-pair LFP ===")
    c = part_c_stacked(sae)
    (RES_DIR / "lfp_stacked_cross_pair.json").write_text(json.dumps(c, indent=2))
    print(f"  cross-pair ID: raw={c['ID_raw']:.2f}  SAE={c['ID_sae']:.2f}")


if __name__ == "__main__":
    main()
