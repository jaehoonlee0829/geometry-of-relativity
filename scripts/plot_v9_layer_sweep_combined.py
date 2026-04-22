"""v9 §13 — combined layer-sweep figure: one plot with all key curves.

A single 4-panel figure showing the story:
  (a) CV R²(z) per layer        — "where is z encoded?"
  (b) Intrinsic dim per layer   — "how curved is the z-manifold at each depth?"
  (c) primal_z steering slope   — "where is z causally potent?"
  (d) cos(primal_z[L], primal_z[L-1])  — "when does the direction stabilize?"

All averaged over 8 pairs with shaded min/max band.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"


def agg(values_per_pair):
    a = np.array(values_per_pair)  # (n_pairs, n_layers)
    return a.mean(axis=0), a.min(axis=0), a.max(axis=0)


def main():
    # Geometry data: R²(z), ID, primal_norm, cos_prev per (pair, layer)
    geom = json.loads((RES_DIR / "layer_sweep_geometry.json").read_text())
    L = geom["pairs"][0]["n_layers"]
    layers = np.arange(L)

    r2_z = np.array([[rec["r2_cv_z"] for rec in p["layer_records"]]
                      for p in geom["pairs"]])
    id_2nn = np.array([[rec["id_cell_means_TWONN"] if np.isfinite(
                         rec["id_cell_means_TWONN"]) else np.nan
                        for rec in p["layer_records"]] for p in geom["pairs"]])
    pnorm = np.array([[rec["primal_norm"] for rec in p["layer_records"]]
                       for p in geom["pairs"]])
    cos_prev = np.array([[rec["primal_cos_prev_layer"]
                          if np.isfinite(rec["primal_cos_prev_layer"]) else np.nan
                          for rec in p["layer_records"]]
                         for p in geom["pairs"]])
    lfp_pr = np.array([[rec["lfp_participation_ratio"]
                        for rec in p["layer_records"]] for p in geom["pairs"]])

    # Steering data — only certain layers
    steer = json.loads((RES_DIR / "layer_sweep_steering.json").read_text())
    steer_layers = steer["layers"]
    steer_primal = np.array([[steer["per_pair"][name][f"layer_{L_}"]["primal_slope"]
                              for L_ in steer_layers]
                             for name in steer["per_pair"].keys()])
    steer_probe = np.array([[steer["per_pair"][name][f"layer_{L_}"]["probe_slope"]
                             for L_ in steer_layers]
                            for name in steer["per_pair"].keys()])

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # (a) R²(z) per layer
    ax = axes[0, 0]
    m = r2_z.mean(axis=0)
    ax.plot(layers, m, "-o", color="C0", lw=2, ms=4, label="mean")
    ax.fill_between(layers, r2_z.min(axis=0), r2_z.max(axis=0),
                    color="C0", alpha=0.2, label="min / max across 8 pairs")
    ax.axhline(0.9, color="k", ls=":", lw=0.5, alpha=0.5)
    ax.set_title("(a) CV R²(z) per layer — where z is encodable",
                 fontsize=10)
    ax.set_xlabel("layer"); ax.set_ylabel("R²(z)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Intrinsic dim (TWO-NN on cell-means)
    ax = axes[0, 1]
    m = np.nanmean(id_2nn, axis=0)
    ax.plot(layers, m, "-o", color="C2", lw=2, ms=4, label="mean")
    ax.fill_between(layers, np.nanmin(id_2nn, axis=0), np.nanmax(id_2nn, axis=0),
                    color="C2", alpha=0.2)
    ax.set_title("(b) Intrinsic dim (TWO-NN on cell-means) — curvature by depth\n"
                 "Goodfire prediction: rises mid, drops at last layer",
                 fontsize=9)
    ax.set_xlabel("layer"); ax.set_ylabel("ID")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (c) LFP 5-probe Gram participation ratio
    ax = axes[0, 2]
    m = lfp_pr.mean(axis=0)
    ax.plot(layers, m, "-o", color="C3", lw=2, ms=4, label="mean")
    ax.fill_between(layers, lfp_pr.min(axis=0), lfp_pr.max(axis=0),
                    color="C3", alpha=0.2)
    ax.axhline(5, color="k", ls=":", lw=0.6)
    ax.axhline(1, color="k", ls=":", lw=0.6)
    ax.set_title("(c) LFP 5-probe Gram PR — near 5 = per-z-value probes orthogonal",
                 fontsize=9)
    ax.set_xlabel("layer"); ax.set_ylabel("PR (max=5)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (d) steering slope per layer (primal vs probe)
    ax = axes[1, 0]
    m_p = steer_primal.mean(axis=0)
    m_b = steer_probe.mean(axis=0)
    ax.plot(steer_layers, m_p, "-o", color="C0", lw=2, ms=6,
            label="primal_z (mean across pairs)")
    ax.plot(steer_layers, m_b, "-s", color="C1", lw=2, ms=6, label="probe_z (Ridge)")
    ax.fill_between(steer_layers, steer_primal.min(axis=0), steer_primal.max(axis=0),
                    color="C0", alpha=0.2)
    ax.fill_between(steer_layers, steer_probe.min(axis=0), steer_probe.max(axis=0),
                    color="C1", alpha=0.2)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_title("(d) Causal steering slope by layer — where z is USED",
                 fontsize=10)
    ax.set_xlabel("layer"); ax.set_ylabel("Δlogit_diff per α")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) primal_z norm grows with depth
    ax = axes[1, 1]
    m = pnorm.mean(axis=0)
    ax.plot(layers, m, "-o", color="C4", lw=2, ms=4, label="mean")
    ax.fill_between(layers, pnorm.min(axis=0), pnorm.max(axis=0),
                    color="C4", alpha=0.2)
    ax.set_title("(e) ‖primal_z‖ by layer — amplification with depth", fontsize=10)
    ax.set_xlabel("layer"); ax.set_ylabel("‖primal_z‖")
    ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (f) cos(primal_z[L], primal_z[L-1])
    ax = axes[1, 2]
    m = np.nanmean(cos_prev, axis=0)
    ax.plot(layers, m, "-o", color="C5", lw=2, ms=4, label="mean")
    ax.fill_between(layers, np.nanmin(cos_prev, axis=0),
                    np.nanmax(cos_prev, axis=0), color="C5", alpha=0.2)
    ax.axhline(1, color="k", ls=":", lw=0.5, alpha=0.5)
    ax.axhline(0, color="k", ls=":", lw=0.5, alpha=0.5)
    ax.set_title("(f) cos(primal_z[L], primal_z[L−1]) — direction stability",
                 fontsize=10)
    ax.set_xlabel("layer"); ax.set_ylabel("cosine")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("v9 §13 — Gemma 2 2B layer sweep: encode vs. use vs. curvature",
                 fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "layer_sweep_combined.png"
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
