"""Regenerate all v7 figures from pre-computed JSON results (no .npz needed).

Reads:
  results/v7_analysis/direction_confound_audit_clean.json
  results/v7_analysis/inlp_clean.json
  results/v7_analysis/park_fisher_clean.json
  results/v7_steering/clean_direction_comparison.json
  results/v7_steering/clean_transfer_matrix.json

Writes into figures/v7/:
  direction_confound_matrix_gridB.png   — Grid B only (clean)
  cross_grid_direction_stability.png    — how stable each direction is across grids
  inlp_clean_curves.png                — INLP R² decay curves
  fisher_entropy_bins_clean.png         — Fisher-metric cos(w_z, w_ld) by entropy bin
  seven_direction_curves_clean_8pair.png — 8-direction steering curves
  clean_transfer_heatmap.png            — cross-pair transfer matrix
  steering_slopes_all_directions_clean.png — bar chart of |slope| per direction

Run:
  python scripts/replot_v7_from_json.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "v7"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = ["height", "age", "experience", "size", "speed", "wealth", "weight", "bmi_abs"]
DIR_NAMES = ["primal_z", "primal_x", "probe_z", "probe_x", "pc1", "pc2", "zeroshot_wx"]


# ---------------------------------------------------------------------------
# 1. Direction confound matrix — Grid B only
# ---------------------------------------------------------------------------
def plot_confound_matrix():
    data = json.loads((ROOT / "results/v7_analysis/direction_confound_audit_clean.json").read_text())
    matrices = [np.array(data["per_pair_gridB"][p]) for p in PAIRS if p in data["per_pair_gridB"]]
    mean_B = np.mean(matrices, axis=0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(np.abs(mean_B), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(DIR_NAMES)))
    ax.set_xticklabels(DIR_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(DIR_NAMES)))
    ax.set_yticklabels(DIR_NAMES)
    for i in range(len(DIR_NAMES)):
        for j in range(len(DIR_NAMES)):
            ax.text(j, i, f"{abs(mean_B[i][j]):.2f}", ha="center", va="center",
                    color="white" if abs(mean_B[i][j]) > 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Direction cosines — Grid B (x, z) — v7 clean", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "direction_confound_matrix_gridB.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'direction_confound_matrix_gridB.png'}")


# ---------------------------------------------------------------------------
# 2. Cross-grid direction stability
# ---------------------------------------------------------------------------
def plot_cross_grid_stability():
    data = json.loads((ROOT / "results/v7_analysis/direction_confound_audit_clean.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 5))
    xpos = np.arange(len(DIR_NAMES))
    w = 1.0 / (len(PAIRS) + 1)
    for i, pname in enumerate(PAIRS):
        if pname not in data["cross_grid"]:
            continue
        vals = [abs(data["cross_grid"][pname]["cos_gridA_gridB_per_direction"][d]) for d in DIR_NAMES]
        ax.bar(xpos + i * w, vals, w, label=pname)
    ax.set_xticks(xpos + 3.5 * w)
    ax.set_xticklabels(DIR_NAMES, rotation=30, ha="right")
    ax.set_ylabel("|cos| (same direction: Grid A vs Grid B)")
    ax.set_title("How stable is each direction across grid designs?")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cross_grid_direction_stability.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'cross_grid_direction_stability.png'}")


# ---------------------------------------------------------------------------
# 3. INLP clean curves
# ---------------------------------------------------------------------------
def plot_inlp():
    data = json.loads((ROOT / "results/v7_analysis/inlp_clean.json").read_text())
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pname in zip(axes.ravel(), PAIRS):
        if pname not in data:
            ax.set_visible(False)
            continue
        d = data[pname]
        r2_inlp = d["r2_inlp_z"]
        r2_rand = d["r2_random_mean"]
        r2_std = d["r2_random_std"]
        iters = np.arange(len(r2_inlp))
        ax.plot(iters, r2_inlp, marker="o", lw=2, label="INLP (project out z-direction)")
        ax.errorbar(iters, r2_rand, yerr=r2_std, marker="s", lw=1.5,
                     alpha=0.7, capsize=3, label="random null (avg 3)")
        ax.set_title(pname, fontsize=10)
        ax.set_xlabel("iteration")
        ax.set_ylabel("CV R²(z)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("INLP on clean (Grid B) activations — 8 iterations of z-direction projection")
    fig.tight_layout()
    fig.savefig(OUT / "inlp_clean_curves.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'inlp_clean_curves.png'}")


# ---------------------------------------------------------------------------
# 4. Fisher entropy bins
# ---------------------------------------------------------------------------
def plot_fisher_entropy():
    data = json.loads((ROOT / "results/v7_analysis/park_fisher_clean.json").read_text())
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pname in zip(axes.ravel(), PAIRS):
        if pname not in data:
            ax.set_visible(False)
            continue
        d = data[pname]
        # 4-metric bar: euclid cos(z,ld), park cos(z,ld), euclid cos(adj,z), park cos(adj,z)
        metrics = {
            "euclid\ncos(z,ld)": d["euclid_cos_z_ld"],
            "park\ncos(z,ld)": d["park_cos_z_ld"],
            "euclid\ncos(adj,z)": d["euclid_cos_adj_z"],
            "park\ncos(adj,z)": d["park_cos_adj_z"],
        }
        # Low vs high entropy bins
        if "bin_low" in d and "bin_high" in d:
            lo = d["bin_low"]
            hi = d["bin_high"]
            xpos = np.arange(2)
            vals = [lo["F_cos_z_ld_mean"], hi["F_cos_z_ld_mean"]]
            errs = [lo["F_cos_z_ld_std"], hi["F_cos_z_ld_std"]]
            ax.bar(xpos, vals, yerr=errs, capsize=4, color=["#1f77b4", "#ff7f0e"])
            ax.set_xticks(xpos)
            ax.set_xticklabels([f"low ent\n(μ={lo['entropy_mean']:.2f})",
                                f"high ent\n(μ={hi['entropy_mean']:.2f})"], fontsize=7)
            ax.set_ylabel("F-cos(w_z, w_ld)", fontsize=8)
        ax.set_title(pname, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Fisher-metric cos(w_z, w_ld) by entropy bin — Grid B clean", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fisher_entropy_bins_clean.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'fisher_entropy_bins_clean.png'}")


# ---------------------------------------------------------------------------
# 5. Seven direction steering curves
# ---------------------------------------------------------------------------
def plot_steering_curves():
    data = json.loads((ROOT / "results/v7_steering/clean_direction_comparison.json").read_text())
    # Determine direction names from first pair
    first_pair = next(iter(data))
    dir_names = list(data[first_pair].keys())
    # Determine alpha range from first direction's curve
    alphas = sorted(float(a) for a in data[first_pair][dir_names[0]]["curve"].keys())

    colors = {"primal_z": "#1f77b4", "primal_x": "#ff7f0e", "probe_z": "#2ca02c",
              "probe_x": "#d62728", "meta_w1": "#9467bd", "zeroshot_wx": "#8c564b",
              "pc2": "#e377c2", "pc1": "#17becf"}

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pname in zip(axes.ravel(), PAIRS):
        if pname not in data:
            ax.set_visible(False)
            continue
        for dname in dir_names:
            if dname not in data[pname]:
                continue
            curve = data[pname][dname]["curve"]
            ys = [curve[str(a)]["ld_mean"] for a in alphas]
            ax.plot(alphas, ys, marker="o", color=colors.get(dname, "gray"),
                    label=dname, lw=1.5, markersize=3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_title(pname, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("α", fontsize=8)
    axes.ravel()[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("v7 clean-grid: 8-direction steering curves (E4B layer 32)")
    fig.tight_layout()
    fig.savefig(OUT / "seven_direction_curves_clean_8pair.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {OUT / 'seven_direction_curves_clean_8pair.png'}")


# ---------------------------------------------------------------------------
# 6. Transfer heatmap
# ---------------------------------------------------------------------------
def plot_transfer_heatmap():
    data = json.loads((ROOT / "results/v7_steering/clean_transfer_matrix.json").read_text())
    M = np.array(data["transfer_matrix_slope_per_alpha_unit"])
    pair_names = data["pair_names"]
    rand_slopes = np.array(data["random_null_slopes_per_pair_per_seed"])
    rand_mean = np.abs(rand_slopes).mean()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(pair_names)))
    ax.set_xticklabels(pair_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels(pair_names, fontsize=9)
    ax.set_xlabel("target pair (steered)")
    ax.set_ylabel("source pair (direction from)")
    for i in range(len(pair_names)):
        for j in range(len(pair_names)):
            color = "white" if M[i, j] > M.max() * 0.6 else "black"
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, fraction=0.04, label="slope (Δld / α)")
    s = data["summary"]
    ax.set_title(
        f"Cross-pair transfer — Grid B clean\n"
        f"diag={s['diagonal_mean_abs']:.3f}  off={s['offdiag_mean_abs']:.3f}  "
        f"ratio={s['transfer_ratio']:.1%}  null={s['random_null_mean_abs']:.4f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "clean_transfer_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'clean_transfer_heatmap.png'}")


# ---------------------------------------------------------------------------
# 7. Steering slopes bar chart (all directions)
# ---------------------------------------------------------------------------
def plot_steering_slopes():
    data = json.loads((ROOT / "results/v7_steering/clean_direction_comparison.json").read_text())
    first_pair = next(iter(data))
    dir_names = list(data[first_pair].keys())

    colors = {"primal_z": "#1f77b4", "primal_x": "#ff7f0e", "probe_z": "#2ca02c",
              "probe_x": "#d62728", "meta_w1": "#9467bd", "zeroshot_wx": "#8c564b",
              "pc2": "#e377c2", "pc1": "#17becf"}

    fig, ax = plt.subplots(figsize=(14, 6))
    xpos = np.arange(len(PAIRS))
    n_dirs = len(dir_names)
    w = 0.8 / n_dirs
    for i, dname in enumerate(dir_names):
        slopes = []
        for pname in PAIRS:
            if pname in data and dname in data[pname]:
                slopes.append(abs(data[pname][dname]["slope_ld"]))
            else:
                slopes.append(0)
        ax.bar(xpos + i * w - 0.4 + w / 2, slopes, w,
               label=dname, color=colors.get(dname, "gray"))
    ax.set_xticks(xpos)
    ax.set_xticklabels(PAIRS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("|slope(ld vs α)|")
    ax.set_title("v7 clean-grid steering |slopes| per direction per pair")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "steering_slopes_all_directions_clean.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'steering_slopes_all_directions_clean.png'}")


# ---------------------------------------------------------------------------
# 8. Entropy vs alpha (how much does steering perturb the distribution?)
# ---------------------------------------------------------------------------
def plot_entropy_vs_alpha():
    data = json.loads((ROOT / "results/v7_steering/clean_direction_comparison.json").read_text())
    first_pair = next(iter(data))
    dir_names = list(data[first_pair].keys())
    alphas = sorted(float(a) for a in data[first_pair][dir_names[0]]["curve"].keys())

    colors = {"primal_z": "#1f77b4", "primal_x": "#ff7f0e", "probe_z": "#2ca02c",
              "probe_x": "#d62728", "meta_w1": "#9467bd", "zeroshot_wx": "#8c564b",
              "pc2": "#e377c2", "pc1": "#17becf"}

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pname in zip(axes.ravel(), PAIRS):
        if pname not in data:
            ax.set_visible(False)
            continue
        for dname in dir_names:
            if dname not in data[pname]:
                continue
            curve = data[pname][dname]["curve"]
            ys = [curve[str(a)]["entropy_mean"] for a in alphas]
            ax.plot(alphas, ys, marker="o", color=colors.get(dname, "gray"),
                    label=dname, lw=1.5, markersize=3)
        ax.set_title(pname, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("α", fontsize=8)
        ax.set_ylabel("entropy", fontsize=8)
    axes.ravel()[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("v7 clean-grid: entropy vs steering α (E4B layer 32)")
    fig.tight_layout()
    fig.savefig(OUT / "entropy_vs_alpha_clean.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {OUT / 'entropy_vs_alpha_clean.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Regenerating v7 figures from JSON...")
    plot_confound_matrix()
    plot_cross_grid_stability()
    plot_inlp()
    plot_fisher_entropy()
    plot_steering_curves()
    plot_transfer_heatmap()
    plot_steering_slopes()
    plot_entropy_vs_alpha()
    # Clean up orphaned figures that no script produces anymore
    for orphan in ["cos_wz_wld_4metrics_clean.png", "transfer_matrix_detailed.png"]:
        p = OUT / orphan
        if p.exists():
            p.unlink()
            print(f"  removed orphaned {orphan}")
    print("done")


if __name__ == "__main__":
    main()
