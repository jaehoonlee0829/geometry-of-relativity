"""v7 Priority 2: clean confound audit — (x, z) grid directions.

Computes 7 directions per pair from Grid B activations and compares to
the v6 Grid A directions. Tests whether:
  1. cos(primal_z, primal_x) drops from 0.91 (v6) to lower (v7)
  2. PC1 becomes less confounded with x
  3. meta_w1 changes

Outputs:
  results/v7_analysis/direction_confound_audit_clean.json
  figures/v7/direction_confound_matrix_gridB.png
  figures/v7/cos_primal_z_clean_vs_v6.png  (cross-grid comparison)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_v4_adjpairs import PAIRS  # noqa: E402

V7 = REPO / "results" / "v7_xz_grid"
V4 = REPO / "results" / "v4_adjpairs"
ZS_EXPANDED = REPO / "results" / "v4_zeroshot_expanded"
OUT = REPO / "results" / "v7_analysis"
OUT_FIG = REPO / "figures" / "v7"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

DIR_NAMES = ["primal_z", "primal_x", "probe_z", "probe_x", "pc1", "pc2", "zeroshot_wx"]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def load_grid(pair_name: str, grid_root: Path, trials_fname: str, layer: str):
    trials_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in (grid_root / trials_fname).open()}
    npz_name = f"e4b_{pair_name}_{'implicit_' if grid_root.name == 'v4_adjpairs' else ''}{layer}.npz"
    npz = np.load(grid_root / npz_name, allow_pickle=True)
    acts = npz["activations"].astype(np.float64)
    ids = [str(s) for s in npz["ids"]]
    xs = np.array([trials_by_id[i]["x"] for i in ids])
    zs = np.array([trials_by_id[i]["z"] for i in ids])
    mus = np.array([trials_by_id[i]["mu"] for i in ids])
    return acts, xs, zs, mus


def compute_7_dirs(acts, xs, zs, zs_npz_acts=None, zs_npz_xs=None):
    primal_z = acts[zs > +1.0].mean(0) - acts[zs < -1.0].mean(0)
    x_hi, x_lo = np.percentile(xs, 75), np.percentile(xs, 25)
    primal_x = acts[xs >= x_hi].mean(0) - acts[xs <= x_lo].mean(0)
    w_z = Ridge(alpha=1.0).fit(acts, zs).coef_
    w_x = Ridge(alpha=1.0).fit(acts, xs).coef_
    centered = acts - acts.mean(0)
    pca = PCA(n_components=2).fit(centered)
    pc1, pc2 = pca.components_[0], pca.components_[1]
    if np.corrcoef(centered @ pc1, zs)[0, 1] < 0:
        pc1, pc2 = -pc1, -pc2
    if zs_npz_acts is not None:
        w_x_zs = Ridge(alpha=1.0).fit(zs_npz_acts, zs_npz_xs).coef_
    else:
        w_x_zs = np.zeros_like(w_x)
    return {
        "primal_z": unit(primal_z),
        "primal_x": unit(primal_x),
        "probe_z":  unit(w_z),
        "probe_x":  unit(w_x),
        "pc1":      unit(pc1),
        "pc2":      unit(pc2),
        "zeroshot_wx": unit(w_x_zs),
    }


def main():
    # Load zero-shot expanded activations once (for zeroshot_wx direction)
    zs_trials = {json.loads(l)["id"]: json.loads(l)
                 for l in (ZS_EXPANDED / "e4b_trials.jsonl").open()}

    result = {"per_pair_gridB": {}, "per_pair_gridA": {}, "cross_grid": {}}
    confound_xz_gridA = {}
    confound_xz_gridB = {}
    for pair_obj in PAIRS:
        pname = pair_obj.name
        # Grid B (v7)
        acts_B, xs_B, zs_B, mus_B = load_grid(pname, V7, "e4b_trials.jsonl", "late")
        # Compute correlation between x and z in Grid B (should be ~0 by construction)
        corr_xz_B = float(np.corrcoef(xs_B, zs_B)[0, 1])
        # Grid A (v4)
        acts_A, xs_A, zs_A, mus_A = load_grid(pname, V4, "e4b_trials.jsonl", "late")
        corr_xz_A = float(np.corrcoef(xs_A, zs_A)[0, 1])

        # Zero-shot activations per pair
        zs_npz = np.load(ZS_EXPANDED / f"e4b_{pname}_late.npz", allow_pickle=True)
        zs_acts = zs_npz["activations"].astype(np.float64)
        zs_ids = [str(s) for s in zs_npz["ids"]]
        zs_xs = np.array([zs_trials[i]["x"] for i in zs_ids])

        dirs_A = compute_7_dirs(acts_A, xs_A, zs_A, zs_acts, zs_xs)
        dirs_B = compute_7_dirs(acts_B, xs_B, zs_B, zs_acts, zs_xs)

        # 7×7 matrix per grid
        M_B = np.array([[float(np.dot(dirs_B[a], dirs_B[b])) for b in DIR_NAMES] for a in DIR_NAMES])
        M_A = np.array([[float(np.dot(dirs_A[a], dirs_A[b])) for b in DIR_NAMES] for a in DIR_NAMES])
        result["per_pair_gridB"][pname] = M_B.tolist()
        result["per_pair_gridA"][pname] = M_A.tolist()

        # Cross-grid cosines
        cross = {d: float(np.dot(dirs_A[d], dirs_B[d])) for d in DIR_NAMES}
        result["cross_grid"][pname] = {
            "corr_xz_gridA": corr_xz_A,
            "corr_xz_gridB": corr_xz_B,
            "cos_gridA_gridB_per_direction": cross,
        }
        confound_xz_gridA[pname] = corr_xz_A
        confound_xz_gridB[pname] = corr_xz_B

        print(f"[{pname:12s}]  corr(x,z):  A={corr_xz_A:+.3f}  B={corr_xz_B:+.3f}   "
              f"cos(primal_z): A→B={cross['primal_z']:+.3f}   "
              f"cos(primal_z, primal_x): A={M_A[0,1]:+.3f}  B={M_B[0,1]:+.3f}")

    # Aggregate: mean matrices across 8 pairs
    mean_A = np.mean([np.array(M) for M in result["per_pair_gridA"].values()], axis=0)
    mean_B = np.mean([np.array(M) for M in result["per_pair_gridB"].values()], axis=0)
    result["mean_cos_gridA"] = mean_A.tolist()
    result["mean_cos_gridB"] = mean_B.tolist()

    # Replicate v6 meta_w1 on both grids (via PCA stacked SVD)
    for grid_name, which in [("gridA", "pc1_gridA"), ("gridB", "pc1_gridB")]:
        # we need PC1 per pair from each grid
        pc1_stack = []
        for pair_obj in PAIRS:
            pname = pair_obj.name
            if grid_name == "gridA":
                acts, xs, zs, _ = load_grid(pname, V4, "e4b_trials.jsonl", "late")
            else:
                acts, xs, zs, _ = load_grid(pname, V7, "e4b_trials.jsonl", "late")
            centered = acts - acts.mean(0)
            pc1 = PCA(1).fit(centered).components_[0]
            if np.corrcoef(centered @ pc1, zs)[0, 1] < 0: pc1 = -pc1
            pc1_stack.append(pc1 / np.linalg.norm(pc1))
        V = np.stack(pc1_stack)
        _, _, Wt = np.linalg.svd(V, full_matrices=False)
        w1 = Wt[0] / np.linalg.norm(Wt[0])
        # cos with mean primal_z
        mean_primal_z = np.mean([np.array([[dirs_B[d] for d in [dir_name]] for dir_name in [dir_name]]) for pair_obj in PAIRS for dir_name in ["primal_z"]], axis=0) if grid_name == "gridB" else None
        result[f"meta_w1_{grid_name}"] = w1.tolist()
        result[f"meta_w1_{grid_name}_cos_with_PC1s"] = [float(np.dot(w1, pc)) for pc in pc1_stack]

    (OUT / "direction_confound_audit_clean.json").write_text(json.dumps(result, indent=2))

    # Print summary
    print("\n=== MEAN |cos| ACROSS 8 PAIRS ===")
    print(f"{'':15s}" + "  ".join(f"{d:>11s}" for d in DIR_NAMES))
    print("-- Grid A (x, μ) — v4/v6 — CONFOUNDED --")
    for i, a in enumerate(DIR_NAMES):
        row = "  ".join(f"{abs(mean_A[i][j]):11.3f}" for j in range(len(DIR_NAMES)))
        print(f"  {a:12s} {row}")
    print("-- Grid B (x, z) — v7 — CLEAN --")
    for i, a in enumerate(DIR_NAMES):
        row = "  ".join(f"{abs(mean_B[i][j]):11.3f}" for j in range(len(DIR_NAMES)))
        print(f"  {a:12s} {row}")
    print()
    print(f"corr(x, z) in each pair (should be ~0 for Grid B):")
    for p in PAIRS:
        print(f"  {p.name:12s}  A={confound_xz_gridA[p.name]:+.3f}  B={confound_xz_gridB[p.name]:+.3f}")

    # Figure: Grid B matrix heatmap (clean grid only)
    fig, ax = plt.subplots(figsize=(9, 7))
    M = mean_B
    im = ax.imshow(np.abs(M), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(DIR_NAMES))); ax.set_xticklabels(DIR_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(DIR_NAMES))); ax.set_yticklabels(DIR_NAMES)
    for i in range(len(DIR_NAMES)):
        for j in range(len(DIR_NAMES)):
            ax.text(j, i, f"{abs(M[i][j]):.2f}", ha="center", va="center",
                    color="white" if abs(M[i][j]) > 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Direction cosines — Grid B (x, z) — v7 clean", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "direction_confound_matrix_gridB.png", dpi=140)
    plt.close(fig)

    # Cross-grid: for each direction, plot cos(grid A direction, grid B direction)
    fig, ax = plt.subplots(figsize=(11, 5))
    xpos = np.arange(len(DIR_NAMES))
    w = 1.0 / (len(PAIRS) + 1)
    for i, p in enumerate(PAIRS):
        pname = p.name
        vals = [abs(result["cross_grid"][pname]["cos_gridA_gridB_per_direction"][d]) for d in DIR_NAMES]
        ax.bar(xpos + i*w, vals, w, label=pname)
    ax.set_xticks(xpos + 3.5*w); ax.set_xticklabels(DIR_NAMES, rotation=30, ha="right")
    ax.set_ylabel("|cos| (same direction computed on Grid A vs Grid B, same pair)")
    ax.set_title("How stable is each direction across grid designs?")
    ax.legend(fontsize=7, ncol=4); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "cross_grid_direction_stability.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {OUT/'direction_confound_audit_clean.json'}")


if __name__ == "__main__":
    main()
