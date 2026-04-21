"""v6 post-hoc: confirm critic-flagged confounds in the 7 "steering directions".

Computes cosines between the 7 candidate directions introduced in
exp_v6_7dir_steering.py to audit how independent they actually are.

Findings this script confirms:
  1. primal_z ≈ primal_x  (by construction, since z = (x-μ)/σ and sample
     z>+1 biases toward upper x quartile): cos 0.71-0.99
  2. meta_w1 ≈ −mean(primal_z across pairs): cos ≈ −0.98
  3. probe_z ⊥ primal_z:  cos 0.05-0.10   (Ridge shrinkage picks
     low-variance directions; diff-of-means picks high-variance ones)
  4. PC1 ≈ primal_z: cos 0.77-0.99  (PC1 captures the dominant linear
     z-variation in cell-mean activations)

Writes:
  results/v6_steering/direction_confound_audit.json
  figures/v6/direction_confound_matrix.png
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

ADJPAIRS = REPO / "results" / "v4_adjpairs"
ZS_EXPANDED = REPO / "results" / "v4_zeroshot_expanded"
OUT = REPO / "results" / "v6_steering"
OUT_FIG = REPO / "figures" / "v6"

DIR_NAMES = ["primal_z", "primal_x", "probe_z", "probe_x", "pc1", "pc2", "zeroshot_wx"]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def compute_directions(pair_name: str, trials_by_id):
    npz = np.load(ADJPAIRS / f"e4b_{pair_name}_implicit_late.npz", allow_pickle=True)
    acts = npz["activations"].astype(np.float64)
    ids = [str(s) for s in npz["ids"]]
    xs = np.array([trials_by_id[i]["x"] for i in ids])
    zs = np.array([trials_by_id[i]["z"] for i in ids])

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

    zs_npz = np.load(ZS_EXPANDED / f"e4b_{pair_name}_late.npz", allow_pickle=True)
    zs_trials = {json.loads(l)["id"]: json.loads(l)
                 for l in (ZS_EXPANDED / "e4b_trials.jsonl").open()}
    zs_acts = zs_npz["activations"].astype(np.float64)
    zs_ids = [str(s) for s in zs_npz["ids"]]
    zs_xs = np.array([zs_trials[i]["x"] for i in zs_ids])
    w_x_zs = Ridge(alpha=1.0).fit(zs_acts, zs_xs).coef_

    dirs = {
        "primal_z": unit(primal_z),
        "primal_x": unit(primal_x),
        "probe_z":  unit(w_z),
        "probe_x":  unit(w_x),
        "pc1":      unit(pc1),
        "pc2":      unit(pc2),
        "zeroshot_wx": unit(w_x_zs),
    }
    return dirs


def main():
    trials_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in (ADJPAIRS / "e4b_trials.jsonl").open()}
    per_pair: dict[str, dict] = {}
    for p_obj in PAIRS:
        dirs = compute_directions(p_obj.name, trials_by_id)
        M = np.array([[float(np.dot(dirs[a], dirs[b])) for b in DIR_NAMES] for a in DIR_NAMES])
        per_pair[p_obj.name] = M.tolist()

    # Replicate v6 meta_w1: SVD of stacked sign-aligned PC1s
    pc1_list = []
    primal_z_list = []
    for p_obj in PAIRS:
        dirs = compute_directions(p_obj.name, trials_by_id)
        pc1_list.append(dirs["pc1"])
        primal_z_list.append(dirs["primal_z"])
    V = np.stack(pc1_list)
    _, _, Wt = np.linalg.svd(V, full_matrices=False)
    w1 = Wt[0]; w1 = w1 / np.linalg.norm(w1)

    mean_primal_z = np.mean(primal_z_list, axis=0)
    mean_primal_z_unit = unit(mean_primal_z)

    cos_w1_mean_primal = float(np.dot(w1, mean_primal_z_unit))
    cos_w1_per_pair_primal = {
        p.name: float(np.dot(w1, primal_z_list[i])) for i, p in enumerate(PAIRS)
    }

    summary = {
        "direction_names": DIR_NAMES,
        "per_pair_cosine_matrices": per_pair,
        "meta_w1_vs_mean_primal_z": cos_w1_mean_primal,
        "meta_w1_vs_per_pair_primal_z": cos_w1_per_pair_primal,
        "mean_offdiag_abs_cos_over_pairs": {},
    }

    # Aggregate: mean |cos| per direction-pair across 8 concepts
    mean_matrix = np.mean([np.array(M) for M in per_pair.values()], axis=0)
    summary["mean_cos_across_pairs"] = mean_matrix.tolist()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "direction_confound_audit.json").write_text(json.dumps(summary, indent=2))

    # Print a nice table
    print(f"Mean cos across 8 pairs (each cell is |cos| averaged across pairs):")
    print("               " + "  ".join(f"{n:>12s}" for n in DIR_NAMES))
    for i, a in enumerate(DIR_NAMES):
        row = "  ".join(f"{abs(mean_matrix[i][j]):12.3f}" for j in range(len(DIR_NAMES)))
        print(f"  {a:12s}  {row}")

    print(f"\ncos(meta_w1, mean(primal_z across pairs)) = {cos_w1_mean_primal:+.4f}")

    # Plot heatmap
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(np.abs(mean_matrix), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(DIR_NAMES))); ax.set_xticklabels(DIR_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(DIR_NAMES))); ax.set_yticklabels(DIR_NAMES)
    for i in range(len(DIR_NAMES)):
        for j in range(len(DIR_NAMES)):
            ax.text(j, i, f"{abs(mean_matrix[i][j]):.2f}", ha="center", va="center",
                    color="white" if abs(mean_matrix[i][j]) > 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04, label="|cos| (averaged across 8 pairs)")
    ax.set_title("How independent are the 7 'steering directions'?")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "direction_confound_matrix.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'direction_confound_matrix.png'}")


if __name__ == "__main__":
    main()
