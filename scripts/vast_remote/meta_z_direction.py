#!/usr/bin/env python3
"""Meta z-direction: SVD of stacked, sign-aligned PC1 vectors across 8 pairs.

Pipeline:
  1. For each pair, load implicit_late.npz -> cell-mean activations (bin by x, mu).
  2. PC1 of cell means = per-pair z-direction (2560-d).
  3. Sign-align each PC1 so its projection positively correlates with z.
  4. SVD the 8×2560 stack -> right singular vectors = ordered "shared directions".
  5. Meta-direction w1 = top right singular vector.
  6. For each pair, compare:
        - R²(proj_onto_w1       ~ z)   = how much z a SHARED direction captures
        - R²(proj_onto_own_PC1  ~ z)   = how much z the CONCEPT-SPECIFIC direction captures
     The gap tells us how concept-specific the z-encoding really is.

Outputs:
  - results/v4_adjpairs_analysis/figures/meta_z_direction.png
  - results/v4_adjpairs_analysis/meta_z_direction.json
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

REPO = Path("/workspace/repo")
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import generate_all_prompts  # noqa: E402

trials = generate_all_prompts()
id2trial = {t["id"]: t for t in trials}
PAIRS_ORDER = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]


def lstsq_r2(X, y):
    Xi = np.column_stack([X, np.ones(len(y))])
    coef, _, _, _ = np.linalg.lstsq(Xi, y, rcond=None)
    yhat = Xi @ coef
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else None


# --- Gather per-pair cell means and PC1 directions ---
per_pair = {}

for pname in PAIRS_ORDER:
    npz_path = REPO / "results" / "v4_adjpairs" / f"e4b_{pname}_implicit_late.npz"
    if not npz_path.exists():
        continue
    d = np.load(npz_path)
    acts, ids = d["activations"], d["ids"]
    xs  = np.array([id2trial[i]["x"]  for i in ids])
    mus = np.array([id2trial[i]["mu"] for i in ids])
    zs  = np.array([id2trial[i]["z"]  for i in ids])

    x_vals  = sorted(set(xs.tolist()))
    mu_vals = sorted(set(mus.tolist()))
    cell_acts, cx, cm, cz = [], [], [], []
    for x in x_vals:
        for mu in mu_vals:
            m = (xs == x) & (mus == mu)
            if m.sum() == 0:
                continue
            cell_acts.append(acts[m].mean(0))
            cx.append(x); cm.append(mu); cz.append(zs[m].mean())
    cell_acts = np.stack(cell_acts)
    cx = np.array(cx, float); cm = np.array(cm, float); cz = np.array(cz, float)

    centered = cell_acts - cell_acts.mean(0)
    pca = PCA(n_components=1)
    pca.fit(centered)
    pc1 = pca.components_[0]

    # Sign-align: make (centered @ pc1) positively correlate with z
    proj = centered @ pc1
    if np.corrcoef(proj, cz)[0, 1] < 0:
        pc1 = -pc1
        proj = -proj

    r2_own = lstsq_r2(proj.reshape(-1, 1), cz)
    per_pair[pname] = {
        "pc1": pc1,
        "centered": centered,
        "mean": cell_acts.mean(0),
        "cz": cz,
        "cx": cx,
        "cm": cm,
        "r2_own_pc1": r2_own,
    }

names = [n for n in PAIRS_ORDER if n in per_pair]
print(f"Loaded {len(names)} pairs: {names}")

# --- SVD the stacked PC1 matrix ---
V = np.stack([per_pair[n]["pc1"] for n in names])   # (n_pairs, d)
U, S, Wt = np.linalg.svd(V, full_matrices=False)
W = Wt.T                                            # (d, n_pairs)  each column = shared dir
var_ratio = (S ** 2) / (S ** 2).sum()
print(f"SVD singular values: {S}")
print(f"Variance ratio: {var_ratio}")

w1 = W[:, 0]
w1 = w1 / np.linalg.norm(w1)

# --- Project each pair's cells onto meta direction and measure R²(proj ~ z) ---
r2_meta = {}
r2_own = {}
cos_with_meta = {}
for n in names:
    centered = per_pair[n]["centered"]
    cz = per_pair[n]["cz"]

    proj_meta = centered @ w1
    # Sign-align meta projection per pair for fair R² (R² is sign-invariant anyway)
    r2_meta[n] = lstsq_r2(proj_meta.reshape(-1, 1), cz)
    r2_own[n] = per_pair[n]["r2_own_pc1"]

    v = per_pair[n]["pc1"]
    cos_with_meta[n] = float(abs(np.dot(v / np.linalg.norm(v), w1)))

# --- Figure ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6))

# Panel 1: scree of the SVD
ax1.bar(range(1, len(S) + 1), var_ratio * 100, color="#3b6fb5")
ax1.set_xlabel("Shared direction index")
ax1.set_ylabel("Variance explained across pair-PC1s (%)")
ax1.set_title("SVD of stacked PC1 vectors\n(how much of the 8 per-pair z-axes\ncollapses onto a single shared direction)")
ax1.set_xticks(range(1, len(S) + 1))
for i, v in enumerate(var_ratio * 100):
    ax1.text(i + 1, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
ax1.set_ylim(0, max(var_ratio * 100) * 1.15)
ax1.grid(True, axis="y", alpha=0.3)

# Panel 2: R²(own PC1 ~ z) vs R²(meta dir ~ z) per pair
xs = np.arange(len(names))
w = 0.38
own_vals  = [r2_own[n]  if r2_own[n]  is not None else 0 for n in names]
meta_vals = [r2_meta[n] if r2_meta[n] is not None else 0 for n in names]
ax2.bar(xs - w / 2, own_vals,  w, label="R²(own PC1 ~ z)",      color="#2ca02c")
ax2.bar(xs + w / 2, meta_vals, w, label="R²(meta w₁ ~ z)",       color="#d62728")
ax2.set_xticks(xs); ax2.set_xticklabels(names, rotation=45, ha="right")
ax2.set_ylabel("R²"); ax2.set_ylim(0, 1.05)
ax2.legend(loc="lower right", fontsize=10)
ax2.set_title("z-encoding: concept-specific axis vs shared meta-axis\n(gap = part of z that each concept puts in a private direction)")
ax2.grid(True, axis="y", alpha=0.3)

# Panel 3: |cos(own_PC1, w1)| per pair
cos_vals = [cos_with_meta[n] for n in names]
ax3.bar(xs, cos_vals, color="#9467bd")
ax3.set_xticks(xs); ax3.set_xticklabels(names, rotation=45, ha="right")
ax3.set_ylabel("|cos(own_PC1, meta_w₁)|")
ax3.set_ylim(0, 1.05)
ax3.set_title("Alignment of each pair's z-axis with the meta direction")
for i, v in enumerate(cos_vals):
    ax3.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax3.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
out_dir = REPO / "results" / "v4_adjpairs_analysis" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "meta_z_direction.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Saved {out_path}")

# --- JSON summary ---
summary = {
    "pairs": names,
    "svd_singular_values": S.tolist(),
    "svd_variance_ratio":  var_ratio.tolist(),
    "top_shared_variance_pct": float(var_ratio[0] * 100),
    "per_pair": {
        n: {
            "r2_own_pc1_vs_z": r2_own[n],
            "r2_meta_w1_vs_z": r2_meta[n],
            "cos_own_pc1_with_meta": cos_with_meta[n],
        }
        for n in names
    },
}
json_path = REPO / "results" / "v4_adjpairs_analysis" / "meta_z_direction.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {json_path}")

# --- Console table ---
print("\n=== Per-pair: own vs meta z-axis ===")
print(f"{'pair':<12} {'R²(own~z)':>10} {'R²(meta~z)':>11} {'|cos(own,meta)|':>16}")
for n in names:
    ro = r2_own[n]
    rm = r2_meta[n]
    c  = cos_with_meta[n]
    ro_s = f"{ro:.3f}" if ro is not None else "  -  "
    rm_s = f"{rm:.3f}" if rm is not None else "  -  "
    print(f"{n:<12} {ro_s:>10} {rm_s:>11} {c:>16.3f}")

print(f"\nTop shared direction explains {var_ratio[0]*100:.1f}% of the variance")
print(f"across the 8 per-pair z-axes (|cos|-squared sum interpretation).")
