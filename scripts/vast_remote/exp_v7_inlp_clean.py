"""v7 Priority 6: INLP on Grid B (clean) activations.

Iteratively project out the clean z-direction from activations and measure
how quickly R²(z) drops. Compare to:
  - random-direction null (should keep R² stable)
  - v5 INLP on Grid A (confounded) if available

Writes:
  results/v7_analysis/inlp_clean.json
  figures/v7/inlp_clean_curves.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_v4_adjpairs import PAIRS  # noqa: E402

V7 = REPO / "results" / "v7_xz_grid"
OUT = REPO / "results" / "v7_analysis"
OUT_FIG = REPO / "figures" / "v7"
N_ITER = 8
N_RAND = 3


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def cv_r2(X, y, seed=0):
    cv = KFold(5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X):
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        yp = m.predict(X[te])
        ss_res = ((y[te] - yp) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(scores))


def project_out(X, d):
    d = unit(d)
    return X - np.outer(X @ d, d)


def inlp_run(X, y, n_iter, seed=0, random=False):
    Xi = X.copy()
    r2s = [cv_r2(Xi, y, seed=seed)]
    rng = np.random.default_rng(seed)
    for _ in range(n_iter):
        if random:
            d = unit(rng.standard_normal(X.shape[1]))
        else:
            w = Ridge(alpha=1.0).fit(Xi, y).coef_
            d = unit(w)
        Xi = project_out(Xi, d)
        r2s.append(cv_r2(Xi, y, seed=seed))
    return r2s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    trials_by_id = {json.loads(l)["id"]: json.loads(l) for l in (V7 / "e4b_trials.jsonl").open()}

    result = {}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, p_obj in zip(axes.ravel(), PAIRS):
        pn = p_obj.name
        npz = np.load(V7 / f"e4b_{pn}_late.npz", allow_pickle=True)
        acts = npz["activations"].astype(np.float64)
        ids = [str(s) for s in npz["ids"]]
        zs = np.array([trials_by_id[i]["z"] for i in ids])
        # INLP on z-direction
        r2_inlp = inlp_run(acts, zs, N_ITER, seed=0, random=False)
        # Random null: average across N_RAND seeds
        r2_rand_runs = [inlp_run(acts, zs, N_ITER, seed=100+s, random=True) for s in range(N_RAND)]
        r2_rand_mean = np.mean(r2_rand_runs, axis=0)
        r2_rand_std = np.std(r2_rand_runs, axis=0)
        result[pn] = {
            "r2_inlp_z": r2_inlp,
            "r2_random_mean": r2_rand_mean.tolist(),
            "r2_random_std":  r2_rand_std.tolist(),
            "r2_init": r2_inlp[0],
            "r2_final_inlp": r2_inlp[-1],
            "r2_final_rand": float(r2_rand_mean[-1]),
        }
        print(f"[{pn:12s}] R²(z): init={r2_inlp[0]:.3f}  "
              f"inlp_end={r2_inlp[-1]:.3f}  rand_end={r2_rand_mean[-1]:.3f}  "
              f"(Δ_inlp={r2_inlp[0]-r2_inlp[-1]:+.3f}  Δ_rand={r2_inlp[0]-r2_rand_mean[-1]:+.3f})")
        iters = np.arange(N_ITER+1)
        ax.plot(iters, r2_inlp, marker="o", lw=2, label="INLP (project out z-direction)")
        ax.errorbar(iters, r2_rand_mean, yerr=r2_rand_std, marker="s", lw=1.5,
                     alpha=0.7, capsize=3, label="random null (avg 3)")
        ax.set_title(pn, fontsize=10); ax.set_xlabel("iteration"); ax.set_ylabel("CV R²(z)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("INLP on clean (Grid B) activations — 8 iterations of z-direction projection")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "inlp_clean_curves.png", dpi=140)
    plt.close(fig)

    (OUT / "inlp_clean.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT/'inlp_clean.json'} + figure")


if __name__ == "__main__":
    main()
