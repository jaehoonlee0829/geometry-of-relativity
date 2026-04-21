"""Exp 3a: compute Σ⁻¹ (data-covariance) metric cosines and persist to JSON.

Re-runs the probe training from analyze_v4_adjpairs but adds Σ⁻¹ cosines:
    cos_Σ⁻¹(w_adj, w_z), cos_Σ⁻¹(w_adj, w_x), cos_Σ⁻¹(w_z, w_x),
    cos_Σ⁻¹(w_z, w_ld)

For each pair × layer. Σ = cov(implicit_activations); regularize with λI.
Writes: results/v4_adjpairs_analysis/sigma_inv_cosines.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ADJPAIRS = Path("results/v4_adjpairs")
OUT = Path("results/v4_adjpairs_analysis")

PAIRS = ["height", "age", "experience", "size", "speed", "wealth", "weight", "bmi_abs"]
LAYERS = ["mid", "late"]


def load_pair_layer(pair: str, layer: str):
    trials_by_id = {json.loads(l)["id"]: json.loads(l) for l in (ADJPAIRS / "e4b_trials.jsonl").open()}
    ld_by_id = {r["id"]: r["logit_diff"] for r in map(json.loads, (ADJPAIRS / f"e4b_{pair}_implicit_logits.jsonl").open())}
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


def train_ridge(X, y, alpha=1.0, seed=0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    w = model.coef_.astype(np.float64)
    # cv R²
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        yp = m.predict(X[te])
        ss_res = ((y[te] - yp) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return w, float(np.mean(scores))


def eucl_cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def sigma_inv_solve(acts: np.ndarray, reg_scale: float = 1e-3):
    Sigma = np.cov(acts, rowvar=False)
    d = Sigma.shape[0]
    reg = reg_scale * np.trace(Sigma) / d
    Sigma_reg = Sigma + reg * np.eye(d)
    L = cho_factor(Sigma_reg, lower=True)

    def solve(v):
        return cho_solve(L, v)

    return solve


def sigma_inv_cos(u, v, Sinv):
    gu = Sinv(u)
    gv = Sinv(v)
    return float(np.dot(u, gv) / (np.sqrt(np.dot(u, gu)) * np.sqrt(np.dot(v, gv)) + 1e-12))


def main():
    result = {}
    for pair in PAIRS:
        result[pair] = {}
        for layer in LAYERS:
            print(f"[{pair}/{layer}]", end=" ", flush=True)
            acts, xs, zs, ld = load_pair_layer(pair, layer)
            # Binary adjective label: sign(z) — the "tautological" probe per Exp 6
            y_adj = np.sign(zs).astype(np.float64)
            y_adj[y_adj == 0] = 1.0
            w_x, _ = train_ridge(acts, xs)
            w_z, _ = train_ridge(acts, zs)
            w_adj, _ = train_ridge(acts, y_adj)
            w_ld, _ = train_ridge(acts, ld)
            Sinv = sigma_inv_solve(acts)
            result[pair][layer] = {
                "cos_adj_z_euclid": eucl_cos(w_adj, w_z),
                "cos_adj_x_euclid": eucl_cos(w_adj, w_x),
                "cos_z_x_euclid": eucl_cos(w_z, w_x),
                "cos_z_ld_euclid": eucl_cos(w_z, w_ld),
                "cos_adj_z_sigma_inv": sigma_inv_cos(w_adj, w_z, Sinv),
                "cos_adj_x_sigma_inv": sigma_inv_cos(w_adj, w_x, Sinv),
                "cos_z_x_sigma_inv": sigma_inv_cos(w_z, w_x, Sinv),
                "cos_z_ld_sigma_inv": sigma_inv_cos(w_z, w_ld, Sinv),
                "n": int(acts.shape[0]),
            }
            r = result[pair][layer]
            print(f"Σ⁻¹ cos(adj,z)={r['cos_adj_z_sigma_inv']:+.3f} (E={r['cos_adj_z_euclid']:+.3f}); "
                  f"Σ⁻¹ cos(z,ld)={r['cos_z_ld_sigma_inv']:+.3f} (E={r['cos_z_ld_euclid']:+.3f})")

    (OUT / "sigma_inv_cosines.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT/'sigma_inv_cosines.json'}")

    # Summary: per layer, average Σ⁻¹ lift
    print("\nLift: Σ⁻¹ − Euclidean (averaged across 8 pairs, |cos|):")
    for layer in LAYERS:
        ae = np.mean([abs(result[p][layer]["cos_adj_z_euclid"]) for p in PAIRS])
        as_ = np.mean([abs(result[p][layer]["cos_adj_z_sigma_inv"]) for p in PAIRS])
        ze = np.mean([abs(result[p][layer]["cos_z_ld_euclid"]) for p in PAIRS])
        zs = np.mean([abs(result[p][layer]["cos_z_ld_sigma_inv"]) for p in PAIRS])
        print(f"  [{layer}] |cos(adj,z)|: E={ae:.3f} → Σ⁻¹={as_:.3f}  (lift {as_-ae:+.3f})")
        print(f"  [{layer}] |cos(z,ld)|:  E={ze:.3f} → Σ⁻¹={zs:.3f}  (lift {zs-ze:+.3f})")


if __name__ == "__main__":
    main()
