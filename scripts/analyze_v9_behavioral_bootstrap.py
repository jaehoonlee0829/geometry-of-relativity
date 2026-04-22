"""v9 robustness for §10.1 — bootstrap CI on R + Grid-B simulation.

Addresses critic concerns:
  (STATS-1) R = −slope(μ)/slope(x) on Grid B may be structurally biased
            toward 1 for any pure-z responder — simulate pure-z and pure-x
            synthetic models on each pair's exact design matrix and check
            the R that these ideal cases produce.
  (SCI-C1) Bootstrap a 95% CI on R per pair to see whether `age`'s R≈1.03
            vs E4B's smaller R is statistically distinguishable.

Writes
  results/v9_gemma2/behavioral_bootstrap.json
  figures/v9/behavioral_R_ci_vs_simulation.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import PAIRS, LOG_SPACE_PAIRS  # noqa: E402

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"
N_BOOT = 1000
RNG = np.random.default_rng(0)


def load_pair_rows(pair_name: str):
    trials = {}
    with (RES_DIR / "gemma2_trials.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["pair"] == pair_name:
                trials[t["id"]] = t
    log_rows = [json.loads(l)
                for l in (RES_DIR / f"gemma2_{pair_name}_logits.jsonl").open()]
    rows = []
    for r in log_rows:
        if r["id"] not in trials:
            continue
        t = trials[r["id"]]
        rows.append({"x": t["x"], "mu": t["mu"], "z": t["z"],
                     "logit_diff": r["logit_diff"]})
    return rows


def R_from_rows(rows, pair_name):
    xs = np.array([r["x"] for r in rows])
    mus = np.array([r["mu"] for r in rows])
    lds = np.array([r["logit_diff"] for r in rows])
    if pair_name in LOG_SPACE_PAIRS:
        xs = np.log(xs); mus = np.log(mus)
    A = np.column_stack([np.ones_like(lds), xs, mus])
    coef, *_ = np.linalg.lstsq(A, lds, rcond=None)
    slope_x = float(coef[1]); slope_mu = float(coef[2])
    R = -slope_mu / slope_x if abs(slope_x) > 1e-9 else None
    return R, slope_x, slope_mu


def bootstrap_R(rows, pair_name, n=N_BOOT):
    rs = []
    idxs = np.arange(len(rows))
    for _ in range(n):
        sample_idx = RNG.choice(idxs, size=len(idxs), replace=True)
        sample = [rows[i] for i in sample_idx]
        R, *_ = R_from_rows(sample, pair_name)
        if R is not None and np.isfinite(R):
            rs.append(R)
    rs = np.array(rs)
    return {
        "R_mean": float(rs.mean()),
        "R_q025": float(np.quantile(rs, 0.025)),
        "R_q975": float(np.quantile(rs, 0.975)),
        "R_std": float(rs.std()),
        "n_valid": int(len(rs)),
    }


def simulate_R(rows, pair_name, kind: str, noise_std: float = 0.1):
    """Given the grid (x, μ, z), simulate ld = β·<driver> + noise and report R."""
    xs = np.array([r["x"] for r in rows])
    mus = np.array([r["mu"] for r in rows])
    zs = np.array([r["z"] for r in rows])

    if kind == "pure_z":
        driver = zs
    elif kind == "pure_x":
        driver = xs
    elif kind == "pure_mu":
        driver = mus
    elif kind == "half_x_half_z":
        driver = 0.5 * (xs - xs.mean()) / (xs.std() + 1e-9) + \
                 0.5 * (zs - zs.mean()) / (zs.std() + 1e-9)
    else:
        raise ValueError(kind)

    sigma = driver.std() + 1e-9
    ld = driver / sigma  # unit-scale signal
    ld = ld + RNG.normal(0, noise_std, size=ld.shape)
    sim_rows = [{"x": xs[i], "mu": mus[i], "z": zs[i],
                 "logit_diff": ld[i]} for i in range(len(xs))]
    R, *_ = R_from_rows(sim_rows, pair_name)
    return float(R) if R is not None else float("nan")


def main():
    results = {}
    for p in PAIRS:
        rows = load_pair_rows(p.name)
        R_real, sx, sm = R_from_rows(rows, p.name)
        boot = bootstrap_R(rows, p.name)
        sim = {
            kind: simulate_R(rows, p.name, kind, noise_std=0.1)
            for kind in ["pure_z", "pure_x", "pure_mu", "half_x_half_z"]
        }
        results[p.name] = {
            "R_observed": float(R_real),
            "slope_x": sx, "slope_mu": sm,
            "bootstrap": boot,
            "simulation_noise01": sim,
        }
        print(f"{p.name:12s}  R={R_real:+.3f}  "
              f"CI95=[{boot['R_q025']:+.3f}, {boot['R_q975']:+.3f}]  "
              f"sim pure_z={sim['pure_z']:+.3f}  pure_x={sim['pure_x']:+.3f}  "
              f"half={sim['half_x_half_z']:+.3f}")

    out = RES_DIR / "behavioral_bootstrap.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    # Plot
    pairs = [p.name for p in PAIRS]
    xs = np.arange(len(pairs))
    R_obs = [results[n]["R_observed"] for n in pairs]
    lo = [results[n]["bootstrap"]["R_q025"] for n in pairs]
    hi = [results[n]["bootstrap"]["R_q975"] for n in pairs]
    sim_pz = [results[n]["simulation_noise01"]["pure_z"] for n in pairs]
    sim_px = [results[n]["simulation_noise01"]["pure_x"] for n in pairs]
    sim_half = [results[n]["simulation_noise01"]["half_x_half_z"] for n in pairs]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.errorbar(xs, R_obs,
                yerr=[np.array(R_obs) - np.array(lo),
                      np.array(hi) - np.array(R_obs)],
                fmt="o", color="C0", ecolor="C0", capsize=3,
                label="observed R ± 95% bootstrap CI", markersize=7)
    ax.scatter(xs, sim_pz, marker="^", color="C2", s=50,
               label="simulated pure-z model", zorder=5)
    ax.scatter(xs, sim_px, marker="v", color="C3", s=50,
               label="simulated pure-x model", zorder=5)
    ax.scatter(xs, sim_half, marker="s", color="C1", s=50,
               label="simulated half-x + half-z", zorder=5)
    ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.4)
    ax.axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.4)
    ax.axhline(0.3, color="red", ls="--", lw=0.8, alpha=0.5,
               label="v9 acceptance threshold R=0.3")
    ax.set_xticks(xs); ax.set_xticklabels(pairs, rotation=30, ha="right")
    ax.set_ylabel("R = −slope(μ) / slope(x)")
    ax.set_title(
        "v9 §10.1 robustness — observed R (with bootstrap CI) vs simulated\n"
        "idealized models on each pair's Grid B design.   "
        "(pure-z ≈ 1 and pure-x ≈ 0 confirms R is a well-defined statistic.)",
        fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "behavioral_R_ci_vs_simulation.png", dpi=140)
    print(f"  wrote {FIG_DIR}/behavioral_R_ci_vs_simulation.png")


if __name__ == "__main__":
    main()
