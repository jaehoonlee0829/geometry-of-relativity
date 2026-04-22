"""v9 Priority 1 analysis: behavioral signal on Gemma 2 2B Grid B.

Reads the outputs of `scripts/vast_remote/extract_v9_gemma2_xz_grid.py` and:
  - Regresses logit_diff ~ x + μ per pair  →  R = -slope(μ)/slope(x)
  - Regresses logit_diff ~ z per pair       →  R²(z), slope(z)
  - Writes results/v9_gemma2/behavioral_summary.json
  - Plots 8-panel heatmap of cell-mean logit_diff in (x, z) grid
  - Plots 8-panel heatmap of cell-mean logit_diff in (x, μ) grid
  - Prints pass/fail vs acceptance criterion: R > 0.3 for at least 5/8 pairs

No GPU needed — operates on the .npz + .jsonl outputs from the GPU extraction.
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
FIG_DIR.mkdir(parents=True, exist_ok=True)


def regress(y, *xs):
    """OLS with intercept. Returns dict with r2, intercept, and per-x slopes."""
    y = np.asarray(y, dtype=np.float64)
    A = np.column_stack([np.ones_like(y)] + [np.asarray(x, dtype=np.float64) for x in xs])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"intercept": float(coef[0]),
            "slopes": [float(c) for c in coef[1:]],
            "r2": r2}


def load_pair(pair_name: str) -> dict:
    """Load trials + per-prompt logit_diff for one pair."""
    trials = {}
    with (RES_DIR / "gemma2_trials.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["pair"] == pair_name:
                trials[t["id"]] = t

    log_rows = [json.loads(l) for l in (RES_DIR / f"gemma2_{pair_name}_logits.jsonl").open()]
    rows = []
    for r in log_rows:
        if r["id"] not in trials:
            continue
        t = trials[r["id"]]
        rows.append({
            "id": r["id"],
            "x": t["x"], "mu": t["mu"], "z": t["z"],
            "logit_diff": r["logit_diff"], "entropy": r["entropy"],
        })
    return {"pair": pair_name, "rows": rows}


def analyze_pair(pair_name: str) -> dict:
    data = load_pair(pair_name)
    rows = data["rows"]
    if not rows:
        return {"pair": pair_name, "error": "no data"}

    xs = np.array([r["x"] for r in rows])
    mus = np.array([r["mu"] for r in rows])
    zs = np.array([r["z"] for r in rows])
    lds = np.array([r["logit_diff"] for r in rows])

    # For log-space pairs, regress on log-values (matches how z is defined)
    if pair_name in LOG_SPACE_PAIRS:
        xs_use = np.log(xs)
        mus_use = np.log(mus)
    else:
        xs_use = xs
        mus_use = mus

    fit_xmu = regress(lds, xs_use, mus_use)
    slope_x, slope_mu = fit_xmu["slopes"]
    R = (-slope_mu / slope_x) if abs(slope_x) > 1e-9 else None

    fit_z = regress(lds, zs)
    fit_x = regress(lds, xs_use)

    return {
        "pair": pair_name,
        "n_trials": len(rows),
        "logit_diff_mean": float(lds.mean()),
        "logit_diff_std": float(lds.std()),
        "entropy_mean": float(np.mean([r["entropy"] for r in rows])),
        "slope_x": slope_x, "slope_mu": slope_mu,
        "relativity_ratio": R,
        "r2_xmu": fit_xmu["r2"],
        "r2_z": fit_z["r2"],
        "r2_x_alone": fit_x["r2"],
        "slope_z": fit_z["slopes"][0],
    }


def plot_heatmaps(all_results: list[dict]):
    """8-panel (x, z) cell-mean heatmap of logit_diff."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, res in zip(axes.flat, all_results):
        pair = res["pair"]
        data = load_pair(pair)
        rows = data["rows"]
        if not rows:
            ax.set_title(f"{pair} (no data)")
            ax.axis("off")
            continue
        # Discretize to the Grid B cells (unique x and z values)
        xs = np.array([r["x"] for r in rows])
        zs = np.array([r["z"] for r in rows])
        lds = np.array([r["logit_diff"] for r in rows])
        x_vals = sorted(set(xs))
        z_vals = sorted(set(zs))
        grid = np.full((len(z_vals), len(x_vals)), np.nan)
        for r in rows:
            i = z_vals.index(r["z"])
            j = x_vals.index(r["x"])
            if np.isnan(grid[i, j]):
                grid[i, j] = 0.0
            grid[i, j] += r["logit_diff"]
        counts = np.zeros_like(grid)
        for r in rows:
            i = z_vals.index(r["z"])
            j = x_vals.index(r["x"])
            counts[i, j] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = grid / counts
        vmax = np.nanmax(np.abs(grid))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[0, len(x_vals), 0, len(z_vals)])
        ax.set_xticks(np.arange(len(x_vals)) + 0.5)
        ax.set_xticklabels([f"{x:g}" for x in x_vals], rotation=45, fontsize=7)
        ax.set_yticks(np.arange(len(z_vals)) + 0.5)
        ax.set_yticklabels([f"{z:+.1f}" for z in z_vals], fontsize=7)
        ax.set_xlabel("x (raw value)", fontsize=8)
        ax.set_ylabel("z (context-relative)", fontsize=8)
        R = res["relativity_ratio"]
        R_str = f"R={R:+.2f}" if R is not None else "R=N/A"
        ax.set_title(f"{pair}: {R_str}  r²(z)={res['r2_z']:.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        "v9 P1 — Gemma 2 2B Grid B: cell-mean logit_diff (high − low token)",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "gemma2_logit_diff_xz_8panel.png"
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")


def main():
    all_results = []
    for p in PAIRS:
        print(f"\n=== {p.name} ===")
        res = analyze_pair(p.name)
        all_results.append(res)
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue
        print(f"  n={res['n_trials']}  ld_mean={res['logit_diff_mean']:+.3f}  "
              f"ent_mean={res['entropy_mean']:.2f}")
        R = res["relativity_ratio"]
        print(f"  slope_x={res['slope_x']:+.4f}  slope_μ={res['slope_mu']:+.4f}  "
              f"R=-slope_μ/slope_x={R:+.3f}" if R is not None else
              f"  slope_x≈0 — ratio undefined")
        print(f"  r²(z)={res['r2_z']:.3f}  r²(x,μ)={res['r2_xmu']:.3f}  "
              f"r²(x)={res['r2_x_alone']:.3f}")

    # Acceptance criterion
    R_vals = {r["pair"]: r.get("relativity_ratio") for r in all_results}
    n_passing = sum(
        1 for R in R_vals.values() if R is not None and R > 0.3
    )
    print(f"\n{'='*60}")
    print(f"ACCEPTANCE: R > 0.3 on {n_passing}/8 pairs "
          f"({'PASS' if n_passing >= 5 else 'FAIL'})")
    print(f"{'='*60}")
    for pair_name, R in R_vals.items():
        status = " (>=0.3)" if (R is not None and R > 0.3) else ""
        R_fmt = f"{R:+.3f}" if R is not None else "   N/A"
        print(f"  {pair_name:12s}  R = {R_fmt}{status}")

    summary = {
        "model": "google/gemma-2-2b",
        "layer": "late (index 20)",
        "grid": "Grid B (5x × 5z × 30 seeds)",
        "pairs": all_results,
        "acceptance_n_passing_R_gt_0.3": n_passing,
        "acceptance_pass": n_passing >= 5,
    }
    out_json = RES_DIR / "behavioral_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out_json}")

    plot_heatmaps(all_results)


if __name__ == "__main__":
    main()
