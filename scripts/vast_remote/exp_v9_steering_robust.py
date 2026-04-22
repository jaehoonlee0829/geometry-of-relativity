"""v9 Robustness: extended α, multi-seed null, held-out CV for steering.

Addresses three critic concerns on P3/P4:

  (A) α = ±2 is underpowered for the "on-manifold kinder to entropy"
      hypothesis — extend to α ∈ {±4, ±6, ±8} to enter the off-manifold
      regime.
  (B) The "random null" was a single Gaussian draw — run 30 seeds and
      report per-pair 95% quantile; redo primal-vs-random significance.
  (C) primal_z / probe_z were fit on the same activations they then
      steered into — run 5-fold CV where each fold's directions are
      fit on 80% of trials and steering is evaluated on held-out 20%.

Outputs
  results/v9_gemma2/steering_robust_rows.jsonl
  results/v9_gemma2/steering_robust_summary.json
  figures/v9/steering_extended_alpha.png
  figures/v9/steering_multiseed_null.png
  figures/v9/steering_heldout_cv.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS  # noqa: E402
from extract_v7_xz_grid import Z_VALUES  # noqa: E402
from exp_v9_on_manifold_steering import (  # noqa: E402
    load_trials_with_z, LATE_LAYER, BATCH_SIZE, z_to_bin, run_steering,
)

MODEL_ID = "google/gemma-2-2b"
EXTENDED_ALPHAS = [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
NULL_ALPHAS = [-2.0, 0.0, 2.0]
N_NULL_SEEDS = 30
SUBSET_PER_Z = 20
N_FOLDS = 5

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"


def stratified_subset(rows_all, zs, seed_rng: np.random.Generator):
    """Return chosen indices, stratified by z-value."""
    chosen = []
    for z_val in Z_VALUES:
        idx = np.where(np.isclose(zs, z_val, atol=1e-6))[0]
        if len(idx) == 0:
            continue
        pick = seed_rng.choice(idx, size=min(SUBSET_PER_Z, len(idx)), replace=False)
        chosen.extend(pick.tolist())
    return np.array(sorted(chosen), dtype=int)


def rescale_to(v, target_norm):
    n = np.linalg.norm(v)
    return v * (target_norm / n) if n > 1e-9 else v


def directions_for(acts, zs, pair_norm_p=None):
    """Compute primal_z and Ridge probe_z from the given (acts, zs).

    Returns primal_z (raw), probe_z (Ridge coef), and ||primal_z||.
    Note: probe_z here is NOT rescaled yet — caller does that if needed.
    """
    hi = zs > 0
    lo = zs < 0
    primal_z = (acts[hi].mean(0) - acts[lo].mean(0)).astype(np.float32)
    rid = Ridge(alpha=1.0, fit_intercept=True).fit(acts, zs)
    probe_z = rid.coef_.astype(np.float32)
    return primal_z, probe_z, float(np.linalg.norm(primal_z))


def tangents_for(acts, zs):
    """Per-z-value cell-means → 4 finite-difference tangents (rescaled later)."""
    cell_means = []
    for z_val in Z_VALUES:
        mask = np.isclose(zs, z_val, atol=1e-6)
        if mask.any():
            cell_means.append(acts[mask].mean(axis=0))
        else:
            cell_means.append(np.zeros(acts.shape[1], dtype=acts.dtype))
    cell_means = np.stack(cell_means)
    return np.diff(cell_means, axis=0).astype(np.float32)


def slope_of_alpha_means(ld_by_alpha: dict, alphas):
    xs = np.array(alphas)
    ys = np.array([np.mean(ld_by_alpha[a]) for a in alphas])
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    summary = {
        "model": MODEL_ID,
        "layer": LATE_LAYER,
        "extended_alphas": EXTENDED_ALPHAS,
        "null_alphas": NULL_ALPHAS,
        "n_null_seeds": N_NULL_SEEDS,
        "n_folds": N_FOLDS,
        "per_pair": {},
    }
    all_rows = []
    rng_master = np.random.default_rng(0)

    for pair in PAIRS:
        print(f"\n==================== {pair.name} ====================", flush=True)
        rows_all, acts = load_trials_with_z(pair.name)
        zs = np.array([r["z"] for r in rows_all])

        # Fixed stratified subset (same recipe as P3)
        chosen = stratified_subset(rows_all, zs, rng_master)
        sub_rows = [rows_all[i] for i in chosen]
        sub_zs = zs[chosen]

        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)

        # ============  (A) extended α  ============
        primal, probe, norm_p = directions_for(acts, zs)
        tangents = tangents_for(acts, zs)
        # Rescale each tangent to norm_p
        tangents_scaled = np.stack([
            rescale_to(tangents[k], norm_p) for k in range(tangents.shape[0])
        ])

        primal_per_row = np.tile(primal, (len(sub_rows), 1)).astype(np.float32)
        tangent_per_row = np.array(
            [tangents_scaled[z_to_bin(r["z"])] for r in sub_rows],
            dtype=np.float32,
        )

        ext_ld = {"primal": {}, "tangent": {}}
        ext_ent = {"primal": {}, "tangent": {}}
        for dir_name, dir_arr in [("primal", primal_per_row),
                                  ("tangent", tangent_per_row)]:
            for alpha in EXTENDED_ALPHAS:
                t1 = time.time()
                ld, ent = run_steering(model, tok, sub_rows, dir_arr, alpha,
                                       high_id, low_id)
                print(f"  [A] {dir_name:7s} α={alpha:+.1f}  ld={ld.mean():+.3f}  "
                      f"ent={ent.mean():.3f}  ({time.time() - t1:.1f}s)", flush=True)
                ext_ld[dir_name][alpha] = ld.tolist()
                ext_ent[dir_name][alpha] = ent.tolist()
                for k, r in enumerate(sub_rows):
                    all_rows.append({
                        "id": r["id"], "pair": pair.name, "z": float(sub_zs[k]),
                        "experiment": "extended_alpha",
                        "direction": dir_name, "alpha": float(alpha),
                        "logit_diff": float(ld[k]), "entropy": float(ent[k]),
                    })

        # ============  (B) Multi-seed random null  ============
        null_slopes = []  # one slope per seed, computed over NULL_ALPHAS
        null_ent_shifts = []
        for seed in range(N_NULL_SEEDS):
            rng_s = np.random.default_rng(1000 + seed)
            v = rng_s.standard_normal(acts.shape[1]).astype(np.float32)
            v = rescale_to(v, norm_p)
            dir_per_row = np.tile(v, (len(sub_rows), 1)).astype(np.float32)
            ld_by_alpha = {}
            ent_by_alpha = {}
            for alpha in NULL_ALPHAS:
                ld, ent = run_steering(model, tok, sub_rows, dir_per_row, alpha,
                                       high_id, low_id)
                ld_by_alpha[alpha] = ld
                ent_by_alpha[alpha] = ent
            slope = slope_of_alpha_means(ld_by_alpha, NULL_ALPHAS)
            ent_shift = float(
                0.5 * (ent_by_alpha[-2.0].mean() + ent_by_alpha[2.0].mean())
                - ent_by_alpha[0.0].mean()
            )
            null_slopes.append(slope)
            null_ent_shifts.append(ent_shift)
        null_slopes = np.array(null_slopes)
        null_ent_shifts = np.array(null_ent_shifts)
        print(f"  [B] random null over {N_NULL_SEEDS} seeds: "
              f"slope 2.5%={np.quantile(null_slopes, 0.025):+.3f}  "
              f"50%={np.quantile(null_slopes, 0.5):+.3f}  "
              f"97.5%={np.quantile(null_slopes, 0.975):+.3f}")

        # ============  (C) Held-out CV (primal vs probe)  ============
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1234)
        cv_slopes = {"primal_in": [], "primal_out": [],
                     "probe_in": [],  "probe_out": []}
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(rows_all)))):
            acts_tr, zs_tr = acts[train_idx], zs[train_idx]
            acts_te, zs_te = acts[test_idx],  zs[test_idx]
            rows_te = [rows_all[i] for i in test_idx]
            primal_tr, probe_tr, norm_tr = directions_for(acts_tr, zs_tr)
            probe_tr_scaled = rescale_to(probe_tr, norm_tr)
            # Stratified subset ON TEST fold
            chosen_te = stratified_subset(rows_te,
                                          np.array([r["z"] for r in rows_te]),
                                          rng_master)
            if len(chosen_te) == 0:
                continue
            sub_te = [rows_te[i] for i in chosen_te]
            for dir_name, v in [("primal", primal_tr), ("probe", probe_tr_scaled)]:
                dir_per_row = np.tile(v, (len(sub_te), 1)).astype(np.float32)
                ld_by_a = {}
                for alpha in NULL_ALPHAS:  # reuse -2, 0, +2
                    ld, _ = run_steering(model, tok, sub_te, dir_per_row, alpha,
                                         high_id, low_id)
                    ld_by_a[alpha] = ld
                slope = slope_of_alpha_means(ld_by_a, NULL_ALPHAS)
                cv_slopes[f"{dir_name}_out"].append(slope)

            # In-sample baseline: use directions fit on WHOLE data, evaluate on
            # the same test subset (so the only difference vs. "out" is fit set).
            primal_full, probe_full, norm_full = directions_for(acts, zs)
            probe_full_scaled = rescale_to(probe_full, norm_full)
            for dir_name, v in [("primal", primal_full),
                                ("probe", probe_full_scaled)]:
                dir_per_row = np.tile(v, (len(sub_te), 1)).astype(np.float32)
                ld_by_a = {}
                for alpha in NULL_ALPHAS:
                    ld, _ = run_steering(model, tok, sub_te, dir_per_row, alpha,
                                         high_id, low_id)
                    ld_by_a[alpha] = ld
                slope = slope_of_alpha_means(ld_by_a, NULL_ALPHAS)
                cv_slopes[f"{dir_name}_in"].append(slope)

        def mean_std(xs):
            a = np.array(xs) if xs else np.zeros(1)
            return float(a.mean()), float(a.std()), a.tolist()

        cv_stats = {k: dict(zip(("mean", "std", "values"), mean_std(v)))
                    for k, v in cv_slopes.items()}
        print(f"  [C] held-out slope (mean ± sd over {N_FOLDS} folds):")
        for k in ("primal_in", "primal_out", "probe_in", "probe_out"):
            s = cv_stats[k]
            print(f"       {k:12s}  {s['mean']:+.3f} ± {s['std']:.3f}")

        # Compact per-pair summary
        def mean_slope_over_alphas(d):
            xs = np.array(EXTENDED_ALPHAS)
            ys = np.array([float(np.mean(d[a])) for a in EXTENDED_ALPHAS])
            # slope using all 11 α
            return float(np.polyfit(xs, ys, 1)[0])

        def ent_shift_at(alpha, base_alpha, d_ent):
            return float(np.mean(d_ent[alpha]) - np.mean(d_ent[base_alpha]))

        pair_summary = {
            "norm_primal": norm_p,
            # Extended α
            "slope_primal_ext":  mean_slope_over_alphas(ext_ld["primal"]),
            "slope_tangent_ext": mean_slope_over_alphas(ext_ld["tangent"]),
            "ent_primal_at_alpha2":  ent_shift_at( 2.0, 0.0, ext_ent["primal"]),
            "ent_primal_at_alpha4":  ent_shift_at( 4.0, 0.0, ext_ent["primal"]),
            "ent_primal_at_alpha6":  ent_shift_at( 6.0, 0.0, ext_ent["primal"]),
            "ent_primal_at_alpha8":  ent_shift_at( 8.0, 0.0, ext_ent["primal"]),
            "ent_tangent_at_alpha2": ent_shift_at( 2.0, 0.0, ext_ent["tangent"]),
            "ent_tangent_at_alpha4": ent_shift_at( 4.0, 0.0, ext_ent["tangent"]),
            "ent_tangent_at_alpha6": ent_shift_at( 6.0, 0.0, ext_ent["tangent"]),
            "ent_tangent_at_alpha8": ent_shift_at( 8.0, 0.0, ext_ent["tangent"]),
            # Multi-seed null
            "null_slopes_seeds":    null_slopes.tolist(),
            "null_ent_shifts":      null_ent_shifts.tolist(),
            "null_slope_q025":      float(np.quantile(null_slopes, 0.025)),
            "null_slope_q975":      float(np.quantile(null_slopes, 0.975)),
            "null_slope_abs_q95":   float(np.quantile(np.abs(null_slopes), 0.95)),
            # Held-out CV
            "cv_slopes": cv_stats,
        }
        summary["per_pair"][pair.name] = pair_summary

    (RES_DIR / "steering_robust_rows.jsonl").open("w").write(
        "\n".join(json.dumps(r) for r in all_rows) + "\n"
    )
    (RES_DIR / "steering_robust_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {RES_DIR}/steering_robust_summary.json")


if __name__ == "__main__":
    main()
