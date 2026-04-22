"""v9 Priority 4: Park causal inner product steering.

Idea (Park et al., ICML 2024): the "causal" inner product on the residual
stream is induced by the unembedding matrix W_U:

    <u, v>_causal = u^T (W_U W_U^T) v

A covector that "decodes" z (the Ridge probe_z) need not be the same as the
*direction* along which z is causally encoded (primal_z). If the causal
metric bridges them, then transforming probe_z by (W_U W_U^T)^-1 should
make it steer as effectively as primal_z.

    probe_z_causal = (W_U W_U^T + λI)^{-1} probe_z

This script:
  1. Extracts W_U (= lm_head weight, shape (vocab, d_model)).
  2. Computes M = W_U^T W_U (NOT W_U W_U^T — we want (d_model, d_model)).
     NOTE: W_U is stored as lm_head.weight which has shape (vocab, d).
     The causal-metric matrix is W_U.T @ W_U in that convention, i.e.
     the Gram of columns-of-W_U-viewed-as-rows-of-lm_head — same object.
  3. For each pair computes probe_z (Ridge) and probe_z_causal =
     solve(M + λI, probe_z), rescaled to ||primal_z||.
  4. Runs steering with primal_z, probe_z, and probe_z_causal at the same
     α grid as P3 so the comparison is apples-to-apples.

Outputs
    results/v9_gemma2/park_causal_rows.jsonl
    results/v9_gemma2/park_causal_summary.json
    figures/v9/park_causal_slopes.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS  # noqa: E402
from extract_v7_xz_grid import Z_VALUES  # noqa: E402
from exp_v9_on_manifold_steering import (  # noqa: E402
    load_trials_with_z, get_layers, run_steering, ALPHAS,
    LATE_LAYER, BATCH_SIZE,
)

MODEL_ID = "google/gemma-2-2b"
RIDGE_ALPHA = 1.0
CAUSAL_LAMBDA = 1e-2   # regularizer on the causal inner-product inversion
RNG = np.random.default_rng(0)

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"


def extract_WU(model) -> np.ndarray:
    """Return W_U = lm_head.weight.detach() as float32 on CPU, shape (vocab, d)."""
    W = model.get_output_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    return W


def compute_causal_metric(W_U: np.ndarray, lam: float) -> np.ndarray:
    """Return M = W_U^T W_U + λ I (d, d)."""
    M = W_U.T @ W_U   # (d, d)
    M += lam * np.eye(W_U.shape[1], dtype=W_U.dtype)
    return M


def main():
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    print("\nExtracting W_U…", flush=True)
    W_U = extract_WU(model)
    print(f"  W_U shape: {W_U.shape}  |W_U|={np.linalg.norm(W_U):.1f}")
    print(f"\nComputing causal metric M = W_U^T W_U + {CAUSAL_LAMBDA}·I (λ for stability)…")
    M = compute_causal_metric(W_U, CAUSAL_LAMBDA)
    # Cache Cholesky for fast solves
    from scipy.linalg import cho_factor, cho_solve  # noqa: E402 (lazy import)
    c_and_low = cho_factor(M, lower=True)
    print(f"  cond(M) estimate via eigmin/max on a random subspace: skipping for speed.")

    SUBSET_PER_Z = 20
    out_rows = []
    summary = {"model": MODEL_ID, "layer": LATE_LAYER, "alphas": ALPHAS,
               "ridge_alpha": RIDGE_ALPHA, "causal_lambda": CAUSAL_LAMBDA,
               "per_pair": {}}

    for pair in PAIRS:
        print(f"\n=== {pair.name} ===", flush=True)
        rows_all, acts = load_trials_with_z(pair.name)
        zs = np.array([r["z"] for r in rows_all])

        # Directions
        hi = zs > 0
        lo = zs < 0
        primal_z = (acts[hi].mean(0) - acts[lo].mean(0)).astype(np.float32)
        rid = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(acts, zs)
        probe_z = rid.coef_.astype(np.float32)
        probe_z_causal = cho_solve(c_and_low, probe_z).astype(np.float32)

        norm_p = np.linalg.norm(primal_z)

        def rescale_to_norm_p(v):
            n = np.linalg.norm(v)
            return v * (norm_p / n) if n > 1e-9 else v

        probe_z_n = rescale_to_norm_p(probe_z)
        probe_z_causal_n = rescale_to_norm_p(probe_z_causal)

        cos_probe_primal = float(
            probe_z @ primal_z / (np.linalg.norm(probe_z) * norm_p + 1e-12))
        cos_causal_primal = float(
            probe_z_causal @ primal_z / (np.linalg.norm(probe_z_causal) * norm_p + 1e-12))
        print(f"  ||primal||={norm_p:.2f}  "
              f"cos(probe,primal)={cos_probe_primal:+.3f}  "
              f"cos(probe_causal,primal)={cos_causal_primal:+.3f}")

        # Stratified subset (same recipe as P3 so results are comparable)
        chosen = []
        for z_val in Z_VALUES:
            idx = np.where(np.isclose(zs, z_val, atol=1e-6))[0]
            if len(idx) == 0:
                continue
            pick = RNG.choice(idx, size=min(SUBSET_PER_Z, len(idx)), replace=False)
            chosen.extend(pick.tolist())
        chosen = np.array(sorted(chosen), dtype=int)
        sub_rows = [rows_all[i] for i in chosen]
        sub_zs = zs[chosen]

        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)

        for dir_name, dir_vec in [
            ("primal", primal_z),
            ("probe", probe_z_n),
            ("probe_causal", probe_z_causal_n),
        ]:
            dir_per_row = np.tile(dir_vec, (len(sub_rows), 1)).astype(np.float32)
            by_alpha = {}
            ent_by_alpha = {}
            for alpha in ALPHAS:
                t1 = time.time()
                ld, ent = run_steering(model, tok, sub_rows, dir_per_row, alpha,
                                       high_id, low_id)
                print(f"  {dir_name:12s} α={alpha:+.1f}  "
                      f"ld_mean={ld.mean():+.3f}  ent_mean={ent.mean():.3f}  "
                      f"({time.time() - t1:.1f}s)", flush=True)
                by_alpha[alpha] = ld.mean()
                ent_by_alpha[alpha] = ent.mean()
                for k, r in enumerate(sub_rows):
                    out_rows.append({
                        "id": r["id"], "pair": pair.name, "z": float(sub_zs[k]),
                        "direction": dir_name, "alpha": float(alpha),
                        "logit_diff": float(ld[k]), "entropy": float(ent[k]),
                    })
            xs = np.array(ALPHAS)
            ys = np.array([by_alpha[a] for a in ALPHAS])
            slope = float(np.polyfit(xs, ys, 1)[0])
            ent_0 = ent_by_alpha[0.0]
            ent_abs2 = 0.5 * (ent_by_alpha[-2.0] + ent_by_alpha[2.0])
            summary["per_pair"].setdefault(pair.name, {})[dir_name] = {
                "slope": slope,
                "entropy_at_0": float(ent_0),
                "entropy_at_abs2": float(ent_abs2),
                "entropy_shift_2": float(ent_abs2 - ent_0),
                "cos_with_primal":
                    cos_probe_primal if dir_name == "probe"
                    else (cos_causal_primal if dir_name == "probe_causal" else 1.0),
            }
        p = summary["per_pair"][pair.name]
        print(f"  slopes:    primal={p['primal']['slope']:+.3f}   "
              f"probe={p['probe']['slope']:+.3f}   "
              f"probe_causal={p['probe_causal']['slope']:+.3f}")

    with (RES_DIR / "park_causal_rows.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    (RES_DIR / "park_causal_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote rows + summary to {RES_DIR}/park_causal_*.")


if __name__ == "__main__":
    main()
