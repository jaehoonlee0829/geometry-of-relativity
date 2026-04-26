"""v11 P3d — z-direction vs lexical-direction disentanglement (CPU).

Methodology-critic fix vs the original v11 plan: the lexical direction is
defined in **model-output space** as ``W_U[:, high_id] - W_U[:, low_id]``
(unembedding row difference), not as ``mean(h[pred==high]) - mean(h[pred==low])``.

The post-hoc model-output split *forces* high cosine with z whenever the model
is accurate (because pred ≈ sign(z)), which would beg the very question P3d
is meant to answer ("is primal_z just a lexical readout direction?").

We compute three lexical-direction variants and report all three:
  1. ``w_lexical_unembed``  = W_U[high] - W_U[low]    (in d_model space; pull
                              into hidden-state space directly — same shape)
  2. ``w_lexical_zero_z``   = mean(h[|z|<eps, pred==high]) - mean(h[|z|<eps, pred==low])
                              (post-hoc split BUT only on ambiguous cells where
                              z is near zero — there pred and z decorrelate).
  3. ``w_lexical_naive``    = mean(h[pred==high]) - mean(h[pred==low])
                              (the originally-planned variant; reported for
                              completeness so we can contrast with #1.)

Both #2 and #3 live in d_model space (hidden-state). #1 lives in d_model space
too (W_U has shape ``(vocab, d_model)`` for Gemma).

For each pair × model × layer:
  primal_z = mean(h[z>+1]) - mean(h[z<-1])
  cos(primal_z, w_lexical_*)

Outputs:
  results/v11/<model_short>/z_vs_lexical_per_layer.json
  figures/v11/disentanglement/z_vs_lexical_<model_short>.png

Key claim the layout is meant to support: cos(primal_z, w_lexical_unembed) is
**low** even when cos(primal_z, w_lexical_naive) is high → primal_z is NOT
just the lexical readout direction; the model has a separate z-code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
EPS_Z = 0.3   # |z| < EPS_Z  is the "ambiguous cells" set for variant #2


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two flat vectors. Returns 0 if either is degenerate."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def directions_for_pair(model_short: str, pair: str) -> dict | None:
    pair_dir = REPO / "results" / "v11" / model_short / pair
    res_path = pair_dir / f"{model_short}_{pair}_v11_residuals.npz"
    wu_path = REPO / "results" / "v11" / model_short / f"{model_short}_W_U.npz"
    if not res_path.exists():
        print(f"[{model_short}/{pair}] residuals missing — skip")
        return None
    if not wu_path.exists():
        print(f"[{model_short}/{pair}] W_U missing — skip")
        return None

    d = np.load(res_path)
    wu_d = np.load(wu_path)
    acts = d["activations"]  # (N, n_layers, d_model)
    z = d["z"]
    next_argmax = d["next_argmax"]

    W_U = wu_d["W_U"]  # (vocab, d_model)
    low_id = int(wu_d["low_id"])
    high_id = int(wu_d["high_id"])
    w_lexical_unembed = (W_U[high_id] - W_U[low_id]).astype(np.float64)

    n_layers = acts.shape[1]

    # Boolean masks
    high_z = z > +1.0
    low_z = z < -1.0
    if high_z.sum() < 30 or low_z.sum() < 30:
        print(f"[{model_short}/{pair}] insufficient z-extreme prompts "
              f"(>+1: {high_z.sum()}, <-1: {low_z.sum()}) — skip")
        return None

    pred_high = next_argmax == high_id
    pred_low = next_argmax == low_id

    near_zero = np.abs(z) < EPS_Z
    pred_high_amb = pred_high & near_zero
    pred_low_amb = pred_low & near_zero

    per_layer: list[dict] = []
    for L in range(n_layers):
        h = acts[:, L, :].astype(np.float64)
        primal_z = h[high_z].mean(0) - h[low_z].mean(0)

        # Variant 1: from unembedding (model-output space)
        cos_unembed = cos(primal_z, w_lexical_unembed)

        # Variant 2: ambiguous-cells post-hoc split (decorrelated from z)
        if pred_high_amb.sum() >= 5 and pred_low_amb.sum() >= 5:
            w_amb = h[pred_high_amb].mean(0) - h[pred_low_amb].mean(0)
            cos_amb = cos(primal_z, w_amb)
            n_amb = (int(pred_high_amb.sum()), int(pred_low_amb.sum()))
        else:
            cos_amb = float("nan")
            n_amb = (int(pred_high_amb.sum()), int(pred_low_amb.sum()))

        # Variant 3: naive post-hoc split (begged-question version, for contrast)
        if pred_high.sum() >= 5 and pred_low.sum() >= 5:
            w_naive = h[pred_high].mean(0) - h[pred_low].mean(0)
            cos_naive = cos(primal_z, w_naive)
        else:
            cos_naive = float("nan")

        per_layer.append({
            "layer": L,
            "cos_primal_lexical_unembed": cos_unembed,
            "cos_primal_lexical_zero_z": cos_amb,
            "cos_primal_lexical_naive": cos_naive,
            "n_pred_high_amb": n_amb[0],
            "n_pred_low_amb": n_amb[1],
        })

    return {
        "model_short": model_short,
        "pair": pair,
        "n_prompts": int(acts.shape[0]),
        "n_high_z": int(high_z.sum()),
        "n_low_z": int(low_z.sum()),
        "n_pred_high": int(pred_high.sum()),
        "n_pred_low": int(pred_low.sum()),
        "n_near_zero": int(near_zero.sum()),
        "per_layer": per_layer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True,
                    choices=["gemma2-2b", "gemma2-9b"],
                    help="model_short matching results/v11/<model_short>/")
    ap.add_argument("--pairs", default="all",
                    help="comma-separated list, or 'all'")
    args = ap.parse_args()

    pairs = ALL_PAIRS if args.pairs == "all" else args.pairs.split(",")

    out: dict[str, dict] = {}
    for p in pairs:
        info = directions_for_pair(args.model_short, p)
        if info is not None:
            out[p] = info
            # Quick summary at the canonical late layer
            late = 20 if args.model_short == "gemma2-2b" else 33
            row = next((r for r in info["per_layer"] if r["layer"] == late), None)
            if row:
                print(f"[{args.model_short}/{p}] L{late}  "
                      f"cos(primal,W_U) = {row['cos_primal_lexical_unembed']:+.3f}  "
                      f"cos(primal,naive) = {row['cos_primal_lexical_naive']:+.3f}  "
                      f"cos(primal,zero_z) = {row['cos_primal_lexical_zero_z']:+.3f}",
                      flush=True)

    out_dir = REPO / "results" / "v11" / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "z_vs_lexical_per_layer.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
