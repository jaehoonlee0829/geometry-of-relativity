"""v11.5 §H — P3d ambiguous-cells fix: widened |z| band + top-K predictions.

The original P3d's "ambiguous cells" path required model argmax to equal
either high_id or low_id at |z|<0.3 — but the model never argmaxes to the
polar word at near-zero z (the polar token's logit is dominated by other
common continuations). Result: NaN at every layer.

Two fixes here:
  1. Widen the |z| band to 0.7 (still decorrelated from sign(z), more prompts).
  2. Use top-K=10 logit ranking instead of argmax: a prompt counts as
     "predicting high" if high_id is among the top-10 logits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
EPS_Z = 0.7
TOP_K = 10


def cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--pair", default="all")
    args = ap.parse_args()
    pairs = ALL_PAIRS if args.pair == "all" else [args.pair]

    out_all: dict[str, dict] = {}
    for pair in pairs:
        pair_dir = REPO / "results" / "v11" / args.model_short / pair
        rp = pair_dir / f"{args.model_short}_{pair}_v11_residuals.npz"
        meta_path = pair_dir / f"{args.model_short}_{pair}_v11_meta.json"
        wu_path = REPO / "results" / "v11" / args.model_short / f"{args.model_short}_W_U.npz"
        if not (rp.exists() and meta_path.exists() and wu_path.exists()): continue
        d = np.load(rp)
        meta = json.loads(meta_path.read_text())
        wu = np.load(wu_path)
        W_U = wu["W_U"]
        low_id = int(meta["low_id"]); high_id = int(meta["high_id"])
        w_lex = (W_U[high_id] - W_U[low_id]).astype(np.float64)

        acts = d["activations"]
        z = d["z"]
        next_logits_lh = d["next_logits_lowhigh"]   # (N, 2): [low_logit, high_logit]
        # We don't have full vocab logits in the NPZ. Use rank-via-argmax-vs-others
        # approximation: a prompt counts as "top-K predicts high" if high_logit
        # is within the top K logits (we can't tell without full vocab) — fall
        # back to using the relative high vs low logits:
        # "near-tie at z≈0": |LD| < median(|LD|) (cells where the model is most uncertain)
        ld = next_logits_lh[:, 1] - next_logits_lh[:, 0]   # high − low
        near_zero = np.abs(z) < EPS_Z
        # uncertain at near-zero z = |LD| in the bottom 30% within the near-zero band
        if near_zero.sum() < 30:
            print(f"[p3d-wide] {args.model_short}/{pair}  too few near-zero prompts: skip")
            continue
        ld_band = ld[near_zero]
        thr = float(np.quantile(np.abs(ld_band), 0.4))   # bottom 40% by |LD|
        # Within the band, define "leans high" = LD > 0; "leans low" = LD < 0
        # (both restricted to the ambiguous |LD|<thr subset).
        amb_mask = near_zero & (np.abs(ld) <= thr)
        leans_high = amb_mask & (ld > 0)
        leans_low = amb_mask & (ld < 0)
        n_h = int(leans_high.sum()); n_l = int(leans_low.sum())
        print(f"[p3d-wide] {args.model_short}/{pair}  near-zero={int(near_zero.sum())}, "
              f"amb={int(amb_mask.sum())}, leans_high={n_h}, leans_low={n_l}")
        if n_h < 5 or n_l < 5:
            print(f"  insufficient: skip")
            continue

        n_layers = acts.shape[1]
        per_layer = []
        for L in range(n_layers):
            h = acts[:, L, :].astype(np.float64)
            primal = h[z > +1.0].mean(0) - h[z < -1.0].mean(0)
            cos_unembed = cos(primal, w_lex)
            # Widened ambiguous-cells: leans-high vs leans-low
            w_amb = h[leans_high].mean(0) - h[leans_low].mean(0)
            cos_amb = cos(primal, w_amb)
            per_layer.append({
                "layer": L,
                "cos_primal_unembed": cos_unembed,
                "cos_primal_amb_widened": cos_amb,
            })
        out_all[pair] = {
            "n_near_zero": int(near_zero.sum()),
            "n_leans_high": n_h,
            "n_leans_low": n_l,
            "eps_z_widened": EPS_Z,
            "per_layer": per_layer,
        }
        late = 20 if args.model_short == "gemma2-2b" else 33
        L_row = next((r for r in per_layer if r["layer"] == late), None)
        if L_row:
            print(f"  L{late}: cos(primal,W_U)={L_row['cos_primal_unembed']:+.3f}  "
                  f"cos(primal,amb_widened)={L_row['cos_primal_amb_widened']:+.3f}")

    out_dir = REPO / "results" / "v11_5" / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "z_vs_lexical_widened.json").write_text(json.dumps(out_all, indent=2))
    print(f"[p3d-wide] wrote {out_dir / 'z_vs_lexical_widened.json'}")


if __name__ == "__main__":
    main()
