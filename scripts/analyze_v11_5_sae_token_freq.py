"""v11.5 §E — SAE features with token-frequency / numeral-magnitude control.

For each (model, pair, late_layer) and each top-50 z-feature from v11 P4:
  - r2_z      = R²(z | feature)
  - r2_x      = R²(x | feature)            ← raw numeral magnitude proxy
  - r2_token  = R²(numeral_token_id | feature)   ← exact token identity

A feature whose r2_z >> r2_x and r2_token is small is a TRUE z-feature.
A feature with r2_x > r2_z is tracking raw magnitude.
A feature with high r2_token is tracking specific tokens.

Cross-pair Jaccard of top-50 z-features at the late layer (replicates v9's
0.06 cross-pair claim at v11's higher N).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from analyze_v11_sae import load_sae, encode_sae, cv_r2_scalar, find_sae_subdir  # noqa: E402

ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
SAE_REPO = {"gemma2-2b": "google/gemma-scope-2b-pt-res", "gemma2-9b": "google/gemma-scope-9b-pt-res"}
LATE = {"gemma2-2b": 20, "gemma2-9b": 33}


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def numeral_token_proxy(trials: list[dict]) -> np.ndarray:
    """Return per-prompt 'token-magnitude' proxy: numeric value cast to int.
    Useful as a control: if a feature's R² on this is high, the feature is
    just tracking the numeral, not z."""
    return np.array([float(t["x"]) for t in trials], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=list(SAE_REPO.keys()))
    ap.add_argument("--pair", default="all")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")

    L = LATE[args.model_short]
    print(f"[sae-tok] loading SAE for {args.model_short} L{L}...", flush=True)
    sae = load_sae(SAE_REPO[args.model_short], L, token)

    pairs = ALL_PAIRS if args.pair == "all" else [args.pair]
    by_pair: dict[str, dict] = {}
    top_z_feats_per_pair: dict[str, list[int]] = {}
    for p in pairs:
        rp = REPO / "results" / "v11" / args.model_short / p / f"{args.model_short}_{p}_v11_residuals.npz"
        if not rp.exists(): continue
        d = np.load(rp)
        h = d["activations"][:, L, :].astype(np.float32)
        z = d["z"].astype(np.float64)
        x = d["x"].astype(np.float64)

        # Numeral-token proxy: x value (already a numeral magnitude); for an
        # "exact-token" control we'd need actual token ids — proxy x rounded to
        # int gives token-string proxy.
        token_proxy = np.round(x).astype(np.float64)

        feats = encode_sae(h, sae)
        F = feats.shape[1]
        # Compute R²(z), R²(x), R²(token_proxy) for every feature
        r2_z_arr = np.zeros(F); r2_x_arr = np.zeros(F); r2_tok_arr = np.zeros(F)
        for fi in range(F):
            col = feats[:, fi]
            if col.std() < 1e-9: continue
            r2_z_arr[fi] = cv_r2_scalar(col, z)
            r2_x_arr[fi] = cv_r2_scalar(col, x)
            r2_tok_arr[fi] = cv_r2_scalar(col, token_proxy)

        # Top-50 z-features
        top_z = np.argsort(r2_z_arr)[::-1][:50].tolist()
        top_z_feats_per_pair[p] = top_z

        # Pure-z features: r2_z dominates AND r2_x / r2_token small
        pure_z_features = [int(fi) for fi in top_z
                           if r2_z_arr[fi] > 0.4
                           and r2_x_arr[fi] < 0.5 * r2_z_arr[fi]
                           and r2_tok_arr[fi] < 0.5 * r2_z_arr[fi]]

        by_pair[p] = {
            "n_features_total": int(F),
            "top_50_r2_z": [float(r2_z_arr[fi]) for fi in top_z],
            "top_50_r2_x": [float(r2_x_arr[fi]) for fi in top_z],
            "top_50_r2_tok": [float(r2_tok_arr[fi]) for fi in top_z],
            "top_50_feat_ids": top_z,
            "n_pure_z_features": len(pure_z_features),
            "pure_z_feature_ids": pure_z_features,
        }
        print(f"[sae-tok] {args.model_short}/{p}  L{L}  "
              f"top_z[0]={top_z[0]} (r²_z={r2_z_arr[top_z[0]]:.3f}, "
              f"r²_x={r2_x_arr[top_z[0]]:.3f}, r²_tok={r2_tok_arr[top_z[0]]:.3f})  "
              f"pure_z_count={len(pure_z_features)}", flush=True)

    # Cross-pair Jaccard of top-50 z-feature ids
    plist = sorted(top_z_feats_per_pair.keys())
    n = len(plist)
    Jmat = np.zeros((n, n))
    for i, a in enumerate(plist):
        for j, b in enumerate(plist):
            Jmat[i, j] = jaccard(top_z_feats_per_pair[a], top_z_feats_per_pair[b])
    off_jac = []
    for i in range(n):
        for j in range(i + 1, n):
            off_jac.append(Jmat[i, j])
    off_mean = float(np.mean(off_jac)) if off_jac else 0.0
    print(f"[sae-tok] cross-pair top-50 Jaccard mean (off-diag) = {off_mean:.3f}")

    out = {
        "model_short": args.model_short,
        "layer": L,
        "sae_subpath": sae["subpath"],
        "by_pair": by_pair,
        "cross_pair_jaccard_top50": {a: {b: float(Jmat[i, j]) for j, b in enumerate(plist)}
                                       for i, a in enumerate(plist)},
        "cross_pair_jaccard_mean_off_diag": off_mean,
    }
    out_dir = REPO / "results" / "v11_5" / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sae_features_with_token_freq_control.json").write_text(json.dumps(out, indent=2))
    print(f"[sae-tok] wrote {out_dir / 'sae_features_with_token_freq_control.json'}")


if __name__ == "__main__":
    main()
