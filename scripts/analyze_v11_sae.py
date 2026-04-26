"""v11 P4 — SAE feature analysis at L7 vs L20 (2B equiv) / L11 vs L33 (9B equiv).

For each (model, pair, layer):
  1. Project residuals through Gemma Scope SAE encoder (JumpReLU SAE).
  2. Identify features with high R²(z) (linear) vs high entropy + nonzero on
     |z| < 1 (place-cell / bump features).
  3. Compute cross-pair Jaccard of top-50 z-features.

Outputs:
  results/v11/<model_short>/sae_L<early>_L<late>_overlap.json
  figures/v11/sae/L<early>_vs_L<late>_feature_profiles_<pair>_<model_short>.png
  figures/v11/sae/L<early>_placecell_vs_linear_<model_short>.png
  figures/v11/sae/cross_pair_feature_overlap_<model_short>.png

Layer choices:
  2B (26 layers): L7 (early) and L20 (late) — same as v9
  9B (42 layers): L11 (= 7×42/26) and L33 (= 20×42/26) — proportional depth
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parent.parent

SAE_REPO = {"gemma2-2b": "google/gemma-scope-2b-pt-res", "gemma2-9b": "google/gemma-scope-9b-pt-res"}
LAYER_PAIR = {"gemma2-2b": (7, 20), "gemma2-9b": (11, 33)}
SAE_WIDTH = "16k"
TARGET_L0_PRIORITY = ["average_l0_71", "average_l0_75", "average_l0_82", "average_l0_88",
                      "average_l0_141", "average_l0_29", "average_l0_22", "average_l0_60",
                      "average_l0_39", "average_l0_68"]
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]


def find_sae_subdir(repo_id: str, layer: int, token: str) -> str:
    """Locate the canonical SAE subdir for (repo, layer, width). Picks the
    first available L0 setting from TARGET_L0_PRIORITY, falling back to
    whatever exists."""
    from huggingface_hub import HfApi
    files = HfApi(token=token).list_repo_files(repo_id)
    prefix = f"layer_{layer}/width_{SAE_WIDTH}/"
    candidates = sorted({f.split("/")[2] for f in files if f.startswith(prefix) and f.endswith("/params.npz")})
    if not candidates:
        raise RuntimeError(f"no SAE found in {repo_id} for layer {layer} width {SAE_WIDTH}")
    for pref in TARGET_L0_PRIORITY:
        if pref in candidates:
            return f"{prefix}{pref}/params.npz"
    return f"{prefix}{candidates[0]}/params.npz"


def load_sae(repo_id: str, layer: int, token: str) -> dict:
    """Returns dict with W_enc(d, F), b_enc(F,), W_dec(F, d), b_dec(d,), threshold(F,)."""
    sub = find_sae_subdir(repo_id, layer, token)
    path = hf_hub_download(repo_id, sub, token=token)
    p = np.load(path)
    keys = list(p.keys())
    return {
        "W_enc": p["W_enc"].astype(np.float32),
        "b_enc": p["b_enc"].astype(np.float32),
        "W_dec": p["W_dec"].astype(np.float32),
        "b_dec": p["b_dec"].astype(np.float32),
        "threshold": p["threshold"].astype(np.float32) if "threshold" in keys else None,
        "subpath": sub,
    }


def encode_sae(h: np.ndarray, sae: dict, batch: int = 1000) -> np.ndarray:
    """JumpReLU encoding. Returns (N, F) feature activations."""
    h = h.astype(np.float32) - sae["b_dec"]   # subtract decoder bias for proper SAE input
    feats = np.zeros((h.shape[0], sae["W_enc"].shape[1]), dtype=np.float32)
    for b0 in range(0, h.shape[0], batch):
        chunk = h[b0:b0 + batch]
        pre = chunk @ sae["W_enc"] + sae["b_enc"]
        if sae["threshold"] is not None:
            # JumpReLU: f = pre if pre > threshold else 0
            mask = pre > sae["threshold"][None, :]
            feats[b0:b0 + chunk.shape[0]] = np.where(mask, pre, 0.0)
        else:
            feats[b0:b0 + chunk.shape[0]] = np.maximum(pre, 0.0)
    return feats


def cv_r2_scalar(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """R²(y | x) where x is a single feature column. Uses 5-fold ridge."""
    if x.std() < 1e-9: return 0.0
    X = x.reshape(-1, 1)
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=(0.1, 1.0, 10.0)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ss = ((y - pred) ** 2).sum(); ss0 = ((y - y.mean()) ** 2).sum()
    return float(1 - ss / max(ss0, 1e-12))


def feature_profile_score(feat: np.ndarray, z: np.ndarray) -> dict:
    """Categorize feature: linear-z (high R²(z)), bump (peaks near z=0),
    monotonic (sign-matching cumulative)."""
    nonzero = (feat > 1e-6).mean()
    if nonzero < 0.005:
        return {"r2_z": 0.0, "kind": "dead", "nonzero_frac": float(nonzero)}
    r2 = cv_r2_scalar(feat, z)
    # Determine peak location of mean(feat | z-bin)
    bins = np.linspace(z.min(), z.max(), 11)
    midp = 0.5 * (bins[:-1] + bins[1:])
    profile = np.array([feat[(z >= bins[i]) & (z < bins[i + 1])].mean() if (z >= bins[i]).any() else 0.0
                        for i in range(len(bins) - 1)])
    peak_z = midp[int(np.argmax(profile))]
    kind = "linear" if r2 > 0.4 else ("bump" if abs(peak_z) < 0.7 and profile.max() > profile.min() * 2 else "other")
    return {"r2_z": r2, "kind": kind, "peak_z": float(peak_z), "nonzero_frac": float(nonzero)}


def analyze_layer(model_short: str, pair: str, layer: int, sae: dict, top_k: int = 50) -> dict:
    res_path = (REPO / "results" / "v11" / model_short / pair /
                f"{model_short}_{pair}_v11_residuals.npz")
    if not res_path.exists():
        return {"available": False, "pair": pair, "layer": layer}
    d = np.load(res_path)
    h = d["activations"][:, layer, :].astype(np.float32)
    z = d["z"].astype(np.float64)

    feats = encode_sae(h, sae)
    F = feats.shape[1]
    profiles = []
    for fi in range(F):
        col = feats[:, fi]
        if col.std() < 1e-9:
            profiles.append({"r2_z": 0.0, "kind": "dead", "nonzero_frac": 0.0, "feat_id": fi})
            continue
        info = feature_profile_score(col, z)
        info["feat_id"] = fi
        profiles.append(info)

    # Counts and top features
    n_linear = sum(1 for p in profiles if p["kind"] == "linear")
    n_bump = sum(1 for p in profiles if p["kind"] == "bump")
    n_dead = sum(1 for p in profiles if p["kind"] == "dead")
    top = sorted(profiles, key=lambda p: p["r2_z"], reverse=True)[:top_k]
    top_ids = [p["feat_id"] for p in top]

    return {
        "available": True,
        "pair": pair,
        "layer": layer,
        "n_features": F,
        "n_linear": n_linear,
        "n_bump": n_bump,
        "n_dead": n_dead,
        "top_z_features": top_ids,
        "top_z_features_r2": [float(p["r2_z"]) for p in top],
    }


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=list(SAE_REPO.keys()))
    ap.add_argument("--pair", default="all", help="pair or 'all'")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")

    pairs = ALL_PAIRS if args.pair == "all" else [args.pair]
    L_early, L_late = LAYER_PAIR[args.model_short]

    print(f"[P4] loading SAEs for {args.model_short}: L{L_early}, L{L_late}...", flush=True)
    sae_early = load_sae(SAE_REPO[args.model_short], L_early, token)
    sae_late = load_sae(SAE_REPO[args.model_short], L_late, token)
    print(f"[P4]   early: {sae_early['subpath']}  ({sae_early['W_enc'].shape})", flush=True)
    print(f"[P4]   late:  {sae_late['subpath']}   ({sae_late['W_enc'].shape})", flush=True)

    by_pair: dict[str, dict] = {}
    for p in pairs:
        early = analyze_layer(args.model_short, p, L_early, sae_early)
        late = analyze_layer(args.model_short, p, L_late, sae_late)
        if not (early["available"] and late["available"]): continue
        by_pair[p] = {"early": early, "late": late}
        print(f"[P4] {args.model_short}/{p}  L{L_early}: linear={early['n_linear']:4d} "
              f"bump={early['n_bump']:4d}  |  L{L_late}: linear={late['n_linear']:4d} "
              f"bump={late['n_bump']:4d}", flush=True)

    # Cross-pair Jaccard of top-50 z-features at the LATE layer
    late_top = {p: by_pair[p]["late"]["top_z_features"] for p in by_pair}
    pair_list = sorted(late_top.keys())
    n = len(pair_list)
    Jmat = np.zeros((n, n))
    for i, a in enumerate(pair_list):
        for j, b in enumerate(pair_list):
            Jmat[i, j] = jaccard(late_top[a], late_top[b])

    out_dir = REPO / "results" / "v11" / args.model_short
    out_path = out_dir / f"sae_L{L_early}_L{L_late}_overlap.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model_short": args.model_short,
        "L_early": L_early, "L_late": L_late,
        "sae_subpath_early": sae_early["subpath"],
        "sae_subpath_late": sae_late["subpath"],
        "by_pair": by_pair,
        "cross_pair_late_jaccard": {a: {b: float(Jmat[i, j]) for j, b in enumerate(pair_list)}
                                     for i, a in enumerate(pair_list)},
    }, indent=2))
    print(f"\n[P4] wrote {out_path.relative_to(REPO)}")

    # Heatmap of cross-pair feature overlap
    fig_dir = REPO / "figures" / "v11" / "sae"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(Jmat, cmap="viridis", vmin=0, vmax=1)
    plt.xticks(range(n), pair_list, rotation=45, ha="right")
    plt.yticks(range(n), pair_list)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{Jmat[i, j]:.2f}", ha="center", va="center",
                     color="white" if Jmat[i, j] < 0.4 else "black", fontsize=7)
    plt.title(f"{args.model_short}  cross-pair top-50 z-feature Jaccard @ L{L_late}")
    plt.colorbar(label="Jaccard")
    plt.tight_layout()
    plt.savefig(fig_dir / f"cross_pair_feature_overlap_{args.model_short}.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
