"""v9 Robustness check for §10.2 (SAE decomposition of z).

Addresses critic concerns:
  (i)  `sae_project` used W_enc (encoder response) but the "direction in SAE
       basis" question is better answered via W_dec rows (feature
       contribution directions). Rerun with BOTH projections side-by-side.
  (ii) The claim "SAE sparsity refuted" was scoped to ONE SAE
       (layer 20, width 65k, avg_l0 61). Repeat on layer-13 SAE and on the
       lowest-L0 width-65k variant for layer 20 to see whether a different
       dictionary gives a different answer.

Outputs:
  results/v9_gemma2/sae_sensitivity.json
  figures/v9/sae_sensitivity_participation.png
  figures/v9/sae_sensitivity_energy_in_top20.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from extract_v4_adjpairs import PAIRS  # noqa: E402

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"
SAE_REPO = "google/gemma-scope-2b-pt-res"

# Sensitivity sweep: (layer, l0, layer_name_for_activations)
VARIANTS = [
    ("layer_20", "average_l0_61", "late"),    # baseline (matches P2)
    ("layer_20", "average_l0_20", "late"),    # sparser dictionary
    ("layer_13", "average_l0_74", "mid"),     # mid-layer SAE
]
TOP_K = 20


def sae_encode(h: np.ndarray, W_enc, b_enc, b_dec, threshold):
    x = h - b_dec
    pre = x @ W_enc + b_enc
    return np.where(pre > threshold, pre, 0.0).astype(np.float32)


def project_enc(v: np.ndarray, W_enc) -> np.ndarray:
    """Encoder-column response: v @ W_enc[:, i]."""
    return (v @ W_enc).astype(np.float32)


def project_dec(v: np.ndarray, W_dec) -> np.ndarray:
    """Decoder-row projection: <v, W_dec[i, :]>.

    Gemma-Scope W_dec rows are unit-normalized, so this is cosine-proportional
    to "how much of v is explained by feature i's contribution direction."
    """
    return (v @ W_dec.T).astype(np.float32)


def participation_ratio(v: np.ndarray) -> float:
    p = v.astype(np.float64) ** 2
    s = p.sum()
    if s < 1e-18:
        return float(len(v))
    return float(s * s / (p * p).sum())


def energy_in_idx(v: np.ndarray, idx) -> float:
    p = v.astype(np.float64) ** 2
    total = p.sum()
    if total < 1e-18:
        return 0.0
    return float(p[list(idx)].sum() / total)


def corrcoef_safe(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 1e-12 else 0.0


def load_pair(pair_name: str, layer_name: str):
    """Return (acts, zs) for one pair at the requested captured layer."""
    with np.load(RES_DIR / f"gemma2_{pair_name}_{layer_name}.npz", allow_pickle=True) as z_:
        acts = z_["activations"].astype(np.float32)
        ids = z_["ids"].tolist()
    trials = {}
    with (RES_DIR / "gemma2_trials.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["pair"] == pair_name:
                trials[t["id"]] = t
    zs = np.array([trials[i]["z"] for i in ids], dtype=np.float64)
    return acts, zs


def run_variant(layer: str, l0: str, layer_name: str):
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    path = f"{layer}/width_65k/{l0}/params.npz"
    print(f"\n### {path}  (captured activations: '{layer_name}')")
    sae_p = hf_hub_download(SAE_REPO, path)
    P = np.load(sae_p)
    W_enc = P["W_enc"].astype(np.float32)
    W_dec = P["W_dec"].astype(np.float32)
    b_enc = P["b_enc"].astype(np.float32)
    b_dec = P["b_dec"].astype(np.float32)
    thr = P["threshold"].astype(np.float32)
    n_feat = W_enc.shape[1]

    per_pair = {}
    for p in PAIRS:
        acts, zs = load_pair(p.name, layer_name)
        coeffs = sae_encode(acts, W_enc, b_enc, b_dec, thr)
        active_idx = np.where(coeffs.any(axis=0))[0]
        corrs = np.zeros(n_feat, dtype=np.float32)
        for i in active_idx:
            corrs[i] = corrcoef_safe(coeffs[:, i], zs)
        top = np.argsort(-np.abs(corrs))[:TOP_K]

        # directions
        hi = zs > 0
        lo = zs < 0
        primal_z = (acts[hi].mean(0) - acts[lo].mean(0)).astype(np.float32)
        rid = Ridge(alpha=1.0, fit_intercept=True).fit(acts, zs)
        probe_z = rid.coef_.astype(np.float32)

        per_pair[p.name] = {
            "n_active": int(len(active_idx)),
            "top_feat_corrs":     [float(corrs[i]) for i in top],
            "top_features":       [int(i) for i in top],
            # Via encoder (original analysis)
            "enc_primal_PR":      participation_ratio(project_enc(primal_z, W_enc)),
            "enc_probe_PR":       participation_ratio(project_enc(probe_z,  W_enc)),
            "enc_primal_ez":      energy_in_idx(project_enc(primal_z, W_enc), top),
            "enc_probe_ez":       energy_in_idx(project_enc(probe_z,  W_enc), top),
            # Via decoder (scientifically more defensible)
            "dec_primal_PR":      participation_ratio(project_dec(primal_z, W_dec)),
            "dec_probe_PR":       participation_ratio(project_dec(probe_z,  W_dec)),
            "dec_primal_ez":      energy_in_idx(project_dec(primal_z, W_dec), top),
            "dec_probe_ez":       energy_in_idx(project_dec(probe_z,  W_dec), top),
        }
        q = per_pair[p.name]
        print(f"  {p.name:12s}  active={q['n_active']:4d}  "
              f"PR[dec]: primal={q['dec_primal_PR']:5.0f} probe={q['dec_probe_PR']:5.0f} "
              f"(probe/primal={q['dec_probe_PR']/max(q['dec_primal_PR'],1):.2f}x)   "
              f"ez[dec]: primal={q['dec_primal_ez']:.4f} probe={q['dec_probe_ez']:.4f}")

    return {
        "sae_path": path,
        "n_features": n_feat,
        "uniform_baseline_ez": TOP_K / n_feat,
        "per_pair": per_pair,
    }


def main():
    results = {}
    for layer, l0, lname in VARIANTS:
        key = f"{layer}_{l0}"
        results[key] = run_variant(layer, l0, lname)
    out = RES_DIR / "sae_sensitivity.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    # Plot: side-by-side participation ratio for 3 variants, dec projection
    pairs = [p.name for p in PAIRS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    width = 0.8 / (2 * len(VARIANTS))  # 2 bars per variant (primal + probe)
    for vi, (layer, l0, lname) in enumerate(VARIANTS):
        key = f"{layer}_{l0}"
        primal = [results[key]["per_pair"][n]["dec_primal_PR"] for n in pairs]
        probe  = [results[key]["per_pair"][n]["dec_probe_PR"]  for n in pairs]
        off_p = (2 * vi - (len(VARIANTS) * 2 - 1) / 2) * width
        off_b = off_p + width
        lbl = f"{layer} {l0.replace('average_','')}"
        ax1.bar(np.arange(len(pairs)) + off_p, primal, width=width,
                label=f"primal · {lbl}", alpha=0.9)
        ax1.bar(np.arange(len(pairs)) + off_b, probe, width=width,
                label=f"probe · {lbl}",  alpha=0.55)
    ax1.set_xticks(np.arange(len(pairs)))
    ax1.set_xticklabels(pairs, rotation=30, ha="right")
    ax1.set_ylabel("Participation ratio (decoder projection)")
    ax1.set_title("v9 P2 sensitivity — participation ratio across 3 SAE variants",
                  fontsize=10)
    ax1.legend(fontsize=6, ncol=2)

    # Second panel: energy fraction in top-20 z-features
    for vi, (layer, l0, lname) in enumerate(VARIANTS):
        key = f"{layer}_{l0}"
        ez_p = [results[key]["per_pair"][n]["dec_primal_ez"] for n in pairs]
        ez_b = [results[key]["per_pair"][n]["dec_probe_ez"]  for n in pairs]
        off_p = (2 * vi - (len(VARIANTS) * 2 - 1) / 2) * width
        off_b = off_p + width
        lbl = f"{layer} {l0.replace('average_','')}"
        ax2.bar(np.arange(len(pairs)) + off_p, ez_p, width=width,
                label=f"primal · {lbl}", alpha=0.9)
        ax2.bar(np.arange(len(pairs)) + off_b, ez_b, width=width,
                label=f"probe · {lbl}",  alpha=0.55)
    ax2.set_xticks(np.arange(len(pairs)))
    ax2.set_xticklabels(pairs, rotation=30, ha="right")
    ax2.set_ylabel("Energy fraction in top-20 z-features (decoder proj)")
    ax2.set_title("v9 P2 sensitivity — z-feature energy concentration",
                  fontsize=10)
    # Uniform baseline line (per-variant, same 20/65536 so identical)
    baseline = TOP_K / list(results.values())[0]["n_features"]
    ax2.axhline(baseline, color="k", ls=":", lw=0.8,
                label=f"uniform {baseline:.4f}")
    ax2.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "sae_sensitivity_participation_and_energy.png", dpi=140)
    print(f"  wrote {FIG_DIR}/sae_sensitivity_participation_and_energy.png")


if __name__ == "__main__":
    main()
