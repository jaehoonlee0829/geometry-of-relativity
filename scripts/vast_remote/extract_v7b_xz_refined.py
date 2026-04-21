"""v7b: fixed Grid B for experience and size pairs.

v7 Grid B had residual corr(x, z) for size (0.13) and experience (0.20)
because asymmetric cell-dropping left some (x) values with fewer z-bins.

Fix: per-pair, restrict to x values that allow ALL 5 z-values to produce
plausible μ. Drop the problematic x values entirely rather than individual
cells. Preserves symmetry.

  experience: drop x=1 (x=1 forces z≤0.125 for μ≥0.5). Keep x ∈ {5,10,15,25}.
              4 × 5 × 30 = 600 prompts.
  size:       drop x=5 (x=5 forces z≤0.67 for μ≥1). Keep x ∈ {15,25,40,60}.
              4 × 5 × 30 = 600 prompts.

Other pairs unchanged from v7.

Writes (only for the 2 fixed pairs):
  results/v7b_xz_grid/e4b_{pair}_{layer}.npz
  results/v7b_xz_grid/e4b_{pair}_logits.jsonl
  results/v7b_xz_grid/e4b_{pair}_trials.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS, LOG_SPACE_PAIRS, fmt_num, sample_context  # noqa: E402
from extract_v7_xz_grid import (  # noqa: E402
    LAYER_INDICES, BATCH_SIZE, Z_VALUES, N_SEEDS,
    make_implicit_prompt, get_layers, extract_and_score,
)

MODEL_ID = "google/gemma-4-E4B"
OUT = REPO / "results" / "v7b_xz_grid"
OUT.mkdir(parents=True, exist_ok=True)

# Custom x-subsets for the 2 problematic pairs
FIXED_X = {
    "experience": [5.0, 10.0, 15.0, 25.0],     # drops x=1
    "size":       [15.0, 25.0, 40.0, 60.0],    # drops x=5
}


def build_trials_for_pair(pair):
    xs = FIXED_X[pair.name]
    trials = []
    idx = 0
    for x in xs:
        for z in Z_VALUES:
            mu = x - pair.sigma * z
            for s in range(N_SEEDS):
                trials.append({
                    "id": f"{pair.name}_v7b_{idx:06d}",
                    "pair": pair.name,
                    "condition": "implicit_xz_refined",
                    "prompt": make_implicit_prompt(pair, x, mu, s),
                    "x": float(x), "mu": float(mu), "z": float(z),
                    "sigma": pair.sigma, "seed": s,
                    "low_word": pair.low_word, "high_word": pair.high_word,
                })
                idx += 1
    return trials


def main():
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    for pair in PAIRS:
        if pair.name not in FIXED_X:
            continue
        trials = build_trials_for_pair(pair)
        xs = np.array([t["x"] for t in trials])
        zs = np.array([t["z"] for t in trials])
        corr_xz = float(np.corrcoef(xs, zs)[0, 1])
        print(f"\n=== {pair.name}: {len(trials)} trials  corr(x,z)={corr_xz:+.4f} ===", flush=True)

        with (OUT / f"e4b_{pair.name}_trials.jsonl").open("w") as f:
            for t in trials: f.write(json.dumps(t) + "\n")

        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)
        t1 = time.time()
        acts, ld, ent = extract_and_score(model, tok, trials, high_id, low_id)
        print(f"  extracted in {time.time()-t1:.1f}s  ld_mean={ld.mean():+.3f}", flush=True)

        ids_arr = np.array([t["id"] for t in trials])
        for layer, a in acts.items():
            np.savez(OUT / f"e4b_{pair.name}_{layer}.npz",
                     activations=a.astype(np.float32),
                     ids=ids_arr,
                     layer_index=LAYER_INDICES[layer], layer_name=layer)
        with (OUT / f"e4b_{pair.name}_logits.jsonl").open("w") as f:
            for t, l, e in zip(trials, ld, ent):
                f.write(json.dumps({"id": t["id"], "logit_diff": float(l), "entropy": float(e)}) + "\n")

    # Audit: re-compute cos(primal_z, primal_x) on fixed pairs
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA

    def unit(v): return v / (np.linalg.norm(v) + 1e-12)

    print("\n=== RE-AUDIT (v7 → v7b for fixed pairs) ===")
    print(f"{'pair':12s}  corr_v7  corr_v7b   cos(pz,px)_v7  cos(pz,px)_v7b")
    v7_audit = json.load(open(REPO / "results" / "v7_analysis" / "direction_confound_audit_clean.json"))
    for pair in PAIRS:
        if pair.name not in FIXED_X:
            continue
        trials = build_trials_for_pair(pair)
        trials_by_id = {t["id"]: t for t in trials}
        npz = np.load(OUT / f"e4b_{pair.name}_late.npz", allow_pickle=True)
        acts = npz["activations"].astype(np.float64)
        ids = [str(s) for s in npz["ids"]]
        xs = np.array([trials_by_id[i]["x"] for i in ids])
        zs = np.array([trials_by_id[i]["z"] for i in ids])
        corr_v7b = float(np.corrcoef(xs, zs)[0, 1])

        primal_z = unit(acts[zs > +1.0].mean(0) - acts[zs < -1.0].mean(0))
        x_hi, x_lo = np.percentile(xs, 75), np.percentile(xs, 25)
        primal_x = unit(acts[xs >= x_hi].mean(0) - acts[xs <= x_lo].mean(0))
        cos_pz_px = float(np.dot(primal_z, primal_x))

        corr_v7 = v7_audit["cross_grid"][pair.name]["corr_xz_gridB"]
        cos_v7 = v7_audit["per_pair_gridB"][pair.name][0][1]
        print(f"  {pair.name:10s}   {corr_v7:+.3f}   {corr_v7b:+.3f}      {cos_v7:+.3f}         {cos_pz_px:+.3f}")


if __name__ == "__main__":
    main()
