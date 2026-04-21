"""v7 addendum: rerun posneg (pos/neg math) on the (x, z) clean grid.

v6 Block B used a (x, μ) grid for posneg_abs with corr(x, z) ≈ 0.7 — same
confound as adjpairs. Alt-interp critic flagged that v6 R=0.42 may be
contaminated. Redo with z-independent grid.

Grid B for posneg:
  x ∈ {-8, -3, 0, 3, 8}; z ∈ {-2, -1, 0, 1, 2}; σ=3 → μ = x - σ·z
  μ ranges: x=-8,z=+2→-14; x=+8,z=-2→+14. All within reasonable number range.

Writes:
  results/v4_abs_controls/e4b_posneg_abs_v7_{trials,logits}.jsonl
  results/v4_abs_controls/e4b_posneg_abs_v7_{mid,late}.npz
  results/v4_adjpairs_analysis/posneg_abs_v7_result.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402

MODEL_ID = "google/gemma-4-E4B"
LAYER_INDICES = {"mid": 21, "late": 32}
BATCH_SIZE = 16
X_VALUES = [-8.0, -3.0, 0.0, 3.0, 8.0]
Z_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
SIGMA = 3.0
N_SEEDS = 30
OUT = REPO / "results" / "v4_abs_controls"
OUT_ANALYSIS = REPO / "results" / "v4_adjpairs_analysis"
OUT_FIG = REPO / "figures" / "v7"
OUT.mkdir(parents=True, exist_ok=True)


def fmt(v): return str(int(v)) if v == int(v) else f"{v:.1f}"


def sample_context(mu, seed, n=15):
    rng = np.random.default_rng(seed + 98765)
    out = []
    while len(out) < n:
        v = rng.normal(mu, SIGMA)
        if -40 <= v <= 40:
            out.append(round(v, 1))
    return out


def build_implicit_prompt(x, mu, seed):
    sample = sample_context(mu, seed)
    items = [f"Number {i+1}: {v:g}" for i, v in enumerate(sample)]
    return "\n".join(items) + f"\nNumber {len(items)+1}: {fmt(x)}. This number is"


def build_trials():
    trials = []
    idx = 0
    for x in X_VALUES:
        for z in Z_VALUES:
            mu = x - SIGMA * z
            for s in range(N_SEEDS):
                trials.append({
                    "id": f"posneg_v7_{idx:06d}",
                    "pair": "posneg_abs", "condition": "implicit_xz_grid",
                    "prompt": build_implicit_prompt(x, mu, s),
                    "x": float(x), "mu": float(mu), "z": float(z),
                    "sigma": SIGMA, "seed": s,
                    "low_word": "negative", "high_word": "positive",
                })
                idx += 1
    return trials


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def extract_and_score(model, tok, trials_sub, high_id, low_id):
    prompts = [t["prompt"] for t in trials_sub]
    layers = get_layers(model)
    captured = {k: [] for k in LAYER_INDICES}
    handles = []
    for k, idx in LAYER_INDICES.items():
        def make_hook(kk):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[kk].append(h.detach())
            return hook
        handles.append(layers[idx].register_forward_hook(make_hook(k)))
    per_layer = {k: [] for k in LAYER_INDICES}
    lds, ents = [], []
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            for k in captured: captured[k].clear()
            with torch.no_grad():
                out = model(**enc)
            last = out.logits[:, -1, :]
            logprobs = torch.log_softmax(last.double(), dim=-1)
            ent = -(logprobs.exp() * logprobs).sum(-1).float().cpu().numpy()
            ld = (last[:, high_id] - last[:, low_id]).float().cpu().numpy()
            lds.append(ld); ents.append(ent)
            for k in LAYER_INDICES:
                h = captured[k][0]
                per_layer[k].append(h[:, -1, :].float().cpu().numpy())
    finally:
        for h in handles: h.remove()
    return ({k: np.concatenate(v, axis=0) for k, v in per_layer.items()},
            np.concatenate(lds), np.concatenate(ents))


def main():
    trials = build_trials()
    print(f"Built {len(trials)} prompts (Grid B for posneg)", flush=True)
    xs = np.array([t["x"] for t in trials])
    zs = np.array([t["z"] for t in trials])
    print(f"  corr(x, z) in trials: {np.corrcoef(xs, zs)[0,1]:.4f} (target: 0)")

    with (OUT / "e4b_posneg_abs_v7_trials.jsonl").open("w") as f:
        for t in trials: f.write(json.dumps(t) + "\n")

    print(f"\nLoading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    high_id = first_token_id(tok, "positive")
    low_id = first_token_id(tok, "negative")

    t1 = time.time()
    acts, ld, ent = extract_and_score(model, tok, trials, high_id, low_id)
    print(f"  extracted in {time.time()-t1:.1f}s  ld_mean={ld.mean():+.3f}", flush=True)

    ids_arr = np.array([t["id"] for t in trials])
    for layer, a in acts.items():
        np.savez(OUT / f"e4b_posneg_abs_v7_{layer}.npz",
                 activations=a.astype(np.float32),
                 ids=ids_arr,
                 layer_index=LAYER_INDICES[layer], layer_name=layer)
    with (OUT / "e4b_posneg_abs_v7_logits.jsonl").open("w") as f:
        for t, l, e in zip(trials, ld, ent):
            f.write(json.dumps({"id": t["id"], "logit_diff": float(l), "entropy": float(e)}) + "\n")

    # Compute relativity ratio (project convention)
    x = np.array([t["x"] for t in trials])
    mu = np.array([t["mu"] for t in trials])
    X = np.column_stack([x, mu, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, ld, rcond=None)
    slope_x, slope_mu, _ = beta
    rr = -slope_mu / slope_x if abs(slope_x) > 1e-9 else None
    print(f"\nposneg_abs on Grid B:  slope_x={slope_x:+.4f}  slope_μ={slope_mu:+.4f}  relativity_ratio={rr:+.4f}")

    # Compare to v6 (Grid A) posneg_abs
    v6 = json.load(open(OUT_ANALYSIS / "posneg_abs_result.json"))
    v6_rr = v6.get("relativity_ratio")
    print(f"  v6 (Grid A): relativity_ratio = {v6_rr:+.4f}")
    print(f"  v7 (Grid B) −  v6:    Δ = {rr - v6_rr:+.4f}")

    result = {
        "grid": "B (x, z independent)",
        "pair": "posneg_abs",
        "slope_x": float(slope_x),
        "slope_mu": float(slope_mu),
        "relativity_ratio": float(rr),
        "v6_grid_A_relativity_ratio": v6_rr,
        "n_trials": len(trials),
        "corr_xz": float(np.corrcoef(xs, zs)[0, 1]),
    }
    (OUT_ANALYSIS / "posneg_abs_v7_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_ANALYSIS/'posneg_abs_v7_result.json'}")


if __name__ == "__main__":
    main()
