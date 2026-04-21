"""v7 Grid B extraction: iterate (x, z), derive μ. Breaks the v4–v6 confound
where z was derived from (x, μ), creating x⟂z ≠ 0.

For each pair: 5 x-values × 5 z-values × 30 seeds = 750 prompts.
Derived μ values are sanity-checked; cells with implausible μ are dropped
and logged.

Writes:
  results/v7_xz_grid/e4b_{pair}_{layer}.npz
  results/v7_xz_grid/e4b_{pair}_logits.jsonl   (per-prompt logit_diff + softmax entropy)
  results/v7_xz_grid/e4b_trials.jsonl
  results/v7_xz_grid/dropped_cells.json
"""
from __future__ import annotations

import json
import math
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
from extract_v4_adjpairs import (  # noqa: E402
    PAIRS, LOG_SPACE_PAIRS, fmt_num, sample_context, compute_z,
)

MODEL_ID = "google/gemma-4-E4B"
LAYER_INDICES = {"mid": 21, "late": 32}
BATCH_SIZE = 16
Z_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
N_SEEDS = 30
OUT = REPO / "results" / "v7_xz_grid"
OUT.mkdir(parents=True, exist_ok=True)


# Per-pair domain constraints: μ must be in this range, else cell is dropped.
MU_BOUNDS = {
    "height":     (100.0, 210.0),
    "age":        (1.0,   80.0),
    "weight":     (30.0,  130.0),
    "size":       (1.0,   100.0),
    "speed":      (5.0,   220.0),
    "wealth":     (1000.0, 5_000_000.0),  # log-space
    "experience": (0.5,   40.0),
    "bmi_abs":    (10.0,  50.0),
}


def derive_mu(pair, x: float, z: float):
    """Given (x, z), derive μ using pair's sigma. Log-space handles wealth."""
    if pair.name in LOG_SPACE_PAIRS:
        # z = (log x - log μ)/log σ  ⇒  μ = x / σ^z
        mu = x / (pair.sigma ** z)
    else:
        mu = x - pair.sigma * z
    return mu


def is_plausible_mu(pair, mu: float) -> bool:
    lo, hi = MU_BOUNDS[pair.name]
    return lo <= mu <= hi


def build_implicit_items(pair, mu: float, seed: int, n: int = 15) -> list[str]:
    """Reuse v4 context sampler."""
    low = pair.target_values[0] * 0.4 if pair.target_values[0] > 0 else pair.target_values[0] - pair.sigma * 3
    high = pair.target_values[-1] * 2.5
    log_space = pair.name in LOG_SPACE_PAIRS
    sample = sample_context(mu, pair.sigma, seed, n, low, high, log_space=log_space)
    if pair.name == "height":
        return [f"Person {i+1}: {int(v)} cm" for i, v in enumerate(sample)]
    elif pair.name == "age":
        return [f"Person {i+1}: {int(v)} years old" for i, v in enumerate(sample)]
    elif pair.name == "weight":
        return [f"Person {i+1}: {int(v)} kg" for i, v in enumerate(sample)]
    elif pair.name == "size":
        return [f"Object {i+1}: {int(v)} cm across" for i, v in enumerate(sample)]
    elif pair.name == "speed":
        return [f"Vehicle {i+1}: {int(v)} km/h" for i, v in enumerate(sample)]
    elif pair.name == "wealth":
        return [f"Person {i+1} earns ${int(v)}/year" for i, v in enumerate(sample)]
    elif pair.name == "experience":
        return [f"Worker {i+1}: {int(v)} years experience" for i, v in enumerate(sample)]
    elif pair.name == "bmi_abs":
        return [f"Person {i+1}: BMI {v:.1f}" for i, v in enumerate(sample)]
    return [f"Item {i+1}: {v}" for i, v in enumerate(sample)]


def make_implicit_prompt(pair, x: float, mu: float, seed: int) -> str:
    items = build_implicit_items(pair, mu, seed)
    items_block = "\n".join(items)
    return pair.format_prompt_implicit.format(
        items=items_block, n_last=len(items) + 1, x_str=fmt_num(x),
    )


def build_trials():
    trials = []
    idx = 0
    dropped = {p.name: [] for p in PAIRS}
    for pair in PAIRS:
        for x in pair.target_values:
            for z in Z_VALUES:
                mu = derive_mu(pair, x, z)
                if not is_plausible_mu(pair, mu):
                    dropped[pair.name].append({"x": x, "z": z, "derived_mu": mu})
                    continue
                for s in range(N_SEEDS):
                    trials.append({
                        "id": f"{pair.name}_v7_{idx:06d}",
                        "pair": pair.name,
                        "condition": "implicit_xz_grid",
                        "prompt": make_implicit_prompt(pair, x, mu, s),
                        "x": float(x), "mu": float(mu), "z": float(z),
                        "sigma": pair.sigma, "seed": s,
                        "low_word": pair.low_word, "high_word": pair.high_word,
                    })
                    idx += 1
    return trials, dropped


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
    logit_diffs, entropies = [], []
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
            logit_diffs.append(ld); entropies.append(ent)
            for k in LAYER_INDICES:
                h = captured[k][0]
                per_layer[k].append(h[:, -1, :].float().cpu().numpy())
    finally:
        for h in handles: h.remove()
    return ({k: np.concatenate(v, axis=0) for k, v in per_layer.items()},
            np.concatenate(logit_diffs), np.concatenate(entropies))


def main():
    trials, dropped = build_trials()
    print(f"Built {len(trials)} prompts across {len(PAIRS)} pairs", flush=True)
    for p in PAIRS:
        n_kept = sum(1 for t in trials if t["pair"] == p.name)
        n_drop = len(dropped[p.name])
        print(f"  {p.name:12s}  kept={n_kept:4d}  dropped_cells={n_drop}/{5*5}  "
              f"{[(d['x'], d['z'], round(d['derived_mu'],1)) for d in dropped[p.name]]}")
    with (OUT / "e4b_trials.jsonl").open("w") as f:
        for t in trials: f.write(json.dumps(t) + "\n")
    (OUT / "dropped_cells.json").write_text(json.dumps(dropped, indent=2))

    print(f"\nLoading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    for pair in PAIRS:
        sub = [t for t in trials if t["pair"] == pair.name]
        if not sub:
            print(f"\n=== {pair.name}: 0 trials (all cells dropped), skip ===")
            continue
        print(f"\n=== {pair.name} ({pair.low_word}/{pair.high_word})  n={len(sub)} ===", flush=True)
        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)
        t1 = time.time()
        acts, ld, ent = extract_and_score(model, tok, sub, high_id, low_id)
        print(f"  extracted in {time.time()-t1:.1f}s  ld_mean={ld.mean():+.3f}  ent_mean={ent.mean():.3f}", flush=True)
        ids_arr = np.array([t["id"] for t in sub])
        for layer, a in acts.items():
            np.savez(OUT / f"e4b_{pair.name}_{layer}.npz",
                     activations=a.astype(np.float32),
                     ids=ids_arr,
                     layer_index=LAYER_INDICES[layer], layer_name=layer)
        with (OUT / f"e4b_{pair.name}_logits.jsonl").open("w") as f:
            for t, l, e in zip(sub, ld, ent):
                f.write(json.dumps({"id": t["id"],
                                     "logit_diff": float(l),
                                     "entropy": float(e)}) + "\n")
    print("\nDONE v7 extraction.")


if __name__ == "__main__":
    main()
