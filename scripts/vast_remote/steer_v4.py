"""Causal steering test: does adding α·ŵ_z to activations shift logit_diff?

Loads:
  - Probe weights from results/v4_analysis/probes/implicit_{mid,late}_probes.npz
    (writes via analyze_v4.py)
  - Tests steering on a small held-out set of prompts — the 35 explicit
    prompts + zero-shot set, which probes weren't trained on.

For each α in a sweep [-5, -2, -1, 0, +1, +2, +5]:
  1. Run the model once per prompt WITHOUT a hook, record baseline logit_diff.
  2. Register a forward hook on layer L that adds α·ŵ_z to the LAST token
     of the residual stream at that layer.
  3. Re-run, record steered logit_diff.

Expected: if w_z is causal at layer L, response curve is monotone.
          if w_z is purely correlational, response is flat.

Usage:
  python scripts/vast_remote/steer_v4.py [--layer late|mid]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E4B"
REPO = Path(__file__).resolve().parent.parent.parent
V4_DIR = REPO / "results" / "v4_dense"
PROBE_DIR = REPO / "results" / "v4_analysis" / "probes"
OUT_DIR = REPO / "results" / "v4_steering"

LAYER_TO_IDX = {"mid": 21, "late": 32}
ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError


def steered_forward(model, tokenizer, prompts, layer_idx, direction_unit, alpha, tall_id, short_id):
    """Run forward with an additive hook at `layer_idx` on the last token."""
    layers_mod = get_layers(model)
    direction_t = torch.tensor(direction_unit, dtype=model.dtype, device=model.device)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # Additive on last token only (leaves context untouched)
        h[:, -1, :] = h[:, -1, :] + alpha * direction_t
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = layers_mod[layer_idx].register_forward_hook(hook)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=1024).to(model.device)
    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    logits = out.logits[:, -1, :]
    tall = logits[:, tall_id].cpu().float().numpy()
    short = logits[:, short_id].cpu().float().numpy()
    return tall - short, tall, short


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", default="late", choices=["mid", "late"])
    ap.add_argument("--n-prompts", type=int, default=60,
                    help="Number of held-out prompts to steer on.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_idx = LAYER_TO_IDX[args.layer]
    probe_path = PROBE_DIR / f"implicit_{args.layer}_probes.npz"
    if not probe_path.exists():
        raise FileNotFoundError(f"Run analyze_v4.py first to produce {probe_path}")

    with np.load(probe_path, allow_pickle=True) as z:
        w_z = z["w_z"]
        w_adj = z["w_adj"]
    w_z_unit = w_z / (np.linalg.norm(w_z) + 1e-12)
    w_adj_unit = w_adj / (np.linalg.norm(w_adj) + 1e-12)

    # Held-out prompts: explicit (35) + zero_shot (5) from v4_dense trials
    # These were NOT used for probe training (probes trained on implicit).
    trials_path = V4_DIR / "e4b_trials.jsonl"
    trials = [json.loads(l) for l in trials_path.open()]
    held_out = [t for t in trials if t["condition"] in ("explicit", "zero_shot")]
    held_out = held_out[: args.n_prompts]
    # Reconstruct prompts on-the-fly
    from importlib import import_module
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
    ev4 = import_module("extract_v4_dense")
    prompts = []
    for t in held_out:
        if t["condition"] == "explicit":
            prompts.append(ev4.make_explicit_prompt(t["x"], t["mu"]))
        else:
            prompts.append(ev4.make_zero_shot_prompt(t["x"]))

    print(f"Loading {MODEL_ID}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    tall_id = tok.encode("tall", add_special_tokens=False)[0]
    short_id = tok.encode("short", add_special_tokens=False)[0]

    results = {"layer": args.layer, "layer_idx": layer_idx,
               "n_prompts": len(prompts), "alphas": ALPHAS,
               "probes": ["w_z", "w_adj"],
               "curves": {}}

    for probe_name, direction in [("w_z", w_z_unit), ("w_adj", w_adj_unit)]:
        print(f"\n=== Steering along {probe_name} @ layer {args.layer} (idx {layer_idx}) ===")
        curve = []
        for alpha in ALPHAS:
            # Batch for speed
            batch = 16
            all_diffs = []
            for i in range(0, len(prompts), batch):
                diffs, _, _ = steered_forward(
                    model, tok, prompts[i:i+batch],
                    layer_idx, direction, alpha, tall_id, short_id)
                all_diffs.extend(diffs.tolist())
            mean_diff = float(np.mean(all_diffs))
            std_diff = float(np.std(all_diffs, ddof=1))
            curve.append({"alpha": alpha, "mean_logit_diff": mean_diff, "std": std_diff})
            print(f"  α={alpha:+6.2f}  mean_logit_diff={mean_diff:+7.3f}  sd={std_diff:.3f}")
        results["curves"][probe_name] = curve

    out_path = OUT_DIR / f"steering_{args.layer}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Simple monotonicity check
    for probe_name, curve in results["curves"].items():
        ds = [c["mean_logit_diff"] for c in curve]
        mono = all(ds[i] <= ds[i+1] for i in range(len(ds)-1)) or \
               all(ds[i] >= ds[i+1] for i in range(len(ds)-1))
        slope_per_alpha = (ds[-1] - ds[0]) / (ALPHAS[-1] - ALPHAS[0])
        print(f"{probe_name}: monotone={mono}  slope={slope_per_alpha:+.3f}/α")


if __name__ == "__main__":
    main()
