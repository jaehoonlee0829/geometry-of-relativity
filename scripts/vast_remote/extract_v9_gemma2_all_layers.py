"""v9 §13: extract Gemma 2 2B activations at ALL 26 decoder layers on Grid B.

The v9 P1 extraction captured only layers 13 and 20. For a full layer-sweep
(probe-R² by depth, intrinsic-dim by depth, primal-direction continuity,
LFP-Gram evolution), we need the activation at the last content token of
EVERY decoder block.

Strategy: register a forward hook on every `model.model.layers[k]` once,
run the same Grid B trials we built for P1, and dump a single
(n_prompts, 26, d_model=2304) array per pair.

Writes
  results/v9_gemma2/gemma2_{pair}_alllayers.npz
    activations: (n, 26, 2304) float32
    ids: (n,) str
    layer_indices: (26,) int

Expected wall time: ~2 min on H100 (single forward pass per prompt; the
26 hooks are cheap).
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
from extract_v4_adjpairs import PAIRS  # noqa: E402
from extract_v7_xz_grid import build_trials  # noqa: E402
from exp_v9_on_manifold_steering import get_layers  # noqa: E402

MODEL_ID = "google/gemma-2-2b"
BATCH_SIZE = 16
RES_DIR = REPO / "results" / "v9_gemma2"
RES_DIR.mkdir(parents=True, exist_ok=True)


def extract_all_layers(model, tok, trials):
    """Return (acts, ids) where acts has shape (n, n_layers, d_model)."""
    layers = get_layers(model)
    n_layers = len(layers)
    captured = [[] for _ in range(n_layers)]

    handles = []
    for k in range(n_layers):
        def make_hook(kk):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[kk].append(h[:, -1, :].detach().float().cpu().numpy())
            return hook
        handles.append(layers[k].register_forward_hook(make_hook(k)))

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = [t["prompt"] for t in trials]
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                _ = model(**enc)
    finally:
        for h in handles:
            h.remove()

    per_layer = [np.concatenate(v, axis=0) for v in captured]
    acts = np.stack(per_layer, axis=1)  # (n, n_layers, d)
    return acts, np.array([t["id"] for t in trials])


def main():
    # Re-materialize the exact Grid B trials v9 P1 used.
    trials, _ = build_trials()
    trials_by_pair = {}
    for t in trials:
        trials_by_pair.setdefault(t["pair"], []).append(t)

    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    n_layers = len(get_layers(model))
    print(f"Model has {n_layers} decoder layers", flush=True)

    for pair in PAIRS:
        pair_trials = trials_by_pair.get(pair.name, [])
        if not pair_trials:
            continue
        print(f"\n=== {pair.name}  n={len(pair_trials)} ===", flush=True)
        t1 = time.time()
        acts, ids = extract_all_layers(model, tok, pair_trials)
        print(f"  extracted in {time.time() - t1:.1f}s  shape={acts.shape}",
              flush=True)
        out = RES_DIR / f"gemma2_{pair.name}_alllayers.npz"
        np.savez(out,
                 activations=acts.astype(np.float32),
                 ids=ids,
                 layer_indices=np.arange(n_layers))
        print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
