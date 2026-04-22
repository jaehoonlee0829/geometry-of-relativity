"""v9 Priority 1: Replicate v7 Grid B extraction on `google/gemma-2-2b`.

Motivation: enables SAE analysis via pretrained Gemma Scope SAEs. If the
behavioral signal (logit_diff vs z) reproduces on Gemma 2 2B, we can use
the Gemma Scope residual-stream SAEs to decompose z-direction features.

Uses the SAME Grid B trials as v7 (5 x-values × 5 z-values × 30 seeds per
pair, μ derived and bounds-checked) so acceptance comparisons are apples to
apples with v7's E4B results.

Writes:
  results/v9_gemma2/gemma2_{pair}_{layer}.npz
  results/v9_gemma2/gemma2_{pair}_logits.jsonl
  results/v9_gemma2/gemma2_trials.jsonl
  results/v9_gemma2/dropped_cells.json

Acceptance criteria (per docs/NEXT_GPU_SESSION_v9.md):
  - Relativity ratio R > 0.3 for at least 5/8 pairs
  - logit_diff heatmap shows z-gradient

Wall time: ~5 min on a single H100 (2B params, 26 layers, ~5600 prompts).
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
from _token_utils import first_token_id, report_tokenization  # noqa: E402
from extract_v4_adjpairs import PAIRS, LOG_SPACE_PAIRS, fmt_num, sample_context  # noqa: E402
from extract_v7_xz_grid import (  # noqa: E402
    Z_VALUES, N_SEEDS, MU_BOUNDS, derive_mu, is_plausible_mu,
    build_implicit_items, make_implicit_prompt, build_trials, get_layers,
)

MODEL_ID = "google/gemma-2-2b"
# Gemma 2 2B has 26 transformer blocks, hidden size 2304.
# Use mid=13 and late=20 per v9 plan (matches Gemma Scope SAE availability).
LAYER_INDICES = {"mid": 13, "late": 20}
BATCH_SIZE = 16
OUT = REPO / "results" / "v9_gemma2"
OUT.mkdir(parents=True, exist_ok=True)


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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(model.device)
            for k in captured:
                captured[k].clear()
            with torch.no_grad():
                out = model(**enc)
            last = out.logits[:, -1, :]
            logprobs = torch.log_softmax(last.double(), dim=-1)
            ent = -(logprobs.exp() * logprobs).sum(-1).float().cpu().numpy()
            ld = (last[:, high_id] - last[:, low_id]).float().cpu().numpy()
            logit_diffs.append(ld)
            entropies.append(ent)
            for k in LAYER_INDICES:
                h = captured[k][0]
                per_layer[k].append(h[:, -1, :].float().cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    return (
        {k: np.concatenate(v, axis=0) for k, v in per_layer.items()},
        np.concatenate(logit_diffs),
        np.concatenate(entropies),
    )


def main():
    trials, dropped = build_trials()
    print(f"Built {len(trials)} prompts across {len(PAIRS)} pairs", flush=True)
    for p in PAIRS:
        n_kept = sum(1 for t in trials if t["pair"] == p.name)
        n_drop = len(dropped[p.name])
        print(f"  {p.name:12s}  kept={n_kept:4d}  dropped_cells={n_drop}/{5*5}")

    with (OUT / "gemma2_trials.jsonl").open("w") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")
    (OUT / "dropped_cells.json").write_text(json.dumps(dropped, indent=2))

    print(f"\nLoading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    # Sanity-check tokenization of all adjective words on THIS tokenizer.
    words = sorted({w for p in PAIRS for w in (p.low_word, p.high_word)})
    print("\nTokenization check (gemma-2-2b):")
    report_tokenization(tok, words)

    for pair in PAIRS:
        sub = [t for t in trials if t["pair"] == pair.name]
        if not sub:
            print(f"\n=== {pair.name}: 0 trials (all cells dropped), skip ===")
            continue
        print(f"\n=== {pair.name} ({pair.low_word}/{pair.high_word})  n={len(sub)} ===",
              flush=True)
        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)
        t1 = time.time()
        acts, ld, ent = extract_and_score(model, tok, sub, high_id, low_id)
        print(f"  extracted in {time.time() - t1:.1f}s  "
              f"ld_mean={ld.mean():+.3f}  ent_mean={ent.mean():.3f}", flush=True)
        ids_arr = np.array([t["id"] for t in sub])
        for layer, a in acts.items():
            np.savez(OUT / f"gemma2_{pair.name}_{layer}.npz",
                     activations=a.astype(np.float32),
                     ids=ids_arr,
                     layer_index=LAYER_INDICES[layer], layer_name=layer)
        with (OUT / f"gemma2_{pair.name}_logits.jsonl").open("w") as f:
            for t, l, e in zip(sub, ld, ent):
                f.write(json.dumps({"id": t["id"],
                                    "logit_diff": float(l),
                                    "entropy": float(e)}) + "\n")
    print("\nDONE v9 Priority 1 extraction.")


if __name__ == "__main__":
    main()
