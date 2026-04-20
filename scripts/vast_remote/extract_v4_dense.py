"""v4 dense extraction: 100 seeds per (x, mu) cell, three conditions, activations + logits.

Run on Vast GPU instance. Self-contained — no imports from src/.

Conditions:
  1. implicit: list of 15 heights sampled from N(mu, sigma), then target
  2. explicit: "In a group where heights cluster around {mu} cm (±{sigma} cm), a person who is {x} cm is"
  3. zero_shot: "A person who is {x} cm is" (no context — control)

Extracts:
  - Last-token activation at "is" (shape: (N, d))
  - logit("tall") - logit("short") per prompt
  - Top-5 token logits per prompt (to see what the model actually predicts)

Output:
  results/v4_dense/{model}_{condition}_{layer}.npz  — activations + metadata
  results/v4_dense/{model}_logits.jsonl              — per-prompt logit info
"""
import json
import math
import random
import sys
import time
import gc
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Shared tokenization helper (co-located). Spacing-aware first_token_id
# avoids the `encode("tall")[0]` pitfall of grabbing a no-space prefix token
# that the model never predicts after "... is".
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _token_utils import first_token_id, report_tokenization  # noqa: E402

# ---- Config ----
MODEL_ID = "google/gemma-4-E4B"
SHORT = "e4b"
LAYER_INDICES = {"mid": 21, "late": 32}  # Focus on 2 layers, not 4
BATCH_SIZE = 16

# Experiment grid
X_VALUES = [150.0, 160.0, 165.0, 170.0, 180.0]  # 5 target heights
MU_VALUES = [145.0, 150.0, 160.0, 165.0, 170.0, 180.0, 185.0]  # 7 context means
SIGMA = 10.0
N_SEEDS = 100  # replicates per (x, mu) cell

OUT = Path("results/v4_dense")
OUT.mkdir(parents=True, exist_ok=True)


# ---- Prompt generation ----

def sample_context(mu: float, sigma: float, seed: int, n: int = 15) -> list[float]:
    """Sample n heights from N(mu, sigma), rounded to int, clipped."""
    rng = random.Random(seed)
    return [float(max(120, min(210, round(rng.gauss(mu, sigma))))) for _ in range(n)]


def make_implicit_prompt(x: float, mu: float, seed: int) -> str:
    """Implicit context: list 15 people, then target."""
    sample = sample_context(mu, SIGMA, seed)
    lines = [f"Person {i+1}: {int(v)} cm" for i, v in enumerate(sample)]
    lines.append(f"Person {len(sample)+1}: {int(x)} cm. This person is")
    return "\n".join(lines)


def make_explicit_prompt(x: float, mu: float) -> str:
    """Explicit context: state mu and sigma."""
    return (
        f"In a group where most people's heights cluster around {int(mu)} cm "
        f"(give or take {int(SIGMA)} cm), a person who is {int(x)} cm is"
    )


def make_zero_shot_prompt(x: float) -> str:
    """Zero-shot: no context at all (control)."""
    return f"A person who is {int(x)} cm is"


def generate_all_prompts():
    """Generate all prompts. Returns list of dicts with id, prompt, x, mu, z, condition, seed."""
    trials = []
    idx = 0

    # Implicit: 5 x * 7 mu * 100 seeds = 3500
    for x in X_VALUES:
        for mu in MU_VALUES:
            z = (x - mu) / SIGMA
            for seed in range(N_SEEDS):
                trials.append({
                    "id": f"implicit_{idx:05d}",
                    "condition": "implicit",
                    "prompt": make_implicit_prompt(x, mu, seed),
                    "x": x, "mu": mu, "z": z, "seed": seed,
                })
                idx += 1

    # Explicit: 5 x * 7 mu * 1 (deterministic) = 35
    for x in X_VALUES:
        for mu in MU_VALUES:
            z = (x - mu) / SIGMA
            trials.append({
                "id": f"explicit_{idx:05d}",
                "condition": "explicit",
                "prompt": make_explicit_prompt(x, mu),
                "x": x, "mu": mu, "z": z, "seed": -1,
            })
            idx += 1

    # Zero-shot: 5 x * 1 = 5
    for x in X_VALUES:
        trials.append({
            "id": f"zeroshot_{idx:05d}",
            "condition": "zero_shot",
            "prompt": make_zero_shot_prompt(x),
            "x": x, "mu": 0.0, "z": 0.0, "seed": -1,
        })
        idx += 1

    return trials


# ---- Model helpers ----

def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError(f"no layers on {type(model).__name__}")


def get_unembedding(model):
    if hasattr(model, "lm_head") and model.lm_head.weight is not None:
        return model.lm_head.weight
    return model.get_input_embeddings().weight


# ---- Main extraction ----

def extract_batch(model, tokenizer, prompts, layer_indices, tall_id, short_id):
    """Run a batch of prompts. Returns activations per layer + logit info per prompt."""
    layers_mod = get_layers(model)
    captured = {k: [] for k in layer_indices}
    handles = []

    def make_hook(key):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[key].append(h.detach())
        return hook

    for key, idx in layer_indices.items():
        handles.append(layers_mod[idx].register_forward_hook(make_hook(key)))

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=1024).to(model.device)

    for k in captured:
        captured[k].clear()

    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    # Extract last-token activations
    acts = {}
    for k in layer_indices:
        h = captured[k][0]  # (batch, seq, d)
        last = h[:, -1, :].float().cpu().numpy()
        acts[k] = last

    # Extract logits at last position
    logits = outputs.logits[:, -1, :]  # (batch, vocab)

    logit_info = []
    for i in range(len(prompts)):
        row_logits = logits[i]
        tall_logit = float(row_logits[tall_id].cpu())
        short_logit = float(row_logits[short_id].cpu())

        # Top 5 tokens
        top5_vals, top5_ids = torch.topk(row_logits, 5)
        top5_tokens = tokenizer.convert_ids_to_tokens(top5_ids.cpu().tolist())
        top5_logits = top5_vals.cpu().tolist()

        logit_info.append({
            "logit_tall": tall_logit,
            "logit_short": short_logit,
            "logit_diff": tall_logit - short_logit,
            "top5_tokens": top5_tokens,
            "top5_logits": [round(v, 3) for v in top5_logits],
        })

    for h in handles:
        h.remove()

    return acts, logit_info


def main():
    print(f"Generating prompts...", flush=True)
    trials = generate_all_prompts()
    print(f"  Total: {len(trials)} prompts", flush=True)
    print(f"  Implicit: {sum(1 for t in trials if t['condition']=='implicit')}", flush=True)
    print(f"  Explicit: {sum(1 for t in trials if t['condition']=='explicit')}", flush=True)
    print(f"  Zero-shot: {sum(1 for t in trials if t['condition']=='zero_shot')}", flush=True)

    # Save trial metadata
    meta_path = OUT / f"{SHORT}_trials.jsonl"
    with meta_path.open("w") as f:
        for t in trials:
            # Don't save prompt text (too large), save everything else
            row = {k: v for k, v in t.items() if k != "prompt"}
            f.write(json.dumps(row) + "\n")
    print(f"  Metadata: {meta_path}", flush=True)

    print(f"\nLoading {MODEL_ID}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # Find token IDs for "tall" and "short" via the unified spacing-aware
    # helper. This matches extract_v4_adjpairs.py so logit_diff values are
    # directly comparable across the two extraction scripts.
    print("  Tokenization check:", flush=True)
    report_tokenization(tokenizer, ["tall", "short"])
    tall_id = first_token_id(tokenizer, "tall")
    short_id = first_token_id(tokenizer, "short")

    # Process by condition
    for condition in ["implicit", "explicit", "zero_shot"]:
        cond_trials = [t for t in trials if t["condition"] == condition]
        if not cond_trials:
            continue

        print(f"\n=== {condition} ({len(cond_trials)} prompts) ===", flush=True)
        prompts = [t["prompt"] for t in cond_trials]
        ids = [t["id"] for t in cond_trials]

        all_acts = {k: [] for k in LAYER_INDICES}
        all_logit_info = []

        t0 = time.time()
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            acts, logit_info = extract_batch(
                model, tokenizer, batch_prompts, LAYER_INDICES, tall_id, short_id
            )
            for k in LAYER_INDICES:
                all_acts[k].append(acts[k])
            all_logit_info.extend(logit_info)

            if (i // BATCH_SIZE) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + len(batch_prompts)) / elapsed if elapsed > 0 else 0
                print(f"  {i+len(batch_prompts)}/{len(prompts)} ({rate:.1f} p/s)", flush=True)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} p/s)", flush=True)

        # Save activations per layer
        for layer_key in LAYER_INDICES:
            stacked = np.concatenate(all_acts[layer_key], axis=0)
            out_path = OUT / f"{SHORT}_{condition}_{layer_key}.npz"
            np.savez(
                out_path,
                activations=stacked.astype(np.float32),
                ids=np.array(ids),
                layer_index=LAYER_INDICES[layer_key],
            )
            print(f"  Saved {out_path}: {stacked.shape}", flush=True)

        # Save logit info
        logit_path = OUT / f"{SHORT}_{condition}_logits.jsonl"
        with logit_path.open("w") as f:
            for tid, info in zip(ids, all_logit_info):
                row = {"id": tid, **info}
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {logit_path}", flush=True)

    # ---- Quick analysis ----
    print(f"\n{'='*60}")
    print("QUICK ANALYSIS")
    print(f"{'='*60}")

    # Load implicit logits and check behavioral signal
    logit_path = OUT / f"{SHORT}_implicit_logits.jsonl"
    with logit_path.open() as f:
        logit_data = [json.loads(l) for l in f]

    # Load implicit trial metadata
    meta = []
    with meta_path.open() as f:
        meta = [json.loads(l) for l in f]
    implicit_meta = [m for m in meta if m["condition"] == "implicit"]

    # Compute mean logit_diff per (x, mu) cell
    from collections import defaultdict
    cell_diffs = defaultdict(list)
    for m, ld in zip(implicit_meta, logit_data):
        cell_diffs[(m["x"], m["mu"])].append(ld["logit_diff"])

    print(f"\nMean logit(tall)-logit(short) per (x, mu) cell:")
    print(f"{'x':>6}  ", end="")
    for mu in sorted(set(m["mu"] for m in implicit_meta)):
        print(f"{'mu='+str(int(mu)):>10}", end="")
    print()

    for x in sorted(set(m["x"] for m in implicit_meta)):
        print(f"{int(x):>6}  ", end="")
        for mu in sorted(set(m["mu"] for m in implicit_meta)):
            diffs = cell_diffs.get((x, mu), [])
            if diffs:
                mean_diff = np.mean(diffs)
                print(f"{mean_diff:>+10.2f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    # Top-5 tokens summary for a few cells
    print(f"\nTop-5 predicted tokens (first seed of selected cells):")
    for x, mu in [(165, 145), (165, 165), (165, 185)]:
        matching = [(m, ld) for m, ld in zip(implicit_meta, logit_data)
                    if m["x"] == x and m["mu"] == mu]
        if matching:
            m, ld = matching[0]
            print(f"  x={int(x)}, mu={int(mu)}, z={m['z']:+.1f}: "
                  f"{list(zip(ld['top5_tokens'], ld['top5_logits']))}")

    print(f"\nDONE. All outputs in {OUT}/", flush=True)


if __name__ == "__main__":
    main()
