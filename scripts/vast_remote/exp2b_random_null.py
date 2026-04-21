"""Exp 2b: random-direction null for the meta-direction steering.

Addresses the statistical-critic flag that exp2_meta_steer.py lacks a null.
Steers with 3 random unit vectors in d=2560 at the same α grid and layer.
If the meta-direction effect is specific (not a generic "perturb the last
token" effect), slopes should be larger for w₁ than for random w's.

Uses exp2's existing per-pair 100-prompt subsample for direct comparison.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS, compute_z, make_implicit_prompt  # noqa: E402

MODEL_ID = "google/gemma-4-E4B"
LAYER_IDX = 32
ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
N_PROMPTS_PER_PAIR = 100
BATCH_SIZE = 16
N_RANDOM = 3
D = 2560

OUT = REPO / "results" / "v4_steering"
OUT_FIG = REPO / "figures" / "v4_adjpairs"


def subsample_pair_prompts(pair_obj, n: int, rng: np.random.Generator):
    all_trials = []
    idx = 0
    for x in pair_obj.target_values:
        for mu in pair_obj.mu_values:
            z = compute_z(pair_obj, x, mu)
            for s in range(30):
                all_trials.append({
                    "prompt": make_implicit_prompt(pair_obj, x, mu, s),
                    "x": x, "mu": mu, "z": z, "seed": s,
                    "low_word": pair_obj.low_word, "high_word": pair_obj.high_word,
                })
                idx += 1
    picks = rng.choice(len(all_trials), size=min(n, len(all_trials)), replace=False)
    return [all_trials[i] for i in picks]


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def steered_ld(model, tok, prompts, layer_idx, direction, alpha, high_id, low_id):
    layers = get_layers(model)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = direction.to(device=h.device, dtype=h.dtype)
        h[:, -1, :] = h[:, -1, :] + alpha * d
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = layers[layer_idx].register_forward_hook(hook)
    out_ld = []
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            ld = (logits[:, high_id] - logits[:, low_id]).float().cpu().numpy()
            out_ld.append(ld)
    finally:
        handle.remove()
    return np.concatenate(out_ld)


def main():
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # 3 random unit directions (same norm as w₁, which is unit)
    rng_dirs = np.random.default_rng(42)
    random_dirs = []
    for ri in range(N_RANDOM):
        v = rng_dirs.standard_normal(D).astype(np.float64)
        v = v / np.linalg.norm(v)
        random_dirs.append(v)

    # Same prompt subsample RNG as Exp 2 (seed=0)
    rng_p = np.random.default_rng(0)
    result = {"alphas": ALPHAS, "n_random": N_RANDOM, "random": {}}
    for pair_obj in PAIRS:
        result["random"][pair_obj.name] = {"per_random": []}
        trials = subsample_pair_prompts(pair_obj, N_PROMPTS_PER_PAIR, rng_p)
        prompts = [t["prompt"] for t in trials]
        low_id = first_token_id(tok, pair_obj.low_word)
        high_id = first_token_id(tok, pair_obj.high_word)
        print(f"\n=== {pair_obj.name} ({pair_obj.low_word}/{pair_obj.high_word}) ===", flush=True)
        for ri, v_np in enumerate(random_dirs):
            direction = torch.from_numpy(v_np).to(model.device)
            per_alpha = {}
            t_start = time.time()
            for alpha in ALPHAS:
                ld = steered_ld(model, tok, prompts, LAYER_IDX, direction, alpha, high_id, low_id)
                per_alpha[str(alpha)] = {"logit_diff_mean": float(ld.mean()),
                                          "logit_diff_std": float(ld.std())}
            xs = np.array(ALPHAS)
            means = np.array([per_alpha[str(a)]["logit_diff_mean"] for a in ALPHAS])
            slope = float(np.polyfit(xs, means, 1)[0])
            result["random"][pair_obj.name]["per_random"].append({
                "random_seed_idx": ri,
                "curve": per_alpha,
                "slope": slope,
                "range": float(means.max() - means.min()),
            })
            print(f"  random#{ri}: slope={slope:+.4f}  range={means.max()-means.min():.3f}  ({time.time()-t_start:.1f}s)", flush=True)

    # Load meta-w1 slope summary
    meta = json.load((OUT / "meta_w1_steering.json").open())
    meta_slopes = {}
    for p, r in meta["per_pair"].items():
        ys = [r["curve"][str(a)]["logit_diff_mean"] for a in ALPHAS]
        meta_slopes[p] = float(np.polyfit(ALPHAS, ys, 1)[0])

    # Summarize: is w1 slope larger than random?
    print("\n=== META vs RANDOM null ===")
    print(f"{'pair':15s}  w1_slope    rand_slope_mean±std      excess (|w1|-|rand|)")
    for p in meta_slopes:
        rs = [r["slope"] for r in result["random"][p]["per_random"]]
        rs_mean, rs_std = float(np.mean(rs)), float(np.std(rs, ddof=1))
        excess = abs(meta_slopes[p]) - np.mean([abs(s) for s in rs])
        print(f"{p:15s}  {meta_slopes[p]:+.4f}     {rs_mean:+.4f}±{rs_std:.4f}         {excess:+.4f}")
        result["random"][p]["w1_slope"] = meta_slopes[p]
        result["random"][p]["random_slopes"] = rs
        result["random"][p]["excess_abs_slope"] = float(excess)

    (OUT / "meta_w1_random_null.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT/'meta_w1_random_null.json'}")

    # Plot: |slope| of w1 vs 3 random directions
    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(meta_slopes)
    xpos = np.arange(len(names))
    w1 = [abs(meta_slopes[p]) for p in names]
    rand_mean = [float(np.mean([abs(s) for s in result["random"][p]["random_slopes"]])) for p in names]
    rand_std = [float(np.std([abs(s) for s in result["random"][p]["random_slopes"]], ddof=1)) for p in names]
    ax.bar(xpos - 0.2, w1, 0.4, label="|slope| along w₁ (meta direction)")
    ax.bar(xpos + 0.2, rand_mean, 0.4, yerr=rand_std, capsize=3,
           label=f"|slope| along {N_RANDOM} random directions (mean±std)")
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("|slope of logit_diff vs α|  (per α-unit)")
    ax.set_title("Exp 2b: is the meta-direction specific, or does ANY perturbation steer?")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "meta_w1_vs_random_null.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'meta_w1_vs_random_null.png'}")


if __name__ == "__main__":
    main()
