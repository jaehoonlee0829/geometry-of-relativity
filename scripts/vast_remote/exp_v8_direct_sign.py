"""v8 Priority 1+2: direct sign classification — 4 prompt variants + top-K tokens.

Tests whether v7's posneg R=0.47 is driven by prompt ambiguity. If the
original prompt "This number is ___" elicits relative-position adjectives
rather than sign classification, a prompt that EXPLICITLY asks about
sign should show R ≈ 0. Otherwise, the model genuinely treats sign as
context-relative.

Grid B (x, z): 5 x × 5 z × 30 seeds = 750 prompts per variant.
4 variants × 750 = 3000 prompts.
Logs top-10 tokens per prompt for qualitative analysis.

Writes:
  results/v8_direct_sign/e4b_{variant}_logits.jsonl
  results/v8_direct_sign/e4b_{variant}_top10.jsonl
  results/v8_direct_sign/e4b_trials.jsonl   (shared across variants)
  results/v8_direct_sign/summary.json
  figures/v8/direct_sign_comparison.png
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
BATCH_SIZE = 16
X_VALUES = [-8.0, -3.0, 0.0, 3.0, 8.0]
Z_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
SIGMA = 3.0
N_SEEDS = 30
TOP_K = 10

OUT = REPO / "results" / "v8_direct_sign"
OUT_FIG = REPO / "figures" / "v8"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


# Each variant: name, template (uses {items} and {x_str}), high_word, low_word.
# Template ends at the position where the next token is what we score.
VARIANTS = [
    {
        "name": "orig",
        "template": "{items}\nNumber {n_last}: {x_str}. This number is",
        "high_word": "positive",
        "low_word":  "negative",
    },
    {
        "name": "compared",
        "template": "{items}\nNumber {n_last}: {x_str}. Compared to zero, this number is",
        "high_word": "above",
        "low_word":  "below",
    },
    {
        "name": "relative",
        "template": "{items}\nNumber {n_last}: {x_str}. Relative to zero, this number is",
        "high_word": "higher",
        "low_word":  "lower",
    },
    {
        "name": "forced_qa",
        "template": "{items}\nNumber {n_last}: {x_str}. Is this number above or below zero? Answer:",
        "high_word": "Above",
        "low_word":  "Below",
    },
]


def fmt(v): return str(int(v)) if v == int(v) else f"{v:.1f}"


def sample_context(mu, seed, n=15):
    rng = np.random.default_rng(seed + 98765)
    out = []
    while len(out) < n:
        v = rng.normal(mu, SIGMA)
        if -40 <= v <= 40:
            out.append(round(v, 1))
    return out


def build_trials():
    trials = []
    idx = 0
    for x in X_VALUES:
        for z in Z_VALUES:
            mu = x - SIGMA * z
            for s in range(N_SEEDS):
                sample = sample_context(mu, s)
                items_block = "\n".join(f"Number {i+1}: {v:g}" for i, v in enumerate(sample))
                trials.append({
                    "id": f"posneg_v8_{idx:06d}",
                    "x": float(x), "mu": float(mu), "z": float(z),
                    "sigma": SIGMA, "seed": s,
                    "items": items_block,
                    "n_last": len(sample) + 1,
                    "x_str": fmt(x),
                })
                idx += 1
    return trials


def run_variant(model, tok, trials, variant):
    prompts = [variant["template"].format(**{"items": t["items"], "n_last": t["n_last"], "x_str": t["x_str"]})
               for t in trials]
    high_id = first_token_id(tok, variant["high_word"])
    low_id = first_token_id(tok, variant["low_word"])

    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lds, top10s = [], []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        logprobs = torch.log_softmax(logits.double(), dim=-1)
        ld = (logits[:, high_id] - logits[:, low_id]).float().cpu().numpy()
        topv, topi = logprobs.topk(TOP_K, dim=-1)
        lds.append(ld)
        for b in range(topi.shape[0]):
            top10s.append({
                "tokens": [tok.decode([int(idx)]) for idx in topi[b].tolist()],
                "logprobs": topv[b].float().cpu().numpy().tolist(),
            })
    ld_arr = np.concatenate(lds)
    return ld_arr, top10s, high_id, low_id


def analyze(trials, ld):
    x = np.array([t["x"] for t in trials])
    mu = np.array([t["mu"] for t in trials])
    z = np.array([t["z"] for t in trials])
    X = np.column_stack([x, mu, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, ld, rcond=None)
    slope_x, slope_mu, _ = beta
    rr = -slope_mu / slope_x if abs(slope_x) > 1e-9 else None
    # Classification accuracy (x ≠ 0 only)
    nonzero = x != 0
    acc = float(np.mean(np.sign(ld[nonzero]) == np.sign(x[nonzero])))
    return {
        "slope_x": float(slope_x),
        "slope_mu": float(slope_mu),
        "relativity_ratio": float(rr) if rr is not None else None,
        "classification_accuracy_nonzero": acc,
        "ld_mean": float(ld.mean()),
        "ld_std": float(ld.std()),
        "ld_at_x_minus_8": float(ld[x == -8].mean()),
        "ld_at_x_0":       float(ld[x == 0].mean()),
        "ld_at_x_plus_8":  float(ld[x == +8].mean()),
    }


def main():
    trials = build_trials()
    print(f"Built {len(trials)} prompts (Grid B)")
    xs = np.array([t["x"] for t in trials])
    zs = np.array([t["z"] for t in trials])
    print(f"  corr(x, z) = {np.corrcoef(xs, zs)[0,1]:+.4f} (should be 0)")
    with (OUT / "e4b_trials.jsonl").open("w") as f:
        for t in trials: f.write(json.dumps(t) + "\n")

    print(f"\nLoading {MODEL_ID}…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    summary = {}
    for v in VARIANTS:
        print(f"\n=== variant: {v['name']}  (score {v['high_word']!r} − {v['low_word']!r}) ===")
        t1 = time.time()
        ld, top10s, hi, lo = run_variant(model, tok, trials, v)
        print(f"  {len(trials)} prompts in {time.time()-t1:.1f}s   ld_mean={ld.mean():+.3f}")

        a = analyze(trials, ld)
        a["high_word"] = v["high_word"]; a["low_word"] = v["low_word"]
        a["high_id"] = int(hi); a["low_id"] = int(lo)
        a["template"] = v["template"]
        summary[v["name"]] = a
        print(f"  R = {a['relativity_ratio']:+.3f}  accuracy = {a['classification_accuracy_nonzero']:.3f}")
        print(f"  ld @ x=-8: {a['ld_at_x_minus_8']:+.3f}   ld @ x=0: {a['ld_at_x_0']:+.3f}   ld @ x=+8: {a['ld_at_x_plus_8']:+.3f}")

        with (OUT / f"e4b_{v['name']}_logits.jsonl").open("w") as f:
            for t, l in zip(trials, ld):
                f.write(json.dumps({"id": t["id"], "logit_diff": float(l)}) + "\n")
        with (OUT / f"e4b_{v['name']}_top10.jsonl").open("w") as f:
            for t, top in zip(trials, top10s):
                f.write(json.dumps({"id": t["id"], "top10": top}) + "\n")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'summary.json'}")

    # Compare R across variants
    print("\n=== ACROSS-VARIANT COMPARISON ===")
    print(f"{'variant':12s}  high/low         R         acc     ld(-8)    ld(0)     ld(+8)")
    for v in VARIANTS:
        a = summary[v["name"]]
        print(f"  {v['name']:10s}  {v['high_word']:>8s}/{v['low_word']:<8s}  {a['relativity_ratio']:+.3f}   {a['classification_accuracy_nonzero']:.3f}   "
              f"{a['ld_at_x_minus_8']:+.3f}   {a['ld_at_x_0']:+.3f}   {a['ld_at_x_plus_8']:+.3f}")

    # Figure: R per variant + ld curve per variant
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [v["name"] for v in VARIANTS]
    Rs = [summary[n]["relativity_ratio"] for n in names]
    accs = [summary[n]["classification_accuracy_nonzero"] for n in names]
    xpos = np.arange(len(names))
    axes[0].bar(xpos-0.2, Rs, 0.4, label="relativity ratio R")
    axes[0].bar(xpos+0.2, accs, 0.4, label="classification accuracy")
    axes[0].set_xticks(xpos); axes[0].set_xticklabels(names, rotation=20)
    axes[0].axhline(0, color="k", lw=0.5); axes[0].axhline(1, color="k", ls="--", lw=0.5, alpha=0.3)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Relativity vs accuracy per prompt variant")

    # ld by x (averaged across z)
    xs_arr = np.array([t["x"] for t in trials])
    for v in VARIANTS:
        lds = np.array([json.loads(l)["logit_diff"] for l in (OUT / f"e4b_{v['name']}_logits.jsonl").open()])
        x_unique = sorted(set(xs_arr))
        y = [lds[xs_arr == xv].mean() for xv in x_unique]
        axes[1].plot(x_unique, y, marker="o", label=f"{v['name']}  R={summary[v['name']]['relativity_ratio']:+.2f}")
    axes[1].axhline(0, color="k", lw=0.5, alpha=0.5)
    axes[1].axvline(0, color="k", lw=0.5, alpha=0.5)
    axes[1].set_xlabel("x (signed value)"); axes[1].set_ylabel("logit_diff (high - low)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_title("logit_diff vs x, averaged over z")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "direct_sign_comparison.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'direct_sign_comparison.png'}")


if __name__ == "__main__":
    main()
