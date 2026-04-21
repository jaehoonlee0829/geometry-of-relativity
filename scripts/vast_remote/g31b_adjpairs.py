"""G31B secondary pipeline: extract adjpair activations + logit_diff for all 8
pairs (implicit only) and verify the meta-direction finding replicates.

Uses existing prompt generation from extract_v4_adjpairs (same trials).

Writes:
  results/v4_adjpairs/g31b_{pair}_implicit_{mid|late}.npz
  results/v4_adjpairs/g31b_{pair}_implicit_logits.jsonl
  results/v4_adjpairs/g31b_trials.jsonl  (same schema as e4b_trials.jsonl)
  results/v4_adjpairs_analysis/g31b_relativity_ratios.json
  figures/v4_adjpairs/g31b_relativity_ratios.png

Layer budget for G31B (60 layers): mid=30, late=45.
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

MODEL_ID = "google/gemma-4-31B"
LAYER_INDICES = {"mid": 30, "late": 45}  # G31B has 60 layers
BATCH_SIZE = 4
OUT = REPO / "results" / "v4_adjpairs"
OUT_FIG = REPO / "figures" / "v4_adjpairs"
OUT_ANALYSIS = REPO / "results" / "v4_adjpairs_analysis"


def build_implicit_trials():
    trials = []
    idx = 0
    for pair in PAIRS:
        for x in pair.target_values:
            for mu in pair.mu_values:
                z = compute_z(pair, x, mu)
                for s in range(30):
                    trials.append({
                        "id": f"{pair.name}_implicit_{idx:06d}",
                        "pair": pair.name,
                        "condition": "implicit",
                        "prompt": make_implicit_prompt(pair, x, mu, s),
                        "x": float(x), "mu": float(mu), "z": float(z),
                        "sigma": pair.sigma, "seed": s,
                        "low_word": pair.low_word, "high_word": pair.high_word,
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


def extract_and_score(model, tok, trials_for_pair, high_id, low_id):
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
    prompts = [t["prompt"] for t in trials_for_pair]
    per_layer = {k: [] for k in LAYER_INDICES}
    logit_diffs = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            for k in captured: captured[k].clear()
            with torch.no_grad():
                out = model(**enc)
            last = out.logits[:, -1, :]
            ld = (last[:, high_id] - last[:, low_id]).float().cpu().numpy()
            logit_diffs.append(ld)
            for k in LAYER_INDICES:
                h = captured[k][0]
                per_layer[k].append(h[:, -1, :].float().cpu().numpy())
    finally:
        for h in handles: h.remove()
    return ({k: np.concatenate(v, axis=0) for k, v in per_layer.items()},
            np.concatenate(logit_diffs))


def relativity_ratio(trials, ld):
    """project convention: -slope_μ / slope_x"""
    x = np.array([t["x"] for t in trials])
    mu = np.array([t["mu"] for t in trials])
    X = np.column_stack([x, mu, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, ld, rcond=None)
    slope_x, slope_mu, _ = beta
    return float(-slope_mu / slope_x) if abs(slope_x) > 1e-9 else None, float(slope_x), float(slope_mu)


def main() -> None:
    trials = build_implicit_trials()
    by_pair = defaultdict(list)
    for t in trials: by_pair[t["pair"]].append(t)
    print(f"{len(trials)} implicit trials across {len(by_pair)} pairs", flush=True)

    with (OUT / "g31b_trials.jsonl").open("w") as f:
        for t in trials: f.write(json.dumps(t) + "\n")

    print(f"\nLoading {MODEL_ID}… (≈1 min, 31B bf16 sharded)", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s  dtype={model.dtype}", flush=True)

    per_pair_stats = {}
    for pair_obj in PAIRS:
        pair = pair_obj.name
        print(f"\n=== G31B × {pair} ({pair_obj.low_word}/{pair_obj.high_word}) ===", flush=True)
        t_list = by_pair[pair]
        high_id = first_token_id(tok, pair_obj.high_word)
        low_id = first_token_id(tok, pair_obj.low_word)
        t1 = time.time()
        acts, ld = extract_and_score(model, tok, t_list, high_id, low_id)
        print(f"  {len(t_list)} prompts, {time.time()-t1:.1f}s  ld_mean={ld.mean():+.3f}", flush=True)
        for layer, a in acts.items():
            np.savez(OUT / f"g31b_{pair}_implicit_{layer}.npz",
                     activations=a.astype(np.float32),
                     ids=np.array([t["id"] for t in t_list]),
                     layer_index=LAYER_INDICES[layer],
                     layer_name=layer)
        with (OUT / f"g31b_{pair}_implicit_logits.jsonl").open("w") as f:
            for t, l in zip(t_list, ld):
                f.write(json.dumps({"id": t["id"], "logit_diff": float(l)}) + "\n")
        rr, sx, smu = relativity_ratio(t_list, ld)
        per_pair_stats[pair] = {"relativity_ratio": rr, "slope_x": sx, "slope_mu": smu,
                                 "ld_mean": float(ld.mean()), "ld_std": float(ld.std())}
        print(f"  relativity_ratio = {rr:.3f}  (slope_x={sx:+.4f}, slope_μ={smu:+.4f})", flush=True)

    # Compare G31B vs E4B relativity ratios
    e4b_summary = json.load(open(OUT_ANALYSIS / "summary.json"))
    e4b_rr = {r["pair"]: r["relativity_ratio"] for r in e4b_summary if r["layer"] == "late"}

    final = {"per_pair_g31b": per_pair_stats, "e4b_late": e4b_rr, "model": MODEL_ID}
    (OUT_ANALYSIS / "g31b_relativity_ratios.json").write_text(json.dumps(final, indent=2))
    print(f"\nwrote {OUT_ANALYSIS/'g31b_relativity_ratios.json'}")

    # Plot G31B vs E4B per pair
    names = list(per_pair_stats.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [e4b_rr.get(n) for n in names], 0.4, label="E4B (late L32)")
    ax.bar(xs + 0.2, [per_pair_stats[n]["relativity_ratio"] for n in names], 0.4, label="G31B (late L45)")
    ax.axhline(1.0, ls="--", color="gray", alpha=0.4, label="perfect relativity")
    ax.axhline(0.0, ls="--", color="gray", alpha=0.4, label="perfect absolute")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("relativity_ratio = −slope_μ / slope_x")
    ax.set_title("Relativity ratio per pair — E4B vs G31B (late layers)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "g31b_vs_e4b_relativity.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'g31b_vs_e4b_relativity.png'}")


if __name__ == "__main__":
    main()
