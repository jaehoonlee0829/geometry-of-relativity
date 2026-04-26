"""v11.5 §A — domain-agnostic shared z-direction.

Build w_shared three ways and compare all three:
  1. mean: w_shared = ℓ2-normalize( Σ_p primal_z[p] )
  2. pc1:  w_shared = first PC of the 8 per-pair primal_z's stacked as rows
  3. proc: Procrustes-aligned mean (sign-aligned per pair to maximize agreement)

Then steer all 8 pairs simultaneously with α · w_shared at α ∈ {-4, 0, +4}
and compare each pair's slope to its within-pair primal_z slope.

Domain-general claim: w_shared slope ≥ 50% of within-pair slope on 8/8 pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
MODEL_BY_SHORT = {"gemma2-2b": "google/gemma-2-2b", "gemma2-9b": "google/gemma-2-9b"}
LATE_BY_SHORT = {"gemma2-2b": 20, "gemma2-9b": 33}


def primal_z(model_short: str, pair: str, layer: int) -> np.ndarray | None:
    p = REPO / "results" / "v11" / model_short / pair / f"{model_short}_{pair}_v11_residuals.npz"
    if not p.exists(): return None
    d = np.load(p)
    h = d["activations"][:, layer, :].astype(np.float64)
    z = d["z"]
    return h[z > +1.0].mean(0) - h[z < -1.0].mean(0)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def cell_seed_subset(pair: str, seed: int = 0) -> list[dict]:
    p = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    out, seen = [], set()
    for line in p.open():
        t = json.loads(line)
        if t.get("cell_seed", t.get("seed")) != seed: continue
        key = (round(t["x"], 4), round(t["z"], 4))
        if key in seen: continue
        seen.add(key); out.append(t)
    return out


def get_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr): m = getattr(m, attr)
    return m.layers


def steer(model, tok, prompts, hi_id, lo_id, direction, layer, alpha, bs, max_seq):
    if direction is not None and alpha != 0:
        d = unit(direction)
        d_t = torch.tensor(d, dtype=torch.bfloat16, device=model.device)
        layers = get_layers(model)
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h + alpha * d_t
            return (h,) + output[1:] if isinstance(output, tuple) else h
        handle = layers[layer].register_forward_hook(hook)
    else:
        handle = None
    out = np.zeros(len(prompts), dtype=np.float32)
    try:
        for b0 in range(0, len(prompts), bs):
            batch = prompts[b0:b0 + bs]
            enc = tok(batch, return_tensors="pt", padding="max_length",
                      max_length=max_seq, truncation=True).to(model.device)
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            out[b0:b0 + len(batch)] = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
    finally:
        if handle is not None: handle.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=list(MODEL_BY_SHORT.keys()))
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    L = LATE_BY_SHORT[args.model_short]
    pairs = []
    primals = []
    for p in ALL_PAIRS:
        d = primal_z(args.model_short, p, L)
        if d is not None:
            pairs.append(p); primals.append(d)
    if len(pairs) < 2:
        print("not enough pairs"); return
    P = np.stack(primals)  # (n_pairs, d_model)

    # Construct shared directions
    w_mean = unit(P.mean(0))
    w_pc1 = unit(PCA(n_components=1).fit(P).components_[0])
    # Procrustes: sign-align each pair's primal to w_mean, then re-mean
    P_aligned = np.array([p if (p @ w_mean) >= 0 else -p for p in P])
    w_proc = unit(P_aligned.mean(0))
    shared_dirs = {"mean": w_mean, "pc1": w_pc1, "proc": w_proc}

    # Cosine of shared directions with each per-pair primal_z
    cos_table: dict[str, dict[str, float]] = {}
    for kind, w in shared_dirs.items():
        cos_table[kind] = {p: float(unit(P[i]) @ w) for i, p in enumerate(pairs)}

    # Average pairwise cos among per-pair primals (the upper bound on "how shared can w_shared be")
    pairwise = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            pairwise.append(float(unit(P[i]) @ unit(P[j])))
    pairwise_mean = float(np.mean(pairwise))

    # Load model and steer
    print(f"[shared-z] {args.model_short} L{L} | mean cos(P_i, P_j)={pairwise_mean:+.3f}", flush=True)
    print(f"[shared-z] cos(w_shared, primal_z[pair]) per pair, per kind:")
    for kind in shared_dirs:
        print(f"  {kind}: " + ", ".join(f"{p}={cos_table[kind][p]:+.2f}" for p in pairs))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = MODEL_BY_SHORT[args.model_short]
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()

    # Per-pair: within-pair slope and shared-direction slope (both kinds)
    by_pair: dict[str, dict] = {}
    for i, pair in enumerate(pairs):
        sub = cell_seed_subset(pair, seed=0)
        prompts = [t["prompt"] for t in sub]
        hi = first_token_id(tok, sub[0]["high_word"])
        lo = first_token_id(tok, sub[0]["low_word"])
        t1 = time.time()

        # Within-pair (steering with own primal_z)
        ld_pos = steer(model, tok, prompts, hi, lo, P[i], L, +args.alpha,
                       args.batch_size, args.max_seq)
        ld_neg = steer(model, tok, prompts, hi, lo, P[i], L, -args.alpha,
                       args.batch_size, args.max_seq)
        within_slope = float((ld_pos - ld_neg).mean() / (2 * args.alpha))

        # w_shared (mean variant — the cleanest interpretation)
        ld_pos_s = steer(model, tok, prompts, hi, lo, w_proc, L, +args.alpha,
                         args.batch_size, args.max_seq)
        ld_neg_s = steer(model, tok, prompts, hi, lo, w_proc, L, -args.alpha,
                         args.batch_size, args.max_seq)
        shared_slope = float((ld_pos_s - ld_neg_s).mean() / (2 * args.alpha))

        ratio = shared_slope / within_slope if abs(within_slope) > 1e-12 else float("nan")
        by_pair[pair] = {
            "within_slope": within_slope,
            "shared_slope_proc": shared_slope,
            "ratio_shared_to_within": ratio,
            "n_prompts": len(prompts),
        }
        print(f"[shared-z]   {pair:11s}  within={within_slope:+.3f}  "
              f"shared(proc)={shared_slope:+.3f}  ratio={ratio:+.2f}  "
              f"({time.time() - t1:.1f}s)", flush=True)

    # Aggregate verdict: domain-general if ≥6/8 ratios > 0.5
    n_pass = sum(1 for v in by_pair.values()
                 if v["ratio_shared_to_within"] is not None and v["ratio_shared_to_within"] > 0.5)
    domain_general_8of8 = n_pass == len(pairs)
    domain_general_6of8 = n_pass >= 6

    out = {
        "model_short": args.model_short,
        "layer": L,
        "alpha": args.alpha,
        "pairs": pairs,
        "pairwise_primal_cos_mean": pairwise_mean,
        "cos_w_shared_vs_primal_per_pair": cos_table,
        "by_pair_steering": by_pair,
        "n_pass_ratio_gt_0p5": n_pass,
        "domain_general_8of8": domain_general_8of8,
        "domain_general_6of8": domain_general_6of8,
    }
    out_dir = REPO / "results" / "v11_5" / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shared_z_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\n[shared-z] {n_pass}/{len(pairs)} pairs pass ratio>0.5  "
          f"(domain-general 8/8={domain_general_8of8}, 6/8={domain_general_6of8})")
    print(f"[shared-z] wrote {out_dir / 'shared_z_analysis.json'}")


if __name__ == "__main__":
    main()
