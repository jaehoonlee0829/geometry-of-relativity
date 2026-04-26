"""v11 P3e — cross-pair transfer steering (GPU).

For each (source_pair, target_pair) with source != target:
  1. Compute primal_z from source_pair activations at canonical late layer
     (L=20 for 2B, L=33 for 9B).
  2. Steer target_pair seed=0 prompts (one per cell) at α ∈ {-4, 0, +4}.
  3. Measure Δlogit_diff slope.

Stores 8×8 transfer matrix per model. Diagonal = within-pair (gold standard).

Inputs:
  results/v11/<model_short>/<pair>/<base>_residuals.npz   (for each pair)
  data_gen/v11_<pair>_trials.jsonl                        (for each pair)
Outputs:
  results/v11/<model_short>/cross_pair_transfer_dense.json
  figures/v11/steering/cross_pair_transfer_8x8_<model_short>.png
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
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
MODEL_BY_SHORT = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
}
LATE_BY_SHORT = {"gemma2-2b": 20, "gemma2-9b": 33}


def primal_z_at(model_short: str, pair: str, layer: int) -> np.ndarray | None:
    p = (REPO / "results" / "v11" / model_short / pair /
         f"{model_short}_{pair}_v11_residuals.npz")
    if not p.exists():
        return None
    d = np.load(p)
    h = d["activations"][:, layer, :].astype(np.float64)
    z = d["z"]
    return h[z > +1.0].mean(0) - h[z < -1.0].mean(0)


def get_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr): m = getattr(m, attr)
    if hasattr(m, "layers"): return m.layers
    raise AttributeError("decoder layers not found")


def cell_seed0_subset(pair: str) -> list[dict]:
    path = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    out = []
    seen = set()
    for line in path.open():
        t = json.loads(line)
        if t.get("cell_seed", t.get("seed")) != 0 and "cell_seed" in t and t["cell_seed"] != 0:
            continue
        if t.get("cell_seed") == 0:
            key = (round(t["x"], 4), round(t["z"], 4))
            if key in seen: continue
            seen.add(key)
            out.append(t)
        elif "cell_seed" not in t and t["seed"] == 0:
            key = (round(t["x"], 4), round(t["z"], 4))
            if key in seen: continue
            seen.add(key)
            out.append(t)
    return out


def steer_run(model, tok, prompts: list[str], hi_id: int, lo_id: int,
              direction: np.ndarray | None, layer: int, alpha: float,
              batch_size: int, max_seq: int) -> np.ndarray:
    """Return per-prompt logit_diff after adding alpha * direction at `layer`."""
    if direction is not None:
        d_unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
        d_t = torch.tensor(d_unit, dtype=torch.bfloat16, device=model.device)
    else:
        d_t = None

    layers = get_layers(model)
    handle = None
    if d_t is not None and alpha != 0:
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h + alpha * d_t  # broadcast over (bsz, seq)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        handle = layers[layer].register_forward_hook(hook)

    out = np.zeros(len(prompts), dtype=np.float32)
    try:
        for b0 in range(0, len(prompts), batch_size):
            batch = prompts[b0:b0 + batch_size]
            enc = tok(batch, return_tensors="pt",
                      padding="max_length", max_length=max_seq,
                      truncation=True).to(model.device)
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            ld = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
            out[b0:b0 + len(batch)] = ld
    finally:
        if handle is not None: handle.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=list(MODEL_BY_SHORT.keys()))
    ap.add_argument("--alpha", type=float, default=4.0,
                    help="steering magnitude (we run -alpha, 0, +alpha)")
    ap.add_argument("--max-seq", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    model_short = args.model_short
    model_id = MODEL_BY_SHORT[model_short]
    layer = LATE_BY_SHORT[model_short]

    # Pre-compute primal_z for all available pairs
    primal: dict[str, np.ndarray] = {}
    for p in ALL_PAIRS:
        d = primal_z_at(model_short, p, layer)
        if d is not None: primal[p] = d
    available = sorted(primal.keys())
    print(f"[transfer] {model_short}: primal_z computed at L{layer} for {len(available)} pairs",
          flush=True)
    if len(available) < 2:
        print("[transfer] insufficient pairs — abort"); return

    # Load model
    print(f"[transfer] loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()

    # For each TARGET pair, steer with each SOURCE pair's direction
    transfer_slope = {tgt: {src: float("nan") for src in available} for tgt in available}
    baseline_slope = {tgt: float("nan") for tgt in available}

    for target in available:
        target_subset = cell_seed0_subset(target)
        if not target_subset:
            print(f"[transfer] {target}: no seed-0 subset — skip"); continue
        prompts = [t["prompt"] for t in target_subset]
        z_vec = np.array([t["z"] for t in target_subset], dtype=np.float64)
        hi_id = first_token_id(tok, target_subset[0]["high_word"])
        lo_id = first_token_id(tok, target_subset[0]["low_word"])

        t1 = time.time()
        # baseline (alpha=0)
        ld0 = steer_run(model, tok, prompts, hi_id, lo_id, None, layer, 0.0,
                        args.batch_size, args.max_seq)
        # within-pair gold (steer with target's own direction)
        ld_pos = steer_run(model, tok, prompts, hi_id, lo_id, primal[target], layer,
                           +args.alpha, args.batch_size, args.max_seq)
        ld_neg = steer_run(model, tok, prompts, hi_id, lo_id, primal[target], layer,
                           -args.alpha, args.batch_size, args.max_seq)
        baseline_slope[target] = float((ld_pos - ld_neg).mean() / (2 * args.alpha))
        transfer_slope[target][target] = baseline_slope[target]

        for src in available:
            if src == target: continue
            ld_pos = steer_run(model, tok, prompts, hi_id, lo_id, primal[src], layer,
                               +args.alpha, args.batch_size, args.max_seq)
            ld_neg = steer_run(model, tok, prompts, hi_id, lo_id, primal[src], layer,
                               -args.alpha, args.batch_size, args.max_seq)
            transfer_slope[target][src] = float((ld_pos - ld_neg).mean() / (2 * args.alpha))

        print(f"[transfer] target={target} (n={len(prompts)}): "
              f"within-slope={baseline_slope[target]:+.3f}  "
              f"avg-cross={np.nanmean([v for k, v in transfer_slope[target].items() if k != target]):+.3f}  "
              f"({time.time() - t1:.1f}s)", flush=True)

    out_path = REPO / "results" / "v11" / model_short / "cross_pair_transfer_dense.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model_short": model_short,
        "layer": layer,
        "alpha": args.alpha,
        "pairs": available,
        "transfer_slope_target_by_source": transfer_slope,
        "within_pair_slope": baseline_slope,
    }, indent=2))
    print(f"\nwrote {out_path.relative_to(REPO)}")

    # 8x8 heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(available)
    M = np.full((n, n), np.nan)
    for i, tgt in enumerate(available):
        for j, src in enumerate(available):
            M[i, j] = transfer_slope[tgt].get(src, np.nan)
    fig_dir = REPO / "figures" / "v11" / "steering"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 6))
    im = plt.imshow(M, cmap="RdBu_r", vmin=-abs(np.nanmax(np.abs(M))), vmax=abs(np.nanmax(np.abs(M))))
    plt.xticks(range(n), available, rotation=45, ha="right")
    plt.yticks(range(n), available)
    plt.xlabel("steering source pair")
    plt.ylabel("target pair (steered)")
    plt.title(f"{model_short}  cross-pair transfer @ L{layer}  α=±{args.alpha}")
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                plt.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                         color="black", fontsize=7)
    plt.colorbar(im, label="Δlogit_diff slope")
    plt.tight_layout()
    plt.savefig(fig_dir / f"cross_pair_transfer_8x8_{model_short}.png", dpi=110)
    plt.close()
    print(f"wrote {fig_dir / f'cross_pair_transfer_8x8_{model_short}.png'}")


if __name__ == "__main__":
    main()
