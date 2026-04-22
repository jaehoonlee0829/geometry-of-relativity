"""v9 §13.5: per-layer causal steering.

For a subset of strategic layers, compute primal_z and probe_z at that
layer (from the saved `gemma2_{pair}_alllayers.npz`) and run causal
steering by hooking the same layer during a new forward pass.

Alphas: {-2, -1, 0, +1, +2}.
Subset: 20 prompts per z-value × 5 z-values = up to 100/pair.
Layers: [5, 10, 13, 17, 20, 22, 24] by default (configurable via env).

Outputs
  results/v9_gemma2/layer_sweep_steering.json
  figures/v9/layer_sweep_steering_slopes.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS  # noqa: E402
from extract_v7_xz_grid import Z_VALUES  # noqa: E402
from exp_v9_on_manifold_steering import BATCH_SIZE, get_layers  # noqa: E402

MODEL_ID = "google/gemma-2-2b"
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
DEFAULT_LAYERS = [5, 10, 13, 17, 20, 22, 24]
SUBSET_PER_Z = 20

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"


def run_steer_at_layer(model, tok, layer_idx, prompts, dir_per_row, alpha,
                       high_id, low_id, batch_size=BATCH_SIZE):
    layers = get_layers(model)
    steer_tensor = torch.tensor(dir_per_row, dtype=torch.bfloat16,
                                device=model.device)
    batch_slice = [0]

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        n = h.shape[0]
        delta = steer_tensor[batch_slice[0]:batch_slice[0] + n] * alpha
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + delta
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    handle = layers[layer_idx].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lds = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch_slice[0] = i
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model(**enc)
            last = out.logits[:, -1, :]
            ld = (last[:, high_id] - last[:, low_id]).float().cpu().numpy()
            lds.append(ld)
    finally:
        handle.remove()
    return np.concatenate(lds)


def main():
    layers_override = os.environ.get("V9_LAYERS")
    layers_to_steer = (json.loads(layers_override) if layers_override
                       else DEFAULT_LAYERS)
    print(f"Steering at layers: {layers_to_steer}")

    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    summary = {
        "model": MODEL_ID,
        "layers": layers_to_steer,
        "alphas": ALPHAS,
        "subset_per_z": SUBSET_PER_Z,
        "per_pair": {},
    }
    rng = np.random.default_rng(0)

    for pair in PAIRS:
        print(f"\n==================== {pair.name} ====================", flush=True)
        # Load all-layer activations + trials
        with np.load(RES_DIR / f"gemma2_{pair.name}_alllayers.npz",
                     allow_pickle=True) as z_:
            acts_all = z_["activations"].astype(np.float32)  # (n, L, d)
            ids = z_["ids"].tolist()
        trials = {}
        with (RES_DIR / "gemma2_trials.jsonl").open() as f:
            for line in f:
                t = json.loads(line)
                if t["pair"] == pair.name:
                    trials[t["id"]] = t
        zs = np.array([trials[i]["z"] for i in ids], dtype=np.float64)
        rows = [trials[i] for i in ids]

        # Stratified subset by z
        chosen = []
        for z_val in Z_VALUES:
            idx = np.where(np.isclose(zs, z_val, atol=1e-6))[0]
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, size=min(SUBSET_PER_Z, len(idx)), replace=False)
            chosen.extend(pick.tolist())
        chosen = np.array(sorted(chosen), dtype=int)
        sub_rows = [rows[i] for i in chosen]
        sub_prompts = [r["prompt"] for r in sub_rows]
        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)

        pair_result = {}
        for layer in layers_to_steer:
            acts_L = acts_all[:, layer, :]
            hi = zs > 0
            lo = zs < 0
            primal_z = (acts_L[hi].mean(0) - acts_L[lo].mean(0)).astype(np.float32)
            rid = Ridge(alpha=1.0, fit_intercept=True).fit(acts_L, zs)
            probe_z = rid.coef_.astype(np.float32)
            norm_p = float(np.linalg.norm(primal_z))
            probe_z_n = probe_z * (norm_p /
                                   max(np.linalg.norm(probe_z), 1e-9))

            dir_primal = np.tile(primal_z, (len(sub_rows), 1)).astype(np.float32)
            dir_probe = np.tile(probe_z_n, (len(sub_rows), 1)).astype(np.float32)

            layer_result = {"norm_primal": norm_p}
            for dir_name, dir_arr in [("primal", dir_primal), ("probe", dir_probe)]:
                means = {}
                for alpha in ALPHAS:
                    t1 = time.time()
                    ld = run_steer_at_layer(model, tok, layer, sub_prompts,
                                             dir_arr, alpha, high_id, low_id)
                    means[alpha] = float(ld.mean())
                    print(f"  L{layer:02d} {dir_name:7s} α={alpha:+.1f}  "
                          f"ld={ld.mean():+.3f}  ({time.time() - t1:.1f}s)",
                          flush=True)
                xs = np.array(ALPHAS)
                ys = np.array([means[a] for a in ALPHAS])
                slope = float(np.polyfit(xs, ys, 1)[0])
                layer_result[f"{dir_name}_slope"] = slope
                layer_result[f"{dir_name}_means_per_alpha"] = means
            pair_result[f"layer_{layer}"] = layer_result
            print(f"  → L{layer:02d}: primal={layer_result['primal_slope']:+.3f}  "
                  f"probe={layer_result['probe_slope']:+.3f}",
                  flush=True)
        summary["per_pair"][pair.name] = pair_result

    (RES_DIR / "layer_sweep_steering.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nWrote {RES_DIR}/layer_sweep_steering.json")

    # Plot
    pairs = [p.name for p in PAIRS]
    fig, (ax1, ax2) = plt_for_pairs(summary, layers_to_steer, pairs)
    fig.savefig(FIG_DIR / "layer_sweep_steering_slopes.png", dpi=140)
    print(f"  wrote {FIG_DIR}/layer_sweep_steering_slopes.png")


def plt_for_pairs(summary, layers_to_steer, pairs):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for name in pairs:
        p = summary["per_pair"][name]
        ys_primal = [p[f"layer_{L}"]["primal_slope"] for L in layers_to_steer]
        ys_probe = [p[f"layer_{L}"]["probe_slope"]  for L in layers_to_steer]
        ax1.plot(layers_to_steer, ys_primal, "-o", label=name, ms=4)
        ax2.plot(layers_to_steer, ys_probe, "-o", label=name, ms=4)
    ax1.set_title("primal_z steering slope by layer", fontsize=11)
    ax2.set_title("probe_z (Ridge) steering slope by layer", fontsize=11)
    for a in (ax1, ax2):
        a.set_xlabel("layer"); a.set_ylabel("Δlogit_diff per α")
        a.legend(fontsize=7, ncol=2); a.grid(alpha=0.3)
        a.axhline(0, color="k", lw=0.3)
    fig.suptitle("v9 §13.5 — which layer is most causally potent? "
                 "(direction refit per layer)", fontsize=12)
    fig.tight_layout()
    return fig, (ax1, ax2)


if __name__ == "__main__":
    main()
