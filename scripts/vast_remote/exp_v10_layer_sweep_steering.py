"""v10 P5 — layer-sweep steering on the dense height grid (GPU).

For each layer L ∈ [0, 25] and each α ∈ {-8, -4, 0, +4, +8}, add α·d_L to the
residual at L (during forward) and measure how logit_diff(tall - short)
changes vs the unsteered baseline.

Two direction families:
  primal_z  = mean activation where z > +1  −  mean where z < −1   (per layer)
  probe_z   = closed-form ridge probe direction for z              (per layer)

Both are normalized to unit ℓ2.

To keep wall time under ~5 min we steer the 400 seed-0 prompts (one per
(x, z) cell). This matches v10's cell-mean granularity and is sufficient
for the layer-sweep statistics. Per-cell measurements then give a
mean-and-std curve per (direction, α, layer).

Inputs:
  results/v10/gemma2_height_v10_residuals.npz
  data_gen/v10_dense_height_trials.jsonl
Outputs:
  results/v10/steering_layer_sweep.npz
  figures/v10/steering_layer_sweep.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402

MODEL_ID = "google/gemma-2-2b"
TRIALS_PATH = REPO / "data_gen" / "v10_dense_height_trials.jsonl"
RES = REPO / "results" / "v10"
FIG = REPO / "figures" / "v10"
FIG.mkdir(parents=True, exist_ok=True)

ALPHAS = [-8.0, -4.0, 0.0, 4.0, 8.0]
BATCH = 16
MAX_SEQ = 192


def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def closed_form_ridge(X: np.ndarray, y: np.ndarray, lam: float = 100.0) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    yc = y - y.mean()
    A = Xc.T @ Xc
    A.flat[::A.shape[0] + 1] += lam
    return np.linalg.solve(A, Xc.T @ yc).astype(np.float32)


def main():
    t_start = time.time()
    print("[P5] loading residuals...", flush=True)
    res = np.load(RES / "gemma2_height_v10_residuals.npz")
    acts = res["activations"].astype(np.float32)        # (4000, 26, 2304)
    z = res["z"].astype(np.float32)
    x = res["x"].astype(np.float32)
    ids = res["ids"]
    seeds = res["seed"]
    n_layers = acts.shape[1]
    print(f"[P5]   acts {acts.shape}", flush=True)

    # --- compute primal_z per layer (mean(z>+1) - mean(z<-1)), L2-normalized
    # --- compute probe_z per layer (closed-form ridge), L2-normalized
    print("[P5] computing per-layer steering directions...", flush=True)
    primal = np.zeros((n_layers, acts.shape[2]), dtype=np.float32)
    probe = np.zeros_like(primal)
    pos = z > 1.0
    neg = z < -1.0
    print(f"[P5]   z > +1: n={pos.sum()},  z < -1: n={neg.sum()}", flush=True)
    for L in range(n_layers):
        d = acts[pos, L].mean(0) - acts[neg, L].mean(0)
        primal[L] = d / (np.linalg.norm(d) + 1e-12)
        w = closed_form_ridge(acts[:, L], z, lam=100.0)
        probe[L] = w / (np.linalg.norm(w) + 1e-12)
    print(f"[P5]   primal/probe shapes: {primal.shape}", flush=True)

    # --- subsample to seed=0 (one prompt per (x, z) cell)
    sub_mask = seeds == 0
    sub_idx = np.where(sub_mask)[0]
    print(f"[P5] steering subsample: seed=0 → {len(sub_idx)} prompts", flush=True)

    # --- load trials
    all_trials = [json.loads(l) for l in TRIALS_PATH.open()]
    sub_trials = [all_trials[i] for i in sub_idx]
    sub_z = z[sub_idx]
    sub_x = x[sub_idx]

    # --- model
    print(f"[P5] loading {MODEL_ID}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    print(f"[P5]   loaded in {time.time() - t0:.1f}s", flush=True)
    layers = get_decoder_layers(model)

    tall_id = first_token_id(tok, "tall")
    short_id = first_token_id(tok, "short")

    # --- helper: run forward over sub_trials with optional steering
    def run_forward(L_steer: int | None, direction: np.ndarray | None,
                    alpha: float) -> np.ndarray:
        """Return logit_diff(tall - short) per prompt."""
        out = np.zeros(len(sub_trials), dtype=np.float32)
        handle = None
        if L_steer is not None and direction is not None and alpha != 0.0:
            d_t = torch.tensor(direction, dtype=torch.bfloat16,
                               device=model.device)
            scale = float(alpha)

            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                # Add α·d to every position in the residual at this layer.
                h_steered = h + scale * d_t
                if isinstance(output, tuple):
                    return (h_steered,) + output[1:]
                return h_steered
            handle = layers[L_steer].register_forward_hook(hook)
        try:
            for i in range(0, len(sub_trials), BATCH):
                batch = [t["prompt"] for t in sub_trials[i:i + BATCH]]
                enc = tok(batch, return_tensors="pt", padding="max_length",
                          max_length=MAX_SEQ, truncation=True).to(model.device)
                with torch.no_grad():
                    o = model(**enc, use_cache=False, output_attentions=False)
                last = o.logits[:, -1, :].float()
                ld = (last[:, tall_id] - last[:, short_id]).cpu().numpy()
                out[i:i + len(batch)] = ld
        finally:
            if handle is not None:
                handle.remove()
        return out

    # --- baseline (no steering)
    print("[P5] running baseline (α=0)...", flush=True)
    baseline = run_forward(None, None, 0.0)
    print(f"[P5]   baseline corr(ld, z) = "
          f"{np.corrcoef(baseline, sub_z)[0, 1]:.3f}", flush=True)

    # --- layer × direction × alpha sweep
    print("[P5] sweeping all 26 layers × 2 directions × "
          f"{len(ALPHAS) - 1} non-zero alphas ...", flush=True)
    n_a = len(ALPHAS)
    sweep = np.zeros((n_layers, 2, n_a, len(sub_trials)), dtype=np.float32)
    direction_names = ["primal_z", "probe_z"]
    direction_arrs = [primal, probe]
    for L in range(n_layers):
        t_L = time.time()
        for di, dname in enumerate(direction_names):
            for ai, a in enumerate(ALPHAS):
                if a == 0.0:
                    sweep[L, di, ai] = baseline
                else:
                    sweep[L, di, ai] = run_forward(L, direction_arrs[di][L], a)
        print(f"[P5]  L{L:2d}  done in {time.time() - t_L:.1f}s", flush=True)

    # --- aggregate: mean Δlogit_diff per (L, dir, α)
    # Slope of mean(Δlogit_diff) vs α gives steering strength.
    mean_ld = sweep.mean(-1)                        # (L, 2, A)
    delta = mean_ld - mean_ld[:, :, ALPHAS.index(0.0)][..., None]   # baseline subtracted
    # Slope via least-squares against alpha
    A_arr = np.array(ALPHAS, dtype=np.float32)
    slope = np.zeros((n_layers, 2), dtype=np.float32)
    for L in range(n_layers):
        for di in range(2):
            slope[L, di], _ = np.polyfit(A_arr, delta[L, di], 1)

    # --- save
    np.savez(RES / "steering_layer_sweep.npz",
             alphas=A_arr,
             sub_idx=sub_idx,
             baseline=baseline,
             sweep=sweep,
             mean_ld=mean_ld, delta=delta, slope=slope,
             primal_dirs=primal, probe_dirs=probe,
             tall_id=np.int32(tall_id), short_id=np.int32(short_id))
    print(f"[P5] wrote {RES}/steering_layer_sweep.npz "
          f"({(RES / 'steering_layer_sweep.npz').stat().st_size / 1e6:.1f} MB)",
          flush=True)

    # --- figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].plot(np.arange(n_layers), slope[:, 0], "o-", color="C0", lw=2,
                 label="primal_z")
    axes[0].plot(np.arange(n_layers), slope[:, 1], "s--", color="C3", lw=2,
                 label="probe_z")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_xlabel("layer where steering is injected")
    axes[0].set_ylabel("steering slope (Δ logit_diff per unit α)")
    axes[0].set_title("Layer-sweep steering strength")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Strategic α-sweep at L7 / L13 / L20
    for li, L in enumerate([7, 13, 20]):
        axes[1].plot(A_arr, delta[L, 0], "o-",
                     label=f"primal_z @ L{L}", color=f"C{li}")
        axes[1].plot(A_arr, delta[L, 1], "s--",
                     label=f"probe_z  @ L{L}", color=f"C{li}", alpha=0.5)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("α (steering magnitude)")
    axes[1].set_ylabel("Δ logit_diff (mean over 400 cells)")
    axes[1].set_title("α-sweep at strategic layers")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "steering_layer_sweep.png", dpi=120)
    plt.close()
    print(f"[P5] wrote {FIG}/steering_layer_sweep.png", flush=True)
    print(f"[P5] TOTAL: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
