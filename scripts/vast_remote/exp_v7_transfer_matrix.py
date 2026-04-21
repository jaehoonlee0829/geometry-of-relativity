"""v7 Priority 4: clean cross-pair transfer matrix with random-direction null.

Uses primal_z_clean (from Grid B) for transfer — the strongest direction.
Adds 3 random unit vectors as null.

For each (pair_A direction source) × (pair_B evaluation): steer pair_B
prompts with pair_A's primal_z_clean at α=±4, measure slope.

Writes:
  results/v7_steering/clean_transfer_matrix.json
  figures/v7/clean_transfer_heatmap.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS as PAIR_OBJS, compute_z, make_implicit_prompt  # noqa: E402

MODEL_ID = "google/gemma-4-E4B"
LAYER_IDX = 32
N_PROMPTS = 60
BATCH_SIZE = 16
N_RANDOM = 3

V7 = REPO / "results" / "v7_xz_grid"
OUT_JSON = REPO / "results" / "v7_steering"
OUT_FIG = REPO / "figures" / "v7"

PAIR_NAMES = [p.name for p in PAIR_OBJS]
D = 2560


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def compute_primal_z_clean(pair_name: str):
    trials = {json.loads(l)["id"]: json.loads(l) for l in (V7 / "e4b_trials.jsonl").open()}
    npz = np.load(V7 / f"e4b_{pair_name}_late.npz", allow_pickle=True)
    acts = npz["activations"].astype(np.float64)
    ids = [str(s) for s in npz["ids"]]
    zs = np.array([trials[i]["z"] for i in ids])
    return unit(acts[zs > +1.0].mean(0) - acts[zs < -1.0].mean(0))


def subsample_prompts(pair_obj, n, rng):
    all_trials = []
    for x in pair_obj.target_values:
        for mu in pair_obj.mu_values:
            z = compute_z(pair_obj, x, mu)
            for s in range(30):
                all_trials.append({
                    "prompt": make_implicit_prompt(pair_obj, x, mu, s),
                    "x": x, "mu": mu, "z": z, "seed": s,
                })
    picks = rng.choice(len(all_trials), size=n, replace=False)
    return [all_trials[i] for i in picks]


def steered_ld(model, tok, prompts, direction, alpha, high_id, low_id):
    layers = get_layers(model)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = direction.to(device=h.device, dtype=h.dtype)
        h[:, -1, :] = h[:, -1, :] + alpha * d
        if isinstance(out, tuple): return (h,) + out[1:]
        return h

    handle = layers[LAYER_IDX].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lds = []
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            ld = (logits[:, high_id] - logits[:, low_id]).float().cpu().numpy()
            lds.append(ld)
    finally:
        handle.remove()
    return np.concatenate(lds)


def main():
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Compute primal_z_clean per pair
    dirs = {n: compute_primal_z_clean(n) for n in PAIR_NAMES}

    # Random null directions
    rng_dirs = np.random.default_rng(42)
    random_dirs = [unit(rng_dirs.standard_normal(D).astype(np.float64)) for _ in range(N_RANDOM)]

    # Prompts per pair
    rng_p = np.random.default_rng(0)
    prompts_by = {po.name: subsample_prompts(po, N_PROMPTS, rng_p) for po in PAIR_OBJS}
    tokenset = {po.name: (first_token_id(tok, po.high_word),
                           first_token_id(tok, po.low_word)) for po in PAIR_OBJS}

    # Transfer matrix: rows = direction source, cols = eval pair
    T = np.zeros((len(PAIR_NAMES), len(PAIR_NAMES)))
    # Random null matrix (same shape, using each random dir then averaging)
    R = np.zeros((len(PAIR_NAMES), N_RANDOM))   # one per eval pair, per random dir

    print("\n=== clean primal_z transfer matrix ===")
    for i, p_src in enumerate(PAIR_NAMES):
        print(f"\nfrom {p_src}", flush=True)
        direction = torch.from_numpy(dirs[p_src]).to(model.device)
        for j, p_tgt in enumerate(PAIR_NAMES):
            prompts = [t["prompt"] for t in prompts_by[p_tgt]]
            high_id, low_id = tokenset[p_tgt]
            ld_m4 = steered_ld(model, tok, prompts, direction, -4.0, high_id, low_id)
            ld_p4 = steered_ld(model, tok, prompts, direction, +4.0, high_id, low_id)
            slope = (ld_p4.mean() - ld_m4.mean()) / 8.0
            T[i, j] = slope
            print(f"  → {p_tgt:12s}: slope = {slope:+.4f}", end="", flush=True)
        print()

    # Random null: for each eval pair, steer with 3 random dirs
    print("\n=== random-direction null ===")
    for ri, vd in enumerate(random_dirs):
        direction = torch.from_numpy(vd).to(model.device)
        print(f"random #{ri}")
        for j, p_tgt in enumerate(PAIR_NAMES):
            prompts = [t["prompt"] for t in prompts_by[p_tgt]]
            high_id, low_id = tokenset[p_tgt]
            ld_m4 = steered_ld(model, tok, prompts, direction, -4.0, high_id, low_id)
            ld_p4 = steered_ld(model, tok, prompts, direction, +4.0, high_id, low_id)
            slope = (ld_p4.mean() - ld_m4.mean()) / 8.0
            R[j, ri] = slope

    # Summary
    diag_mean = float(np.mean(np.abs(np.diagonal(T))))
    offdiag_mean = float(np.mean(np.abs(T[~np.eye(len(PAIR_NAMES), dtype=bool)])))
    random_mean = float(np.mean(np.abs(R)))
    print(f"\nSUMMARY:")
    print(f"  diagonal |slope|     mean = {diag_mean:.4f}   (primal_z on own pair)")
    print(f"  off-diag |slope|     mean = {offdiag_mean:.4f}   (primal_z of A on pair B)")
    print(f"  random null |slope|  mean = {random_mean:.4f}   (random unit vector)")
    print(f"  transfer ratio: off-diag / diagonal = {offdiag_mean/diag_mean:.2f}")
    print(f"  signal/null: off-diag / random = {offdiag_mean/random_mean:.2f}")

    result = {
        "direction_used": "primal_z_clean (Grid B)",
        "alpha_range": 8,
        "pair_names": PAIR_NAMES,
        "transfer_matrix_slope_per_alpha_unit": T.tolist(),
        "random_null_slopes_per_pair_per_seed": R.tolist(),
        "summary": {
            "diagonal_mean_abs": diag_mean,
            "offdiag_mean_abs": offdiag_mean,
            "random_null_mean_abs": random_mean,
            "transfer_ratio": offdiag_mean/diag_mean,
            "signal_to_null_ratio": offdiag_mean/random_mean,
        },
    }
    (OUT_JSON / "clean_transfer_matrix.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON/'clean_transfer_matrix.json'}")

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    vmax = max(abs(T.min()), abs(T.max()))
    im = axes[0].imshow(T, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_xticks(range(len(PAIR_NAMES))); axes[0].set_xticklabels(PAIR_NAMES, rotation=30, ha="right")
    axes[0].set_yticks(range(len(PAIR_NAMES))); axes[0].set_yticklabels(PAIR_NAMES)
    axes[0].set_xlabel("evaluation pair (B)"); axes[0].set_ylabel("direction source (A)")
    axes[0].set_title("primal_z_clean transfer matrix  (slope per α-unit)")
    for i in range(len(PAIR_NAMES)):
        for j in range(len(PAIR_NAMES)):
            axes[0].text(j, i, f"{T[i,j]:+.3f}", ha="center", va="center",
                          color="white" if abs(T[i,j]) > vmax*0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=axes[0], fraction=0.04)

    # Random null comparison
    xpos = np.arange(len(PAIR_NAMES))
    rand_means = np.abs(R).mean(axis=1)
    rand_stds = np.abs(R).std(axis=1)
    primal_own = np.abs(np.diagonal(T))
    axes[1].bar(xpos - 0.2, primal_own, 0.4, label="|primal_z| on own pair")
    axes[1].bar(xpos + 0.2, rand_means, 0.4, yerr=rand_stds, capsize=3, label="|random unit vec| (mean±std of 3)")
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(PAIR_NAMES, rotation=30, ha="right")
    axes[1].set_ylabel("|slope per α-unit|")
    axes[1].set_title("primal_z_clean vs random null")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "clean_transfer_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'clean_transfer_heatmap.png'}")


if __name__ == "__main__":
    main()
