"""Exp 2: steer with meta-direction w₁ across all 8 adjective pairs.

w₁ = top right singular vector of the stacked per-pair PC1 matrix (late layer).
If w₁ is a genuine "relativity knob", adding α·w₁ to the residual stream at
layer 32 should uniformly shift logit_diff(high − low) for every pair.

Output: results/v4_steering/meta_w1_steering.json
        figures/v4_adjpairs/meta_w1_steering_curves.png
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS, compute_z, make_implicit_prompt  # noqa: E402

REPO = SCRIPT_DIR.parent.parent
ADJPAIRS = REPO / "results" / "v4_adjpairs"
OUT_JSON = REPO / "results" / "v4_steering"
OUT_FIG = REPO / "figures" / "v4_adjpairs"
OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

MODEL_ID = "google/gemma-4-E4B"
LAYER_IDX = 32  # "late" layer for E4B
ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
N_PROMPTS_PER_PAIR = 100  # subsample from 750 implicit trials per pair
BATCH_SIZE = 16


def compute_meta_w1() -> tuple[np.ndarray, dict]:
    """Replicate meta_z_direction.py's SVD and return w1 plus diagnostics."""
    pc1s = []
    names = []
    for pair in [p.name for p in PAIRS]:
        trials_by_id = {json.loads(l)["id"]: json.loads(l) for l in (ADJPAIRS / "e4b_trials.jsonl").open()}
        npz = np.load(ADJPAIRS / f"e4b_{pair}_implicit_late.npz", allow_pickle=True)
        acts = npz["activations"].astype(np.float64)
        ids = [str(s) for s in npz["ids"]]
        zs = np.array([trials_by_id[i]["z"] for i in ids])
        # Cell-mean: group by (x, mu)
        by_cell: dict = defaultdict(list)
        for a, i in zip(acts, ids):
            t = trials_by_id[i]
            by_cell[(t["x"], t["mu"], t["z"])].append(a)
        cell_acts, cell_zs = [], []
        for (x, mu, z), aa in by_cell.items():
            cell_acts.append(np.mean(aa, axis=0))
            cell_zs.append(z)
        cell_acts = np.array(cell_acts)
        cell_zs = np.array(cell_zs)
        centered = cell_acts - cell_acts.mean(0)
        pc1 = PCA(n_components=1).fit(centered).components_[0]
        proj = centered @ pc1
        if np.corrcoef(proj, cell_zs)[0, 1] < 0:
            pc1 = -pc1
        pc1s.append(pc1)
        names.append(pair)
    V = np.stack(pc1s)  # (n_pairs, d)
    U, S, Wt = np.linalg.svd(V, full_matrices=False)
    w1 = Wt[0]
    w1 = w1 / np.linalg.norm(w1)
    return w1, {"pairs": names, "singular_values": S.tolist(),
                "variance_ratio": (S**2 / (S**2).sum()).tolist()}


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("layers not found")


def subsample_pair_prompts(pair_obj, n: int, rng: np.random.Generator) -> list[dict]:
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


def steered_logit_diff(model, tok, prompts: list[str], layer_idx: int,
                       direction: torch.Tensor, alpha: float,
                       high_id: int, low_id: int) -> np.ndarray:
    layers = get_layers(model)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # device_map=auto can place layer 32 on a different GPU than where `direction`
        # was created; move to match h at hook time (no-op if already aligned).
        d = direction.to(device=h.device, dtype=h.dtype)
        h[:, -1, :] = h[:, -1, :] + alpha * d
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = layers[layer_idx].register_forward_hook(hook)
    out_ld = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits  # (B, T, V)
            last = logits[:, -1, :]
            ld = (last[:, high_id] - last[:, low_id]).float().cpu().numpy()
            out_ld.append(ld)
    finally:
        handle.remove()
    return np.concatenate(out_ld)


def main():
    print("Computing meta-direction w₁ from cached activations…", flush=True)
    w1_np, diag = compute_meta_w1()
    print(f"  w1 shape={w1_np.shape}  top σ={diag['singular_values'][0]:.3f}  "
          f"variance_ratio[0]={diag['variance_ratio'][0]*100:.1f}%", flush=True)

    # Persist w1 as .npy for reuse
    np.save(OUT_JSON / "meta_w1.npy", w1_np)

    print(f"\nLoading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    direction = torch.from_numpy(w1_np).to(model.device)

    rng = np.random.default_rng(0)
    results: dict[str, dict] = {}
    for pair_obj in PAIRS:
        print(f"\n=== {pair_obj.name} ({pair_obj.low_word}/{pair_obj.high_word}) ===", flush=True)
        trials = subsample_pair_prompts(pair_obj, N_PROMPTS_PER_PAIR, rng)
        prompts = [t["prompt"] for t in trials]
        low_id = first_token_id(tok, pair_obj.low_word)
        high_id = first_token_id(tok, pair_obj.high_word)
        pair_curve: dict = {}
        for alpha in ALPHAS:
            t1 = time.time()
            ld = steered_logit_diff(model, tok, prompts, LAYER_IDX, direction, alpha, high_id, low_id)
            pair_curve[str(alpha)] = {
                "logit_diff_mean": float(ld.mean()),
                "logit_diff_std": float(ld.std()),
                "n": int(len(ld)),
            }
            print(f"  α={alpha:+.1f}  ld={ld.mean():+.3f}±{ld.std():.3f}  ({time.time()-t1:.1f}s)", flush=True)
        results[pair_obj.name] = {
            "low_word": pair_obj.low_word,
            "high_word": pair_obj.high_word,
            "curve": pair_curve,
        }

    # Persist
    out = {
        "layer_idx": LAYER_IDX,
        "alphas": ALPHAS,
        "meta_w1_diag": diag,
        "n_prompts_per_pair": N_PROMPTS_PER_PAIR,
        "per_pair": results,
    }
    (OUT_JSON / "meta_w1_steering.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON/'meta_w1_steering.json'}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))
    for name, r in results.items():
        ys = [r["curve"][str(a)]["logit_diff_mean"] for a in ALPHAS]
        yerr = [r["curve"][str(a)]["logit_diff_std"] for a in ALPHAS]
        ax.errorbar(ALPHAS, ys, yerr=yerr, marker="o", label=f"{name} ({r['low_word']}/{r['high_word']})",
                    capsize=3, alpha=0.85)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("α (steering magnitude along meta-direction w₁)")
    ax.set_ylabel("logit_diff(high − low)  (mean ± 1 std)")
    ax.set_title("Meta-direction w₁ steering across 8 adjective pairs (E4B layer 32)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "meta_w1_steering_curves.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'meta_w1_steering_curves.png'}")


if __name__ == "__main__":
    main()
