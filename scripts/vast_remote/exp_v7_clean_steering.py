"""v7 Priority 3: 7-direction steering with CLEAN (Grid B) directions.

Reuses v6 Block D protocol (α-sweep, entropy, on-manifold) but with the
7 directions computed from Grid B activations. Key question: does the
13× slope gap between primal_z and probe_z survive on the clean grid?

We also ALWAYS evaluate on the same (Grid A style) implicit prompts, so
logit_diff measurements are comparable to v6.

Writes:
  results/v7_steering/clean_direction_comparison.json
  figures/v7/seven_direction_curves_clean_8pair.png
  figures/v7/clean_vs_v6_primal_probe_slopes.png
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS as PAIR_OBJS, compute_z, make_implicit_prompt  # noqa: E402

MODEL_ID = "google/gemma-4-E4B"
LAYER_IDX = 32
ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
N_PROMPTS_PER_PAIR = 60
BATCH_SIZE = 16

V7 = REPO / "results" / "v7_xz_grid"
ZS_EXPANDED = REPO / "results" / "v4_zeroshot_expanded"
OUT_JSON = REPO / "results" / "v7_steering"
OUT_FIG = REPO / "figures" / "v7"
OUT_JSON.mkdir(parents=True, exist_ok=True)

PAIR_NAMES = [p.name for p in PAIR_OBJS]
DIR_NAMES = ["primal_z", "primal_x", "probe_z", "probe_x", "pc1", "pc2", "meta_w1", "zeroshot_wx"]


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def compute_dirs_clean(pair_name: str):
    """From Grid B activations."""
    trials_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in (V7 / "e4b_trials.jsonl").open()}
    npz = np.load(V7 / f"e4b_{pair_name}_late.npz", allow_pickle=True)
    acts = npz["activations"].astype(np.float64)
    ids = [str(s) for s in npz["ids"]]
    xs = np.array([trials_by_id[i]["x"] for i in ids])
    zs = np.array([trials_by_id[i]["z"] for i in ids])

    primal_z = acts[zs > +1.0].mean(0) - acts[zs < -1.0].mean(0)
    x_hi, x_lo = np.percentile(xs, 75), np.percentile(xs, 25)
    primal_x = acts[xs >= x_hi].mean(0) - acts[xs <= x_lo].mean(0)
    w_z = Ridge(alpha=1.0).fit(acts, zs).coef_
    w_x = Ridge(alpha=1.0).fit(acts, xs).coef_
    centered = acts - acts.mean(0)
    pca = PCA(n_components=2).fit(centered)
    pc1, pc2 = pca.components_[0], pca.components_[1]
    if np.corrcoef(centered @ pc1, zs)[0, 1] < 0:
        pc1, pc2 = -pc1, -pc2

    zs_npz = np.load(ZS_EXPANDED / f"e4b_{pair_name}_late.npz", allow_pickle=True)
    zs_trials = {json.loads(l)["id"]: json.loads(l)
                 for l in (ZS_EXPANDED / "e4b_trials.jsonl").open()}
    zs_acts = zs_npz["activations"].astype(np.float64)
    zs_ids = [str(s) for s in zs_npz["ids"]]
    zs_xs = np.array([zs_trials[i]["x"] for i in zs_ids])
    w_x_zs = Ridge(alpha=1.0).fit(zs_acts, zs_xs).coef_

    return {"primal_z": unit(primal_z), "primal_x": unit(primal_x),
            "probe_z":  unit(w_z),      "probe_x":  unit(w_x),
            "pc1":      unit(pc1),      "pc2":      unit(pc2),
            "zeroshot_wx": unit(w_x_zs)}, pc1, pc2


def compute_meta_w1_clean(pc1_list):
    V = np.stack(pc1_list)
    _, _, Wt = np.linalg.svd(V, full_matrices=False)
    w1 = Wt[0] / np.linalg.norm(Wt[0])
    # Sign-align w1 to match mean PC1 direction (SVD sign is arbitrary)
    if np.dot(w1, V.mean(axis=0)) < 0:
        w1 = -w1
    return w1


def subsample_prompts(pair_obj, n, rng):
    all_trials = []
    for x in pair_obj.target_values:
        for mu in pair_obj.mu_values:
            z = compute_z(pair_obj, x, mu)
            for s in range(30):
                all_trials.append({
                    "prompt": make_implicit_prompt(pair_obj, x, mu, s),
                    "x": x, "mu": mu, "z": z, "seed": s,
                    "low_word": pair_obj.low_word, "high_word": pair_obj.high_word,
                })
    picks = rng.choice(len(all_trials), size=min(n, len(all_trials)), replace=False)
    return [all_trials[i] for i in picks]


def steered_measure(model, tok, prompts, layer_idx, direction, alpha, high_id, low_id):
    layers = get_layers(model)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = direction.to(device=h.device, dtype=h.dtype)
        h[:, -1, :] = h[:, -1, :] + alpha * d
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = layers[layer_idx].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lds, ents = [], []
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            logprobs = torch.log_softmax(logits.double(), dim=-1)
            ent = -(logprobs.exp() * logprobs).sum(-1).float().cpu().numpy()
            ld = (logits[:, high_id] - logits[:, low_id]).float().cpu().numpy()
            lds.append(ld); ents.append(ent)
    finally:
        handle.remove()
    return np.concatenate(lds), np.concatenate(ents)


def main():
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print("Computing CLEAN directions (from Grid B)…", flush=True)
    per_pair_dirs = {}
    pc1_store = {}
    for po in PAIR_OBJS:
        dirs, pc1, pc2 = compute_dirs_clean(po.name)
        per_pair_dirs[po.name] = dirs
        pc1_store[po.name] = pc1 / np.linalg.norm(pc1)
    meta_w1 = compute_meta_w1_clean([pc1_store[n] for n in PAIR_NAMES])

    rng = np.random.default_rng(0)
    prompts_by_pair = {po.name: subsample_prompts(po, N_PROMPTS_PER_PAIR, rng) for po in PAIR_OBJS}

    results = {}
    total_t0 = time.time()
    for pair_obj in PAIR_OBJS:
        pair = pair_obj.name
        t_pair = time.time()
        print(f"\n=== {pair} ===", flush=True)
        prompts = [t["prompt"] for t in prompts_by_pair[pair]]
        high_id = first_token_id(tok, pair_obj.high_word)
        low_id = first_token_id(tok, pair_obj.low_word)

        pair_out = {}
        for dname in DIR_NAMES:
            if dname == "meta_w1":
                direction_np = meta_w1
            else:
                direction_np = per_pair_dirs[pair][dname]
            direction = torch.from_numpy(direction_np).to(model.device)
            per_alpha = {}
            for alpha in ALPHAS:
                ld, ent = steered_measure(model, tok, prompts, LAYER_IDX, direction, alpha, high_id, low_id)
                per_alpha[str(alpha)] = {"ld_mean": float(ld.mean()), "ld_std": float(ld.std()),
                                          "entropy_mean": float(ent.mean())}
            xs_a = np.array(ALPHAS)
            ys_ld = np.array([per_alpha[str(a)]["ld_mean"] for a in ALPHAS])
            slope = float(np.polyfit(xs_a, ys_ld, 1)[0])
            pair_out[dname] = {
                "slope_ld": slope,
                "range_ld": float(ys_ld.max() - ys_ld.min()),
                "max_entropy_shift": max(abs(per_alpha[str(a)]["entropy_mean"] - per_alpha["0.0"]["entropy_mean"]) for a in ALPHAS),
                "curve": per_alpha,
            }
            print(f"  {dname:13s}  slope={slope:+.4f}  range={ys_ld.max()-ys_ld.min():.3f}",
                  flush=True)
        results[pair] = pair_out
        print(f"  (pair took {time.time()-t_pair:.1f}s)", flush=True)

    (OUT_JSON / "clean_direction_comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON/'clean_direction_comparison.json'}  (total {time.time()-total_t0:.1f}s)")

    # Compare to v6 results
    v6_path = REPO / "results" / "v6_steering" / "direction_comparison.json"
    v6 = json.load(v6_path.open()) if v6_path.exists() else None

    # Slope comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    colors = {"primal_z": "#1f77b4", "primal_x": "#ff7f0e", "probe_z": "#2ca02c",
              "probe_x": "#d62728", "meta_w1": "#9467bd", "zeroshot_wx": "#8c564b",
              "pc2": "#e377c2", "pc1": "#17becf"}
    for ax, pair in zip(axes.ravel(), PAIR_NAMES):
        for dname in DIR_NAMES:
            ys = [results[pair][dname]["curve"][str(a)]["ld_mean"] for a in ALPHAS]
            ax.plot(ALPHAS, ys, marker="o", color=colors[dname], label=dname, lw=1.5)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_title(pair, fontsize=10); ax.grid(alpha=0.3); ax.set_xlabel("α", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("v7 clean-grid: 8-direction steering curves (E4B layer 32)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "seven_direction_curves_clean_8pair.png", dpi=130)
    plt.close(fig)

    # Primal-vs-probe slopes: v6 (confounded) vs v7 (clean)
    if v6:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, which in zip(axes, ["primal_z", "probe_z"]):
            v6_slopes = [v6[p][which]["slope_ld"] if which in v6[p] else np.nan for p in PAIR_NAMES]
            v7_slopes = [results[p][which]["slope_ld"] for p in PAIR_NAMES]
            xpos = np.arange(len(PAIR_NAMES))
            ax.bar(xpos-0.2, np.abs(v6_slopes), 0.4, label=f"v6 (confounded)")
            ax.bar(xpos+0.2, np.abs(v7_slopes), 0.4, label=f"v7 (clean)")
            ax.set_xticks(xpos); ax.set_xticklabels(PAIR_NAMES, rotation=30, ha="right", fontsize=9)
            ax.set_title(f"|slope_ld| — {which}")
            ax.legend(); ax.grid(alpha=0.3)
        fig.suptitle("v6 vs v7 steering slopes — does the 13× primal/probe gap survive cleaning?")
        fig.tight_layout()
        fig.savefig(OUT_FIG / "clean_vs_v6_primal_probe_slopes.png", dpi=130)
        plt.close(fig)

    # Print summary: v7 primal vs probe
    print("\nv7 per-pair slope comparison (primal_z vs probe_z):")
    print(f"{'pair':15s}  primal_z  probe_z   ratio")
    for p in PAIR_NAMES:
        pz = results[p]["primal_z"]["slope_ld"]
        qz = results[p]["probe_z"]["slope_ld"]
        ratio = abs(pz) / (abs(qz) + 1e-6)
        print(f"  {p:12s}  {pz:+.4f}  {qz:+.4f}   {ratio:.1f}x")


if __name__ == "__main__":
    main()
