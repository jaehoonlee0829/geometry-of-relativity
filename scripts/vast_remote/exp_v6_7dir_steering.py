"""v6 Block D: 7-direction steering × 8 pairs × on-manifold + entropy + transfer.

Seven candidate adjective-axes per pair:
  1. primal_z      — mean(acts | z>+1) − mean(acts | z<−1)  (diff-of-means on z)
  2. primal_x      — mean(acts | x high quartile) − mean(acts | x low quartile)
  3. probe_z       — Ridge(acts, z).coef_
  4. probe_x       — Ridge(acts, x).coef_
  5. meta_w1       — top right singular vector of stacked per-pair PC1s (v5 obj)
  6. zeroshot_wx   — Ridge(zeroshot_acts, x).coef_  (from v4_zeroshot_expanded)
  7. pair_PC2      — PC2 of cell-mean activations for this pair (surprise axis)

For each (pair × direction × α ∈ {-8, -4, -2, -1, 0, 1, 2, 4, 8}):
  - forward-hook-steer at layer 32 (last token only), measure:
      * logit_diff(high - low)
      * softmax entropy at last position
      * on-manifold distance: project the steered residual at layer 32 onto
        the pair's unsteered (PC1, PC2) basis, measure perpendicular distance
        to the fitted parabola PC2 = a·PC1² + b·PC1 + c

Cross-pair transfer matrix: steer age-pair prompts using height's probe_z,
measure shift slope. Repeat height × age, age × wealth, etc.

Outputs:
  results/v6_steering/direction_comparison.json
  results/v6_steering/transfer_matrix.json
  figures/v6/seven_direction_curves_8pair.png
  figures/v6/on_manifold_distance.png
  figures/v6/entropy_vs_alpha.png
  figures/v6/transfer_heatmap.png
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
N_PROMPTS_PER_PAIR = 60   # smaller than v5's 100 because 7 directions × 9 α × 8 pairs × 60 = 30k forward passes
BATCH_SIZE = 16

ADJPAIRS = REPO / "results" / "v4_adjpairs"
ZS_EXPANDED = REPO / "results" / "v4_zeroshot_expanded"
OUT_JSON = REPO / "results" / "v6_steering"
OUT_FIG = REPO / "figures" / "v6"
OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

PAIR_NAMES = [p.name for p in PAIR_OBJS]


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def load_implicit_acts(pair_name: str, layer: str):
    trials_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in (ADJPAIRS / "e4b_trials.jsonl").open()}
    npz = np.load(ADJPAIRS / f"e4b_{pair_name}_implicit_{layer}.npz", allow_pickle=True)
    acts = npz["activations"].astype(np.float64)
    ids = [str(s) for s in npz["ids"]]
    xs = np.array([trials_by_id[i]["x"] for i in ids])
    zs = np.array([trials_by_id[i]["z"] for i in ids])
    return acts, xs, zs


def compute_per_pair_directions(pair_name: str):
    """Return dict of 7 unit-norm direction vectors for this pair at layer=late."""
    acts, xs, zs = load_implicit_acts(pair_name, "late")
    # Primal z diff-of-means
    hi_mask = zs > +1.0
    lo_mask = zs < -1.0
    primal_z = acts[hi_mask].mean(0) - acts[lo_mask].mean(0)
    # Primal x diff-of-means (top/bottom quartile)
    x_hi = np.percentile(xs, 75); x_lo = np.percentile(xs, 25)
    primal_x = acts[xs >= x_hi].mean(0) - acts[xs <= x_lo].mean(0)
    # Probes
    w_z = Ridge(alpha=1.0).fit(acts, zs).coef_
    w_x = Ridge(alpha=1.0).fit(acts, xs).coef_
    # PC2 (surprise)
    centered = acts - acts.mean(0)
    pca = PCA(n_components=2).fit(centered)
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    # sign-align pc1 with z (so pc2's sign is reproducible)
    if np.corrcoef(centered @ pc1, zs)[0, 1] < 0:
        pc1 = -pc1
        pc2 = -pc2

    # Zero-shot wx (from v4_zeroshot_expanded)
    zs_npz = np.load(ZS_EXPANDED / f"e4b_{pair_name}_late.npz", allow_pickle=True)
    zs_acts = zs_npz["activations"].astype(np.float64)
    zs_ids = [str(s) for s in zs_npz["ids"]]
    zs_trials = {json.loads(l)["id"]: json.loads(l)
                 for l in (ZS_EXPANDED / "e4b_trials.jsonl").open()}
    zs_xs = np.array([zs_trials[i]["x"] for i in zs_ids])
    w_x_zs = Ridge(alpha=1.0).fit(zs_acts, zs_xs).coef_

    def unit(v): return v / (np.linalg.norm(v) + 1e-12)

    return {
        "primal_z":    unit(primal_z),
        "primal_x":    unit(primal_x),
        "probe_z":     unit(w_z),
        "probe_x":     unit(w_x),
        "pc2":         unit(pc2),
        "zeroshot_wx": unit(w_x_zs),
    }, pc1, pc2


def compute_meta_w1(pc1_dict):
    """Replicate v5 meta-direction: SVD of stacked PC1s, take w1."""
    V = np.stack([pc1_dict[n] for n in PAIR_NAMES])
    U, S, Wt = np.linalg.svd(V, full_matrices=False)
    w1 = Wt[0]
    return w1 / np.linalg.norm(w1)


def subsample_prompts(pair_obj, n: int, rng):
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


def steered_measure(model, tok, prompts, layer_idx, direction, alpha,
                    high_id, low_id, pc1_np=None, pc2_np=None):
    """Run forward with an additive hook at `layer_idx`; also capture the
    steered residual so we can project onto (PC1, PC2).

    Returns: ld array, entropy array, pc_coords array (None if pc1 not given)
    """
    layers = get_layers(model)
    captured = []

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = direction.to(device=h.device, dtype=h.dtype)
        h[:, -1, :] = h[:, -1, :] + alpha * d
        captured.append(h[:, -1, :].detach().float().cpu().numpy())
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = layers[layer_idx].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lds, entropies = [], []
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]   # (B, V)
            logprobs = torch.log_softmax(logits.double(), dim=-1)
            p = logprobs.exp()
            entropy = -(p * logprobs).sum(-1).float().cpu().numpy()
            ld = (logits[:, high_id] - logits[:, low_id]).float().cpu().numpy()
            lds.append(ld)
            entropies.append(entropy)
    finally:
        handle.remove()
    ld_arr = np.concatenate(lds)
    ent_arr = np.concatenate(entropies)
    acts_arr = np.concatenate(captured, axis=0) if captured else np.zeros((0,))
    pc_coords = None
    if pc1_np is not None and len(acts_arr):
        pc_coords = np.column_stack([acts_arr @ pc1_np, acts_arr @ pc2_np])
    return ld_arr, ent_arr, pc_coords


def fit_parabola(pc_coords):
    """Fit pc2 = a·pc1² + b·pc1 + c; return coefficients + residual dist fn."""
    x1 = pc_coords[:, 0]; x2 = pc_coords[:, 1]
    A = np.column_stack([x1**2, x1, np.ones_like(x1)])
    coef, *_ = np.linalg.lstsq(A, x2, rcond=None)

    def dist(coords):
        x1 = coords[:, 0]; x2 = coords[:, 1]
        pred = coef[0]*x1**2 + coef[1]*x1 + coef[2]
        return np.abs(x2 - pred)
    return coef, dist


def main():
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Compute per-pair directions (pair_name → {direction_name → unit vector})
    print("\nComputing per-pair directions…", flush=True)
    per_pair_dirs = {}
    pc1_store = {}
    pc2_store = {}
    for pair_obj in PAIR_OBJS:
        dirs, pc1, pc2 = compute_per_pair_directions(pair_obj.name)
        per_pair_dirs[pair_obj.name] = dirs
        pc1_store[pair_obj.name] = pc1
        pc2_store[pair_obj.name] = pc2

    # Compute meta w1 across pairs (signed per v5 convention)
    w1 = compute_meta_w1(pc1_store)

    # Parabola fit (for on-manifold distance) per pair — use unsteered cell means
    parabolas = {}
    for pair_name in PAIR_NAMES:
        acts, xs, zs = load_implicit_acts(pair_name, "late")
        centered = acts - acts.mean(0)
        coords = np.column_stack([centered @ pc1_store[pair_name],
                                  centered @ pc2_store[pair_name]])
        coefs, _ = fit_parabola(coords)
        parabolas[pair_name] = {"coefs": coefs.tolist(),
                                 "mean": acts.mean(0).tolist()}
    # Prompts
    rng = np.random.default_rng(0)
    prompts_by_pair = {po.name: subsample_prompts(po, N_PROMPTS_PER_PAIR, rng) for po in PAIR_OBJS}

    DIRECTION_NAMES = ["primal_z", "primal_x", "probe_z", "probe_x", "meta_w1", "zeroshot_wx", "pc2"]
    results: dict = {}
    total_t0 = time.time()

    for pair_obj in PAIR_OBJS:
        pair = pair_obj.name
        t_pair = time.time()
        print(f"\n=== steering {pair} ===", flush=True)
        prompts = [t["prompt"] for t in prompts_by_pair[pair]]
        high_id = first_token_id(tok, pair_obj.high_word)
        low_id = first_token_id(tok, pair_obj.low_word)
        pc1_np = pc1_store[pair]; pc2_np = pc2_store[pair]
        # For on-manifold distance we need the ACTIVATION-SPACE direction used for
        # projection. We use (pc1, pc2) in pair-local centered space; subtract
        # acts.mean(0) from captured hidden states before projecting.
        acts_mean = np.array(parabolas[pair]["mean"])

        pair_out: dict = {}
        for dname in DIRECTION_NAMES:
            if dname == "meta_w1":
                direction_np = w1
            else:
                direction_np = per_pair_dirs[pair][dname]
            direction = torch.from_numpy(direction_np).to(model.device)
            per_alpha = {}
            for alpha in ALPHAS:
                ld, ent, pcc = steered_measure(
                    model, tok, prompts, LAYER_IDX, direction, alpha,
                    high_id, low_id, pc1_np, pc2_np,
                )
                # On-manifold distance: project (pc1, pc2) after centering with acts_mean
                pcc_centered = pcc - np.array([acts_mean @ pc1_np, acts_mean @ pc2_np])
                # Distance from parabola
                coefs = np.array(parabolas[pair]["coefs"])
                pred = coefs[0]*pcc_centered[:, 0]**2 + coefs[1]*pcc_centered[:, 0] + coefs[2]
                off_manifold = np.abs(pcc_centered[:, 1] - pred)
                per_alpha[str(alpha)] = {
                    "ld_mean": float(ld.mean()), "ld_std": float(ld.std()),
                    "entropy_mean": float(ent.mean()), "entropy_std": float(ent.std()),
                    "off_manifold_mean": float(off_manifold.mean()),
                    "off_manifold_std": float(off_manifold.std()),
                }
            xs_alpha = np.array(ALPHAS)
            ys_ld = np.array([per_alpha[str(a)]["ld_mean"] for a in ALPHAS])
            slope = float(np.polyfit(xs_alpha, ys_ld, 1)[0])
            # baseline off-manifold (alpha=0)
            off0 = per_alpha["0.0"]["off_manifold_mean"]
            off_max = max(per_alpha[str(a)]["off_manifold_mean"] for a in ALPHAS)
            pair_out[dname] = {
                "slope_ld": slope,
                "range_ld": float(ys_ld.max() - ys_ld.min()),
                "entropy_at_alpha0": per_alpha["0.0"]["entropy_mean"],
                "max_entropy_shift": max(abs(per_alpha[str(a)]["entropy_mean"] - per_alpha["0.0"]["entropy_mean"]) for a in ALPHAS),
                "off_manifold_at_alpha0": off0,
                "off_manifold_max": off_max,
                "off_manifold_ratio_max_vs_0": off_max / (off0 + 1e-12),
                "curve": per_alpha,
            }
            print(f"  {dname:13s}  slope={slope:+.4f}  range={ys_ld.max()-ys_ld.min():.3f}  "
                  f"Δent_max={pair_out[dname]['max_entropy_shift']:.3f}  "
                  f"off_max/off0={pair_out[dname]['off_manifold_ratio_max_vs_0']:.1f}×",
                  flush=True)
        results[pair] = pair_out
        print(f"  (pair took {time.time()-t_pair:.1f}s)", flush=True)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal steering time: {total_elapsed:.1f}s", flush=True)

    # Cross-pair transfer: steer pair_B prompts with pair_A's probe_z
    print("\n=== cross-pair transfer (probe_z only, α=±4) ===", flush=True)
    transfer = {}
    for pair_a in PAIR_NAMES:
        transfer[pair_a] = {}
        direction_np = per_pair_dirs[pair_a]["probe_z"]
        direction = torch.from_numpy(direction_np).to(model.device)
        for pair_b_obj in PAIR_OBJS:
            pair_b = pair_b_obj.name
            prompts = [t["prompt"] for t in prompts_by_pair[pair_b]]
            high_id = first_token_id(tok, pair_b_obj.high_word)
            low_id = first_token_id(tok, pair_b_obj.low_word)
            ld_m4, _, _ = steered_measure(model, tok, prompts, LAYER_IDX,
                                           direction, -4.0, high_id, low_id)
            ld_p4, _, _ = steered_measure(model, tok, prompts, LAYER_IDX,
                                           direction, +4.0, high_id, low_id)
            transfer[pair_a][pair_b] = float(ld_p4.mean() - ld_m4.mean()) / 8.0
        print(f"  from {pair_a}: " +
              "  ".join(f"{p}={transfer[pair_a][p]:+.3f}" for p in PAIR_NAMES), flush=True)

    (OUT_JSON / "direction_comparison.json").write_text(json.dumps(results, indent=2))
    (OUT_JSON / "transfer_matrix.json").write_text(json.dumps({
        "direction_used": "probe_z",
        "alpha_range": 8,
        "effect_per_alpha_unit": transfer,
    }, indent=2))
    print(f"\nwrote direction_comparison.json + transfer_matrix.json")

    # ---------- figures ----------
    colors = {"primal_z": "#1f77b4", "primal_x": "#ff7f0e", "probe_z": "#2ca02c",
              "probe_x": "#d62728", "meta_w1": "#9467bd", "zeroshot_wx": "#8c564b",
              "pc2": "#e377c2"}

    # 7-direction curves × 8 pairs
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, pair in zip(axes.ravel(), PAIR_NAMES):
        for dname in DIRECTION_NAMES:
            ys = [results[pair][dname]["curve"][str(a)]["ld_mean"] for a in ALPHAS]
            ax.plot(ALPHAS, ys, marker="o", color=colors[dname], label=dname, lw=1.5)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_title(pair, fontsize=10); ax.grid(alpha=0.3); ax.set_xlabel("α", fontsize=8)
        ax.set_ylabel("logit_diff", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("7-direction steering curves (α-sweep, E4B layer 32)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "seven_direction_curves_8pair.png", dpi=130)
    plt.close(fig)

    # Off-manifold plot: max off-manifold / off-manifold at α=0 per (pair × direction)
    fig, ax = plt.subplots(figsize=(12, 6))
    xpos = np.arange(len(PAIR_NAMES))
    w = 1.0 / (len(DIRECTION_NAMES) + 1)
    for i, dname in enumerate(DIRECTION_NAMES):
        ratios = [results[p][dname]["off_manifold_ratio_max_vs_0"] for p in PAIR_NAMES]
        ax.bar(xpos + i*w, ratios, w, label=dname, color=colors[dname])
    ax.axhline(1.0, color="k", ls="--", alpha=0.3, label="on-manifold")
    ax.set_xticks(xpos + 3*w); ax.set_xticklabels(PAIR_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("off-manifold distance at max α / distance at α=0  (×)")
    ax.set_title("On-manifold check: does steering stay on the PCA parabola?")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "on_manifold_distance.png", dpi=130)
    plt.close(fig)

    # Entropy shift plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, dname in enumerate(DIRECTION_NAMES):
        shifts = [results[p][dname]["max_entropy_shift"] for p in PAIR_NAMES]
        ax.bar(xpos + i*w, shifts, w, label=dname, color=colors[dname])
    ax.set_xticks(xpos + 3*w); ax.set_xticklabels(PAIR_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("max |entropy(steered) − entropy(α=0)|  (nats)")
    ax.set_title("Softmax entropy shift at max α")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "entropy_vs_alpha.png", dpi=130)
    plt.close(fig)

    # Transfer heatmap
    T = np.array([[transfer[a][b] for b in PAIR_NAMES] for a in PAIR_NAMES])
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(T.min()), abs(T.max()))
    im = ax.imshow(T, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(PAIR_NAMES))); ax.set_xticklabels(PAIR_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(PAIR_NAMES))); ax.set_yticklabels(PAIR_NAMES)
    ax.set_xlabel("prompts evaluated on pair (B)")
    ax.set_ylabel("direction learned on pair (A)")
    ax.set_title("Cross-pair transfer: slope of ld vs α for probe_z  (α∈[−4,+4])")
    for i in range(len(PAIR_NAMES)):
        for j in range(len(PAIR_NAMES)):
            ax.text(j, i, f"{T[i,j]:+.2f}", ha="center", va="center",
                    color="white" if abs(T[i,j]) > vmax*0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "transfer_heatmap.png", dpi=130)
    plt.close(fig)

    print("wrote 4 figures in figures/v6/")


if __name__ == "__main__":
    main()
