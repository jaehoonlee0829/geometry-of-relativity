"""v11 P5 — derive head taxonomy + causal head ablation.

Two stages:

  Stage A — taxonomy (CPU, from already-extracted v11 attention NPZs)
    Per (strategic-layer L, head h):
      - attn_mass_ctx, attn_mass_tgt  (last-token softmax mass on the 15
        context value tokens vs the target value token)
      - r2_z_head   = R²(z) of a ridge fit on head_outputs[L, h] (4000, head_dim)
      - r2_mu_head  = R²(μ)
      - r2_x_head   = R²(x)
      - dla_score   = mean over prompts of (probe_z @ (W_O[:, slice] @ head_out)),
                      where probe_z is the ridge probe at the canonical late layer

    Pick top-quartile thresholds and tag heads:
      - μ-aggregator: r2_mu_head high, attn_mass_ctx high, layer ≤ 5
      - comparator  : r2_z_head high, attn_mass_ctx ≈ attn_mass_tgt, mid layer
      - z-writer    : |dla_score| high, r2_z_head high, layer ≥ 10

  Stage B — ablation (GPU)
    For 2B, ablate v10's canonical heads L13h2 / L3h0 / L0h6 (override-able).
    For 9B, ablate the top z-writer / comparator / μ-aggregator from Stage A.
    Measure ΔR²(z) on cell-mean logit_diff at every downstream layer.

Inputs:
  results/v11/<model_short>/height/<base>_attention.npz    (height as the fingerprint pair)
  results/v11/<model_short>/height/<base>_residuals.npz
  results/v11/<model_short>/height/<base>_W_O_strategic.npz
  data_gen/v11_height_trials.jsonl                         (for ablation forward pass)
Outputs:
  results/v11/<model_short>/head_taxonomy.json
  results/v11/<model_short>/head_ablation_causal.json
  figures/v11/attention/head_taxonomy_<model_short>.png
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

MODEL_BY_SHORT = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
}
LATE_BY_SHORT = {"gemma2-2b": 20, "gemma2-9b": 33}
# 2B canonical heads from v10 §14.6 — kept for reference / sanity check
CANONICAL_2B = {"comparator": (13, 2), "early_writer": (3, 0), "mu_aggregator": (0, 6)}


def cv_r2(X, y, k=5):
    if y.std() < 1e-12 or X.shape[0] < k * 2: return 0.0
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ss = ((y - pred) ** 2).sum(); ss0 = ((y - y.mean()) ** 2).sum()
    return float(1 - ss / max(ss0, 1e-12))


def get_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr): m = getattr(m, attr)
    return m.layers


def derive_taxonomy(model_short: str) -> dict:
    """Stage A — produce per-head metrics and top-quartile taxonomy."""
    base = REPO / "results" / "v11" / model_short / "height"
    pfx = f"{model_short}_height_v11"
    res = np.load(base / f"{pfx}_residuals.npz")
    attn = np.load(base / f"{pfx}_attention.npz")
    wo = np.load(base / f"{pfx}_W_O_strategic.npz")

    z = res["z"].astype(np.float64)
    mu = res["mu"].astype(np.float64)
    x = res["x"].astype(np.float64)
    acts = res["activations"]
    late = LATE_BY_SHORT[model_short]
    h_late = acts[:, late, :].astype(np.float64)

    # Ridge probe for z at the canonical late layer (used for DLA scoring)
    probe = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(h_late, z)
    w_z = probe.coef_.astype(np.float64)

    head_outputs = attn["head_outputs"]   # (N, n_strat, n_heads, head_dim)
    attn_last = attn["attn_last_row"]     # (N, n_strat, n_heads, max_seq)
    strat_layers = list(map(int, attn["attn_layers"]))
    seq_len_un = attn["seq_len_unpadded"]
    seq_pad = int(attn["seq_len_padded"])
    target_pos = attn["target_pos_unpadded"]
    context_pos = attn["context_pos_unpadded"]
    n_prompts, n_strat, n_heads, head_dim = head_outputs.shape

    # Per-prompt offset to convert unpadded position → padded position
    pad_off = (seq_pad - seq_len_un).astype(np.int32)

    rows: list[dict] = []
    for k_L, L in enumerate(strat_layers):
        # W_O slice for this layer: (d_model, n_heads*head_dim)
        W_O = wo[f"L{L}"]  # already fp32
        for h in range(n_heads):
            ho = head_outputs[:, k_L, h, :].astype(np.float64)  # (N, head_dim)
            # DLA contribution: per-prompt residual contribution = W_O slice · head_out
            slc = W_O[:, h * head_dim:(h + 1) * head_dim]  # (d_model, head_dim)
            contrib = ho @ slc.T   # (N, d_model)
            dla = (contrib @ w_z).astype(np.float64)   # (N,)

            # Attention mass: sum over context positions vs target position
            mass_ctx = np.zeros(n_prompts, dtype=np.float64)
            mass_tgt = np.zeros(n_prompts, dtype=np.float64)
            attn_row = attn_last[:, k_L, h, :].astype(np.float64)  # (N, max_seq)
            for i in range(n_prompts):
                off = pad_off[i]
                # context: 15 ranges in unpadded coords
                for c in range(context_pos.shape[1]):
                    s, e = context_pos[i, c]
                    mass_ctx[i] += attn_row[i, off + s:off + e].sum()
                ts, te = target_pos[i]
                mass_tgt[i] += attn_row[i, off + ts:off + te].sum()

            rows.append({
                "layer": int(L),
                "head": int(h),
                "attn_mass_ctx_mean": float(mass_ctx.mean()),
                "attn_mass_tgt_mean": float(mass_tgt.mean()),
                "r2_z_head": cv_r2(ho, z),
                "r2_mu_head": cv_r2(ho, mu),
                "r2_x_head": cv_r2(ho, x),
                "dla_signed_mean": float(dla.mean()),
                "dla_abs_mean": float(np.abs(dla).mean()),
            })

    # Top-quartile thresholds
    arr = lambda key: np.array([r[key] for r in rows])
    thr_ctx = float(np.quantile(arr("attn_mass_ctx_mean"), 0.75))
    thr_tgt = float(np.quantile(arr("attn_mass_tgt_mean"), 0.75))
    thr_dla = float(np.quantile(arr("dla_abs_mean"), 0.75))
    for r in rows:
        tags = []
        L = r["layer"]
        if r["attn_mass_ctx_mean"] > thr_ctx and r["r2_mu_head"] > 0.5 and L <= 5:
            tags.append("mu_aggregator")
        if r["r2_z_head"] > 0.5 and r["attn_mass_ctx_mean"] > thr_ctx * 0.6 and r["attn_mass_tgt_mean"] > thr_tgt * 0.6:
            tags.append("comparator")
        if r["dla_abs_mean"] > thr_dla and r["r2_z_head"] > 0.4 and L >= max(7, LATE_BY_SHORT[model_short] - 12):
            tags.append("z_writer")
        r["tags"] = tags

    # Pick top candidate per role (by the metric most diagnostic of that role)
    def top(role: str, key: str, k: int = 5) -> list[tuple[int, int]]:
        cands = [r for r in rows if role in r["tags"]]
        cands.sort(key=lambda r: r[key], reverse=True)
        return [(r["layer"], r["head"]) for r in cands[:k]]

    taxonomy = {
        "model_short": model_short,
        "thresholds": {"ctx": thr_ctx, "tgt": thr_tgt, "dla_abs": thr_dla},
        "n_heads_per_layer": n_heads,
        "strategic_layers": strat_layers,
        "rows": rows,
        "top_z_writers": top("z_writer", "dla_abs_mean", 8),
        "top_comparators": top("comparator", "r2_z_head", 8),
        "top_mu_aggregators": top("mu_aggregator", "r2_mu_head", 8),
    }
    return taxonomy


def ablate_heads(model_short: str, ablation_targets: list[tuple[str, int, int]]) -> dict:
    """Stage B — zero a head's output during forward, measure ΔR²(z) at late layer.

    ablation_targets: list of (label, layer, head) tuples to zero in turn.
    Each ablation runs the full 4000-prompt height set.
    """
    model_id = MODEL_BY_SHORT[model_short]
    pair = "height"
    trials_path = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    trials = [json.loads(l) for l in trials_path.open()]
    print(f"[ablate] loading {model_id}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()

    layers_mod = get_layers(model)
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    hi_id = first_token_id(tok, "tall")
    lo_id = first_token_id(tok, "short")

    def measure(per_prompt_logit_diff: np.ndarray) -> float:
        """corr(cell-mean ld, z) using the trials' z labels."""
        zs = np.array([t["z"] for t in trials], dtype=np.float64)
        xs = np.array([t["x"] for t in trials], dtype=np.float64)
        keys = [(round(float(x), 4), round(float(z), 4)) for x, z in zip(xs, zs)]
        from collections import defaultdict
        cell = defaultdict(list)
        for i, k in enumerate(keys): cell[k].append(per_prompt_logit_diff[i])
        means = np.array([np.mean(v) for v in cell.values()])
        cell_zs = np.array([k[1] for k in cell.keys()])
        return float(np.corrcoef(means, cell_zs)[0, 1])

    def forward_with_ablation(layer: int | None, head: int | None) -> np.ndarray:
        handle = None
        if layer is not None and head is not None:
            o_proj = layers_mod[layer].self_attn.o_proj
            def pre_hook(module, args):
                x = args[0]
                # zero head's slice across all positions
                xm = x.clone()
                xm[..., head * head_dim:(head + 1) * head_dim] = 0
                return (xm,) + args[1:]
            handle = o_proj.register_forward_pre_hook(pre_hook)
        try:
            out = np.zeros(len(trials), dtype=np.float32)
            BS = 16
            for b0 in range(0, len(trials), BS):
                batch = trials[b0:b0 + BS]
                prompts = [t["prompt"] for t in batch]
                enc = tok(prompts, return_tensors="pt",
                          padding="max_length", max_length=224,
                          truncation=True).to(model.device)
                with torch.no_grad():
                    logits = model(**enc, use_cache=False).logits[:, -1, :].float()
                ld = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
                out[b0:b0 + len(batch)] = ld
        finally:
            if handle is not None: handle.remove()
        return out

    print(f"[ablate] baseline (no ablation) ...", flush=True)
    t0 = time.time()
    base_ld = forward_with_ablation(None, None)
    base_r = measure(base_ld)
    print(f"[ablate]   baseline corr(ld_mean, z) = {base_r:.3f}  ({time.time() - t0:.1f}s)",
          flush=True)

    results = []
    for label, L, H in ablation_targets:
        t1 = time.time()
        ld = forward_with_ablation(L, H)
        r = measure(ld)
        delta = r - base_r
        ent_mean = float(np.std(ld))  # poor man's effect-size proxy
        print(f"[ablate]   {label}=L{L}h{H}  corr(z) = {r:+.3f}  Δ = {delta:+.3f}  "
              f"({time.time() - t1:.1f}s)", flush=True)
        results.append({
            "label": label, "layer": L, "head": H,
            "corr_z_after_ablation": r,
            "delta_corr_z_vs_baseline": delta,
            "logit_diff_std": ent_mean,
        })
    return {
        "model_short": model_short,
        "baseline_corr_z": base_r,
        "ablations": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=list(MODEL_BY_SHORT.keys()))
    ap.add_argument("--skip-ablation", action="store_true",
                    help="only derive taxonomy (no GPU)")
    args = ap.parse_args()

    print(f"[P5] {args.model_short}: stage A — head taxonomy ...", flush=True)
    tax = derive_taxonomy(args.model_short)
    out_dir = REPO / "results" / "v11" / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "head_taxonomy.json").write_text(json.dumps(tax, indent=2))
    print(f"[P5] wrote {out_dir / 'head_taxonomy.json'}", flush=True)
    print(f"[P5]   top z-writers:    {tax['top_z_writers'][:3]}")
    print(f"[P5]   top comparators:  {tax['top_comparators'][:3]}")
    print(f"[P5]   top μ-aggregators:{tax['top_mu_aggregators'][:3]}")

    if args.skip_ablation:
        return

    # Build ablation targets
    targets: list[tuple[str, int, int]] = []
    if args.model_short == "gemma2-2b":
        # Use v10's canonical heads (L13h2/L3h0/L0h6) per CANONICAL_2B.
        for label, (L, H) in CANONICAL_2B.items():
            targets.append((label, L, H))
    else:
        # 9B — use top derived candidates from Stage A.
        if tax["top_z_writers"]:    targets.append(("z_writer_top",     *tax["top_z_writers"][0]))
        if tax["top_comparators"]:  targets.append(("comparator_top",   *tax["top_comparators"][0]))
        if tax["top_mu_aggregators"]:targets.append(("mu_aggregator_top",*tax["top_mu_aggregators"][0]))

    if not targets:
        print(f"[P5] no ablation targets found — skipping stage B")
        return

    print(f"[P5] stage B — causal ablation of {len(targets)} heads...", flush=True)
    abl = ablate_heads(args.model_short, targets)
    (out_dir / "head_ablation_causal.json").write_text(json.dumps(abl, indent=2))
    print(f"[P5] wrote {out_dir / 'head_ablation_causal.json'}")


if __name__ == "__main__":
    main()
