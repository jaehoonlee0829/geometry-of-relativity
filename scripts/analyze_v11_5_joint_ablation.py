"""v11.5 §I — joint head-set ablation with held-out threshold split.

Single-head ablations from v11 P5 were null. Try ablating the FULL tag-set
jointly (all comparators, all z-writers, all μ-aggregators) to test whether
ANY group of heads is causally necessary.

To avoid circularity (taxonomy thresholds picked on the same data the
ablation measures), split prompts in half by cell_seed parity:
  - threshold-fitting fold:  cell_seed in {0, 2, 4, 6, 8}
  - ablation-measurement fold: cell_seed in {1, 3, 5, 7, 9}

Tags (top-quartile thresholds) are picked on the threshold-fitting fold;
ablation effect is measured on the held-out fold.

Output: results/v11_5/<model_short>/joint_head_ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

MODEL_BY_SHORT = {"gemma2-2b": "google/gemma-2-2b", "gemma2-9b": "google/gemma-2-9b"}
LATE_BY_SHORT = {"gemma2-2b": 20, "gemma2-9b": 33}


def cv_r2(X, y, k=5):
    if y.std() < 1e-12: return 0.0
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ss = ((y - pred) ** 2).sum(); ss0 = ((y - y.mean()) ** 2).sum()
    return float(1 - ss / max(ss0, 1e-12))


def derive_tags_on_fold(model_short: str, train_mask: np.ndarray) -> dict:
    """Re-derive head taxonomy using only train_mask=True prompts."""
    base = REPO / "results" / "v11" / model_short / "height"
    pfx = f"{model_short}_height_v11"
    res = np.load(base / f"{pfx}_residuals.npz")
    attn = np.load(base / f"{pfx}_attention.npz")
    wo = np.load(base / f"{pfx}_W_O_strategic.npz")

    z = res["z"].astype(np.float64)[train_mask]
    mu = res["mu"].astype(np.float64)[train_mask]
    acts = res["activations"][train_mask]
    late = LATE_BY_SHORT[model_short]
    h_late = acts[:, late, :].astype(np.float64)
    probe = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(h_late, z)
    w_z = probe.coef_.astype(np.float64)

    head_outputs = attn["head_outputs"][train_mask]   # (N, n_strat, n_heads, head_dim)
    attn_last = attn["attn_last_row"][train_mask]
    seq_len_un = attn["seq_len_unpadded"][train_mask]
    seq_pad = int(attn["seq_len_padded"])
    target_pos = attn["target_pos_unpadded"][train_mask]
    context_pos = attn["context_pos_unpadded"][train_mask]
    strat_layers = list(map(int, attn["attn_layers"]))
    n_prompts, n_strat, n_heads, head_dim = head_outputs.shape
    pad_off = (seq_pad - seq_len_un).astype(np.int32)

    rows = []
    for k_L, L in enumerate(strat_layers):
        W_O = wo[f"L{L}"]
        for h in range(n_heads):
            ho = head_outputs[:, k_L, h, :].astype(np.float64)
            slc = W_O[:, h * head_dim:(h + 1) * head_dim]
            dla = (ho @ slc.T) @ w_z

            mass_ctx = np.zeros(n_prompts)
            mass_tgt = np.zeros(n_prompts)
            attn_row = attn_last[:, k_L, h, :].astype(np.float64)
            for i in range(n_prompts):
                off = pad_off[i]
                for c in range(context_pos.shape[1]):
                    s, e = context_pos[i, c]
                    mass_ctx[i] += attn_row[i, off + s:off + e].sum()
                ts, te = target_pos[i]
                mass_tgt[i] += attn_row[i, off + ts:off + te].sum()

            rows.append({
                "layer": int(L), "head": int(h),
                "ctx": float(mass_ctx.mean()),
                "tgt": float(mass_tgt.mean()),
                "r2_z_head": cv_r2(ho, z),
                "r2_mu_head": cv_r2(ho, mu),
                "dla_abs": float(np.abs(dla).mean()),
            })

    arr = lambda key: np.array([r[key] for r in rows])
    thr_ctx = float(np.quantile(arr("ctx"), 0.75))
    thr_dla = float(np.quantile(arr("dla_abs"), 0.75))

    tag_sets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        L = r["layer"]
        if r["ctx"] > thr_ctx and r["r2_mu_head"] > 0.5 and L <= 5:
            tag_sets["mu_aggregator"].append((L, r["head"]))
        if r["r2_z_head"] > 0.5 and r["ctx"] > thr_ctx * 0.6 and r["tgt"] > 0.6 * float(np.quantile(arr("tgt"), 0.75)):
            tag_sets["comparator"].append((L, r["head"]))
        if r["dla_abs"] > thr_dla and r["r2_z_head"] > 0.4 and L >= max(7, LATE_BY_SHORT[model_short] - 12):
            tag_sets["z_writer"].append((L, r["head"]))
    return {"tag_sets": {k: list(v) for k, v in tag_sets.items()}, "thr_ctx": thr_ctx, "thr_dla": thr_dla}


def get_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr): m = getattr(m, attr)
    return m.layers


def fwd_with_ablation(model, tok, prompts, ablate: list[tuple[int, int]],
                      hi_id, lo_id, head_dim, bs=16, max_seq=288):
    handles = []
    if ablate:
        layers_mod = get_layers(model)
        # Group ablations by layer so we register one pre-hook per layer.
        by_layer: dict[int, list[int]] = defaultdict(list)
        for L, H in ablate: by_layer[L].append(H)
        for L, heads in by_layer.items():
            o_proj = layers_mod[L].self_attn.o_proj
            heads_local = list(heads)
            def make_hook(heads_zero):
                def pre(module, args):
                    x = args[0].clone()
                    for H in heads_zero:
                        x[..., H * head_dim:(H + 1) * head_dim] = 0
                    return (x,) + args[1:]
                return pre
            handles.append(o_proj.register_forward_pre_hook(make_hook(heads_local)))
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
        for h in handles: h.remove()
    return out


def cell_corr(ld: np.ndarray, trials: list[dict]) -> float:
    keys = [(round(float(t["x"]), 4), round(float(t["z"]), 4)) for t in trials]
    cell = defaultdict(list)
    for i, k in enumerate(keys): cell[k].append(ld[i])
    means = np.array([np.mean(v) for v in cell.values()])
    cell_zs = np.array([k[1] for k in cell.keys()])
    return float(np.corrcoef(means, cell_zs)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=list(MODEL_BY_SHORT.keys()))
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    model_id = MODEL_BY_SHORT[args.model_short]
    pair = "height"
    trials_path = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    trials = [json.loads(l) for l in trials_path.open()]

    # Held-out split by cell_seed parity
    seeds = np.array([t.get("cell_seed", t.get("seed")) for t in trials])
    train_mask = (seeds % 2 == 0)
    test_mask = (seeds % 2 == 1)
    train_trials = [t for t, m in zip(trials, train_mask) if m]
    test_trials = [t for t, m in zip(trials, test_mask) if m]
    print(f"[joint-abl] {args.model_short} held-out split: train={len(train_trials)}, "
          f"test={len(test_trials)}", flush=True)

    print(f"[joint-abl] deriving taxonomy on train fold...", flush=True)
    tax = derive_tags_on_fold(args.model_short, train_mask)
    for k, v in tax["tag_sets"].items():
        print(f"  tag_set {k}: {len(v)} heads — {v[:5]}{'...' if len(v) > 5 else ''}")

    # Load model
    print(f"[joint-abl] loading {model_id}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    hi_id = first_token_id(tok, "tall")
    lo_id = first_token_id(tok, "short")
    test_prompts = [t["prompt"] for t in test_trials]

    # Baseline (no ablation) on test fold
    print(f"[joint-abl] baseline on test fold...", flush=True)
    t0 = time.time()
    base_ld = fwd_with_ablation(model, tok, test_prompts, [], hi_id, lo_id, head_dim,
                                args.batch_size, args.max_seq)
    base_r = cell_corr(base_ld, test_trials)
    print(f"[joint-abl]   baseline corr(z) = {base_r:.4f}  ({time.time() - t0:.1f}s)",
          flush=True)

    # Ablate each tag-set
    results = []
    for tag, heads in tax["tag_sets"].items():
        if not heads:
            continue
        t1 = time.time()
        ld = fwd_with_ablation(model, tok, test_prompts, heads, hi_id, lo_id, head_dim,
                               args.batch_size, args.max_seq)
        r = cell_corr(ld, test_trials)
        delta = r - base_r
        print(f"[joint-abl]   ablate ALL {tag} ({len(heads)} heads)  corr(z) = {r:.4f}  "
              f"Δ = {delta:+.4f}  ({time.time() - t1:.1f}s)", flush=True)
        results.append({
            "tag": tag, "n_heads_ablated": len(heads), "heads": heads,
            "corr_z_after": r, "delta_corr_z_vs_baseline": delta,
        })

    # Also ablate the UNION of all three tag-sets
    union = sorted({h for v in tax["tag_sets"].values() for h in v})
    if union:
        t1 = time.time()
        ld = fwd_with_ablation(model, tok, test_prompts, union, hi_id, lo_id, head_dim,
                               args.batch_size, args.max_seq)
        r = cell_corr(ld, test_trials)
        delta = r - base_r
        print(f"[joint-abl]   ablate UNION ({len(union)} heads)  corr(z) = {r:.4f}  "
              f"Δ = {delta:+.4f}  ({time.time() - t1:.1f}s)", flush=True)
        results.append({
            "tag": "union", "n_heads_ablated": len(union), "heads": union,
            "corr_z_after": r, "delta_corr_z_vs_baseline": delta,
        })

    out = {
        "model_short": args.model_short,
        "split_strategy": "cell_seed_parity",
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "tags_derived_on_train": {k: list(v) for k, v in tax["tag_sets"].items()},
        "thresholds_train": {"thr_ctx": tax["thr_ctx"], "thr_dla": tax["thr_dla"]},
        "baseline_corr_z_test": base_r,
        "ablations": results,
    }
    out_dir = REPO / "results" / "v11_5" / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "joint_head_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"[joint-abl] wrote {out_dir / 'joint_head_ablation.json'}")


if __name__ == "__main__":
    main()
