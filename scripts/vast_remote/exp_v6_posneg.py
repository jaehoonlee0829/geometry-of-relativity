"""v6 Block B: positive/negative math pair — cleanest possible absolute control.

Threshold at exactly 0 (mathematically absolute). No polysemy, no cultural
priors, no RLHF politeness. If this pair STILL shows relativity (R > 0.3),
the "all adjectives are context-relative to some degree" reframing has very
strong evidence.

Pair: posneg_abs, low=negative, high=positive.
x ∈ {-8, -3, 0, 3, 8}; μ ∈ {-6, -2, 0, 2, 6}; σ = 3.
5×5×30 seeds = 750 implicit + 25 explicit + 5 zero-shot = 780 prompts.

Writes:
  results/v4_abs_controls/e4b_posneg_abs_{cond}_{layer}.npz
  results/v4_abs_controls/e4b_posneg_abs_{cond}_logits.jsonl
  results/v4_abs_controls/e4b_posneg_abs_trials.jsonl
  results/v4_adjpairs_analysis/posneg_abs_result.json
  figures/v4_adjpairs/posneg_abs_panel.png
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
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402

MODEL_ID = "google/gemma-4-E4B"
LAYER_INDICES = {"mid": 21, "late": 32}
BATCH_SIZE = 16
OUT = REPO / "results" / "v4_abs_controls"
OUT_ANALYSIS = REPO / "results" / "v4_adjpairs_analysis"
OUT_FIG = REPO / "figures" / "v4_adjpairs"
OUT.mkdir(parents=True, exist_ok=True)

X_VALUES = [-8.0, -3.0, 0.0, 3.0, 8.0]
MU_VALUES = [-6.0, -2.0, 0.0, 2.0, 6.0]
SIGMA = 3.0
N_SEEDS = 30


def fmt_num(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.1f}"


def sample_context(mu: float, seed: int, n: int = 15) -> list[float]:
    rng = np.random.default_rng(seed + 98765)
    out = []
    while len(out) < n:
        v = rng.normal(mu, SIGMA)
        if -20 <= v <= 20:
            out.append(round(v, 1))
    return out


def build_implicit(x: float, mu: float, seed: int) -> str:
    sample = sample_context(mu, seed)
    items = [f"Number {i+1}: {v:g}" for i, v in enumerate(sample)]
    return "\n".join(items) + f"\nNumber {len(items)+1}: {fmt_num(x)}. This number is"


def build_explicit(x: float, mu: float) -> str:
    return (f"In a list where numbers cluster around {fmt_num(mu)} "
            f"(give or take {fmt_num(SIGMA)}), a number {fmt_num(x)} is")


def build_zeroshot(x: float) -> str:
    return f"A number {fmt_num(x)} is"


def build_trials():
    trials = []
    idx = 0
    for x in X_VALUES:
        for mu in MU_VALUES:
            z = (x - mu) / SIGMA
            for s in range(N_SEEDS):
                trials.append({
                    "id": f"posneg_abs_implicit_{idx:06d}",
                    "pair": "posneg_abs", "condition": "implicit",
                    "prompt": build_implicit(x, mu, s),
                    "x": x, "mu": mu, "z": z, "sigma": SIGMA, "seed": s,
                    "low_word": "negative", "high_word": "positive",
                })
                idx += 1
            trials.append({
                "id": f"posneg_abs_explicit_{idx:06d}",
                "pair": "posneg_abs", "condition": "explicit",
                "prompt": build_explicit(x, mu),
                "x": x, "mu": mu, "z": (x - mu) / SIGMA, "sigma": SIGMA, "seed": -1,
                "low_word": "negative", "high_word": "positive",
            })
            idx += 1
        trials.append({
            "id": f"posneg_abs_zero_shot_{idx:06d}",
            "pair": "posneg_abs", "condition": "zero_shot",
            "prompt": build_zeroshot(x),
            "x": x, "mu": float("nan"), "z": float("nan"), "sigma": SIGMA, "seed": -2,
            "low_word": "negative", "high_word": "positive",
        })
        idx += 1
    return trials


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def extract_and_score(model, tok, trials_sub, high_id, low_id):
    prompts = [t["prompt"] for t in trials_sub]
    layers = get_layers(model)
    captured = {k: [] for k in LAYER_INDICES}
    handles = []
    for k, idx in LAYER_INDICES.items():
        def make_hook(kk):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[kk].append(h.detach())
            return hook
        handles.append(layers[idx].register_forward_hook(make_hook(k)))
    per_layer = {k: [] for k in LAYER_INDICES}
    logit_diffs = []
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            for k in captured: captured[k].clear()
            with torch.no_grad():
                out = model(**enc)
            ld = (out.logits[:, -1, high_id] - out.logits[:, -1, low_id]).float().cpu().numpy()
            logit_diffs.append(ld)
            for k in LAYER_INDICES:
                h = captured[k][0]
                per_layer[k].append(h[:, -1, :].float().cpu().numpy())
    finally:
        for h in handles: h.remove()
    return ({k: np.concatenate(v, axis=0) for k, v in per_layer.items()},
            np.concatenate(logit_diffs))


def relativity_ratio(trials, ld):
    x = np.array([t["x"] for t in trials])
    mu = np.array([t["mu"] for t in trials])
    X = np.column_stack([x, mu, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, ld, rcond=None)
    slope_x, slope_mu, _ = beta
    return float(-slope_mu / slope_x) if abs(slope_x) > 1e-9 else None, float(slope_x), float(slope_mu)


def main():
    trials = build_trials()
    print(f"Built {len(trials)} prompts", flush=True)
    with (OUT / "e4b_posneg_abs_trials.jsonl").open("w") as f:
        for t in trials: f.write(json.dumps(t) + "\n")

    print(f"\nLoading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    high_id = first_token_id(tok, "positive")
    low_id = first_token_id(tok, "negative")
    print(f"  high_id (positive)={high_id}  low_id (negative)={low_id}", flush=True)

    for cond in ("implicit", "explicit", "zero_shot"):
        sub = [t for t in trials if t["condition"] == cond]
        t1 = time.time()
        acts, ld = extract_and_score(model, tok, sub, high_id, low_id)
        print(f"  [{cond}] {len(sub)} prompts, {time.time()-t1:.1f}s  "
              f"ld_mean={ld.mean():+.3f}±{ld.std():.3f}", flush=True)
        for layer, a in acts.items():
            np.savez(OUT / f"e4b_posneg_abs_{cond}_{layer}.npz",
                     activations=a.astype(np.float32),
                     ids=np.array([t["id"] for t in sub]),
                     layer_index=LAYER_INDICES[layer], layer_name=layer)
        with (OUT / f"e4b_posneg_abs_{cond}_logits.jsonl").open("w") as f:
            for t, l in zip(sub, ld):
                f.write(json.dumps({"id": t["id"], "logit_diff": float(l)}) + "\n")

    # Analyze relativity on implicit condition
    imp = [t for t in trials if t["condition"] == "implicit"]
    imp_ld = {json.loads(l)["id"]: json.loads(l)["logit_diff"]
              for l in (OUT / "e4b_posneg_abs_implicit_logits.jsonl").open()}
    imp_ld_ordered = [imp_ld[t["id"]] for t in imp]
    rr, sx, smu = relativity_ratio(imp, imp_ld_ordered)

    # Zero-shot ld at x=-8, 0, +8 for bias check
    zs = [t for t in trials if t["condition"] == "zero_shot"]
    zs_ld = {json.loads(l)["id"]: json.loads(l)["logit_diff"]
             for l in (OUT / "e4b_posneg_abs_zero_shot_logits.jsonl").open()}
    zs_points = [(t["x"], zs_ld[t["id"]]) for t in zs]
    zs_points.sort()

    result = {
        "pair": "posneg_abs",
        "tokens": {"low_word": "negative", "high_word": "positive",
                   "low_id": int(low_id), "high_id": int(high_id)},
        "grid": {"x_values": X_VALUES, "mu_values": MU_VALUES, "sigma": SIGMA,
                 "n_seeds_implicit": N_SEEDS},
        "ld_stats_implicit": {
            "n": len(imp), "mean": float(np.mean(imp_ld_ordered)),
            "std": float(np.std(imp_ld_ordered)),
        },
        "slope_x": sx,
        "slope_mu": smu,
        "relativity_ratio": rr,
        "zero_shot_by_x": zs_points,
    }
    (OUT_ANALYSIS / "posneg_abs_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nposneg_abs relativity_ratio = {rr:.3f}  "
          f"(slope_x={sx:+.4f}, slope_μ={smu:+.4f})")
    print(f"zero-shot by x: {zs_points}")

    # Combine with existing 10 pairs for updated Welch t-test
    try:
        v5 = json.load(open(OUT_ANALYSIS / "exp7_abs_controls.json"))
        all_abs_rr = list(v5["absolute_ratios_proj"].values()) + [rr]
        all_rel_rr = list(v5["relative_ratios_proj"].values())
        from scipy import stats
        t, p = stats.ttest_ind(all_abs_rr, all_rel_rr, equal_var=False)
        print(f"\nUpdated Welch t-test (with posneg_abs as 5th absolute pair):")
        print(f"  relative n={len(all_rel_rr)} mean={np.mean(all_rel_rr):.3f}")
        print(f"  absolute n={len(all_abs_rr)} mean={np.mean(all_abs_rr):.3f}")
        print(f"  Welch t={t:+.3f}  p={p:.4f}")
        result["updated_welch"] = {
            "relative": {"n": len(all_rel_rr), "mean": float(np.mean(all_rel_rr))},
            "absolute": {"n": len(all_abs_rr), "mean": float(np.mean(all_abs_rr))},
            "welch_t": float(t), "welch_p": float(p),
        }
        (OUT_ANALYSIS / "posneg_abs_result.json").write_text(json.dumps(result, indent=2))
    except Exception as e:
        print(f"(skipped extended t-test: {e})")

    # Quick plot: zero-shot logit_diff by x, overlay with slope_x / slope_mu table
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    xs_zs, ys_zs = zip(*zs_points)
    ax.plot(xs_zs, ys_zs, "o-", lw=2)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("x"); ax.set_ylabel("zero-shot logit_diff(positive − negative)")
    ax.set_title(f"Zero-shot bias (5 points)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(["slope_x", "|slope_μ|", "relativity_ratio"],
           [abs(sx), abs(smu), rr], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title(f"posneg_abs  ratio={rr:.3f}")
    ax.grid(alpha=0.3)
    fig.suptitle("positive/negative math — is this a clean absolute control?")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "posneg_abs_panel.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'posneg_abs_panel.png'}")


if __name__ == "__main__":
    main()
