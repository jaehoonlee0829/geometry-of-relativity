"""Exp 7: add 3 more absolute-adjective controls.

Pairs (all single-token on E4B tokenizer, verified):
  - temp_abs:     cold / hot        (temperature °C, anchors: water freezes at 0, boils at 100)
  - legal_abs:    minor / adult      (age years, anchor: 18)
  - grade_abs:    failing / passing  (exam score %, anchor: 60)

Uses same grid as the existing adjpair extraction: 5 x × 5 μ × 30 seeds = 750
implicit + 25 explicit + 5 zero-shot = 780 per pair. Extracts mid+late activations
and logit_diff(high−low). With n=4 absolute pairs (incl. bmi_abs) and 7 relative
pairs, we can do a t-test on relativity_ratio.

Writes:
  results/v4_abs_controls/e4b_{pair}_{cond}_{layer}.npz
  results/v4_abs_controls/e4b_{pair}_{cond}_logits.jsonl
  results/v4_abs_controls/e4b_trials.jsonl
  results/v4_adjpairs_analysis/exp7_abs_controls.json
  figures/v4_adjpairs/relativity_ratio_absolute_vs_relative.png
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402

MODEL_ID = "google/gemma-4-E4B"
LAYER_INDICES = {"mid": 21, "late": 32}
BATCH_SIZE = 16
N_SEEDS = 30

OUT = REPO / "results" / "v4_abs_controls"
OUT.mkdir(parents=True, exist_ok=True)
OUT_ANALYSIS = REPO / "results" / "v4_adjpairs_analysis"
OUT_FIG = REPO / "figures" / "v4_adjpairs"


class AbsPair(NamedTuple):
    name: str
    low_word: str
    high_word: str
    x_values: list[float]
    mu_values: list[float]
    sigma: float
    unit: str
    item_template: str              # "Liquid {i}: {v:g}°C"
    implicit_template: str          # final prompt with {items} and {x_str}
    explicit_template: str
    zero_shot_template: str


ABS_PAIRS: list[AbsPair] = [
    AbsPair(
        name="temp_abs",
        low_word="cold",
        high_word="hot",
        x_values=[10, 30, 50, 70, 90],
        mu_values=[20, 40, 60, 80, 95],
        sigma=15.0,
        unit="°C",
        item_template="Liquid {i}: {v:g}°C",
        implicit_template=(
            "{items}\n"
            "Liquid {n_last}: {x_str}°C. This liquid is"
        ),
        explicit_template=(
            "In a collection of liquids whose temperatures average around "
            "{mu_str}°C (give or take {sigma_str}°C), a liquid at {x_str}°C is"
        ),
        zero_shot_template="A liquid at {x_str}°C is",
    ),
    AbsPair(
        name="legal_abs",
        low_word="minor",
        high_word="adult",
        x_values=[12, 16, 18, 22, 30],
        mu_values=[15, 18, 21, 25, 35],
        sigma=4.0,
        unit="years",
        item_template="Person {i}: {v:g} years old",
        implicit_template=(
            "{items}\n"
            "Person {n_last}: {x_str} years old. This person is a"
        ),
        explicit_template=(
            "In a group where people's ages cluster around {mu_str} years "
            "(give or take {sigma_str}), a person who is {x_str} years old is a"
        ),
        zero_shot_template="A person who is {x_str} years old is a",
    ),
    AbsPair(
        name="grade_abs",
        low_word="failing",
        high_word="passing",
        x_values=[30, 50, 60, 75, 95],
        mu_values=[40, 55, 65, 75, 85],
        sigma=12.0,
        unit="%",
        item_template="Student {i}: {v:g}%",
        implicit_template=(
            "{items}\n"
            "Student {n_last}: {x_str}%. This student is"
        ),
        explicit_template=(
            "In a class where scores average around {mu_str}% "
            "(give or take {sigma_str}%), a student with a score of {x_str}% is"
        ),
        zero_shot_template="A student with a score of {x_str}% is",
    ),
]


def fmt_num(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:.1f}"


def sample_context(pair: AbsPair, mu: float, seed: int, n: int = 15) -> list[float]:
    rng = np.random.default_rng(seed + hash(pair.name) % 1000)
    low = pair.x_values[0] * 0.4 if pair.x_values[0] > 0 else pair.x_values[0] - pair.sigma * 3
    high = pair.x_values[-1] * 2.0 if pair.x_values[-1] > 0 else pair.x_values[-1] + pair.sigma * 3
    out = []
    while len(out) < n:
        v = rng.normal(mu, pair.sigma)
        if low <= v <= high:
            out.append(round(v, 1) if abs(v) < 100 else round(v))
    return out


def build_implicit_prompt(pair: AbsPair, x: float, mu: float, seed: int) -> str:
    sample = sample_context(pair, mu, seed)
    items = [pair.item_template.format(i=i + 1, v=v) for i, v in enumerate(sample)]
    return pair.implicit_template.format(items="\n".join(items),
                                         n_last=len(items) + 1,
                                         x_str=fmt_num(x))


def build_explicit_prompt(pair: AbsPair, x: float, mu: float) -> str:
    return pair.explicit_template.format(mu_str=fmt_num(mu), sigma_str=fmt_num(pair.sigma),
                                         x_str=fmt_num(x))


def build_zero_shot_prompt(pair: AbsPair, x: float) -> str:
    return pair.zero_shot_template.format(x_str=fmt_num(x))


def build_all_trials() -> list[dict]:
    trials = []
    idx = 0
    for pair in ABS_PAIRS:
        for x in pair.x_values:
            for mu in pair.mu_values:
                z = (x - mu) / pair.sigma
                for s in range(N_SEEDS):
                    trials.append({
                        "id": f"{pair.name}_implicit_{idx:06d}",
                        "pair": pair.name, "condition": "implicit",
                        "prompt": build_implicit_prompt(pair, x, mu, s),
                        "x": float(x), "mu": float(mu), "z": float(z),
                        "sigma": pair.sigma, "seed": s,
                        "low_word": pair.low_word, "high_word": pair.high_word,
                    })
                    idx += 1
                trials.append({
                    "id": f"{pair.name}_explicit_{idx:06d}",
                    "pair": pair.name, "condition": "explicit",
                    "prompt": build_explicit_prompt(pair, x, mu),
                    "x": float(x), "mu": float(mu), "z": float((x - mu) / pair.sigma),
                    "sigma": pair.sigma, "seed": -1,
                    "low_word": pair.low_word, "high_word": pair.high_word,
                })
                idx += 1
            trials.append({
                "id": f"{pair.name}_zero_shot_{idx:06d}",
                "pair": pair.name, "condition": "zero_shot",
                "prompt": build_zero_shot_prompt(pair, x),
                "x": float(x), "mu": float("nan"), "z": float("nan"),
                "sigma": pair.sigma, "seed": -2,
                "low_word": pair.low_word, "high_word": pair.high_word,
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


def extract_and_score(model, tok, trials_sub: list[dict], high_id: int, low_id: int):
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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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


def analyze_pair(trials_by_id, ld_by_id, acts_late, pair_name: str) -> dict:
    """Compute relativity_ratio via the x+μ regression → slope_x / (slope_x + |slope_μ|)."""
    rows = [trials_by_id[i] for i in acts_late.keys() if trials_by_id[i]["condition"] == "implicit"]
    # Actually use trial order from the activations dir
    ids_ordered = list(acts_late.keys())
    rows = [trials_by_id[i] for i in ids_ordered if trials_by_id[i]["condition"] == "implicit"]
    ld = [ld_by_id[i] for i in ids_ordered if trials_by_id[i]["condition"] == "implicit"]
    x = np.array([r["x"] for r in rows])
    mu = np.array([r["mu"] for r in rows])
    z = np.array([r["z"] for r in rows])
    ld = np.array(ld)
    # Regression ld ~ x + μ
    X = np.column_stack([x, mu, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, ld, rcond=None)
    slope_x, slope_mu, _ = beta
    # Regression ld ~ z alone
    slope_z_alone = np.polyfit(z, ld, 1)[0]
    # Relativity ratio: higher means more x-driven (absolute); lower means more z-driven (relative)
    relativity_ratio = abs(slope_x) / (abs(slope_x) + abs(slope_mu) + 1e-12)
    return {
        "slope_x": float(slope_x),
        "slope_mu": float(slope_mu),
        "slope_z_alone": float(slope_z_alone),
        "relativity_ratio": float(relativity_ratio),
        "n_implicit": int(len(ld)),
    }


def main():
    trials = build_all_trials()
    print(f"Built {len(trials)} trials across {len(ABS_PAIRS)} abs pairs", flush=True)
    for p in ABS_PAIRS:
        n = sum(1 for t in trials if t["pair"] == p.name)
        print(f"  {p.name}: {n} trials ({p.low_word}/{p.high_word})")

    # Save trials
    with (OUT / "e4b_trials.jsonl").open("w") as f:
        for t in trials: f.write(json.dumps(t) + "\n")

    print(f"\nLoading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    per_pair_analysis = {}
    trials_by_id = {t["id"]: t for t in trials}
    for pair in ABS_PAIRS:
        print(f"\n=== {pair.name} ({pair.low_word}/{pair.high_word}) ===", flush=True)
        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)
        for cond in ("implicit", "explicit", "zero_shot"):
            sub = [t for t in trials if t["pair"] == pair.name and t["condition"] == cond]
            if not sub:
                continue
            t1 = time.time()
            acts, ld = extract_and_score(model, tok, sub, high_id, low_id)
            print(f"  [{cond}] {len(sub)} prompts, {time.time()-t1:.1f}s  "
                  f"ld_mean={ld.mean():+.3f}±{ld.std():.3f}", flush=True)
            ids_arr = np.array([t["id"] for t in sub])
            for layer, a in acts.items():
                np.savez(OUT / f"e4b_{pair.name}_{cond}_{layer}.npz",
                         activations=a.astype(np.float32),
                         ids=ids_arr,
                         layer_index=LAYER_INDICES[layer],
                         layer_name=layer)
            with (OUT / f"e4b_{pair.name}_{cond}_logits.jsonl").open("w") as f:
                for t, l in zip(sub, ld):
                    f.write(json.dumps({"id": t["id"], "logit_diff": float(l)}) + "\n")

        # Analyze: compute relativity_ratio on implicit condition
        imp_logits = {}
        for rec in map(json.loads, (OUT / f"e4b_{pair.name}_implicit_logits.jsonl").open()):
            imp_logits[rec["id"]] = rec["logit_diff"]
        imp_npz = np.load(OUT / f"e4b_{pair.name}_implicit_late.npz", allow_pickle=True)
        imp_acts_dict = {str(i): True for i in imp_npz["ids"]}
        analysis = analyze_pair(trials_by_id, imp_logits, imp_acts_dict, pair.name)
        per_pair_analysis[pair.name] = analysis
        print(f"  relativity_ratio = {analysis['relativity_ratio']:.3f}  "
              f"(slope_x={analysis['slope_x']:+.4f}, slope_μ={analysis['slope_mu']:+.4f})", flush=True)

    # Combine with existing 8 pairs' relativity_ratios from v4_adjpairs summary_main
    existing = json.load(open(OUT_ANALYSIS / "summary.json"))
    existing_rr = {r["pair"]: r["relativity_ratio"] for r in existing if r["layer"] == "late"}

    # Classification: relative = (height, age, experience, size, speed, wealth, weight)
    # Absolute = (bmi_abs + 3 new)
    RELATIVE = ["height", "age", "experience", "size", "speed", "wealth", "weight"]
    ABSOLUTE_OLD = ["bmi_abs"]
    ABSOLUTE_NEW = [p.name for p in ABS_PAIRS]
    all_rel = [existing_rr[p] for p in RELATIVE if p in existing_rr]
    all_abs = [existing_rr[p] for p in ABSOLUTE_OLD if p in existing_rr] + \
              [per_pair_analysis[p]["relativity_ratio"] for p in ABSOLUTE_NEW]

    # Welch t-test
    t_stat, p_val = stats.ttest_ind(all_abs, all_rel, equal_var=False)
    final = {
        "per_pair_new": per_pair_analysis,
        "relative_pairs": RELATIVE,
        "relative_relativity_ratios": all_rel,
        "absolute_pairs": ABSOLUTE_OLD + ABSOLUTE_NEW,
        "absolute_relativity_ratios": all_abs,
        "welch_t": float(t_stat),
        "welch_p_twosided": float(p_val),
        "relative_mean": float(np.mean(all_rel)),
        "relative_std": float(np.std(all_rel, ddof=1)),
        "absolute_mean": float(np.mean(all_abs)),
        "absolute_std": float(np.std(all_abs, ddof=1)),
        "note": (
            "relativity_ratio = |slope_x| / (|slope_x| + |slope_μ|). "
            "Absolute pairs should have ratio→1 (pure-x); relative should have ratio<<1 "
            "(μ matters as much as x)."
        ),
    }
    (OUT_ANALYSIS / "exp7_abs_controls.json").write_text(json.dumps(final, indent=2))
    print(f"\n=== STAT TEST ===")
    print(f"  relative  (n={len(all_rel)}): ratio mean={np.mean(all_rel):.3f}±{np.std(all_rel, ddof=1):.3f}")
    print(f"  absolute  (n={len(all_abs)}): ratio mean={np.mean(all_abs):.3f}±{np.std(all_abs, ddof=1):.3f}")
    print(f"  Welch t = {t_stat:+.3f}   p = {p_val:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    xs1 = np.full(len(all_rel), 0) + np.random.default_rng(0).normal(0, 0.05, len(all_rel))
    xs2 = np.full(len(all_abs), 1) + np.random.default_rng(0).normal(0, 0.05, len(all_abs))
    ax.scatter(xs1, all_rel, s=60, alpha=0.8, label=f"relative (n={len(all_rel)})")
    ax.scatter(xs2, all_abs, s=60, alpha=0.8, color="red", label=f"absolute (n={len(all_abs)})")
    for i, p in enumerate(RELATIVE):
        if p in existing_rr:
            ax.annotate(p, (xs1[i], all_rel[i]), fontsize=8, xytext=(5, 0), textcoords="offset points")
    all_abs_names = ABSOLUTE_OLD + ABSOLUTE_NEW
    for i, p in enumerate(all_abs_names):
        ax.annotate(p, (xs2[i], all_abs[i]), fontsize=8, xytext=(5, 0), textcoords="offset points")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["relative", "absolute"])
    ax.set_ylabel("relativity_ratio = |slope_x| / (|slope_x| + |slope_μ|)")
    ax.set_title(f"Relativity ratio: relative vs absolute adjective pairs\nWelch t={t_stat:+.2f}, p={p_val:.3g}")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "relativity_ratio_absolute_vs_relative.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'relativity_ratio_absolute_vs_relative.png'}")


if __name__ == "__main__":
    main()
