"""v4 adjective-pair extension: test whether contextual relativity generalizes
beyond tall/short to other gradable adjective pairs.

Adjective pairs tested (domain, low, high):
  Relative (7, expected to show contextual relativity):
    - height       short / tall
    - age          young / old
    - weight       light / heavy
    - size         small / big
    - speed        slow  / fast
    - wealth       poor  / rich
    - experience   novice / expert
  Absolute control (1, expected NOT to show relativity):
    - bmi_abs      underweight / obese   (anchored to clinical thresholds)

For each pair we sweep:
  5 target values × 5 context means × 30 seeds = 750 implicit trials
  + 5×5 explicit (deterministic) = 25
  + 5 zero-shot = 5
  = 780 prompts per pair, 6240 total across 8 pairs.

Core checks per pair:
  (a) cell-mean logit_diff monotone in z?  (linearity of z-signal)
  (b) relativity ratio  -slope(mu)/slope(x) ≈ 1 for relative, ≈ 0 for absolute?
  (c) does the zero-shot baseline already saturate the high/low token?

Uses the same E4B model as extract_v4_dense.py. Runs on Vast H100.
Expected wall time: 6,240 prompts at ~50 p/s ≈ 2 min.

Output: results/v4_adjpairs/e4b_{domain}_{condition}_{layer}.npz
        results/v4_adjpairs/e4b_{domain}_{condition}_logits.jsonl
        results/v4_adjpairs/e4b_trials.jsonl
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Shared tokenization helper (co-located with this script). sys.path needs to
# include this directory since tests import this module directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _token_utils import tokens_of_word, first_token_id, report_tokenization  # noqa: E402
# torch / transformers imported lazily inside main() so tests can import this
# module on CPU-only machines without the heavy deps.

# ---- Config ----
MODEL_ID = "google/gemma-4-E4B"
SHORT = "e4b"
LAYER_INDICES = {"mid": 21, "late": 32}
BATCH_SIZE = 16
N_SEEDS_IMPLICIT = 30     # per (x, mu) cell — smaller than v4_dense
N_SEEDS_EXPLICIT = 1       # deterministic

OUT = Path("results/v4_adjpairs")
OUT.mkdir(parents=True, exist_ok=True)


class Pair(NamedTuple):
    name: str              # key for filenames
    domain: str            # English noun ("age", "weight", ...)
    unit: str              # "cm", "years old", "kg", ""
    low_word: str          # "short"
    high_word: str         # "tall"
    target_values: list[float]   # 5 target x values
    mu_values: list[float]       # 5 context means
    sigma: float           # if name in LOG_SPACE_PAIRS: multiplicative factor; else additive stdev
    format_prompt_implicit: str   # template; placeholders {items} and {x_unit}
    format_prompt_explicit: str
    format_prompt_zero: str
    value_label: str       # "tall", "old", "heavy" etc. — how model is asked


# PLANNING.md specifies wealth in log-space: z = (log(x) - log(μ)) / log(σ_factor).
# For log-space pairs, `sigma` on the Pair is interpreted as a multiplicative
# factor (e.g. 2.0 ≈ "spread is doubling/halving"), context samples are drawn
# from a log-normal, and the z-score uses natural logarithms.
LOG_SPACE_PAIRS: set[str] = {"wealth"}


def compute_z(pair: "Pair", x: float, mu: float) -> float:
    """Domain-appropriate z-score. Log-space for wealth per PLANNING.md §Domain 2."""
    if pair.name in LOG_SPACE_PAIRS:
        if x <= 0 or mu <= 0 or pair.sigma <= 1.0:
            raise ValueError(f"log-space pair {pair.name} needs x>0, mu>0, sigma>1")
        return (math.log(x) - math.log(mu)) / math.log(pair.sigma)
    return (x - mu) / pair.sigma


PAIRS: list[Pair] = [
    Pair(
        name="height",
        domain="height",
        unit="cm",
        low_word="short",
        high_word="tall",
        target_values=[150.0, 160.0, 165.0, 170.0, 180.0],
        mu_values=[145.0, 155.0, 165.0, 175.0, 185.0],
        sigma=10.0,
        format_prompt_implicit=(
            "{items}\n"
            "Person {n_last}: {x_str} cm. This person is"
        ),
        format_prompt_explicit=(
            "In a group where most people's heights cluster around {mu_str} cm "
            "(give or take {sigma_str} cm), a person who is {x_str} cm is"
        ),
        format_prompt_zero="A person who is {x_str} cm is",
        value_label="height",
    ),
    Pair(
        name="age",
        domain="age",
        unit="years old",
        low_word="young",
        high_word="old",
        target_values=[20.0, 30.0, 40.0, 50.0, 60.0],
        mu_values=[25.0, 35.0, 40.0, 45.0, 55.0],
        sigma=5.0,
        format_prompt_implicit=(
            "{items}\n"
            "Person {n_last}: {x_str} years old. This person is"
        ),
        format_prompt_explicit=(
            "In a group where most people's ages cluster around {mu_str} years "
            "(give or take {sigma_str} years), a person who is {x_str} years old is"
        ),
        format_prompt_zero="A person who is {x_str} years old is",
        value_label="age",
    ),
    Pair(
        name="weight",
        domain="weight",
        unit="kg",
        low_word="light",
        high_word="heavy",
        target_values=[50.0, 65.0, 75.0, 85.0, 100.0],
        mu_values=[55.0, 65.0, 75.0, 85.0, 95.0],
        sigma=8.0,
        format_prompt_implicit=(
            "{items}\n"
            "Person {n_last}: {x_str} kg. This person is"
        ),
        format_prompt_explicit=(
            "In a group where most people's weights cluster around {mu_str} kg "
            "(give or take {sigma_str} kg), a person who weighs {x_str} kg is"
        ),
        format_prompt_zero="A person who weighs {x_str} kg is",
        value_label="weight",
    ),
    Pair(
        name="size",
        domain="size",
        unit="cm diameter",
        low_word="small",
        high_word="big",
        target_values=[5.0, 15.0, 25.0, 40.0, 60.0],
        mu_values=[10.0, 20.0, 30.0, 40.0, 55.0],
        sigma=6.0,
        format_prompt_implicit=(
            "{items}\n"
            "Object {n_last}: {x_str} cm across. This object is"
        ),
        format_prompt_explicit=(
            "In a group of objects whose sizes cluster around {mu_str} cm "
            "(give or take {sigma_str} cm), an object that is {x_str} cm across is"
        ),
        format_prompt_zero="An object that is {x_str} cm across is",
        value_label="size",
    ),
    Pair(
        name="speed",
        domain="speed",
        unit="km/h",
        low_word="slow",
        high_word="fast",
        target_values=[20.0, 50.0, 80.0, 110.0, 150.0],
        mu_values=[30.0, 60.0, 80.0, 100.0, 140.0],
        sigma=15.0,
        format_prompt_implicit=(
            "{items}\n"
            "Vehicle {n_last}: {x_str} km/h. This vehicle is"
        ),
        format_prompt_explicit=(
            "In a group of vehicles whose speeds cluster around {mu_str} km/h "
            "(give or take {sigma_str} km/h), a vehicle going {x_str} km/h is"
        ),
        format_prompt_zero="A vehicle going {x_str} km/h is",
        value_label="speed",
    ),
    Pair(
        # Wealth uses LOG-SPACE z per PLANNING.md §Domain 2:
        #   z = (log(x) - log(μ)) / log(σ_factor), σ_factor = 2.0.
        # target_values & mu_values are roughly log-spaced (factor ~2-3 apart).
        # The context sampler uses a log-normal distribution (see sample_context).
        name="wealth",
        domain="wealth",
        unit="USD annual income",
        low_word="poor",
        high_word="rich",
        target_values=[20000.0, 50000.0, 100000.0, 250000.0, 600000.0],
        mu_values=[30000.0, 70000.0, 150000.0, 400000.0, 1000000.0],
        sigma=2.0,  # multiplicative factor — in log-space this is log(2) ≈ 0.693
        format_prompt_implicit=(
            "{items}\n"
            "Person {n_last} earns ${x_str}/year. This person is"
        ),
        format_prompt_explicit=(
            "In a group where most people earn around ${mu_str}/year "
            "(incomes spread by roughly a factor of {sigma_str}), "
            "a person earning ${x_str}/year is"
        ),
        format_prompt_zero="A person earning ${x_str}/year is",
        value_label="wealth",
    ),
    Pair(
        name="experience",
        domain="experience",
        unit="years experience",
        low_word="novice",
        high_word="expert",
        target_values=[1.0, 5.0, 10.0, 15.0, 25.0],
        mu_values=[2.0, 7.0, 12.0, 18.0, 25.0],
        sigma=4.0,
        format_prompt_implicit=(
            "{items}\n"
            "Worker {n_last}: {x_str} years experience. This worker is"
        ),
        format_prompt_explicit=(
            "In a team where most have around {mu_str} years experience "
            "(give or take {sigma_str}), a worker with {x_str} years experience is"
        ),
        format_prompt_zero="A worker with {x_str} years experience is",
        value_label="experience",
    ),
    # --- ABSOLUTE ADJECTIVE CONTROL ---
    # BMI with "thin"/"obese" — "obese" is clinically anchored at BMI ≥ 30,
    # "thin" is paired to keep both words single-token under the Gemma 4
    # tokenizer (the clinical term "underweight" splits into 3+ subtokens,
    # which biases a next-token logit lookup). The relativity ratio depends
    # mostly on the threshold-anchored HIGH side, so "thin" vs "obese"
    # still tests the absolute-adjective prediction even though "thin"
    # carries some residual context dependence.
    # A startup tokenization check (see main()) asserts both words are
    # single-token on the actual tokenizer at extract time.
    Pair(
        name="bmi_abs",
        domain="BMI",
        unit="BMI",
        low_word="thin",
        high_word="obese",
        target_values=[17.0, 22.0, 27.0, 32.0, 38.0],
        mu_values=[20.0, 25.0, 28.0, 30.0, 33.0],
        sigma=3.0,
        format_prompt_implicit=(
            "{items}\n"
            "Person {n_last}: BMI {x_str}. This person is"
        ),
        format_prompt_explicit=(
            "In a group where most people's BMIs cluster around {mu_str} "
            "(give or take {sigma_str}), a person with BMI {x_str} is"
        ),
        format_prompt_zero="A person with BMI {x_str} is",
        value_label="bmi",
    ),
]


def fmt_num(v: float) -> str:
    """Render a number without trailing .0 for integer-valued floats."""
    if v == int(v):
        return str(int(v))
    return f"{v:.1f}"


def sample_context(mu: float, sigma: float, seed: int, n: int = 15,
                   low: float = None, high: float = None,
                   log_space: bool = False) -> list[float]:
    """Sample `n` values from a per-person distribution around `mu`.

    log_space=False: v ~ Normal(mu, sigma).
    log_space=True:  v ~ LogNormal(log(mu), log(sigma))
                     i.e. log(v) ~ Normal(log(mu), log(sigma_factor)).
                     sigma is the multiplicative factor; for sigma=2.0 about
                     68% of samples fall in [mu/2, mu*2].
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if log_space:
            log_v = rng.gauss(math.log(mu), math.log(sigma))
            v = math.exp(log_v)
        else:
            v = rng.gauss(mu, sigma)
        if low is not None:
            v = max(low, v)
        if high is not None:
            v = min(high, v)
        out.append(round(v, 1) if abs(v) < 100 else round(v))
    return out


def build_implicit_items(pair: Pair, mu: float, seed: int, n: int = 15) -> list[str]:
    """Build the 'items' block for an implicit prompt."""
    low = pair.target_values[0] * 0.4
    high = pair.target_values[-1] * 2.5
    sample = sample_context(mu, pair.sigma, seed, n, low, high,
                            log_space=(pair.name in LOG_SPACE_PAIRS))
    if pair.name == "height":
        return [f"Person {i+1}: {int(v)} cm" for i, v in enumerate(sample)]
    elif pair.name == "age":
        return [f"Person {i+1}: {int(v)} years old" for i, v in enumerate(sample)]
    elif pair.name == "weight":
        return [f"Person {i+1}: {int(v)} kg" for i, v in enumerate(sample)]
    elif pair.name == "size":
        return [f"Object {i+1}: {int(v)} cm across" for i, v in enumerate(sample)]
    elif pair.name == "speed":
        return [f"Vehicle {i+1}: {int(v)} km/h" for i, v in enumerate(sample)]
    elif pair.name == "wealth":
        return [f"Person {i+1} earns ${int(v)}/year" for i, v in enumerate(sample)]
    elif pair.name == "experience":
        return [f"Worker {i+1}: {int(v)} years experience" for i, v in enumerate(sample)]
    elif pair.name == "bmi_abs":
        # BMI is a rational number typically shown with one decimal. Keep .1 precision.
        return [f"Person {i+1}: BMI {v:.1f}" for i, v in enumerate(sample)]
    else:
        return [f"Item {i+1}: {v}" for i, v in enumerate(sample)]


def make_implicit_prompt(pair: Pair, x: float, mu: float, seed: int) -> str:
    items = build_implicit_items(pair, mu, seed)
    items_block = "\n".join(items)
    return pair.format_prompt_implicit.format(
        items=items_block,
        n_last=len(items) + 1,
        x_str=fmt_num(x),
    )


def make_explicit_prompt(pair: Pair, x: float, mu: float) -> str:
    return pair.format_prompt_explicit.format(
        mu_str=fmt_num(mu),
        sigma_str=fmt_num(pair.sigma),
        x_str=fmt_num(x),
    )


def make_zero_shot_prompt(pair: Pair, x: float) -> str:
    return pair.format_prompt_zero.format(x_str=fmt_num(x))


def generate_all_prompts() -> list[dict]:
    """Generate prompts for every pair × condition × cell × seed."""
    trials = []
    idx = 0
    for pair in PAIRS:
        for x in pair.target_values:
            for mu in pair.mu_values:
                z = compute_z(pair, x, mu)
                for s in range(N_SEEDS_IMPLICIT):
                    trials.append({
                        "id": f"{pair.name}_implicit_{idx:06d}",
                        "pair": pair.name,
                        "condition": "implicit",
                        "prompt": make_implicit_prompt(pair, x, mu, s),
                        "x": x, "mu": mu, "z": z, "sigma": pair.sigma, "seed": s,
                        "low_word": pair.low_word, "high_word": pair.high_word,
                    })
                    idx += 1
                trials.append({
                    "id": f"{pair.name}_explicit_{idx:06d}",
                    "pair": pair.name,
                    "condition": "explicit",
                    "prompt": make_explicit_prompt(pair, x, mu),
                    "x": x, "mu": mu, "z": z, "sigma": pair.sigma, "seed": -1,
                    "low_word": pair.low_word, "high_word": pair.high_word,
                })
                idx += 1
        for x in pair.target_values:
            trials.append({
                "id": f"{pair.name}_zeroshot_{idx:06d}",
                "pair": pair.name,
                "condition": "zero_shot",
                "prompt": make_zero_shot_prompt(pair, x),
                "x": x, "mu": 0.0, "z": 0.0, "sigma": pair.sigma, "seed": -1,
                "low_word": pair.low_word, "high_word": pair.high_word,
            })
            idx += 1
    return trials


# ---- Model helpers ----

def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError(f"no layers on {type(model).__name__}")


def extract_batch(model, tokenizer, prompts, layer_indices, token_ids):
    """Run a batch. Returns acts per layer + logits per prompt at last position.

    token_ids: dict mapping label -> token_id to score (multiple labels per prompt).
    """
    import torch
    layers_mod = get_layers(model)
    captured = {k: [] for k in layer_indices}
    handles = []

    def make_hook(key):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[key].append(h.detach())
        return hook

    for key, idx in layer_indices.items():
        handles.append(layers_mod[idx].register_forward_hook(make_hook(key)))

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=1024).to(model.device)

    for k in captured:
        captured[k].clear()

    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    acts = {}
    for k in layer_indices:
        h = captured[k][0]
        last = h[:, -1, :].float().cpu().numpy()
        acts[k] = last

    logits = outputs.logits[:, -1, :]

    logit_info = []
    for i in range(len(prompts)):
        row_logits = logits[i]
        per_label = {}
        for label, tid in token_ids.items():
            per_label[label] = float(row_logits[tid].cpu())

        top5_vals, top5_ids = torch.topk(row_logits, 5)
        top5_tokens = tokenizer.convert_ids_to_tokens(top5_ids.cpu().tolist())
        logit_info.append({
            "per_label": per_label,
            "top5_tokens": top5_tokens,
            "top5_logits": [round(float(v), 3) for v in top5_vals.cpu().tolist()],
        })

    for h in handles:
        h.remove()
    return acts, logit_info


def get_first_token(tokenizer, word: str) -> int:
    """Backward-compat alias for scripts.vast_remote._token_utils.first_token_id."""
    return first_token_id(tokenizer, word)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Generating prompts for {len(PAIRS)} adjective pairs...", flush=True)
    trials = generate_all_prompts()
    print(f"  Total: {len(trials)} prompts", flush=True)
    for pair in PAIRS:
        n = sum(1 for t in trials if t["pair"] == pair.name)
        print(f"    {pair.name:12s}: {n} (low={pair.low_word}, high={pair.high_word})", flush=True)

    meta_path = OUT / f"{SHORT}_trials.jsonl"
    with meta_path.open("w") as f:
        for t in trials:
            row = {k: v for k, v in t.items() if k != "prompt"}
            f.write(json.dumps(row) + "\n")
    print(f"  Metadata: {meta_path}", flush=True)

    print(f"\nLoading {MODEL_ID}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # Build per-pair token mapping. Startup tokenization check:
    # every scored adjective MUST be single-token under the best spacing
    # variant, otherwise logit_diff lookup becomes a sub-token lookup and
    # the relativity ratio for that pair is polluted.
    all_labels = sorted({w for p in PAIRS for w in (p.low_word, p.high_word)})
    print("\n  Tokenization check:", flush=True)
    token_table = report_tokenization(tokenizer, all_labels)
    multi = {w: ids for w, ids in token_table.items() if len(ids) > 1}
    if multi:
        print(f"\n[FATAL] Multi-token adjectives detected: {multi}", flush=True)
        print("Either substitute single-token alternatives, or extend "
              "extract_batch to score log P(word | prompt) autoregressively.",
              flush=True)
        raise SystemExit(2)
    label_to_id = {w: ids[0] for w, ids in token_table.items()}

    # Process per-pair × per-condition
    for pair in PAIRS:
        for condition in ["implicit", "explicit", "zero_shot"]:
            cond_trials = [t for t in trials if t["pair"] == pair.name and t["condition"] == condition]
            if not cond_trials:
                continue
            print(f"\n=== {pair.name}/{condition} ({len(cond_trials)}) ===", flush=True)

            prompts = [t["prompt"] for t in cond_trials]
            ids = [t["id"] for t in cond_trials]
            # Just score the two pair-specific labels + always a few common ones
            token_ids = {
                pair.low_word: label_to_id[pair.low_word],
                pair.high_word: label_to_id[pair.high_word],
            }

            all_acts = {k: [] for k in LAYER_INDICES}
            all_info = []
            t0 = time.time()
            for i in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[i:i+BATCH_SIZE]
                acts, info = extract_batch(model, tokenizer, batch, LAYER_INDICES, token_ids)
                for k in LAYER_INDICES:
                    all_acts[k].append(acts[k])
                all_info.extend(info)
            print(f"  {len(prompts)} in {time.time()-t0:.1f}s", flush=True)

            for layer_key in LAYER_INDICES:
                stacked = np.concatenate(all_acts[layer_key], axis=0)
                out_path = OUT / f"{SHORT}_{pair.name}_{condition}_{layer_key}.npz"
                np.savez(out_path,
                         activations=stacked.astype(np.float32),
                         ids=np.array(ids),
                         layer_index=LAYER_INDICES[layer_key])

            logit_path = OUT / f"{SHORT}_{pair.name}_{condition}_logits.jsonl"
            with logit_path.open("w") as f:
                for tid, inf in zip(ids, all_info):
                    row = {
                        "id": tid,
                        "pair": pair.name,
                        "low_word": pair.low_word,
                        "high_word": pair.high_word,
                        "logit_low": inf["per_label"][pair.low_word],
                        "logit_high": inf["per_label"][pair.high_word],
                        "logit_diff": inf["per_label"][pair.high_word] - inf["per_label"][pair.low_word],
                        "top5_tokens": inf["top5_tokens"],
                        "top5_logits": inf["top5_logits"],
                    }
                    f.write(json.dumps(row) + "\n")
            print(f"  Saved {logit_path.name}", flush=True)

    # ---- Quick per-pair summary ----
    print(f"\n{'='*70}")
    print("QUICK ANALYSIS: per-pair implicit cell means")
    print(f"{'='*70}")
    for pair in PAIRS:
        lp = OUT / f"{SHORT}_{pair.name}_implicit_logits.jsonl"
        if not lp.exists():
            continue
        rows = [json.loads(l) for l in lp.open()]
        # Map id -> trial
        trial_lookup = {t["id"]: t for t in trials}
        cells = {}
        for r in rows:
            tr = trial_lookup[r["id"]]
            cells.setdefault((tr["x"], tr["mu"]), []).append(r["logit_diff"])
        print(f"\n  {pair.name}  ({pair.low_word}/{pair.high_word})")
        print(f"    {'x':>10}   " + "  ".join(f"mu={fmt_num(m):>5}" for m in pair.mu_values))
        for x in pair.target_values:
            row_str = f"    {fmt_num(x):>10}   "
            for m in pair.mu_values:
                vals = cells.get((x, m), [])
                if vals:
                    row_str += f"{np.mean(vals):+8.2f}  "
                else:
                    row_str += "    N/A  "
            print(row_str)

    print(f"\nDONE. Outputs in {OUT}/", flush=True)


if __name__ == "__main__":
    main()
