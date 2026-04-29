"""v12.1 lexical disentanglement follow-up.

Implements the minimum viable v12.1 plan:

1. Token-position lexical capture at Gemma 2 9B L33.
2. Lexical-subspace residualization of `primal_z` and steering.

Outputs:
  results/v12_1/token_position_lexical_capture.json
  results/v12_1/lexical_subspace_residualization.json
  figures/v12_1/token_position_lexical_cosines.png
  figures/v12_1/lexical_subspace_residualization_steering.png
  figures/v12_1/lexical_subspace_fraction_removed.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

MODEL_ID = "google/gemma-2-9b"
MODEL_SHORT = "gemma2-9b"
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
DEFAULT_LAYER = 33

RESULTS = REPO / "results" / "v12_1"
FIGS = REPO / "figures" / "v12_1"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SYNONYMS = {
    "height": (["tall", "high", "large"], ["short", "low", "small"]),
    "age": (["old", "elderly", "aged"], ["young", "new", "youthful"]),
    "weight": (["heavy", "weighty", "large"], ["light", "thin", "small"]),
    "size": (["big", "large", "huge"], ["small", "little", "tiny"]),
    "speed": (["fast", "quick", "rapid"], ["slow", "sluggish", "unhurried"]),
    "wealth": (["rich", "wealthy", "affluent"], ["poor", "low-income", "broke"]),
    "experience": (["expert", "experienced", "veteran"], ["novice", "new", "inexperienced"]),
    "bmi_abs": (["obese", "heavy", "large"], ["thin", "lean", "light"]),
}

DOMAIN_TERMS = {
    "height": "height",
    "age": "age",
    "weight": "weight",
    "size": "size",
    "speed": "speed",
    "wealth": "wealth",
    "experience": "experience",
    "bmi_abs": "BMI",
}


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("decoder layers not found")


def load_npz(pair: str):
    return np.load(REPO / "results" / "v11" / MODEL_SHORT / pair / f"{MODEL_SHORT}_{pair}_v11_residuals.npz")


def load_meta(pair: str) -> dict:
    return json.loads(
        (REPO / "results" / "v11" / MODEL_SHORT / pair / f"{MODEL_SHORT}_{pair}_v11_meta.json").read_text()
    )


def load_trials(pair: str) -> list[dict]:
    p = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    return [json.loads(line) for line in p.open()]


def seed0_trials(pair: str, max_n: int) -> list[dict]:
    out = []
    seen = set()
    for t in load_trials(pair):
        if t.get("cell_seed") != 0:
            continue
        key = (round(float(t["x"]), 4), round(float(t["z"]), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    if len(out) > max_n:
        idx = np.linspace(0, len(out) - 1, max_n).round().astype(int)
        out = [out[i] for i in idx]
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float64) / n if n > 1e-12 else v.astype(np.float64)


def primal_z(pair: str, layer: int) -> np.ndarray:
    d = load_npz(pair)
    h = d["activations"][:, layer, :].astype(np.float64)
    z = d["z"].astype(np.float64)
    return h[z > 1.0].mean(0) - h[z < -1.0].mean(0)


def primal_x(pair: str, layer: int) -> np.ndarray:
    d = load_npz(pair)
    h = d["activations"][:, layer, :].astype(np.float64)
    x = d["x"].astype(np.float64)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return h[x >= q75].mean(0) - h[x <= q25].mean(0)


def lexical_prompt_rows(pair: str) -> list[dict]:
    meta = load_meta(pair)
    high = meta["high_word"]
    low = meta["low_word"]
    high_syn, low_syn = SYNONYMS[pair]
    domain = DOMAIN_TERMS[pair]
    rows: list[dict] = []

    def add(kind: str, polarity: str, prompt: str, phrase: str) -> None:
        start = prompt.index(phrase)
        rows.append(
            {
                "pair": pair,
                "kind": kind,
                "polarity": polarity,
                "prompt": prompt,
                "target_phrase": phrase,
                "char_start": start,
                "char_end": start + len(phrase),
            }
        )

    for polarity, word in [("high", high), ("low", low)]:
        add("word", polarity, f"The adjective is {word}", word)
        add("sentence", polarity, f"This person is {word}.", word)
        add("sentence", polarity, f"A described case is {word}.", word)

    for polarity, words in [("high", high_syn), ("low", low_syn)]:
        for word in words:
            add("synonym", polarity, f"The adjective is {word}", word)
            add("synonym_sentence", polarity, f"A described case is {word}.", word)

    for prompt, phrase in [
        (f"The property is {domain}", domain),
        (f"The measurement is {domain}", domain),
        (f"The concept is {domain}", domain),
    ]:
        add("domain", "domain", prompt, phrase)
    for prompt, phrase in [
        ("The property is neutral", "neutral"),
        ("The measurement is baseline", "baseline"),
        ("The concept is generic", "generic"),
    ]:
        add("domain", "baseline", prompt, phrase)
    return rows


def token_span_for_offsets(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> list[int]:
    idxs = []
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if e > char_start and s < char_end:
            idxs.append(i)
    if not idxs:
        raise ValueError(f"no token span for chars {char_start}:{char_end}")
    return idxs


def capture_lexical_states(model, tok, layer: int, batch_size: int) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    rows = []
    pair_slices = {}
    for pair in ALL_PAIRS:
        start = len(rows)
        rows.extend(lexical_prompt_rows(pair))
        pair_slices[pair] = (start, len(rows))

    prompts = [r["prompt"] for r in rows]
    token_spans: list[list[int]] = []
    for r in rows:
        enc = tok(r["prompt"], return_offsets_mapping=True, add_special_tokens=True)
        span = token_span_for_offsets(enc["offset_mapping"], int(r["char_start"]), int(r["char_end"]))
        token_spans.append(span)
        r["token_span_unpadded"] = [int(span[0]), int(span[-1] + 1)]
        r["token_count"] = int(len(span))
        r["target_token_texts"] = tok.convert_ids_to_tokens([int(enc["input_ids"][i]) for i in span])
        final_idx = len(enc["input_ids"]) - 1
        r["final_token_unpadded"] = int(final_idx)
        r["final_token_text"] = tok.convert_ids_to_tokens([int(enc["input_ids"][final_idx])])[0]
        r["offset_mapping"] = [[int(s), int(e)] for s, e in enc["offset_mapping"]]

    captured: list[np.ndarray] = []
    layers = get_layers(model)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().float().cpu().numpy())

    handle = layers[layer].register_forward_hook(hook)
    adjective_states = []
    final_states = []
    try:
        for b0 in range(0, len(prompts), batch_size):
            batch = prompts[b0 : b0 + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
            lengths = enc["attention_mask"].sum(dim=1).cpu().numpy().astype(int)
            # left padding is active, so unpadded token indices are shifted right by pad count.
            pad_counts = enc["attention_mask"].shape[1] - lengths
            captured.clear()
            with torch.no_grad():
                model(input_ids=enc["input_ids"].to(model.device), attention_mask=enc["attention_mask"].to(model.device), use_cache=False)
            h = captured[0]
            for j in range(len(batch)):
                row_i = b0 + j
                shifted = [idx + int(pad_counts[j]) for idx in token_spans[row_i]]
                adjective_states.append(h[j, shifted, :].mean(axis=0))
                final_states.append(h[j, int(lengths[j] + pad_counts[j] - 1), :])
    finally:
        handle.remove()

    adj = np.stack(adjective_states).astype(np.float64)
    final = np.stack(final_states).astype(np.float64)

    by_pair: dict[str, dict[str, np.ndarray]] = {}
    capture_json = {
        "model_id": MODEL_ID,
        "model_short": MODEL_SHORT,
        "layer": layer,
        "capture_rule": "adjective state is the mean over tokenizer offset tokens overlapping target_phrase; final state is the last non-padding token.",
        "by_pair": {},
    }
    for pair, (s0, s1) in pair_slices.items():
        prs = rows[s0:s1]
        A = adj[s0:s1]
        F = final[s0:s1]

        def diff(kind: str, state: np.ndarray, high_label: str = "high", low_label: str = "low") -> np.ndarray:
            hi = np.array([r["kind"] == kind and r["polarity"] == high_label for r in prs])
            lo = np.array([r["kind"] == kind and r["polarity"] == low_label for r in prs])
            if hi.sum() == 0 or lo.sum() == 0:
                return np.zeros(state.shape[1], dtype=np.float64)
            return state[hi].mean(axis=0) - state[lo].mean(axis=0)

        directions = {
            "d_word_token": diff("word", A),
            "d_sentence_token": diff("sentence", A),
            "d_sentence_final": diff("sentence", F),
            "d_synonym_token": diff("synonym", A),
            "d_synonym_sentence_token": diff("synonym_sentence", A),
            "d_domain_token": diff("domain", A, "domain", "baseline"),
        }
        pz = primal_z(pair, layer)
        px = primal_x(pair, layer)
        capture_json["by_pair"][pair] = {
            "n_prompts": len(prs),
            "prompt_rows": prs,
            "cosines": {
                f"cos_primal_z_{name}": cos(pz, vec) for name, vec in directions.items()
            }
            | {f"cos_primal_x_{name}": cos(px, vec) for name, vec in directions.items()},
        }
        by_pair[pair] = directions
        print(f"[v12.1] captured lexical states {pair}", flush=True)

    (RESULTS / "token_position_lexical_capture.json").write_text(json.dumps(capture_json, indent=2))
    return capture_json, by_pair


def orthonormal_basis(vectors: list[np.ndarray]) -> np.ndarray:
    X = np.stack([unit(v) for v in vectors if np.linalg.norm(v) > 1e-9], axis=1)
    if X.size == 0:
        return np.zeros((vectors[0].shape[0], 0), dtype=np.float64)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    rank = int((s > 1e-5).sum())
    return u[:, :rank].astype(np.float64)


def steer_logits(model, tok, prompts: list[str], hi_id: int, lo_id: int, direction: np.ndarray, layer: int, alpha: float, batch_size: int, max_seq: int) -> np.ndarray:
    d = torch.tensor(unit(direction), dtype=torch.bfloat16, device=model.device)
    layers = get_layers(model)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * d
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    vals = np.zeros(len(prompts), dtype=np.float32)
    handle = layers[layer].register_forward_hook(hook)
    try:
        for b0 in range(0, len(prompts), batch_size):
            batch = prompts[b0 : b0 + batch_size]
            enc = tok(batch, return_tensors="pt", padding="max_length", max_length=max_seq, truncation=True).to(model.device)
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            vals[b0 : b0 + len(batch)] = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
    finally:
        handle.remove()
    return vals


def steering_slope(model, tok, trials: list[dict], direction: np.ndarray, layer: int, args) -> float:
    prompts = [t["prompt"] for t in trials]
    hi_id = first_token_id(tok, trials[0]["high_word"])
    lo_id = first_token_id(tok, trials[0]["low_word"])
    pos = steer_logits(model, tok, prompts, hi_id, lo_id, direction, layer, args.alpha, args.batch_size, args.max_seq)
    neg = steer_logits(model, tok, prompts, hi_id, lo_id, direction, layer, -args.alpha, args.batch_size, args.max_seq)
    return float((pos - neg).mean() / (2 * args.alpha))


def residualization_steering(model, tok, lexical_dirs: dict[str, dict[str, np.ndarray]], layer: int, args) -> dict:
    rng = np.random.default_rng(121)
    out = {
        "model_id": MODEL_ID,
        "model_short": MODEL_SHORT,
        "layer": layer,
        "alpha": args.alpha,
        "prompts_per_pair": args.prompts_per_pair,
        "lexical_basis_directions": [
            "d_word_token",
            "d_sentence_token",
            "d_sentence_final",
            "d_synonym_token",
            "d_synonym_sentence_token",
            "d_domain_token",
        ],
        "by_pair": {},
    }
    for pair in ALL_PAIRS:
        pz = unit(primal_z(pair, layer))
        px = primal_x(pair, layer)
        dirs = lexical_dirs[pair]
        Q = orthonormal_basis([dirs[k] for k in out["lexical_basis_directions"]])
        pz_lex = Q @ (Q.T @ pz) if Q.shape[1] else np.zeros_like(pz)
        pz_resid = pz - pz_lex
        lex_norm = float(np.linalg.norm(pz_lex))
        resid_norm = float(np.linalg.norm(pz_resid))
        vectors = {
            "primal_z_context": pz,
            "lexical_projection_primal_z": unit(pz_lex) if lex_norm > 1e-9 else pz_lex,
            "lexical_residual_primal_z": unit(pz_resid) if resid_norm > 1e-9 else pz_resid,
            "primal_x_context": px,
            "d_word_token": dirs["d_word_token"],
            "d_sentence_token": dirs["d_sentence_token"],
            "d_sentence_final": dirs["d_sentence_final"],
            "random_null": rng.normal(size=pz.shape[0]),
        }
        trials = seed0_trials(pair, args.prompts_per_pair)
        slopes = {name: steering_slope(model, tok, trials, vec, layer, args) for name, vec in vectors.items()}
        out["by_pair"][pair] = {
            "n_prompts": len(trials),
            "lexical_subspace_rank": int(Q.shape[1]),
            "fraction_of_primal_z_norm_in_lexical_subspace": lex_norm,
            "fraction_of_primal_z_norm_squared_in_lexical_subspace": float(lex_norm**2),
            "residual_norm_fraction": resid_norm,
            "cos_primal_z_lexical_projection": cos(pz, pz_lex),
            "cos_primal_z_lexical_residual": cos(pz, pz_resid),
            "steering_slopes": slopes,
            "residual_over_primal_steering_ratio": float(slopes["lexical_residual_primal_z"] / slopes["primal_z_context"])
            if abs(slopes["primal_z_context"]) > 1e-9
            else None,
            "lexical_projection_over_primal_steering_ratio": float(slopes["lexical_projection_primal_z"] / slopes["primal_z_context"])
            if abs(slopes["primal_z_context"]) > 1e-9
            else None,
        }
        print(f"[v12.1] residualization {pair}: {out['by_pair'][pair]}", flush=True)
        (RESULTS / "lexical_subspace_residualization.json").write_text(json.dumps(out, indent=2))
    return out


def plot_token_capture(capture: dict) -> None:
    keys = ["d_word_token", "d_sentence_token", "d_sentence_final", "d_synonym_token", "d_domain_token"]
    M = np.array(
        [[capture["by_pair"][p]["cosines"].get(f"cos_primal_z_{k}", 0.0) for k in keys] for p in ALL_PAIRS],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(ALL_PAIRS)), ALL_PAIRS)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("v12.1 token-position lexical capture: cos(primal_z, lexical direction)")
    fig.colorbar(im, ax=ax, label="cosine")
    fig.tight_layout()
    fig.savefig(FIGS / "token_position_lexical_cosines.png", dpi=150)
    plt.close(fig)


def plot_residualization(res: dict) -> None:
    slope_keys = [
        "primal_z_context",
        "lexical_projection_primal_z",
        "lexical_residual_primal_z",
        "primal_x_context",
        "d_word_token",
        "d_sentence_token",
        "d_sentence_final",
        "random_null",
    ]
    M = np.array([[res["by_pair"][p]["steering_slopes"].get(k, 0.0) for k in slope_keys] for p in ALL_PAIRS], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    lim = max(0.01, float(np.nanmax(np.abs(M))))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(slope_keys)), slope_keys, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(ALL_PAIRS)), ALL_PAIRS)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("v12.1 lexical-subspace residualization steering @ L33")
    fig.colorbar(im, ax=ax, label="Delta logit-diff per alpha")
    fig.tight_layout()
    fig.savefig(FIGS / "lexical_subspace_residualization_steering.png", dpi=150)
    plt.close(fig)

    pairs = ALL_PAIRS
    lex_frac = [res["by_pair"][p]["fraction_of_primal_z_norm_squared_in_lexical_subspace"] for p in pairs]
    resid_ratio = [res["by_pair"][p]["residual_over_primal_steering_ratio"] for p in pairs]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(pairs))
    ax.bar(x - 0.18, lex_frac, width=0.36, label="norm^2 in lexical subspace")
    ax.bar(x + 0.18, resid_ratio, width=0.36, label="residual/primal steering")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, pairs, rotation=35, ha="right")
    ax.set_title("v12.1 lexical fraction removed vs residual steering")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "lexical_subspace_fraction_removed.png", dpi=150)
    plt.close(fig)


def write_summary(capture: dict, res: dict) -> None:
    z_word = np.array([capture["by_pair"][p]["cosines"]["cos_primal_z_d_word_token"] for p in ALL_PAIRS])
    z_sent_tok = np.array([capture["by_pair"][p]["cosines"]["cos_primal_z_d_sentence_token"] for p in ALL_PAIRS])
    z_sent_final = np.array([capture["by_pair"][p]["cosines"]["cos_primal_z_d_sentence_final"] for p in ALL_PAIRS])
    residual_ratios = np.array([res["by_pair"][p]["residual_over_primal_steering_ratio"] for p in ALL_PAIRS], dtype=float)
    lexical_ratios = np.array([res["by_pair"][p]["lexical_projection_over_primal_steering_ratio"] for p in ALL_PAIRS], dtype=float)
    lex_norm2 = np.array([res["by_pair"][p]["fraction_of_primal_z_norm_squared_in_lexical_subspace"] for p in ALL_PAIRS])
    lines = [
        "# V12.1 Lexical Disentanglement Summary",
        "",
        "V12.1 tests whether V12 lexical steering came from literal adjective-token",
        "semantics, sentence-final/template state, or a non-lexical residual of",
        "`primal_z` after removing a lexical subspace.",
        "",
        "## Aggregate Results",
        "",
        f"- mean cos(primal_z, word-token lexical direction): {z_word.mean():+.3f}",
        f"- mean cos(primal_z, sentence-token lexical direction): {z_sent_tok.mean():+.3f}",
        f"- mean cos(primal_z, sentence-final lexical direction): {z_sent_final.mean():+.3f}",
        f"- mean fraction of primal_z norm^2 in lexical subspace: {lex_norm2.mean():.3f}",
        f"- mean lexical-projection/primal steering ratio: {lexical_ratios.mean():+.3f}",
        f"- mean lexical-residual/primal steering ratio: {residual_ratios.mean():+.3f}",
        "",
        "## Interpretation Guardrails",
        "",
        "- If residual steering is strong, V12 lexical effects do not exhaust the causal direction.",
        "- If lexical projection dominates and residual steering is weak, the causal direction is mostly lexical/adjective geometry.",
        "- If both are nonzero, use the mixed-mechanism framing.",
        "",
        "## Per-pair Steering Ratios",
        "",
    ]
    for p in ALL_PAIRS:
        row = res["by_pair"][p]
        lines.append(
            f"- {p}: lexical_projection/primal={row['lexical_projection_over_primal_steering_ratio']:+.3f}, "
            f"residual/primal={row['residual_over_primal_steering_ratio']:+.3f}, "
            f"lexical_norm2={row['fraction_of_primal_z_norm_squared_in_lexical_subspace']:.3f}"
        )
    (REPO / "docs" / "V12_1_RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--prompts-per-pair", type=int, default=160)
    args = ap.parse_args()

    print(f"[v12.1] loading {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    print("[v12.1] model loaded", flush=True)

    capture, lexical_dirs = capture_lexical_states(model, tok, args.layer, args.batch_size)
    plot_token_capture(capture)
    res = residualization_steering(model, tok, lexical_dirs, args.layer, args)
    plot_residualization(res)
    write_summary(capture, res)
    print("[v12.1] complete", flush=True)


if __name__ == "__main__":
    main()
