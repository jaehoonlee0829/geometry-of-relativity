"""v12 GPU experiments for Gemma 2 9B.

Loads the model once and runs the GPU-bound V12 checks:

  * strategic-layer steering sweep for primal_z/probe_z/random null;
  * lexical prompt activation directions for the direction red-team;
  * z/x/lexical steering subtest on dense context prompts;
  * pure-x / fixed-mu / fixed-x / matched-z transfer controls.

The script is intentionally restartable: each section writes its JSON as soon
as it completes, and figures are generated at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

RESULTS = REPO / "results" / "v12"
FIGS = REPO / "figures" / "v12"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

MODEL_SHORT = "gemma2-9b"
MODEL_ID = "google/gemma-2-9b"
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
STRATEGIC_LAYERS = [0, 1, 3, 5, 7, 10, 13, 14, 17, 21, 25, 29, 33, 37, 41]
LATE = 33


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("decoder layers not found")


def load_npz(pair: str):
    p = REPO / "results" / "v11" / MODEL_SHORT / pair / f"{MODEL_SHORT}_{pair}_v11_residuals.npz"
    return np.load(p)


def load_meta(pair: str) -> dict:
    p = REPO / "results" / "v11" / MODEL_SHORT / pair / f"{MODEL_SHORT}_{pair}_v11_meta.json"
    return json.loads(p.read_text())


def load_trials(pair: str) -> list[dict]:
    p = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    return [json.loads(line) for line in p.open()]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.astype(np.float64)
    return (v / n).astype(np.float64)


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


def probe_z(pair: str, layer: int) -> np.ndarray:
    d = load_npz(pair)
    h = d["activations"][:, layer, :].astype(np.float64)
    z = d["z"].astype(np.float64)
    return Ridge(alpha=1.0).fit(h, z).coef_.astype(np.float64)


def seed0_trials(pair: str) -> list[dict]:
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
    return out


def controlled_subset(pair: str, condition: str, max_n: int = 96) -> list[dict]:
    rows = seed0_trials(pair)
    if not rows:
        return []
    x = np.array([float(t["x"]) for t in rows])
    z = np.array([float(t["z"]) for t in rows])
    mu = np.array([float(t["mu"]) for t in rows])
    if condition == "full_grid":
        chosen = rows
    elif condition == "fixed_mu":
        med = float(np.median(mu))
        keep = np.argsort(np.abs(mu - med))[: max_n * 2]
        chosen = [rows[i] for i in keep]
    elif condition == "fixed_x":
        values, counts = np.unique(np.round(x, 4), return_counts=True)
        center = values[np.argmin(np.abs(values - np.median(x)))]
        chosen = [t for t in rows if round(float(t["x"]), 4) == round(float(center), 4)]
    elif condition == "matched_z":
        values, counts = np.unique(np.round(z, 4), return_counts=True)
        center = values[np.argmin(np.abs(values))]
        chosen = [t for t in rows if round(float(t["z"]), 4) == round(float(center), 4)]
    else:
        raise ValueError(condition)
    if len(chosen) > max_n:
        idx = np.linspace(0, len(chosen) - 1, max_n).round().astype(int)
        chosen = [chosen[i] for i in idx]
    return chosen


def ids_for_pair(tok, pair: str) -> tuple[int, int]:
    meta = load_meta(pair)
    return first_token_id(tok, meta["high_word"]), first_token_id(tok, meta["low_word"])


def steer_logits(
    model,
    tok,
    prompts: list[str],
    hi_id: int,
    lo_id: int,
    direction: np.ndarray | None,
    layer: int,
    alpha: float,
    batch_size: int,
    max_seq: int,
) -> np.ndarray:
    if direction is not None and alpha != 0:
        d = torch.tensor(unit(direction), dtype=torch.bfloat16, device=model.device)
    else:
        d = None
    layers = get_layers(model)
    handle = None
    if d is not None:
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h + alpha * d
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        handle = layers[layer].register_forward_hook(hook)
    vals = np.zeros(len(prompts), dtype=np.float32)
    try:
        for b0 in range(0, len(prompts), batch_size):
            batch = prompts[b0 : b0 + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding="max_length",
                max_length=max_seq,
                truncation=True,
            ).to(model.device)
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            vals[b0 : b0 + len(batch)] = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()
    return vals


def slope_for_direction(model, tok, trials: list[dict], direction: np.ndarray, layer: int, args) -> float:
    if not trials:
        return float("nan")
    prompts = [t["prompt"] for t in trials]
    hi_id = first_token_id(tok, trials[0]["high_word"])
    lo_id = first_token_id(tok, trials[0]["low_word"])
    pos = steer_logits(model, tok, prompts, hi_id, lo_id, direction, layer, args.alpha, args.batch_size, args.max_seq)
    neg = steer_logits(model, tok, prompts, hi_id, lo_id, direction, layer, -args.alpha, args.batch_size, args.max_seq)
    return float((pos - neg).mean() / (2 * args.alpha))


def capture_prompt_directions(model, tok, args) -> dict:
    prompts_by_pair = {}
    meta_by_pair = {p: load_meta(p) for p in ALL_PAIRS}
    synonym = {
        "height": (["tall", "large", "high"], ["short", "small", "low"]),
        "age": (["old", "elderly", "aged"], ["young", "new", "youthful"]),
        "weight": (["heavy", "large", "weighty"], ["light", "small", "thin"]),
        "size": (["big", "large", "huge"], ["small", "little", "tiny"]),
        "speed": (["fast", "quick", "rapid"], ["slow", "sluggish", "unhurried"]),
        "wealth": (["rich", "wealthy", "affluent"], ["poor", "low-income", "broke"]),
        "experience": (["expert", "experienced", "veteran"], ["novice", "new", "inexperienced"]),
        "bmi_abs": (["obese", "heavy", "large"], ["thin", "lean", "light"]),
    }
    all_prompts = []
    slices = {}
    for pair in ALL_PAIRS:
        m = meta_by_pair[pair]
        high = m["high_word"]
        low = m["low_word"]
        highs, lows = synonym[pair]
        entries = {
            "lex_high_low_word_high": [f"The word is: {high}", f"This adjective is {high}", f"A description: {high}"],
            "lex_high_low_word_low": [f"The word is: {low}", f"This adjective is {low}", f"A description: {low}"],
            "lex_sentence_high": [f"This person is {high}.", f"The described case is {high}."],
            "lex_sentence_low": [f"This person is {low}.", f"The described case is {low}."],
            "lex_synonym_high": [f"The word is: {w}" for w in highs],
            "lex_synonym_low": [f"The word is: {w}" for w in lows],
            "lex_domain_high": [f"The property is {pair}.", f"The measurement is {pair}."],
            "lex_domain_low": ["The property is neutral.", "The measurement is baseline."],
        }
        prompts_by_pair[pair] = entries
        slices[pair] = {}
        for name, ps in entries.items():
            start = len(all_prompts)
            all_prompts.extend(ps)
            slices[pair][name] = (start, len(all_prompts))

    layers = get_layers(model)
    captures = []
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captures.append(h[:, -1, :].detach().float().cpu().numpy())
    handle = layers[LATE].register_forward_hook(hook)
    try:
        for b0 in range(0, len(all_prompts), args.batch_size):
            enc = tok(
                all_prompts[b0 : b0 + args.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(model.device)
            with torch.no_grad():
                model(**enc, use_cache=False)
    finally:
        handle.remove()
    H = np.concatenate(captures, axis=0).astype(np.float64)

    out = {"model_short": MODEL_SHORT, "layer": LATE, "by_pair": {}}
    for pair in ALL_PAIRS:
        d = {}
        dirs = {}
        prompt_rows = []
        for base in ["lex_high_low_word", "lex_sentence", "lex_synonym", "lex_domain"]:
            a0, a1 = slices[pair][f"{base}_high"]
            b0, b1 = slices[pair][f"{base}_low"]
            dirs[base] = H[a0:a1].mean(0) - H[b0:b1].mean(0)
        for name, (s0, s1) in slices[pair].items():
            for idx in range(s0, s1):
                prompt_rows.append({"kind": name, "prompt": all_prompts[idx], "global_index": idx})
        pz = primal_z(pair, LATE)
        px = primal_x(pair, LATE)
        for name, vec in dirs.items():
            d[f"cos_primal_z_{name}"] = cos(pz, vec)
            d[f"cos_primal_x_{name}"] = cos(px, vec)
        out["by_pair"][pair] = d
        np.save(RESULTS / f"{MODEL_SHORT}_{pair}_L{LATE}_lexical_dirs.npy", np.stack([dirs[k] for k in sorted(dirs)]))
        (RESULTS / f"{MODEL_SHORT}_{pair}_L{LATE}_lexical_dir_names.json").write_text(json.dumps(sorted(dirs), indent=2))
        idxs = [r["global_index"] for r in prompt_rows]
        np.savez(
            RESULTS / f"{MODEL_SHORT}_{pair}_L{LATE}_lexical_prompt_activations.npz",
            activations=H[idxs].astype(np.float32),
            kinds=np.array([r["kind"] for r in prompt_rows]),
            prompts=np.array([r["prompt"] for r in prompt_rows]),
        )
        print(f"[v12-gpu] lexical directions {pair}", flush=True)
    (RESULTS / "direction_redteam_lexical_activations.json").write_text(json.dumps(out, indent=2))
    return out


def layer_steering(model, tok, args) -> dict:
    out = {"model_short": MODEL_SHORT, "layers": STRATEGIC_LAYERS, "pairs": ALL_PAIRS, "alpha": args.alpha, "by_pair": {}}
    rng = np.random.default_rng(12)
    for pair in ALL_PAIRS:
        trials = controlled_subset(pair, "full_grid", max_n=args.layer_sweep_prompts)
        out["by_pair"][pair] = {}
        d_model = load_npz(pair)["activations"].shape[-1]
        null = rng.normal(size=d_model)
        for layer in STRATEGIC_LAYERS:
            pz = primal_z(pair, layer)
            qz = probe_z(pair, layer)
            out["by_pair"][pair][str(layer)] = {
                "n_prompts": len(trials),
                "primal_z": slope_for_direction(model, tok, trials, pz, layer, args),
                "probe_z": slope_for_direction(model, tok, trials, qz, layer, args),
                "random_null": slope_for_direction(model, tok, trials, null, layer, args),
            }
            print(f"[v12-gpu] layer steering {pair} L{layer}: {out['by_pair'][pair][str(layer)]}", flush=True)
        (RESULTS / "layer_sweep_9b_steering.json").write_text(json.dumps(out, indent=2))
    return out


def redteam_steering(model, tok, lexical: dict, args) -> dict:
    out = {"model_short": MODEL_SHORT, "layer": LATE, "alpha": args.alpha, "by_pair": {}}
    rng = np.random.default_rng(13)
    for pair in ALL_PAIRS:
        trials = controlled_subset(pair, "full_grid", max_n=args.redteam_prompts)
        names = json.loads((RESULTS / f"{MODEL_SHORT}_{pair}_L{LATE}_lexical_dir_names.json").read_text())
        dirs = np.load(RESULTS / f"{MODEL_SHORT}_{pair}_L{LATE}_lexical_dirs.npy")
        lex = {name: dirs[i] for i, name in enumerate(names)}
        d_model = load_npz(pair)["activations"].shape[-1]
        vectors = {
            "primal_z_context": primal_z(pair, LATE),
            "primal_x_context": primal_x(pair, LATE),
            "lex_high_low_word": lex["lex_high_low_word"],
            "lex_sentence_high_low": lex["lex_sentence"],
            "random_null": rng.normal(size=d_model),
        }
        out["by_pair"][pair] = {
            name: slope_for_direction(model, tok, trials, vec, LATE, args) for name, vec in vectors.items()
        }
        out["by_pair"][pair]["n_prompts"] = len(trials)
        print(f"[v12-gpu] direction steering {pair}: {out['by_pair'][pair]}", flush=True)
        (RESULTS / "direction_redteam_steering.json").write_text(json.dumps(out, indent=2))
    return out


def pure_x_transfer(model, tok, args) -> dict:
    conditions = ["full_grid", "fixed_mu", "fixed_x", "matched_z"]
    primals = {p: primal_z(p, LATE) for p in ALL_PAIRS}
    out = {
        "model_short": MODEL_SHORT,
        "layer": LATE,
        "alpha": args.alpha,
        "conditions": conditions,
        "target_by_source_by_condition": {},
    }
    for cond in conditions:
        out["target_by_source_by_condition"][cond] = {}
        for target in ALL_PAIRS:
            trials = controlled_subset(target, cond, max_n=args.transfer_prompts)
            out["target_by_source_by_condition"][cond][target] = {}
            for source in ALL_PAIRS:
                out["target_by_source_by_condition"][cond][target][source] = slope_for_direction(
                    model, tok, trials, primals[source], LATE, args
                )
            out["target_by_source_by_condition"][cond][target]["n_prompts"] = len(trials)
            vals = [out["target_by_source_by_condition"][cond][target][s] for s in ALL_PAIRS if s != target]
            print(
                f"[v12-gpu] transfer {cond}/{target}: within="
                f"{out['target_by_source_by_condition'][cond][target][target]:+.3f} "
                f"cross={float(np.nanmean(vals)):+.3f} n={len(trials)}",
                flush=True,
            )
            (RESULTS / "pure_x_transfer_control.json").write_text(json.dumps(out, indent=2))
    return out


def plot_gpu_outputs() -> None:
    red_path = RESULTS / "direction_redteam_steering.json"
    if red_path.exists():
        red = json.loads(red_path.read_text())
        keys = ["primal_z_context", "primal_x_context", "lex_high_low_word", "lex_sentence_high_low", "random_null"]
        M = np.array([[red["by_pair"][p].get(k, np.nan) for k in keys] for p in ALL_PAIRS])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        lim = max(0.01, float(np.nanmax(np.abs(M))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(ALL_PAIRS)), ALL_PAIRS)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("v12 direction red-team steering @ L33")
        fig.colorbar(im, ax=ax, label="Delta logit-diff per alpha")
        fig.tight_layout()
        fig.savefig(FIGS / "direction_redteam_steering.png", dpi=150)
        plt.close(fig)

    tr_path = RESULTS / "pure_x_transfer_control.json"
    if tr_path.exists():
        tr = json.loads(tr_path.read_text())
        conds = tr["conditions"]
        fig, axes = plt.subplots(1, len(conds), figsize=(18, 4.7), sharey=True)
        for ax, cond in zip(axes, conds):
            M = np.array(
                [[tr["target_by_source_by_condition"][cond][t].get(s, np.nan) for s in ALL_PAIRS] for t in ALL_PAIRS]
            )
            lim = max(0.01, float(np.nanmax(np.abs(M))))
            im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(cond)
            ax.set_xticks(range(len(ALL_PAIRS)), ALL_PAIRS, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(ALL_PAIRS)), ALL_PAIRS, fontsize=7)
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Delta logit-diff per alpha")
        fig.suptitle("v12 pure-x / fixed-mu transfer controls")
        fig.savefig(FIGS / "pure_x_transfer_control.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="all", help="comma list: lexical,layer,redteam,transfer")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--layer-sweep-prompts", type=int, default=160)
    ap.add_argument("--redteam-prompts", type=int, default=160)
    ap.add_argument("--transfer-prompts", type=int, default=72)
    args = ap.parse_args()
    sections = {"lexical", "layer", "redteam", "transfer"} if args.sections == "all" else set(args.sections.split(","))

    print(f"[v12-gpu] loading {MODEL_ID}", flush=True)
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
    print("[v12-gpu] model loaded", flush=True)

    lexical = None
    if "lexical" in sections:
        lexical = capture_prompt_directions(model, tok, args)
    if "layer" in sections:
        layer_steering(model, tok, args)
    if "redteam" in sections:
        if lexical is None and not (RESULTS / "direction_redteam_lexical_activations.json").exists():
            lexical = capture_prompt_directions(model, tok, args)
        redteam_steering(model, tok, lexical or {}, args)
    if "transfer" in sections:
        pure_x_transfer(model, tok, args)
    plot_gpu_outputs()
    print("[v12-gpu] complete", flush=True)


if __name__ == "__main__":
    main()
