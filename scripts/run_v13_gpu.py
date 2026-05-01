"""V13 GPU session: affine/OOD relativity, x-transfer controls, top logits.

This is a restartable minimum-viable implementation of
docs/NEXT_GPU_SESSION_v13.md. It writes JSON after each section and renders all
figures it can from whatever results already exist.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS, LOG_SPACE_PAIRS, build_implicit_items, fmt_num, sample_context  # noqa: E402

MODEL_SHORT = "gemma2-9b"
MODEL_ID = "google/gemma-2-9b"
PRIMARY_PAIRS = ["height", "weight", "speed", "experience"]
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
LAYERS = [25, 33]
LATE = 33

RESULTS = REPO / "results" / "v13"
FIGS = REPO / "figures" / "v13"
for sub in ["affine_shift", "x_transfer", "top_logits", "domain_extension"]:
    (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    (FIGS / sub).mkdir(parents=True, exist_ok=True)

PAIR_BY_NAME = {p.name: p for p in PAIRS}

NEW_DOMAINS = {
    "brightness": {
        "unit": "lux",
        "low_word": "dim",
        "high_word": "bright",
        "sigma": 150.0,
        "xs": [120.0, 250.0, 500.0, 900.0, 1500.0],
        "entity": "Lamp",
        "value_text": "{x_str} lux",
        "prompt": "{items}\nLamp {n_last}: {x_str} lux. This lamp is",
    },
    "temperature": {
        "unit": "C",
        "low_word": "cold",
        "high_word": "hot",
        "sigma": 8.0,
        "xs": [0.0, 8.0, 16.0, 25.0, 36.0],
        "entity": "Room",
        "value_text": "{x_str} C",
        "prompt": "{items}\nRoom {n_last}: {x_str} C. This room is",
    },
    "loudness": {
        "unit": "dB",
        "low_word": "quiet",
        "high_word": "loud",
        "sigma": 8.0,
        "xs": [35.0, 45.0, 55.0, 70.0, 85.0],
        "entity": "Sound",
        "value_text": "{x_str} dB",
        "prompt": "{items}\nSound {n_last}: {x_str} dB. This sound is",
    },
    "price": {
        "unit": "dollars",
        "low_word": "cheap",
        "high_word": "expensive",
        "sigma": 40.0,
        "xs": [15.0, 35.0, 75.0, 150.0, 300.0],
        "entity": "Item",
        "value_text": "${x_str}",
        "prompt": "{items}\nItem {n_last}: ${x_str}. This item is",
    },
}


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("decoder layers not found")


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float64) if n < 1e-12 else (v / n).astype(np.float64)


def corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def load_npz(pair: str, model_short: str = MODEL_SHORT):
    return np.load(REPO / "results" / "v11" / model_short / pair / f"{model_short}_{pair}_v11_residuals.npz")


def load_meta(pair: str, model_short: str = MODEL_SHORT) -> dict:
    return json.loads((REPO / "results" / "v11" / model_short / pair / f"{model_short}_{pair}_v11_meta.json").read_text())


def load_trials(pair: str) -> list[dict]:
    return [json.loads(line) for line in (REPO / "data_gen" / f"v11_{pair}_trials.jsonl").open()]


def seed0_trials(pair: str, max_n: int) -> list[dict]:
    rows = []
    seen = set()
    for t in load_trials(pair):
        if t.get("cell_seed") != 0:
            continue
        key = (round(float(t["x"]), 4), round(float(t["z"]), 4))
        if key in seen:
            continue
        seen.add(key)
        rows.append(t)
    if len(rows) > max_n:
        idx = np.linspace(0, len(rows) - 1, max_n).round().astype(int)
        rows = [rows[i] for i in idx]
    return rows


def primal(pair: str, layer: int, key: str) -> np.ndarray:
    d = load_npz(pair)
    h = d["activations"][:, layer, :].astype(np.float64)
    if key == "z":
        z = d["z"].astype(np.float64)
        return h[z > 1.0].mean(0) - h[z < -1.0].mean(0)
    if key == "x":
        x = d["x"].astype(np.float64)
        q25, q75 = np.quantile(x, [0.25, 0.75])
        return h[x >= q75].mean(0) - h[x <= q25].mean(0)
    raise ValueError(key)


def primal_x_resid_z(pair: str, layer: int) -> np.ndarray:
    d = load_npz(pair)
    h = d["activations"][:, layer, :].astype(np.float64)
    x = d["x"].astype(np.float64)
    z = d["z"].astype(np.float64)
    X = np.column_stack([np.ones_like(z), z])
    x_resid = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    q25, q75 = np.quantile(x_resid, [0.25, 0.75])
    return h[x_resid >= q75].mean(0) - h[x_resid <= q25].mean(0)


def build_items_with_sigma(pair_name: str, mu: float, sigma: float, seed: int, n: int = 15) -> list[str]:
    pair = PAIR_BY_NAME[pair_name]
    low = pair.target_values[0] * 0.4
    high = pair.target_values[-1] * 2.5
    vals = sample_context(mu, sigma, seed, n, low, high, log_space=(pair.name in LOG_SPACE_PAIRS))
    if pair.name == "height":
        return [f"Person {i + 1}: {int(v)} cm" for i, v in enumerate(vals)]
    if pair.name == "age":
        return [f"Person {i + 1}: {int(v)} years old" for i, v in enumerate(vals)]
    if pair.name == "weight":
        return [f"Person {i + 1}: {int(v)} kg" for i, v in enumerate(vals)]
    if pair.name == "size":
        return [f"Object {i + 1}: {int(v)} cm across" for i, v in enumerate(vals)]
    if pair.name == "speed":
        return [f"Vehicle {i + 1}: {int(v)} km/h" for i, v in enumerate(vals)]
    if pair.name == "wealth":
        return [f"Person {i + 1} earns ${int(v)}/year" for i, v in enumerate(vals)]
    if pair.name == "experience":
        return [f"Worker {i + 1}: {int(v)} years experience" for i, v in enumerate(vals)]
    if pair.name == "bmi_abs":
        return [f"Person {i + 1}: BMI {v:.1f}" for i, v in enumerate(vals)]
    return [f"Item {i + 1}: {v}" for i, v in enumerate(vals)]


def make_v11_prompt(pair_name: str, x: float, mu: float, seed: int, sigma: float | None = None) -> str:
    pair = PAIR_BY_NAME[pair_name]
    items = build_implicit_items(pair, mu, seed) if sigma is None else build_items_with_sigma(pair_name, mu, sigma, seed)
    return pair.format_prompt_implicit.format(
        items="\n".join(items),
        n_last=len(items) + 1,
        x_str=fmt_num(x),
    )


def affine_trials(pair_name: str, max_per_condition: int) -> list[dict]:
    base = seed0_trials(pair_name, 9999)
    pair = PAIR_BY_NAME[pair_name]
    seed0 = int(base[0]["seed"])
    xs = np.array([float(t["x"]) for t in base])
    mus = np.array([float(t["mu"]) for t in base])
    sigma = float(base[0]["sigma"])
    span = float(np.nanmax(xs) - np.nanmin(xs))
    delta = max(span, sigma * 6)
    rows = []
    for t in base:
        x, mu, z = float(t["x"]), float(t["mu"]), float(t["z"])
        transforms = {
            "base": (x, mu, sigma, z),
            "parallel_shift": (x + delta, mu + delta, sigma, z),
            "scale_up": (x * 2.0, mu * 2.0, sigma * 2.0, z),
            "world_ood": (x + 2.0 * delta, mu + 2.0 * delta, sigma, z),
            "target_ood": (mu + np.sign(z if z != 0 else 1.0) * sigma * 5.0, mu, sigma, np.sign(z if z != 0 else 1.0) * 5.0),
        }
        if pair_name not in LOG_SPACE_PAIRS and min(x - delta, mu - delta) > 0:
            transforms["negative_shift"] = (x - delta, mu - delta, sigma, z)
        if pair_name not in LOG_SPACE_PAIRS and min(x * 0.5, mu * 0.5, sigma * 0.5) > 0:
            transforms["scale_down"] = (x * 0.5, mu * 0.5, sigma * 0.5, z)
        for condition, (x2, mu2, sig2, z_eff) in transforms.items():
            rows.append({
                "pair": pair_name,
                "condition": condition,
                "x": float(x2),
                "mu": float(mu2),
                "sigma": float(sig2),
                "z": z,
                "z_eff": float(z_eff),
                "seed": seed0,
                "low_word": pair.low_word,
                "high_word": pair.high_word,
                "prompt": make_v11_prompt(pair_name, float(x2), float(mu2), seed0, float(sig2)),
            })
    out = []
    for condition in sorted({r["condition"] for r in rows}):
        cr = [r for r in rows if r["condition"] == condition]
        if len(cr) > max_per_condition:
            idx = np.linspace(0, len(cr) - 1, max_per_condition).round().astype(int)
            cr = [cr[i] for i in idx]
        out.extend(cr)
    return out


def new_domain_trials(domains: list[str], per_domain: int) -> list[dict]:
    zs = np.linspace(-3, 3, 13).round(2)
    rows = []
    for name in domains:
        spec = NEW_DOMAINS[name]
        xs = np.array(spec["xs"], dtype=float)
        sigma = float(spec["sigma"])
        for x0 in xs:
            for z in zs:
                mu = float(x0 - sigma * z)
                vals = np.random.default_rng(int(abs(x0 * 10 + z * 1000))).normal(mu, sigma, 5)
                items = [f"{spec['entity']} {i + 1}: {fmt_num(max(0.1, v))} {spec['unit']}." for i, v in enumerate(vals)]
                prompt = spec["prompt"].format(items="\n".join(items), n_last=len(items) + 1, x_str=fmt_num(x0))
                rows.append({
                    "pair": name,
                    "condition": "relative_domain",
                    "x": float(x0),
                    "mu": mu,
                    "sigma": sigma,
                    "z": float(z),
                    "z_eff": float(z),
                    "low_word": spec["low_word"],
                    "high_word": spec["high_word"],
                    "prompt": prompt,
                })
    if per_domain:
        keep = []
        for name in domains:
            dr = [r for r in rows if r["pair"] == name]
            if len(dr) > per_domain:
                idx = np.linspace(0, len(dr) - 1, per_domain).round().astype(int)
                dr = [dr[i] for i in idx]
            keep.extend(dr)
        rows = keep
    return rows


def objective_trials(per_task: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(13)
    for task in ["positive_negative", "even_odd"]:
        for x in np.linspace(-9, 9, 19):
            if task == "even_odd":
                x = int(round(x + 20))
            for mu in [-6, 0, 6]:
                context = rng.normal(mu, 4.0, 5)
                if task == "positive_negative":
                    items = "\n".join([f"Number {i + 1}: {v:+.1f}." for i, v in enumerate(context)])
                    prompt = f"{items}\nNumber 6: {x:+.1f}. Forced choice: this number is"
                    low, high = "negative", "positive"
                    z = (float(x) - mu) / 4.0
                    objective = 1.0 if x > 0 else -1.0 if x < 0 else 0.0
                else:
                    vals = [int(round(v + 20)) for v in context]
                    items = "\n".join([f"Number {i + 1}: {v}." for i, v in enumerate(vals)])
                    prompt = f"{items}\nNumber 6: {x}. Forced choice: this number is"
                    low, high = "odd", "even"
                    z = (float(x) - (mu + 20)) / 4.0
                    objective = 1.0 if int(x) % 2 == 0 else -1.0
                rows.append({
                    "pair": task,
                    "condition": "objective_control",
                    "x": float(x),
                    "mu": float(mu),
                    "sigma": 4.0,
                    "z": float(z),
                    "z_eff": float(z),
                    "objective": objective,
                    "low_word": low,
                    "high_word": high,
                    "prompt": prompt,
                })
    if per_task:
        keep = []
        for task in ["positive_negative", "even_odd"]:
            tr = [r for r in rows if r["pair"] == task]
            if len(tr) > per_task:
                idx = np.linspace(0, len(tr) - 1, per_task).round().astype(int)
                tr = [tr[i] for i in idx]
            keep.extend(tr)
        rows = keep
    return rows


def run_prompts(model, tok, rows: list[dict], layers_to_capture: list[int], batch_size: int, max_seq: int, top_k: int = 0):
    layers = get_layers(model)
    captures = {layer: [] for layer in layers_to_capture}
    handles = []
    for layer in layers_to_capture:
        def make_hook(layer):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                captures[layer].append(h[:, -1, :].detach().float().cpu().numpy())
            return hook
        handles.append(layers[layer].register_forward_hook(make_hook(layer)))
    ld = np.zeros(len(rows), dtype=np.float32)
    top_rows = []
    try:
        for b0 in range(0, len(rows), batch_size):
            batch_rows = rows[b0:b0 + batch_size]
            enc = tok([r["prompt"] for r in batch_rows], return_tensors="pt", padding="max_length", max_length=max_seq, truncation=True).to(model.device)
            hi = [first_token_id(tok, r["high_word"]) for r in batch_rows]
            lo = [first_token_id(tok, r["low_word"]) for r in batch_rows]
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            for i, r in enumerate(batch_rows):
                ld[b0 + i] = float(logits[i, hi[i]] - logits[i, lo[i]])
                if top_k:
                    vals, ids = torch.topk(logits[i], k=top_k)
                    probs = torch.softmax(logits[i], dim=-1)[ids]
                    for rank, (tid, val, prob) in enumerate(zip(ids.cpu().tolist(), vals.cpu().tolist(), probs.cpu().tolist()), start=1):
                        top_rows.append({
                            "row_index": b0 + i,
                            "rank": rank,
                            "token_id": int(tid),
                            "token": tok.decode([tid]),
                            "logit": float(val),
                            "probability": float(prob),
                        })
    finally:
        for h in handles:
            h.remove()
    H = {layer: np.concatenate(chunks, axis=0).astype(np.float32) for layer, chunks in captures.items()}
    return ld, H, top_rows


def steer_ld(model, tok, rows: list[dict], direction: np.ndarray, layer: int, alpha: float, batch_size: int, max_seq: int) -> np.ndarray:
    d = torch.tensor(unit(direction), dtype=torch.bfloat16, device=model.device)
    layers = get_layers(model)
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * d
        return (h,) + output[1:] if isinstance(output, tuple) else h
    handle = layers[layer].register_forward_hook(hook)
    vals = np.zeros(len(rows), dtype=np.float32)
    try:
        for b0 in range(0, len(rows), batch_size):
            batch = rows[b0:b0 + batch_size]
            enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding="max_length", max_length=max_seq, truncation=True).to(model.device)
            hi = [first_token_id(tok, r["high_word"]) for r in batch]
            lo = [first_token_id(tok, r["low_word"]) for r in batch]
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            vals[b0:b0 + len(batch)] = np.array([float(logits[i, hi[i]] - logits[i, lo[i]]) for i in range(len(batch))], dtype=np.float32)
    finally:
        handle.remove()
    return vals


def steering_slope(model, tok, rows: list[dict], direction: np.ndarray, layer: int, args) -> float:
    if not rows:
        return float("nan")
    pos = steer_ld(model, tok, rows, direction, layer, args.alpha, args.batch_size, args.max_seq)
    neg = steer_ld(model, tok, rows, direction, layer, -args.alpha, args.batch_size, args.max_seq)
    return float((pos - neg).mean() / (2.0 * args.alpha))


def experiment_affine(model, tok, args) -> None:
    rows = []
    for pair in args.pairs:
        rows.extend(affine_trials(pair, args.affine_prompts_per_condition))
    ld, H, top = run_prompts(model, tok, rows, LAYERS, args.batch_size, args.max_seq, top_k=args.top_k)
    for i, r in enumerate(rows):
        r["ld"] = float(ld[i])
    metrics = {"model_id": MODEL_ID, "model_short": MODEL_SHORT, "layers": LAYERS, "pairs": args.pairs, "conditions": {}, "by_pair": {}}
    probe_transfer = {"model_short": MODEL_SHORT, "layer": LATE, "by_pair": {}}
    steering = {"model_short": MODEL_SHORT, "layer": LATE, "alpha": args.alpha, "by_pair": {}}
    for pair in args.pairs:
        pr = [i for i, r in enumerate(rows) if r["pair"] == pair]
        metrics["by_pair"][pair] = {}
        for condition in sorted({rows[i]["condition"] for i in pr}):
            idx = [i for i in pr if rows[i]["condition"] == condition]
            xs = [rows[i]["x"] for i in idx]
            mus = [rows[i]["mu"] for i in idx]
            zs = [rows[i]["z"] for i in idx]
            ze = [rows[i]["z_eff"] for i in idx]
            lds = [rows[i]["ld"] for i in idx]
            metrics["by_pair"][pair][condition] = {
                "n": len(idx),
                "corr_ld_z": corr(lds, zs),
                "corr_ld_z_eff": corr(lds, ze),
                "corr_ld_x": corr(lds, xs),
                "corr_ld_mu": corr(lds, mus),
                "mean_ld": float(np.mean(lds)),
            }
        base_idx = [i for i in pr if rows[i]["condition"] == "base"]
        base_z = np.array([rows[i]["z"] for i in base_idx], dtype=np.float64)
        pz_base = H[LATE][base_idx][base_z > 1.0].mean(0) - H[LATE][base_idx][base_z < -1.0].mean(0)
        probe = Ridge(alpha=1.0).fit(H[LATE][base_idx].astype(np.float64), base_z)
        probe_transfer["by_pair"][pair] = {}
        steering["by_pair"][pair] = {}
        for condition in sorted({rows[i]["condition"] for i in pr}):
            idx = [i for i in pr if rows[i]["condition"] == condition]
            cz = np.array([rows[i]["z_eff"] for i in idx], dtype=np.float64)
            Hc = H[LATE][idx].astype(np.float64)
            pc = Hc[cz > 1.0].mean(0) - Hc[cz < -1.0].mean(0) if (cz > 1.0).any() and (cz < -1.0).any() else np.zeros(Hc.shape[1])
            pred = probe.predict(Hc)
            probe_transfer["by_pair"][pair][condition] = {
                "cos_primal_z_base_condition": float(unit(pz_base) @ unit(pc)),
                "base_probe_corr_with_z_eff": corr(pred, cz),
            }
            cond_rows = [rows[i] for i in idx]
            steering["by_pair"][pair][condition] = {
                "base_primal_z_slope": steering_slope(model, tok, cond_rows, pz_base, LATE, args),
                "condition_primal_z_slope": steering_slope(model, tok, cond_rows, pc, LATE, args),
            }
        print(f"[v13] affine {pair} done", flush=True)
        (RESULTS / "affine_shift" / "affine_shift_metrics.json").write_text(json.dumps(metrics, indent=2))
        (RESULTS / "affine_shift" / "affine_shift_probe_transfer.json").write_text(json.dumps(probe_transfer, indent=2))
        (RESULTS / "affine_shift" / "affine_shift_steering.json").write_text(json.dumps(steering, indent=2))
    with (RESULTS / "top_logits" / "top_logits_by_condition.jsonl").open("w") as f:
        for tr in top:
            row = rows[tr["row_index"]]
            f.write(json.dumps({**{k: row[k] for k in ["pair", "condition", "x", "mu", "sigma", "z", "z_eff"]}, **tr}) + "\n")


def experiment_x_transfer(model, tok, args) -> None:
    pairs = ALL_PAIRS
    directions = {
        "primal_z": {p: primal(p, LATE, "z") for p in pairs},
        "primal_x_naive": {p: primal(p, LATE, "x") for p in pairs},
        "primal_x_resid_z": {p: primal_x_resid_z(p, LATE) for p in pairs},
    }
    matrices = {fam: {t: {} for t in pairs} for fam in directions}
    n_prompts = {}
    for target in pairs:
        rows = seed0_trials(target, args.transfer_prompts)
        n_prompts[target] = len(rows)
        for source in pairs:
            for fam, by_source in directions.items():
                matrices[fam][target][source] = steering_slope(model, tok, rows, by_source[source], LATE, args)
        vals_z = [matrices["primal_z"][target][s] for s in pairs if s != target]
        vals_x = [matrices["primal_x_naive"][target][s] for s in pairs if s != target]
        print(f"[v13] x-transfer target={target} z_cross={np.mean(vals_z):+.3f} x_cross={np.mean(vals_x):+.3f}", flush=True)
        (RESULTS / "x_transfer" / "cross_pair_transfer_x_8x8.json").write_text(json.dumps({
            "model_id": MODEL_ID,
            "model_short": MODEL_SHORT,
            "layer": LATE,
            "alpha": args.alpha,
            "pairs": pairs,
            "n_prompts_by_target": n_prompts,
            "matrices": matrices,
        }, indent=2))
    summary = summarize_transfer(matrices, pairs)
    (RESULTS / "x_transfer" / "cross_pair_transfer_x_vs_z_summary.json").write_text(json.dumps(summary, indent=2))


def summarize_transfer(matrices: dict, pairs: list[str]) -> dict:
    n = len(pairs)
    off = ~np.eye(n, dtype=bool)
    out = {"families": {}, "paired_offdiag": {}}
    arrs = {}
    for fam, mat in matrices.items():
        M = np.array([[mat[t][s] for s in pairs] for t in pairs], dtype=np.float64)
        arrs[fam] = M
        out["families"][fam] = {
            "mean_diagonal": float(np.diag(M).mean()),
            "mean_off_diagonal": float(M[off].mean()),
            "off_diagonal_positive_fraction": float((M[off] > 0).mean()),
            "target_wise_off_diagonal_mean": {t: float(np.delete(M[i], i).mean()) for i, t in enumerate(pairs)},
            "source_wise_off_diagonal_mean": {s: float(np.delete(M[:, j], j).mean()) for j, s in enumerate(pairs)},
        }
    for fam in ["primal_x_naive", "primal_x_resid_z"]:
        d = arrs["primal_z"][off] - arrs[fam][off]
        out["paired_offdiag"][f"z_minus_{fam}"] = {
            "mean": float(d.mean()),
            "positive_fraction": float((d > 0).mean()),
            "ci95": bootstrap_ci(d),
        }
    return out


def bootstrap_ci(x: np.ndarray, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(130)
    vals = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def token_group(token: str, pair: str) -> str:
    s = token.strip().lower()
    groups = {
        "high": {"tall", "heavy", "fast", "expert", "old", "big", "rich", "obese", "bright", "hot", "loud", "expensive"},
        "low": {"short", "light", "slow", "novice", "young", "small", "poor", "thin", "dim", "cold", "quiet", "cheap"},
        "neutral": {"average", "normal", "moderate", "typical", "medium"},
        "extreme_high": {"giant", "huge", "massive", "extreme", "very", "exceptionally"},
        "extreme_low": {"tiny", "minuscule", "weak", "barely"},
        "ood": {"impossible", "weird", "unusual", "unrealistic", "invalid"},
    }
    for name, toks in groups.items():
        if s in toks:
            return name
    if any(ch.isdigit() for ch in s):
        return "number"
    if s in {"cm", "kg", "km", "year", "years", "$", "c", "db", "lux"}:
        return "unit"
    return "other"


def analyze_top_logits() -> None:
    path = RESULTS / "top_logits" / "top_logits_by_condition.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.open()]
    grouped = {}
    for r in rows:
        key = (r["pair"], r["condition"], round(float(r["z_eff"]), 1))
        grouped.setdefault(key, {}).setdefault(token_group(r["token"], r["pair"]), []).append(float(r["logit"]))
    out = []
    for (pair, condition, z), by_group in grouped.items():
        scores = {}
        for group, logits in by_group.items():
            m = max(logits)
            scores[group] = float(m + math.log(sum(math.exp(v - m) for v in logits)))
        out.append({"pair": pair, "condition": condition, "z_bin": z, "group_scores": scores, "group_ld": scores.get("high", np.nan) - scores.get("low", np.nan)})
    (RESULTS / "top_logits" / "top_logit_group_scores.json").write_text(json.dumps({"rows": out}, indent=2))


def experiment_domain(model, tok, args) -> None:
    rel_rows = new_domain_trials(args.new_domains, args.domain_prompts_per_task)
    obj_rows = objective_trials(args.domain_prompts_per_task)
    ld, H, _ = run_prompts(model, tok, rel_rows + obj_rows, [LATE], args.batch_size, args.max_seq, top_k=0)
    rows = rel_rows + obj_rows
    for i, r in enumerate(rows):
        r["ld"] = float(ld[i])
    rel = {"model_short": MODEL_SHORT, "layer": LATE, "by_domain": {}}
    obj = {"model_short": MODEL_SHORT, "by_task": {}}
    for name in args.new_domains:
        idx = [i for i, r in enumerate(rows) if r["pair"] == name]
        rel["by_domain"][name] = {
            "n": len(idx),
            "corr_ld_z": corr([rows[i]["ld"] for i in idx], [rows[i]["z"] for i in idx]),
            "corr_ld_x": corr([rows[i]["ld"] for i in idx], [rows[i]["x"] for i in idx]),
            "corr_ld_mu": corr([rows[i]["ld"] for i in idx], [rows[i]["mu"] for i in idx]),
        }
    for task in ["positive_negative", "even_odd"]:
        idx = [i for i, r in enumerate(rows) if r["pair"] == task]
        obj["by_task"][task] = {
            "n": len(idx),
            "corr_ld_z": corr([rows[i]["ld"] for i in idx], [rows[i]["z"] for i in idx]),
            "corr_ld_objective": corr([rows[i]["ld"] for i in idx], [rows[i]["objective"] for i in idx]),
        }
    with (RESULTS / "domain_extension" / "domain_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez(
        RESULTS / "domain_extension" / "domain_activations_L33.npz",
        activations=H[LATE].astype(np.float16),
        pair=np.array([r["pair"] for r in rows]),
        condition=np.array([r["condition"] for r in rows]),
        z=np.array([r["z"] for r in rows], dtype=np.float32),
        ld=np.array([r["ld"] for r in rows], dtype=np.float32),
    )
    (RESULTS / "domain_extension" / "domain_metrics.json").write_text(json.dumps(rel, indent=2))
    (RESULTS / "domain_extension" / "objective_control_metrics.json").write_text(json.dumps(obj, indent=2))
    transfer = {"model_short": MODEL_SHORT, "layer": LATE, "alpha": args.alpha, "targets": args.new_domains, "sources": PRIMARY_PAIRS + args.new_domains, "matrix": {}}
    domain_dirs = {}
    for name in args.new_domains:
        idx = [i for i, r in enumerate(rows) if r["pair"] == name]
        z = np.array([rows[i]["z"] for i in idx], dtype=np.float64)
        Ha = H[LATE][idx].astype(np.float64)
        domain_dirs[name] = Ha[z > 1.0].mean(0) - Ha[z < -1.0].mean(0)
    source_dirs = {p: primal(p, LATE, "z") for p in PRIMARY_PAIRS}
    source_dirs.update(domain_dirs)
    for target in args.new_domains:
        target_rows = [r for r in rel_rows if r["pair"] == target]
        transfer["matrix"][target] = {}
        for source, vec in source_dirs.items():
            transfer["matrix"][target][source] = steering_slope(model, tok, target_rows, vec, LATE, args)
        print(f"[v13] domain-transfer target={target}: {transfer['matrix'][target]}", flush=True)
    (RESULTS / "domain_extension" / "new_domain_cross_pair_transfer.json").write_text(json.dumps(transfer, indent=2))


def plot_all() -> None:
    aff_path = RESULTS / "affine_shift" / "affine_shift_metrics.json"
    if aff_path.exists():
        aff = json.loads(aff_path.read_text())
        pairs = list(aff["by_pair"].keys())
        conds = sorted({c for p in pairs for c in aff["by_pair"][p].keys()})
        M = np.array([[aff["by_pair"][p].get(c, {}).get("corr_ld_z_eff", np.nan) for c in conds] for p in pairs])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(M, cmap="viridis", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(conds)), conds, rotation=35, ha="right")
        ax.set_yticks(range(len(pairs)), pairs)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=8, color="white" if M[i,j] < 0.45 else "black")
        ax.set_title("V13 affine/OOD corr(LD, z_eff)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGS / "affine_shift" / "affine_shift_corr_bars.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.5), sharey=True)
        if len(pairs) == 1:
            axes = [axes]
        for ax, pair in zip(axes, pairs):
            for cond, row in aff["by_pair"][pair].items():
                ax.scatter([cond], [row["corr_ld_z_eff"]], label=cond, s=35)
            ax.set_title(pair)
            ax.set_ylim(-1, 1)
            ax.tick_params(axis="x", labelrotation=70)
        fig.suptitle("Affine/OOD z correlation by condition")
        fig.tight_layout()
        fig.savefig(FIGS / "affine_shift" / "affine_shift_ld_vs_z.png", dpi=150)
        plt.close(fig)
    probe_path = RESULTS / "affine_shift" / "affine_shift_probe_transfer.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        pairs = list(probe["by_pair"].keys())
        conds = sorted({c for p in pairs for c in probe["by_pair"][p].keys()})
        M = np.array([[probe["by_pair"][p].get(c, {}).get("base_probe_corr_with_z_eff", np.nan) for c in conds] for p in pairs])
        heatmap(M, pairs, conds, "Base z probe transfer corr", FIGS / "affine_shift" / "affine_shift_probe_transfer.png")
    steer_path = RESULTS / "affine_shift" / "affine_shift_steering.json"
    if steer_path.exists():
        st = json.loads(steer_path.read_text())
        pairs = list(st["by_pair"].keys())
        conds = sorted({c for p in pairs for c in st["by_pair"][p].keys()})
        M = np.array([[st["by_pair"][p].get(c, {}).get("base_primal_z_slope", np.nan) for c in conds] for p in pairs])
        heatmap(M, pairs, conds, "Base primal_z steering slope", FIGS / "affine_shift" / "affine_shift_steering.png")
    x_path = RESULTS / "x_transfer" / "cross_pair_transfer_x_8x8.json"
    if x_path.exists():
        xres = json.loads(x_path.read_text())
        pairs = xres["pairs"]
        for fam, fname in [("primal_x_naive", "cross_pair_transfer_x_8x8_gemma2-9b.png"), ("primal_z", "cross_pair_transfer_z_vs_x_8x8_gemma2-9b.png")]:
            M = np.array([[xres["matrices"][fam][t][s] for s in pairs] for t in pairs])
            heatmap(M, pairs, pairs, f"{fam} transfer @ L33", FIGS / "x_transfer" / fname, diverge=True)
    analyze_top_logits()
    top_path = RESULTS / "top_logits" / "top_logit_group_scores.json"
    if top_path.exists():
        top = json.loads(top_path.read_text())["rows"]
        for fname, groups in [("semantic_mass_by_z.png", ["high", "low", "neutral", "ood"]), ("top_token_trajectories_by_z.png", ["high", "low", "extreme_high", "extreme_low"])]:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            for g in groups:
                xs, ys = [], []
                for r in top:
                    if r["condition"] in {"base", "world_ood", "target_ood"} and g in r["group_scores"]:
                        xs.append(r["z_bin"])
                        ys.append(r["group_scores"][g])
                if xs:
                    order = np.argsort(xs)
                    ax.plot(np.array(xs)[order], np.array(ys)[order], ".", alpha=0.45, label=g)
            ax.set_xlabel("z bin")
            ax.set_ylabel("top-k group logsumexp")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGS / "top_logits" / fname, dpi=150)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter([r["z_bin"] for r in top], [r.get("group_ld", np.nan) for r in top], s=8, alpha=0.5)
        ax.set_xlabel("z bin")
        ax.set_ylabel("group high-low score")
        fig.tight_layout()
        fig.savefig(FIGS / "top_logits" / "classic_ld_vs_group_ld.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ood = [r for r in top if r["condition"] in {"target_ood", "world_ood"}][:40]
        ax.axis("off")
        ax.text(0.01, 0.98, "\n".join([f"{r['pair']} {r['condition']} z={r['z_bin']:+.1f} groupLD={r.get('group_ld', np.nan):+.2f}" for r in ood]), va="top", family="monospace", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "top_logits" / "ood_top_tokens_examples.png", dpi=150)
        plt.close(fig)
    dom_path = RESULTS / "domain_extension" / "domain_metrics.json"
    obj_path = RESULTS / "domain_extension" / "objective_control_metrics.json"
    if dom_path.exists() and obj_path.exists():
        dom = json.loads(dom_path.read_text())
        obj = json.loads(obj_path.read_text())
        labels = list(dom["by_domain"].keys()) + list(obj["by_task"].keys())
        zvals = [dom["by_domain"][k]["corr_ld_z"] for k in dom["by_domain"]] + [obj["by_task"][k]["corr_ld_z"] for k in obj["by_task"]]
        oval = [np.nan] * len(dom["by_domain"]) + [obj["by_task"][k]["corr_ld_objective"] for k in obj["by_task"]]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.18, zvals, width=0.36, label="corr LD,z")
        ax.bar(x + 0.18, oval, width=0.36, label="corr LD,objective")
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylim(-1, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "domain_extension" / "objective_vs_relative_summary.png", dpi=150)
        fig.savefig(FIGS / "domain_extension" / "domain_corr_summary.png", dpi=150)
        plt.close(fig)
        act_path = RESULTS / "domain_extension" / "domain_activations_L33.npz"
        if act_path.exists():
            d = np.load(act_path)
            A = d["activations"].astype(np.float64)
            pair = d["pair"].astype(str)
            z = d["z"].astype(np.float64)
            domains = list(dom["by_domain"].keys())
            fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4), squeeze=False)
            for ax, name in zip(axes.ravel(), domains):
                idx = np.where(pair == name)[0]
                if len(idx) >= 3:
                    Y = PCA(n_components=2).fit_transform(A[idx])
                    sc = ax.scatter(Y[:, 0], Y[:, 1], c=z[idx], cmap="coolwarm", s=24)
                    ax.set_title(f"{name} PCA @ L33")
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    fig.colorbar(sc, ax=ax, label="z")
            fig.tight_layout()
            fig.savefig(FIGS / "domain_extension" / "domain_pca_panels.png", dpi=150)
            plt.close(fig)
        transfer_path = RESULTS / "domain_extension" / "new_domain_cross_pair_transfer.json"
        if transfer_path.exists():
            transfer = json.loads(transfer_path.read_text())
            M = np.array([[transfer["matrix"][t][s] for s in transfer["sources"]] for t in transfer["targets"]], dtype=np.float64)
            heatmap(M, transfer["targets"], transfer["sources"], "New-domain cross-pair steering @ L33", FIGS / "domain_extension" / "new_domain_cross_pair_transfer.png", diverge=True)


def heatmap(M, rows, cols, title, path, diverge=False):
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.8), max(4, len(rows) * 0.55)))
    if diverge:
        lim = max(0.01, float(np.nanmax(np.abs(M))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def placeholder_plot(path: Path, text: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="all", help="comma list: affine,x_transfer,domain,plot")
    ap.add_argument("--pairs", nargs="+", default=PRIMARY_PAIRS)
    ap.add_argument("--new-domains", nargs="+", default=["brightness", "temperature"])
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-seq", type=int, default=320)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--affine-prompts-per-condition", type=int, default=72)
    ap.add_argument("--transfer-prompts", type=int, default=72)
    ap.add_argument("--domain-prompts-per-task", type=int, default=80)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()
    sections = {"affine", "x_transfer", "domain", "plot"} if args.sections == "all" else set(args.sections.split(","))
    if sections - {"plot"}:
        print(f"[v13] loading {MODEL_ID}", flush=True)
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
        print("[v13] model loaded", flush=True)
    else:
        tok = model = None
    if "affine" in sections:
        experiment_affine(model, tok, args)
    if "x_transfer" in sections:
        experiment_x_transfer(model, tok, args)
    if "domain" in sections:
        experiment_domain(model, tok, args)
    if "plot" in sections:
        plot_all()
    print("[v13] complete", flush=True)


if __name__ == "__main__":
    main()
