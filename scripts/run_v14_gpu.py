#!/usr/bin/env python3
"""V14 GPU session: distribution, order, and OOD controls.

This script is deliberately restartable. Each section writes JSON and figures
under results/v14 and figures/v14. It reuses the v13 scoring/steering helpers
so that v14 remains compatible with the existing prompt and activation format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # Allow CPU-only plotting/row inspection environments.
    torch = None
    AutoModelForCausalLM = AutoTokenizer = None

try:
    from run_v13_gpu import (
        ALL_PAIRS,
        LAYERS,
        LATE,
        LOG_SPACE_PAIRS,
        MODEL_ID,
        MODEL_SHORT,
        PAIR_BY_NAME,
        REPO,
        corr,
        first_token_id,
        fmt_num,
        get_layers,
        run_prompts,
        unit,
    )
except ModuleNotFoundError:
    # CPU-only fallback for plotting and prompt-design checks.
    import sys

    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
    from _token_utils import first_token_id  # type: ignore
    from extract_v4_adjpairs import LOG_SPACE_PAIRS, PAIRS, fmt_num  # type: ignore

    ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
    LAYERS = [25, 33]
    LATE = 33
    MODEL_ID = "google/gemma-2-9b"
    MODEL_SHORT = "gemma2-9b"
    PAIR_BY_NAME = {p.name: p for p in PAIRS}

    def corr(a, b) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    def unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v.astype(np.float64) if n < 1e-12 else (v / n).astype(np.float64)

    def get_layers(model):
        raise RuntimeError("get_layers requires GPU dependencies")

    def run_prompts(*args, **kwargs):
        raise RuntimeError("run_prompts requires GPU dependencies")


RESULTS = REPO / "results" / "v14"
FIGS = REPO / "figures" / "v14"
for sub in ["distribution", "order", "affine_ood"]:
    (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    (FIGS / sub).mkdir(parents=True, exist_ok=True)

DIST_KINDS = ["normal", "uniform", "beta_u", "beta_low", "beta_high", "bimodal"]
ORDER_KINDS = ["random", "ascending", "descending", "alternating_low_high", "near_target_first", "near_target_last"]
OOD_CONDITIONS = [
    "base",
    "parallel_shift_high",
    "world_extreme_high",
    "world_extreme_low",
    "target_extreme_high",
    "target_extreme_low",
]

PAIR_SHIFTS = {
    "height": 150.0,
    "age": 100.0,
    "weight": 150.0,
    "size": 150.0,
    "speed": 300.0,
    "wealth": 100.0,  # multiplicative, handled separately
    "experience": 100.0,
    "bmi_abs": 50.0,
}

PAIR_WORLD_HIGH = {
    "height": 320.0,
    "age": 150.0,
    "weight": 250.0,
    "size": 180.0,
    "speed": 500.0,
    "wealth": 1_000_000_000_000.0,
    "experience": 150.0,
    "bmi_abs": 80.0,
}

PAIR_WORLD_LOW = {
    "height": 20.0,
    "age": 3.0,
    "weight": 10.0,
    "size": 20.0,
    "speed": 5.0,
    "wealth": 500.0,
    "experience": 1.0,
    "bmi_abs": 8.0,
}


def stable_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def choose_cells(pair: str, n: int) -> list[dict]:
    rows = []
    seen = set()
    for r in load_v11_trials(pair):
        if int(r.get("cell_seed", -1)) != 0:
            continue
        key = (round(float(r["x"]), 4), round(float(r["z"]), 4))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    if n and len(rows) > n:
        idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
        rows = [rows[int(i)] for i in idx]
    return rows


def load_v11_trials(pair: str) -> list[dict]:
    path = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.open()]
    return synthetic_cells(pair, 12, 9)


def synthetic_cells(pair: str, n_x: int, n_z: int) -> list[dict]:
    """Fallback cells for branches without committed V11 prompt JSONL files."""
    spec = PAIR_BY_NAME[pair]
    xs = np.linspace(min(spec.target_values), max(spec.target_values), n_x)
    zs = np.linspace(-3.0, 3.0, n_z)
    rows = []
    for x in xs:
        for z in zs:
            if pair in LOG_SPACE_PAIRS:
                mu = float(x / (spec.sigma ** z))
            else:
                mu = float(x - z * spec.sigma)
            if x <= 0 or mu <= 0:
                continue
            rows.append(
                {
                    "x": float(x),
                    "mu": mu,
                    "sigma": float(spec.sigma),
                    "z": float(z),
                    "cell_seed": 0,
                    "seed": 0,
                }
            )
    return rows


def z_value(pair: str, x: float, mu: float, sigma: float) -> float:
    if pair in LOG_SPACE_PAIRS:
        return (math.log(x) - math.log(mu)) / math.log(sigma)
    return (x - mu) / sigma


def x_from_z(pair: str, mu: float, sigma: float, z: float) -> float:
    if pair in LOG_SPACE_PAIRS:
        return float(mu * (sigma ** z))
    return float(mu + z * sigma)


def standardized_samples(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "normal":
        vals = rng.normal(0, 1, n)
    elif kind == "uniform":
        vals = rng.uniform(-math.sqrt(3), math.sqrt(3), n)
    elif kind == "beta_u":
        vals = rng.beta(0.45, 0.45, n)
    elif kind == "beta_low":
        vals = rng.beta(2.0, 5.0, n)
    elif kind == "beta_high":
        vals = rng.beta(5.0, 2.0, n)
    elif kind == "bimodal":
        side = rng.choice([-1.0, 1.0], size=n)
        vals = side * 1.4 + rng.normal(0, 0.35, n)
    else:
        raise ValueError(kind)
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals - vals.mean()
    sd = vals.std(ddof=1)
    return vals / sd if sd > 1e-8 else vals


def context_values(pair: str, mu: float, sigma: float, kind: str, n: int, seed: int, allow_invalid: bool = False) -> np.ndarray | None:
    floor = 1.0 if pair in {"height", "age", "weight", "size", "speed", "experience", "wealth"} else 0.1
    for attempt in range(20):
        rng = np.random.default_rng(seed + attempt * 9973)
        z = standardized_samples(kind, n, rng)
        if pair in LOG_SPACE_PAIRS:
            vals = mu * (sigma ** z)
        else:
            vals = mu + sigma * z
        if allow_invalid or float(np.min(vals)) > floor:
            return vals.astype(np.float64)
    return None


def format_items(pair_name: str, values: Iterable[float]) -> list[str]:
    pair = PAIR_BY_NAME[pair_name]
    vals = list(values)
    if pair.name == "height":
        return [f"Person {i + 1}: {int(round(v))} cm" for i, v in enumerate(vals)]
    if pair.name == "age":
        return [f"Person {i + 1}: {int(round(v))} years old" for i, v in enumerate(vals)]
    if pair.name == "weight":
        return [f"Person {i + 1}: {int(round(v))} kg" for i, v in enumerate(vals)]
    if pair.name == "size":
        return [f"Object {i + 1}: {fmt_num(v)} cm across" for i, v in enumerate(vals)]
    if pair.name == "speed":
        return [f"Vehicle {i + 1}: {int(round(v))} km/h" for i, v in enumerate(vals)]
    if pair.name == "wealth":
        return [f"Person {i + 1} earns ${int(round(v))}/year" for i, v in enumerate(vals)]
    if pair.name == "experience":
        return [f"Worker {i + 1}: {fmt_num(v)} years experience" for i, v in enumerate(vals)]
    if pair.name == "bmi_abs":
        return [f"Person {i + 1}: BMI {v:.1f}" for i, v in enumerate(vals)]
    raise KeyError(pair_name)


def make_prompt(pair_name: str, x: float, values: Iterable[float]) -> str:
    pair = PAIR_BY_NAME[pair_name]
    items = format_items(pair_name, values)
    return pair.format_prompt_implicit.format(items="\n".join(items), n_last=len(items) + 1, x_str=fmt_num(x))


def row_from_context(pair: str, condition: str, x: float, mu: float, sigma: float, z: float, values: np.ndarray, **extra) -> dict:
    emp_mu = float(np.mean(values))
    emp_sigma = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    if pair in LOG_SPACE_PAIRS:
        logs = np.log(np.asarray(values, dtype=np.float64))
        log_sd = float(np.std(logs, ddof=1))
        z_emp = (math.log(x) - float(np.mean(logs))) / log_sd if log_sd > 1e-8 else float("nan")
        emp_mu_for_log = float(math.exp(np.mean(logs)))
        emp_sigma_for_log = float(math.exp(log_sd))
    else:
        z_emp = z_value(pair, x, emp_mu, emp_sigma) if emp_sigma > 1e-8 else float("nan")
        emp_mu_for_log = float("nan")
        emp_sigma_for_log = float("nan")
    rank = float(np.mean(values <= x))
    p = PAIR_BY_NAME[pair]
    return {
        "pair": pair,
        "condition": condition,
        "x": float(x),
        "mu": float(mu),
        "sigma": float(sigma),
        "z": float(z),
        "empirical_mu": emp_mu,
        "empirical_sigma": emp_sigma,
        "empirical_log_mu": emp_mu_for_log,
        "empirical_log_sigma_factor": emp_sigma_for_log,
        "z_empirical": float(z_emp),
        "target_rank": rank,
        "low_word": p.low_word,
        "high_word": p.high_word,
        "prompt": make_prompt(pair, x, values),
        **extra,
    }


def build_distribution_rows(args) -> list[dict]:
    rows = []
    for pair in args.pairs:
        for cell in choose_cells(pair, args.dist_cells_per_pair):
            x, mu, sigma, z = float(cell["x"]), float(cell["mu"]), float(cell["sigma"]), float(cell["z"])
            for kind in DIST_KINDS:
                for seed_idx in range(args.dist_seeds):
                    seed = stable_seed("dist", pair, x, z, kind, seed_idx)
                    vals = context_values(pair, mu, sigma, kind, args.context_n, seed)
                    if vals is None:
                        continue
                    cell_id = f"{pair}|{x:.6g}|{z:.3f}|{seed_idx}"
                    rows.append(row_from_context(pair, kind, x, mu, sigma, z, vals, dist_kind=kind, seed_index=seed_idx, cell_id=cell_id))
    return rows


def order_values(kind: str, values: np.ndarray, x: float, seed: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if kind == "ascending":
        return np.sort(vals)
    if kind == "descending":
        return np.sort(vals)[::-1]
    if kind == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(vals)
    if kind == "alternating_low_high":
        sorted_vals = list(np.sort(vals))
        out = []
        while sorted_vals:
            out.append(sorted_vals.pop(0))
            if sorted_vals:
                out.append(sorted_vals.pop(-1))
        return np.array(out, dtype=np.float64)
    by_dist = vals[np.argsort(np.abs(vals - x))]
    if kind == "near_target_first":
        return by_dist
    if kind == "near_target_last":
        return by_dist[::-1]
    raise ValueError(kind)


def build_order_rows(args) -> list[dict]:
    rows = []
    for pair in args.pairs:
        for cell in choose_cells(pair, args.order_cells_per_pair):
            x, mu, sigma, z = float(cell["x"]), float(cell["mu"]), float(cell["sigma"]), float(cell["z"])
            for seed_idx in range(args.order_seeds):
                base_seed = stable_seed("order-base", pair, x, z, seed_idx)
                base_vals = context_values(pair, mu, sigma, "normal", args.context_n, base_seed)
                if base_vals is None:
                    continue
                for kind in ORDER_KINDS:
                    vals = order_values(kind, base_vals, x, stable_seed("order", pair, x, z, seed_idx, kind))
                    cell_id = f"{pair}|{x:.6g}|{z:.3f}|{seed_idx}"
                    rows.append(row_from_context(pair, kind, x, mu, sigma, z, vals, order_kind=kind, seed_index=seed_idx, cell_id=cell_id))
    return rows


def affine_transform(pair: str, condition: str, x: float, mu: float, sigma: float, z: float) -> tuple[float, float, float, float] | None:
    if condition == "base":
        return x, mu, sigma, z
    if condition == "parallel_shift_high":
        if pair in LOG_SPACE_PAIRS:
            factor = PAIR_SHIFTS[pair]
            return x * factor, mu * factor, sigma, z
        delta = PAIR_SHIFTS[pair]
        return x + delta, mu + delta, sigma, z
    if condition == "world_extreme_high":
        mu2 = PAIR_WORLD_HIGH[pair]
        return x_from_z(pair, mu2, sigma, z), mu2, sigma, z
    if condition == "world_extreme_low":
        mu2 = PAIR_WORLD_LOW[pair]
        x2 = x_from_z(pair, mu2, sigma, z)
        if x2 <= 0:
            return None
        return x2, mu2, sigma, z
    if condition == "target_extreme_high":
        z2 = 5.0
        return x_from_z(pair, mu, sigma, z2), mu, sigma, z2
    if condition == "target_extreme_low":
        z2 = -5.0
        x2 = x_from_z(pair, mu, sigma, z2)
        if x2 <= 0:
            return None
        return x2, mu, sigma, z2
    raise ValueError(condition)


def build_affine_rows(args) -> list[dict]:
    rows = []
    for pair in args.pairs:
        for cell in choose_cells(pair, args.ood_cells_per_pair):
            x, mu, sigma, z = float(cell["x"]), float(cell["mu"]), float(cell["sigma"]), float(cell["z"])
            for n_context in args.ood_context_ns:
                for seed_idx in range(args.ood_seeds):
                    for condition in OOD_CONDITIONS:
                        tr = affine_transform(pair, condition, x, mu, sigma, z)
                        if tr is None:
                            continue
                        x2, mu2, sigma2, z2 = tr
                        vals = context_values(pair, mu2, sigma2, "normal", n_context, stable_seed("ood", pair, x, z, condition, n_context, seed_idx))
                        if vals is None:
                            continue
                        cell_id = f"{pair}|{x:.6g}|{z:.3f}|{n_context}|{seed_idx}"
                        rows.append(
                            row_from_context(
                                pair,
                                condition,
                                x2,
                                mu2,
                                sigma2,
                                z2,
                                vals,
                                ood_condition=condition,
                                n_context=n_context,
                                seed_index=seed_idx,
                                cell_id=cell_id,
                                base_x=x,
                                base_mu=mu,
                                base_z=z,
                            )
                        )
    return rows


def add_outputs(rows: list[dict], ld: np.ndarray, H: dict[int, np.ndarray]) -> None:
    for i, row in enumerate(rows):
        row["ld"] = float(ld[i])
    for layer in H:
        assert H[layer].shape[0] == len(rows)


def group_metrics(rows: list[dict], group_keys: list[str]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = {}
    for key, group in groups.items():
        d = out
        for part in key[:-1]:
            d = d.setdefault(str(part), {})
        name = str(key[-1])
        ld = [r["ld"] for r in group]
        d[name] = {
            "n": len(group),
            "corr_ld_z": corr(ld, [r["z"] for r in group]),
            "corr_ld_base_z": corr(ld, [r.get("base_z", r["z"]) for r in group]),
            "corr_ld_z_empirical": corr(ld, [r["z_empirical"] for r in group]),
            "corr_ld_rank": corr(ld, [r["target_rank"] for r in group]),
            "corr_ld_x": corr(ld, [r["x"] for r in group]),
            "mean_ld": float(np.mean(ld)),
        }
    return out


def matched_delta_metrics(rows: list[dict], condition_key: str, baseline: str, group_key: str = "pair") -> dict:
    by_cell: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        by_cell.setdefault((str(row[group_key]), str(row["cell_id"])), {})[str(row[condition_key])] = row
    deltas: dict[str, dict[str, list[float]]] = {}
    for (group, _cell), by_cond in by_cell.items():
        base = by_cond.get(baseline)
        if base is None:
            continue
        for cond, row in by_cond.items():
            if cond == baseline:
                continue
            deltas.setdefault(group, {}).setdefault(cond, []).append(float(row["ld"] - base["ld"]))
    out: dict[str, dict] = {}
    for group, by_cond in deltas.items():
        out[group] = {}
        for cond, vals in by_cond.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[group][cond] = {
                "n_matched": int(arr.size),
                "mean_delta_ld": float(np.mean(arr)),
                "mean_abs_delta_ld": float(np.mean(np.abs(arr))),
                "se_delta_ld": float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else None,
            }
    return out


def direction_diagnostics(rows: list[dict], H: dict[int, np.ndarray], base_condition: str, condition_key: str) -> dict:
    out: dict[str, dict] = {}
    for pair in sorted({r["pair"] for r in rows}):
        idx_pair = [i for i, r in enumerate(rows) if r["pair"] == pair]
        base_idx = [i for i in idx_pair if rows[i][condition_key] == base_condition]
        if len(base_idx) < 8:
            continue
        z_base = np.array([rows[i]["z"] for i in base_idx], dtype=np.float64)
        Hb = H[LATE][base_idx].astype(np.float64)
        pz_base = Hb[z_base > 1.0].mean(0) - Hb[z_base < -1.0].mean(0)
        probe = Ridge(alpha=1.0).fit(Hb, z_base)
        out[pair] = {}
        for condition in sorted({rows[i][condition_key] for i in idx_pair}):
            idx = [i for i in idx_pair if rows[i][condition_key] == condition]
            zc = np.array([rows[i]["z"] for i in idx], dtype=np.float64)
            Hc = H[LATE][idx].astype(np.float64)
            if (zc > 1.0).any() and (zc < -1.0).any():
                pc = Hc[zc > 1.0].mean(0) - Hc[zc < -1.0].mean(0)
                cos_val = float(unit(pz_base) @ unit(pc))
            else:
                cos_val = float("nan")
            out[pair][condition] = {
                "cos_primal_z_base_condition": cos_val,
                "base_probe_corr_z": corr(probe.predict(Hc), zc),
            }
    return out


def write_rows_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            slim = {k: v for k, v in row.items() if k != "prompt"}
            f.write(json.dumps(clean_json(slim), allow_nan=False) + "\n")


def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(clean_json(obj), indent=2, allow_nan=False))


def experiment_distribution(model, tok, args) -> None:
    rows = build_distribution_rows(args)
    print(f"[v14] distribution prompts={len(rows)} context_n={args.context_n}", flush=True)
    ld, H, _ = run_prompts(model, tok, rows, LAYERS, args.batch_size, args.max_seq, top_k=0)
    add_outputs(rows, ld, H)
    result = {
        "model_id": MODEL_ID,
        "model_short": MODEL_SHORT,
        "pairs": args.pairs,
        "context_n": args.context_n,
        "distribution_kinds": DIST_KINDS,
        "metrics_by_pair_distribution": group_metrics(rows, ["pair", "dist_kind"]),
        "matched_delta_vs_normal": matched_delta_metrics(rows, "dist_kind", "normal"),
        "direction_diagnostics": direction_diagnostics(rows, H, "normal", "dist_kind"),
    }
    write_json(RESULTS / "distribution" / "distribution_metrics.json", result)
    write_rows_jsonl(rows, RESULTS / "distribution" / "distribution_rows.jsonl")


def experiment_order(model, tok, args) -> None:
    rows = build_order_rows(args)
    print(f"[v14] order prompts={len(rows)} context_n={args.context_n}", flush=True)
    ld, H, _ = run_prompts(model, tok, rows, LAYERS, args.batch_size, args.max_seq, top_k=0)
    add_outputs(rows, ld, H)
    result = {
        "model_id": MODEL_ID,
        "model_short": MODEL_SHORT,
        "pairs": args.pairs,
        "context_n": args.context_n,
        "order_kinds": ORDER_KINDS,
        "metrics_by_pair_order": group_metrics(rows, ["pair", "order_kind"]),
        "matched_delta_vs_random": matched_delta_metrics(rows, "order_kind", "random"),
        "direction_diagnostics": direction_diagnostics(rows, H, "random", "order_kind"),
    }
    write_json(RESULTS / "order" / "order_metrics.json", result)
    write_rows_jsonl(rows, RESULTS / "order" / "order_rows.jsonl")


def experiment_affine_ood(model, tok, args) -> None:
    rows = build_affine_rows(args)
    print(f"[v14] affine/OOD prompts={len(rows)}", flush=True)
    ld, H, top = run_prompts(model, tok, rows, LAYERS, args.batch_size, args.max_seq, top_k=args.top_k)
    add_outputs(rows, ld, H)
    result = {
        "model_id": MODEL_ID,
        "model_short": MODEL_SHORT,
        "pairs": args.pairs,
        "conditions": OOD_CONDITIONS,
        "context_ns": args.ood_context_ns,
        "metrics_by_pair_condition_n": group_metrics(rows, ["pair", "ood_condition", "n_context"]),
        "matched_delta_vs_base": matched_delta_metrics(rows, "ood_condition", "base"),
        "direction_diagnostics": direction_diagnostics(rows, H, "base", "ood_condition"),
        "steering_by_pair_condition": {},
    }
    for pair in args.pairs:
        pair_idx = [i for i, r in enumerate(rows) if r["pair"] == pair and r.get("ood_condition") == "base"]
        if not pair_idx:
            continue
        bz = np.array([rows[i]["z"] for i in pair_idx], dtype=np.float64)
        Hb = H[LATE][pair_idx].astype(np.float64)
        if not ((bz > 1.0).any() and (bz < -1.0).any()):
            continue
        base_pz = Hb[bz > 1.0].mean(0) - Hb[bz < -1.0].mean(0)
        result["steering_by_pair_condition"][pair] = {}
        for condition in sorted({r["ood_condition"] for r in rows if r["pair"] == pair}):
            cond_rows = [r for r in rows if r["pair"] == pair and r["ood_condition"] == condition and r.get("n_context") == max(args.ood_context_ns)]
            if cond_rows:
                result["steering_by_pair_condition"][pair][condition] = {
                    "base_primal_z_slope": steering_slope_local(model, tok, cond_rows, base_pz, LATE, args)
                }
    write_json(RESULTS / "affine_ood" / "affine_ood_metrics.json", result)
    write_rows_jsonl(rows, RESULTS / "affine_ood" / "affine_ood_rows.jsonl")
    with (RESULTS / "affine_ood" / "affine_ood_top_logits.jsonl").open("w") as f:
        for tr in top:
            row = rows[tr["row_index"]]
            f.write(json.dumps(clean_json({**{k: row[k] for k in ["pair", "condition", "ood_condition", "n_context", "x", "mu", "sigma", "z", "z_empirical"]}, **tr}), allow_nan=False) + "\n")


def steer_ld_local(model, tok, rows: list[dict], direction: np.ndarray, layer: int, alpha: float, batch_size: int, max_seq: int) -> np.ndarray:
    layers = get_layers(model)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        d = torch.as_tensor(unit(direction), dtype=h.dtype, device=h.device)
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


def steering_slope_local(model, tok, rows: list[dict], direction: np.ndarray, layer: int, args) -> float:
    pos = steer_ld_local(model, tok, rows, direction, layer, args.alpha, args.batch_size, args.max_seq)
    neg = steer_ld_local(model, tok, rows, direction, layer, -args.alpha, args.batch_size, args.max_seq)
    return float((pos - neg).mean() / (2.0 * args.alpha))


def plot_distribution() -> None:
    path = RESULTS / "distribution" / "distribution_metrics.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    pairs = d["pairs"]
    mat = np.array([[d["metrics_by_pair_distribution"][p][kind]["corr_ld_z"] for kind in DIST_KINDS] for p in pairs], dtype=float)
    heatmap(mat, pairs, DIST_KINDS, "V14 distribution shape: corr(LD, population z)", FIGS / "distribution" / "distribution_corr_bars.png", vmin=-1, vmax=1)

    # Visualize one standardized draw for human inspection.
    fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)
    for ax, kind in zip(axes.flat, DIST_KINDS):
        vals = standardized_samples(kind, 1000, np.random.default_rng(140))
        ax.hist(vals, bins=30, color="#4C78A8", alpha=0.85)
        ax.set_title(kind)
    fig.suptitle("Distribution-shape controls after standardization")
    fig.tight_layout()
    fig.savefig(FIGS / "distribution" / "distribution_shape_examples.png", dpi=150)
    plt.close(fig)

    diag = d.get("direction_diagnostics", {})
    M = np.array([[diag.get(p, {}).get(kind, {}).get("cos_primal_z_base_condition", np.nan) for kind in DIST_KINDS] for p in pairs], dtype=float)
    heatmap(M, pairs, DIST_KINDS, "cos(primal_z normal, condition)", FIGS / "distribution" / "distribution_primal_alignment.png", vmin=-1, vmax=1)

    delta = d.get("matched_delta_vs_normal", {})
    delta_cols = [k for k in DIST_KINDS if k != "normal"]
    D = np.array([[delta.get(p, {}).get(k, {}).get("mean_abs_delta_ld", np.nan) for k in delta_cols] for p in pairs], dtype=float)
    heatmap(D, pairs, delta_cols, "Mean |LD shift| vs normal at matched cells", FIGS / "distribution" / "distribution_rank_vs_z.png", vmin=0, vmax=None)


def plot_order() -> None:
    path = RESULTS / "order" / "order_metrics.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    pairs = d["pairs"]
    mat = np.array([[d["metrics_by_pair_order"][p][kind]["corr_ld_z"] for kind in ORDER_KINDS] for p in pairs], dtype=float)
    heatmap(mat, pairs, ORDER_KINDS, "V14 order sensitivity: corr(LD,z)", FIGS / "order" / "order_corr_bars.png", vmin=-1, vmax=1)
    diag = d.get("direction_diagnostics", {})
    M = np.array([[diag.get(p, {}).get(kind, {}).get("cos_primal_z_base_condition", np.nan) for kind in ORDER_KINDS] for p in pairs], dtype=float)
    heatmap(M, pairs, ORDER_KINDS, "cos(primal_z random-order, order)", FIGS / "order" / "order_primal_alignment.png", vmin=-1, vmax=1)
    delta = d.get("matched_delta_vs_random", {})
    delta_cols = [k for k in ORDER_KINDS if k != "random"]
    D = np.array([[delta.get(p, {}).get(k, {}).get("mean_abs_delta_ld", np.nan) for k in delta_cols] for p in pairs], dtype=float)
    heatmap(D, pairs, delta_cols, "Mean |LD shift| vs random order", FIGS / "order" / "order_ld_shift.png", vmin=0, vmax=None)


def plot_affine_ood() -> None:
    path = RESULTS / "affine_ood" / "affine_ood_metrics.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    pairs = d["pairs"]
    n_max = str(max(d["context_ns"]))
    cols = OOD_CONDITIONS
    mat = np.array([[d["metrics_by_pair_condition_n"].get(p, {}).get(cond, {}).get(n_max, {}).get("corr_ld_z") for cond in cols] for p in pairs], dtype=float)
    heatmap(mat, pairs, cols, f"V14 affine/OOD corr(LD,z), N={n_max}", FIGS / "affine_ood" / "affine_ood_corr_heatmap.png", vmin=-1, vmax=1)

    diag = d.get("direction_diagnostics", {})
    M = np.array([[diag.get(p, {}).get(cond, {}).get("cos_primal_z_base_condition", np.nan) for cond in cols] for p in pairs], dtype=float)
    heatmap(M, pairs, cols, "cos(primal_z base, condition)", FIGS / "affine_ood" / "affine_ood_primal_alignment.png", vmin=-1, vmax=1)
    delta = d.get("matched_delta_vs_base", {})
    delta_cols = [k for k in cols if k != "base"]
    D = np.array([[delta.get(p, {}).get(k, {}).get("mean_abs_delta_ld", np.nan) for k in delta_cols] for p in pairs], dtype=float)
    heatmap(D, pairs, delta_cols, "Mean |LD shift| vs base at matched cells", FIGS / "affine_ood" / "affine_ood_by_context_n.png", vmin=0, vmax=None)
    steering = d.get("steering_by_pair_condition", {})
    S = np.array([[steering.get(p, {}).get(cond, {}).get("base_primal_z_slope", np.nan) for cond in cols] for p in pairs], dtype=float)
    heatmap(S, pairs, cols, "Base primal_z steering slope by OOD condition", FIGS / "affine_ood" / "affine_ood_steering.png", vmin=None, vmax=None)
    plot_ood_top_tokens()
    write_summary(d)


def simple_token_group(token: str) -> str:
    s = token.strip().lower()
    if s in {"tall", "old", "heavy", "big", "fast", "rich", "expert", "obese"}:
        return "high_adj"
    if s in {"short", "young", "light", "small", "slow", "poor", "novice", "thin"}:
        return "low_adj"
    if s in {"average", "normal", "typical", "moderate", "medium"}:
        return "neutral"
    if s in {"giant", "huge", "tiny", "impossible", "unusual", "weird", "extreme"}:
        return "ood_semantic"
    return "other"


def plot_ood_top_tokens() -> None:
    path = RESULTS / "affine_ood" / "affine_ood_top_logits.jsonl"
    if not path.exists():
        return
    counts: dict[tuple[str, str], dict[str, int]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if int(r.get("rank", 0)) != 1:
                continue
            key = (r["pair"], r["ood_condition"])
            counts.setdefault(key, {})
            group = simple_token_group(r.get("token", ""))
            counts[key][group] = counts[key].get(group, 0) + 1
    pairs = sorted({k[0] for k in counts})
    cols = OOD_CONDITIONS
    M = np.zeros((len(pairs), len(cols)), dtype=float)
    for i, pair in enumerate(pairs):
        for j, cond in enumerate(cols):
            by_group = counts.get((pair, cond), {})
            total = sum(by_group.values())
            M[i, j] = by_group.get("ood_semantic", 0) / total if total else np.nan
    heatmap(M, pairs, cols, "Top-1 OOD-semantic token fraction", FIGS / "affine_ood" / "affine_ood_top_tokens.png", vmin=0, vmax=1)


def heatmap(M: np.ndarray, rows: list[str], cols: list[str], title: str, path: Path, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(cols) * 1.1), max(4, len(rows) * 0.55)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(affine: dict | None = None) -> None:
    lines = [
        "# V14 Auto Summary",
        "",
        "This file is generated by `scripts/run_v14_gpu.py --sections plot`.",
        "Treat it as a first-pass interpretation checklist, not final paper prose.",
        "",
    ]
    for name, path in [
        ("distribution", RESULTS / "distribution" / "distribution_metrics.json"),
        ("order", RESULTS / "order" / "order_metrics.json"),
        ("affine_ood", RESULTS / "affine_ood" / "affine_ood_metrics.json"),
    ]:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        lines.append(f"## {name}")
        if name == "distribution":
            metrics = d.get("metrics_by_pair_distribution", {})
            for pair, by_cond in metrics.items():
                vals = [v.get("corr_ld_z") for v in by_cond.values() if isinstance(v.get("corr_ld_z"), (int, float))]
                if vals:
                    lines.append(f"- {pair}: corr(LD,z) range across distributions = {min(vals):+.3f} to {max(vals):+.3f}.")
        elif name == "order":
            delta = d.get("matched_delta_vs_random", {})
            for pair, by_cond in delta.items():
                vals = [v.get("mean_abs_delta_ld") for v in by_cond.values() if isinstance(v.get("mean_abs_delta_ld"), (int, float))]
                if vals:
                    lines.append(f"- {pair}: max mean |LD shift| from order = {max(vals):.3f}.")
        elif name == "affine_ood":
            metrics = d.get("metrics_by_pair_condition_n", {})
            for pair, by_cond in metrics.items():
                vals = []
                for by_n in by_cond.values():
                    for v in by_n.values():
                        val = v.get("corr_ld_z")
                        if isinstance(val, (int, float)):
                            vals.append(val)
                if vals:
                    lines.append(f"- {pair}: corr(LD,z) range across OOD/N = {min(vals):+.3f} to {max(vals):+.3f}.")
        lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines))


def plot_all() -> None:
    plot_distribution()
    plot_order()
    plot_affine_ood()
    write_summary()


def load_model():
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("V14 GPU sections require torch and transformers. Use --sections plot for CPU-only plotting.")
    print(f"[v14] loading {MODEL_ID}", flush=True)
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
    print("[v14] model loaded", flush=True)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="all", help="comma list: distribution,order,affine_ood,plot")
    ap.add_argument("--pairs", nargs="+", default=ALL_PAIRS)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-seq", type=int, default=640)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--context-n", type=int, default=31)
    ap.add_argument("--dist-cells-per-pair", type=int, default=96)
    ap.add_argument("--dist-seeds", type=int, default=4)
    ap.add_argument("--order-cells-per-pair", type=int, default=72)
    ap.add_argument("--order-seeds", type=int, default=4)
    ap.add_argument("--ood-cells-per-pair", type=int, default=72)
    ap.add_argument("--ood-seeds", type=int, default=3)
    ap.add_argument("--ood-context-ns", nargs="+", type=int, default=[5, 15, 31])
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    sections = {"distribution", "order", "affine_ood", "plot"} if args.sections == "all" else set(args.sections.split(","))
    model = tok = None
    if sections - {"plot"}:
        model, tok = load_model()
    if "distribution" in sections:
        experiment_distribution(model, tok, args)
    if "order" in sections:
        experiment_order(model, tok, args)
    if "affine_ood" in sections:
        experiment_affine_ood(model, tok, args)
    if "plot" in sections:
        plot_all()
    print("[v14] complete", flush=True)


if __name__ == "__main__":
    main()
