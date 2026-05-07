#!/usr/bin/env python3
"""V14.1 cleanup runner.

V14.1 keeps the V14 robustness experiments but fixes two paper-facing issues:

* affine/OOD low-world contexts are generated with valid support by construction;
* order rows include local-context z diagnostics;
* Figure 5 keeps the original V12 2 x 3 layout while replacing probe_z steering
  with primal_x when available.

GPU sections write fresh rows/metrics under results/v14_1. Plot-only mode can
also read existing V14 artifacts to generate the V14.1 story figures.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_v14_gpu as v14


RESULTS = REPO / "results" / "v14_1"
FIGS = REPO / "figures" / "v14_1"
V14_RESULTS = REPO / "results" / "v14"

for sub in ["affine_ood", "order", "distribution", "fig5"]:
    (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    (FIGS / sub).mkdir(parents=True, exist_ok=True)

PAIRS = v14.ALL_PAIRS
OOD_CONDITIONS = v14.OOD_CONDITIONS
ORDER_KINDS = v14.ORDER_KINDS
DIST_KINDS = v14.DIST_KINDS
LOG_SPACE_PAIRS = v14.LOG_SPACE_PAIRS
MODEL_ID = v14.MODEL_ID
MODEL_SHORT = v14.MODEL_SHORT
LAYERS = v14.LAYERS
LATE = v14.LATE
DEFAULT_FIG5_LAYERS = [0, 1, 3, 5, 7, 10, 13, 14, 17, 21, 25, 29, 33, 37, 41]


def floor_for_pair(pair: str) -> float:
    return 0.1 if pair == "bmi_abs" else 1.0


def local_z(pair: str, x: float, values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size < 2:
        return float("nan")
    if pair in LOG_SPACE_PAIRS:
        if x <= 0 or np.any(vals <= 0):
            return float("nan")
        logs = np.log(vals)
        sd = float(np.std(logs, ddof=1))
        return (math.log(x) - float(np.mean(logs))) / sd if sd > 1e-8 else float("nan")
    sd = float(np.std(vals, ddof=1))
    return (x - float(np.mean(vals))) / sd if sd > 1e-8 else float("nan")


def bounded_standardized_normal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = np.clip(rng.normal(0.0, 1.0, n), -2.25, 2.25)
    vals = vals - vals.mean()
    sd = vals.std(ddof=1)
    return vals / sd if sd > 1e-8 else vals


def positive_context_and_target(
    pair: str,
    mu: float,
    sigma: float,
    z: float,
    n: int,
    seed: int,
) -> tuple[float, float, np.ndarray, str]:
    """Return x, effective_sigma, valid context values, sampling note."""
    if pair in LOG_SPACE_PAIRS:
        samples = bounded_standardized_normal(n, seed)
        vals = mu * (sigma ** samples)
        return float(mu * (sigma ** z)), float(sigma), vals.astype(np.float64), "log_space_bounded_normal"

    samples = bounded_standardized_normal(n, seed)
    floor = floor_for_pair(pair)
    min_sample = float(np.min(samples))
    lower_need = max(0.0, -min_sample, -float(z))
    sigma_eff = float(sigma)
    note = "bounded_normal"
    if lower_need > 0 and mu - sigma_eff * lower_need <= floor:
        sigma_eff = max((mu - floor) / (lower_need + 0.25), 1e-6)
        note = "bounded_normal_sigma_shrunk"
    vals = mu + sigma_eff * samples
    x = mu + sigma_eff * z
    if float(np.min(vals)) <= floor or x <= floor:
        raise ValueError(f"invalid support after shrink: {pair} mu={mu} sigma={sigma_eff} z={z}")
    return float(x), sigma_eff, vals.astype(np.float64), note


def affine_mu_sigma_z(pair: str, condition: str, x: float, mu: float, sigma: float, z: float) -> tuple[float, float, float]:
    if condition == "base":
        return mu, sigma, z
    if condition == "parallel_shift_high":
        if pair in LOG_SPACE_PAIRS:
            return mu * v14.PAIR_SHIFTS[pair], sigma, z
        return mu + v14.PAIR_SHIFTS[pair], sigma, z
    if condition == "world_extreme_high":
        return v14.PAIR_WORLD_HIGH[pair], sigma, z
    if condition == "world_extreme_low":
        return v14.PAIR_WORLD_LOW[pair], sigma, z
    if condition == "target_extreme_high":
        return mu, sigma, 5.0
    if condition == "target_extreme_low":
        return mu, sigma, -5.0
    raise ValueError(condition)


def v14_1_row_from_context(pair: str, condition: str, x: float, mu: float, sigma: float, z: float, values: np.ndarray, **extra) -> dict:
    row = v14.row_from_context(pair, condition, x, mu, sigma, z, values, **extra)
    row["population_z"] = float(z)
    row["z_full"] = local_z(pair, x, values)
    row["z_empirical_from_rendered_context"] = row["z_full"]
    row["valid_row_fraction"] = 1.0
    return row


def add_order_local_z(row: dict, values: np.ndarray) -> None:
    pair, x = row["pair"], float(row["x"])
    row["z_full"] = local_z(pair, x, values)
    row["z_first5"] = local_z(pair, x, values[:5])
    for k in [3, 5, 10, 15]:
        row[f"z_last{k}"] = local_z(pair, x, values[-k:])


def build_affine_rows(args) -> list[dict]:
    rows: list[dict] = []
    skipped: list[dict] = []
    for pair in args.pairs:
        for cell in v14.choose_cells(pair, args.ood_cells_per_pair):
            base_x, base_mu, base_sigma, base_z = (
                float(cell["x"]),
                float(cell["mu"]),
                float(cell["sigma"]),
                float(cell["z"]),
            )
            for n_context in args.ood_context_ns:
                for seed_idx in range(args.ood_seeds):
                    for condition in OOD_CONDITIONS:
                        mu2, sigma_requested, z2 = affine_mu_sigma_z(
                            pair, condition, base_x, base_mu, base_sigma, base_z
                        )
                        try:
                            seed = v14.stable_seed("v14.1-ood", pair, base_x, base_z, condition, n_context, seed_idx)
                            x2, sigma2, vals, note = positive_context_and_target(
                                pair, mu2, sigma_requested, z2, n_context, seed
                            )
                        except ValueError as exc:
                            skipped.append(
                                {
                                    "pair": pair,
                                    "condition": condition,
                                    "base_x": base_x,
                                    "base_z": base_z,
                                    "n_context": n_context,
                                    "seed_index": seed_idx,
                                    "reason": str(exc),
                                }
                            )
                            continue
                        cell_id = f"{pair}|{base_x:.6g}|{base_z:.3f}|{n_context}|{seed_idx}"
                        rows.append(
                            v14_1_row_from_context(
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
                                base_x=base_x,
                                base_mu=base_mu,
                                base_z=base_z,
                                requested_sigma=base_sigma,
                                sampling_note=note,
                            )
                        )
    if skipped:
        (RESULTS / "affine_ood" / "affine_ood_skipped_rows.json").write_text(json.dumps(skipped, indent=2))
    return rows


def build_order_rows(args) -> list[dict]:
    rows: list[dict] = []
    for pair in args.pairs:
        for cell in v14.choose_cells(pair, args.order_cells_per_pair):
            x, mu, sigma, z = float(cell["x"]), float(cell["mu"]), float(cell["sigma"]), float(cell["z"])
            for seed_idx in range(args.order_seeds):
                base_seed = v14.stable_seed("v14.1-order-base", pair, x, z, seed_idx)
                base_vals = v14.context_values(pair, mu, sigma, "normal", args.context_n, base_seed)
                if base_vals is None:
                    continue
                for kind in ORDER_KINDS:
                    vals = v14.order_values(kind, base_vals, x, v14.stable_seed("v14.1-order", pair, x, z, seed_idx, kind))
                    cell_id = f"{pair}|{x:.6g}|{z:.3f}|{seed_idx}"
                    row = v14_1_row_from_context(
                        pair,
                        kind,
                        x,
                        mu,
                        sigma,
                        z,
                        vals,
                        order_kind=kind,
                        seed_index=seed_idx,
                        cell_id=cell_id,
                    )
                    add_order_local_z(row, vals)
                    rows.append(row)
    return rows


def group_metrics(rows: list[dict], group_keys: list[str]) -> dict:
    out = v14.group_metrics(rows, group_keys)
    if group_keys == ["pair", "order_kind"]:
        for pair, by_cond in out.items():
            for cond, rec in by_cond.items():
                subset = [r for r in rows if r["pair"] == pair and r["order_kind"] == cond]
                ld = [r["ld"] for r in subset]
                for key in ["z_full", "z_first5", "z_last3", "z_last5", "z_last10", "z_last15"]:
                    rec[f"corr_ld_{key}"] = v14.corr(ld, [r.get(key) for r in subset])
    return out


def order_gap_metrics(rows: list[dict]) -> dict:
    by_cell: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        by_cell.setdefault((row["pair"], row["cell_id"]), {})[row["order_kind"]] = row
    out: dict[str, dict] = {}
    for (pair, _), by_order in by_cell.items():
        base = by_order.get("random")
        if not base:
            continue
        out.setdefault(pair, {})
        for cond, row in by_order.items():
            if cond == "random":
                continue
            out[pair].setdefault(cond, {f"gap_{k}": [] for k in [3, 5, 10, 15]})
            out[pair][cond].setdefault("delta_ld", []).append(float(row["ld"] - base["ld"]))
            for k in [3, 5, 10, 15]:
                out[pair][cond][f"gap_{k}"].append(float(row.get(f"z_last{k}", np.nan) - row.get("z_full", np.nan)))
    final: dict[str, dict] = {}
    for pair, by_cond in out.items():
        final[pair] = {}
        for cond, vals in by_cond.items():
            delta = vals.pop("delta_ld")
            final[pair][cond] = {
                "n_matched": len(delta),
                **{f"corr_delta_ld_gap_last{k}": v14.corr(delta, vals[f"gap_{k}"]) for k in [3, 5, 10, 15]},
            }
    return final


def experiment_affine_ood(model, tok, args) -> None:
    rows = build_affine_rows(args)
    print(f"[v14.1] affine/OOD prompts={len(rows)}", flush=True)
    ld, H, _ = v14.run_prompts(model, tok, rows, LAYERS, args.batch_size, args.max_seq, top_k=0)
    v14.add_outputs(rows, ld, H)
    result = {
        "model_id": MODEL_ID,
        "model_short": MODEL_SHORT,
        "pairs": args.pairs,
        "conditions": OOD_CONDITIONS,
        "context_ns": args.ood_context_ns,
        "metrics_by_pair_condition_n": group_metrics(rows, ["pair", "ood_condition", "n_context"]),
        "matched_delta_vs_base": v14.matched_delta_metrics(rows, "ood_condition", "base"),
        "direction_diagnostics": v14.direction_diagnostics(rows, H, "base", "ood_condition"),
        "support_fix": "bounded-normal contexts with per-row sigma shrink when needed to keep targets/context positive",
    }
    v14.write_json(RESULTS / "affine_ood" / "affine_ood_metrics.json", result)
    v14.write_rows_jsonl(rows, RESULTS / "affine_ood" / "affine_ood_rows.jsonl")


def experiment_order(model, tok, args) -> None:
    rows = build_order_rows(args)
    print(f"[v14.1] order prompts={len(rows)} context_n={args.context_n}", flush=True)
    ld, H, _ = v14.run_prompts(model, tok, rows, LAYERS, args.batch_size, args.max_seq, top_k=0)
    v14.add_outputs(rows, ld, H)
    result = {
        "model_id": MODEL_ID,
        "model_short": MODEL_SHORT,
        "pairs": args.pairs,
        "context_n": args.context_n,
        "order_kinds": ORDER_KINDS,
        "metrics_by_pair_order": group_metrics(rows, ["pair", "order_kind"]),
        "matched_delta_vs_random": v14.matched_delta_metrics(rows, "order_kind", "random"),
        "local_z_gap_metrics": order_gap_metrics(rows),
        "direction_diagnostics": v14.direction_diagnostics(rows, H, "random", "order_kind"),
    }
    v14.write_json(RESULTS / "order" / "order_metrics.json", result)
    v14.write_rows_jsonl(rows, RESULTS / "order" / "order_rows.jsonl")


def experiment_distribution(model, tok, args) -> None:
    rows = v14.build_distribution_rows(args)
    print(f"[v14.1] distribution prompts={len(rows)} context_n={args.context_n}", flush=True)
    ld, H, _ = v14.run_prompts(model, tok, rows, LAYERS, args.batch_size, args.max_seq, top_k=0)
    v14.add_outputs(rows, ld, H)
    result = {
        "model_id": MODEL_ID,
        "model_short": MODEL_SHORT,
        "pairs": args.pairs,
        "context_n": args.context_n,
        "distribution_kinds": DIST_KINDS,
        "metrics_by_pair_distribution": group_metrics(rows, ["pair", "dist_kind"]),
        "matched_delta_vs_normal": v14.matched_delta_metrics(rows, "dist_kind", "normal"),
        "direction_diagnostics": v14.direction_diagnostics(rows, H, "normal", "dist_kind"),
    }
    v14.write_json(RESULTS / "distribution" / "distribution_metrics.json", result)
    v14.write_rows_jsonl(rows, RESULTS / "distribution" / "distribution_rows.jsonl")


def experiment_fig5_primal_x(model, tok, args) -> None:
    old_results, old_figs = v14.RESULTS, v14.FIGS
    v14.RESULTS, v14.FIGS = RESULTS, FIGS
    try:
        v14.experiment_fig5_gpu(model, tok, args)
    finally:
        v14.RESULTS, v14.FIGS = old_results, old_figs
    src = RESULTS / "fig5" / "fig5_layer_x_z_metrics.json"
    if src.exists():
        data = json.loads(src.read_text())
        out = {
            "model_short": MODEL_SHORT,
            "layers": data["layers"],
            "pairs": data["pairs"],
            "alpha": args.alpha,
            "by_pair": {},
            "source": "generated by v14 fig5_gpu section, normalized to V12 layer_sweep_9b_steering shape",
        }
        for pair, by_layer in data["by_pair_layer"].items():
            out["by_pair"][pair] = {}
            for layer, rec in by_layer.items():
                out["by_pair"][pair][layer] = {
                    "n_prompts": None,
                    "primal_x": rec.get("primal_x_steering_slope"),
                }
        v14.write_json(RESULTS / "fig5" / "layer_sweep_9b_steering_primal_x.json", out)


def source_path(section: str, filename: str) -> Path:
    own = RESULTS / section / filename
    return own if own.exists() else V14_RESULTS / section / filename


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def sem(xs: list[float]) -> float:
    return float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")


def binned_line(rows: list[dict], condition_key: str, conditions: list[str], title: str, path: Path, z_key: str = "z") -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True)
    axes = axes.ravel()
    for ax, pair in zip(axes, PAIRS):
        pair_rows = [r for r in rows if r["pair"] == pair and r.get("n_context", 31) == 31]
        for cond in conditions:
            cond_rows = [
                r for r in pair_rows if r.get(condition_key) == cond and -5.1 <= float(r.get(z_key, r["z"])) <= 5.1
            ]
            zs = sorted({round(float(r.get(z_key, r["z"])), 2) for r in cond_rows})
            xs, ys = [], []
            for z in zs:
                vals = [float(r["ld"]) for r in cond_rows if round(float(r.get(z_key, r["z"])), 2) == z]
                if len(vals) >= 2:
                    xs.append(z)
                    ys.append(mean(vals))
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, lw=1.35, label=cond)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(pair)
        ax.set_xlabel("z")
        ax.set_ylabel("LD(high-low)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_affine_ood() -> None:
    path = source_path("affine_ood", "affine_ood_rows.jsonl")
    if not path.exists():
        return
    rows = load_jsonl(path)
    out_dir = FIGS / "affine_ood"
    conditions = ["base", "parallel_shift_high", "world_extreme_high", "world_extreme_low"]
    binned_line(rows, "ood_condition", conditions, "V14.1 affine/OOD: LD by relative standing", out_dir / "affine_ood_ld_by_z_lines.png")

    for suffix, pred, ylabel in [
        ("high", lambda r: float(r["z"]) > 1.0, "mean LD(high-low), z > 1"),
        ("low", lambda r: float(r["z"]) < -1.0, "mean LD(high-low), z < -1"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(PAIRS))
        width = 0.22
        for cond, offset in [("base", -width), ("world_extreme_low", 0.0), ("world_extreme_high", width)]:
            vals, errs = [], []
            for pair in PAIRS:
                subset = [
                    float(r["ld"])
                    for r in rows
                    if r["pair"] == pair and r.get("n_context", 31) == 31 and r.get("ood_condition") == cond and pred(r)
                ]
                vals.append(mean(subset))
                errs.append(sem(subset))
            ax.bar(x + offset, vals, width=width, yerr=errs, capsize=2, label=cond)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xticks(x, PAIRS, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Conflict test: relatively {suffix} targets in strange worlds")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"affine_ood_conflict_z_{suffix}.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(PAIRS))
    for cond, offset in [("base", -0.27), ("target_extreme_low", 0.0), ("target_extreme_high", 0.27)]:
        vals, errs = [], []
        for pair in PAIRS:
            if cond == "base":
                subset = [
                    float(r["ld"])
                    for r in rows
                    if r["pair"] == pair and r.get("n_context", 31) == 31 and r.get("ood_condition") == cond and abs(float(r["z"])) < 0.25
                ]
            else:
                subset = [
                    float(r["ld"])
                    for r in rows
                    if r["pair"] == pair and r.get("n_context", 31) == 31 and r.get("ood_condition") == cond
                ]
            vals.append(mean(subset))
            errs.append(sem(subset))
        ax.bar(x + offset, vals, width=0.27, yerr=errs, capsize=2, label=cond)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(x, PAIRS, rotation=30, ha="right")
    ax.set_ylabel("mean LD(high-low)")
    ax.set_title("Target-only extremes: mean LD, not corr(LD,z)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "affine_ood_target_extremes_mean_ld.png", dpi=170)
    plt.close(fig)

    counts = np.zeros((len(PAIRS), len(OOD_CONDITIONS)), dtype=float)
    for i, pair in enumerate(PAIRS):
        expected = max(
            len([r for r in rows if r["pair"] == pair and r.get("n_context", 31) == 31 and r.get("ood_condition") == cond])
            for cond in OOD_CONDITIONS
        )
        for j, cond in enumerate(OOD_CONDITIONS):
            n = len([r for r in rows if r["pair"] == pair and r.get("n_context", 31) == 31 and r.get("ood_condition") == cond])
            counts[i, j] = n / expected if expected else np.nan
    heatmap(counts, PAIRS, OOD_CONDITIONS, "V14.1 valid-row fraction at N=31", out_dir / "affine_ood_valid_row_fraction.png", vmin=0, vmax=1, cmap="viridis")


def plot_distribution() -> None:
    rows_path = source_path("distribution", "distribution_rows.jsonl")
    metrics_path = source_path("distribution", "distribution_metrics.json")
    if not rows_path.exists() or not metrics_path.exists():
        return
    rows = load_jsonl(rows_path)
    out_dir = FIGS / "distribution"
    binned_line(rows, "dist_kind", DIST_KINDS, "V14.1 distribution controls: LD-z by context shape", out_dir / "distribution_ld_by_z_lines.png")
    metrics = json.loads(metrics_path.read_text())
    mins, maxs = [], []
    for pair in PAIRS:
        vals = [
            rec["corr_ld_z"]
            for rec in metrics["metrics_by_pair_distribution"][pair].values()
            if isinstance(rec.get("corr_ld_z"), (int, float))
        ]
        mins.append(min(vals))
        maxs.append(max(vals))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(PAIRS))
    ax.vlines(x, mins, maxs, color="#4C78A8", lw=5)
    ax.scatter(x, mins, color="#1F4E79", s=28, label="min")
    ax.scatter(x, maxs, color="#F58518", s=28, label="max")
    ax.set_xticks(x, PAIRS, rotation=30, ha="right")
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("corr(LD,z)")
    ax.set_title("Distribution robustness: correlation range across shapes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "distribution_corr_range_by_pair.png", dpi=170)
    plt.close(fig)


def plot_order() -> None:
    rows_path = source_path("order", "order_rows.jsonl")
    metrics_path = source_path("order", "order_metrics.json")
    if not rows_path.exists() or not metrics_path.exists():
        return
    rows = load_jsonl(rows_path)
    out_dir = FIGS / "order"
    binned_line(rows, "order_kind", ORDER_KINDS, "V14.1 order controls: LD by full-context z", out_dir / "order_ld_by_z_lines.png")

    metrics = json.loads(metrics_path.read_text())
    has_local = any(
        f"corr_ld_{key}" in rec
        for by_order in metrics.get("metrics_by_pair_order", {}).values()
        for rec in by_order.values()
        for key in ["z_full", "z_first5", "z_last3", "z_last5", "z_last10", "z_last15"]
    )
    if not has_local:
        write_placeholder(
            out_dir / "order_full_vs_local_z_corr.png",
            "Order local-z diagnostics require V14.1 order rows",
            "Run: python scripts/run_v14_1_gpu.py --sections order,plot",
        )
        write_placeholder(
            out_dir / "order_delta_ld_vs_local_z_gap.png",
            "Order delta-vs-local-gap diagnostics require V14.1 order rows",
            "Existing V14 rows do not store rendered context values.",
        )
        return

    keys = ["z_full", "z_first5", "z_last3", "z_last5", "z_last10", "z_last15"]
    mat = np.full((len(PAIRS), len(keys)), np.nan)
    for i, pair in enumerate(PAIRS):
        by_order = metrics.get("metrics_by_pair_order", {}).get(pair, {})
        for j, key in enumerate(keys):
            vals = []
            for rec in by_order.values():
                val = rec.get(f"corr_ld_{key}")
                if isinstance(val, (int, float)):
                    vals.append(float(val))
            mat[i, j] = float(np.nanmean(vals)) if vals else np.nan
    heatmap(mat, PAIRS, keys, "Order diagnostics: corr(LD, local z)", out_dir / "order_full_vs_local_z_corr.png", vmin=0, vmax=1)

    gap = metrics.get("local_z_gap_metrics", {})
    conds = [c for c in ORDER_KINDS if c != "random"]
    mat = np.full((len(PAIRS), len(conds)), np.nan)
    for i, pair in enumerate(PAIRS):
        for j, cond in enumerate(conds):
            vals = [
                gap.get(pair, {}).get(cond, {}).get(f"corr_delta_ld_gap_last{k}")
                for k in [3, 5, 10, 15]
            ]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            mat[i, j] = float(np.nanmean(vals)) if vals else np.nan
    heatmap(mat, PAIRS, conds, "Order effect vs local-z gap", out_dir / "order_delta_ld_vs_local_z_gap.png", vmin=-1, vmax=1)


def plot_fig5_original_format() -> None:
    layer_path = REPO / "results" / "v12" / "layer_sweep_9b.json"
    steer_path = REPO / "results" / "v12" / "layer_sweep_9b_steering.json"
    if not layer_path.exists() or not steer_path.exists():
        return
    layer = json.loads(layer_path.read_text())
    steer = json.loads(steer_path.read_text())
    fig5_path = RESULTS / "fig5" / "layer_sweep_9b_steering_primal_x.json"
    if not fig5_path.exists() and (V14_RESULTS / "fig5" / "fig5_layer_x_z_metrics.json").exists():
        v14_fig5 = json.loads((V14_RESULTS / "fig5" / "fig5_layer_x_z_metrics.json").read_text())
        primal_x = {"layers": v14_fig5["layers"], "pairs": v14_fig5["pairs"], "by_pair": {}}
        for pair, by_layer in v14_fig5["by_pair_layer"].items():
            primal_x["by_pair"][pair] = {
                str(layer_id): {"primal_x": rec.get("primal_x_steering_slope")}
                for layer_id, rec in by_layer.items()
            }
    elif fig5_path.exists():
        primal_x = json.loads(fig5_path.read_text())
    else:
        primal_x = None

    pairs = list(layer["pairs"])
    n_layers = next(iter(layer["pairs"].values()))["n_layers"]
    xs = np.arange(n_layers)

    def layer_mat(key: str) -> np.ndarray:
        return np.array([[r[key] for r in layer["pairs"][p]["layer_records"]] for p in pairs], dtype=float)

    r2z = layer_mat("r2_cv_z")
    incr = layer_mat("increment_r2_fold_aware")
    pnorm = layer_mat("primal_norm")
    cprev = layer_mat("primal_cos_prev_layer")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    panels = [
        (axes[0, 0], r2z, "CV R2(z): availability", "R2", "C0", False),
        (axes[0, 1], incr, "Fold-aware incremental R2(z)", "incremental R2", "C1", False),
        (axes[0, 2], pnorm, "primal_z norm", "norm", "C2", True),
        (axes[1, 1], cprev, "cos(primal_z[L], primal_z[L-1])", "cosine", "C4", False),
    ]
    for ax, mat, title, ylabel, color, logy in panels:
        center, lo, hi = nan_summary(mat, axis=0)
        ax.plot(xs, center, "-o", ms=3, lw=2, color=color)
        ax.fill_between(xs, lo, hi, alpha=0.18, color=color)
        ax.axhline(0, color="k", lw=0.5)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    ax = axes[1, 0]
    layers = steer["layers"]
    for direction, color in [("primal_z", "C0"), ("primal_x", "C1"), ("random_null", "0.4")]:
        vals = []
        for pair in steer["pairs"]:
            row = []
            for L in layers:
                if direction == "primal_x" and primal_x is not None:
                    row.append(primal_x.get("by_pair", {}).get(pair, {}).get(str(L), {}).get("primal_x", np.nan))
                else:
                    row.append(steer["by_pair"][pair][str(L)].get(direction, np.nan))
            vals.append(row)
        vals = np.asarray(vals, dtype=float)
        center, lo, hi = nan_summary(vals, axis=0)
        ax.plot(layers, center, "-o", ms=4, lw=2, label=direction, color=color)
        ax.fill_between(layers, lo, hi, alpha=0.15, color=color)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Causal steering slope")
    ax.set_xlabel("layer")
    ax.set_ylabel("Delta logit-diff per alpha")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    vals = np.array([[steer["by_pair"][p][str(L)].get("primal_z", np.nan) for L in layers] for p in steer["pairs"]])
    r2z_center, _, _ = nan_summary(r2z, axis=0)
    vals_center, _, _ = nan_summary(vals, axis=0)
    ax.scatter(r2z_center[layers], vals_center, s=50)
    for L, a, b in zip(layers, r2z_center[layers], vals_center):
        ax.text(a, b, str(L), fontsize=8)
    ax.set_xlabel("mean CV R2(z)")
    ax.set_ylabel("mean primal_z steering slope")
    ax.set_title("Decodability vs use")
    ax.grid(alpha=0.25)

    fig.suptitle("v14.1 Gemma 2 9B layer sweep: original layout with primal_x control", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5" / "layer_sweep_9b_combined_primal_x.png", dpi=150)
    plt.close(fig)


def nan_summary(M: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(M, dtype=float)
    moved = np.moveaxis(arr, axis, 0)
    rest_shape = moved.shape[1:]
    center = np.full(rest_shape, np.nan, dtype=float)
    lo = np.full(rest_shape, np.nan, dtype=float)
    hi = np.full(rest_shape, np.nan, dtype=float)
    for idx in np.ndindex(rest_shape):
        vals = moved[(slice(None),) + idx]
        valid = vals[np.isfinite(vals)]
        if valid.size:
            center[idx] = float(np.mean(valid))
            lo[idx] = float(np.min(valid))
            hi[idx] = float(np.max(valid))
    return center, lo, hi


def write_placeholder(path: Path, title: str, detail: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=13, weight="bold", wrap=True)
    ax.text(0.5, 0.42, detail, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(M: np.ndarray, rows: list[str], cols: list[str], title: str, path: Path, vmin=None, vmax=None, cmap="RdBu_r") -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(cols) * 1.05), max(4, len(rows) * 0.55)))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
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


def plot_all() -> None:
    plot_affine_ood()
    plot_distribution()
    plot_order()
    plot_fig5_original_format()
    print("Wrote V14.1 plots under figures/v14_1/")


def load_model():
    return v14.load_model()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="plot", help="comma list: distribution,order,affine_ood,fig5_primal_x,plot")
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
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
    ap.add_argument("--fig5-cells-per-pair", type=int, default=72)
    ap.add_argument("--fig5-seeds", type=int, default=1)
    ap.add_argument("--fig5-layers", nargs="+", type=int, default=DEFAULT_FIG5_LAYERS)
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args()

    sections = {"distribution", "order", "affine_ood", "fig5_primal_x", "plot"} if args.sections == "all" else set(args.sections.split(","))
    model = tok = None
    if sections - {"plot"}:
        model, tok = load_model()
    if "distribution" in sections:
        experiment_distribution(model, tok, args)
    if "order" in sections:
        experiment_order(model, tok, args)
    if "affine_ood" in sections:
        experiment_affine_ood(model, tok, args)
    if "fig5_primal_x" in sections:
        experiment_fig5_primal_x(model, tok, args)
    if "plot" in sections:
        plot_all()
    print("[v14.1] complete", flush=True)


if __name__ == "__main__":
    main()
