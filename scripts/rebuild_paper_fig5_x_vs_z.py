#!/usr/bin/env python3
"""Rebuild a paper-facing layer figure using existing V12 artifacts.

This intentionally avoids the underdefined `probe_z` steering comparison in the
paper draft. The available artifacts contain R^2(z), R^2(x), and primal_z
steering by layer. They do not currently contain primal_x steering by layer, so
the figure marks that panel as missing instead of inventing data.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures" / "v14"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_ORDER = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def mean_by_layer(layer_records: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    by_layer: dict[int, list[float]] = {}
    for pair in PAIR_ORDER:
        for rec in layer_records["pairs"][pair]["layer_records"]:
            val = rec.get(key)
            if val is None or not np.isfinite(val):
                continue
            by_layer.setdefault(int(rec["layer"]), []).append(float(val))
    layers = np.array(sorted(by_layer), dtype=int)
    vals = np.array([np.mean(by_layer[int(layer)]) for layer in layers], dtype=float)
    return layers, vals


def mean_steering_by_layer(steering: dict, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    by_layer: dict[int, list[float]] = {}
    for pair in PAIR_ORDER:
        pair_rows = steering["by_pair"].get(pair, {})
        for layer_str, rec in pair_rows.items():
            if key not in rec:
                continue
            val = rec[key]
            if val is None or not np.isfinite(val):
                continue
            by_layer.setdefault(int(layer_str), []).append(float(val))
    if not by_layer:
        return None
    layers = np.array(sorted(by_layer), dtype=int)
    vals = np.array([np.mean(by_layer[int(layer)]) for layer in layers], dtype=float)
    return layers, vals


def main() -> None:
    layer = load_json(REPO / "results" / "v12" / "layer_sweep_9b.json")
    steering = load_json(REPO / "results" / "v12" / "layer_sweep_9b_steering.json")

    lz, r2z = mean_by_layer(layer, "r2_cv_z")
    lx, r2x = mean_by_layer(layer, "r2_cv_x")
    primal_z = mean_steering_by_layer(steering, "primal_z")
    primal_x = mean_steering_by_layer(steering, "primal_x")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=180)

    ax = axes[0]
    ax.plot(lz, r2z, marker="o", label=r"$R^2(z)$", color="#1f77b4")
    ax.plot(lx, r2x, marker="o", label=r"$R^2(x)$", color="#d62728")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-validated $R^2$")
    ax.set_title("Linear availability")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    if primal_z is not None:
        layers, vals = primal_z
        ax.plot(layers, vals, marker="o", label=r"$d^{primal}_z$", color="#1f77b4")
    if primal_x is not None:
        layers, vals = primal_x
        ax.plot(layers, vals, marker="o", label=r"$d^{primal}_x$", color="#d62728")
    else:
        ax.text(0.03, 0.94, r"$d^{primal}_x$ not in existing artifacts", ha="left", va="top", transform=ax.transAxes, fontsize=8)
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Steering slope")
    ax.set_title("Causal steering")
    ax.grid(alpha=0.25)
    if primal_z is not None or primal_x is not None:
        ax.legend(frameon=False)

    fig.suptitle("Layer evidence: compare relative standing against raw magnitude", y=1.03)
    fig.tight_layout()
    out = OUT_DIR / "paper_fig5_layer_x_z_cleanup.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    if primal_x is None:
        print("NOTE: primal_x steering by layer is missing from existing artifacts.")


if __name__ == "__main__":
    main()
