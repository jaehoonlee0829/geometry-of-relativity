"""README story figure for v11.5.

This plot is intentionally 9B-first and CPU-only. It uses committed JSON
summaries to show the experiment design and the three most readable results:
early z encoding, shared/cross-pair steering, and SAE controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemma2-9b"
PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
PAIR_LABELS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "BMI"]


def load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text())


def mean_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    naive, orth = [], []
    for pair in PAIRS:
        data = load_json(f"results/v11_5/{MODEL}/{pair}/increment_r2_fold_aware.json")
        naive.append(np.array(data["naive_r2_per_layer"], dtype=float))
        orth.append(np.array(data["orth_r2_per_layer_FOLD_AWARE"], dtype=float))
    naive_arr = np.vstack(naive)
    orth_arr = np.vstack(orth)
    return np.arange(naive_arr.shape[1]), naive_arr.mean(axis=0), orth_arr.mean(axis=0)


def steering_ratios() -> tuple[list[float], list[float]]:
    shared = load_json(f"results/v11_5/{MODEL}/shared_z_analysis.json")
    transfer = load_json(f"results/v11_5/{MODEL}/multiseed_transfer.json")

    shared_ratios = [
        shared["by_pair_steering"][pair]["ratio_shared_to_within"] for pair in PAIRS
    ]
    cross_ratios = []
    for target in PAIRS:
        summary = transfer["transfer_summary"][target]
        within = summary[target]["mean_slope"]
        off = [summary[source]["mean_slope"] for source in PAIRS if source != target]
        cross_ratios.append(float(np.mean(off) / within))
    return shared_ratios, cross_ratios


def sae_controls() -> tuple[list[float], list[float], list[float], float]:
    data = load_json(f"results/v11_5/{MODEL}/sae_features_with_token_freq_control.json")
    r2_z, r2_x, r2_tok = [], [], []
    for pair in PAIRS:
        rec = data["by_pair"][pair]
        r2_z.append(rec["top_50_r2_z"][0])
        r2_x.append(rec["top_50_r2_x"][0])
        r2_tok.append(rec["top_50_r2_tok"][0])
    return r2_z, r2_x, r2_tok, data["cross_pair_jaccard_mean_off_diag"]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # A. Experiment design / variables.
    ax = axes[0, 0]
    x_vals = np.linspace(-2.5, 2.5, 20)
    z_vals = np.linspace(-3.0, 3.0, 20)
    xx, zz = np.meshgrid(x_vals, z_vals)
    ax.scatter(xx.flatten(), zz.flatten(), s=8, alpha=0.45, color="#4C72B0")
    ax.set_title("(a) Eval design: disentangle raw value from context")
    ax.set_xlabel("raw value axis x (schematic)")
    ax.set_ylabel("relative standing z")
    ax.text(
        0.02,
        0.98,
        r"$z=(x-\mu)/\sigma$" "\n"
        r"$x$: target value" "\n"
        r"$\mu$: context mean" "\n"
        r"$\sigma$: context spread" "\n\n"
        "v11: 20 x-values x 20 z-values x 10 seeds\n"
        "per adjective pair and model",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
    )
    ax.grid(alpha=0.25)

    # B. Early encoding.
    ax = axes[0, 1]
    layers, naive, orth = mean_curves()
    ax.plot(layers, naive, color="#4C72B0", lw=2.2, label="cumulative decodability")
    ax.plot(layers, orth, color="#DD8452", lw=2.2, label="new information per layer")
    ax.set_title("(b) z is available early, then carried forward")
    ax.set_xlabel("Gemma 2 9B layer")
    ax.set_ylabel("mean R²(z) across pairs")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # C. Steering.
    ax = axes[1, 0]
    shared_ratios, cross_ratios = steering_ratios()
    pos = np.arange(len(PAIRS))
    width = 0.36
    ax.bar(pos - width / 2, shared_ratios, width, color="#55A868", label="single shared direction / within")
    ax.bar(pos + width / 2, cross_ratios, width, color="#C44E52", label="mean cross-pair transfer / within")
    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.set_title("(c) Layer-33 steering transfers across adjective pairs")
    ax.set_xticks(pos)
    ax.set_xticklabels(PAIR_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("relative steering efficiency")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # D. SAE controls.
    ax = axes[1, 1]
    r2_z, r2_x, r2_tok, jaccard = sae_controls()
    means = [np.mean(r2_z), np.mean(r2_x), np.mean(r2_tok)]
    errs = [np.std(r2_z), np.std(r2_x), np.std(r2_tok)]
    ax.bar(["z", "raw x", "token magnitude"], means, yerr=errs, capsize=3, color=["#4C72B0", "#999999", "#999999"])
    ax.set_title("(d) SAE features track z, not numeral magnitude")
    ax.set_ylabel("top feature R², mean ± sd across pairs")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.5,
        0.78,
        f"9B top-50 feature overlap\nmean Jaccard = {jaccard:.3f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
    )

    fig.suptitle("v11.5: context-normalized z in Gemma 2 9B", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = ROOT / "figures" / "v11_5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "readme_9b_story.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
