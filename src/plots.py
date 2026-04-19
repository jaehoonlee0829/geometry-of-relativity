"""Plotting utilities for mech-interp relativity study.

Hero figure: Fisher-normalized cosine alignment between probe covectors and gradients.
Faceted by layer and model.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def plot_hero(
    df: pd.DataFrame,
    out_path: str,
) -> None:
    """Plot Fisher-normalized cosine ρ between probe covectors and gradients.

    Args:
        df: DataFrame with columns:
            - model (str): "gemma" or "llama"
            - layer (str): "early", "mid", "late", "final"
            - adjective_class (str): "relative" or "absolute"
            - context_mu (float): context mean (150, 165, 180)
            - rho_rel (float): cosine alignment for relative (tall/short)
            - rho_abs (float): cosine alignment for absolute (obese)
        out_path (str): output PDF path
    """

    # Try to use seaborn-v0_8 style with fallback
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    # Create 2x2 facet grid for layers
    layer_order = ["early", "mid", "late", "final"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Line style map
    line_styles = {"gemma": "-", "llama": "--"}
    colors = {"relative": "C0", "absolute": "C3"}  # blue and red

    for idx, layer in enumerate(layer_order):
        ax = axes[idx]
        layer_data = df[df["layer"] == layer]

        # Plot curves for each adjective class
        for adj_class in ["relative", "absolute"]:
            class_data = layer_data[layer_data["adjective_class"] == adj_class].copy()

            # For each model, plot separately to get proper line styles
            for model in ["gemma", "llama"]:
                model_data = class_data[class_data["model"] == model].sort_values("context_mu")

                if len(model_data) > 0:
                    col = "rho_rel" if adj_class == "relative" else "rho_abs"
                    ax.plot(
                        model_data["context_mu"],
                        model_data[col],
                        linestyle=line_styles[model],
                        color=colors[adj_class],
                        marker="o",
                        markersize=5,
                        linewidth=1.5,
                        label=f"{adj_class} ({model})" if idx == 0 else "",
                    )

        ax.set_xlabel("Context mean μ (cm)", fontsize=9)
        ax.set_ylabel("Fisher-normalized ρ", fontsize=9)
        ax.set_title(f"{layer.capitalize()} layer", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([150, 165, 180])

    # Create custom legend handles for the entire figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="C0", linewidth=1.5, label="relative (tall/short)"),
        Line2D([0], [0], color="C3", linewidth=1.5, label="absolute (obese)"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-", label="gemma (solid)"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="llama (dashed)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=9,
    )

    plt.suptitle("Fisher-normalized alignment: probe covector vs. gradient", fontsize=10, y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()


def plot_probe_shift(
    df: pd.DataFrame,
    out_path: str,
) -> None:
    """Visualize H1 kill-test: probe logit as function of context_mu, by adjective.

    Args:
        df: DataFrame with columns:
            - model (str): "gemma" or "llama"
            - layer (str): layer identifier
            - adjective (str): adjective name
            - context_mu (float): context mean
            - probe_logit (float): probe logit score
        out_path (str): output PDF path
    """

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(8, 5))

    line_styles = {"gemma": "-", "llama": "--"}

    # Group by adjective and plot
    for adjective in df["adjective"].unique():
        adj_data = df[df["adjective"] == adjective].sort_values("context_mu")

        for model in ["gemma", "llama"]:
            model_data = adj_data[adj_data["model"] == model]
            if len(model_data) > 0:
                ax.plot(
                    model_data["context_mu"],
                    model_data["probe_logit"],
                    linestyle=line_styles[model],
                    marker="o",
                    markersize=6,
                    linewidth=1.5,
                    label=f"{adjective} ({model})",
                )

    ax.set_xlabel("Context mean μ (cm)", fontsize=9)
    ax.set_ylabel("Probe logit", fontsize=9)
    ax.set_title("Probe shift across contexts (H1 kill-test)", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()


if __name__ == "__main__":
    # Synthetic data for smoke test
    import numpy as np

    np.random.seed(42)

    layers = ["early", "mid", "late", "final"]
    models = ["gemma", "llama"]
    contexts = [150, 165, 180]
    adj_classes = ["relative", "absolute"]

    rows = []
    for layer in layers:
        for model in models:
            for context_mu in contexts:
                for adj_class in adj_classes:
                    # Synthetic: relative should track ∇z_C (context-dependent)
                    # absolute should track ∇x (stable)
                    if adj_class == "relative":
                        # Tall at 180, short at 150, intermediate at 165
                        base = (context_mu - 165) / 15
                        noise = np.random.normal(0, 0.1)
                        rho_rel = 0.8 + base * 0.3 + noise
                        rho_abs = 0.2 + np.random.normal(0, 0.1)
                    else:
                        rho_rel = 0.15 + np.random.normal(0, 0.1)
                        rho_abs = 0.7 + np.random.normal(0, 0.08)

                    rows.append({
                        "model": model,
                        "layer": layer,
                        "adjective_class": adj_class,
                        "context_mu": context_mu,
                        "rho_rel": max(-1, min(1, rho_rel)),
                        "rho_abs": max(-1, min(1, rho_abs)),
                    })

    df = pd.DataFrame(rows)

    # Create figures directory if it doesn't exist
    import os
    os.makedirs("figures", exist_ok=True)

    plot_hero(df, "figures/hero_smoke.pdf")
    print("Smoke test: hero_smoke.pdf created successfully")
