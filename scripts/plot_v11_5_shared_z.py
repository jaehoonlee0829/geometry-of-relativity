"""Plot shared z-direction steering efficiency (v11.5) — grouped bar chart."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
MODELS = {
    "gemma2-2b": ROOT / "results/v11_5/gemma2-2b/shared_z_analysis.json",
    "gemma2-9b": ROOT / "results/v11_5/gemma2-9b/shared_z_analysis.json",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, (model_name, path) in zip(axes, MODELS.items()):
    with open(path) as f:
        data = json.load(f)

    steering = data["by_pair_steering"]
    within_slopes = [steering[p]["within_slope"] for p in PAIRS]
    shared_slopes = [steering[p]["shared_slope_proc"] for p in PAIRS]
    ratios = [steering[p]["ratio_shared_to_within"] for p in PAIRS]

    x = np.arange(len(PAIRS))
    width = 0.35

    bars1 = ax.bar(x - width / 2, within_slopes, width, label="within-pair slope", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, shared_slopes, width, label="shared-direction slope", color="#DD8452")

    # Horizontal dashed line at 50% of each within-pair slope (use mean as reference)
    threshold_vals = [w * 0.5 for w in within_slopes]
    # Draw a single reference line at 50% of the mean within-pair slope
    mean_within = np.mean(within_slopes)
    ax.axhline(y=mean_within * 0.5, color="gray", linestyle="--", linewidth=1,
               label="50% of mean within-pair")

    # Annotate each pair with its ratio percentage
    for i, (ratio, ws, ss) in enumerate(zip(ratios, within_slopes, shared_slopes)):
        top = max(ws, ss)
        ax.annotate(
            f"{ratio * 100:.0f}%",
            xy=(x[i], top),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#2ca02c" if ratio >= 0.5 else "#d62728",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(PAIRS, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Slope (prob shift per alpha)" if ax is axes[0] else "")
    ax.set_title(f"{model_name} (layer {data['layer']})", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Shared z-direction steering efficiency", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])

out = ROOT / "figures/v11_5/shared_z_steering_ratios.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved -> {out}")
