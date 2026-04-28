"""Plot primal_z: W_U-orthogonal but decision-aligned (height) — v11 + v11.5."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# We use the widened file as the primary source: it has both cos_primal_unembed
# and cos_primal_amb_widened per layer.  The v11 per_layer file has
# cos_primal_lexical_unembed (same values) but the "decision" direction there
# had too few samples — widened fixes that.

MODELS = {
    "gemma2-2b": ROOT / "results/v11_5/gemma2-2b/z_vs_lexical_widened.json",
    "gemma2-9b": ROOT / "results/v11_5/gemma2-9b/z_vs_lexical_widened.json",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (model_name, path) in zip(axes, MODELS.items()):
    with open(path) as f:
        data = json.load(f)

    height = data["height"]
    per_layer = height["per_layer"]

    layers = [d["layer"] for d in per_layer]
    cos_unembed = [abs(d["cos_primal_unembed"]) for d in per_layer]
    cos_decision = [abs(d["cos_primal_amb_widened"]) for d in per_layer]

    ax.plot(layers, cos_unembed, "o-", markersize=3, linewidth=1.5,
            color="#4C72B0", label=r"$W_U$ lexical direction")
    ax.plot(layers, cos_decision, "s-", markersize=3, linewidth=1.5,
            color="#DD8452", label="data-derived decision direction")

    ax.set_xlabel("Layer")
    ax.set_ylabel(r"|cos(primal$_z$, direction)|" if ax is axes[0] else "")
    ax.set_title(f"{model_name}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(y=0.7, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    r"primal$_z$: $W_U$-orthogonal but decision-aligned (height)",
    fontsize=13, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out = ROOT / "figures/v11_5/z_vs_lexical_height.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved -> {out}")
