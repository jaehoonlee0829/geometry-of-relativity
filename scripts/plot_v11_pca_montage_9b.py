"""Create a README montage of v11 Gemma 2 9B PCA plots.

The source plots already exist under figures/v11/pca. This script composes the
eight 2D PCA panels into one 2x4 image so README can show the cross-pair
geometry without embedding eight separate files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
TITLES = ["height", "age", "weight", "size", "speed", "wealth", "experience", "BMI"]


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, pair, title in zip(axes.flat, PAIRS, TITLES):
        path = ROOT / "figures" / "v11" / "pca" / f"{pair}_gemma2-9b_2d_L33.png"
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("v11 Gemma 2 9B: PCA of dense cell-mean activations", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / "figures" / "v11" / "pca" / "montage_gemma2-9b_2d_L33.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
