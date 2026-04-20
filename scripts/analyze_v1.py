"""Analyze v1 spectrum scan: compute per-(family, mu) completion class fractions,
fit sigmoids to the obese response curves, generate plots + markdown summary.
"""
from __future__ import annotations
import json
import pathlib
import re
from collections import Counter, defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "behavioral_v1"
OUT_SUMMARY = REPO / "results" / "behavioral_v1_summary.md"
OUT_CSV = REPO / "results" / "behavioral_v1_per_mu.csv"
OUT_PLOT = REPO / "figures" / "spectrum_v1.pdf"

# Completion categories (case-insensitive substring match, checked in order).
CATEGORIES = [
    ("extremely_tall", [r"extremely tall", r"very tall", r"exceptionally tall", r"unusually tall"]),
    ("tall", [r"\btall\b", r"\btaller\b"]),
    ("extremely_short", [r"extremely short", r"very short"]),
    ("short", [r"\bshort\b", r"\bshorter\b"]),
    ("average_normal", [r"\baverage\b", r"\bnormal\b", r"\btypical\b", r"\bordinary\b", r"\baverage height\b", r"\baverage weight\b", r"\bhealthy\b", r"\bmedium\b"]),
    ("obese", [r"\bobese\b", r"\bobesity\b", r"\boverweight\b"]),
    ("underweight", [r"\bunderweight\b", r"\bskinny\b", r"\bemaciated\b", r"\bthin\b"]),
]


def classify(completion: str) -> str:
    s = completion.lower()
    for label, patterns in CATEGORIES:
        for p in patterns:
            if re.search(p, s):
                return label
    return "other"


def load_all():
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_sigmoid(xs, ys):
    """Fit 4-param logistic: y = L / (1 + exp(-k (x - x0))) + b.
    Uses scipy if available, else returns None."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    # Initial guesses
    L0 = max(ys) - min(ys) if max(ys) > min(ys) else 1.0
    b0 = min(ys)
    x00 = xs[np.argmin(np.abs(ys - (max(ys) + min(ys)) / 2))]
    k0 = -1.0  # falling sigmoid (fraction "obese" decreases with mu)
    try:
        popt, _ = curve_fit(sigmoid, xs, ys, p0=[L0, x00, k0, b0], maxfev=10000)
        return popt
    except Exception:
        return None


def main():
    rows = load_all()
    if not rows:
        print(f"No JSON in {RESULTS_DIR}")
        return

    # Group by family (tall_spectrum, obese_spectrum_bmi, obese_spectrum_hw), mu, template
    groups = defaultdict(list)  # (family, mu) -> list of completions (all templates pooled)
    for r in rows:
        key = (r["family"], r["context_mu"])
        groups[key].extend(r["completions"])

    # Per-mu aggregates
    per_mu = []
    for (family, mu), comps in sorted(groups.items()):
        counts = Counter(classify(c) for c in comps)
        n = len(comps)
        rec = {"family": family, "mu": mu, "n": n}
        for label, _ in CATEGORIES:
            rec[label] = counts.get(label, 0) / n
        rec["other"] = counts.get("other", 0) / n
        per_mu.append(rec)

    # Write CSV
    import csv
    with OUT_CSV.open("w", newline="") as f:
        cat_labels = [label for label, _ in CATEGORIES] + ["other"]
        writer = csv.DictWriter(f, fieldnames=["family", "mu", "n"] + cat_labels)
        writer.writeheader()
        for rec in per_mu:
            writer.writerow(rec)

    # Build sigmoid fits
    fits = {}
    for family in ("tall_spectrum", "obese_spectrum_bmi", "obese_spectrum_hw"):
        rows_f = [r for r in per_mu if r["family"] == family]
        if not rows_f:
            continue
        xs = [r["mu"] for r in rows_f]
        target_key = "tall" if family == "tall_spectrum" else "obese"
        ys = [r[target_key] for r in rows_f]
        popt = fit_sigmoid(xs, ys)
        fits[family] = (xs, ys, popt)

    # Write summary
    lines = []
    lines.append("# v1 Spectrum Analysis — Sonnet 4.6, 81 prompts × 10 samples\n")
    lines.append("## Key question\n")
    lines.append("Do relative ('tall') and absolute ('obese') gradable adjectives respond differently")
    lines.append("to a spectrum of context means μ? Fair symmetric design: both adjectives tested across")
    lines.append("9 μ values symmetric around the target, at narrow σ.\n")

    for family, key, desc in [
        ("tall_spectrum", "tall", "TALL / target = 165 cm"),
        ("obese_spectrum_bmi", "obese", "OBESE (BMI-direct) / target = BMI 32"),
        ("obese_spectrum_hw", "obese", "OBESE (height+weight, BMI derivation required) / target = 170cm/92kg"),
    ]:
        lines.append(f"\n## {desc}\n")
        rows_f = sorted([r for r in per_mu if r["family"] == family], key=lambda r: r["mu"])
        if not rows_f:
            lines.append("(no data)\n")
            continue
        lines.append(f"| μ | fraction `{key}` | fraction average/normal | fraction other | n |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in rows_f:
            lines.append(f"| {r['mu']} | {r[key]:.2f} | {r['average_normal']:.2f} | {r['other']:.2f} | {r['n']} |")
        if family in fits and fits[family][2] is not None:
            L, x0, k, b = fits[family][2]
            lines.append(f"\nSigmoid fit: L={L:.3f}, x0={x0:.2f} (transition center), k={k:.3f} (slope), b={b:.3f}")
            lines.append(f"Interpretation: 50% transition at μ≈{x0:.1f}. Slope magnitude |k|={abs(k):.2f}.")

    # Compare obese_bmi vs obese_hw
    lines.append("\n## Paired comparison: BMI-direct vs height+weight (obese)\n")
    lines.append("If 'obese' is anchored to the literal number 32 (pattern-match), the BMI-direct curve stays high.")
    lines.append("If 'obese' is anchored to body habitus relative to context, both curves track each other.")
    lines.append("The difference between the two curves at each μ measures the contribution of literal-numeric-anchoring.\n")
    bmi_by_mu = {r["mu"]: r["obese"] for r in per_mu if r["family"] == "obese_spectrum_bmi"}
    hw_by_mu = {r["mu"]: r["obese"] for r in per_mu if r["family"] == "obese_spectrum_hw"}
    lines.append("| μ | obese% BMI-direct | obese% height+weight | Δ (bmi − hw) |")
    lines.append("|---|---:|---:|---:|")
    for mu in sorted(bmi_by_mu.keys()):
        b = bmi_by_mu.get(mu, float("nan"))
        h = hw_by_mu.get(mu, float("nan"))
        d = b - h if not (np.isnan(b) or np.isnan(h)) else float("nan")
        lines.append(f"| {mu} | {b:.2f} | {h:.2f} | {d:+.2f} |")

    # Verdict for H2 (fair design)
    lines.append("\n## Verdict on H2 (revised, fair-design test)\n")
    obese_at_mu40 = next((r["obese"] for r in per_mu if r["family"] == "obese_spectrum_bmi" and r["mu"] == 40), None)
    obese_at_mu43 = next((r["obese"] for r in per_mu if r["family"] == "obese_spectrum_bmi" and r["mu"] == 43), None)
    lines.append(f"- fraction 'obese' at μ=40 (BMI-direct): {obese_at_mu40:.2f}" if obese_at_mu40 is not None else "- μ=40 BMI-direct missing")
    lines.append(f"- fraction 'obese' at μ=43 (BMI-direct): {obese_at_mu43:.2f}" if obese_at_mu43 is not None else "- μ=43 BMI-direct missing")
    lines.append("If obese were fully absolute, both should be ≈1.00 (BMI=32 is always medically obese).")
    lines.append("If obese is fully relative, both should be ≈0.00 (BMI=32 is slim relative to μ=43).")

    OUT_SUMMARY.write_text("\n".join(lines))
    print(f"summary → {OUT_SUMMARY}")
    print(f"csv → {OUT_CSV}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        # Left: tall spectrum
        rows_t = sorted([r for r in per_mu if r["family"] == "tall_spectrum"], key=lambda r: r["mu"])
        xs = [r["mu"] for r in rows_t]
        axes[0].plot(xs, [r["tall"] for r in rows_t], "o-", label="tall", color="C0")
        axes[0].plot(xs, [r["short"] for r in rows_t], "o-", label="short", color="C3")
        axes[0].plot(xs, [r["average_normal"] for r in rows_t], "o-", label="average/normal", color="C7")
        axes[0].axvline(165, color="k", ls=":", alpha=0.5, label="target=165 cm")
        axes[0].set_xlabel("context μ (cm)")
        axes[0].set_ylabel("fraction of completions")
        axes[0].set_title("TALL spectrum (target = 165 cm)")
        axes[0].legend(fontsize=9)
        axes[0].set_ylim(-0.05, 1.05)
        # Right: obese spectrum, both styles overlaid
        rows_b = sorted([r for r in per_mu if r["family"] == "obese_spectrum_bmi"], key=lambda r: r["mu"])
        rows_h = sorted([r for r in per_mu if r["family"] == "obese_spectrum_hw"], key=lambda r: r["mu"])
        xs_b = [r["mu"] for r in rows_b]
        xs_h = [r["mu"] for r in rows_h]
        axes[1].plot(xs_b, [r["obese"] for r in rows_b], "o-", label="obese (BMI-direct)", color="C1")
        axes[1].plot(xs_h, [r["obese"] for r in rows_h], "s--", label="obese (height+weight)", color="C2")
        axes[1].plot(xs_b, [r["average_normal"] for r in rows_b], "o-", label="avg/normal (BMI)", color="C7", alpha=0.4)
        axes[1].plot(xs_h, [r["average_normal"] for r in rows_h], "s--", label="avg/normal (hw)", color="C8", alpha=0.4)
        axes[1].axvline(32, color="k", ls=":", alpha=0.5, label="target BMI=32")
        axes[1].set_xlabel("context μ (BMI)")
        axes[1].set_title("OBESE spectrum (target = BMI 32)")
        axes[1].legend(fontsize=8, loc="best")
        axes[1].set_ylim(-0.05, 1.05)
        fig.suptitle("Behavioral spectrum scan — Claude Sonnet 4.6, 10 samples per (μ × template)", fontsize=11)
        fig.tight_layout()
        OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PLOT, bbox_inches="tight")
        print(f"plot → {OUT_PLOT}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
