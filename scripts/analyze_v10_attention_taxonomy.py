"""v10 P4 follow-up — re-tune head taxonomy with adaptive thresholds.

The first pass used absolute thresholds that were calibrated for a different
attention regime. Here we use top-quartile cuts per metric so the taxonomy
captures *relative* head specialization on this 4000-prompt grid.

Inputs:  results/v10/attention_per_head.json
Outputs: results/v10/attention_per_head_taxonomy.json
         figures/v10/attention_taxonomy_grid.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "v10"
FIG = REPO / "figures" / "v10"


def main() -> None:
    src = json.loads((RES / "attention_per_head.json").read_text())
    heads = src["heads"]
    layers_used = sorted(set(h["layer"] for h in heads))
    n_heads = src["n_heads"]

    # Distributions
    am_ctx = np.array([h["attn_mass_ctx"] for h in heads])
    am_tgt = np.array([h["attn_mass_tgt"] for h in heads])
    r2z = np.array([h["r2_z"] for h in heads])
    r2mu = np.array([h["r2_mu"] for h in heads])
    r2x = np.array([h["r2_x"] for h in heads])
    dla_signed = np.array([h["dla_mean_signed"] for h in heads])
    dla_abs = np.array([h["dla_mean_abs"] for h in heads])

    # Thresholds (top-quartile of distribution where larger = more specialized)
    thr_ctx = np.quantile(am_ctx, 0.75)
    thr_tgt = np.quantile(am_tgt, 0.75)
    thr_dla = np.quantile(dla_abs, 0.75)
    print(f"thr_ctx={thr_ctx:.4f}  thr_tgt={thr_tgt:.4f}  thr_dla={thr_dla:.4f}",
          flush=True)

    new = []
    for i, h in enumerate(heads):
        tags = []
        # μ-aggregator: high context attention + can decode μ from output
        if am_ctx[i] > thr_ctx and r2mu[i] > 0.7:
            tags.append("mu-aggregator")
        # comparator: above-median attention to target AND >median attention to ctx
        if am_tgt[i] > np.quantile(am_tgt, 0.5) and am_ctx[i] > np.quantile(am_ctx, 0.5):
            tags.append("comparator")
        # z-writer: large absolute DLA score + at L >=10 (before that the
        # readout direction may be very different)
        if dla_abs[i] > thr_dla and h["layer"] >= 10 and r2z[i] > 0.6:
            tags.append("z-writer")
        rec = dict(h)
        rec["tags_v2"] = tags
        new.append(rec)

    # Save
    out = dict(src)
    out["heads"] = new
    out["thresholds_v2"] = {
        "attn_mass_ctx_top25": float(thr_ctx),
        "attn_mass_tgt_top25": float(thr_tgt),
        "dla_mean_abs_top25": float(thr_dla),
    }
    (RES / "attention_per_head_taxonomy.json").write_text(json.dumps(out, indent=2))

    n_ww = sum("z-writer" in r["tags_v2"] for r in new)
    n_cmp = sum("comparator" in r["tags_v2"] for r in new)
    n_mu = sum("mu-aggregator" in r["tags_v2"] for r in new)
    print(f"v2 taxonomy: {n_mu} μ-aggregators, {n_cmp} comparators, "
          f"{n_ww} z-writers", flush=True)

    # ---- figure: layer × head grid coloured by primary tag
    color = {"mu-aggregator": "tab:blue", "comparator": "tab:orange",
             "z-writer": "tab:red"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in new:
        L = r["layer"]; h = r["head"]
        tags = r["tags_v2"]
        if not tags:
            ax.scatter(L, h, s=80, marker="x", c="lightgray", alpha=0.6)
            continue
        # Draw layered markers per tag
        for ti, tag in enumerate(tags):
            ax.scatter(L + (ti - len(tags) / 2 + 0.5) * 0.25, h,
                       s=80 + 200 * abs(r["dla_mean_signed"]),
                       c=color[tag], alpha=0.7, edgecolors="black", lw=0.5,
                       label=tag if (L, h, ti) == (new[0]["layer"], new[0]["head"], 0) else None)
    # Legend
    handles = [plt.scatter([], [], c=v, s=60, label=k, edgecolors="black",
                           lw=0.5)
               for k, v in color.items()]
    handles.append(plt.scatter([], [], c="lightgray", marker="x", s=60,
                               label="untagged"))
    ax.legend(handles=handles, loc="upper right")
    ax.set_xticks(layers_used); ax.set_xlabel("layer")
    ax.set_yticks(range(n_heads)); ax.set_ylabel("head index")
    ax.set_title("v10 attention head taxonomy (size ∝ |DLA score|)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "attention_taxonomy_grid.png", dpi=120)
    plt.close()
    print(f"wrote {RES}/attention_per_head_taxonomy.json + figure")

    print("\n--- top-10 |DLA| heads ---", flush=True)
    for r in sorted(new, key=lambda x: -x["dla_mean_abs"])[:10]:
        print(f"  L{r['layer']:>2}h{r['head']}  |DLA|={r['dla_mean_abs']:.3f}  "
              f"DLA_signed={r['dla_mean_signed']:+.3f}  R²(z)={r['r2_z']:.2f}  "
              f"R²(μ)={r['r2_mu']:.2f}  attn_ctx={r['attn_mass_ctx']:.3f}  "
              f"attn_tgt={r['attn_mass_tgt']:.3f}  tags={r['tags_v2']}")


if __name__ == "__main__":
    main()
