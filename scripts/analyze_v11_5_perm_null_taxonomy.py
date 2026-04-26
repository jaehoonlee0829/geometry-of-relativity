"""v11.5 §B — permutation null on head taxonomy thresholds.

Statistical critic flag: top-quartile thresholds + tag intersections (e.g.
"5 z-writers requires top-quartile on multiple axes") look mechanically
near the quartile expectation. Test whether observed co-tag counts exceed
the 95th percentile of a permutation null.

Procedure: shuffle (layer, head) → (ctx_mass, tgt_mass, r2_z, r2_mu, dla_abs)
mappings 1000× and count how often each tag intersection size meets or
exceeds the observed count.

Output: results/v11_5/<model_short>/taxonomy_perm_null.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

LATE_BY_SHORT = {"gemma2-2b": 20, "gemma2-9b": 33}
N_SHUFFLE = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", required=True, choices=list(LATE_BY_SHORT.keys()))
    args = ap.parse_args()

    tax_path = REPO / "results" / "v11" / args.model_short / "head_taxonomy.json"
    if not tax_path.exists():
        print(f"missing {tax_path}"); return
    tax = json.loads(tax_path.read_text())
    rows = tax["rows"]
    n = len(rows)
    rng = np.random.default_rng(0)

    ctx = np.array([r["attn_mass_ctx_mean"] for r in rows])
    tgt = np.array([r["attn_mass_tgt_mean"] for r in rows])
    r2z = np.array([r["r2_z_head"] for r in rows])
    r2mu = np.array([r["r2_mu_head"] for r in rows])
    dla = np.array([r["dla_abs_mean"] for r in rows])
    layer = np.array([r["layer"] for r in rows])
    late = LATE_BY_SHORT[args.model_short]

    def count_tags(c, t, rz, rm, dl):
        thr_ctx = np.quantile(c, 0.75); thr_tgt = np.quantile(t, 0.75)
        thr_dla = np.quantile(dl, 0.75)
        mu_agg = np.sum((c > thr_ctx) & (rm > 0.5) & (layer <= 5))
        comp = np.sum((rz > 0.5) & (c > thr_ctx * 0.6) & (t > thr_tgt * 0.6))
        zw = np.sum((dl > thr_dla) & (rz > 0.4) & (layer >= max(7, late - 12)))
        return int(mu_agg), int(comp), int(zw)

    obs_mu, obs_comp, obs_zw = count_tags(ctx, tgt, r2z, r2mu, dla)
    print(f"[perm-null] {args.model_short} observed:  μ-agg={obs_mu}  comp={obs_comp}  z-writer={obs_zw}",
          flush=True)

    # Shuffle each metric INDEPENDENTLY across heads — null = "metrics are
    # uncorrelated with which (layer, head) they belong to."
    null_mu, null_comp, null_zw = [], [], []
    for s in range(N_SHUFFLE):
        c_s = rng.permutation(ctx)
        t_s = rng.permutation(tgt)
        rz_s = rng.permutation(r2z)
        rm_s = rng.permutation(r2mu)
        dl_s = rng.permutation(dla)
        m, cp, zw = count_tags(c_s, t_s, rz_s, rm_s, dl_s)
        null_mu.append(m); null_comp.append(cp); null_zw.append(zw)
    null_mu = np.array(null_mu); null_comp = np.array(null_comp); null_zw = np.array(null_zw)

    out = {
        "model_short": args.model_short,
        "n_shuffle": N_SHUFFLE,
        "n_heads_total": int(n),
        "observed": {"mu_aggregator": obs_mu, "comparator": obs_comp, "z_writer": obs_zw},
        "null_mean": {"mu_aggregator": float(null_mu.mean()),
                      "comparator": float(null_comp.mean()),
                      "z_writer": float(null_zw.mean())},
        "null_q95": {"mu_aggregator": float(np.quantile(null_mu, 0.95)),
                     "comparator": float(np.quantile(null_comp, 0.95)),
                     "z_writer": float(np.quantile(null_zw, 0.95))},
        "p_value_observed_geq": {
            "mu_aggregator": float((null_mu >= obs_mu).mean()),
            "comparator": float((null_comp >= obs_comp).mean()),
            "z_writer": float((null_zw >= obs_zw).mean()),
        },
        "interpretation": (
            "p < 0.05 → observed tag count exceeds 95% of permutation null → "
            "the (layer, head) → metric mapping carries real structure for that tag. "
            "p ≥ 0.05 → tag count is consistent with quartile-cutoff structure alone."
        ),
    }
    print(f"[perm-null] null mean: μ-agg={out['null_mean']['mu_aggregator']:.1f}  "
          f"comp={out['null_mean']['comparator']:.1f}  z-writer={out['null_mean']['z_writer']:.1f}")
    print(f"[perm-null] p(observed≥): μ-agg={out['p_value_observed_geq']['mu_aggregator']:.3f}  "
          f"comp={out['p_value_observed_geq']['comparator']:.3f}  "
          f"z-writer={out['p_value_observed_geq']['z_writer']:.3f}")

    out_dir = REPO / "results" / "v11_5" / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "taxonomy_perm_null.json").write_text(json.dumps(out, indent=2))
    print(f"[perm-null] wrote {out_dir / 'taxonomy_perm_null.json'}")


if __name__ == "__main__":
    main()
