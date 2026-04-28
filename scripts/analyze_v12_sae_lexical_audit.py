"""v12 SAE lexical/domain audit.

Uses v11.5 top z-SAE features plus lexical prompt activations produced by
scripts/run_v12_gpu.py. The goal is deliberately conservative: identify whether
top z-correlated sparse features also activate on direct lexical/domain probes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from analyze_v11_sae import SAE_REPO, encode_sae, load_sae  # noqa: E402

RESULTS = REPO / "results" / "v12"
FIGS = REPO / "figures" / "v12"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

MODEL_SHORT = "gemma2-9b"
LATE = 33
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    sae = load_sae(SAE_REPO[MODEL_SHORT], LATE, token)
    src = json.loads((REPO / "results" / "v11_5" / MODEL_SHORT / "sae_features_with_token_freq_control.json").read_text())
    by_pair = {}
    for pair in ALL_PAIRS:
        pair_src = src["by_pair"].get(pair)
        act_path = RESULTS / f"{MODEL_SHORT}_{pair}_L{LATE}_lexical_prompt_activations.npz"
        dense_path = REPO / "results" / "v11" / MODEL_SHORT / pair / f"{MODEL_SHORT}_{pair}_v11_residuals.npz"
        if pair_src is None or not act_path.exists() or not dense_path.exists():
            by_pair[pair] = {"available": False, "reason": "missing top-feature source or lexical activations"}
            continue

        top_ids = [int(x) for x in pair_src["top_50_feat_ids"][:25]]
        dense = np.load(dense_path)
        z = dense["z"].astype(np.float64)
        x = dense["x"].astype(np.float64)
        h = dense["activations"][:, LATE, :].astype(np.float32)
        feats = encode_sae(h, sae)[:, top_ids]
        lex = np.load(act_path)
        lex_feats = encode_sae(lex["activations"].astype(np.float32), sae)[:, top_ids]
        kinds = lex["kinds"].astype(str)
        prompts = lex["prompts"].astype(str)

        base_mean = feats.mean(axis=0)
        base_std = feats.std(axis=0) + 1e-6
        lex_zscore = (lex_feats - base_mean[None, :]) / base_std[None, :]
        kind_scores = {}
        for kind in sorted(set(kinds)):
            mask = kinds == kind
            kind_scores[kind] = {
                "mean_abs_zscore": float(np.mean(np.abs(lex_zscore[mask]))),
                "max_abs_zscore": float(np.max(np.abs(lex_zscore[mask]))),
                "mean_activation": float(np.mean(lex_feats[mask])),
            }

        feature_rows = []
        for j, feat_id in enumerate(top_ids):
            high_lex = float(np.max(np.abs(lex_zscore[:, j])))
            prompt_idx = int(np.argmax(np.abs(lex_zscore[:, j])))
            r2_z = float(pair_src["top_50_r2_z"][j])
            r2_x = float(pair_src["top_50_r2_x"][j])
            r2_tok = float(pair_src["top_50_r2_tok"][j])
            if r2_x > 0.5 * max(r2_z, 1e-9) or r2_tok > 0.5 * max(r2_z, 1e-9):
                label = "raw numeric"
            elif high_lex >= 4.0:
                label = "lexical z-like"
            elif r2_z >= 0.4:
                label = "pure-ish z"
            else:
                label = "mixed/polysemantic"
            feature_rows.append(
                {
                    "feature_id": feat_id,
                    "r2_z": r2_z,
                    "r2_x": r2_x,
                    "r2_token": r2_tok,
                    "max_abs_lexical_zscore": high_lex,
                    "top_lexical_probe_kind": str(kinds[prompt_idx]),
                    "top_lexical_probe": str(prompts[prompt_idx]),
                    "classification": label,
                }
            )

        labels = [r["classification"] for r in feature_rows]
        by_pair[pair] = {
            "available": True,
            "n_features_audited": len(feature_rows),
            "classification_counts": {label: labels.count(label) for label in sorted(set(labels))},
            "lexical_probe_kind_scores": kind_scores,
            "features": feature_rows,
        }
        print(f"[v12-sae] {pair}: {by_pair[pair]['classification_counts']}", flush=True)

    out = {
        "model_short": MODEL_SHORT,
        "layer": LATE,
        "sae_subpath": sae["subpath"],
        "lexical_zscore_threshold_for_lexical_z_like": 4.0,
        "by_pair": by_pair,
    }
    (RESULTS / "sae_feature_lexical_audit.json").write_text(json.dumps(out, indent=2))

    pairs = [p for p in ALL_PAIRS if by_pair.get(p, {}).get("available")]
    labels = ["pure-ish z", "lexical z-like", "raw numeric", "mixed/polysemantic"]
    M = np.array([[by_pair[p]["classification_counts"].get(label, 0) for label in labels] for p in pairs])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(pairs))
    colors = ["C0", "C1", "C3", "0.5"]
    for j, label in enumerate(labels):
        ax.bar(pairs, M[:, j], bottom=bottom, label=label, color=colors[j])
        bottom += M[:, j]
    ax.set_ylabel("top z-SAE features audited")
    ax.set_title("v12 SAE lexical/domain audit @ Gemma 2 9B L33")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "sae_feature_interpretation_examples.png", dpi=150)
    plt.close(fig)

    notes = ["# v12 SAE Feature Notes", ""]
    for pair in pairs:
        notes.append(f"## {pair}")
        notes.append("")
        for row in by_pair[pair]["features"][:5]:
            notes.append(
                f"- feature {row['feature_id']}: {row['classification']}; "
                f"R2(z)={row['r2_z']:.3f}, R2(x)={row['r2_x']:.3f}, "
                f"max lexical z={row['max_abs_lexical_zscore']:.2f}; "
                f"top probe `{row['top_lexical_probe_kind']}` / `{row['top_lexical_probe']}`"
            )
        notes.append("")
    (REPO / "docs" / "v12_sae_feature_notes.md").write_text("\n".join(notes))
    print("[v12-sae] complete")


if __name__ == "__main__":
    main()
