"""v12.2 residual-vs-lexical cross-pair transfer.

Minimum viable implementation of docs/NEXT_GPU_SESSION_v12_2.md:

1. Build full / lexical-projection / lexical-residual directions at Gemma 2 9B L33.
2. Run 8x8 cross-pair transfer matrices on seed0 dense target prompts.
3. Compute bootstrap summary metrics over off-diagonal cells.
4. Check target lexical-subspace leakage for every source-target pair.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402
from run_v12_1_lexical_disentanglement import (  # noqa: E402
    ALL_PAIRS,
    MODEL_ID,
    MODEL_SHORT,
    capture_lexical_states,
    get_layers,
    load_npz,
    load_trials,
    primal_z,
    unit,
)

RESULTS = REPO / "results" / "v12_2"
FIGS = REPO / "figures" / "v12_2"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

LEXICAL_BASIS_DIRECTIONS = [
    "d_word_token",
    "d_sentence_token",
    "d_sentence_final",
    "d_synonym_token",
    "d_synonym_sentence_token",
    "d_domain_token",
]


def load_meta(pair: str) -> dict:
    return json.loads(
        (REPO / "results" / "v11" / MODEL_SHORT / pair / f"{MODEL_SHORT}_{pair}_v11_meta.json").read_text()
    )


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def seed0_trials(pair: str, max_n: int) -> list[dict]:
    out = []
    seen = set()
    for t in load_trials(pair):
        if t.get("cell_seed") != 0:
            continue
        key = (round(float(t["x"]), 4), round(float(t["z"]), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    if len(out) > max_n:
        idx = np.linspace(0, len(out) - 1, max_n).round().astype(int)
        out = [out[i] for i in idx]
    return out


def orthonormal_basis(vectors: list[np.ndarray]) -> np.ndarray:
    usable = [unit(v) for v in vectors if np.linalg.norm(v) > 1e-9]
    if not usable:
        return np.zeros((vectors[0].shape[0], 0), dtype=np.float64)
    X = np.stack(usable, axis=1)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, : int((s > 1e-5).sum())].astype(np.float64)


def build_directions(lexical_dirs: dict[str, dict[str, np.ndarray]], layer: int) -> dict:
    by_pair = {}
    for pair in ALL_PAIRS:
        full = unit(primal_z(pair, layer))
        Q = orthonormal_basis([lexical_dirs[pair][k] for k in LEXICAL_BASIS_DIRECTIONS])
        lex = Q @ (Q.T @ full) if Q.shape[1] else np.zeros_like(full)
        resid = full - lex
        by_pair[pair] = {
            "Q": Q,
            "full": full,
            "lexical_projection": unit(lex) if np.linalg.norm(lex) > 1e-9 else lex,
            "lexical_residual": unit(resid) if np.linalg.norm(resid) > 1e-9 else resid,
            "lexical_norm2": float(np.linalg.norm(lex) ** 2),
            "residual_norm": float(np.linalg.norm(resid)),
            "lexical_rank": int(Q.shape[1]),
        }
    return by_pair


def steer_logits(
    model,
    tok,
    prompts: list[str],
    hi_id: int,
    lo_id: int,
    direction: np.ndarray,
    layer: int,
    alpha: float,
    batch_size: int,
    max_seq: int,
) -> np.ndarray:
    d = torch.tensor(unit(direction), dtype=torch.bfloat16, device=model.device)
    layers = get_layers(model)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * d
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    vals = np.zeros(len(prompts), dtype=np.float32)
    handle = layers[layer].register_forward_hook(hook)
    try:
        for b0 in range(0, len(prompts), batch_size):
            batch = prompts[b0 : b0 + batch_size]
            enc = tok(batch, return_tensors="pt", padding="max_length", max_length=max_seq, truncation=True).to(model.device)
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            vals[b0 : b0 + len(batch)] = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
    finally:
        handle.remove()
    return vals


def steering_slope(model, tok, trials: list[dict], direction: np.ndarray, layer: int, args) -> float:
    prompts = [t["prompt"] for t in trials]
    hi_id = first_token_id(tok, trials[0]["high_word"])
    lo_id = first_token_id(tok, trials[0]["low_word"])
    pos = steer_logits(model, tok, prompts, hi_id, lo_id, direction, layer, args.alpha, args.batch_size, args.max_seq)
    neg = steer_logits(model, tok, prompts, hi_id, lo_id, direction, layer, -args.alpha, args.batch_size, args.max_seq)
    return float((pos - neg).mean() / (2 * args.alpha))


def transfer_matrices(model, tok, directions: dict, layer: int, args) -> tuple[dict, dict]:
    rng = np.random.default_rng(122)
    families = ["full", "lexical_projection", "lexical_residual", "random_null"]
    matrices = {fam: {target: {} for target in ALL_PAIRS} for fam in families}
    n_prompts = {}
    random_dirs = {src: rng.normal(size=directions[src]["full"].shape[0]) for src in ALL_PAIRS}
    for target in ALL_PAIRS:
        trials = seed0_trials(target, args.prompts_per_pair)
        n_prompts[target] = len(trials)
        for source in ALL_PAIRS:
            vectors = {
                "full": directions[source]["full"],
                "lexical_projection": directions[source]["lexical_projection"],
                "lexical_residual": directions[source]["lexical_residual"],
                "random_null": random_dirs[source],
            }
            for fam, vec in vectors.items():
                matrices[fam][target][source] = steering_slope(model, tok, trials, vec, layer, args)
            print(
                f"[v12.2] target={target:10s} source={source:10s} "
                f"full={matrices['full'][target][source]:+.3f} "
                f"lex={matrices['lexical_projection'][target][source]:+.3f} "
                f"resid={matrices['lexical_residual'][target][source]:+.3f} "
                f"null={matrices['random_null'][target][source]:+.3f}",
                flush=True,
            )
        (RESULTS / "residual_vs_lexical_transfer.json").write_text(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "model_short": MODEL_SHORT,
                    "layer": layer,
                    "alpha": args.alpha,
                    "single_seed_followup": True,
                    "pairs": ALL_PAIRS,
                    "n_prompts_by_target": n_prompts,
                    "matrices": matrices,
                    "source_direction_metadata": {
                        p: {
                            "lexical_subspace_rank": directions[p]["lexical_rank"],
                            "source_lexical_norm2": directions[p]["lexical_norm2"],
                            "source_residual_norm": directions[p]["residual_norm"],
                        }
                        for p in ALL_PAIRS
                    },
                },
                indent=2,
            )
        )
    out = json.loads((RESULTS / "residual_vs_lexical_transfer.json").read_text())
    return out, matrices


def matrix_array(matrices: dict, fam: str) -> np.ndarray:
    return np.array([[matrices[fam][t][s] for s in ALL_PAIRS] for t in ALL_PAIRS], dtype=np.float64)


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    vals = np.zeros(n_boot)
    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def summary_metrics(matrices: dict, n_boot: int) -> dict:
    rng = np.random.default_rng(123)
    null_off = []
    out = {"families": {}, "paired_offdiag_comparisons": {}, "n_bootstrap": n_boot, "single_seed_followup": True}
    off_mask = ~np.eye(len(ALL_PAIRS), dtype=bool)
    for fam in ["full", "lexical_projection", "lexical_residual", "random_null"]:
        M = matrix_array(matrices, fam)
        diag = np.diag(M)
        off = M[off_mask]
        if fam == "random_null":
            null_off = off
        out["families"][fam] = {
            "mean_diagonal": float(diag.mean()),
            "mean_off_diagonal": float(off.mean()),
            "diagonal_off_diagonal_ratio": float(diag.mean() / off.mean()) if abs(off.mean()) > 1e-12 else None,
            "off_diagonal_positive_fraction": float((off > 0).mean()),
            "off_diagonal_mean_ci95": bootstrap_ci(off, n_boot, 123),
            "target_wise_off_diagonal_mean": {
                t: float(np.delete(M[i, :], i).mean()) for i, t in enumerate(ALL_PAIRS)
            },
            "source_wise_off_diagonal_mean": {
                s: float(np.delete(M[:, j], j).mean()) for j, s in enumerate(ALL_PAIRS)
            },
        }
    null_mean = float(np.mean(null_off))
    null_std = float(np.std(null_off, ddof=1))
    for fam in ["full", "lexical_projection", "lexical_residual"]:
        off = matrix_array(matrices, fam)[off_mask]
        out["families"][fam]["off_diagonal_z_score_vs_random_null"] = (
            float((off.mean() - null_mean) / null_std) if null_std > 1e-12 else None
        )
    comparisons = {
        "residual_minus_lexical": matrix_array(matrices, "lexical_residual")[off_mask]
        - matrix_array(matrices, "lexical_projection")[off_mask],
        "residual_minus_full": matrix_array(matrices, "lexical_residual")[off_mask]
        - matrix_array(matrices, "full")[off_mask],
        "lexical_minus_full": matrix_array(matrices, "lexical_projection")[off_mask]
        - matrix_array(matrices, "full")[off_mask],
    }
    for name, diff in comparisons.items():
        out["paired_offdiag_comparisons"][name] = {
            "mean_difference": float(diff.mean()),
            "ci95": bootstrap_ci(diff, n_boot, 456),
            "positive_fraction": float((diff > 0).mean()),
        }
    (RESULTS / "residual_vs_lexical_transfer_summary.json").write_text(json.dumps(out, indent=2))
    return out


def leakage_check(directions: dict, matrices: dict) -> dict:
    out = {
        "model_short": MODEL_SHORT,
        "layer": 33,
        "pairs": ALL_PAIRS,
        "target_lexical_overlap": {fam: {t: {} for t in ALL_PAIRS} for fam in ["full", "lexical_projection", "lexical_residual"]},
        "offdiag_correlations_with_transfer": {},
    }
    for target in ALL_PAIRS:
        Qt = directions[target]["Q"]
        for source in ALL_PAIRS:
            for fam in ["full", "lexical_projection", "lexical_residual"]:
                d = directions[source][fam]
                val = float(np.linalg.norm(Qt.T @ unit(d)) ** 2) if Qt.shape[1] else 0.0
                out["target_lexical_overlap"][fam][target][source] = val
    off_mask = ~np.eye(len(ALL_PAIRS), dtype=bool)
    for fam in ["full", "lexical_projection", "lexical_residual"]:
        overlap = np.array(
            [[out["target_lexical_overlap"][fam][t][s] for s in ALL_PAIRS] for t in ALL_PAIRS],
            dtype=float,
        )[off_mask]
        transfer = matrix_array(matrices, fam)[off_mask]
        corr = float(np.corrcoef(overlap, transfer)[0, 1]) if overlap.std() > 1e-12 and transfer.std() > 1e-12 else 0.0
        out["offdiag_correlations_with_transfer"][fam] = corr
    (RESULTS / "target_lexical_subspace_leakage.json").write_text(json.dumps(out, indent=2))
    return out


def plot_matrices(matrices: dict) -> None:
    fams = ["full", "lexical_projection", "lexical_residual", "random_null"]
    arrays = [matrix_array(matrices, fam) for fam in fams]
    lim = max(0.01, max(float(np.nanmax(np.abs(a))) for a in arrays))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.7), sharey=True)
    for ax, fam, M in zip(axes, fams, arrays):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(fam)
        ax.set_xticks(range(len(ALL_PAIRS)), ALL_PAIRS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(ALL_PAIRS)), ALL_PAIRS, fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Delta logit-diff per alpha")
    fig.suptitle("v12.2 residual vs lexical cross-pair transfer @ L33")
    fig.savefig(FIGS / "residual_vs_lexical_transfer_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: dict) -> None:
    fams = ["full", "lexical_projection", "lexical_residual", "random_null"]
    diag = [summary["families"][f]["mean_diagonal"] for f in fams]
    off = [summary["families"][f]["mean_off_diagonal"] for f in fams]
    ci = [summary["families"][f]["off_diagonal_mean_ci95"] for f in fams]
    yerr = np.array([[off[i] - ci[i][0] for i in range(len(fams))], [ci[i][1] - off[i] for i in range(len(fams))]])
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, diag, width=0.36, label="diagonal mean")
    ax.bar(x + 0.18, off, width=0.36, yerr=yerr, label="off-diagonal mean (bootstrap CI)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, fams, rotation=25, ha="right")
    ax.set_ylabel("Delta logit-diff per alpha")
    ax.set_title("v12.2 transfer summary (single-seed follow-up)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "residual_vs_lexical_transfer_summary.png", dpi=150)
    plt.close(fig)


def plot_leakage(leak: dict, matrices: dict) -> None:
    fams = ["full", "lexical_projection", "lexical_residual"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, fam in zip(axes[0], fams):
        M = np.array([[leak["target_lexical_overlap"][fam][t][s] for s in ALL_PAIRS] for t in ALL_PAIRS])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(0.01, float(np.nanmax(M))))
        ax.set_title(f"{fam}: target lexical overlap")
        ax.set_xticks(range(len(ALL_PAIRS)), ALL_PAIRS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(ALL_PAIRS)), ALL_PAIRS, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    off_mask = ~np.eye(len(ALL_PAIRS), dtype=bool)
    for ax, fam in zip(axes[1], fams):
        overlap = np.array([[leak["target_lexical_overlap"][fam][t][s] for s in ALL_PAIRS] for t in ALL_PAIRS])[off_mask]
        transfer = matrix_array(matrices, fam)[off_mask]
        ax.scatter(overlap, transfer, s=18, alpha=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("target lexical overlap")
        ax.set_ylabel("transfer slope")
        ax.set_title(f"{fam}: r={leak['offdiag_correlations_with_transfer'][fam]:+.2f}")
        ax.grid(alpha=0.25)
    fig.suptitle("v12.2 target lexical-subspace leakage")
    fig.tight_layout()
    fig.savefig(FIGS / "target_lexical_subspace_leakage.png", dpi=150)
    plt.close(fig)


def write_summary_doc(summary: dict, leak: dict) -> None:
    fams = ["full", "lexical_projection", "lexical_residual", "random_null"]
    lines = [
        "# V12.2 Residual vs Lexical Transfer Summary",
        "",
        "V12.2 compares cross-pair transfer from full `primal_z`, its lexical",
        "projection, and its lexical residual at Gemma 2 9B L33. This is a",
        "single-seed steering follow-up using seed0 unique target cells.",
        "",
        "## Aggregate Transfer",
        "",
    ]
    for fam in fams:
        row = summary["families"][fam]
        lines.append(
            f"- {fam}: diag={row['mean_diagonal']:+.3f}, offdiag={row['mean_off_diagonal']:+.3f}, "
            f"ratio={row['diagonal_off_diagonal_ratio'] if row['diagonal_off_diagonal_ratio'] is not None else 'n/a'}, "
            f"offdiag_pos={row['off_diagonal_positive_fraction']:.2f}"
        )
    lines += [
        "",
        "## Paired Off-diagonal Comparisons",
        "",
    ]
    for name, row in summary["paired_offdiag_comparisons"].items():
        lines.append(f"- {name}: mean={row['mean_difference']:+.3f}, CI95=[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}], positive_fraction={row['positive_fraction']:.2f}")
    lines += [
        "",
        "## Target Lexical-subspace Leakage",
        "",
    ]
    for fam, corr in leak["offdiag_correlations_with_transfer"].items():
        lines.append(f"- corr({fam} offdiag transfer, target lexical overlap) = {corr:+.3f}")
    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- These are single-seed transfer matrices, not BH-FDR significance claims.",
        "- Residual transfer should not be called non-lexical unless target-side lexical overlap is low.",
        "- Projection/residual directions are normalized before steering, so matrix values compare intervention potency, not vector-energy fractions.",
    ]
    (REPO / "docs" / "V12_2_RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--prompts-per-pair", type=int, default=160)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    print(f"[v12.2] loading {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    print("[v12.2] model loaded", flush=True)

    _, lexical_dirs = capture_lexical_states(model, tok, args.layer, args.batch_size)
    directions = build_directions(lexical_dirs, args.layer)
    _, matrices = transfer_matrices(model, tok, directions, args.layer, args)
    summary = summary_metrics(matrices, args.n_bootstrap)
    leak = leakage_check(directions, matrices)
    plot_matrices(matrices)
    plot_summary(summary)
    plot_leakage(leak, matrices)
    write_summary_doc(summary, leak)
    print("[v12.2] complete", flush=True)


if __name__ == "__main__":
    main()
