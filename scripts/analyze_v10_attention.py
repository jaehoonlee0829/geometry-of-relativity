"""v10 P4 — attention-circuit analysis on dumped activations.

Implements the DLA-on-dumped-activations recipe from the research agent:

Per (strategic-layer ℓ, head h):
  - attn_mass_ctx, attn_mass_tgt, attn_mass_other  (last-token softmax mass)
  - attn_entropy                                   (selectivity)
  - comparison_score   = mean_i (attn_to_ctx_i × attn_to_tgt)
  - R²(z|head_out), R²(x|head_out), R²(μ|head_out) per head
  - DLA score          = w_z @ head_contribution_to_residual
                         where head_contribution = W_O[:, h*hd:(h+1)*hd] @ head_out
                         and  w_z is the ridge probe for z fit at L=20
  - faithfulness_topk  = sum of |DLA score| over top-k heads vs total probe logit

Classifier:
  - μ-aggregator : high attn_mass_ctx, high R²(μ|head_out), low entropy variance
                   across context positions
  - comparator   : high comparison_score, high R²(z|head_out)
  - z-writer     : high |DLA score| at layer ≥13, high R²(z|head_out)

Inputs:
  results/v10/gemma2_height_v10_residuals.npz
  results/v10/gemma2_height_v10_attention.npz
  results/v10/gemma2_height_v10_W_O_strategic.npz

Outputs:
  results/v10/attention_per_head.json
  figures/v10/attention_mass_heatmaps.png
  figures/v10/head_r2z_heatmap.png
  figures/v10/dla_score_heatmap.png
  figures/v10/head_taxonomy_scatter.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "v10"
FIG = REPO / "figures" / "v10"
FIG.mkdir(parents=True, exist_ok=True)


def cv_r2(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    preds = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def fit_z_probe(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit ridge probe of z on residual; return (w, intercept)."""
    m = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(X, y)
    return m.coef_.astype(np.float64), float(m.intercept_)


def main() -> None:
    print("[P4] loading...", flush=True)
    res = np.load(RES / "gemma2_height_v10_residuals.npz")
    atn = np.load(RES / "gemma2_height_v10_attention.npz")
    wos = np.load(RES / "gemma2_height_v10_W_O_strategic.npz")
    meta = json.loads((RES / "gemma2_height_v10_meta.json").read_text())

    acts = res["activations"].astype(np.float32)        # (N, 26, 2304)
    z = res["z"].astype(np.float64)
    x = res["x"].astype(np.float64)
    mu = res["mu"].astype(np.float64)
    N = acts.shape[0]

    attn_last = atn["attn_last_row"]                     # (N, 8L, 8H, MAX_SEQ)
    head_out = atn["head_outputs"].astype(np.float32)    # (N, 8L, 8H, head_dim)
    strat = atn["attn_layers"]                            # (8,)
    seq_unpad = atn["seq_len_unpadded"]
    seq_pad = int(atn["seq_len_padded"])
    target_pos_unpad = atn["target_pos_unpadded"]         # (N, 2)
    context_pos_unpad = atn["context_pos_unpadded"]       # (N, 15, 2)

    n_layers_meta = meta["n_layers"]
    n_heads = meta["n_heads"]
    head_dim = meta["head_dim"]
    d_model = meta["d_model"]

    print(f"[P4] N={N}  strategic_layers={list(strat)}  n_heads={n_heads}  "
          f"head_dim={head_dim}  d_model={d_model}", flush=True)

    # ---- precompute padded positions per prompt
    pad_off = seq_pad - seq_unpad   # (N,)

    # Build attention-mass aggregators
    print("[P4] computing per-prompt attention masses...", flush=True)
    attn_mass_ctx = np.zeros((N, len(strat), n_heads), dtype=np.float64)
    attn_mass_tgt = np.zeros_like(attn_mass_ctx)
    attn_entropy = np.zeros_like(attn_mass_ctx)
    comparison = np.zeros_like(attn_mass_ctx)

    for i in range(N):
        po = int(pad_off[i])
        # Padded positions of context-number tokens (sum of windows)
        ctx_padded_idx: list[int] = []
        for k in range(15):
            s, e = context_pos_unpad[i, k]
            ctx_padded_idx.extend(range(po + int(s), po + int(e)))
        ctx_padded_idx = np.array(ctx_padded_idx, dtype=np.int64)

        ts, te = target_pos_unpad[i]
        tgt_padded_idx = np.arange(po + int(ts), po + int(te), dtype=np.int64)

        for li in range(len(strat)):
            for h in range(n_heads):
                row = attn_last[i, li, h].astype(np.float64)
                m_ctx = row[ctx_padded_idx].sum()
                m_tgt = row[tgt_padded_idx].sum()
                attn_mass_ctx[i, li, h] = m_ctx
                attn_mass_tgt[i, li, h] = m_tgt
                # entropy of valid (unpadded) tail
                p = row[po:]                # actual prompt positions
                p = p / max(p.sum(), 1e-12)
                attn_entropy[i, li, h] = float(-(p * np.log(p + 1e-30)).sum())
                # comparison_score: mean_i [attn_ctx_i × attn_tgt]
                # attn_tgt is a scalar per head; comparison = m_tgt × mean(ctx attn)
                # use per-context-token product mean: (ctx_attn.mean()) × m_tgt
                if len(ctx_padded_idx) > 0:
                    comparison[i, li, h] = (
                        row[ctx_padded_idx].mean() * m_tgt
                    )

    # Aggregate over prompts → per-(layer, head) summary
    am_ctx_mean = attn_mass_ctx.mean(0)
    am_tgt_mean = attn_mass_tgt.mean(0)
    am_other_mean = 1.0 - am_ctx_mean - am_tgt_mean
    ent_mean = attn_entropy.mean(0)
    comp_mean = comparison.mean(0)

    # ---- R²(z|head_out), R²(x|head_out), R²(μ|head_out)
    print("[P4] computing per-head R² for z/x/mu...", flush=True)
    r2z = np.zeros((len(strat), n_heads))
    r2x = np.zeros_like(r2z)
    r2mu = np.zeros_like(r2z)
    for li in range(len(strat)):
        for h in range(n_heads):
            X = head_out[:, li, h, :]   # (N, head_dim)
            r2z[li, h] = cv_r2(X, z)
            r2x[li, h] = cv_r2(X, x)
            r2mu[li, h] = cv_r2(X, mu)
        print(f"[P4]   layer {int(strat[li]):2d}: R²(z) per-head = "
              + " ".join(f"{v:.2f}" for v in r2z[li]), flush=True)

    # ---- DLA score: project head contribution onto z-probe direction at L20
    print("[P4] fitting z-probe at layer 20 + DLA scores...", flush=True)
    L_target = 20
    w_z, b_z = fit_z_probe(acts[:, L_target, :], z)
    # For each strategic layer ℓ ≤ L_target:
    #   head_contribution_residual[h] = W_O_ℓ[:, h*hd:(h+1)*hd] @ head_out_ℓ[h]
    #   dla_h = w_z @ head_contribution_residual
    dla = np.zeros((len(strat), n_heads, N), dtype=np.float64)
    for li, L in enumerate(strat):
        WO = wos[f"L{int(L)}"]   # (d_model, n_heads*head_dim)
        for h in range(n_heads):
            WO_h = WO[:, h * head_dim:(h + 1) * head_dim]  # (d_model, head_dim)
            ho = head_out[:, li, h, :]                       # (N, head_dim)
            contrib = ho @ WO_h.T                             # (N, d_model)
            dla[li, h] = contrib @ w_z                        # (N,)

    dla_mean_abs = np.abs(dla).mean(2)         # (8L, n_heads)
    dla_mean_signed = dla.mean(2)              # (8L, n_heads)
    dla_corr_z = np.array([
        [np.corrcoef(dla[li, h], z)[0, 1] for h in range(n_heads)]
        for li in range(len(strat))
    ])

    # ---- Faithfulness: sum of top-k DLA scores vs probe-direction logit
    # probe-direction logit per prompt = w_z @ resid_L = z_pred (prior to intercept)
    z_pred = acts[:, L_target, :] @ w_z + b_z
    dla_total_per_prompt = dla.sum((0, 1))   # (N,)
    # variance explained of z_pred by sum of DLA scores from sub-target layers
    fa_corr = float(np.corrcoef(z_pred, dla_total_per_prompt)[0, 1])
    print(f"[P4] faithfulness corr(z_pred, sum_DLA) = {fa_corr:.3f}", flush=True)

    # ---- Classify heads
    classifications: list[dict] = []
    for li, L in enumerate(strat):
        for h in range(n_heads):
            tags = []
            if am_ctx_mean[li, h] > 0.4 and r2mu[li, h] > 0.3:
                tags.append("mu-aggregator")
            if comp_mean[li, h] > 0.005 and r2z[li, h] > 0.2:
                tags.append("comparator")
            if int(L) >= 13 and abs(dla_mean_signed[li, h]) > 0.05 and r2z[li, h] > 0.3:
                tags.append("z-writer")
            classifications.append({
                "layer": int(L), "head": int(h), "tags": tags,
                "attn_mass_ctx": float(am_ctx_mean[li, h]),
                "attn_mass_tgt": float(am_tgt_mean[li, h]),
                "attn_entropy_mean": float(ent_mean[li, h]),
                "comparison_score": float(comp_mean[li, h]),
                "r2_z": float(r2z[li, h]),
                "r2_x": float(r2x[li, h]),
                "r2_mu": float(r2mu[li, h]),
                "dla_mean_signed": float(dla_mean_signed[li, h]),
                "dla_mean_abs": float(dla_mean_abs[li, h]),
                "dla_corr_z": float(dla_corr_z[li, h]),
            })

    out = {
        "strategic_layers": list(map(int, strat)),
        "n_heads": int(n_heads),
        "L_target_for_probe": int(L_target),
        "faithfulness_corr_z_pred_sum_dla": fa_corr,
        "heads": classifications,
    }
    json_path = RES / "attention_per_head.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"[P4] wrote {json_path}", flush=True)

    # ---- figures
    layer_labels = [f"L{int(L)}" for L in strat]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, M, title in [
        (axes[0, 0], am_ctx_mean, "attn mass to context"),
        (axes[0, 1], am_tgt_mean, "attn mass to target"),
        (axes[1, 0], ent_mean, "attention entropy"),
        (axes[1, 1], comp_mean, "comparison score (ctx_mean × tgt)"),
    ]:
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(strat)))
        ax.set_yticklabels(layer_labels)
        ax.set_xticks(range(n_heads))
        ax.set_xlabel("head")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIG / "attention_mass_heatmaps.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, M, title, cmap in [
        (axes[0], r2z, "R²(z | head_out)", "magma"),
        (axes[1], r2x, "R²(x | head_out)", "magma"),
        (axes[2], r2mu, "R²(μ | head_out)", "magma"),
    ]:
        im = ax.imshow(np.clip(M, 0, 1), aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(len(strat)))
        ax.set_yticklabels(layer_labels)
        ax.set_xticks(range(n_heads))
        ax.set_xlabel("head")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIG / "head_r2z_heatmap.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    vmax = max(abs(dla_mean_signed.min()), abs(dla_mean_signed.max()))
    im0 = axes[0].imshow(dla_mean_signed, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax)
    axes[0].set_yticks(range(len(strat))); axes[0].set_yticklabels(layer_labels)
    axes[0].set_xticks(range(n_heads)); axes[0].set_xlabel("head")
    axes[0].set_title("DLA score on z-probe (mean signed)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(dla_corr_z, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_yticks(range(len(strat))); axes[1].set_yticklabels(layer_labels)
    axes[1].set_xticks(range(n_heads)); axes[1].set_xlabel("head")
    axes[1].set_title("corr(DLA_score, z) per head")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIG / "dla_score_heatmap.png", dpi=120)
    plt.close()

    # taxonomy scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for li, L in enumerate(strat):
        for h in range(n_heads):
            colour = "C0"
            tags = next((c["tags"] for c in classifications
                         if c["layer"] == int(L) and c["head"] == h), [])
            if "z-writer" in tags:
                colour = "red"
            elif "comparator" in tags:
                colour = "orange"
            elif "mu-aggregator" in tags:
                colour = "blue"
            ax.scatter(r2mu[li, h], r2z[li, h],
                       s=40 + 200 * abs(dla_mean_signed[li, h]),
                       c=colour, alpha=0.7, edgecolors="black", lw=0.4)
            if abs(dla_mean_signed[li, h]) > 0.05 or r2z[li, h] > 0.5:
                ax.annotate(f"L{int(L)}h{h}", (r2mu[li, h], r2z[li, h]),
                            fontsize=7, alpha=0.8)
    ax.set_xlabel("R²(μ | head_out)")
    ax.set_ylabel("R²(z | head_out)")
    ax.set_title("Head taxonomy: red=z-writer, orange=comparator, blue=μ-aggregator\n"
                 "(size ∝ |DLA score| on z-probe)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "head_taxonomy_scatter.png", dpi=120)
    plt.close()

    print(f"[P4] wrote 4 figures to {FIG}", flush=True)
    n_ww = sum(1 for c in classifications if "z-writer" in c["tags"])
    n_cmp = sum(1 for c in classifications if "comparator" in c["tags"])
    n_mu = sum(1 for c in classifications if "mu-aggregator" in c["tags"])
    print(f"[P4] head taxonomy: {n_mu} μ-aggregators, "
          f"{n_cmp} comparators, {n_ww} z-writers", flush=True)


if __name__ == "__main__":
    main()
