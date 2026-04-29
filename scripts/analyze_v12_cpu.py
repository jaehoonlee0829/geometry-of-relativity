"""v12 CPU analyses and paper-grade figures.

This script consumes the existing v11/v11.5 dense activation dumps and writes
the CPU-side v12 deliverables:

  results/v12/layer_sweep_9b.json
  figures/v12/layer_sweep_9b_combined.png
  results/v12/direction_redteam_x_lexical_z.json
  figures/v12/direction_redteam_cosines.png
  results/v12/pc_extremeness_x_z_audit.json
  figures/v12/pc_extremeness_x_z_grid.png

GPU-only steering outputs are merged into the same v12 namespace by
scripts/run_v12_gpu.py when available.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "v12"
FIGS = REPO / "figures" / "v12"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
MODELS = ["gemma2-2b", "gemma2-9b"]
LATE = {"gemma2-2b": 20, "gemma2-9b": 33}


def load_npz(model_short: str, pair: str):
    p = REPO / "results" / "v11" / model_short / pair / f"{model_short}_{pair}_v11_residuals.npz"
    return np.load(p) if p.exists() else None


def load_meta(model_short: str, pair: str) -> dict:
    p = REPO / "results" / "v11" / model_short / pair / f"{model_short}_{pair}_v11_meta.json"
    return json.loads(p.read_text())


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def r2(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.std() < 1e-12 or v.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(u, v)[0, 1] ** 2)


def cell_groups(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    keys = np.array([f"{round(float(xi), 4)}_{round(float(zi), 4)}" for xi, zi in zip(x, z)])
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def group_cv_r2(h: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return float("nan")
    preds = np.zeros_like(y, dtype=np.float64)
    for tr, te in GroupKFold(n_splits=n_splits).split(h, y, groups):
        m = Ridge(alpha=1.0)
        m.fit(h[tr], y[tr])
        preds[te] = m.predict(h[te])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def cell_mean(acts: np.ndarray, x: np.ndarray, z: np.ndarray):
    groups = cell_groups(x, z)
    n = int(groups.max() + 1)
    out = np.zeros((n, acts.shape[1], acts.shape[2]), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int32)
    cx = np.zeros(n, dtype=np.float64)
    cz = np.zeros(n, dtype=np.float64)
    for i, g in enumerate(groups):
        out[g] += acts[i].astype(np.float64)
        cnt[g] += 1
        cx[g] = float(x[i])
        cz[g] = float(z[i])
    out /= np.maximum(cnt, 1)[:, None, None]
    return out, cx, cz


def primal_direction(acts: np.ndarray, z: np.ndarray, layer: int) -> np.ndarray:
    h = acts[:, layer, :].astype(np.float64)
    return h[z > 1.0].mean(0) - h[z < -1.0].mean(0)


def raw_x_direction(acts: np.ndarray, x: np.ndarray, layer: int) -> np.ndarray:
    h = acts[:, layer, :].astype(np.float64)
    q25, q75 = np.quantile(x.astype(np.float64), [0.25, 0.75])
    return h[x >= q75].mean(0) - h[x <= q25].mean(0)


def probe_direction(acts: np.ndarray, y: np.ndarray, layer: int) -> np.ndarray:
    h = acts[:, layer, :].astype(np.float64)
    m = Ridge(alpha=1.0).fit(h, y.astype(np.float64))
    return m.coef_.astype(np.float64)


def layer_sweep_9b() -> dict:
    by_pair = {}
    for pair in ALL_PAIRS:
        d = load_npz("gemma2-9b", pair)
        if d is None:
            continue
        acts = d["activations"]
        z = d["z"].astype(np.float64)
        x = d["x"].astype(np.float64)
        groups = cell_groups(x, z)
        n_layers = acts.shape[1]
        fold_path = REPO / "results" / "v11_5" / "gemma2-9b" / pair / "increment_r2_fold_aware.json"
        fold = json.loads(fold_path.read_text()) if fold_path.exists() else {}
        records = []
        prev = None
        for L in range(n_layers):
            h = acts[:, L, :].astype(np.float64)
            pz = h[z > 1.0].mean(0) - h[z < -1.0].mean(0)
            records.append(
                {
                    "layer": L,
                    "r2_cv_z": float(fold.get("naive_r2_per_layer", [])[L])
                    if fold.get("naive_r2_per_layer")
                    else group_cv_r2(h, z, groups),
                    "increment_r2_fold_aware": float(
                        fold.get("orth_r2_per_layer_FOLD_AWARE", fold.get("orth_r2_per_layer", []))[L]
                    )
                    if (fold.get("orth_r2_per_layer_FOLD_AWARE") or fold.get("orth_r2_per_layer"))
                    else float("nan"),
                    "primal_norm": float(np.linalg.norm(pz)),
                    "primal_cos_prev_layer": cos(pz, prev) if prev is not None else float("nan"),
                    "r2_cv_x": group_cv_r2(h, x, groups),
                }
            )
            prev = pz
        by_pair[pair] = {
            "n_prompts": int(acts.shape[0]),
            "n_layers": int(n_layers),
            "layer_records": records,
        }
        print(f"[v12-cpu] layer sweep loaded gemma2-9b/{pair}")
    out = {"model_short": "gemma2-9b", "pairs": by_pair}
    (RESULTS / "layer_sweep_9b.json").write_text(json.dumps(strict_json_ready(out), indent=2, allow_nan=False))
    return out


def plot_layer_sweep(layer: dict) -> None:
    pairs = list(layer["pairs"])
    n_layers = next(iter(layer["pairs"].values()))["n_layers"]
    xs = np.arange(n_layers)

    def mat(key):
        return np.array([[r[key] for r in layer["pairs"][p]["layer_records"]] for p in pairs], dtype=float)

    r2z = mat("r2_cv_z")
    incr = mat("increment_r2_fold_aware")
    pnorm = mat("primal_norm")
    cprev = mat("primal_cos_prev_layer")
    steer_path = RESULTS / "layer_sweep_9b_steering.json"
    steer = json.loads(steer_path.read_text()) if steer_path.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    ax = axes[0, 0]
    ax.plot(xs, np.nanmean(r2z, 0), "-o", ms=3, lw=2)
    ax.fill_between(xs, np.nanmin(r2z, 0), np.nanmax(r2z, 0), alpha=0.18)
    ax.set_title("CV R2(z): availability")
    ax.set_xlabel("layer")
    ax.set_ylabel("R2")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(xs, np.nanmean(incr, 0), "-o", ms=3, lw=2, color="C1")
    ax.fill_between(xs, np.nanmin(incr, 0), np.nanmax(incr, 0), alpha=0.18, color="C1")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Fold-aware incremental R2(z)")
    ax.set_xlabel("layer")
    ax.set_ylabel("incremental R2")
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    ax.plot(xs, np.nanmean(pnorm, 0), "-o", ms=3, lw=2, color="C2")
    ax.fill_between(xs, np.nanmin(pnorm, 0), np.nanmax(pnorm, 0), alpha=0.18, color="C2")
    ax.set_yscale("log")
    ax.set_title("primal_z norm")
    ax.set_xlabel("layer")
    ax.set_ylabel("norm")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    if steer:
        layers = steer["layers"]
        for direction, color in [("primal_z", "C0"), ("probe_z", "C3"), ("random_null", "0.4")]:
            vals = np.array(
                [[steer["by_pair"][p][str(L)].get(direction, np.nan) for L in layers] for p in steer["pairs"]]
            )
            ax.plot(layers, np.nanmean(vals, 0), "-o", ms=4, lw=2, label=direction, color=color)
            ax.fill_between(layers, np.nanmin(vals, 0), np.nanmax(vals, 0), alpha=0.15, color=color)
    else:
        ax.text(0.5, 0.5, "steering JSON not present yet", ha="center", va="center")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Causal steering slope")
    ax.set_xlabel("layer")
    ax.set_ylabel("Delta logit-diff per alpha")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(xs, np.nanmean(cprev, 0), "-o", ms=3, lw=2, color="C4")
    ax.fill_between(xs, np.nanmin(cprev, 0), np.nanmax(cprev, 0), alpha=0.18, color="C4")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("cos(primal_z[L], primal_z[L-1])")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    if steer:
        layers = steer["layers"]
        vals = np.array([[steer["by_pair"][p][str(L)].get("primal_z", np.nan) for L in layers] for p in steer["pairs"]])
        ax.scatter(np.nanmean(r2z, 0)[layers], np.nanmean(vals, 0), s=50)
        for L, a, b in zip(layers, np.nanmean(r2z, 0)[layers], np.nanmean(vals, 0)):
            ax.text(a, b, str(L), fontsize=8)
        ax.set_xlabel("mean CV R2(z)")
        ax.set_ylabel("mean primal steering slope")
    else:
        ax.text(0.5, 0.5, "rerun after steering", ha="center", va="center")
    ax.set_title("Decodability vs use")
    ax.grid(alpha=0.25)

    fig.suptitle("v12 Gemma 2 9B layer sweep: encode vs use", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "layer_sweep_9b_combined.png", dpi=150)
    plt.close(fig)


def direction_redteam() -> dict:
    out = {"models": {}}
    for model_short in MODELS:
        late = LATE[model_short]
        model_out = {}
        wu_path = REPO / "results" / "v11" / model_short / f"{model_short}_W_U.npz"
        wu = np.load(wu_path)["W_U"].astype(np.float64) if wu_path.exists() else None
        for pair in ALL_PAIRS:
            d = load_npz(model_short, pair)
            if d is None:
                continue
            meta = load_meta(model_short, pair)
            acts = d["activations"]
            z = d["z"].astype(np.float64)
            x = d["x"].astype(np.float64)
            pz = primal_direction(acts, z, late)
            px = raw_x_direction(acts, x, late)
            qz = probe_direction(acts, z, late)
            qx = probe_direction(acts, x, late)
            row = {
                "late_layer": late,
                "cos_primal_z_primal_x": cos(pz, px),
                "cos_probe_z_probe_x": cos(qz, qx),
                "cos_primal_z_probe_x": cos(pz, qx),
                "cos_probe_z_primal_x": cos(qz, px),
            }
            if wu is not None:
                lex = wu[int(meta["high_id"])] - wu[int(meta["low_id"])]
                row["cos_primal_z_unembed_high_low"] = cos(pz, lex)
                row["cos_primal_x_unembed_high_low"] = cos(px, lex)
            model_out[pair] = row
            print(f"[v12-cpu] direction redteam {model_short}/{pair}")
        out["models"][model_short] = model_out

    gpu_path = RESULTS / "direction_redteam_lexical_activations.json"
    if gpu_path.exists():
        out["lexical_activation_directions"] = json.loads(gpu_path.read_text())
    steer_path = RESULTS / "direction_redteam_steering.json"
    if steer_path.exists():
        out["steering"] = json.loads(steer_path.read_text())
    (RESULTS / "direction_redteam_x_lexical_z.json").write_text(json.dumps(strict_json_ready(out), indent=2, allow_nan=False))
    return out


def plot_direction_redteam(red: dict) -> None:
    cosine_keys = [
        "cos_primal_z_primal_x",
        "cos_probe_z_probe_x",
        "cos_primal_z_unembed_high_low",
    ]
    rows = []
    labels = []
    for model_short in MODELS:
        for pair in ALL_PAIRS:
            row = red.get("models", {}).get(model_short, {}).get(pair)
            if row and all(k in row for k in cosine_keys):
                rows.append([row[k] for k in cosine_keys])
                labels.append(f"{model_short}/{pair}")
    if not rows:
        return
    M = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(labels))))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    ax.set_xticks(range(len(cosine_keys)), labels=[k.replace("cos_", "").replace("_", "\n") for k in cosine_keys], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("v12 direction red-team: cosine similarities")
    fig.colorbar(im, ax=ax, label="cosine")
    fig.tight_layout()
    fig.savefig(FIGS / "direction_redteam_cosines.png", dpi=150)
    plt.close(fig)

    lex_rows = []
    lex_labels = []
    lex_keys = [
        "cos_primal_z_lex_high_low_word",
        "cos_primal_z_lex_sentence",
        "cos_primal_z_lex_synonym",
        "cos_primal_z_lex_domain",
        "cos_primal_x_lex_high_low_word",
        "cos_primal_x_lex_sentence",
    ]
    lex_by_pair = red.get("lexical_activation_directions", {}).get("by_pair", {})
    for pair in ALL_PAIRS:
        row = lex_by_pair.get(pair, {})
        if all(k in row for k in lex_keys):
            lex_rows.append([row[k] for k in lex_keys])
            lex_labels.append(pair)
    if lex_rows:
        M = np.array(lex_rows, dtype=float)
        fig, ax = plt.subplots(figsize=(8.5, max(4, 0.35 * len(lex_labels))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(lex_labels)), labels=lex_labels, fontsize=8)
        ax.set_xticks(
            range(len(lex_keys)),
            labels=[k.replace("cos_", "").replace("_", "\n") for k in lex_keys],
            fontsize=8,
        )
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("v12 Gemma 2 9B lexical direction cosine similarities")
        fig.colorbar(im, ax=ax, label="cosine")
        fig.tight_layout()
        fig.savefig(FIGS / "direction_redteam_lexical_cosines_9b.png", dpi=150)
        plt.close(fig)


def pc_extremeness() -> dict:
    out = {"models": {}}
    for model_short in MODELS:
        late = LATE[model_short]
        model_out = {}
        for pair in ALL_PAIRS:
            d = load_npz(model_short, pair)
            if d is None:
                continue
            acts = d["activations"]
            x = d["x"].astype(np.float64)
            z = d["z"].astype(np.float64)
            cm, cx, cz = cell_mean(acts, x, z)
            X = cm[:, late, :]
            pca = PCA(n_components=3).fit(X)
            proj = pca.transform(X)
            pz = primal_direction(acts, z, late)
            px = raw_x_direction(acts, x, late)
            metrics = {}
            targets = {
                "z": cz,
                "x": cx,
                "z2": cz**2,
                "x2": cx**2,
                "abs_z": np.abs(cz),
                "abs_x": np.abs(cx),
            }
            for k in range(3):
                pc = proj[:, k]
                for name, y in targets.items():
                    metrics[f"PC{k+1}_vs_{name}"] = r2(pc, y)
                metrics[f"cos_PC{k+1}_primal_z"] = cos(pca.components_[k], pz)
                metrics[f"cos_PC{k+1}_primal_x"] = cos(pca.components_[k], px)
            metrics["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
            model_out[pair] = {"late_layer": late, "metrics": metrics}
            print(f"[v12-cpu] pc audit {model_short}/{pair}")
        out["models"][model_short] = model_out
    (RESULTS / "pc_extremeness_x_z_audit.json").write_text(json.dumps(strict_json_ready(out), indent=2, allow_nan=False))
    return out


def plot_pc_extremeness(pc: dict) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    for row_i, model_short in enumerate(MODELS):
        data = pc.get("models", {}).get(model_short, {})
        pairs = [p for p in ALL_PAIRS if p in data]
        metric_groups = [("PC1", "z"), ("PC2", "z"), ("PC2", "abs_z"), ("PC2", "x"), ("PC3", "abs_z")]
        for col_i, metric_group in enumerate(metric_groups):
            ax = axes[row_i, col_i]
            pc_name, target = metric_group
            vals = [data[p]["metrics"].get(f"{pc_name}_vs_{target}", np.nan) for p in pairs]
            ax.bar(range(len(pairs)), vals, color=f"C{col_i}")
            ax.set_xticks(range(len(pairs)), pairs, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("R2")
            ax.set_title(f"{model_short}: {pc_name} vs {target}")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("v12 PC extremeness / raw-x audit")
    fig.tight_layout()
    fig.savefig(FIGS / "pc_extremeness_x_z_grid.png", dpi=150)
    plt.close(fig)


def strict_json_ready(obj):
    if isinstance(obj, dict):
        return {k: strict_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strict_json_ready(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main() -> None:
    layer = layer_sweep_9b()
    plot_layer_sweep(layer)
    red = direction_redteam()
    plot_direction_redteam(red)
    pc = pc_extremeness()
    plot_pc_extremeness(pc)
    print("[v12-cpu] complete")


if __name__ == "__main__":
    main()
