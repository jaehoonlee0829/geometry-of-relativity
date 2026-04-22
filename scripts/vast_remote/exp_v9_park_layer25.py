"""v9 robustness for Park's causal metric: apply at layer 25 (pre-unembedding).

Addresses critic concern SCI-C4:
  Park's causal inner product is sharpest at the layer immediately before
  unembedding, because the derivation uses W_U geometry. v9 P4 applied
  the transformation at layer 20 and then let 5 more transformer blocks
  re-mix the vector — weaker test. We repeat at layer 25 (last transformer
  block), refit probe_z on layer-25 activations, and additionally sweep
  λ ∈ {1e-5, 1e-3, 1e-1, 1, 10}.

Also reports eigenvalue diagnostics of M = W_U^T W_U + λI so we can see
whether λ=1e-2 was effectively zero (critic claim) or effectively large.

Writes
  results/v9_gemma2/park_layer25_summary.json
  results/v9_gemma2/park_layer25_rows.jsonl
  figures/v9/park_layer25_lambda_sweep.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS  # noqa: E402
from extract_v7_xz_grid import Z_VALUES  # noqa: E402
from exp_v9_on_manifold_steering import BATCH_SIZE, get_layers  # noqa: E402

MODEL_ID = "google/gemma-2-2b"
LATE25_LAYER = 25
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
LAMBDAS = [1e-5, 1e-3, 1e-1, 1.0, 10.0]
SUBSET_PER_Z = 20
RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"


def extract_layer25(model, tok, rows, batch_size=BATCH_SIZE):
    """Single forward pass; grab layer-25 output and last-token logits."""
    layers = get_layers(model)
    captured = []
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())
    handle = layers[LATE25_LAYER].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    acts_list = []
    try:
        for i in range(0, len(rows), batch_size):
            batch = [r["prompt"] for r in rows[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(model.device)
            captured.clear()
            with torch.no_grad():
                _ = model(**enc)
            h = captured[0]
            acts_list.append(h[:, -1, :].float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(acts_list, axis=0)


def run_steering_at_layer25(model, tok, rows, direction_per_row, alpha,
                             high_id, low_id, batch_size=BATCH_SIZE):
    prompts = [r["prompt"] for r in rows]
    layers = get_layers(model)
    steer_tensor = torch.tensor(direction_per_row, dtype=torch.bfloat16,
                                device=model.device)
    batch_slice = [0]

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        n = h.shape[0]
        delta = steer_tensor[batch_slice[0]:batch_slice[0] + n] * alpha
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + delta
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    handle = layers[LATE25_LAYER].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lds, ents = [], []
    try:
        for i in range(0, len(prompts), batch_size):
            batch_slice[0] = i
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model(**enc)
            last = out.logits[:, -1, :]
            logp = torch.log_softmax(last.double(), dim=-1)
            ent = -(logp.exp() * logp).sum(-1).float().cpu().numpy()
            ld = (last[:, high_id] - last[:, low_id]).float().cpu().numpy()
            lds.append(ld); ents.append(ent)
    finally:
        handle.remove()
    return np.concatenate(lds), np.concatenate(ents)


def main():
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    # W_U & eigenvalue diagnostic (shared across pairs)
    print("\nExtracting W_U and computing W_U^T W_U eigen-diagnostic…", flush=True)
    W_U = model.get_output_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    print(f"  W_U shape: {W_U.shape}")
    G = W_U.T @ W_U    # (d, d)
    eigvals = np.linalg.eigvalsh(G).astype(np.float64)  # ascending
    print(f"  eigvals(W_U^T W_U) min={eigvals[0]:.3e}  "
          f"median={np.median(eigvals):.3e}  max={eigvals[-1]:.3e}")
    diagnostics = {
        "eig_min": float(eigvals[0]),
        "eig_10pct": float(np.quantile(eigvals, 0.10)),
        "eig_50pct": float(np.quantile(eigvals, 0.50)),
        "eig_90pct": float(np.quantile(eigvals, 0.90)),
        "eig_max": float(eigvals[-1]),
        "eigvals_first_20": eigvals[:20].tolist(),
        "eigvals_last_20":  eigvals[-20:].tolist(),
        "lambdas_swept": LAMBDAS,
        "lambda_regime_per_lam": {
            str(l): {
                "frac_eig_below_lam": float((eigvals < l).mean()),
                "lambda_vs_median_ratio": l / float(np.median(eigvals)),
            }
            for l in LAMBDAS
        },
    }

    # Cache cho_factor per λ
    cho = {}
    for lam in LAMBDAS:
        M = G + lam * np.eye(G.shape[0], dtype=G.dtype)
        cho[lam] = cho_factor(M, lower=True)
    print("  Cholesky factored for all λs")

    summary = {
        "model": MODEL_ID,
        "layer": LATE25_LAYER,
        "alphas": ALPHAS,
        "lambdas": LAMBDAS,
        "eigen_diagnostics": diagnostics,
        "per_pair": {},
    }
    all_rows = []
    rng = np.random.default_rng(0)

    for pair in PAIRS:
        print(f"\n==================== {pair.name} ====================", flush=True)
        # Load trials for this pair
        trials = []
        with (RES_DIR / "gemma2_trials.jsonl").open() as f:
            for line in f:
                t = json.loads(line)
                if t["pair"] == pair.name:
                    trials.append(t)

        print(f"  extracting layer-{LATE25_LAYER} activations for n={len(trials)}…",
              flush=True)
        t1 = time.time()
        acts = extract_layer25(model, tok, trials)
        print(f"    done in {time.time() - t1:.1f}s  shape={acts.shape}", flush=True)
        zs = np.array([t["z"] for t in trials], dtype=np.float64)

        # Directions at layer 25
        hi = zs > 0
        lo = zs < 0
        primal_z = (acts[hi].mean(0) - acts[lo].mean(0)).astype(np.float32)
        rid = Ridge(alpha=1.0, fit_intercept=True).fit(acts, zs)
        probe_z = rid.coef_.astype(np.float32)
        norm_p = float(np.linalg.norm(primal_z))

        def rescale(v):
            n = np.linalg.norm(v)
            return v * (norm_p / n) if n > 1e-9 else v

        # Stratified subset
        chosen = []
        for z_val in Z_VALUES:
            idx = np.where(np.isclose(zs, z_val, atol=1e-6))[0]
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, size=min(SUBSET_PER_Z, len(idx)), replace=False)
            chosen.extend(pick.tolist())
        chosen = np.array(sorted(chosen), dtype=int)
        sub_trials = [trials[i] for i in chosen]
        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)

        pair_summary = {
            "norm_primal": norm_p,
            "norm_probe": float(np.linalg.norm(probe_z)),
            "cos_probe_primal": float(
                probe_z @ primal_z / (np.linalg.norm(probe_z) * norm_p + 1e-12)),
            "slopes": {},
            "cos_with_primal": {},
        }

        # 1) Primal baseline
        dir_arr = np.tile(primal_z, (len(sub_trials), 1)).astype(np.float32)
        ld_by_a = {}
        for alpha in ALPHAS:
            ld, ent = run_steering_at_layer25(model, tok, sub_trials, dir_arr,
                                               alpha, high_id, low_id)
            ld_by_a[alpha] = ld.mean()
            for k, r in enumerate(sub_trials):
                all_rows.append({
                    "id": r["id"], "pair": pair.name, "z": float(zs[chosen][k]),
                    "direction": "primal", "lam": None,
                    "alpha": float(alpha),
                    "logit_diff": float(ld[k]), "entropy": float(ent[k]),
                })
        xs = np.array(ALPHAS)
        ys = np.array([ld_by_a[a] for a in ALPHAS])
        primal_slope = float(np.polyfit(xs, ys, 1)[0])
        pair_summary["slopes"]["primal"] = primal_slope
        pair_summary["cos_with_primal"]["primal"] = 1.0
        print(f"  primal slope @ layer 25 = {primal_slope:+.3f}", flush=True)

        # 2) Plain probe
        dir_arr = np.tile(rescale(probe_z), (len(sub_trials), 1)).astype(np.float32)
        ld_by_a = {}
        for alpha in ALPHAS:
            ld, _ = run_steering_at_layer25(model, tok, sub_trials, dir_arr,
                                             alpha, high_id, low_id)
            ld_by_a[alpha] = ld.mean()
            for k, r in enumerate(sub_trials):
                all_rows.append({
                    "id": r["id"], "pair": pair.name, "z": float(zs[chosen][k]),
                    "direction": "probe", "lam": None,
                    "alpha": float(alpha),
                    "logit_diff": float(ld[k]), "entropy": 0.0,
                })
        ys = np.array([ld_by_a[a] for a in ALPHAS])
        probe_slope = float(np.polyfit(xs, ys, 1)[0])
        pair_summary["slopes"]["probe"] = probe_slope
        print(f"  probe slope @ layer 25 = {probe_slope:+.3f}", flush=True)

        # 3) Park causal at each λ
        for lam in LAMBDAS:
            probe_causal = cho_solve(cho[lam], probe_z).astype(np.float32)
            dir_arr = np.tile(rescale(probe_causal),
                              (len(sub_trials), 1)).astype(np.float32)
            cos_cp = float(probe_causal @ primal_z /
                           (np.linalg.norm(probe_causal) * norm_p + 1e-12))
            ld_by_a = {}
            for alpha in ALPHAS:
                ld, _ = run_steering_at_layer25(model, tok, sub_trials, dir_arr,
                                                 alpha, high_id, low_id)
                ld_by_a[alpha] = ld.mean()
                for k, r in enumerate(sub_trials):
                    all_rows.append({
                        "id": r["id"], "pair": pair.name, "z": float(zs[chosen][k]),
                        "direction": "probe_causal", "lam": lam,
                        "alpha": float(alpha),
                        "logit_diff": float(ld[k]), "entropy": 0.0,
                    })
            ys = np.array([ld_by_a[a] for a in ALPHAS])
            slope = float(np.polyfit(xs, ys, 1)[0])
            pair_summary["slopes"][f"probe_causal_lam{lam:g}"] = slope
            pair_summary["cos_with_primal"][f"probe_causal_lam{lam:g}"] = cos_cp
            print(f"  probe_causal (λ={lam:g}): slope={slope:+.3f}  "
                  f"cos_with_primal={cos_cp:+.3f}", flush=True)

        summary["per_pair"][pair.name] = pair_summary

    (RES_DIR / "park_layer25_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_rows) + "\n"
    )
    (RES_DIR / "park_layer25_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {RES_DIR}/park_layer25_summary.json")


if __name__ == "__main__":
    main()
