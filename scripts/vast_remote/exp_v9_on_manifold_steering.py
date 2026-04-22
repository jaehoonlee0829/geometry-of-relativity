"""v9 Priority 3: On-manifold (tangent) vs off-manifold (primal_z) steering.

Hypothesis (from v9 plan):
    Adding α·primal_z to a late-layer activation pushes the hidden state
    off the curved z-trajectory, harming distributional sensibility
    (entropy). A z-conditional tangent — the local direction along the
    per-pair z-trajectory through cell-means — stays on-manifold and
    should produce the same logit_diff slope with less entropy damage.

Design
    Layer: 20 (same as extraction / SAE).
    Directions:
      primal_z   = mean_{z>0}(h) - mean_{z<0}(h)              (global, fixed)
      tangent(z) = piecewise-linear finite-difference tangent along the
                   z-cell-means  (5 knots at Z_VALUES = [-2,-1,0,1,2])
      random     = Gaussian-random unit direction (null control)
    All directions are rescaled to ||primal_z|| so α is comparable.

    Alphas: [-2, -1, 0, +1, +2].

    Steering is applied only to the LAST token position at layer-20 output
    (a forward hook). Subsequent layers 21-25 + norm + lm_head propagate
    the intervention causally to the next-token logits.

Outputs
    results/v9_gemma2/steering_manifold_rows.jsonl
    results/v9_gemma2/steering_manifold_summary.json
    figures/v9/steering_manifold_slopes.png
    figures/v9/steering_manifold_entropy.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS  # noqa: E402
from extract_v7_xz_grid import Z_VALUES  # noqa: E402

MODEL_ID = "google/gemma-2-2b"
LATE_LAYER = 20
BATCH_SIZE = 8
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
RNG = np.random.default_rng(0)

RES_DIR = REPO / "results" / "v9_gemma2"
FIG_DIR = REPO / "figures" / "v9"


def load_trials_with_z(pair_name: str):
    """Return rows [(trial_dict, idx_in_activations)] for a pair."""
    trials = {}
    with (RES_DIR / "gemma2_trials.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["pair"] == pair_name:
                trials[t["id"]] = t
    with np.load(RES_DIR / f"gemma2_{pair_name}_late.npz", allow_pickle=True) as z_:
        ids = z_["ids"].tolist()
        acts = z_["activations"]
    rows = [trials[i] for i in ids]
    return rows, acts


def build_directions(pair_name: str):
    """Compute primal_z, tangent(z), and a random control for one pair.

    tangent(z) is a (5, d)-tensor of finite-difference tangents at the
    five Z_VALUES (−2, −1, 0, +1, +2). For a prompt at z, the local tangent
    is tangent[z-bin], where the bin is 0..3 (segment between consecutive
    Z_VALUES) or 4 (fallback for out-of-range, shouldn't happen on Grid B).
    """
    rows, acts = load_trials_with_z(pair_name)
    zs = np.array([r["z"] for r in rows])
    # Cell-means per z-value
    cell_means = []
    for z_val in Z_VALUES:
        mask = np.isclose(zs, z_val, atol=1e-6)
        if mask.any():
            cell_means.append(acts[mask].mean(axis=0))
        else:
            cell_means.append(np.zeros(acts.shape[1], dtype=acts.dtype))
    cell_means = np.stack(cell_means)  # (5, d)

    hi = zs > 0
    lo = zs < 0
    primal_z = (acts[hi].mean(0) - acts[lo].mean(0)).astype(np.float32)
    norm_p = np.linalg.norm(primal_z)
    # Finite-difference tangents between consecutive z-cell-means.
    # For z in [z_k, z_{k+1}) use tangent[k] = cell_means[k+1]-cell_means[k].
    # For z == z_{K-1} use tangent[K-2].
    tangents = np.diff(cell_means, axis=0).astype(np.float32)  # (4, d)
    # Normalize all tangents to ||primal_z|| so the α-scale is comparable.
    for k in range(tangents.shape[0]):
        n = np.linalg.norm(tangents[k])
        if n > 1e-9:
            tangents[k] = tangents[k] * (norm_p / n)
    # Random direction (null control)
    rand_dir = RNG.standard_normal(size=acts.shape[1]).astype(np.float32)
    rand_dir = rand_dir * (norm_p / (np.linalg.norm(rand_dir) + 1e-9))

    return {
        "rows": rows,
        "acts": acts,
        "zs": zs,
        "primal_z": primal_z,
        "tangents": tangents,
        "random": rand_dir,
        "norm_primal": float(norm_p),
    }


def z_to_bin(z: float) -> int:
    """Which tangent (0..3) applies at z. Z_VALUES = [-2,-1,0,1,2]."""
    for k in range(len(Z_VALUES) - 1):
        if Z_VALUES[k] - 1e-6 <= z < Z_VALUES[k + 1] - 1e-6:
            return k
    return len(Z_VALUES) - 2  # z >= 2 → last segment


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def run_steering(model, tok, rows, direction_per_row, alpha: float,
                 high_id: int, low_id: int):
    """Forward each prompt with last-position steering; return ld, ent arrays."""
    prompts = [r["prompt"] for r in rows]
    layers = get_layers(model)
    steer_tensor = torch.tensor(direction_per_row, dtype=torch.bfloat16,
                                device=model.device)  # (n, d)

    # Batch-level state for the hook:
    batch_slice = [0]  # current batch start index into steer_tensor

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        n = h.shape[0]
        delta = steer_tensor[batch_slice[0]:batch_slice[0] + n] * alpha
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + delta
        if isinstance(out, tuple):
            return (h2,) + tuple(out[1:])
        return h2

    handle = layers[LATE_LAYER].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lds, ents = [], []
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_slice[0] = i
            batch = prompts[i:i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model(**enc)
            last = out.logits[:, -1, :]
            logprobs = torch.log_softmax(last.double(), dim=-1)
            ent = -(logprobs.exp() * logprobs).sum(-1).float().cpu().numpy()
            ld = (last[:, high_id] - last[:, low_id]).float().cpu().numpy()
            lds.append(ld)
            ents.append(ent)
    finally:
        handle.remove()

    return np.concatenate(lds), np.concatenate(ents)


def main():
    # Use 100 stratified prompts per pair (20 per z-level) to keep runtime sane.
    SUBSET_PER_Z = 20
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    out_rows = []
    summary = {
        "model": MODEL_ID,
        "layer": LATE_LAYER,
        "alphas": ALPHAS,
        "subset_per_z": SUBSET_PER_Z,
        "per_pair": {},
    }
    for pair in PAIRS:
        print(f"\n=== {pair.name} ===", flush=True)
        dirs = build_directions(pair.name)
        zs = dirs["zs"]
        all_rows = dirs["rows"]

        # Stratified subset: up to SUBSET_PER_Z per z-value
        chosen = []
        for z_val in Z_VALUES:
            idx = np.where(np.isclose(zs, z_val, atol=1e-6))[0]
            if len(idx) == 0:
                continue
            pick = RNG.choice(idx, size=min(SUBSET_PER_Z, len(idx)), replace=False)
            chosen.extend(pick.tolist())
        chosen = np.array(sorted(chosen), dtype=int)
        sub_rows = [all_rows[i] for i in chosen]
        sub_zs = zs[chosen]
        print(f"  subset n={len(sub_rows)}  norm(primal_z)={dirs['norm_primal']:.2f}")

        high_id = first_token_id(tok, pair.high_word)
        low_id = first_token_id(tok, pair.low_word)

        # Build per-row direction arrays.
        primal_per_row = np.tile(dirs["primal_z"], (len(sub_rows), 1)).astype(np.float32)
        tangent_per_row = np.array(
            [dirs["tangents"][z_to_bin(r["z"])] for r in sub_rows],
            dtype=np.float32,
        )
        random_per_row = np.tile(dirs["random"], (len(sub_rows), 1)).astype(np.float32)

        # Also include α=0 baseline once (same regardless of direction).
        for dir_name, dir_arr in [
            ("primal", primal_per_row),
            ("tangent", tangent_per_row),
            ("random", random_per_row),
        ]:
            for alpha in ALPHAS:
                t1 = time.time()
                ld, ent = run_steering(model, tok, sub_rows, dir_arr, alpha,
                                       high_id, low_id)
                print(f"  {dir_name:7s} α={alpha:+.1f}  "
                      f"ld_mean={ld.mean():+.3f}  ent_mean={ent.mean():.3f}  "
                      f"({time.time() - t1:.1f}s)", flush=True)
                for k, r in enumerate(sub_rows):
                    out_rows.append({
                        "id": r["id"], "pair": pair.name, "z": float(sub_zs[k]),
                        "direction": dir_name, "alpha": float(alpha),
                        "logit_diff": float(ld[k]), "entropy": float(ent[k]),
                    })

        # Per-pair summary: slope of logit_diff vs alpha, mean entropy at α=0 vs |α|=2
        def slope_vs_alpha(rows_all, name):
            sub = [r for r in rows_all if r["direction"] == name]
            if not sub:
                return {"slope": 0.0, "entropy_shift_2": 0.0}
            by_alpha = {a: [] for a in ALPHAS}
            ent_by_alpha = {a: [] for a in ALPHAS}
            for r in sub:
                by_alpha[r["alpha"]].append(r["logit_diff"])
                ent_by_alpha[r["alpha"]].append(r["entropy"])
            xs = np.array(ALPHAS)
            ys = np.array([np.mean(by_alpha[a]) for a in ALPHAS])
            slope = float(np.polyfit(xs, ys, 1)[0])
            ent_0 = float(np.mean(ent_by_alpha[0.0]))
            ent_abs2 = float(0.5 * (np.mean(ent_by_alpha[-2.0]) +
                                    np.mean(ent_by_alpha[ 2.0])))
            return {"slope": slope, "entropy_at_0": ent_0,
                    "entropy_at_abs2": ent_abs2,
                    "entropy_shift_2": ent_abs2 - ent_0}

        pair_rows = [r for r in out_rows if r["pair"] == pair.name]
        summary["per_pair"][pair.name] = {
            "primal":  slope_vs_alpha(pair_rows, "primal"),
            "tangent": slope_vs_alpha(pair_rows, "tangent"),
            "random":  slope_vs_alpha(pair_rows, "random"),
        }
        s = summary["per_pair"][pair.name]
        print(f"  slopes:  primal={s['primal']['slope']:+.3f}  "
              f"tangent={s['tangent']['slope']:+.3f}  "
              f"random={s['random']['slope']:+.3f}")
        print(f"  Δentropy at |α|=2:  primal={s['primal']['entropy_shift_2']:+.3f}  "
              f"tangent={s['tangent']['entropy_shift_2']:+.3f}  "
              f"random={s['random']['entropy_shift_2']:+.3f}")

    with (RES_DIR / "steering_manifold_rows.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    (RES_DIR / "steering_manifold_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote rows + summary to {RES_DIR}/steering_manifold_*.")


if __name__ == "__main__":
    main()
