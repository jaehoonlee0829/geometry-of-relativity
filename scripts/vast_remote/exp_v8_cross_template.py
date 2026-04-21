"""v8 Priority 4: cross-template transfer test for height.

Red-team worry: v7's cross-pair transfer (40% of own-pair) could be driven
by shared prompt template, not shared z-computation. All prompts use
"Person/Number N: X. This __ is ___". If primal_z trained on Template A
also steers Template B (different wording), the direction is semantic,
not syntactic.

Templates for height (both use Grid B x,z,μ):
  A (standard): "Person 16: 170 cm. This person is"
  B (rephrased): "Among the individuals listed, the one measuring 170 cm would be described as"

For each template:
  - Build 750 prompts (5 x × 5 z × 30 seeds)
  - Extract layer-32 activations
  - Compute primal_z = mean(acts|z>+1) − mean(acts|z<-1)

Then: steer Template B prompts with primal_z_from_A; measure slope.
Compare to primal_z_from_B self-steering and random null.

Writes:
  results/v8_cross_template/height_template_A_{acts,logits}.jsonl (+ npz)
  results/v8_cross_template/height_template_B_{acts,logits}.jsonl (+ npz)
  results/v8_cross_template/transfer_result.json
  figures/v8/cross_template_transfer.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS  # noqa: E402

MODEL_ID = "google/gemma-4-E4B"
LAYER_IDX = 32
BATCH_SIZE = 16
ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
N_RANDOM = 3
Z_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
N_SEEDS = 30

HEIGHT = next(p for p in PAIRS if p.name == "height")

OUT = REPO / "results" / "v8_cross_template"
OUT_FIG = REPO / "figures" / "v8"
OUT.mkdir(parents=True, exist_ok=True)


def sample_context(mu, seed, n=15):
    rng = np.random.default_rng(seed + hash("height-v8") % 1000)
    out = []
    while len(out) < n:
        v = rng.normal(mu, HEIGHT.sigma)
        if 60 <= v <= 300:
            out.append(round(v))
    return out


def build_trials(template: str):
    """template has placeholders: {items}, {n_last}, {x_str}"""
    trials = []
    idx = 0
    for x in HEIGHT.target_values:
        for z in Z_VALUES:
            mu = x - HEIGHT.sigma * z
            for s in range(N_SEEDS):
                sample = sample_context(mu, s)
                items_block = "\n".join(f"Person {i+1}: {v} cm" for i, v in enumerate(sample))
                prompt = template.format(items=items_block, n_last=len(sample) + 1, x_str=f"{int(x)}")
                trials.append({
                    "id": f"height_{idx:06d}",
                    "x": float(x), "mu": float(mu), "z": float(z),
                    "sigma": HEIGHT.sigma, "seed": s,
                    "prompt": prompt,
                })
                idx += 1
    return trials


TEMPLATE_A = "{items}\nPerson {n_last}: {x_str} cm. This person is"
TEMPLATE_B = "{items}\nAmong the individuals listed, the one measuring {x_str} cm would be described as"


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def extract_acts_and_ld(model, tok, trials, high_id, low_id):
    prompts = [t["prompt"] for t in trials]
    layers = get_layers(model)
    capt = []

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        capt.append(h[:, -1, :].detach())

    handle = layers[LAYER_IDX].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lds = []
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            ld = (logits[:, high_id] - logits[:, low_id]).float().cpu().numpy()
            lds.append(ld)
    finally:
        handle.remove()
    acts = torch.cat(capt, dim=0).float().cpu().numpy()
    return acts, np.concatenate(lds)


def steered_ld(model, tok, prompts, direction, alpha, high_id, low_id):
    layers = get_layers(model)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = direction.to(device=h.device, dtype=h.dtype)
        h[:, -1, :] = h[:, -1, :] + alpha * d
        if isinstance(out, tuple): return (h,) + out[1:]
        return h

    handle = layers[LAYER_IDX].register_forward_hook(hook)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lds = []
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            ld = (logits[:, high_id] - logits[:, low_id]).float().cpu().numpy()
            lds.append(ld)
    finally:
        handle.remove()
    return np.concatenate(lds)


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def main():
    print(f"Loading {MODEL_ID}…"); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # For Template B, "described as" ends the prompt differently — we still score
    # "tall"/"short" (high/low). Check tokenization is same ID.
    high_id = first_token_id(tok, HEIGHT.high_word)
    low_id = first_token_id(tok, HEIGHT.low_word)
    print(f"  tokens: {HEIGHT.high_word!r} → {high_id}   {HEIGHT.low_word!r} → {low_id}")

    # Build prompts
    trials_A = build_trials(TEMPLATE_A)
    trials_B = build_trials(TEMPLATE_B)
    print(f"  Template A: {len(trials_A)} trials")
    print(f"  Template B: {len(trials_B)} trials")

    # Extract both templates
    print("\nExtracting Template A…")
    t1 = time.time()
    acts_A, ld_A = extract_acts_and_ld(model, tok, trials_A, high_id, low_id)
    print(f"  {time.time()-t1:.1f}s  ld_mean={ld_A.mean():+.3f}")
    print("Extracting Template B…")
    t1 = time.time()
    acts_B, ld_B = extract_acts_and_ld(model, tok, trials_B, high_id, low_id)
    print(f"  {time.time()-t1:.1f}s  ld_mean={ld_B.mean():+.3f}")

    # Compute primal_z per template
    zs = np.array([t["z"] for t in trials_A])    # same zs for both
    pz_A = unit(acts_A[zs > +1].mean(0) - acts_A[zs < -1].mean(0))
    pz_B = unit(acts_B[zs > +1].mean(0) - acts_B[zs < -1].mean(0))
    print(f"\ncos(primal_z_A, primal_z_B) = {float(np.dot(pz_A, pz_B)):+.4f}")

    # Steer Template B prompts with primal_z_A, primal_z_B, and 3 random directions
    D = acts_A.shape[1]
    rng = np.random.default_rng(42)
    random_dirs = [unit(rng.standard_normal(D).astype(np.float64)) for _ in range(N_RANDOM)]

    def sweep(direction, prompts):
        curve = {}
        for alpha in ALPHAS:
            ld = steered_ld(model, tok, prompts, torch.from_numpy(direction).to(model.device), alpha, high_id, low_id)
            curve[str(alpha)] = {"ld_mean": float(ld.mean()), "ld_std": float(ld.std())}
        slope = float(np.polyfit(ALPHAS, [curve[str(a)]["ld_mean"] for a in ALPHAS], 1)[0])
        return curve, slope

    B_prompts = [t["prompt"] for t in trials_B]
    print("\nSteering Template B prompts with various directions…")
    t1 = time.time()
    curve_selfB, slope_selfB = sweep(pz_B, B_prompts)
    print(f"  self (primal_z_B → B): slope={slope_selfB:+.4f}  ({time.time()-t1:.1f}s)")
    t1 = time.time()
    curve_crossAB, slope_crossAB = sweep(pz_A, B_prompts)
    print(f"  cross (primal_z_A → B): slope={slope_crossAB:+.4f}  ({time.time()-t1:.1f}s)")
    rand_slopes = []
    for ri, v in enumerate(random_dirs):
        _, s = sweep(v, B_prompts)
        rand_slopes.append(s)
        print(f"  random #{ri}: slope={s:+.4f}")

    rand_mean_abs = float(np.mean(np.abs(rand_slopes)))
    result = {
        "cos_primal_z_A_B": float(np.dot(pz_A, pz_B)),
        "steering_on_template_B": {
            "self_primal_z_B_slope":  slope_selfB,
            "cross_primal_z_A_slope": slope_crossAB,
            "random_slopes": rand_slopes,
            "random_mean_abs": rand_mean_abs,
        },
        "transfer_ratio_cross_over_self":      abs(slope_crossAB) / (abs(slope_selfB) + 1e-12),
        "transfer_ratio_cross_over_random":    abs(slope_crossAB) / (rand_mean_abs + 1e-12),
    }
    (OUT / "transfer_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nSUMMARY:")
    print(f"  primal_z_A · primal_z_B (cos): {result['cos_primal_z_A_B']:+.4f}")
    print(f"  |slope(pz_A→B)| / |slope(pz_B→B)| = {result['transfer_ratio_cross_over_self']:.3f}")
    print(f"  |slope(pz_A→B)| / |slope(random→B)| = {result['transfer_ratio_cross_over_random']:.2f}×")

    # Also save per-template activations compactly
    np.savez(OUT / "height_template_A_late.npz",
             activations=acts_A.astype(np.float32), ids=np.array([t["id"] for t in trials_A]))
    np.savez(OUT / "height_template_B_late.npz",
             activations=acts_B.astype(np.float32), ids=np.array([t["id"] for t in trials_B]))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ALPHAS, [curve_selfB[str(a)]["ld_mean"] for a in ALPHAS], "o-", label=f"primal_z_B → B  (slope={slope_selfB:+.3f})")
    ax.plot(ALPHAS, [curve_crossAB[str(a)]["ld_mean"] for a in ALPHAS], "s-", label=f"primal_z_A → B  (slope={slope_crossAB:+.3f})")
    ax.set_xlabel("α"); ax.set_ylabel("logit_diff(tall−short)")
    ax.set_title(f"Cross-template transfer for height\ncos(primal_z_A, primal_z_B) = {result['cos_primal_z_A_B']:+.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "cross_template_transfer.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG/'cross_template_transfer.png'}")


if __name__ == "__main__":
    main()
