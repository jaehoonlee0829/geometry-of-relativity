"""Exp 1: zero-shot expansion — 5x × 30 phrasing seeds × 8 pairs = 1200 prompts.

Uses the existing `format_prompt_zero` template from extract_v4_adjpairs.PAIRS
as the core phrasing, and prepends 30 different lead-ins for seed variation.
This lets us measure a zero-shot direction w_x_zeroshot and compare it to the
implicit-context direction w_z_implicit (PC1).

Writes:
  results/v4_zeroshot_expanded/e4b_{pair}_{layer}.npz   (1 per pair × 2 layers)
  results/v4_zeroshot_expanded/e4b_{pair}_logits.jsonl
  results/v4_zeroshot_expanded/e4b_trials.jsonl
  results/v4_adjpairs_analysis/zero_shot_expansion.json   (PCA + probe + cosines)
  figures/v4_adjpairs/zero_shot_vs_implicit_directions.png
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id  # noqa: E402
from extract_v4_adjpairs import PAIRS, make_zero_shot_prompt  # noqa: E402

MODEL_ID = "google/gemma-4-E4B"
LAYER_INDICES = {"mid": 21, "late": 32}
BATCH_SIZE = 16

OUT = REPO / "results" / "v4_zeroshot_expanded"
OUT.mkdir(parents=True, exist_ok=True)
OUT_ANALYSIS = REPO / "results" / "v4_adjpairs_analysis"
OUT_FIG = REPO / "figures" / "v4_adjpairs"

# 30 lead-ins to vary phrasing while preserving the final "... is" token position.
LEAD_INS = [
    "", "Suppose: ", "Consider: ", "Imagine: ", "You see: ",
    "You meet someone: ", "You encounter: ", "Observe: ", "Take an example: ",
    "Picture this: ", "Here we have: ", "In this case: ", "For context: ",
    "Scenario: ", "Fact: ", "Given: ", "Hypothetical: ", "Let's say: ",
    "For example: ", "As an example: ", "Notice this: ", "Example: ",
    "Description: ", "Case: ", "Detail: ", "Statement: ",
    "Here is the information: ", "Here's what we know: ",
    "Now consider: ", "Let me describe: ",
]
assert len(LEAD_INS) == 30


def build_trials() -> list[dict]:
    trials = []
    idx = 0
    for pair in PAIRS:
        for x in pair.target_values:
            base = make_zero_shot_prompt(pair, x)
            for seed, lead in enumerate(LEAD_INS):
                prompt = lead + base
                trials.append({
                    "id": f"{pair.name}_zs_{idx:06d}",
                    "pair": pair.name,
                    "x": float(x),
                    "seed": seed,
                    "prompt": prompt,
                    "low_word": pair.low_word,
                    "high_word": pair.high_word,
                })
                idx += 1
    return trials


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError


def extract_and_score(model, tok, trials_for_pair: list[dict], high_id: int, low_id: int):
    prompts = [t["prompt"] for t in trials_for_pair]
    layers = get_layers(model)
    captured = {k: [] for k in LAYER_INDICES}
    handles = []
    for k, idx in LAYER_INDICES.items():
        def make_hook(kk):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[kk].append(h.detach())
            return hook
        handles.append(layers[idx].register_forward_hook(make_hook(k)))
    per_layer_acts = {k: [] for k in LAYER_INDICES}
    logit_diffs = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            for k in captured: captured[k].clear()
            with torch.no_grad():
                out = model(**enc)
            logits = out.logits[:, -1, :]
            ld = (logits[:, high_id] - logits[:, low_id]).float().cpu().numpy()
            logit_diffs.append(ld)
            for k in LAYER_INDICES:
                h = captured[k][0]
                per_layer_acts[k].append(h[:, -1, :].float().cpu().numpy())
    finally:
        for h in handles: h.remove()
    return ({k: np.concatenate(v, axis=0) for k, v in per_layer_acts.items()},
            np.concatenate(logit_diffs))


def main():
    trials = build_trials()
    by_pair = defaultdict(list)
    for t in trials:
        by_pair[t["pair"]].append(t)
    print(f"{len(trials)} prompts across {len(by_pair)} pairs", flush=True)

    print(f"\nLoading {MODEL_ID}…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    all_trials_out = []
    analysis_result: dict = {}
    for pair_obj in PAIRS:
        pair = pair_obj.name
        print(f"\n=== {pair} ===", flush=True)
        t_list = by_pair[pair]
        high_id = first_token_id(tok, pair_obj.high_word)
        low_id = first_token_id(tok, pair_obj.low_word)
        t0 = time.time()
        acts, ld = extract_and_score(model, tok, t_list, high_id, low_id)
        print(f"  {len(t_list)} prompts, {time.time()-t0:.1f}s", flush=True)
        # Save activations per layer
        for layer, a in acts.items():
            np.savez(OUT / f"e4b_{pair}_{layer}.npz",
                     activations=a.astype(np.float32),
                     ids=np.array([t["id"] for t in t_list]),
                     layer_index=LAYER_INDICES[layer],
                     layer_name=layer)
        # Save logits + trials
        with (OUT / f"e4b_{pair}_logits.jsonl").open("w") as f:
            for t, l in zip(t_list, ld):
                f.write(json.dumps({"id": t["id"], "logit_diff": float(l)}) + "\n")
        all_trials_out.extend(t_list)

        # Analysis: does zero-shot PC1 track x? Compare direction to implicit w_z.
        xs = np.array([t["x"] for t in t_list], dtype=np.float64)
        for layer in LAYER_INDICES:
            a = acts[layer].astype(np.float64)
            # Zero-shot probe for x
            w_x_zs = Ridge(alpha=1.0).fit(a, xs).coef_.astype(np.float64)
            cv_r2_x = _cv_r2(a, xs, seed=0)
            # Zero-shot PCA PC1
            pc1 = PCA(n_components=1).fit(a - a.mean(0)).components_[0].astype(np.float64)
            # Does PC1 track x?
            proj = (a - a.mean(0)) @ pc1
            r2_pc1_x = float(np.corrcoef(proj, xs)[0, 1] ** 2)
            # Compare to implicit direction (load cached implicit activations + compute w_z)
            imp_npz = np.load(REPO / "results" / "v4_adjpairs" / f"e4b_{pair}_implicit_{layer}.npz",
                              allow_pickle=True)
            imp_acts = imp_npz["activations"].astype(np.float64)
            trials_by_id = {json.loads(l)["id"]: json.loads(l) for l in (REPO / "results" / "v4_adjpairs" / "e4b_trials.jsonl").open()}
            imp_zs = np.array([trials_by_id[str(i)]["z"] for i in imp_npz["ids"]])
            w_z_imp = Ridge(alpha=1.0).fit(imp_acts, imp_zs).coef_.astype(np.float64)
            cos = float(np.dot(w_x_zs, w_z_imp) /
                        (np.linalg.norm(w_x_zs) * np.linalg.norm(w_z_imp) + 1e-12))
            cos_pc1 = float(np.dot(pc1, w_z_imp) /
                            (np.linalg.norm(pc1) * np.linalg.norm(w_z_imp) + 1e-12))
            print(f"  [{layer}] R²(zeroshot PC1 ~ x)={r2_pc1_x:.3f}  "
                  f"cv_R²(w_x_zs → x)={cv_r2_x:.3f}  "
                  f"cos(w_x_zs, w_z_imp)={cos:+.3f}  "
                  f"cos(zs_PC1, w_z_imp)={cos_pc1:+.3f}", flush=True)
            analysis_result.setdefault(pair, {})[layer] = {
                "r2_pc1_vs_x": r2_pc1_x,
                "cv_r2_wx_zeroshot": float(cv_r2_x),
                "cos_wx_zs_wz_imp": cos,
                "cos_pc1_zs_wz_imp": cos_pc1,
                "n": int(a.shape[0]),
            }

    with (OUT / "e4b_trials.jsonl").open("w") as f:
        for t in all_trials_out:
            f.write(json.dumps(t) + "\n")
    (OUT_ANALYSIS / "zero_shot_expansion.json").write_text(json.dumps(analysis_result, indent=2))

    # Figure: |cos(zs direction, implicit w_z)| per pair (bars)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, layer in zip(axes, LAYER_INDICES):
        names = list(analysis_result.keys())
        cos_wx = [abs(analysis_result[p][layer]["cos_wx_zs_wz_imp"]) for p in names]
        cos_pc1 = [abs(analysis_result[p][layer]["cos_pc1_zs_wz_imp"]) for p in names]
        r2 = [analysis_result[p][layer]["r2_pc1_vs_x"] for p in names]
        xpos = np.arange(len(names))
        ax.bar(xpos - 0.25, cos_wx, 0.25, label="|cos(w_x_zs, w_z_imp)|")
        ax.bar(xpos,        cos_pc1, 0.25, label="|cos(ZS_PC1, w_z_imp)|")
        ax.bar(xpos + 0.25, r2,      0.25, label="R²(ZS_PC1 ~ x)")
        ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"layer={layer}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Zero-shot direction vs implicit w_z — does ZS PC1 track x, not z?")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "zero_shot_vs_implicit_directions.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {OUT_FIG/'zero_shot_vs_implicit_directions.png'}")


def _cv_r2(X, y, seed=0):
    from sklearn.model_selection import KFold
    cv = KFold(5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X):
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        yp = m.predict(X[te])
        ss_res = ((y[te] - yp) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(scores))


if __name__ == "__main__":
    main()
