"""Gemma 4 31B activation extractor (v1). Same schema as e4b v3: W_U saved once, per-layer activations-only npz.

Layer budget (60 total layers):
  early=14, mid=30, late=45, final=59
"""
import json, time, os, gc
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = Path("data_gen/prompts_v2.jsonl")
OUT = Path("results/activations")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "google/gemma-4-31B"
SHORT = "g31b"
LAYER_INDICES = {"early": 14, "mid": 30, "late": 45, "final": 59}

def get_text_cfg(cfg):
    return getattr(cfg, "text_config", cfg)

def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError(f"no layers on {type(model).__name__}")

def get_unembedding(model):
    if hasattr(model, "lm_head") and model.lm_head.weight is not None:
        return model.lm_head.weight.detach().float().cpu().numpy()
    return model.get_input_embeddings().weight.detach().float().cpu().numpy()

def extract_one_domain(model, tok, prompts, ids, domain, layer_indices, batch_size=4):
    layers_mod = get_layers(model)
    n_total = len(prompts)
    print(f"  {domain}: {n_total} prompts, {len(layers_mod)} total layers", flush=True)
    captured = {k: [] for k in layer_indices}
    handles = []
    def make_hook(key):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[key].append(h.detach())
        return hook
    for key, idx in layer_indices.items():
        handles.append(layers_mod[idx].register_forward_hook(make_hook(key)))
    per_layer_acts = {k: [] for k in layer_indices}
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        for i in range(0, n_total, batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            for k in captured: captured[k].clear()
            with torch.no_grad():
                _ = model(**inputs, use_cache=False)
            for k in layer_indices:
                h = captured[k][0]
                last = h[:, -1, :]
                per_layer_acts[k].append(last.float().cpu().numpy())
            if (i // batch_size) % 8 == 0:
                print(f"    {i+len(batch)}/{n_total}", flush=True)
    finally:
        for h in handles: h.remove()
    return {k: np.concatenate(v, axis=0) for k,v in per_layer_acts.items()}

def main():
    trials = [json.loads(l) for l in PROMPTS.open()]
    print(f"Loaded {len(trials)} trials", flush=True)
    by_dom = {}
    for t in trials:
        by_dom.setdefault(t["domain"], []).append(t)
    print({d: len(ts) for d,ts in by_dom.items()}, flush=True)
    print(f"\nLoading {MODEL_ID}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s; dtype={model.dtype}", flush=True)
    text_cfg = get_text_cfg(model.config)
    print(f"  hidden={getattr(text_cfg,'hidden_size',None)} layers={getattr(text_cfg,'num_hidden_layers',None)}", flush=True)

    W_U = get_unembedding(model).astype(np.float32)
    wu_path = OUT / f"{SHORT}_W_U.npy"
    t0 = time.time()
    np.save(wu_path, W_U)
    print(f"  wrote {wu_path}: {W_U.shape} in {time.time()-t0:.1f}s", flush=True)

    for domain, ts in by_dom.items():
        prompts = [t["prompt"] for t in ts]
        ids = [t["id"] for t in ts]
        print(f"\n=== {domain} ({len(prompts)}) ===", flush=True)
        t0 = time.time()
        per_layer_acts = extract_one_domain(model, tok, prompts, ids, domain, LAYER_INDICES, batch_size=4)
        dt = time.time() - t0
        print(f"  fwd pass done in {dt:.1f}s ({len(prompts)/dt:.2f} p/s)", flush=True)
        for layer_key, acts in per_layer_acts.items():
            out_path = OUT / f"{SHORT}_{domain}_{layer_key}.npz"
            t1 = time.time()
            np.savez(
                out_path,
                activations=acts.astype(np.float32),
                ids=np.array(ids),
                layer_index=LAYER_INDICES[layer_key],
                layer_name=layer_key,
                model_id=MODEL_ID,
            )
            print(f"  wrote {out_path}: {acts.shape} in {time.time()-t1:.2f}s", flush=True)
    print("\nDONE G31B.", flush=True)

if __name__ == "__main__":
    main()
