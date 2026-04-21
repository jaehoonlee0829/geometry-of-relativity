"""On-demand W_U exporter for Gemma 4 E4B or 31B.

Only run this if you need the full unembedding matrix for Fisher-pullback
analysis (src/fisher.py). The extract_*_v*.py scripts no longer cache W_U
because it's just `model.lm_head.weight` — trivially rebuildable but ~2.7 GB
(E4B) / ~5.6 GB (31B) of disk and multiple GB of W&B quota per copy.

Usage:
    python scripts/vast_remote/export_W_U.py e4b
    python scripts/vast_remote/export_W_U.py g31b
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_IDS = {"e4b": "google/gemma-4-E4B", "g31b": "google/gemma-4-31B"}


def get_unembedding(model) -> np.ndarray:
    if hasattr(model, "lm_head") and model.lm_head.weight is not None:
        return model.lm_head.weight.detach().float().cpu().numpy()
    return model.get_input_embeddings().weight.detach().float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("short", choices=sorted(MODEL_IDS), help="model short name")
    ap.add_argument("--out-dir", default="results/activations", type=Path)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.short}_W_U.npy"

    model_id = MODEL_IDS[args.short]
    print(f"Loading {model_id} (bf16, device_map=auto)…", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    np_dtype = {"float32": np.float32, "float16": np.float16}[args.dtype]
    W_U = get_unembedding(model).astype(np_dtype)

    t0 = time.time()
    np.save(out, W_U)
    print(f"wrote {out}: shape={W_U.shape} dtype={W_U.dtype} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
