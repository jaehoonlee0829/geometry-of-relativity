"""v10 P1 — single-pass extraction for the dense height grid.

Captures, in one GPU pass over 4,000 prompts:

  Residuals
    * last-token hidden state at every one of the 26 decoder blocks → (N, 26, 2304) fp16

  Attention pieces at 8 strategic layers {0, 3, 7, 10, 13, 17, 20, 25}
    * attn_last_row[N, 8L, 8H, MAX_SEQ] — softmax-attention from the prediction
      position to every key position, eager attention so this is real (not None
      under SDPA).
    * head_outputs[N, 8L, 8H, head_dim=256] — the per-head value mixed with
      attention BEFORE o_proj. Allows DLA: contribution_to_residual = W_O[:, h*hd:(h+1)*hd] @ head_outputs[h].

  Logits at last position (full vocab argmax + tall/short slice)
    * next_logits_lowhigh[N, 2] = [logit(short), logit(tall)]  fp32
    * next_logit_diff[N] = logit(tall) - logit(short)
    * next_entropy[N]   = entropy of softmax(logits) in nats

  Token-position bookkeeping (for P4 attention analysis)
    * seq_len_unpadded[N], seq_len_padded (single int)
    * target_pos_unpadded[N, 2]    : (start, end) tokens for "Person 16: VAL cm"
    * context_pos_unpadded[N, 15, 2]
    These are recorded against the unpadded prompt; left-padding offset is
    seq_len_padded - seq_len_unpadded[i], so padded position = pad_off + abs.

The script is replicable on any GPU box: every numerical claim later made on
top of these dumps comes from CPU-only analysis files (no further GPU calls).
The wandb integration is OPTIONAL and gated on $WANDB_API_KEY.

Run:
    set -a; source .env; set +a
    .venv/bin/python scripts/vast_remote/extract_v10_dense_height.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from _token_utils import first_token_id, report_tokenization  # noqa: E402

MODEL_ID = "google/gemma-2-2b"
TRIALS_PATH = REPO / "data_gen" / "v10_dense_height_trials.jsonl"
OUT_DIR = REPO / "results" / "v10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIC_LAYERS = [0, 3, 7, 10, 13, 17, 20, 25]
BATCH_SIZE = 16  # eager attention; H100 80GB easily fits seq=192 × bs=16
MAX_SEQ = 192   # actual max is 173; padding to 192 gives slack for safety
N_CONTEXT = 15  # 15 "Person N: VAL" context lines per prompt


# ---------- token-position bookkeeping -----------------------------------------

def find_value_char_ranges(prompt: str) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Return (context_char_ranges_15, target_char_range) in absolute chars.

    Each range is (start, end) referring to the digits of the height value
    (e.g. "184" inside "Person 1: 184 cm").
    """
    ctx: list[tuple[int, int]] = []
    tgt: tuple[int, int] | None = None
    char_pos = 0
    for line in prompt.split("\n"):
        if line.startswith("Person ") and ":" in line and " cm" in line:
            colon = line.index(":")
            cm = line.index(" cm")
            v_start = char_pos + colon + 2  # skip ": "
            v_end = char_pos + cm
            if len(ctx) < N_CONTEXT:
                ctx.append((v_start, v_end))
            else:
                tgt = (v_start, v_end)
        char_pos += len(line) + 1  # +1 for newline
    assert len(ctx) == N_CONTEXT, f"expected 15 context lines, got {len(ctx)}"
    assert tgt is not None, "target line not found"
    return ctx, tgt


def char_to_token_range(offsets: list[tuple[int, int]],
                        char_range: tuple[int, int]) -> tuple[int, int]:
    """Return (tok_start, tok_end) — first/last+1 token covering [c0, c1)."""
    c0, c1 = char_range
    tok_start = None
    tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:  # special token
            continue
        if e > c0 and s < c1:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
    assert tok_start is not None and tok_end is not None, \
        f"no tokens cover {char_range}"
    return tok_start, tok_end


# ---------- model helpers ------------------------------------------------------

def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("could not locate decoder layers")


# ---------- main extraction ----------------------------------------------------

def run() -> None:
    t_start = time.time()

    # --- load trials
    trials = [json.loads(l) for l in TRIALS_PATH.open()]
    n = len(trials)
    print(f"[v10] loaded {n} trials from {TRIALS_PATH.name}", flush=True)

    # --- model + tokenizer
    print(f"[v10] loading {MODEL_ID} (eager attn)...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",   # we need attentions
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    print(f"[v10]   loaded in {time.time() - t0:.1f}s", flush=True)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    d_model = cfg.hidden_size
    vocab_size = cfg.vocab_size
    print(f"[v10]   layers={n_layers} heads={n_heads} head_dim={head_dim} "
          f"d_model={d_model} vocab={vocab_size}", flush=True)
    assert n_layers == 26 and n_heads == 8

    decoder_layers = get_decoder_layers(model)

    # --- output token ids (for logit_diff)
    print("[v10] tokenizer check:", flush=True)
    report_tokenization(tok, ["tall", "short"])
    tall_id = first_token_id(tok, "tall")
    short_id = first_token_id(tok, "short")
    print(f"[v10]   tall_id={tall_id} short_id={short_id}", flush=True)

    # --- pre-tokenize each prompt to get position bookkeeping
    print("[v10] pre-tokenizing for position bookkeeping...", flush=True)
    seq_len_unpad = np.zeros(n, dtype=np.int32)
    target_pos_unpad = np.zeros((n, 2), dtype=np.int32)
    context_pos_unpad = np.zeros((n, N_CONTEXT, 2), dtype=np.int32)
    for i, t in enumerate(trials):
        enc = tok(t["prompt"], return_offsets_mapping=True, add_special_tokens=True)
        seq_len_unpad[i] = len(enc.input_ids)
        ctx_ranges, tgt_range = find_value_char_ranges(t["prompt"])
        offsets = enc.offset_mapping
        target_pos_unpad[i] = char_to_token_range(offsets, tgt_range)
        for k, cr in enumerate(ctx_ranges):
            context_pos_unpad[i, k] = char_to_token_range(offsets, cr)
    print(f"[v10]   seq_len min/max: {seq_len_unpad.min()}/{seq_len_unpad.max()}",
          flush=True)
    assert seq_len_unpad.max() <= MAX_SEQ, \
        f"MAX_SEQ={MAX_SEQ} too small for actual max {seq_len_unpad.max()}"

    # --- output buffers (CPU, fp16/fp32)
    residuals = np.zeros((n, n_layers, d_model), dtype=np.float16)
    attn_last = np.zeros((n, len(STRATEGIC_LAYERS), n_heads, MAX_SEQ),
                         dtype=np.float16)
    head_outs = np.zeros((n, len(STRATEGIC_LAYERS), n_heads, head_dim),
                         dtype=np.float16)
    next_logits_lowhigh = np.zeros((n, 2), dtype=np.float32)
    next_logit_diff = np.zeros(n, dtype=np.float32)
    next_entropy = np.zeros(n, dtype=np.float32)
    next_argmax = np.zeros(n, dtype=np.int32)

    # --- hooks: capture last-token residual at every layer
    captured_resid_buf: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    # --- hooks: capture per-head value mix (input to o_proj) at strategic layers
    captured_head_buf: dict[int, list[np.ndarray]] = {L: [] for L in STRATEGIC_LAYERS}

    def make_layer_hook(layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured_resid_buf[layer_idx].append(
                h[:, -1, :].detach().float().cpu().numpy().astype(np.float16)
            )
        return hook

    def make_o_proj_pre_hook(layer_idx: int):
        # forward_pre_hook signature: (module, args) where args is a tuple
        def pre_hook(module, args):
            x = args[0]  # (bsz, q_len, n_heads*head_dim)
            last = x[:, -1, :].detach().float().cpu().numpy()
            captured_head_buf[layer_idx].append(
                last.reshape(last.shape[0], n_heads, head_dim).astype(np.float16)
            )
        return pre_hook

    handles = []
    for L_idx in range(n_layers):
        handles.append(
            decoder_layers[L_idx].register_forward_hook(make_layer_hook(L_idx))
        )
    for L_idx in STRATEGIC_LAYERS:
        o_proj = decoder_layers[L_idx].self_attn.o_proj
        handles.append(o_proj.register_forward_pre_hook(make_o_proj_pre_hook(L_idx)))

    # --- batched forward pass
    print(f"[v10] forward pass: {n} prompts, batch {BATCH_SIZE}, "
          f"eager attn, MAX_SEQ={MAX_SEQ}...", flush=True)
    t1 = time.time()
    try:
        for b0 in range(0, n, BATCH_SIZE):
            b_trials = trials[b0:b0 + BATCH_SIZE]
            prompts = [t["prompt"] for t in b_trials]
            enc = tok(prompts, return_tensors="pt",
                      padding="max_length", max_length=MAX_SEQ,
                      truncation=True).to(model.device)

            with torch.no_grad():
                out = model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    output_attentions=True,
                    use_cache=False,
                )

            # --- attention from last token, at strategic layers
            for k, L_idx in enumerate(STRATEGIC_LAYERS):
                attn = out.attentions[L_idx]   # (bsz, n_heads, seq, seq)
                attn_last_row = attn[:, :, -1, :].float().cpu().numpy()  # (bsz, h, seq)
                attn_last[b0:b0 + len(b_trials), k] = attn_last_row.astype(np.float16)

            # --- next-token logits at last position
            logits_last = out.logits[:, -1, :].float()   # (bsz, vocab)
            probs = torch.softmax(logits_last, dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1)
            next_logits_lowhigh[b0:b0 + len(b_trials), 0] = (
                logits_last[:, short_id].cpu().numpy()
            )
            next_logits_lowhigh[b0:b0 + len(b_trials), 1] = (
                logits_last[:, tall_id].cpu().numpy()
            )
            next_logit_diff[b0:b0 + len(b_trials)] = (
                (logits_last[:, tall_id] - logits_last[:, short_id]).cpu().numpy()
            )
            next_entropy[b0:b0 + len(b_trials)] = ent.cpu().numpy()
            next_argmax[b0:b0 + len(b_trials)] = (
                logits_last.argmax(-1).cpu().numpy()
            )

            if (b0 // BATCH_SIZE) % 25 == 0:
                done = b0 + len(b_trials)
                rate = done / max(1e-3, time.time() - t1)
                eta = (n - done) / max(1e-3, rate)
                print(f"[v10]   {done}/{n}   rate={rate:.1f} p/s   eta={eta:.0f}s",
                      flush=True)
    finally:
        for h in handles:
            h.remove()
    print(f"[v10] forward pass done in {time.time() - t1:.1f}s", flush=True)

    # --- consolidate buffered captures into the output arrays
    print("[v10] consolidating buffered captures...", flush=True)
    for L_idx in range(n_layers):
        residuals[:, L_idx, :] = np.concatenate(captured_resid_buf[L_idx], axis=0)
    for k, L_idx in enumerate(STRATEGIC_LAYERS):
        head_outs[:, k, :, :] = np.concatenate(captured_head_buf[L_idx], axis=0)

    # --- labels
    ids = np.array([t["id"] for t in trials])
    xs = np.array([t["x"] for t in trials], dtype=np.float32)
    zs = np.array([t["z"] for t in trials], dtype=np.float32)
    mus = np.array([t["mu"] for t in trials], dtype=np.float32)
    sigmas = np.array([t["sigma"] for t in trials], dtype=np.float32)
    seeds = np.array([t["seed"] for t in trials], dtype=np.int32)

    # --- save
    print("[v10] writing outputs...", flush=True)
    res_path = OUT_DIR / "gemma2_height_v10_residuals.npz"
    np.savez(res_path,
             activations=residuals,
             ids=ids, x=xs, z=zs, mu=mus, sigma=sigmas, seed=seeds,
             next_logits_lowhigh=next_logits_lowhigh,
             next_logit_diff=next_logit_diff,
             next_entropy=next_entropy,
             next_argmax=next_argmax,
             layer_indices=np.arange(n_layers))
    print(f"[v10]   {res_path}  ({res_path.stat().st_size/1e6:.1f} MB)", flush=True)

    attn_path = OUT_DIR / "gemma2_height_v10_attention.npz"
    np.savez(attn_path,
             attn_last_row=attn_last,
             head_outputs=head_outs,
             attn_layers=np.array(STRATEGIC_LAYERS, dtype=np.int32),
             seq_len_unpadded=seq_len_unpad,
             seq_len_padded=np.int32(MAX_SEQ),
             target_pos_unpadded=target_pos_unpad,
             context_pos_unpadded=context_pos_unpad)
    print(f"[v10]   {attn_path}  ({attn_path.stat().st_size/1e6:.1f} MB)", flush=True)

    # --- export W_O slices for each strategic layer (needed for DLA in P4)
    wo_path = OUT_DIR / "gemma2_height_v10_W_O_strategic.npz"
    wo_dict = {}
    for L_idx in STRATEGIC_LAYERS:
        W_O = decoder_layers[L_idx].self_attn.o_proj.weight.detach().float().cpu().numpy()
        wo_dict[f"L{L_idx}"] = W_O.astype(np.float32)  # (d_model, n_heads*head_dim)
    np.savez(wo_path, **wo_dict)
    print(f"[v10]   {wo_path}  ({wo_path.stat().st_size/1e6:.1f} MB)", flush=True)

    # --- export W_U for direct logit attribution
    wu_path = OUT_DIR / "gemma2_W_U.npz"
    if not wu_path.exists():
        try:
            wu = model.lm_head.weight.detach().float().cpu().numpy()
        except AttributeError:
            wu = model.get_output_embeddings().weight.detach().float().cpu().numpy()
        np.savez(wu_path, W_U=wu.astype(np.float32),
                 short_id=np.int32(short_id), tall_id=np.int32(tall_id))
        print(f"[v10]   {wu_path}  ({wu_path.stat().st_size/1e6:.1f} MB)", flush=True)

    # --- meta
    meta = {
        "model_id": MODEL_ID,
        "n_prompts": n,
        "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
        "d_model": d_model, "vocab_size": vocab_size,
        "strategic_layers": STRATEGIC_LAYERS,
        "max_seq": MAX_SEQ,
        "batch_size": BATCH_SIZE,
        "tall_id": int(tall_id),
        "short_id": int(short_id),
        "elapsed_sec": round(time.time() - t_start, 1),
        "trials_path": str(TRIALS_PATH),
        "extractor": "scripts/vast_remote/extract_v10_dense_height.py",
    }
    (OUT_DIR / "gemma2_height_v10_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[v10] meta: {meta}", flush=True)

    # --- behavioral sanity check (quick R for full grid)
    from numpy.linalg import lstsq
    # Average logit_diff per (x, z) cell, then R between cell-mean and cell z.
    cell = {}
    for i in range(n):
        key = (round(xs[i], 2), round(zs[i], 2))
        cell.setdefault(key, []).append(next_logit_diff[i])
    means = np.array([np.mean(v) for v in cell.values()])
    cell_zs = np.array([k[1] for k in cell.keys()])
    cell_xs = np.array([k[0] for k in cell.keys()])
    r_z = np.corrcoef(means, cell_zs)[0, 1]
    r_x = np.corrcoef(means, cell_xs)[0, 1]
    print(f"[v10] CELL-MEAN BEHAV: corr(logit_diff_mean, z) = {r_z:.3f}", flush=True)
    print(f"[v10] CELL-MEAN BEHAV: corr(logit_diff_mean, x) = {r_x:.3f}", flush=True)

    # --- optional wandb
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        try:
            import wandb
            wandb.init(project="geometry-of-relativity",
                       name=f"v10-extract-{int(time.time())}",
                       config=meta, dir=str(REPO / "logs"))
            wandb.log({
                "behavioral/r_z_cellmean": r_z,
                "behavioral/r_x_cellmean": r_x,
                "behavioral/entropy_mean": float(next_entropy.mean()),
                "behavioral/logit_diff_mean": float(next_logit_diff.mean()),
                "extraction/elapsed_sec": meta["elapsed_sec"],
                "extraction/n_prompts": n,
            })
            wandb.finish()
            print("[v10] wandb log written", flush=True)
        except Exception as e:
            print(f"[v10] wandb skipped: {e}", flush=True)
    else:
        print("[v10] WANDB_API_KEY not set, skipping wandb", flush=True)

    print(f"[v10] TOTAL elapsed: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    run()
