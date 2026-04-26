"""v11 — generalized dense extractor for any (model, pair).

Drop-in successor to ``extract_v10_dense_height.py`` with:

  * ``--model`` and ``--pair`` CLI args (no MODEL_ID constant)
  * per-model strategic-layer table (no hard-coded ``n_layers==26 and n_heads==8`` assert)
  * per-pair regex for value-char-range detection (no "Person ... cm" assumption)
  * ``--max-seq`` knob (some pairs tokenize longer than height)
  * ``--batch-size`` knob (drop to 8 if 9B + eager attention OOMs at 16)

Captures (matching v10's schema so downstream analyses keep working):

  Residuals: ``(N, n_layers, d_model)`` fp16 — last-token at every decoder block.

  Attention at strategic layers (one set per model):
    - ``attn_last_row[N, len(STRATEGIC), n_heads, MAX_SEQ]`` fp16
    - ``head_outputs[N, len(STRATEGIC), n_heads, head_dim]`` fp16
      (per-head value mix BEFORE o_proj — enables DLA via ``W_O[:, h*hd:(h+1)*hd] @ head_outputs[h]``)

  Logits: ``next_logits_lowhigh[N, 2] = [logit(low_word), logit(high_word)]``,
  ``next_logit_diff = high - low``, ``next_entropy``, ``next_argmax``.

  Token-position bookkeeping for downstream attention-pattern analysis:
  ``seq_len_unpadded``, ``target_pos_unpadded``, ``context_pos_unpadded[N, 15, 2]``.

The script is replicable on any GPU box; every numerical claim later made on
top of these dumps comes from CPU-only analysis files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

N_CONTEXT = 15

STRATEGIC_LAYERS_BY_MODEL: dict[str, list[int]] = {
    "google/gemma-2-2b": [0, 3, 7, 10, 13, 17, 20, 25],
    "google/gemma-2-9b": [0, 5, 10, 16, 21, 28, 34, 41],
}

MODEL_SHORT: dict[str, str] = {
    "google/gemma-2-2b": "gemma2-2b",
    "google/gemma-2-9b": "gemma2-9b",
}

# Per-pair regex used by find_value_char_ranges. Each line that ends a value
# the model later predicts on must contain a single capture group around the
# numeric value's character span. Order independent — applied per-line.
PAIR_LINE_REGEX: dict[str, str] = {
    "height":     r"^Person \d+: (\d+(?:\.\d+)?) cm$",
    "age":        r"^Person \d+: (\d+(?:\.\d+)?) years old$",
    "weight":     r"^Person \d+: (\d+(?:\.\d+)?) kg$",
    "size":       r"^Object \d+: (\d+(?:\.\d+)?) cm across$",
    "speed":      r"^Vehicle \d+: (\d+(?:\.\d+)?) km/h$",
    "wealth":     r"^Person \d+ earns \$(\d+(?:\.\d+)?)/year$",
    "experience": r"^Worker \d+: (\d+(?:\.\d+)?) years experience$",
    "bmi_abs":    r"^Person \d+: BMI (\d+(?:\.\d+)?)$",
}


def find_value_char_ranges(
    prompt: str, pair: str
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Return (context_char_ranges_15, target_char_range) in absolute chars.

    Matches all 16 value-bearing lines (15 context + 1 target). The target is
    the line whose match contains the trailing prompt text — i.e., the last
    matched line.
    """
    pattern = re.compile(PAIR_LINE_REGEX[pair])
    matches: list[tuple[int, int]] = []  # (val_start, val_end) chars
    char_pos = 0
    for line in prompt.split("\n"):
        # Strip trailing fragments like ". This person is" from the target line
        # before matching: match against the line up to the first '.' that
        # isn't inside the value (e.g. "BMI 28.5. This person is" -> "BMI 28.5"
        # is a multi-decimal subtle case; cheaper to match against the rstripped
        # version: split on ". This " and ". An " etc.).
        clean = line
        for sep in (". This ", ". An ", ". A "):
            if sep in clean:
                clean = clean.split(sep, 1)[0]
                break
        m = pattern.match(clean)
        if m:
            v_start = char_pos + m.start(1)
            v_end = char_pos + m.end(1)
            matches.append((v_start, v_end))
        char_pos += len(line) + 1  # +1 for the newline
    if len(matches) < N_CONTEXT + 1:
        raise AssertionError(
            f"expected ≥{N_CONTEXT + 1} value lines for pair={pair}, got {len(matches)}; "
            f"first 200 chars of prompt: {prompt[:200]!r}"
        )
    ctx = matches[:N_CONTEXT]
    tgt = matches[N_CONTEXT]  # the 16th match is the target
    return ctx, tgt


def char_to_token_range(
    offsets: list[tuple[int, int]], char_range: tuple[int, int]
) -> tuple[int, int]:
    c0, c1 = char_range
    tok_start = None
    tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue  # special token
        if e > c0 and s < c1:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
    assert tok_start is not None and tok_end is not None, \
        f"no tokens cover {char_range}"
    return tok_start, tok_end


def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("could not locate decoder layers")


def run(args: argparse.Namespace) -> None:
    t_start = time.time()

    model_id = args.model
    pair = args.pair
    if model_id not in STRATEGIC_LAYERS_BY_MODEL:
        raise ValueError(f"unknown model {model_id!r}; add it to STRATEGIC_LAYERS_BY_MODEL")
    if pair not in PAIR_LINE_REGEX:
        raise ValueError(f"unknown pair {pair!r}; add it to PAIR_LINE_REGEX")

    strategic_layers = STRATEGIC_LAYERS_BY_MODEL[model_id]
    model_short = MODEL_SHORT[model_id]
    out_dir = REPO / "results" / "v11" / model_short / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_path = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    if not trials_path.exists():
        raise FileNotFoundError(
            f"trials not found at {trials_path}. "
            f"Run: python scripts/gen_v11_dense.py --pair {pair}"
        )

    trials = [json.loads(l) for l in trials_path.open()]
    n = len(trials)
    print(f"[v11] {model_short}/{pair}: {n} trials from {trials_path.name}", flush=True)

    low_word = trials[0]["low_word"]
    high_word = trials[0]["high_word"]

    print(f"[v11] loading {model_id} (eager attn)...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    print(f"[v11]   loaded in {time.time() - t0:.1f}s", flush=True)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    d_model = cfg.hidden_size
    vocab_size = cfg.vocab_size
    print(f"[v11]   layers={n_layers} heads={n_heads} head_dim={head_dim} "
          f"d_model={d_model} vocab={vocab_size}", flush=True)

    decoder_layers = get_decoder_layers(model)

    print("[v11] tokenizer check:", flush=True)
    report_tokenization(tok, [low_word, high_word])
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)
    print(f"[v11]   low={low_word!r}({low_id})  high={high_word!r}({high_id})", flush=True)

    print("[v11] pre-tokenizing for position bookkeeping...", flush=True)
    seq_len_unpad = np.zeros(n, dtype=np.int32)
    target_pos_unpad = np.zeros((n, 2), dtype=np.int32)
    context_pos_unpad = np.zeros((n, N_CONTEXT, 2), dtype=np.int32)
    for i, t in enumerate(trials):
        enc = tok(t["prompt"], return_offsets_mapping=True, add_special_tokens=True)
        seq_len_unpad[i] = len(enc.input_ids)
        ctx_ranges, tgt_range = find_value_char_ranges(t["prompt"], pair)
        offsets = enc.offset_mapping
        target_pos_unpad[i] = char_to_token_range(offsets, tgt_range)
        for k, cr in enumerate(ctx_ranges):
            context_pos_unpad[i, k] = char_to_token_range(offsets, cr)
    print(f"[v11]   seq_len min/max: {seq_len_unpad.min()}/{seq_len_unpad.max()}",
          flush=True)
    if seq_len_unpad.max() > args.max_seq:
        raise AssertionError(
            f"--max-seq={args.max_seq} too small for actual max {seq_len_unpad.max()}; "
            f"rerun with --max-seq {seq_len_unpad.max() + 16}"
        )

    residuals = np.zeros((n, n_layers, d_model), dtype=np.float16)
    attn_last = np.zeros((n, len(strategic_layers), n_heads, args.max_seq),
                         dtype=np.float16)
    head_outs = np.zeros((n, len(strategic_layers), n_heads, head_dim),
                         dtype=np.float16)
    next_logits_lowhigh = np.zeros((n, 2), dtype=np.float32)
    next_logit_diff = np.zeros(n, dtype=np.float32)
    next_entropy = np.zeros(n, dtype=np.float32)
    next_argmax = np.zeros(n, dtype=np.int32)

    captured_resid_buf: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    captured_head_buf: dict[int, list[np.ndarray]] = {L: [] for L in strategic_layers}

    def make_layer_hook(layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured_resid_buf[layer_idx].append(
                h[:, -1, :].detach().float().cpu().numpy().astype(np.float16)
            )
        return hook

    def make_o_proj_pre_hook(layer_idx: int):
        def pre_hook(module, args_):
            x = args_[0]  # (bsz, q_len, n_heads*head_dim)
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
    for L_idx in strategic_layers:
        o_proj = decoder_layers[L_idx].self_attn.o_proj
        handles.append(o_proj.register_forward_pre_hook(make_o_proj_pre_hook(L_idx)))

    print(f"[v11] forward pass: {n} prompts, batch {args.batch_size}, "
          f"eager attn, MAX_SEQ={args.max_seq}...", flush=True)
    t1 = time.time()
    try:
        for b0 in range(0, n, args.batch_size):
            b_trials = trials[b0:b0 + args.batch_size]
            prompts = [t["prompt"] for t in b_trials]
            enc = tok(prompts, return_tensors="pt",
                      padding="max_length", max_length=args.max_seq,
                      truncation=True).to(model.device)

            with torch.no_grad():
                out = model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    output_attentions=True,
                    use_cache=False,
                )

            for k, L_idx in enumerate(strategic_layers):
                attn = out.attentions[L_idx]   # (bsz, n_heads, seq, seq)
                attn_last_row = attn[:, :, -1, :].float().cpu().numpy()
                attn_last[b0:b0 + len(b_trials), k] = attn_last_row.astype(np.float16)

            logits_last = out.logits[:, -1, :].float()
            probs = torch.softmax(logits_last, dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1)
            next_logits_lowhigh[b0:b0 + len(b_trials), 0] = (
                logits_last[:, low_id].cpu().numpy()
            )
            next_logits_lowhigh[b0:b0 + len(b_trials), 1] = (
                logits_last[:, high_id].cpu().numpy()
            )
            next_logit_diff[b0:b0 + len(b_trials)] = (
                (logits_last[:, high_id] - logits_last[:, low_id]).cpu().numpy()
            )
            next_entropy[b0:b0 + len(b_trials)] = ent.cpu().numpy()
            next_argmax[b0:b0 + len(b_trials)] = (
                logits_last.argmax(-1).cpu().numpy()
            )

            if (b0 // args.batch_size) % 25 == 0:
                done = b0 + len(b_trials)
                rate = done / max(1e-3, time.time() - t1)
                eta = (n - done) / max(1e-3, rate)
                print(f"[v11]   {done}/{n}   rate={rate:.1f} p/s   eta={eta:.0f}s",
                      flush=True)
    finally:
        for h in handles:
            h.remove()
    print(f"[v11] forward pass done in {time.time() - t1:.1f}s", flush=True)

    print("[v11] consolidating buffered captures...", flush=True)
    for L_idx in range(n_layers):
        residuals[:, L_idx, :] = np.concatenate(captured_resid_buf[L_idx], axis=0)
    for k, L_idx in enumerate(strategic_layers):
        head_outs[:, k, :, :] = np.concatenate(captured_head_buf[L_idx], axis=0)

    ids = np.array([t["id"] for t in trials])
    xs = np.array([t["x"] for t in trials], dtype=np.float32)
    zs = np.array([t["z"] for t in trials], dtype=np.float32)
    mus = np.array([t["mu"] for t in trials], dtype=np.float32)
    sigmas = np.array([t["sigma"] for t in trials], dtype=np.float32)
    seeds = np.array([t["seed"] for t in trials], dtype=np.int32)

    base = f"{model_short}_{pair}_v11"

    print("[v11] writing outputs...", flush=True)
    res_path = out_dir / f"{base}_residuals.npz"
    np.savez(res_path,
             activations=residuals,
             ids=ids, x=xs, z=zs, mu=mus, sigma=sigmas, seed=seeds,
             next_logits_lowhigh=next_logits_lowhigh,
             next_logit_diff=next_logit_diff,
             next_entropy=next_entropy,
             next_argmax=next_argmax,
             layer_indices=np.arange(n_layers))
    print(f"[v11]   {res_path}  ({res_path.stat().st_size/1e6:.1f} MB)", flush=True)

    attn_path = out_dir / f"{base}_attention.npz"
    np.savez(attn_path,
             attn_last_row=attn_last,
             head_outputs=head_outs,
             attn_layers=np.array(strategic_layers, dtype=np.int32),
             seq_len_unpadded=seq_len_unpad,
             seq_len_padded=np.int32(args.max_seq),
             target_pos_unpadded=target_pos_unpad,
             context_pos_unpadded=context_pos_unpad)
    print(f"[v11]   {attn_path}  ({attn_path.stat().st_size/1e6:.1f} MB)", flush=True)

    wo_path = out_dir / f"{base}_W_O_strategic.npz"
    wo_dict = {}
    for L_idx in strategic_layers:
        W_O = decoder_layers[L_idx].self_attn.o_proj.weight.detach().float().cpu().numpy()
        wo_dict[f"L{L_idx}"] = W_O.astype(np.float32)
    np.savez(wo_path, **wo_dict)
    print(f"[v11]   {wo_path}  ({wo_path.stat().st_size/1e6:.1f} MB)", flush=True)

    # W_U is per-model, not per-pair — write once into the model directory.
    wu_path = out_dir.parent / f"{model_short}_W_U.npz"
    if not wu_path.exists():
        try:
            wu = model.lm_head.weight.detach().float().cpu().numpy()
        except AttributeError:
            wu = model.get_output_embeddings().weight.detach().float().cpu().numpy()
        np.savez(wu_path, W_U=wu.astype(np.float32),
                 low_id=np.int32(low_id), high_id=np.int32(high_id),
                 low_word=low_word, high_word=high_word, pair_for_lowhigh=pair)
        print(f"[v11]   {wu_path}  ({wu_path.stat().st_size/1e6:.1f} MB)", flush=True)

    meta = {
        "model_id": model_id,
        "model_short": model_short,
        "pair": pair,
        "low_word": low_word,
        "high_word": high_word,
        "n_prompts": n,
        "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
        "d_model": d_model, "vocab_size": vocab_size,
        "strategic_layers": strategic_layers,
        "max_seq": args.max_seq,
        "batch_size": args.batch_size,
        "low_id": int(low_id),
        "high_id": int(high_id),
        "elapsed_sec": round(time.time() - t_start, 1),
        "trials_path": str(trials_path.relative_to(REPO)),
        "extractor": "scripts/vast_remote/extract_v11_dense.py",
    }
    (out_dir / f"{base}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[v11] meta: {meta}", flush=True)

    cell = {}
    for i in range(n):
        key = (round(xs[i], 4), round(zs[i], 4))
        cell.setdefault(key, []).append(next_logit_diff[i])
    means = np.array([np.mean(v) for v in cell.values()])
    cell_zs = np.array([k[1] for k in cell.keys()])
    cell_xs = np.array([k[0] for k in cell.keys()])
    r_z = np.corrcoef(means, cell_zs)[0, 1]
    r_x = np.corrcoef(means, cell_xs)[0, 1]
    print(f"[v11] CELL-MEAN BEHAV: corr(logit_diff_mean, z) = {r_z:.3f}", flush=True)
    print(f"[v11] CELL-MEAN BEHAV: corr(logit_diff_mean, x) = {r_x:.3f}", flush=True)

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        try:
            import wandb
            wandb.init(project="geometry-of-relativity",
                       name=f"v11-{model_short}-{pair}-{int(time.time())}",
                       config=meta, dir=str(REPO / "logs"),
                       tags=["v11", model_short, pair])
            wandb.log({
                "behavioral/r_z_cellmean": r_z,
                "behavioral/r_x_cellmean": r_x,
                "behavioral/entropy_mean": float(next_entropy.mean()),
                "behavioral/logit_diff_mean": float(next_logit_diff.mean()),
                "extraction/elapsed_sec": meta["elapsed_sec"],
                "extraction/n_prompts": n,
            })
            wandb.finish()
            print("[v11] wandb log written", flush=True)
        except Exception as e:
            print(f"[v11] wandb skipped: {e}", flush=True)
    else:
        print("[v11] WANDB_API_KEY not set, skipping wandb", flush=True)

    print(f"[v11] TOTAL elapsed: {time.time() - t_start:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=sorted(STRATEGIC_LAYERS_BY_MODEL.keys()),
                    help="HuggingFace model id")
    ap.add_argument("--pair", required=True,
                    choices=sorted(PAIR_LINE_REGEX.keys()),
                    help="adjective pair name")
    ap.add_argument("--max-seq", type=int, default=288,
                    help="left-padded sequence length (default 288 — wealth's $-values "
                         "tokenize to ~250)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="forward batch size (drop to 8 if 9B + eager OOMs)")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
