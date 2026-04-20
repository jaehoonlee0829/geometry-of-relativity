# vast_remote — extractor scripts uploaded to Vast instances

These self-contained scripts are designed to run on a Vast.ai GPU box and
produce per-layer activation `.npz` files in `results/activations/` on the
remote (not in this repo — the `.npy`/`.npz` outputs are gitignored).

## Files

- `extract_e4b_v3.py` — Gemma 4 E4B (42 layers, d=2560). Layers early=10, mid=21, late=32, final=41. Saves `e4b_W_U.npy` once (~2.7 GB) + 8 tiny `e4b_{domain}_{layer}.npz` (~2.6 MB each).
- `extract_g31b_v1.py` — Gemma 4 31B (60 layers, d=5376). Layers early=14, mid=30, late=45, final=59. Saves `g31b_W_U.npy` once (~5.6 GB) + 8 `g31b_{domain}_{layer}.npz` (~5.4 MB each).

## Schema

Each `.npz` contains:
- `activations`: `(n_prompts, d)` float32 — last-token hidden state at the target layer
- `ids`: prompt IDs (string array)
- `layer_index`, `layer_name`, `model_id`: metadata scalars

The unembedding matrix `W_U` is saved once per model as a separate `.npy`
(since re-saving it per layer wastes GB of disk and minutes of zlib CPU —
learned the hard way on v2).

## Upload flow

These scripts are uploaded to the remote via Jupyter's Contents API (PUT
/api/contents/...) rather than heredoc paste, which silently truncated
large multi-line scripts. See VAST_INSTRUCTIONS.md for the one-liner.

## Run

```
cd /workspace/repo
nohup python3 -u /workspace/repo/extract_e4b_v3.py > /tmp/extract_e4b_v3.log 2>&1 &
```
