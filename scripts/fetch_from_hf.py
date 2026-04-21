"""Fetch bulk experiment data from the HF dataset.

The `.jsonl` logit+trial files and `.npz` activation files are NOT in git
(too large, regenerable, and already mirrored to HF). Analysis scripts under
`scripts/vast_remote/exp*.py` expect them to be present at `results/...`.

Run this once after cloning:

    python scripts/fetch_from_hf.py                  # downloads everything
    python scripts/fetch_from_hf.py --only v4_dense  # or just one folder

The dataset repo is private; set `HF_TOKEN` in the environment (or use
`huggingface-cli login`) before running.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "xrong1729/mech-interp-relativity-activations"

# HF path → local path  (both relative to repo root)
FOLDERS = {
    "activations":          "results/activations",
    "v4_dense":             "results/v4_dense",
    "v4_adjpairs":          "results/v4_adjpairs",
    "v4_zeroshot_expanded": "results/v4_zeroshot_expanded",
    "v4_abs_controls":      "results/v4_abs_controls",
    "prompts":              "data_gen",  # prompts_v*.jsonl live here in repo
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None,
                    help="subset of folders to fetch (repeatable). Default: all.")
    ap.add_argument("--data-kind", default="both", choices=["npz", "jsonl", "both"],
                    help="fetch only activations (npz), only logits/trials (jsonl), or both")
    args = ap.parse_args()

    selected = args.only or list(FOLDERS)
    missing = [s for s in selected if s not in FOLDERS]
    if missing:
        raise SystemExit(f"unknown folder(s): {missing}. Valid: {list(FOLDERS)}")

    patterns = {"npz": ["*.npz"], "jsonl": ["*.jsonl"],
                "both": ["*.npz", "*.jsonl"]}[args.data_kind]

    token = os.environ.get("HF_TOKEN")
    for hf_name in selected:
        local = Path(FOLDERS[hf_name])
        local.mkdir(parents=True, exist_ok=True)
        allow = [f"{hf_name}/{p}" for p in patterns]
        print(f"fetching {hf_name}/ → {local}/  patterns={allow}")
        # snapshot_download fetches to a cache; we move the files we care about
        cache_root = snapshot_download(
            repo_id=REPO_ID, repo_type="dataset",
            allow_patterns=allow, token=token,
        )
        src = Path(cache_root) / hf_name
        if not src.exists():
            print(f"  (nothing matched in {hf_name}/)")
            continue
        n = 0
        for f in src.iterdir():
            if f.is_file():
                dest = local / f.name
                if not dest.exists():
                    os.symlink(f.resolve(), dest)
                    n += 1
        print(f"  linked {n} files")
    print("done")


if __name__ == "__main__":
    main()
