"""Upload v10 dense-height activations + prompts to the HF dataset.

Mirrors `xrong1729/mech-interp-relativity-activations` for the v10 run, so a
fresh `git clone` + `python scripts/fetch_from_hf.py --only v10` gives you the
NPZs needed to re-run every CPU-only analysis (P2-P4, P6) without re-doing the
GPU extraction.

What gets uploaded (~826 MB total):
  prompts/v10_dense_height_trials.jsonl              ( 2.0 MB)
  v10/gemma2_height_v10_residuals.npz                (458   MB)
  v10/gemma2_height_v10_attention.npz                (220   MB)
  v10/gemma2_height_v10_W_O_strategic.npz            (145   MB)
  v10/gemma2_height_v10_meta.json                    (< 1   KB)
  v10/steering_layer_sweep.npz                       (884   KB)

What is INTENTIONALLY skipped:
  results/v10/gemma2_W_U.npz  (2.2 GB) — derivable from the model in two lines:
      from transformers import AutoModelForCausalLM
      W_U = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b'
              ).lm_head.weight.detach().float().cpu().numpy()
  Skipping saves ~70% of bandwidth without losing reproducibility.

Requirements:
  - HF token with WRITE access to xrong1729/mech-interp-relativity-activations
    (the read-only token used by extract_v10_dense_height.py is NOT enough).
    Generate one at https://huggingface.co/settings/tokens with "Write" role.
  - Set HF_TOKEN in .env then `set -a; source .env; set +a` before running.

Usage:
    python scripts/upload_v10_to_hf.py            # upload all v10 artifacts
    python scripts/upload_v10_to_hf.py --dry-run  # show what would be uploaded
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd

REPO = "xrong1729/mech-interp-relativity-activations"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

UPLOADS: list[tuple[str, str]] = [
    # (local_path relative to repo root, repo_path on HF dataset)
    ("data_gen/v10_dense_height_trials.jsonl",          "prompts/v10_dense_height_trials.jsonl"),
    ("results/v10/gemma2_height_v10_residuals.npz",     "v10/gemma2_height_v10_residuals.npz"),
    ("results/v10/gemma2_height_v10_attention.npz",     "v10/gemma2_height_v10_attention.npz"),
    ("results/v10/gemma2_height_v10_W_O_strategic.npz", "v10/gemma2_height_v10_W_O_strategic.npz"),
    ("results/v10/gemma2_height_v10_meta.json",         "v10/gemma2_height_v10_meta.json"),
    ("results/v10/steering_layer_sweep.npz",            "v10/steering_layer_sweep.npz"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be uploaded without doing it")
    args = ap.parse_args()

    missing = []
    total = 0
    for local, _ in UPLOADS:
        p = ROOT / local
        if not p.exists():
            missing.append(local)
        else:
            total += p.stat().st_size

    if missing:
        print("ERROR: missing local files. Run the v10 pipeline first.",
              file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    print(f"Will upload {len(UPLOADS)} files, {total / 1e9:.2f} GB total:")
    for local, remote in UPLOADS:
        size = (ROOT / local).stat().st_size
        print(f"  {size / 1e6:>7.1f} MB  {local}  →  {REPO}:{remote}")

    if args.dry_run:
        print("\n[dry-run] no upload performed")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not in environment. Source .env first.",
              file=sys.stderr)
        sys.exit(2)

    api = HfApi()
    who = api.whoami(token=token)
    role = who.get("auth", {}).get("accessToken", {}).get("role", "?")
    if role != "write":
        print(f"ERROR: HF token role is '{role}', need 'write'. "
              f"Generate a write token at https://huggingface.co/settings/tokens",
              file=sys.stderr)
        sys.exit(3)

    print(f"\nUploading as {who.get('name')} (token role: {role})...")
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(ROOT / l))
           for l, r in UPLOADS]
    t0 = time.time()
    commit = api.create_commit(
        repo_id=REPO, repo_type="dataset",
        operations=ops,
        commit_message="v10 dense-height: prompts + activations + attention + steering",
        token=token,
    )
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Commit: {commit.commit_url}")
    print(f"\nVerify with:\n"
          f"  python scripts/fetch_from_hf.py --only v10 --data-kind both")


if __name__ == "__main__":
    main()
