"""Upload v11 dense-grid activations + prompts to the HF dataset, per pair × model.

Mirrors the v10 upload script but parameterized over (model, pair) and with a
``--skip-existing`` flag so a partial-run resume just skips files that are
already on the HF side. Per-pair upload after each pair completes — that
way a network blip on pair 5/8 doesn't lose pairs 1–4.

Layout on HF (under ``xrong1729/mech-interp-relativity-activations``):

    prompts/v11_<pair>_trials.jsonl
    v11/<model_short>/<pair>/<base>_residuals.npz
    v11/<model_short>/<pair>/<base>_attention.npz
    v11/<model_short>/<pair>/<base>_W_O_strategic.npz
    v11/<model_short>/<pair>/<base>_meta.json
    v11/<model_short>/<model_short>_W_U.npz   (one per model, not per pair)

where ``<base> = <model_short>_<pair>_v11``.

What is intentionally skipped (matching v10's policy): nothing — for v11 the
W_U file is small (≤2 GB; for 9B, vocab×d_model = 256k×3584 ≈ 1.8 GB fp32) but
still load-bearing for downstream P3d analyses, so we ship it once per model.

Usage:
    python scripts/upload_v11_to_hf.py --model google/gemma-2-2b --pair height
    python scripts/upload_v11_to_hf.py --model google/gemma-2-2b --pair all
    python scripts/upload_v11_to_hf.py --model google/gemma-2-2b --pair height --skip-existing
    python scripts/upload_v11_to_hf.py --model google/gemma-2-2b --pair height --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd

REPO_ID = "xrong1729/mech-interp-relativity-activations"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

MODEL_SHORT: dict[str, str] = {
    "google/gemma-2-2b": "gemma2-2b",
    "google/gemma-2-9b": "gemma2-9b",
}

ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]


def plan_uploads(model_id: str, pair: str) -> list[tuple[str, str]]:
    """Return list of (local_path_relative_to_ROOT, repo_path_on_hf)."""
    short = MODEL_SHORT[model_id]
    base = f"{short}_{pair}_v11"
    pair_dir = f"results/v11/{short}/{pair}"
    plans: list[tuple[str, str]] = [
        (f"data_gen/v11_{pair}_trials.jsonl",
         f"prompts/v11_{pair}_trials.jsonl"),
        (f"{pair_dir}/{base}_residuals.npz",
         f"v11/{short}/{pair}/{base}_residuals.npz"),
        (f"{pair_dir}/{base}_attention.npz",
         f"v11/{short}/{pair}/{base}_attention.npz"),
        (f"{pair_dir}/{base}_W_O_strategic.npz",
         f"v11/{short}/{pair}/{base}_W_O_strategic.npz"),
        (f"{pair_dir}/{base}_meta.json",
         f"v11/{short}/{pair}/{base}_meta.json"),
        # W_U is shared across pairs of the same model, lives one level up.
        (f"results/v11/{short}/{short}_W_U.npz",
         f"v11/{short}/{short}_W_U.npz"),
    ]
    return plans


def upload_one(model_id: str, pair: str, *, dry_run: bool, skip_existing: bool) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    plans = plan_uploads(model_id, pair)

    missing: list[str] = []
    sized: list[tuple[str, str, int]] = []
    for local, remote in plans:
        p = ROOT / local
        if not p.exists():
            missing.append(local)
        else:
            sized.append((local, remote, p.stat().st_size))

    if missing:
        print(f"[upload] {model_id} / {pair}: SKIP ({len(missing)} missing locals)",
              file=sys.stderr)
        for m in missing:
            print(f"           - {m}", file=sys.stderr)
        return

    existing_remote: set[str] = set()
    if skip_existing:
        try:
            existing_remote = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
        except Exception as e:
            print(f"[upload] could not list remote files (will upload all): {e}",
                  file=sys.stderr)
            existing_remote = set()

    to_upload: list[tuple[str, str, int]] = []
    for local, remote, size in sized:
        if skip_existing and remote in existing_remote:
            print(f"[upload]   skip (exists on HF): {remote}")
            continue
        to_upload.append((local, remote, size))

    if not to_upload:
        print(f"[upload] {model_id} / {pair}: nothing to do (all files already present)")
        return

    total = sum(s for _, _, s in to_upload)
    print(f"[upload] {model_id} / {pair}: {len(to_upload)} files, {total/1e9:.2f} GB")
    for local, remote, size in to_upload:
        print(f"           {size/1e6:>8.1f} MB  {local}  ->  {remote}")

    if dry_run:
        print("[upload] DRY RUN — not committing")
        return

    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(ROOT / local))
        for local, remote, _ in to_upload
    ]
    short = MODEL_SHORT[model_id]
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"v11: upload {short}/{pair}",
    )
    print(f"[upload] committed {len(operations)} files for {short}/{pair}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SHORT.keys()))
    ap.add_argument("--pair", required=True,
                    help="pair name or 'all' for every pair")
    ap.add_argument("--skip-existing", action="store_true",
                    help="check the dataset's file list first; only upload new files")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be uploaded without doing it")
    args = ap.parse_args()

    if args.pair == "all":
        pairs = ALL_PAIRS
    else:
        if args.pair not in ALL_PAIRS:
            raise SystemExit(f"unknown pair {args.pair!r}; valid: {ALL_PAIRS}")
        pairs = [args.pair]

    for p in pairs:
        upload_one(args.model, p,
                   dry_run=args.dry_run,
                   skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
