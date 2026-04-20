"""Pull Gemma 4 activation .npz files from W&B Artifact `gemma4-activations:day4`.

By default, pulls only the small per-layer .npz files (~60 MB total) into
`results/activations/`. Skips the big W_U.npy files (2.7 GB + 5.6 GB) unless
`--with-wu` is passed — those are only needed for Fisher pullback math.

After download, verifies shapes, layer_index metadata, and prompt counts
against what we expect from PLANNING.md v2.

Usage:
    # Activations only (recommended default — ~60 MB, fast):
    python scripts/pull_activations_from_wandb.py

    # Activations + W_U matrices (~8 GB, for Fisher math locally):
    python scripts/pull_activations_from_wandb.py --with-wu

    # Different artifact alias:
    python scripts/pull_activations_from_wandb.py --alias v2-prompts

Requires: WANDB_API_KEY in env OR `wandb login` already run on this machine.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ARTIFACT = "xrong-optiver/geometry-of-relativity/gemma4-activations"
DEFAULT_ALIAS = "day4"

# What we expect to see after download — from PLANNING.md v2 and INDIGO-COMPASS results.
EXPECTED = {
    "e4b": {
        "hidden": 2560,
        "layers": {"early": 10, "mid": 21, "late": 32, "final": 41},
        "wu_vocab": 262144,
    },
    "g31b": {
        "hidden": 5376,
        "layers": {"early": 14, "mid": 30, "late": 45, "final": 59},
        "wu_vocab": 262144,
    },
}
N_HEIGHT = 252
N_WEALTH = 196


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alias", default=DEFAULT_ALIAS, help=f"Artifact alias (default: {DEFAULT_ALIAS})")
    p.add_argument(
        "--out-dir",
        default="results/activations",
        help="Local output directory (default: results/activations)",
    )
    p.add_argument(
        "--with-wu",
        action="store_true",
        help="Also download the W_U.npy matrices (~8 GB). Default: skip.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download, just verify files already on disk.",
    )
    return p.parse_args()


def download(alias: str, out_dir: Path, with_wu: bool) -> Path:
    """Download artifact files into `out_dir`. Returns the local artifact directory."""
    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        # wandb.Api() will also fall back to ~/.netrc from `wandb login`, so just warn.
        print("[warn] WANDB_API_KEY not set; relying on cached `wandb login` credentials.", file=sys.stderr)

    api = wandb.Api()
    art = api.artifact(f"{ARTIFACT}:{alias}")
    print(f"[pull] artifact {art.name} ({art.size / 1e9:.2f} GB total across {len(art.manifest.entries)} files)")

    out_dir.mkdir(parents=True, exist_ok=True)

    if with_wu:
        print("[pull] downloading EVERYTHING including W_U.npy matrices")
        local = art.download(root=str(out_dir))
    else:
        # Filter: skip W_U, pull only .npz activation files.
        kept = 0
        skipped = 0
        for name, entry in art.manifest.entries.items():
            if name.endswith("W_U.npy"):
                skipped += 1
                continue
            out_path = out_dir / name
            if out_path.exists() and out_path.stat().st_size == entry.size:
                print(f"  [skip-existing] {name}")
                kept += 1
                continue
            print(f"  [get] {name} ({entry.size / 1e6:.1f} MB)")
            art.get_entry(name).download(root=str(out_dir))
            kept += 1
        print(f"[pull] {kept} files downloaded, {skipped} W_U.npy files skipped (pass --with-wu to fetch them)")
        local = str(out_dir)
    return Path(local)


def verify(out_dir: Path, with_wu: bool) -> bool:
    """Verify every expected file is present, loads cleanly, and has the right shape."""
    all_good = True
    for model, spec in EXPECTED.items():
        for domain, n_expected in [("height", N_HEIGHT), ("wealth", N_WEALTH)]:
            for layer_name, layer_idx in spec["layers"].items():
                fname = f"{model}_{domain}_{layer_name}.npz"
                path = out_dir / fname
                if not path.exists():
                    print(f"  [MISSING] {fname}")
                    all_good = False
                    continue
                with np.load(path, allow_pickle=True) as z:
                    acts = z["activations"]
                    want_shape = (n_expected, spec["hidden"])
                    if acts.shape != want_shape:
                        print(f"  [BAD SHAPE] {fname}: got {acts.shape}, want {want_shape}")
                        all_good = False
                    elif int(z["layer_index"]) != layer_idx:
                        print(f"  [BAD LAYER IDX] {fname}: got {int(z['layer_index'])}, want {layer_idx}")
                        all_good = False
                    else:
                        std = float(acts.std())
                        print(
                            f"  [ok] {fname}  shape={acts.shape}  layer={layer_idx}  "
                            f"mean={float(acts.mean()):+.3f}  std={std:.3f}"
                        )
        if with_wu:
            wu_path = out_dir / f"{model}_W_U.npy"
            if not wu_path.exists():
                print(f"  [MISSING] {wu_path.name}")
                all_good = False
            else:
                wu = np.load(wu_path, mmap_mode="r")
                want_wu = (spec["wu_vocab"], spec["hidden"])
                if wu.shape != want_wu:
                    print(f"  [BAD W_U SHAPE] {wu_path.name}: got {wu.shape}, want {want_wu}")
                    all_good = False
                else:
                    print(f"  [ok] {wu_path.name}  shape={wu.shape}  ({wu_path.stat().st_size / 1e9:.2f} GB)")
    return all_good


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if not args.verify_only:
        download(args.alias, out_dir, args.with_wu)

    print(f"\n[verify] shapes + metadata in {out_dir}")
    ok = verify(out_dir, args.with_wu)
    if ok:
        print("\n[verify] ALL FILES OK.")
        return 0
    else:
        print("\n[verify] PROBLEMS FOUND above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
