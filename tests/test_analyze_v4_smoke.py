"""Smoke test: generate synthetic v4 data and run analyze_v4.py end-to-end.

Validates that the script handles file I/O, sklearn, PCA, and the
Sigma^{-1} cosine Cholesky path without crashing. Does NOT check the
science is right — that's done on real activations.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def make_fake_v4(seed: int = 0):
    """Create small but realistic fake v4 data under results/v4_dense/.

    Structure mirrors extract_v4_dense.py output exactly.
    """
    rng = np.random.default_rng(seed)
    d = 64  # small hidden dim for speed
    sigma = 10.0
    X_VALUES = [150.0, 160.0, 165.0, 170.0, 180.0]
    MU_VALUES = [145.0, 150.0, 160.0, 165.0, 170.0, 180.0, 185.0]
    N_SEEDS = 20  # fewer seeds than real for speed

    v4 = REPO / "results" / "v4_dense"
    v4.mkdir(parents=True, exist_ok=True)

    # Build a ground-truth z-direction + x-direction in activation space
    true_z_dir = rng.normal(size=d)
    true_z_dir /= np.linalg.norm(true_z_dir)
    true_x_dir = rng.normal(size=d)
    true_x_dir /= np.linalg.norm(true_x_dir)

    trials = []
    idx = 0
    implicit_rows = []
    implicit_acts_mid = []
    implicit_acts_late = []
    implicit_logits = []

    for x in X_VALUES:
        for mu in MU_VALUES:
            z = (x - mu) / sigma
            for s in range(N_SEEDS):
                h = (2.0 * z * true_z_dir
                     + 0.05 * (x - 165.0) * true_x_dir
                     + rng.normal(scale=0.3, size=d))
                h_late = h + 0.5 * z * true_z_dir + rng.normal(scale=0.2, size=d)
                ld = 1.2 * z + 0.02 * (x - 165.0) + rng.normal(scale=0.3)
                tid = f"implicit_{idx:05d}"
                trials.append({"id": tid, "condition": "implicit", "x": x, "mu": mu, "z": z, "seed": s})
                implicit_rows.append(tid)
                implicit_acts_mid.append(h)
                implicit_acts_late.append(h_late)
                implicit_logits.append({
                    "id": tid,
                    "logit_tall": float(ld / 2 + 4),
                    "logit_short": float(-ld / 2 + 4),
                    "logit_diff": float(ld),
                    "top5_tokens": ["_a", "_the", "_tall", "_person", "_short"],
                    "top5_logits": [5.1, 4.9, 4.5, 4.0, 3.8],
                })
                idx += 1

    # explicit — 35 points, deterministic
    explicit_acts_mid = []
    explicit_acts_late = []
    explicit_logits = []
    explicit_rows = []
    for x in X_VALUES:
        for mu in MU_VALUES:
            z = (x - mu) / sigma
            tid = f"explicit_{idx:05d}"
            trials.append({"id": tid, "condition": "explicit", "x": x, "mu": mu, "z": z, "seed": -1})
            h = 1.5 * z * true_z_dir + 0.04 * (x - 165.0) * true_x_dir + rng.normal(scale=0.2, size=d)
            h_late = h + 0.4 * z * true_z_dir
            ld = 1.0 * z + 0.01 * (x - 165.0)
            explicit_rows.append(tid)
            explicit_acts_mid.append(h)
            explicit_acts_late.append(h_late)
            explicit_logits.append({
                "id": tid, "logit_tall": float(ld / 2), "logit_short": float(-ld / 2),
                "logit_diff": float(ld),
                "top5_tokens": ["_a", "_the", "_tall", "_person", "_short"],
                "top5_logits": [5.1, 4.9, 4.5, 4.0, 3.8],
            })
            idx += 1

    # zero_shot — 5 points
    zs_rows = []
    zs_acts_mid = []
    zs_acts_late = []
    zs_logits = []
    for x in X_VALUES:
        tid = f"zeroshot_{idx:05d}"
        trials.append({"id": tid, "condition": "zero_shot", "x": x, "mu": 0.0, "z": 0.0, "seed": -1})
        h = 0.03 * (x - 165.0) * true_x_dir + rng.normal(scale=0.2, size=d)
        h_late = h
        ld = 0.02 * (x - 165.0)
        zs_rows.append(tid)
        zs_acts_mid.append(h)
        zs_acts_late.append(h_late)
        zs_logits.append({
            "id": tid, "logit_tall": float(ld / 2), "logit_short": float(-ld / 2),
            "logit_diff": float(ld),
            "top5_tokens": ["_a", "_the", "_tall"], "top5_logits": [5.1, 4.9, 4.5],
        })
        idx += 1

    # Write trials
    with (v4 / "e4b_trials.jsonl").open("w") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")

    # Write activations
    def save_acts(rows, acts_list, cond, layer):
        arr = np.stack(acts_list).astype(np.float32)
        np.savez(v4 / f"e4b_{cond}_{layer}.npz",
                 activations=arr, ids=np.array(rows),
                 layer_index=21 if layer == "mid" else 32)

    save_acts(implicit_rows, implicit_acts_mid, "implicit", "mid")
    save_acts(implicit_rows, implicit_acts_late, "implicit", "late")
    save_acts(explicit_rows, explicit_acts_mid, "explicit", "mid")
    save_acts(explicit_rows, explicit_acts_late, "explicit", "late")
    save_acts(zs_rows, zs_acts_mid, "zero_shot", "mid")
    save_acts(zs_rows, zs_acts_late, "zero_shot", "late")

    # Write logits
    for rows, rows_log, cond in [
        (implicit_rows, implicit_logits, "implicit"),
        (explicit_rows, explicit_logits, "explicit"),
        (zs_rows, zs_logits, "zero_shot"),
    ]:
        with (v4 / f"e4b_{cond}_logits.jsonl").open("w") as f:
            for r in rows_log:
                f.write(json.dumps(r) + "\n")

    return v4


def test_end_to_end(tmp_path, monkeypatch):
    """Run analyze_v4 on synthetic data and verify summary.json is produced."""
    import importlib.util
    import os
    import shutil

    # Swap REPO temporarily: we'll write fake data to real repo results/ but
    # move any existing dir out of the way first.
    v4 = REPO / "results" / "v4_dense"
    analysis_out = REPO / "results" / "v4_analysis"
    # Back up pre-existing dirs
    backups = []
    for d in [v4, analysis_out]:
        if d.exists():
            bak = d.with_suffix(".bak_smoke")
            if bak.exists():
                shutil.rmtree(bak)
            d.rename(bak)
            backups.append((d, bak))

    try:
        make_fake_v4()

        # Run the script as subprocess — simulates real invocation
        result = subprocess.run(
            [sys.executable, "scripts/vast_remote/analyze_v4.py"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"analyze_v4.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "PHASE 1" in result.stdout
        assert "PHASE 2" in result.stdout
        assert "PHASE 3" in result.stdout

        # Verify outputs
        summary = json.loads((analysis_out / "summary.json").read_text())
        assert "phase1_behavioral" in summary
        assert "phase2_probes" in summary
        assert summary["phase1_behavioral"]["implicit"]["r2_on_z"] > 0.2, (
            "Synthetic data had strong z signal — probe should detect it"
        )
        print("[smoke] analyze_v4.py produces valid summary.json")
    finally:
        # Restore backups (best-effort; FUSE mounts sometimes refuse unlink)
        for d in [v4, analysis_out]:
            if d.exists():
                try:
                    shutil.rmtree(d)
                except PermissionError:
                    # FUSE quirk — leave the dir but it's gitignored
                    print(f"[smoke] warning: couldn't remove {d} (FUSE); gitignored anyway")
        for orig, bak in backups:
            if bak.exists() and not orig.exists():
                bak.rename(orig)


if __name__ == "__main__":
    test_end_to_end(None, None)
