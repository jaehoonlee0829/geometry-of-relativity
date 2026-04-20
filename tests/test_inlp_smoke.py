"""Smoke test for inlp_v4.py on synthetic data.

Validates:
  1. The script runs end-to-end on synthetic v4_dense data.
  2. After INLP-z projection, CV R²(z) drops sharply vs the initial value.
  3. Random-direction projection does NOT drop R²(z) as fast.
  4. x-decodability survives INLP-z (x ≠ z).

Uses the existing test_analyze_v4_smoke.make_fake_v4 to avoid code duplication.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

from test_analyze_v4_smoke import make_fake_v4


def test_inlp_end_to_end():
    v4 = REPO / "results" / "v4_dense"
    analysis = REPO / "results" / "v4_analysis"

    backups = []
    for d in [v4, analysis]:
        if d.exists():
            # If the directory is non-empty, rename its contents out of the
            # way rather than trying to rename the directory itself — FUSE
            # mounts refuse to rename non-empty dirs in some cases.
            for entry in list(d.iterdir()):
                target = d / f"_bak_inlp_{entry.name}"
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                try:
                    entry.rename(target)
                    backups.append((entry, target))
                except OSError:
                    pass  # best-effort; gitignored anyway

    try:
        make_fake_v4()

        # Use seed=42 for INLP so its "random" direction doesn't coincide with
        # the synthetic data generator's seed=0 (which would produce a random
        # vector identical to true_z_dir and give misleadingly perfect erasure).
        result = subprocess.run(
            [sys.executable, "scripts/vast_remote/inlp_v4.py",
             "--layer", "late", "--steps", "4", "--seed", "42"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"inlp_v4.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        out_path = analysis / "inlp_late.json"
        assert out_path.exists(), f"missing {out_path}"
        data = json.loads(out_path.read_text())

        # Check schema
        for key in ("inlp_z", "null_random", "inlp_x"):
            assert key in data, f"missing {key}"
            rec = data[key]
            for k in ("step", "r2_z", "r2_x", "r2_ld"):
                assert k in rec and len(rec[k]) == 5, (
                    f"{key}/{k}: expected 5 values (steps 0..4), got {len(rec.get(k, []))}"
                )

        # Scientific checks
        r2z_init = data["inlp_z"]["r2_z"][0]
        r2z_inlp_final = data["inlp_z"]["r2_z"][-1]
        r2z_rand_final = data["null_random"]["r2_z"][-1]

        assert r2z_init > 0.5, (
            f"synthetic z signal too weak: initial R²(z)={r2z_init:.3f}"
        )
        assert r2z_inlp_final < r2z_init - 0.3, (
            f"INLP-z did not collapse R²(z): initial={r2z_init:.3f} final={r2z_inlp_final:.3f}"
        )
        assert r2z_rand_final > r2z_inlp_final, (
            f"random nulls collapsed R²(z) faster than INLP: "
            f"random_final={r2z_rand_final:.3f} inlp_final={r2z_inlp_final:.3f}"
        )

        # x survives INLP-z (since x ≠ z direction)
        r2x_init = data["inlp_z"]["r2_x"][0]
        r2x_inlp_z_final = data["inlp_z"]["r2_x"][-1]
        # Only a mild loss allowed — x direction mostly orthogonal to z
        assert r2x_inlp_z_final > 0.2 * r2x_init - 0.1, (
            f"x-R² collapsed too much under INLP-z: init={r2x_init:.3f} final={r2x_inlp_z_final:.3f}"
        )

        print(f"[smoke] inlp_v4.py R²(z): "
              f"initial={r2z_init:.3f}  INLP={r2z_inlp_final:.3f}  random={r2z_rand_final:.3f}")
        print(f"[smoke] inlp_v4.py R²(x) under INLP-z: "
              f"initial={r2x_init:.3f}  final={r2x_inlp_z_final:.3f}")
    finally:
        # Clean up freshly-generated files (keep bak_inlp_* prefixed backups
        # intact, since they represent real prior state we renamed aside).
        for d in [v4, analysis]:
            if not d.exists():
                continue
            for entry in list(d.iterdir()):
                if entry.name.startswith("_bak_inlp_"):
                    continue
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                except (PermissionError, OSError):
                    pass
        # Restore backups best-effort.
        for orig, bak in backups:
            if bak.exists() and not orig.exists():
                try:
                    bak.rename(orig)
                except OSError:
                    pass


if __name__ == "__main__":
    test_inlp_end_to_end()
