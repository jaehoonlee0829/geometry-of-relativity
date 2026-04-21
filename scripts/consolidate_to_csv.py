"""Consolidate per-experiment JSONL (trials + per-prompt logit_diff) into CSV.

JSONL files are gitignored + live on HF (see scripts/fetch_from_hf.py). But
reviewers benefit from having one small CSV per experiment checked into git
for quick grep / GitHub-table rendering / diff-ability across future reruns.

Outputs (committed to git):
  results/csv/v4_adjpairs_e4b.csv        # 6240 rows — 8 pairs × 3 conds
  results/csv/v4_adjpairs_g31b.csv       # 6000 rows — 8 pairs × 1 cond (implicit)
  results/csv/v4_dense_e4b.csv           # 3580 rows — tall/short dense + expl + zs
  results/csv/v4_abs_controls_e4b.csv    # 2340 rows — 3 new absolute pairs
  results/csv/v4_zeroshot_expanded_e4b.csv  # 1200 rows — 30 phrasing seeds

Schema per CSV (one row = one prompt):
  id, pair, condition, x, mu, z, sigma, seed, low_word, high_word, logit_diff
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "csv"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = ["id", "pair", "condition", "x", "mu", "z", "sigma", "seed",
          "low_word", "high_word", "logit_diff"]


def iter_trials_with_logits(trial_path: Path, logit_paths: list[Path]):
    """Zip a trials.jsonl with one or more *_logits.jsonl files by id."""
    trials = {}
    for line in trial_path.open():
        t = json.loads(line)
        trials[t["id"]] = t
    for lp in logit_paths:
        for line in lp.open():
            r = json.loads(line)
            t = trials.get(r["id"])
            if t is None:
                continue
            yield {
                "id": t["id"],
                "pair": t.get("pair", ""),
                "condition": t.get("condition", ""),
                "x": t.get("x", ""),
                "mu": t.get("mu", ""),
                "z": t.get("z", ""),
                "sigma": t.get("sigma", ""),
                "seed": t.get("seed", ""),
                "low_word": t.get("low_word", ""),
                "high_word": t.get("high_word", ""),
                "logit_diff": r["logit_diff"],
            }


def write_csv(out: Path, rows) -> int:
    n = 0
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def main() -> None:
    base = REPO / "results"

    # 1) v4_adjpairs E4B — 8 pairs × 3 conds = 24 logit jsonls, 1 trials jsonl
    src = base / "v4_adjpairs"
    logits = sorted(src.glob("e4b_*_logits.jsonl"))
    trials = src / "e4b_trials.jsonl"
    n = write_csv(OUT / "v4_adjpairs_e4b.csv",
                  iter_trials_with_logits(trials, logits))
    print(f"  v4_adjpairs_e4b.csv:         {n} rows")

    # 2) v4_adjpairs G31B — implicit only
    logits = sorted(src.glob("g31b_*_logits.jsonl"))
    trials = src / "g31b_trials.jsonl"
    n = write_csv(OUT / "v4_adjpairs_g31b.csv",
                  iter_trials_with_logits(trials, logits))
    print(f"  v4_adjpairs_g31b.csv:        {n} rows")

    # 3) v4_dense — 3 logit jsonls + 1 trials
    src = base / "v4_dense"
    logits = sorted(src.glob("e4b_*_logits.jsonl"))
    trials = src / "e4b_trials.jsonl"
    n = write_csv(OUT / "v4_dense_e4b.csv",
                  iter_trials_with_logits(trials, logits))
    print(f"  v4_dense_e4b.csv:            {n} rows")

    # 4) v4_abs_controls — 3 pairs × 3 conds = 9 logit jsonls
    src = base / "v4_abs_controls"
    logits = sorted(src.glob("e4b_*_logits.jsonl"))
    trials = src / "e4b_trials.jsonl"
    n = write_csv(OUT / "v4_abs_controls_e4b.csv",
                  iter_trials_with_logits(trials, logits))
    print(f"  v4_abs_controls_e4b.csv:     {n} rows")

    # 5) v4_zeroshot_expanded — 8 pairs, 1 condition (zero-shot-style)
    src = base / "v4_zeroshot_expanded"
    logits = sorted(src.glob("e4b_*_logits.jsonl"))
    trials = src / "e4b_trials.jsonl"
    n = write_csv(OUT / "v4_zeroshot_expanded_e4b.csv",
                  iter_trials_with_logits(trials, logits))
    print(f"  v4_zeroshot_expanded_e4b.csv: {n} rows")


if __name__ == "__main__":
    main()
