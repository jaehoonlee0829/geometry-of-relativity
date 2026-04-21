"""Exp 4d: zero-shot logit_diff bias check.

For each pair, report ld at x_min (objectively 'low') and x_max (objectively 'high').
If ld > 0 at x_min, there's a prior bias toward the 'high' word even when the value
is low — i.e. frequency / RLHF bias. For E4B (base model, NOT instruction-tuned)
this should be frequency-based if present.

Writes:
  results/v4_adjpairs_analysis/exp4d_zero_shot_bias.json
"""
import json
from pathlib import Path

ADJPAIRS = Path("results/v4_adjpairs")
OUT = Path("results/v4_adjpairs_analysis")
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = ["height", "age", "experience", "size", "speed", "wealth", "weight", "bmi_abs"]


def main() -> None:
    trials_by_id = {json.loads(l)["id"]: json.loads(l) for l in (ADJPAIRS / "e4b_trials.jsonl").open()}
    result = {"pairs": {}, "summary": {}}
    print(f"{'pair':15s} {'min_x':>10s} {'max_x':>10s}  {'ld@xmin':>9s} {'ld@xmax':>9s}  note")
    print("-" * 90)
    biased_count = 0
    flat_count = 0
    for pair in PAIRS:
        rows = []
        for rec in map(json.loads, (ADJPAIRS / f"e4b_{pair}_zero_shot_logits.jsonl").open()):
            t = trials_by_id[rec["id"]]
            rows.append({"x": t["x"], "ld": rec["logit_diff"], "low": t["low_word"], "high": t["high_word"]})
        rows.sort(key=lambda r: r["x"])
        lo, hi = rows[0], rows[-1]
        ld_range = hi["ld"] - lo["ld"]
        is_biased = lo["ld"] > 0  # ld > 0 at min-x means bias toward 'high' word
        is_flat = abs(ld_range) < 0.5  # zero-shot barely responds to x
        notes = []
        if is_biased: notes.append("bias→high")
        if is_flat: notes.append("x-insensitive")
        if ld_range > 3: notes.append("strong zero-shot prior")
        note = ", ".join(notes) if notes else "ok"
        print(f'{pair:15s} {lo["x"]:>10g} {hi["x"]:>10g}  {lo["ld"]:>9.2f} {hi["ld"]:>9.2f}  {note}')
        biased_count += int(is_biased)
        flat_count += int(is_flat)
        result["pairs"][pair] = {
            "low_word": lo["low"], "high_word": lo["high"],
            "x_min": lo["x"], "x_max": hi["x"],
            "ld_at_xmin": lo["ld"], "ld_at_xmax": hi["ld"],
            "ld_range": ld_range,
            "bias_toward_high": bool(is_biased),
            "x_insensitive": bool(is_flat),
        }
    result["summary"] = {
        "n_pairs": len(PAIRS),
        "n_biased_high": biased_count,
        "n_x_insensitive": flat_count,
        "interpretation": (
            "E4B is a BASE model; any bias here is most likely frequency-based rather "
            "than RLHF-politeness. Confirmed bias in: " +
            ", ".join(p for p in PAIRS if result["pairs"][p]["bias_toward_high"]) +
            "; x-insensitive: " + ", ".join(p for p in PAIRS if result["pairs"][p]["x_insensitive"])
        ),
        "synonym_family_fix_recommendation": (
            "Synonym family aggregation (4a) would help pairs where the tokenizer "
            "splits 'taller', 'tallest', etc. into different IDs. It does NOT fix "
            "x-insensitivity (size/speed/weight). Those pairs likely need prompt-design "
            "redesign or x-range widening."
        ),
    }
    (OUT / "exp4d_zero_shot_bias.json").write_text(json.dumps(result, indent=2))
    print()
    print(f"wrote {OUT/'exp4d_zero_shot_bias.json'}")
    print(f"SUMMARY: {biased_count}/{len(PAIRS)} pairs biased toward high-word at x_min; "
          f"{flat_count}/{len(PAIRS)} x-insensitive in zero-shot.")


if __name__ == "__main__":
    main()
