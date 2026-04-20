"""Smoke test for extract_v4_adjpairs.py prompt generation (no model load).

Verifies:
  1. All 8 pairs generate correctly (7 relative + 1 absolute control).
  2. Per-pair prompt counts: 5*5*30 implicit + 5*5 explicit + 5 zero_shot = 780.
  3. Grand total = 6240.
  4. No prompts have unfilled {placeholder} strings.
  5. Each prompt ends with a recognizable completion cue (e.g. " is").
  6. Implicit prompts actually vary across seeds (non-trivial sampling).
  7. Each pair's context numbers fall near the requested mu.
"""
from __future__ import annotations

import re
import sys
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))

import extract_v4_adjpairs as ev


def test_pair_count():
    assert len(ev.PAIRS) == 8, f"expected 8 pairs, got {len(ev.PAIRS)}"
    names = [p.name for p in ev.PAIRS]
    expected = {"height", "age", "weight", "size", "speed", "wealth",
                "experience", "bmi_abs"}
    assert set(names) == expected, f"pair names mismatch: {set(names)}"


def test_counts_per_pair_and_total():
    trials = ev.generate_all_prompts()
    assert len(trials) == 6240, f"expected 6240 total trials, got {len(trials)}"

    for pair in ev.PAIRS:
        imp = [t for t in trials if t["pair"] == pair.name and t["condition"] == "implicit"]
        exp = [t for t in trials if t["pair"] == pair.name and t["condition"] == "explicit"]
        zs  = [t for t in trials if t["pair"] == pair.name and t["condition"] == "zero_shot"]
        assert len(imp) == 5 * 5 * ev.N_SEEDS_IMPLICIT, (
            f"{pair.name} implicit: got {len(imp)}")
        assert len(exp) == 5 * 5, f"{pair.name} explicit: got {len(exp)}"
        assert len(zs)  == 5, f"{pair.name} zero_shot: got {len(zs)}"


def test_no_unfilled_placeholders():
    """Every rendered prompt should be fully .format-ed — no '{...}' left over."""
    trials = ev.generate_all_prompts()
    placeholder_re = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
    for t in trials[:400]:  # sample for speed; all pairs × several cells covered
        hits = placeholder_re.findall(t["prompt"])
        assert not hits, f"{t['id']} unfilled placeholders {hits}: {t['prompt'][:140]}"


def test_prompts_end_with_is_or_equivalent():
    """Every prompt should end with 'is' (so the next predicted token is the adjective)."""
    trials = ev.generate_all_prompts()
    tails = set()
    for t in trials:
        tail = t["prompt"].rstrip().split()[-1]
        tails.add(tail)
    # All our prompts end with "is" by construction
    assert tails == {"is"}, f"unexpected prompt endings: {tails}"


def test_implicit_sampling_varies_across_seeds():
    """Two different seeds for the same (pair, x, mu) must produce different prompts."""
    for pair in ev.PAIRS:
        x = pair.target_values[0]
        mu = pair.mu_values[0]
        p0 = ev.make_implicit_prompt(pair, x, mu, seed=0)
        p1 = ev.make_implicit_prompt(pair, x, mu, seed=1)
        assert p0 != p1, f"{pair.name}: seed 0 and 1 produced identical prompts"


def test_implicit_sampling_mean_near_mu():
    """Across many seeds, the mean of the sampled context values should be near mu
    (within ~3*sigma / sqrt(n*15) — loose, just to catch gross bugs).

    Lines look like 'Person 7: 165 cm' or 'Worker 3: 12 years experience' or
    'Person 12 earns $75000/year'. The *last* number on the line is always the
    value we care about.
    """
    number_re = re.compile(r"(-?\d+(?:\.\d+)?)")
    for pair in ev.PAIRS:
        mu = pair.mu_values[len(pair.mu_values) // 2]  # middle value
        all_vals = []
        for s in range(30):
            items = ev.build_implicit_items(pair, mu, seed=s, n=15)
            for line in items:
                nums = number_re.findall(line)
                if nums:
                    # Last number = value; first is typically the index.
                    all_vals.append(float(nums[-1]))
        assert len(all_vals) > 200, f"{pair.name}: too few sampled values ({len(all_vals)})"
        sample_mean = statistics.mean(all_vals)
        # Allow a generous slack: clipping effects + SE of the mean.
        slack = 3 * pair.sigma / (len(all_vals) ** 0.5) + 0.15 * abs(mu) + pair.sigma * 0.5
        assert abs(sample_mean - mu) < slack, (
            f"{pair.name}: implicit sample mean {sample_mean:.2f} far from mu {mu:.2f} "
            f"(slack={slack:.2f})")


def test_bmi_abs_prompts_look_like_bmi():
    """Spot-check: BMI implicit prompts should contain 'BMI' and no 'Person 1: 22 cm' etc."""
    trials = ev.generate_all_prompts()
    bmi_trials = [t for t in trials if t["pair"] == "bmi_abs" and t["condition"] == "implicit"]
    assert bmi_trials, "no bmi_abs implicit trials"
    for t in bmi_trials[:5]:
        p = t["prompt"]
        assert "BMI" in p, f"bmi_abs prompt missing 'BMI': {p[:200]}"
        assert " cm" not in p, f"bmi_abs leaked 'cm' unit: {p[:200]}"
        assert " kg" not in p, f"bmi_abs leaked 'kg' unit: {p[:200]}"


def test_wealth_formatting():
    """Wealth prompts should not have scientific notation or trailing '.0' on integers."""
    pair = next(p for p in ev.PAIRS if p.name == "wealth")
    p = ev.make_implicit_prompt(pair, x=50000.0, mu=60000.0, seed=0)
    assert "50000" in p
    # no scientific notation
    assert "e+" not in p.lower()
    # explicit prompt check too
    pe = ev.make_explicit_prompt(pair, x=80000.0, mu=60000.0)
    assert "$80000" in pe
    assert "$60000" in pe


def test_explicit_and_zero_shot_are_deterministic():
    pair = ev.PAIRS[0]  # height
    a = ev.make_explicit_prompt(pair, x=170.0, mu=165.0)
    b = ev.make_explicit_prompt(pair, x=170.0, mu=165.0)
    assert a == b

    c = ev.make_zero_shot_prompt(pair, x=170.0)
    d = ev.make_zero_shot_prompt(pair, x=170.0)
    assert c == d


if __name__ == "__main__":
    # Allow running this file directly without pytest for quick iteration.
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"[ok] {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, e))
            print(f"[FAIL] {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} failed")
        sys.exit(1)
    else:
        print(f"\n{len(tests)}/{len(tests)} passed")
