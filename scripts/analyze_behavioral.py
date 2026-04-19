#!/usr/bin/env python3
"""Analyze behavioral results from LLM completions.

Reads JSON files from results/behavioral_v0/, classifies completions, computes
flip rates, and generates a markdown summary with H1 and H2 kill-test results.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def classify_completion(text: str) -> str:
    """Classify a completion into one of 5 categories.

    Args:
        text: The completion string

    Returns:
        One of: "tall", "short", "obese", "normal/average", "other"
    """
    text_lower = text.lower().strip()

    # Check for tall
    if "tall" in text_lower or "taller" in text_lower:
        return "tall"

    # Check for short
    if "short" in text_lower or "shorter" in text_lower:
        return "short"

    # Check for obese
    if "obese" in text_lower or "obesity" in text_lower or "medically obese" in text_lower:
        return "obese"

    # Check for normal/average
    if any(
        phrase in text_lower
        for phrase in ["normal", "average", "typical", "ordinary", "moderate"]
    ):
        return "normal/average"

    return "other"


def analyze_behavioral_results(results_dir: str) -> dict:
    """Analyze all JSON files in results/behavioral_v0/.

    Args:
        results_dir: Path to behavioral_v0 directory

    Returns:
        Dictionary with analysis results
    """
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("*.json"))

    # Structure: {prompt_id: {category: count, ...}}
    flip_data = defaultdict(lambda: defaultdict(int))
    completions_by_id = {}
    metadata_by_id = {}

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        prompt_id = data["id"]
        completions = data["completions"]
        n_completions = len(completions)

        # Classify each completion
        categories = [classify_completion(c) for c in completions]
        for cat in categories:
            flip_data[prompt_id][cat] += 1

        # Store most common completion
        most_common = max(set(completions), key=completions.count)
        completions_by_id[prompt_id] = most_common

        # Store metadata
        metadata_by_id[prompt_id] = {
            "adjective": data.get("adjective", ""),
            "adjective_class": data.get("adjective_class", ""),
            "context": data.get("context", ""),
            "context_mu": data.get("context_mu", data.get("context_mu_bmi", None)),
            "target_val": data.get("target_val_cm", data.get("target_val_bmi", None)),
            "n_samples": n_completions,
        }

    return {
        "flip_data": dict(flip_data),
        "completions": completions_by_id,
        "metadata": metadata_by_id,
    }


def compute_flip_rates(flip_data: dict) -> dict:
    """Compute fraction of each category per prompt.

    Args:
        flip_data: flip_data from analyze_behavioral_results()

    Returns:
        Dictionary: prompt_id -> {category: fraction, ...}
    """
    flip_rates = {}
    for prompt_id, counts in flip_data.items():
        total = sum(counts.values())
        rates = {cat: cnt / total for cat, cnt in counts.items()}
        flip_rates[prompt_id] = rates
    return flip_rates


def check_h1_kill_test(flip_rates: dict, metadata: dict) -> tuple:
    """Check H1: tall/165cm with narrow-LOW should have ≥60% "tall", narrow-HIGH ≥60% "short".

    Args:
        flip_rates: Flip rates per prompt
        metadata: Metadata per prompt

    Returns:
        (passed: bool, details: str)
    """
    # Filter for tall_165 in narrow-low and narrow-high
    tall_165_low = []
    tall_165_high = []

    for prompt_id in flip_rates:
        meta = metadata[prompt_id]
        if meta["adjective"] == "tall" and meta["target_val"] == 165:
            if meta["context"] == "narrow_low":
                rate_tall = flip_rates[prompt_id].get("tall", 0)
                tall_165_low.append(rate_tall)
            elif meta["context"] == "narrow_high":
                rate_short = flip_rates[prompt_id].get("short", 0)
                tall_165_high.append(rate_short)

    if not tall_165_low or not tall_165_high:
        return False, "Insufficient data for H1 test"

    avg_low = sum(tall_165_low) / len(tall_165_low)
    avg_high = sum(tall_165_high) / len(tall_165_high)

    passed = avg_low >= 0.6 and avg_high >= 0.6
    return passed, f"narrow-LOW tall: {avg_low:.1%}, narrow-HIGH short: {avg_high:.1%}"


def check_h2_control(flip_rates: dict, metadata: dict) -> tuple:
    """Check H2: obese/BMI=32 should have ≥60% "obese" in all non-OOD contexts.

    Args:
        flip_rates: Flip rates per prompt
        metadata: Metadata per prompt

    Returns:
        (passed: bool, details: str)
    """
    obese_rates = defaultdict(list)

    for prompt_id in flip_rates:
        meta = metadata[prompt_id]
        if meta["adjective"] == "obese" and meta["target_val"] == 32:
            if meta["context"] in ["narrow_low", "narrow_high", "wide_symmetric"]:
                rate_obese = flip_rates[prompt_id].get("obese", 0)
                obese_rates[meta["context"]].append(rate_obese)

    if not obese_rates:
        return False, "Insufficient obese data"

    results = {}
    for context in ["narrow_low", "narrow_high", "wide_symmetric"]:
        if context in obese_rates:
            avg = sum(obese_rates[context]) / len(obese_rates[context])
            results[context] = avg

    passed = all(rate >= 0.6 for rate in results.values())
    details = ", ".join(f"{ctx}: {rate:.1%}" for ctx, rate in results.items())
    return passed, details


def write_markdown_summary(
    out_path: str,
    flip_rates: dict,
    metadata: dict,
    completions: dict,
    h1_passed: bool,
    h1_details: str,
    h2_passed: bool,
    h2_details: str,
) -> None:
    """Write markdown summary with tables and kill-test results.

    Args:
        out_path: Output markdown file path
        flip_rates: Flip rates per prompt
        metadata: Metadata per prompt
        completions: Most common completion per prompt
        h1_passed: H1 test result
        h1_details: H1 test details string
        h2_passed: H2 test result
        h2_details: H2 test details string
    """
    lines = []

    # Header
    lines.append("# Behavioral Analysis Summary")
    lines.append("")

    # H1 Kill Test Result
    lines.append("## DAY 1 KILL-TEST RESULT (H1)")
    lines.append("")
    if h1_passed:
        lines.append(f"**PASSED** ✓ {h1_details}")
    else:
        lines.append(f"**FAILED** ✗ {h1_details}")
    lines.append("")

    # H2 Control
    lines.append("## H2 CONTROL (obese)")
    lines.append("")
    if h2_passed:
        lines.append(f"**PASSED** ✓ {h2_details}")
    else:
        lines.append(f"**FAILED** ✗ {h2_details}")
    lines.append("")

    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| prompt_id | n_completions | most_common_completion | fraction_relative_flip |")
    lines.append("|-----------|---------------|------------------------|------------------------|")

    # Sort for readability
    sorted_ids = sorted(flip_rates.keys())
    for prompt_id in sorted_ids:
        n_comp = metadata[prompt_id]["n_samples"]
        most_common = completions[prompt_id]
        fraction_rel = flip_rates[prompt_id].get("tall", 0) + flip_rates[prompt_id].get(
            "short", 0
        )

        lines.append(
            f"| {prompt_id} | {n_comp} | {most_common} | {fraction_rel:.1%} |"
        )

    lines.append("")

    # Write file
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    # Analyze results
    results_dir = "results/behavioral_v0"
    analysis = analyze_behavioral_results(results_dir)

    flip_data = analysis["flip_data"]
    completions = analysis["completions"]
    metadata = analysis["metadata"]

    flip_rates = compute_flip_rates(flip_data)

    # Run kill tests
    h1_passed, h1_details = check_h1_kill_test(flip_rates, metadata)
    h2_passed, h2_details = check_h2_control(flip_rates, metadata)

    # Write summary
    output_path = "results/behavioral_v0_summary.md"
    write_markdown_summary(
        output_path,
        flip_rates,
        metadata,
        completions,
        h1_passed,
        h1_details,
        h2_passed,
        h2_details,
    )

    print(f"Summary written to {output_path}")
    print(f"H1 (tall/short flip): {'PASSED' if h1_passed else 'FAILED'}")
    print(f"H2 (obese control): {'PASSED' if h2_passed else 'FAILED'}")
