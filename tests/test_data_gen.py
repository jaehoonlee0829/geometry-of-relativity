"""Tests for prompt generation via generate_trials (v0/v1) and generate_trials_v2 (v2).

Verifies that the data generation pipeline correctly:
1. Generates sufficient trials
2. Assigns correct adjective_class for each trial
3. Produces non-empty prompt strings
4. Includes context-aware numerical mentions in prompts
"""

import math
import pytest
from src.data_gen import (
    generate_trials,
    generate_trials_v2,
    canonical_v0_contexts,
    z_score,
    HEIGHT_SPEC,
    WEALTH_SPEC,
    DOMAIN_SPECS,
    trial_v2_to_dict,
)


class TestDataGeneration:
    """Verify prompt generation via generate_trials."""

    def test_generate_trials_count(self):
        """Verify that generate_trials produces at least 40 trials."""
        trials = generate_trials(
            adjectives=["tall", "obese", "short"],
            target_values=[155, 165, 175, 32, 3, 10, 300],
            contexts=canonical_v0_contexts(),
            templates_per_context=2,
            seed=0,
        )

        assert len(trials) >= 40, \
            f"Generated {len(trials)} trials, expected >= 40"

    def test_adjective_class_valid(self):
        """Verify all trials have valid adjective_class."""
        trials = generate_trials(
            adjectives=["tall", "obese", "short"],
            target_values=[155, 165, 175, 32, 3, 10, 300],
            contexts=canonical_v0_contexts(),
            templates_per_context=2,
            seed=0,
        )

        valid_classes = {"relative", "absolute"}

        for trial in trials:
            assert trial.adjective_class in valid_classes, \
                f"Trial {trial} has invalid adjective_class: {trial.adjective_class}"

    def test_prompt_non_empty(self):
        """Verify all trials have non-empty prompt strings."""
        trials = generate_trials(
            adjectives=["tall", "obese", "short"],
            target_values=[155, 165, 175, 32, 3, 10, 300],
            contexts=canonical_v0_contexts(),
            templates_per_context=2,
            seed=0,
        )

        for trial in trials:
            assert isinstance(trial.prompt, str), \
                f"Trial {trial} has non-string prompt"
            assert len(trial.prompt) > 0, \
                f"Trial {trial} has empty prompt"

    def test_tall_adjective_relative(self):
        """Verify 'tall' is classified as relative."""
        trials = generate_trials(
            adjectives=["tall"],
            target_values=[165],
            contexts=canonical_v0_contexts(),
            templates_per_context=1,
            seed=0,
        )

        for trial in trials:
            if trial.adjective == "tall":
                assert trial.adjective_class == "relative", \
                    f"tall should be relative, got {trial.adjective_class}"

    def test_obese_adjective_absolute(self):
        """Verify 'obese' is classified as absolute."""
        trials = generate_trials(
            adjectives=["obese"],
            target_values=[30],
            contexts=canonical_v0_contexts(),
            templates_per_context=1,
            seed=0,
        )

        for trial in trials:
            if trial.adjective == "obese":
                assert trial.adjective_class == "absolute", \
                    f"obese should be absolute, got {trial.adjective_class}"

    def test_tall_narrow_low_context_mentions_165(self):
        """Verify that 'tall' @ 165 in narrow_low generates prompts mentioning 165 and 150."""
        trials = generate_trials(
            adjectives=["tall"],
            target_values=[155, 165, 175],
            contexts=canonical_v0_contexts(),
            templates_per_context=3,
            seed=0,
        )

        # Filter for tall, target=165, narrow_low context
        relevant_trials = [
            t for t in trials
            if t.adjective == "tall"
            and t.target_value == 165
            and t.context.name == "narrow_low"
        ]

        assert len(relevant_trials) > 0, \
            "No trials matching tall/165/narrow_low"

        # Collect all prompts for this cell
        prompts = {t.prompt for t in relevant_trials}

        # Check: at least one prompt mentions "165"
        mentions_165 = any("165" in p for p in prompts)
        assert mentions_165, \
            f"No prompt mentions target value 165. Prompts:\n" + \
            "\n".join(prompts)

        # Check: at least one prompt mentions "150" (narrow_low context mean - sigma)
        mentions_150 = any("150" in p for p in prompts)
        assert mentions_150, \
            f"No prompt mentions narrow_low boundary 150. Prompts:\n" + \
            "\n".join(prompts)

    def test_tall_narrow_high_context(self):
        """Verify 'tall' @ 175 in narrow_high has appropriate context."""
        trials = generate_trials(
            adjectives=["tall"],
            target_values=[175],
            contexts=canonical_v0_contexts(),
            templates_per_context=3,
            seed=0,
        )

        relevant_trials = [
            t for t in trials
            if t.adjective == "tall"
            and t.target_value == 175
            and t.context.name == "narrow_high"
        ]

        assert len(relevant_trials) > 0, \
            "No trials matching tall/175/narrow_high"

        for trial in relevant_trials:
            assert "175" in trial.prompt, \
                f"Prompt doesn't mention target value 175: {trial.prompt}"

    def test_z_score_computation(self):
        """Verify z_score is computed correctly for trials."""
        trials = generate_trials(
            adjectives=["tall"],
            target_values=[165],
            contexts=canonical_v0_contexts(),
            templates_per_context=1,
            seed=0,
        )

        for trial in trials:
            z = z_score(trial)
            expected_z = (trial.target_value - trial.context.mu) / trial.context.sigma
            assert abs(z - expected_z) < 1e-12, \
                f"Z-score mismatch for {trial}: {z} != {expected_z}"

    def test_seed_reproducibility(self):
        """Verify that same seed produces same trials."""
        trials1 = generate_trials(
            adjectives=["tall", "short"],
            target_values=[165, 175],
            contexts=canonical_v0_contexts(),
            templates_per_context=2,
            seed=42,
        )

        trials2 = generate_trials(
            adjectives=["tall", "short"],
            target_values=[165, 175],
            contexts=canonical_v0_contexts(),
            templates_per_context=2,
            seed=42,
        )

        assert len(trials1) == len(trials2), \
            f"Different lengths: {len(trials1)} vs {len(trials2)}"

        for t1, t2 in zip(trials1, trials2):
            assert t1.prompt == t2.prompt, \
                f"Different prompts at same seed: {t1.prompt} vs {t2.prompt}"
            assert t1.adjective == t2.adjective, \
                "Different adjectives at same seed"
            assert t1.target_value == t2.target_value, \
                "Different target values at same seed"

    def test_templates_per_context(self):
        """Verify templates_per_context parameter controls trial count."""
        # With 1 adjective, 1 value, 4 contexts, templates_per_context=1:
        # should generate at least 4 trials (one template per context)
        trials1 = generate_trials(
            adjectives=["tall"],
            target_values=[165],
            contexts=canonical_v0_contexts(),
            templates_per_context=1,
            seed=0,
        )

        # With templates_per_context=3:
        # should generate up to 3× as many trials
        trials3 = generate_trials(
            adjectives=["tall"],
            target_values=[165],
            contexts=canonical_v0_contexts(),
            templates_per_context=3,
            seed=0,
        )

        assert len(trials1) >= 4, \
            f"templates_per_context=1 should generate >= 4 trials, got {len(trials1)}"
        assert len(trials3) > len(trials1), \
            f"templates_per_context=3 should generate more than =1, got {len(trials3)} vs {len(trials1)}"

    def test_unknown_adjective_raises(self):
        """Verify that unknown adjectives raise ValueError."""
        with pytest.raises(ValueError, match="Unknown adjective"):
            generate_trials(
                adjectives=["unknown_adjective"],
                target_values=[165],
                contexts=canonical_v0_contexts(),
                templates_per_context=1,
                seed=0,
            )

    def test_canonical_v0_contexts(self):
        """Verify canonical_v0_contexts returns expected structure."""
        contexts = canonical_v0_contexts()

        assert len(contexts) == 4, \
            f"Expected 4 contexts, got {len(contexts)}"

        names = {c.name for c in contexts}
        expected_names = {"narrow_low", "narrow_high", "wide_symmetric", "ood_contaminated"}
        assert names == expected_names, \
            f"Context names mismatch: {names} != {expected_names}"

        # Verify narrow_low and narrow_high have different means
        narrow_low = next(c for c in contexts if c.name == "narrow_low")
        narrow_high = next(c for c in contexts if c.name == "narrow_high")
        assert narrow_low.mu != narrow_high.mu, \
            "narrow_low and narrow_high should have different means"

    def test_obese_context_integration(self):
        """Verify obese works correctly in a full pipeline."""
        trials = generate_trials(
            adjectives=["obese"],
            target_values=[25, 30, 35],
            contexts=canonical_v0_contexts(),
            templates_per_context=1,
            seed=0,
        )

        assert len(trials) > 0, \
            "Should generate trials for obese"

        for trial in trials:
            assert trial.adjective == "obese", \
                f"Wrong adjective: {trial.adjective}"
            assert trial.adjective_class == "absolute", \
                f"obese should be absolute, got {trial.adjective_class}"
            assert isinstance(trial.prompt, str) and len(trial.prompt) > 0, \
                f"Invalid prompt: {trial.prompt}"


class TestGenerateTrialsV2:
    """Verify v2 prompt generator (PLANNING.md spec v2)."""

    def test_height_default_count(self):
        """Default height battery = 2 ctx × 2 frames × 7 x × 9 mu = 252."""
        trials = generate_trials_v2(HEIGHT_SPEC)
        assert len(trials) == 252, f"Expected 252, got {len(trials)}"

    def test_wealth_default_count(self):
        """Default wealth battery = 2 ctx × 2 frames × 7 x × 7 mu = 196."""
        trials = generate_trials_v2(WEALTH_SPEC)
        assert len(trials) == 196, f"Expected 196, got {len(trials)}"

    def test_deterministic_across_runs(self):
        """Same spec → identical trial_id, prompt, context_sample, z across runs."""
        t1 = generate_trials_v2(HEIGHT_SPEC)
        t2 = generate_trials_v2(HEIGHT_SPEC)
        for a, b in zip(t1, t2):
            assert a.trial_id == b.trial_id
            assert a.prompt == b.prompt
            assert a.context_sample == b.context_sample
            assert a.z == b.z

    def test_prompt_ends_at_is_or_considered(self):
        """Every v2 prompt must end at exactly 'is' or 'considered' (no trailing space)."""
        for trial in generate_trials_v2(HEIGHT_SPEC) + generate_trials_v2(WEALTH_SPEC):
            assert trial.prompt.endswith("is") or trial.prompt.endswith("considered"), \
                f"Prompt does not end at is/considered: {trial.prompt[-30:]!r}"
            if trial.prompt_frame == "is":
                assert trial.prompt.endswith(" is"), \
                    f"is-frame should end with ' is': {trial.prompt[-10:]!r}"
            else:
                assert trial.prompt.endswith("is considered"), \
                    f"considered-frame should end with 'is considered': {trial.prompt[-20:]!r}"

    def test_z_score_correctness_height(self):
        """Linear z = (x - mu) / sigma for height."""
        for trial in generate_trials_v2(HEIGHT_SPEC):
            expected = (trial.x - trial.mu) / trial.sigma
            assert abs(trial.z - expected) < 1e-9, \
                f"z mismatch: {trial.z} vs {expected}"

    def test_z_score_correctness_wealth_log_space(self):
        """Log z = (log x - log mu) / log sigma for wealth."""
        for trial in generate_trials_v2(WEALTH_SPEC):
            expected = (math.log(trial.x) - math.log(trial.mu)) / math.log(trial.sigma)
            assert abs(trial.z - expected) < 1e-9, \
                f"log-z mismatch: {trial.z} vs {expected}"

    def test_x_mu_decorrelated(self):
        """At fixed z, raw x spans multiple values — decorrelates x from z.

        This is the whole point of v2: at z=0 (x==mu), we have 7 different x values
        (one per target). At z=+1 and z=-1 we also have multiple x values.
        """
        trials = generate_trials_v2(HEIGHT_SPEC, context_types=("explicit",), prompt_frames=("is",))
        # Group by rounded z
        from collections import defaultdict
        by_z = defaultdict(set)
        for t in trials:
            by_z[round(t.z, 3)].add(t.x)
        # At least one z-bucket should contain multiple x values.
        max_x_count = max(len(xs) for xs in by_z.values())
        assert max_x_count >= 2, \
            f"All z values map to unique x values — not decorrelated. by_z={dict(by_z)}"

    def test_implicit_context_is_deterministic_per_mu(self):
        """All trials with the same (domain, mu, sigma) share the same context_sample."""
        trials = generate_trials_v2(HEIGHT_SPEC, context_types=("implicit",))
        # Group by mu
        samples_by_mu: dict[float, tuple] = {}
        for t in trials:
            if t.mu in samples_by_mu:
                assert samples_by_mu[t.mu] == t.context_sample, \
                    f"mu={t.mu} has different context samples across trials"
            else:
                samples_by_mu[t.mu] = t.context_sample

    def test_explicit_context_has_no_sample(self):
        """Explicit-context trials have empty context_sample tuple."""
        for t in generate_trials_v2(HEIGHT_SPEC, context_types=("explicit",)):
            assert t.context_sample == (), f"explicit trial has sample: {t.context_sample}"

    def test_implicit_context_has_15_samples(self):
        """Implicit-context trials have exactly 15 context samples (default)."""
        for t in generate_trials_v2(HEIGHT_SPEC, context_types=("implicit",)):
            assert len(t.context_sample) == 15, \
                f"Expected 15 samples, got {len(t.context_sample)}"

    def test_height_prompts_contain_target_and_context(self):
        """Height prompts must mention the target x value and (for explicit) the mu."""
        trials = generate_trials_v2(
            HEIGHT_SPEC,
            context_types=("explicit",),
            prompt_frames=("is",),
        )
        for t in trials:
            assert f"{int(round(t.x))} cm" in t.prompt
            assert f"{int(round(t.mu))} cm" in t.prompt

    def test_wealth_prompts_contain_target(self):
        """Wealth prompts must mention the target x value formatted with $ or M."""
        trials = generate_trials_v2(WEALTH_SPEC)
        for t in trials:
            # Either dollar amount like $20,000 or $5.0M appears
            has_dollar = "$" in t.prompt
            assert has_dollar, f"Wealth prompt missing $: {t.prompt[:80]!r}"

    def test_adjective_pair_stored(self):
        """Each trial carries the domain's (high, low) adjective pair."""
        for t in generate_trials_v2(HEIGHT_SPEC):
            assert t.adjective_high == "tall"
            assert t.adjective_low == "short"
        for t in generate_trials_v2(WEALTH_SPEC):
            assert t.adjective_high == "rich"
            assert t.adjective_low == "poor"

    def test_trial_v2_to_dict_round_trip(self):
        """trial_v2_to_dict produces JSON-serialisable output with all required keys."""
        import json
        trials = generate_trials_v2(HEIGHT_SPEC)
        for t in trials[:5]:
            d = trial_v2_to_dict(t)
            # Must be JSON serialisable
            s = json.dumps(d)
            d2 = json.loads(s)
            assert d2 == d
            # Required keys
            for k in [
                "id", "domain", "context_type", "prompt_frame",
                "x", "mu", "sigma", "z", "context_seed", "context_sample",
                "prompt", "adjective_high", "adjective_low",
            ]:
                assert k in d, f"Missing key {k}"

    def test_subset_axes(self):
        """Subsetting context_types or prompt_frames works."""
        t_implicit_is = generate_trials_v2(
            HEIGHT_SPEC, context_types=("implicit",), prompt_frames=("is",)
        )
        # 1 × 1 × 7 × 9 = 63
        assert len(t_implicit_is) == 63
        for t in t_implicit_is:
            assert t.context_type == "implicit"
            assert t.prompt_frame == "is"

    def test_custom_target_values_override(self):
        """Passing explicit target_values overrides spec defaults."""
        trials = generate_trials_v2(
            HEIGHT_SPEC,
            target_values=(160.0,),
            context_types=("explicit",),
            prompt_frames=("is",),
        )
        # 1 × 1 × 1 × 9 = 9
        assert len(trials) == 9
        for t in trials:
            assert t.x == 160.0

    def test_z_scores_span_both_signs(self):
        """The default spec should produce trials with both positive and negative z."""
        height_zs = [t.z for t in generate_trials_v2(HEIGHT_SPEC)]
        assert min(height_zs) < 0
        assert max(height_zs) > 0
        # Expect span ≥ 7 (since x spans 150-180 = 30cm, mu spans 145-185 = 40cm, sigma=10)
        assert max(height_zs) - min(height_zs) >= 7.0

    def test_domain_specs_registry(self):
        """Both canonical specs registered in DOMAIN_SPECS."""
        assert set(DOMAIN_SPECS.keys()) == {"height", "wealth"}
        assert DOMAIN_SPECS["height"] is HEIGHT_SPEC
        assert DOMAIN_SPECS["wealth"] is WEALTH_SPEC
