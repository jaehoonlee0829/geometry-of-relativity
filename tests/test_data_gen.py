"""Tests for prompt generation via generate_trials.

Verifies that the data generation pipeline correctly:
1. Generates sufficient trials
2. Assigns correct adjective_class for each trial
3. Produces non-empty prompt strings
4. Includes context-aware numerical mentions in prompts
"""

import pytest
from src.data_gen import generate_trials, canonical_v0_contexts, z_score


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
