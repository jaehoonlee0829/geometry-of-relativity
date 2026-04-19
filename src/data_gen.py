"""Prompt generator for mech-interp study of gradable adjectives.

Takes structured (adjective, target_value, context) triples and emits natural-language
prompts for small LMs (Gemma-2-2b, Llama-3.2-3B) to complete.
"""

import random
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Context:
    """Context parameters for a gradable adjective trial.

    Attributes:
        name: Type of context distribution.
        mu: Context mean (cm for height, BMI units for weight).
        sigma: Context standard deviation.
        attribute: Whether this context measures height or BMI.
    """

    name: Literal["narrow_low", "narrow_high", "wide_symmetric", "ood_contaminated"]
    mu: float
    sigma: float
    attribute: Literal["height_cm", "bmi"]


@dataclass(frozen=True)
class Trial:
    """A single trial: adjective, target value, context, and prompt.

    Attributes:
        adjective: The gradable adjective ("tall", "obese", etc.).
        adjective_class: Whether the adjective is relative or absolute.
        target_value: The numerical value (cm or BMI) to evaluate.
        context: The context distribution.
        prompt: The natural-language prompt string.
        prompt_template_id: Which template was used (e.g., "narrow_low_1").
    """

    adjective: str
    adjective_class: Literal["relative", "absolute"]
    target_value: float
    context: Context
    prompt: str
    prompt_template_id: str


# Adjective classification: relative vs absolute
ADJECTIVE_CLASS_MAP = {
    "tall": "relative",
    "short": "relative",
    "heavy": "relative",
    "light": "relative",
    "young": "relative",
    "old": "relative",
    "obese": "absolute",
    "underweight": "absolute",
}


def canonical_v0_contexts() -> list[Context]:
    """Return the 4 contexts used in the v0 battery.

    Returns:
        List of 4 Context objects: narrow_low, narrow_high, wide_symmetric, ood_contaminated.
    """
    return [
        Context(
            name="narrow_low",
            mu=150,
            sigma=3,
            attribute="height_cm",
        ),
        Context(
            name="narrow_high",
            mu=180,
            sigma=3,
            attribute="height_cm",
        ),
        Context(
            name="wide_symmetric",
            mu=165,
            sigma=10,
            attribute="height_cm",
        ),
        Context(
            name="ood_contaminated",
            mu=165,
            sigma=10,
            attribute="height_cm",
        ),
    ]


def z_score(trial: Trial) -> float:
    """Compute standardized z-score for a trial.

    Args:
        trial: A Trial object.

    Returns:
        (target_value - context.mu) / context.sigma
    """
    return (trial.target_value - trial.context.mu) / trial.context.sigma


# Template dictionaries per context type.
# Keys are template IDs (e.g., "1", "2", "3"), values are f-string templates.

TEMPLATES_NARROW_LOW_HEIGHT = {
    "1": "Consider a group of people whose heights cluster tightly around {mu} cm (give or take {sigma} cm). In this group, a person who is {target} cm is considered",
    "2": "The people in this community are almost all between {mu_low} cm and {mu_high} cm. Relative to them, someone who is {target} cm tall would be described as",
    "3": "In a village where nearly everyone is about {mu} cm, a newcomer is {target} cm. The villagers describe the newcomer as",
}

TEMPLATES_NARROW_HIGH_HEIGHT = {
    "1": "Consider a group of people whose heights cluster tightly around {mu} cm (give or take {sigma} cm). In this group, a person who is {target} cm is considered",
    "2": "The people in this community are almost all between {mu_low} cm and {mu_high} cm. Relative to them, someone who is {target} cm tall would be described as",
    "3": "In a village where nearly everyone is about {mu} cm, a newcomer is {target} cm. The villagers describe the newcomer as",
}

TEMPLATES_WIDE_SYMMETRIC_HEIGHT = {
    "1": "In a population of adults with heights ranging roughly from {range_low} cm to {range_high} cm (mean around {mu} cm), a person who is {target} cm is considered",
    "2": "Across the general adult population, a person {target} cm tall is considered",
}

TEMPLATES_OOD_HEIGHT = {
    "1": "In a normal population of adult humans, a person who is {target} cm would be described as",
}

TEMPLATES_NARROW_LOW_BMI = {
    "1": "Consider a community where almost everyone has a BMI around {mu} (slim athletic body type). In this community, a person with a BMI of {target} is considered",
    "2": "In a population of marathon runners whose BMIs cluster around {mu}, a person with a BMI of {target} would medically be classified as",
}

TEMPLATES_NARROW_HIGH_BMI = {
    "1": "Consider a community where almost everyone has a BMI around {mu}. In this community, a person with a BMI of {target} is considered",
    "2": "In a community where BMIs cluster around {mu}, a person with a BMI of {target} would medically be classified as",
}

TEMPLATES_WIDE_SYMMETRIC_BMI = {
    "1": "Across the general adult population, a person with a BMI of {target} would medically be classified as",
}

TEMPLATES_OOD_BMI = {
    "1": "In a normal population of adult humans, a person with a BMI of {target} would medically be classified as",
}


def _get_templates_for_context(context: Context) -> dict[str, str]:
    """Get the template dictionary for a given context.

    Args:
        context: A Context object.

    Returns:
        Dictionary mapping template IDs to f-string templates.

    Raises:
        ValueError: If the context combination is not supported.
    """
    if context.attribute == "height_cm":
        if context.name == "narrow_low":
            return TEMPLATES_NARROW_LOW_HEIGHT
        elif context.name == "narrow_high":
            return TEMPLATES_NARROW_HIGH_HEIGHT
        elif context.name == "wide_symmetric":
            return TEMPLATES_WIDE_SYMMETRIC_HEIGHT
        elif context.name == "ood_contaminated":
            return TEMPLATES_OOD_HEIGHT
    elif context.attribute == "bmi":
        if context.name == "narrow_low":
            return TEMPLATES_NARROW_LOW_BMI
        elif context.name == "narrow_high":
            return TEMPLATES_NARROW_HIGH_BMI
        elif context.name == "wide_symmetric":
            return TEMPLATES_WIDE_SYMMETRIC_BMI
        elif context.name == "ood_contaminated":
            return TEMPLATES_OOD_BMI

    raise ValueError(f"Unsupported context: {context}")


def _render_prompt(
    template: str,
    target_value: float,
    context: Context,
) -> str:
    """Render a template into a concrete prompt.

    Args:
        template: An f-string template with placeholders.
        target_value: The target numerical value.
        context: The Context object.

    Returns:
        Rendered prompt string.
    """
    mu = context.mu
    sigma = context.sigma
    target = target_value

    # Compute range bounds for wide_symmetric context
    range_low = mu - 5 * sigma
    range_high = mu + 5 * sigma

    # Compute range bounds for narrow contexts (±1 sigma)
    mu_low = mu - sigma
    mu_high = mu + sigma

    # Format with appropriate precision
    if context.attribute == "height_cm":
        return template.format(
            mu=int(mu),
            sigma=int(sigma),
            target=int(target),
            mu_low=int(mu_low),
            mu_high=int(mu_high),
            range_low=int(range_low),
            range_high=int(range_high),
        )
    else:  # bmi
        return template.format(
            mu=int(mu),
            sigma=int(sigma),
            target=int(target),
            mu_low=int(mu_low),
            mu_high=int(mu_high),
            range_low=int(range_low),
            range_high=int(range_high),
        )


def generate_trials(
    adjectives: list[str],
    target_values: list[float],
    contexts: list[Context],
    templates_per_context: int = 3,
    seed: int = 0,
) -> list[Trial]:
    """Generate trials for all combinations of adjectives, target values, and contexts.

    For each (adjective, target_value, context) triple, generates `templates_per_context`
    phrasings by sampling templates without replacement. The Cartesian product size is
    len(adjectives) × len(target_values) × len(contexts) × templates_per_context.

    Args:
        adjectives: List of adjective strings (e.g., ["tall", "short", "obese"]).
        target_values: List of numerical values to evaluate.
        contexts: List of Context objects.
        templates_per_context: How many template variations per (adj, val, ctx) cell.
        seed: Random seed for reproducible template selection.

    Returns:
        List of Trial objects.

    Raises:
        ValueError: If an adjective is not in ADJECTIVE_CLASS_MAP.
    """
    # Validate adjectives
    for adj in adjectives:
        if adj not in ADJECTIVE_CLASS_MAP:
            raise ValueError(
                f"Unknown adjective '{adj}'. Must be one of: {list(ADJECTIVE_CLASS_MAP.keys())}"
            )

    random.seed(seed)
    trials = []

    for adjective in adjectives:
        adjective_class = ADJECTIVE_CLASS_MAP[adjective]

        for target_value in target_values:
            for context in contexts:
                # Get available templates for this context
                templates = _get_templates_for_context(context)

                # Determine how many templates to actually use
                num_templates = min(templates_per_context, len(templates))

                # Sample template IDs without replacement
                template_ids = random.sample(sorted(templates.keys()), num_templates)

                for template_id in template_ids:
                    template = templates[template_id]
                    prompt = _render_prompt(template, target_value, context)

                    trial = Trial(
                        adjective=adjective,
                        adjective_class=adjective_class,
                        target_value=target_value,
                        context=context,
                        prompt=prompt,
                        prompt_template_id=f"{context.name}_{template_id}",
                    )
                    trials.append(trial)

    return trials


if __name__ == "__main__":
    import json

    # Regenerate v0 battery and compare with prompts_v0.jsonl
    v0_contexts = canonical_v0_contexts()

    # Use specific adjectives and values from v0
    v0_trials = generate_trials(
        adjectives=["tall", "short", "obese"],
        target_values=[165, 175, 155, 32],
        contexts=v0_contexts,
        templates_per_context=3,
        seed=0,
    )

    # Load the reference v0 file
    v0_file = "data_gen/prompts_v0.jsonl"
    v0_ref = {}
    try:
        with open(v0_file) as f:
            for line in f:
                obj = json.loads(line)
                v0_ref[obj["id"]] = obj
    except FileNotFoundError:
        print(f"Warning: {v0_file} not found. Skipping comparison.")
        v0_ref = {}

    # Print generated trials
    print(f"Generated {len(v0_trials)} trials.")
    print("\nSample prompts:")
    for trial in v0_trials[:5]:
        print(
            f"  {trial.adjective} ({trial.adjective_class}) "
            f"@ {trial.target_value} in {trial.context.name}: "
            f"z={z_score(trial):.2f}"
        )
        print(f"    {trial.prompt}")
        print()

    # Basic validation
    if v0_ref:
        print(f"\nComparison with {v0_file}:")
        print(f"  Generated trials: {len(v0_trials)}")
        print(f"  Reference trials: {len(v0_ref)}")

        # Check if any prompts match exactly
        generated_prompts = {t.prompt for t in v0_trials}
        ref_prompts = {v["prompt"] for v in v0_ref.values()}
        overlap = generated_prompts & ref_prompts
        print(f"  Exact prompt matches: {len(overlap)}")
