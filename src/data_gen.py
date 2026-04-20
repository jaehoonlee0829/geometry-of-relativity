"""Prompt generator for mech-interp study of gradable adjectives.

v0/v1: Takes structured (adjective, target_value, context) triples and emits natural-language
prompts for small LMs (Gemma-2-2b, Llama-3.2-3B) to complete. Uses canonical named contexts.

v2 (PLANNING.md spec v2): decorrelates x / mu / sigma with independent axes, supports two
context types (implicit sampled lists vs explicit stated mu+sigma), two prompt frames
("is ___" vs "is considered ___"), and two domains (height, wealth). Designed for
activation probing rather than Claude-API behavioral testing.
"""

import math
import random
from dataclasses import dataclass, field
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


# =====================================================================
# v2 API — PLANNING.md spec v2
# =====================================================================

# Prompt-frame and context-type literals for type hinting.
PromptFrame = Literal["is", "is_considered"]
ContextType = Literal["implicit", "explicit"]
Domain = Literal["height", "wealth"]


@dataclass(frozen=True)
class DomainSpec:
    """Spec for a single gradable-adjective domain in v2 experiments.

    Attributes:
        domain: "height" or "wealth".
        adjective_pair: (high-end, low-end) adjectives, e.g. ("tall", "short") or
            ("rich", "poor").
        target_values: Raw attribute values x to probe. Linear-spaced for height,
            log-spaced for wealth.
        context_means: Context means mu to probe. Chosen to decorrelate from x.
        sigma: Context "spread". For height, this is additive sigma in cm. For wealth,
            this is a log-space multiplicative factor (z = (log x - log mu) / log factor).
        unit_noun: Noun phrase for singular units ("cm", "dollars").
        unit_plural_adj: Short descriptor used inline in implicit lists
            ("Person 1: 148 cm"  vs  "Person 1: earns $48,000").
        z_log_space: If True, z-score computed in log-space (wealth). If False, linear.
        num_context_samples: How many individuals to draw in implicit context.
    """

    domain: Domain
    adjective_pair: tuple[str, str]
    target_values: tuple[float, ...]
    context_means: tuple[float, ...]
    sigma: float
    unit_noun: str
    unit_plural_adj: str
    z_log_space: bool
    num_context_samples: int = 15


# -------- canonical v2 specs --------

HEIGHT_SPEC = DomainSpec(
    domain="height",
    adjective_pair=("tall", "short"),
    # Linear-spaced cm targets
    target_values=(150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0),
    # Linear-spaced means (9 values, spanning -3.5 to +3.5 sigma from central x=165)
    context_means=(145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0),
    sigma=10.0,
    unit_noun="cm",
    unit_plural_adj="cm",
    z_log_space=False,
    num_context_samples=15,
)

WEALTH_SPEC = DomainSpec(
    domain="wealth",
    adjective_pair=("rich", "poor"),
    # Log-spaced annual income targets in USD
    target_values=(20_000.0, 50_000.0, 100_000.0, 250_000.0, 500_000.0, 1_000_000.0, 5_000_000.0),
    # Log-spaced means (7 values)
    context_means=(15_000.0, 30_000.0, 75_000.0, 150_000.0, 300_000.0, 750_000.0, 2_000_000.0),
    # Multiplicative factor: with mu=100K and sigma_factor=2, ~68% of samples fall in [50K, 200K]
    sigma=2.0,
    unit_noun="dollars",
    unit_plural_adj="per year",
    z_log_space=True,
    num_context_samples=15,
)

DOMAIN_SPECS: dict[Domain, DomainSpec] = {
    "height": HEIGHT_SPEC,
    "wealth": WEALTH_SPEC,
}


@dataclass(frozen=True)
class TrialV2:
    """A single v2 trial for activation-probing experiments.

    Unlike Trial (v0/v1), TrialV2 carries the raw (x, mu, sigma) tuple explicitly so
    downstream probe training can regress w_x, w_z, and w_adj on the same activations.

    Attributes:
        trial_id: Stable string ID of the form "{domain}_{context_type}_{frame}_{idx}".
        domain: "height" or "wealth".
        context_type: "implicit" (sampled list) or "explicit" (stated mu,sigma).
        prompt_frame: "is" (factual) or "is_considered" (subjective).
        x: Raw target value (cm or USD).
        mu: Context mean.
        sigma: Context spread (additive for height, multiplicative factor for wealth).
        z: Dimensionless context-normalized score.
        context_seed: Seed used for the deterministic implicit-context sample
            (shared across trials with the same (mu, sigma)).
        context_sample: If implicit, the 15 sampled values that appear in the prompt.
            Empty tuple for explicit context.
        prompt: The natural-language prompt, ending exactly at "is" or "considered".
        adjective_high: Adjective for the high end (e.g. "tall", "rich").
        adjective_low: Adjective for the low end (e.g. "short", "poor").
    """

    trial_id: str
    domain: Domain
    context_type: ContextType
    prompt_frame: PromptFrame
    x: float
    mu: float
    sigma: float
    z: float
    context_seed: int
    context_sample: tuple[float, ...]
    prompt: str
    adjective_high: str
    adjective_low: str


# -------- helpers --------


def _format_value(v: float, spec: DomainSpec) -> str:
    """Format a numeric value in the domain's natural units (for prompt rendering)."""
    if spec.domain == "height":
        return f"{int(round(v))} cm"
    # wealth: render as "$20,000" or "$1.5M"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M".replace(".0M", "M")
    if v >= 1_000:
        return f"${v / 1_000:.0f},000" if v % 1_000 == 0 else f"${int(round(v)):,}"
    return f"${int(round(v))}"


def _z_score_v2(x: float, mu: float, sigma: float, log_space: bool) -> float:
    """Compute v2 z-score. Linear for height, log-space for wealth."""
    if log_space:
        return (math.log(x) - math.log(mu)) / math.log(sigma)
    return (x - mu) / sigma


def _sample_context(
    spec: DomainSpec,
    mu: float,
    seed: int,
) -> tuple[float, ...]:
    """Deterministically draw `spec.num_context_samples` values from the context distribution.

    For height: N(mu, sigma) rounded to nearest cm, clipped to [120, 210].
    For wealth: log-normal centred on log(mu) with scale log(sigma_factor), rounded to
        nearest $1k, clipped to [$5k, $20M].

    The same (mu, seed) always produces the same sample — used to hold the "implicit
    context" constant across trials that share a context mean.

    Args:
        spec: DomainSpec.
        mu: Context mean.
        seed: Deterministic seed for this (mu, sigma) cell.

    Returns:
        Tuple of length spec.num_context_samples.
    """
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(spec.num_context_samples):
        if spec.z_log_space:
            # wealth: log-normal
            log_mu = math.log(mu)
            log_sigma = math.log(spec.sigma)
            log_draw = rng.gauss(log_mu, log_sigma)
            val = math.exp(log_draw)
            val = max(5_000.0, min(20_000_000.0, val))
            # Round to nearest $1k
            val = round(val / 1_000.0) * 1_000.0
        else:
            # height: normal
            val = rng.gauss(mu, spec.sigma)
            val = max(120.0, min(210.0, val))
            val = float(round(val))
        out.append(val)
    return tuple(out)


def _context_seed(domain: Domain, mu: float, sigma: float) -> int:
    """Stable integer seed for a (domain, mu, sigma) cell.

    Same (domain, mu, sigma) always returns the same seed so the implicit context
    is deterministic across runs and across trials that share this cell.
    """
    # Use hash of a tuple; mod into int32 range for determinism.
    key = (domain, round(mu, 3), round(sigma, 3))
    return abs(hash(key)) % (2**31 - 1)


def _render_implicit(
    spec: DomainSpec,
    x: float,
    mu: float,
    sample: tuple[float, ...],
    frame: PromptFrame,
) -> str:
    """Render a prompt where context is a sampled list of 15 people."""
    lines: list[str] = []
    for i, v in enumerate(sample, start=1):
        if spec.domain == "height":
            lines.append(f"Person {i}: {int(round(v))} cm")
        else:  # wealth
            lines.append(f"Person {i}: earns {_format_value(v, spec)} per year")
    # Target person
    idx = len(sample) + 1
    if spec.domain == "height":
        target_desc = f"Person {idx}: {int(round(x))} cm"
    else:
        target_desc = f"Person {idx}: earns {_format_value(x, spec)} per year"
    # Final "is" / "is considered" completion
    if frame == "is":
        tail = f"{target_desc}. This person is"
    else:
        tail = f"{target_desc}. This person is considered"
    return "\n".join(lines) + "\n" + tail


def _render_explicit(
    spec: DomainSpec,
    x: float,
    mu: float,
    frame: PromptFrame,
) -> str:
    """Render a prompt where context is stated explicitly (mu, sigma)."""
    if spec.domain == "height":
        ctx = (
            f"In a group where most people's heights cluster around {int(round(mu))} cm "
            f"(give or take {int(round(spec.sigma))} cm), a person who is "
            f"{int(round(x))} cm"
        )
    else:  # wealth
        ctx = (
            f"In a community where most people earn around {_format_value(mu, spec)} "
            f"per year (most are within a factor of {spec.sigma:g} of that), "
            f"a person who earns {_format_value(x, spec)} per year"
        )
    if frame == "is":
        return ctx + " is"
    return ctx + " is considered"


# -------- public v2 generator --------


def generate_trials_v2(
    spec: DomainSpec,
    context_types: tuple[ContextType, ...] = ("implicit", "explicit"),
    prompt_frames: tuple[PromptFrame, ...] = ("is", "is_considered"),
    target_values: tuple[float, ...] | None = None,
    context_means: tuple[float, ...] | None = None,
) -> list[TrialV2]:
    """Generate the v2 experiment matrix for a single domain.

    Axes (outer-to-inner): context_type × prompt_frame × x × mu.

    The full matrix per domain is |context_types| × |frames| × |x| × |mu|.
    Default height: 2 × 2 × 7 × 9 = 252 trials.
    Default wealth: 2 × 2 × 7 × 7 = 196 trials.

    Args:
        spec: DomainSpec for the target domain.
        context_types: Subset of {"implicit", "explicit"}.
        prompt_frames: Subset of {"is", "is_considered"}.
        target_values: Override spec.target_values.
        context_means: Override spec.context_means.

    Returns:
        List of TrialV2, deterministic across runs.
    """
    xs = target_values if target_values is not None else spec.target_values
    mus = context_means if context_means is not None else spec.context_means

    trials: list[TrialV2] = []
    for ctx_type in context_types:
        for frame in prompt_frames:
            for x in xs:
                for mu in mus:
                    seed = _context_seed(spec.domain, mu, spec.sigma)
                    if ctx_type == "implicit":
                        sample = _sample_context(spec, mu, seed)
                        prompt = _render_implicit(spec, x, mu, sample, frame)
                    else:
                        sample = ()
                        prompt = _render_explicit(spec, x, mu, frame)

                    z = _z_score_v2(x, mu, spec.sigma, spec.z_log_space)

                    idx = len(trials)
                    trial_id = f"{spec.domain}_{ctx_type}_{frame}_{idx:04d}"
                    trials.append(
                        TrialV2(
                            trial_id=trial_id,
                            domain=spec.domain,
                            context_type=ctx_type,
                            prompt_frame=frame,
                            x=float(x),
                            mu=float(mu),
                            sigma=float(spec.sigma),
                            z=float(z),
                            context_seed=seed,
                            context_sample=sample,
                            prompt=prompt,
                            adjective_high=spec.adjective_pair[0],
                            adjective_low=spec.adjective_pair[1],
                        )
                    )
    return trials


def trial_v2_to_dict(trial: TrialV2) -> dict:
    """Convert a TrialV2 to a JSON-serialisable dict (jsonl row)."""
    return {
        "id": trial.trial_id,
        "domain": trial.domain,
        "context_type": trial.context_type,
        "prompt_frame": trial.prompt_frame,
        "x": trial.x,
        "mu": trial.mu,
        "sigma": trial.sigma,
        "z": trial.z,
        "context_seed": trial.context_seed,
        "context_sample": list(trial.context_sample),
        "prompt": trial.prompt,
        "adjective_high": trial.adjective_high,
        "adjective_low": trial.adjective_low,
    }


def write_v2_jsonl(trials: list[TrialV2], path: str) -> None:
    """Write v2 trials to a jsonl file (one trial per line)."""
    import json as _json

    with open(path, "w") as f:
        for t in trials:
            f.write(_json.dumps(trial_v2_to_dict(t)) + "\n")


# =====================================================================
# __main__ entry point
# =====================================================================


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

    # ---------- v2 generation ----------
    print("\n=== v2 battery ===")
    height_trials = generate_trials_v2(HEIGHT_SPEC)
    wealth_trials = generate_trials_v2(WEALTH_SPEC)
    v2_all = height_trials + wealth_trials

    print(f"Height trials:  {len(height_trials)}")
    print(f"Wealth trials:  {len(wealth_trials)}")
    print(f"Total v2:       {len(v2_all)}")

    # Sanity: show one from each (domain, context_type, frame) cell.
    print("\nSample v2 prompts (first of each cell):")
    seen_cells: set[tuple[str, str, str]] = set()
    for t in v2_all:
        cell = (t.domain, t.context_type, t.prompt_frame)
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        print(f"\n--- {cell} | x={t.x} mu={t.mu} z={t.z:+.2f} ---")
        print(t.prompt)

    v2_file = "data_gen/prompts_v2.jsonl"
    write_v2_jsonl(v2_all, v2_file)
    print(f"\nWrote {len(v2_all)} v2 trials to {v2_file}")
