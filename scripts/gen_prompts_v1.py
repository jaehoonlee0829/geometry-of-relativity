"""Generate prompts_v1.jsonl — spectrum scan with paired BMI-direct vs height+weight."""
from __future__ import annotations
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "data_gen" / "prompts_v1.jsonl"

# Spectrum sweep for tall (target = 165 cm). Narrow contexts, sigma = 3.
TALL_MUS = [140, 150, 155, 160, 165, 170, 175, 180, 190]
TARGET_HEIGHT = 165

# Spectrum sweep for obese (target = BMI 32, equivalently ~170 cm / ~92 kg). Narrow contexts, sigma_BMI = 1.
OBESE_MUS = [19, 22, 25, 28, 31, 34, 37, 40, 43]
TARGET_BMI = 32
TARGET_HEIGHT_CM = 170  # fixed
TARGET_WEIGHT_KG = 92   # 92 kg / 1.70^2 = 31.83 BMI ≈ 32

def bmi_to_weight(bmi: float, height_cm: float = 170.0) -> float:
    """Inverse BMI: weight in kg given height in cm."""
    h_m = height_cm / 100.0
    return round(bmi * h_m * h_m, 1)

TALL_TEMPLATES = [
    "Consider a group of people whose heights cluster tightly around {mu} cm. In this group, a person who is {target} cm is considered",
    "In a community where most adults are about {mu} cm tall, a person who is {target} cm would be described as",
    "Imagine a village where nearly everyone is close to {mu} cm in height. A newcomer who is {target} cm would be seen as",
]

OBESE_BMI_TEMPLATES = [
    "Consider a community where most people have a BMI around {mu}. In this community, a person with a BMI of {target} is considered",
    "In a population whose BMIs cluster tightly around {mu}, a person with a BMI of {target} would be medically classified as",
    "Imagine a group of adults whose typical BMI is about {mu}. A person with a BMI of {target} would be described as",
]

# Height+weight style: fixed height 170 cm for everyone; reference weight varies with context BMI.
# Target is 170 cm / 92 kg (BMI ≈ 32). This forces the model to compare weights rather than match the number 32.
OBESE_HW_TEMPLATES = [
    "Consider a community where most adults are 170 cm tall and weigh around {ref_kg} kg. In this community, a person who is 170 cm tall and weighs {target_kg} kg is considered",
    "In a population where adults are typically 170 cm tall and weigh approximately {ref_kg} kg, a person who is 170 cm tall and weighs {target_kg} kg would be medically classified as",
    "Imagine a group where everyone is 170 cm tall and most weigh about {ref_kg} kg. A person who is 170 cm tall and weighs {target_kg} kg would be described as",
]

def emit():
    rows = []
    # Tall/short spectrum
    for mu in TALL_MUS:
        for ti, tpl in enumerate(TALL_TEMPLATES):
            rows.append({
                "id": f"tall_target{TARGET_HEIGHT}_mu{mu}_t{ti}",
                "family": "tall_spectrum",
                "adjective_class_expected": "relative",
                "target_value": TARGET_HEIGHT,
                "target_unit": "cm",
                "context_mu": mu,
                "context_sigma": 3,
                "prompt_style": "cm_direct",
                "template_idx": ti,
                "prompt": tpl.format(mu=mu, target=TARGET_HEIGHT),
            })
    # Obese spectrum — BMI direct
    for mu in OBESE_MUS:
        for ti, tpl in enumerate(OBESE_BMI_TEMPLATES):
            rows.append({
                "id": f"obese_bmi_target{TARGET_BMI}_mu{mu}_t{ti}",
                "family": "obese_spectrum_bmi",
                "adjective_class_expected": "absolute",
                "target_value": TARGET_BMI,
                "target_unit": "BMI",
                "context_mu": mu,
                "context_sigma": 1,
                "prompt_style": "bmi_direct",
                "template_idx": ti,
                "prompt": tpl.format(mu=mu, target=TARGET_BMI),
            })
    # Obese spectrum — height+weight (forces derivation)
    for mu in OBESE_MUS:
        ref_kg = bmi_to_weight(mu, TARGET_HEIGHT_CM)
        for ti, tpl in enumerate(OBESE_HW_TEMPLATES):
            rows.append({
                "id": f"obese_hw_target{TARGET_WEIGHT_KG}kg_mu{mu}_t{ti}",
                "family": "obese_spectrum_hw",
                "adjective_class_expected": "absolute",
                "target_value": TARGET_WEIGHT_KG,
                "target_unit": "kg",
                "target_height_cm": TARGET_HEIGHT_CM,
                "target_implied_bmi": TARGET_BMI,
                "context_mu": mu,  # context mu is still in BMI units for comparability
                "context_mu_kg": ref_kg,
                "context_sigma": 1,
                "prompt_style": "height_weight",
                "template_idx": ti,
                "prompt": tpl.format(ref_kg=ref_kg, target_kg=TARGET_WEIGHT_KG),
            })
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} prompts to {OUT}")

if __name__ == "__main__":
    emit()
