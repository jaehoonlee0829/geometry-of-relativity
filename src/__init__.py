"""mech-interp-relativity: Fisher-geometric study of gradable adjectives in LLMs."""

__version__ = "0.0.1"

MODELS = {
    "gemma-4-31b": "google/gemma-4-31B",
    "gemma-4-e4b": "google/gemma-4-E4B",
    # Legacy (v0/v1 behavioral experiments)
    "gemma-2-2b": "google/gemma-2-2b",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}

# Per-model layer indices for {early, mid, late, final}.
# Gemma-4-31B: 60 layers, d=5376
# Gemma-4-E4B: 42 layers, d=2560
# Gemma-2-2b: 26 layers (legacy)
# Llama-3.2-3B: 28 layers (legacy)
LAYER_INDICES = {
    "gemma-4-31b": {"early": 10, "mid": 30, "late": 45, "final": 59},
    "gemma-4-e4b": {"early": 7, "mid": 21, "late": 33, "final": 41},
    "gemma-2-2b": {"early": 4, "mid": 13, "late": 20, "final": 25},
    "llama-3.2-3b": {"early": 5, "mid": 14, "late": 22, "final": 27},
}
