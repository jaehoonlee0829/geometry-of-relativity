"""mech-interp-relativity: Fisher-geometric study of gradable adjectives in LLMs."""

__version__ = "0.0.1"

MODELS = {
    "gemma-4-31b": "google/gemma-4-31B",
    "gemma-2-9b": "google/gemma-2-9b",
    # Legacy (v0/v1 behavioral experiments)
    "gemma-2-2b": "google/gemma-2-2b",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}

# Per-model layer indices for {early, mid, late, final}.
# Gemma-4-31B has 48 layers, Gemma-2-9b has 42 layers.
# Gemma-2-2b has 26 layers, Llama-3.2-3B has 28 layers.
LAYER_INDICES = {
    "gemma-4-31b": {"early": 8, "mid": 24, "late": 36, "final": 47},
    "gemma-2-9b": {"early": 7, "mid": 21, "late": 33, "final": 41},
    "gemma-2-2b": {"early": 4, "mid": 13, "late": 20, "final": 25},
    "llama-3.2-3b": {"early": 5, "mid": 14, "late": 22, "final": 27},
}
