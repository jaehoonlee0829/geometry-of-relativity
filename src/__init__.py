"""mech-interp-relativity: Fisher-geometric study of gradable adjectives in LLMs."""

__version__ = "0.0.1"

MODELS = {
    "gemma-2-2b": "google/gemma-2-2b",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}

# Per-model layer indices for {early, mid, late, final}.
# Gemma-2-2b has 26 layers, Llama-3.2-3B has 28 layers.
LAYER_INDICES = {
    "gemma-2-2b": {"early": 4, "mid": 13, "late": 20, "final": 25},
    "llama-3.2-3b": {"early": 5, "mid": 14, "late": 22, "final": 27},
}
