"""Extract hidden-state activations from transformer LMs at specific layers."""

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ActivationBatch:
    """Container for activations at a single layer across a prompt batch."""

    model_id: str  # e.g. "google/gemma-2-2b"
    layer_name: str  # "early" | "mid" | "late" | "final"
    layer_index: int
    prompts: list[str]  # input prompts
    activations: np.ndarray  # (N, d) float32, last-token hidden state
    unembedding: np.ndarray  # (V, d) float32, lm_head weight matrix


def extract_activations(
    model_id: str,
    prompts: list[str],
    layer_keys: list[str],
    layer_indices: dict[str, int],
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int = 8,
) -> dict[str, ActivationBatch]:
    """Load model, run forward passes, return {layer_key: ActivationBatch}.

    Args:
        model_id: HuggingFace model ID (e.g. "google/gemma-2-2b").
        prompts: List of input prompts.
        layer_keys: Layer names to extract (e.g. ["early", "mid", "late", "final"]).
        layer_indices: Mapping from layer_key to absolute layer index.
        device: "cuda" or "cpu".
        dtype: "bfloat16" or "float32".
        batch_size: Number of prompts per forward pass.

    Returns:
        Dictionary mapping layer_key -> ActivationBatch.
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers and torch not installed. "
            "Install with: pip install torch transformers"
        )

    # Choose torch dtype
    torch_dtype = (
        torch.bfloat16 if dtype == "bfloat16" and device != "cpu" else torch.float32
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Infer hidden dimension from model
    hidden_dim = model.config.hidden_size

    # Get lm_head weight matrix (V, d)
    with torch.inference_mode():
        unembedding = model.lm_head.weight.detach().float().cpu().numpy()

    # Process prompts in batches
    all_hidden_states = {key: [] for key in layer_keys}

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize with padding
        encodings = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # outputs.hidden_states is tuple of (num_layers+1,) tensors, each (batch, seq, d)
        hidden_states = outputs.hidden_states

        # Extract last token for each layer
        for layer_key in layer_keys:
            layer_idx = layer_indices[layer_key]
            # hidden_states[layer_idx] is (batch, seq, d)
            # Take last token: (batch, d)
            last_token_states = hidden_states[layer_idx][:, -1, :]
            # Cast to float32, move to CPU, convert to numpy
            last_token_np = last_token_states.float().cpu().numpy().astype(
                np.float32
            )
            all_hidden_states[layer_key].append(last_token_np)

    # Stack batches for each layer
    result = {}
    for layer_key in layer_keys:
        stacked = np.concatenate(all_hidden_states[layer_key], axis=0)  # (N, d)
        batch = ActivationBatch(
            model_id=model_id,
            layer_name=layer_key,
            layer_index=layer_indices[layer_key],
            prompts=prompts,
            activations=stacked,
            unembedding=unembedding,
        )
        result[layer_key] = batch

    return result


def save_batch(batch: ActivationBatch, path: str) -> None:
    """Save ActivationBatch to .npz file.

    Args:
        batch: ActivationBatch to save.
        path: Output path (should end in .npz).
    """
    np.savez(
        path,
        activations=batch.activations,
        unembedding=batch.unembedding,
        prompts=np.array(batch.prompts, dtype=object),
        model_id=batch.model_id,
        layer_name=batch.layer_name,
        layer_index=batch.layer_index,
    )


def load_batch(path: str) -> ActivationBatch:
    """Load ActivationBatch from .npz file.

    Args:
        path: Path to .npz file.

    Returns:
        Reconstructed ActivationBatch.
    """
    data = np.load(path, allow_pickle=True)
    return ActivationBatch(
        model_id=str(data["model_id"]),
        layer_name=str(data["layer_name"]),
        layer_index=int(data["layer_index"]),
        prompts=data["prompts"].tolist(),
        activations=data["activations"].astype(np.float32),
        unembedding=data["unembedding"].astype(np.float32),
    )


if __name__ == "__main__":
    """CPU smoke test with tiny stand-in model."""
    if not HAS_TRANSFORMERS:
        print(
            "transformers/torch not available. "
            "Smoke test skipped (import guard OK)."
        )
        sys.exit(0)

    # Hardcode tiny-gpt2 layer indices locally for smoke test
    TINY_LAYER_INDICES = {
        "early": 0,
        "mid": 1,
        "late": 2,
        "final": 3,
    }

    print("=== Activation Extract Smoke Test ===")
    print("Loading tiny-gpt2 (CPU, float32)...")

    try:
        model_id = "sshleifer/tiny-gpt2"
        prompts = ["The color of the sky is", "The answer to life is"]
        layer_keys = ["early", "mid", "late", "final"]

        result = extract_activations(
            model_id=model_id,
            prompts=prompts,
            layer_keys=layer_keys,
            layer_indices=TINY_LAYER_INDICES,
            device="cpu",
            dtype="float32",
            batch_size=2,
        )

        print(f"✓ Extracted {len(result)} layers")
        for key, batch in result.items():
            print(
                f"  {key}: activations {batch.activations.shape}, "
                f"unembedding {batch.unembedding.shape}"
            )

        # Test save/load round-trip
        test_path = "/tmp/test_activation.npz"
        early_batch = result["early"]
        save_batch(early_batch, test_path)
        print(f"✓ Saved to {test_path}")

        loaded = load_batch(test_path)
        print(
            f"✓ Loaded: activations {loaded.activations.shape}, "
            f"prompts {len(loaded.prompts)}"
        )

        # Verify shapes
        assert loaded.activations.shape[0] == len(prompts), "Mismatch in num prompts"
        assert (
            loaded.activations.shape[1] == early_batch.activations.shape[1]
        ), "Mismatch in hidden dim"
        print("✓ All shapes verified")
        print("\n=== Smoke test passed ===")

    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
