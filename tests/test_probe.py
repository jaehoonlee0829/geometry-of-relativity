"""Tests for linear probe training on synthetic data.

Mirrors the __main__ block of src/probe.py: verify that
a probe trained on synthetic data with a known direction
recovers that direction with cosine > 0.9 and test_acc > 0.9.
"""

import numpy as np
import pytest
from src.probe import train_binary_probe


class TestProbeSyntheticRecovery:
    """Verify synthetic probe recovery test."""

    def test_probe_recovers_true_direction(self):
        """Train probe on synthetic data and verify recovery of true direction."""
        # Setup matches __main__ block of src/probe.py
        np.random.seed(0)

        d = 32
        N = 200

        # Generate true direction and normalize
        w_true = np.random.randn(d)
        w_true /= np.linalg.norm(w_true)

        # Generate synthetic data: X ~ N(0, I), labels = (X @ w_true > 0)
        X = np.random.randn(N, d)
        scores = X @ w_true
        y = (scores > 0).astype(int)

        # Check class balance
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        assert n_pos >= 10, f"Insufficient positive examples: {n_pos}"
        assert n_neg >= 10, f"Insufficient negative examples: {n_neg}"

        # Train probe
        result = train_binary_probe(
            X,
            y,
            label_positive="positive",
            label_negative="negative",
            C=1.0,
            test_frac=0.2,
            seed=0,
        )

        # Verify recovery via cosine similarity
        cos_sim = _cosine_similarity(result.w, w_true)

        # Assertions per spec
        assert cos_sim > 0.9, \
            f"Cosine similarity {cos_sim:.6f} not > 0.9"
        assert result.test_acc > 0.9, \
            f"Test accuracy {result.test_acc:.6f} not > 0.9"

    def test_probe_accuracy_metrics(self):
        """Verify that train_acc and test_acc are computed correctly."""
        np.random.seed(1)

        d = 16
        N = 100

        w_true = np.random.randn(d)
        w_true /= np.linalg.norm(w_true)

        X = np.random.randn(N, d)
        scores = X @ w_true
        y = (scores > 0).astype(int)

        result = train_binary_probe(
            X,
            y,
            label_positive="pos",
            label_negative="neg",
            C=1.0,
            test_frac=0.2,
            seed=0,
        )

        # Verify metrics are in [0, 1]
        assert 0.0 <= result.train_acc <= 1.0, \
            f"Train accuracy out of range: {result.train_acc}"
        assert 0.0 <= result.test_acc <= 1.0, \
            f"Test accuracy out of range: {result.test_acc}"

        # Verify counts sum correctly
        assert result.n_train + result.n_test == N, \
            f"Train+test counts {result.n_train + result.n_test} != N={N}"

    def test_probe_with_different_seeds(self):
        """Verify probe training is reproducible and robust across seeds."""
        d = 32
        N = 200

        # Generate fixed synthetic data
        np.random.seed(42)
        w_true = np.random.randn(d)
        w_true /= np.linalg.norm(w_true)
        X = np.random.randn(N, d)
        scores = X @ w_true
        y = (scores > 0).astype(int)

        # Train multiple probes with different seeds
        results = []
        for seed in [0, 1, 2]:
            result = train_binary_probe(
                X,
                y,
                label_positive="pos",
                label_negative="neg",
                C=1.0,
                test_frac=0.2,
                seed=seed,
            )
            results.append(result)

        # All should recover the direction reasonably well
        for i, result in enumerate(results):
            cos_sim = _cosine_similarity(result.w, w_true)
            assert cos_sim > 0.85, \
                f"Seed {i}: cosine {cos_sim:.6f} < 0.85"

    def test_probe_rejects_imbalanced_data(self):
        """Verify probe training rejects data with < 10 examples in any class."""
        np.random.seed(2)

        d = 16
        N = 100

        X = np.random.randn(N, d)
        # Create highly imbalanced labels: only 5 positive examples
        y = np.zeros(N, dtype=int)
        y[:5] = 1

        # Should raise ValueError
        with pytest.raises(ValueError, match="Insufficient class balance"):
            train_binary_probe(
                X,
                y,
                label_positive="pos",
                label_negative="neg",
                C=1.0,
                test_frac=0.2,
                seed=0,
            )

    def test_probe_result_attributes(self):
        """Verify ProbeResult has all expected attributes."""
        np.random.seed(3)

        d = 16
        N = 100

        w_true = np.random.randn(d)
        w_true /= np.linalg.norm(w_true)

        X = np.random.randn(N, d)
        scores = X @ w_true
        y = (scores > 0).astype(int)

        result = train_binary_probe(
            X,
            y,
            label_positive="tall",
            label_negative="short",
            C=1.0,
            test_frac=0.2,
            seed=0,
        )

        # Check all attributes exist and have correct types
        assert hasattr(result, "w") and result.w.ndim == 1, "w should be 1D"
        assert hasattr(result, "b") and isinstance(result.b, float), "b should be float"
        assert hasattr(result, "train_acc") and isinstance(result.train_acc, float), \
            "train_acc should be float"
        assert hasattr(result, "test_acc") and isinstance(result.test_acc, float), \
            "test_acc should be float"
        assert hasattr(result, "n_train") and isinstance(result.n_train, int), \
            "n_train should be int"
        assert hasattr(result, "n_test") and isinstance(result.n_test, int), \
            "n_test should be int"
        assert result.label_positive == "tall", "label_positive mismatch"
        assert result.label_negative == "short", "label_negative mismatch"


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        u: (d,) vector.
        v: (d,) vector.

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-10 or norm_v < 1e-10:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))
