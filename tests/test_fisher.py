"""Tests for Fisher information matrix computation and inversion.

Verifies:
1. Finite-difference accuracy of the Jacobian via Fisher-Jacobian relationship
2. Symmetry and positive semi-definiteness of the Fisher matrix
3. Accuracy of Fisher inverse solve against np.linalg.solve
"""

import numpy as np
import pytest
from src.fisher import (
    softmax_probs,
    fisher_matrix,
    fisher_inv_times_w,
    fisher_normalized_cosine,
)


class TestFisherFiniteDifference:
    """Verify Fisher matrix via finite-difference Jacobian check."""

    def test_jacobian_vector_product(self):
        """Test that (p_{h+εv} - p_{h-εv})/(2ε) ≈ J @ v where J = diag(p) - p p^T @ W_U."""
        # Setup
        np.random.seed(42)
        d = 8
        V = 50
        eps = 1e-5

        # Random h, W_U, unit vector v
        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)
        v = np.random.randn(d).astype(np.float64)
        v /= np.linalg.norm(v)  # normalize to unit vector

        # Compute probabilities at h, h+εv, h-εv
        p = softmax_probs(h, W_U)
        p_plus = softmax_probs(h + eps * v, W_U)
        p_minus = softmax_probs(h - eps * v, W_U)

        # Finite-difference estimate of dp/dv
        dp_dv_fd = (p_plus - p_minus) / (2 * eps)

        # Analytic Jacobian: J = (diag(p) - p p^T) W_U
        # Jacobian-vector product: J @ v = (diag(p) - p p^T) @ W_U @ v
        W_U_v = W_U @ v  # (V,)
        p_W_U_v = p * W_U_v  # element-wise
        p_dot_W_U_v = np.dot(p, W_U_v)  # scalar
        dp_dv_analytic = p_W_U_v - p * p_dot_W_U_v

        # Check they match
        assert np.allclose(dp_dv_fd, dp_dv_analytic, atol=1e-6), \
            f"FD Jacobian mismatch: max error {np.max(np.abs(dp_dv_fd - dp_dv_analytic)):.2e}"

    def test_fisher_jacobian_relationship(self):
        """Verify that F(h) = W_U^T @ (diag(p) - p p^T) @ W_U."""
        np.random.seed(43)
        d = 8
        V = 50

        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)

        # Compute Fisher matrix
        F = fisher_matrix(h, W_U, jitter=0.0)  # no jitter for exact check

        # Compute Jacobian explicitly: J = (diag(p) - p p^T) @ W_U
        p = softmax_probs(h, W_U)
        diag_p = np.diag(p)
        p_outer = np.outer(p, p)
        J = (diag_p - p_outer) @ W_U  # (V, d)

        # F should equal W_U^T @ J
        F_expected = W_U.T @ J

        assert np.allclose(F, F_expected, atol=1e-12), \
            f"Fisher-Jacobian mismatch: max error {np.max(np.abs(F - F_expected)):.2e}"


class TestFisherProperties:
    """Verify mathematical properties of the Fisher matrix."""

    def test_symmetry(self):
        """Fisher matrix should be symmetric."""
        np.random.seed(44)
        d = 8
        V = 50

        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)

        F = fisher_matrix(h, W_U, jitter=1e-6)

        symmetry_error = np.max(np.abs(F - F.T))
        assert symmetry_error < 1e-12, \
            f"Symmetry violated: max |F - F.T| = {symmetry_error:.2e}"

    def test_positive_semidefinite(self):
        """Fisher matrix should be positive semi-definite."""
        np.random.seed(45)
        d = 8
        V = 50

        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)

        F = fisher_matrix(h, W_U, jitter=1e-6)

        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(F)
        min_eigval = np.min(eigvals)

        assert min_eigval > -1e-8, \
            f"Not PSD: min eigenvalue = {min_eigval:.2e}"

    def test_psd_multiple_seeds(self):
        """Fisher matrix should be PSD across multiple random instances."""
        d = 8
        V = 50

        for seed in range(5):
            np.random.seed(seed)
            h = np.random.randn(d).astype(np.float64)
            W_U = np.random.randn(V, d).astype(np.float64)

            F = fisher_matrix(h, W_U, jitter=1e-6)
            eigvals = np.linalg.eigvalsh(F)
            min_eigval = np.min(eigvals)

            assert min_eigval > -1e-8, \
                f"Seed {seed}: min eigenvalue = {min_eigval:.2e}"


class TestFisherInverse:
    """Verify Fisher inverse computation via Cholesky."""

    def test_fisher_inv_times_w_vector(self):
        """Test fisher_inv_times_w matches np.linalg.solve for 1D vector."""
        np.random.seed(46)
        d = 8
        V = 50

        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)
        w = np.random.randn(d).astype(np.float64)

        # Compute via Fisher
        result_fisher = fisher_inv_times_w(h, W_U, w, jitter=1e-6)

        # Compute via np.linalg.solve(F, w)
        F = fisher_matrix(h, W_U, jitter=1e-6)
        result_numpy = np.linalg.solve(F, w)

        assert np.allclose(result_fisher, result_numpy, atol=1e-10), \
            f"1D vector solve mismatch: max error {np.max(np.abs(result_fisher - result_numpy)):.2e}"

    def test_fisher_inv_times_w_matrix(self):
        """Test fisher_inv_times_w matches np.linalg.solve for 2D matrix."""
        np.random.seed(47)
        d = 8
        V = 50
        k = 3

        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)
        W = np.random.randn(d, k).astype(np.float64)

        # Compute via Fisher
        result_fisher = fisher_inv_times_w(h, W_U, W, jitter=1e-6)

        # Compute via np.linalg.solve(F, W)
        F = fisher_matrix(h, W_U, jitter=1e-6)
        result_numpy = np.linalg.solve(F, W)

        assert np.allclose(result_fisher, result_numpy, atol=1e-10), \
            f"2D matrix solve mismatch: max error {np.max(np.abs(result_fisher - result_numpy)):.2e}"

    def test_fisher_inv_solves_identity(self):
        """Test that F^{-1} @ F @ x = x."""
        np.random.seed(48)
        d = 8
        V = 50

        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)
        x = np.random.randn(d).astype(np.float64)

        F = fisher_matrix(h, W_U, jitter=1e-6)
        F_inv_x = fisher_inv_times_w(h, W_U, x, jitter=1e-6)

        # Verify: F @ (F^{-1} @ x) = x
        recovered = F @ F_inv_x

        assert np.allclose(recovered, x, atol=1e-9), \
            f"Identity check failed: max error {np.max(np.abs(recovered - x)):.2e}"


class TestFisherNormalizedCosine:
    """Verify Fisher-normalized cosine similarity."""

    def test_cosine_self_product(self):
        """Cosine of a vector with itself should be 1."""
        np.random.seed(49)
        d = 8
        V = 50

        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)
        u = np.random.randn(d).astype(np.float64)

        F = fisher_matrix(h, W_U, jitter=1e-6)
        cosine = fisher_normalized_cosine(u, u, F)

        assert np.allclose(cosine, 1.0, atol=1e-12), \
            f"Self-cosine should be 1, got {cosine:.6f}"

    def test_cosine_range(self):
        """Cosine should be in [-1, 1]."""
        np.random.seed(50)
        d = 8
        V = 50

        h = np.random.randn(d).astype(np.float64)
        W_U = np.random.randn(V, d).astype(np.float64)
        u = np.random.randn(d).astype(np.float64)
        v = np.random.randn(d).astype(np.float64)

        F = fisher_matrix(h, W_U, jitter=1e-6)
        cosine = fisher_normalized_cosine(u, v, F)

        assert -1.0 <= cosine <= 1.0, \
            f"Cosine out of range: {cosine:.6f}"
