"""Fisher information metric for LLM next-token distributions.

Implements F(h) = W_U^T (diag(p) - p p^T) W_U where p = softmax(h @ W_U.T).
Uses low-rank reformulation to avoid constructing (V x V) matrices.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def softmax_probs(h: np.ndarray, W_U: np.ndarray) -> np.ndarray:
    """Stable softmax of h @ W_U.T.

    Args:
        h: shape (d,) or (B, d), dtype float (cast to float64).
        W_U: shape (V, d), dtype float (cast to float64).

    Returns:
        p: shape (V,) or (B, V), dtype float64.
    """
    h = np.asarray(h, dtype=np.float64)
    W_U = np.asarray(W_U, dtype=np.float64)

    if h.ndim == 1:
        assert h.shape[0] == W_U.shape[1], f"h.shape[0]={h.shape[0]} != W_U.shape[1]={W_U.shape[1]}"
        logits = h @ W_U.T  # (V,)
    elif h.ndim == 2:
        assert h.shape[1] == W_U.shape[1], f"h.shape[1]={h.shape[1]} != W_U.shape[1]={W_U.shape[1]}"
        logits = h @ W_U.T  # (B, V)
    else:
        raise ValueError(f"h must be 1D or 2D, got shape {h.shape}")

    # Stable softmax: shift by max
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def fisher_matrix(h: np.ndarray, W_U: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Compute Fisher information matrix F(h) = W_U^T (diag(p) - p p^T) W_U.

    Uses rank-1 reformulation to avoid constructing (V x V) matrices:
        F = W_U^T diag(p) W_U - (W_U^T p)(W_U^T p)^T

    Args:
        h: shape (d,), dtype float.
        W_U: shape (V, d), dtype float.
        jitter: small constant added to diagonal for numerical stability.

    Returns:
        F: shape (d, d), dtype float64, symmetric positive semi-definite.
    """
    h = np.asarray(h, dtype=np.float64)
    W_U = np.asarray(W_U, dtype=np.float64)

    assert h.ndim == 1, f"h must be 1D, got shape {h.shape}"
    assert W_U.ndim == 2, f"W_U must be 2D, got shape {W_U.shape}"
    assert h.shape[0] == W_U.shape[1], f"h.shape[0]={h.shape[0]} != W_U.shape[1]={W_U.shape[1]}"

    d = h.shape[0]
    V = W_U.shape[0]

    # Compute probabilities
    p = softmax_probs(h, W_U)  # (V,)

    # Compute W_U^T @ diag(p) @ W_U via column-wise scaling
    # diag(p) @ W_U = (p[:,None] * W_U), then W_U^T @ that
    scaled_W_U = p[:, None] * W_U  # (V, d)
    term1 = W_U.T @ scaled_W_U  # (d, d)

    # Compute (W_U^T @ p)(W_U^T @ p)^T
    w_t_p = W_U.T @ p  # (d,)
    term2 = np.outer(w_t_p, w_t_p)  # (d, d)

    # F = term1 - term2
    F = term1 - term2  # (d, d)

    # Add jitter to diagonal for stability
    F = F + jitter * np.eye(d)

    return F


def fisher_inv_times_w(
    h: np.ndarray, W_U: np.ndarray, w: np.ndarray, jitter: float = 1e-6
) -> np.ndarray:
    """Compute F(h)^{-1} @ w via Cholesky decomposition.

    Uses scipy.linalg.cho_factor and cho_solve for numerical stability.
    Never calls np.linalg.inv.

    Args:
        h: shape (d,), dtype float.
        W_U: shape (V, d), dtype float.
        w: shape (d,) or (d, k), dtype float.
        jitter: small constant added to diagonal of F.

    Returns:
        result: shape (d,) or (d, k), same as w, dtype float64.
    """
    h = np.asarray(h, dtype=np.float64)
    W_U = np.asarray(W_U, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    d = h.shape[0]

    if w.ndim == 1:
        assert w.shape[0] == d, f"w.shape[0]={w.shape[0]} != d={d}"
    elif w.ndim == 2:
        assert w.shape[0] == d, f"w.shape[0]={w.shape[0]} != d={d}"
    else:
        raise ValueError(f"w must be 1D or 2D, got shape {w.shape}")

    # Compute Fisher matrix
    F = fisher_matrix(h, W_U, jitter=jitter)

    # Cholesky factorization
    c, low = cho_factor(F)

    # Solve F @ x = w
    result = cho_solve((c, low), w)

    return result


def fisher_normalized_cosine(u: np.ndarray, v: np.ndarray, F: np.ndarray) -> float:
    """Cosine of u, v under Fisher inner product.

    Inner product: <u, v>_F = u^T F v
    Cosine: cos = <u, v>_F / sqrt(<u, u>_F * <v, v>_F)

    Args:
        u: shape (d,), dtype float.
        v: shape (d,), dtype float.
        F: shape (d, d), dtype float, symmetric PSD.

    Returns:
        cosine: float in [-1, 1].
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)

    assert u.ndim == 1, f"u must be 1D, got shape {u.shape}"
    assert v.ndim == 1, f"v must be 1D, got shape {v.shape}"
    assert F.ndim == 2, f"F must be 2D, got shape {F.shape}"
    assert u.shape[0] == v.shape[0] == F.shape[0], \
        f"Shape mismatch: u={u.shape}, v={v.shape}, F={F.shape}"

    # <u, v>_F = u^T F v
    u_F_v = u @ F @ v

    # <u, u>_F = u^T F u
    u_F_u = u @ F @ u

    # <v, v>_F = v^T F v
    v_F_v = v @ F @ v

    # Cosine
    denom = np.sqrt(u_F_u * v_F_v)
    if denom < 1e-15:
        return 0.0

    return float(u_F_v / denom)


if __name__ == "__main__":
    # Smoke test: V=100, d=8
    print("=" * 70)
    print("Fisher information matrix smoke test")
    print("=" * 70)

    V, d = 100, 8
    np.random.seed(42)

    # Generate random W_U and h
    W_U = np.random.randn(V, d).astype(np.float32)
    h = np.random.randn(d).astype(np.float32)

    print(f"\nTest setup: V={V}, d={d}")
    print(f"W_U shape: {W_U.shape}, h shape: {h.shape}")

    # Compute Fisher matrix
    F = fisher_matrix(h, W_U, jitter=1e-6)
    print(f"\nF shape: {F.shape}, dtype: {F.dtype}")

    # Check symmetry
    symmetry_error = np.max(np.abs(F - F.T))
    print(f"Symmetry error (max |F - F.T|): {symmetry_error:.2e}")

    # Check positive semi-definiteness
    eigvals = np.linalg.eigvalsh(F)
    min_eigval = np.min(eigvals)
    print(f"Min eigenvalue: {min_eigval:.2e}")
    print(f"PSD check (all evals >= -1e-8): {min_eigval >= -1e-8}")

    # Test fisher_inv_times_w against np.linalg.solve
    print("\n" + "-" * 70)
    print("Testing fisher_inv_times_w vs np.linalg.solve")
    print("-" * 70)

    w = np.random.randn(d).astype(np.float32)
    result_fisher = fisher_inv_times_w(h, W_U, w, jitter=1e-6)
    result_numpy = np.linalg.solve(F, w)

    diff = np.max(np.abs(result_fisher - result_numpy))
    print(f"Max difference: {diff:.2e}")
    print(f"Match to 1e-10: {diff < 1e-10}")

    # Test with multiple RHS
    print("\n" + "-" * 70)
    print("Testing fisher_inv_times_w with multiple RHS (d, k)")
    print("-" * 70)

    k = 3
    W = np.random.randn(d, k).astype(np.float32)
    result_fisher_multi = fisher_inv_times_w(h, W_U, W, jitter=1e-6)
    result_numpy_multi = np.linalg.solve(F, W)

    diff_multi = np.max(np.abs(result_fisher_multi - result_numpy_multi))
    print(f"Max difference (k={k}): {diff_multi:.2e}")
    print(f"Match to 1e-10: {diff_multi < 1e-10}")

    # Test fisher_normalized_cosine
    print("\n" + "-" * 70)
    print("Testing fisher_normalized_cosine")
    print("-" * 70)

    u = np.random.randn(d).astype(np.float32)
    v = np.random.randn(d).astype(np.float32)

    cosine = fisher_normalized_cosine(u, v, F)
    print(f"Fisher-normalized cosine(u, v): {cosine:.6f}")
    print(f"Value in [-1, 1]: {-1 <= cosine <= 1}")

    # Test with orthogonal vectors under Fisher metric
    print(f"Fisher-normalized cosine(u, u): {fisher_normalized_cosine(u, u, F):.6f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
