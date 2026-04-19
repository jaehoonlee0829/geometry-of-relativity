"""Linear probe training for classifying adjective labels on LLM hidden activations.

This module implements binary linear probes trained via logistic regression
on hidden activation representations to classify gradable adjective labels
(relative: tall/short; absolute: obese).

Key exports:
  - ProbeResult: dataclass holding trained probe parameters and metrics
  - train_binary_probe: train a logistic regression probe on (X, y)
  - probe_logit: compute w^T h + b for a single hidden state
  - probe_shift: compute differential probe response (w^T h_A) - (w^T h_B)
"""

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class ProbeResult:
    """Result of training a binary linear probe.

    Attributes:
        w: (d,) weight vector (probe covector) — the direction to feed into Fisher.
        b: scalar bias term from logistic regression.
        train_acc: training accuracy on train split.
        test_acc: test accuracy on test split.
        n_train: number of training examples.
        n_test: number of test examples.
        label_positive: string label for positive class, e.g., "tall".
        label_negative: string label for negative class, e.g., "short".
    """

    w: np.ndarray
    b: float
    train_acc: float
    test_acc: float
    n_train: int
    n_test: int
    label_positive: str
    label_negative: str


def train_binary_probe(
    X: np.ndarray,
    y: np.ndarray,
    label_positive: str,
    label_negative: str,
    C: float = 1.0,
    test_frac: float = 0.2,
    seed: int = 0,
) -> ProbeResult:
    """Train a binary linear probe on hidden activations via logistic regression.

    Args:
        X: (N, d) array of hidden activations (e.g., from a transformer layer).
        y: (N,) array of binary labels {0, 1}. 1 = positive class, 0 = negative class.
        label_positive: string label for class 1, e.g., "tall".
        label_negative: string label for class 0, e.g., "short".
        C: regularization strength (inverse; smaller C → stronger L2). Default 1.0.
        test_frac: fraction of data to hold out for test. Default 0.2.
        seed: random seed for train/test split. Default 0.

    Returns:
        ProbeResult with trained weights, biases, and accuracies.

    Raises:
        ValueError: if either class has fewer than 10 examples.
    """
    # Check class balance
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos < 10 or n_neg < 10:
        raise ValueError(
            f"Insufficient class balance: positive class has {n_pos} examples, "
            f"negative class has {n_neg} examples. Both classes need ≥10 examples."
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed
    )

    # Train logistic regression
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=5000,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # Extract weights and bias
    w = clf.coef_.flatten()
    b = clf.intercept_[0]

    # Compute accuracies
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    return ProbeResult(
        w=w,
        b=b,
        train_acc=train_acc,
        test_acc=test_acc,
        n_train=len(X_train),
        n_test=len(X_test),
        label_positive=label_positive,
        label_negative=label_negative,
    )


def probe_logit(h: np.ndarray, result: ProbeResult) -> float:
    """Compute the logit (pre-sigmoid score) of the probe.

    Args:
        h: (d,) hidden activation vector.
        result: ProbeResult from train_binary_probe.

    Returns:
        Scalar logit w^T h + b.
    """
    return float(np.dot(result.w, h) + result.b)


def probe_shift(
    h_contextA: np.ndarray, h_contextB: np.ndarray, result: ProbeResult
) -> float:
    """Compute the differential probe response between two contexts.

    Args:
        h_contextA: (d,) hidden activation in context A.
        h_contextB: (d,) hidden activation in context B.
        result: ProbeResult from train_binary_probe.

    Returns:
        Scalar (w^T h_A + b) - (w^T h_B + b) = w^T (h_A - h_B).
    """
    return float(np.dot(result.w, h_contextA - h_contextB))


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


if __name__ == "__main__":
    # Smoke test: synthetic data with known ground truth direction
    np.random.seed(0)

    # Hyperparameters
    d = 32
    N = 200
    w_true = np.random.randn(d)
    w_true /= np.linalg.norm(w_true)  # normalize

    # Generate synthetic data
    X = np.random.randn(N, d)
    scores = X @ w_true
    y = (scores > 0).astype(int)

    # Check class balance
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    print(f"Generated data: N={N}, d={d}")
    print(f"Class distribution: positive={n_pos}, negative={n_neg}")

    # Train probe
    result = train_binary_probe(
        X,
        y,
        label_positive="true_positive",
        label_negative="true_negative",
        C=1.0,
        test_frac=0.2,
        seed=0,
    )

    # Verify recovery of true direction
    cos_sim = _cosine_similarity(result.w, w_true)
    print(f"\nProbe training results:")
    print(f"  Train accuracy: {result.train_acc:.4f}")
    print(f"  Test accuracy:  {result.test_acc:.4f}")
    print(f"  Cosine(w_probe, w_true): {cos_sim:.4f}")
    print(f"  n_train={result.n_train}, n_test={result.n_test}")

    if cos_sim > 0.95:
        print("\n✓ Smoke test PASSED: cosine similarity > 0.95")
    else:
        print(f"\n✗ Smoke test FAILED: cosine similarity {cos_sim:.4f} ≤ 0.95")
