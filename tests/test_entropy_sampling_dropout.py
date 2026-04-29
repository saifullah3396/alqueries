import numpy as np
from alqueries.strategies.entropy_sampling_dropout import EntropySamplingDropout


def test_entropy_sampling_dropout():
    np.random.seed(42)

    T = 5  # Number of stochastic forward passes
    N = 20  # Number of unlabeled samples
    C = 3  # Number of classes

    # Simulate softmax probabilities for T forward passes
    probs = np.random.rand(T, N, C)
    probs = probs / probs.sum(axis=2, keepdims=True)  # Normalize to

    indices = np.arange(N)
    n_samples = 5

    strategy = EntropySamplingDropout()
    selected_indices = strategy.query(
        probs, indices, n_samples
    )
    print("Selected indices:", selected_indices)

    assert len(selected_indices) == n_samples, "Number of selected samples should match n_samples"
    assert all(idx in indices for idx in selected_indices), "Selected indices should be from the unlabeled indices"