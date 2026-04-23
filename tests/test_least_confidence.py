import numpy as np
import torch
from alqueries.strategies.least_confidence import LeastConfidenceSampling

def test_least_confidence():
    strategy = LeastConfidenceSampling()

    # Simulate a dataset of 5 samples and 3 classes.
    # Each row is the softmax probabilities for that sample.
    probs = np.array([
        [0.34, 0.33, 0.33],  # Uncertain (score 0.66)
        [0.95, 0.03, 0.02],  # Confident (score 0.05)
        [0.50, 0.25, 0.25],  # Uncertain (score 0.50)
        [0.80, 0.10, 0.10],  # Confident (score 0.20)
        [0.60, 0.20, 0.20],  # Uncertain (score 0.40)
    ])
    unlabeled_indices = np.array([0, 1, 2, 3, 4])
    n_samples = 2

    selected_indices = strategy.query(
        unlabeled_indices=unlabeled_indices,
        n_samples=n_samples,
        probs=torch.tensor(probs),
    )

    # We expect to select the two most uncertain samples: indices [0, 2].
    assert set(selected_indices) == {0, 2}, f"Expected indices [0,2], got {selected_indices}"
