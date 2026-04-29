import numpy as np
import torch

from alqueries import get_strategy


def test_least_confidence_dropout():
    strategy = get_strategy("least_confidence_dropout")

    # T=3 dropout passes, 3 samples total, 2 classes.
    # We use identical passes so mean == any single pass — easy to verify.
    #   idx 0: [0.9, 0.1]  → uncertainty = 0.10  (confident)
    #   idx 1: [0.5, 0.5]  → uncertainty = 0.50  (most uncertain)
    #   idx 2: [0.6, 0.4]  → uncertainty = 0.40  (second)
    single = torch.tensor(
        [[0.9, 0.1], [0.5, 0.5], [0.6, 0.4]], dtype=torch.float32
    )
    probs = single.unsqueeze(0).expand(3, -1, -1)  # (T=3, N=3, C=2)

    unlabeled = np.array([0, 1, 2])
    picked = strategy.query(unlabeled_indices=unlabeled, n_samples=2, probs=probs)

    assert len(picked) == 2, "Number of queried indices should match n_samples"
    assert set(picked.tolist()) == {1, 2}, "Should pick the two most uncertain samples"
    print("Test passed: least_confidence_dropout returns valid indices.")
