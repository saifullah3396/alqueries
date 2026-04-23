from __future__ import annotations

import numpy as np
import torch

from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy


@register_strategy("least_confidence_dropout")
class LeastConfidenceDropoutSampling(QueryStrategy):
    """
    MC-Dropout variant of Least Confidence.

    The outer pipeline runs the model T times with dropout ENABLED during
    inference and passes all T probability tensors in here. This strategy
    simply averages them and applies least-confidence scoring — it does
    NOT hold a reference to the model or run inference itself.

    score(x) = 1 - max_c  mean_T[ P(y=c | x, theta_t) ]

    Args:
        unlabeled_indices: 1-D array of absolute dataset indices in the pool.
        n_samples:         How many indices to return.
        probs:             Softmax probabilities from T dropout forward passes,
                           shape (T, N_total, C). Sliced inside using
                           unlabeled_indices.
    Returns:
        1-D np.ndarray of n_samples absolute dataset indices.
    """

    def query(
        self,
        unlabeled_indices: np.ndarray,
        n_samples: int,
        *,
        probs: torch.Tensor,
        **_,
    ) -> np.ndarray:
        # probs: (T, N_total, C)
        # Step 1 — average the T dropout passes across the full dataset.
        mean_probs = probs.mean(dim=0)                       # (N_total, C)

        # Step 2 — slice to the unlabelled pool, same pattern as entropy.
        mean_probs = mean_probs[unlabeled_indices]           # (M, C)

        # Step 3 — identical to plain least confidence from here.
        max_probs, _ = mean_probs.max(dim=1)                 # (M,)
        uncertainties = 1.0 - max_probs                      # (M,)

        return unlabeled_indices[uncertainties.argsort(descending=True)[:n_samples]]
