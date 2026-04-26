from __future__ import annotations
import numpy as np
import torch
from alqueries.base import QueryStrategy
from alqueries.registry import register_strategy

@register_strategy("least_confidence")
class LeastConfidenceSampling(QueryStrategy):
    """
    Selects samples where the model is least confident in its top prediction.

    Score(x) = 1 - max_c P(y=c | x)

    A model predicting [0.34, 0.33, 0.33] scores 0.66 (very uncertain).
    A model predicting [0.95, 0.03, 0.02] scores 0.05 (very confident).
    We pick the samples with the HIGHEST scores (most uncertain).

    Args:
        unlabeled_indices: 1-D array of absolute dataset indices in the pool.
        n_samples:         How many indices to return.
        probs:             Softmax probabilities for the FULL dataset,
                           shape (N_total, C). Row i = sample at absolute
                           index i. Sliced inside using unlabeled_indices.
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
        # Slice to only the unlabelled pool rows — same pattern as entropy.
        probs = probs[unlabeled_indices]                     # (M, C)

        # The highest class probability tells us how confident the model is.
        # 1 minus that is the uncertainty score.
        max_probs, _ = probs.max(dim=1)                      # (M,)
        uncertainties = 1.0 - max_probs                      # (M,)
        # Most uncertain = highest score. argsort descending, take first n.
        return unlabeled_indices[uncertainties.argsort(descending=True)[:n_samples]]


