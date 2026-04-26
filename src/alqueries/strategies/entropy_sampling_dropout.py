import numpy as np
from alqueries.base import QueryStrategy


class EntropySamplingDropout(QueryStrategy):
    def query(self, probs, indices, n_samples):
        """
        Select samples based on the average entropy of predictions across multiple stochastic forward passes.

        Args:
            probs (np.ndarray): Array of shape (T, N, C) containing softmax probabilities for T forward passes,
                                N unlabeled samples, and C classes.
            indices (np.ndarray): Array of shape (N,) containing the indices of the unlabeled samples.
            n_samples (int): The number of samples to select.

        Returns:
            np.ndarray: Array of shape (n_samples,) containing the selected sample indices.
        """
        # Calculate the average probability across T forward passes
        avg_probs = np.mean(probs, axis=0)  # Shape: (N, C)

        # Calculate the entropy for each sample
        eps = 1e-10  # Small constant to avoid log(0)
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10), axis=1)  # Shape: (N,)

        # Select the indices of the samples with the highest entropy
        selected_indices = np.argsort(entropy)[-n_samples:]

        return indices[selected_indices]
