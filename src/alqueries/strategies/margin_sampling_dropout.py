import torch

def margin_sampling_dropout(probs: torch.Tensor, n_query: int) -> torch.Tensor:
    """
    Margin Sampling strategy for active learning with dropout.

    Args:
        probs (torch.Tensor): Shape (T, N, C) - predicted probabilities from multiple dropout passes
        n_query (int): Number of samples to select

    Returns:
        torch.Tensor: Indices of selected samples (shape: n_query)
    """

    # Step 1: Average probabilities across dropout passes
    # avg_probs = probs.mean(dim=0)  # Shape: (N, C)

    # Step 2: Compute margin (top1 - top2)
    top2_probs = torch.topk(probs, k=2, dim=2).values
    margins = top2_probs[:, :, 0] - top2_probs[:, :, 1]
    mean_margins = torch.mean(margins, dim=0)  # Average margin across dropout passes

    # Step 3: Get indices of smallest margins (most uncertain)
    query_indices = torch.argsort(mean_margins)

    # Step 4: Select top n_query samples
    return query_indices[:n_query]