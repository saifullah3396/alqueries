import torch

def margin_sampling(probs: torch.Tensor, n_query: int) -> torch.Tensor:
    """
    Margin Sampling strategy for active learning.

    Args:
        probs (torch.Tensor): Shape (N, C) - predicted probabilities
        n_query (int): Number of samples to select

    Returns:
        torch.Tensor: Indices of selected samples (shape: n_query)
    """

    # # Step 1: Sort probabilities in descending order
    # sorted_probs, _ = torch.sort(probs, dim=1, descending=True)

    # # Step 2: Compute margin (top1 - top2)
    # margins = sorted_probs[:, 0] - sorted_probs[:, 1]

    top2_probs, _ = torch.topk(probs, k=2, dim=1)
    margins = top2_probs[:, 0] - top2_probs[:, 1]

    # Step 3: Get indices of smallest margins (most uncertain)
    query_indices = torch.argsort(margins)

    # Step 4: Select top n_query samples
    return query_indices[:n_query]