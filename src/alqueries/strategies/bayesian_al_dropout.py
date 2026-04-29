import torch

def bayesian_al_dropout(probs: torch.Tensor, n_query: int) -> torch.Tensor:

    mean_probs = torch.mean(probs, dim=0)
    entropy_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
    entropy_per_sample = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)
    mean_entropy = torch.mean(entropy_per_sample, dim=0)

    score = entropy_mean - mean_entropy
    query_indices = torch.argsort(score, descending=True)

    return query_indices[:n_query]