import torch

def mean_std(probs: torch.Tensor, n_query: int) -> torch.Tensor:

    var_probs = torch.var(probs, dim=0)
    uncertainties = torch.sum(var_probs, dim=1 )
    query_indices = torch.argsort(uncertainties, descending=True)

    return query_indices[: n_query]