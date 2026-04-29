import torch
from alqueries.strategies.bayesian_al_dropout import bayesian_al_dropout
# from alqueries.strategies import bayesian_al_dropout

def test_bayesian_dropout():
    probs = torch.tensor([
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.5, 0.5]
        ],
        [
            [0.85, 0.15],
            [0.55, 0.45],
            [0.2, 0.8]
        ]
    ])

    selected = bayesian_al_dropout(probs, n_query=2)
    assert selected[0].item() == 2
    assert len(selected) == 2