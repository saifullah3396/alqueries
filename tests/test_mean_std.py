import torch

from alqueries.strategies.mean_std import mean_std

def test_mean_std():
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

    selected = mean_std(probs, n_query=2)

    assert selected[0].item() == 2
    assert len(selected) == 2