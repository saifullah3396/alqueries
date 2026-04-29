import torch

from alqueries.strategies.margin_sampling_dropout import margin_sampling_dropout

def test_margin_sampling_dropout():
   probs = torch.tensor([
      [
      [0.9, 0.1],
      [0.6, 0.4],
      [0.51, 0.49]
    ],
    [
       [0.85, 0.15],
       [0.55, 0.45],
       [0.52, 0.48],
    ]
    ])
   selected = margin_sampling_dropout(probs, n_query=2)

   assert selected[0].item() == 2
   assert selected[1].item() == 1
   assert len(selected) == 2
