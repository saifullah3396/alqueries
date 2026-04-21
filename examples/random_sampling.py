import numpy as np
import torch
from torch.utils.data import TensorDataset

from alqueries import QueryEngine, get_strategy

N_POOL, N_FEATURES, N_CLASSES = 100, 16, 5

dataset = TensorDataset(
    torch.randn(N_POOL, N_FEATURES),
    torch.randint(0, N_CLASSES, (N_POOL,)),
)

engine = QueryEngine(dataset, labeled_indices=np.arange(20))
strategy = get_strategy("random", seed=7)

picked = engine.query(strategy, n_samples=10)
print("random picks:", picked)
