import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import alqueries.strategies.kmeans  # noqa: F401
from alqueries import QueryEngine, get_strategy
from alqueries.extractors.classification import ClassificationFeatureExtractor

N_POOL, N_FEATURES, N_CLASSES = 100, 16, 5

dataset = TensorDataset(
    torch.randn(N_POOL, N_FEATURES),
    torch.randint(0, N_CLASSES, (N_POOL,)),
)

model = nn.Sequential(
    nn.Linear(N_FEATURES, 32),
    nn.ReLU(),
    nn.Linear(32, N_CLASSES),
)
extractor = ClassificationFeatureExtractor(model=model, device="cpu", input_key=0)

engine = QueryEngine(dataset, labeled_indices=np.arange(20), extractor=extractor)
strategy = get_strategy(
    "kmeans",
    pca_dim=None,
    cast_to_float16=False,
    kmeans_kwargs={"n_init": 10, "random_state": 0},
)

picked = engine.query(strategy, n_samples=10)
print("kmeans picks:", picked)
