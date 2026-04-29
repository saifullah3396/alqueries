import torch

def test_margin_sampling():
    from alqueries.strategies.margin_sampling import margin_sampling

    # Sample probabilities for 5 samples and 3 classes
    probs = torch.tensor([
        [0.7, 0.2, 0.1],  # margin = 0.5
        [0.4, 0.35, 0.25], # margin = 0.05 (most uncertain)
        [0.6, 0.3, 0.1],  # margin = 0.3
        [0.5, 0.45, 0.05], # margin = 0.05 (most uncertain)
        [0.8, 0.15, 0.05] # margin = 0.65 (least uncertain)
    ], dtype=torch.float32)

    n_query = 2
    selected_indices = margin_sampling(probs, n_query)

    assert len(selected_indices) == n_query, "Should select the correct number of samples"
    assert set(selected_indices.tolist()) == {1, 3}, "Should select the most uncertain samples"
    print("Test passed: margin_sampling returns valid indices.")