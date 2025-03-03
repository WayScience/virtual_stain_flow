from typing import List, Optional

import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

def evaluate_per_image_metric(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metrics: List[Module],
    indices: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Computes a set of metrics on a per-image basis and returns the results as a pandas DataFrame.

    :param predictions: Predicted images, shape (N, C, H, W).
    :type predictions: torch.Tensor
    :param targets: Target images, shape (N, C, H, W).
    :type targets: torch.Tensor
    :param metrics: List of metric functions to evaluate.
    :type metrics: List[torch.nn.Module]
    :param indices: Optional list of indices to subset the dataset before inference. If None, all images are evaluated.
    :type indices: Optional[List[int]], optional

    :return: A DataFrame where each row corresponds to an image and each column corresponds to a metric.
    :rtype: pd.DataFrame
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")

    results = []

    if indices is None:
        indices = range(predictions.shape[0])

    for i in indices:  # Iterate over images/subset
        pred, target = predictions[i].unsqueeze(0), targets[i].unsqueeze(0)  # Keep batch dimension
        metric_scores = {metric.__class__.__name__: metric.forward(target, pred).item() for metric in metrics}
        results.append(metric_scores)

    return pd.DataFrame(results)