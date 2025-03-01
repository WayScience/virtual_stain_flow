from collections import defaultdict
from typing import List, Dict, Callable, Union

import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

def evaluate_metrics(
    _model: torch.nn.Module,
    _dataset: torch.utils.data.Dataset,
    _metrics: List[Union[Callable, torch.nn.Module]],
    _device:str='cpu'
):    
    metrics = defaultdict(list)
    _model.to(_device)
    _model.eval()

    data_loader = DataLoader(_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():    
        for input, target in data_loader:
            input = input.to(_device)
            target = target.to(_device)
            output = _model(input)            
            for _metric in _metrics:
                metrics[_metric.__class__.__name__].append(_metric(output, target).item())
    
    return pd.DataFrame(metrics)

def evaluate_per_image_metric(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metrics: List[Module]
) -> pd.DataFrame:
    """
    Computes a set of metrics on a per-image basis and returns the results as a pandas DataFrame.

    :param predictions: Predicted images, shape (N, C, H, W).
    :type predictions: torch.Tensor
    :param targets: Target images, shape (N, C, H, W).
    :type targets: torch.Tensor
    :param metrics: List of metric functions to evaluate.
    :type metrics: List[torch.nn.Module]

    :return: A DataFrame where each row corresponds to an image and each column corresponds to a metric.
    :rtype: pd.DataFrame
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")

    results = []

    for i in range(predictions.shape[0]):  # Iterate over images
        pred, target = predictions[i].unsqueeze(0), targets[i].unsqueeze(0)  # Keep batch dimension
        metric_scores = {metric.__class__.__name__: metric.forward(target, pred).item() for metric in metrics}
        results.append(metric_scores)

    return pd.DataFrame(results)