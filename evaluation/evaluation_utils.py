from collections import defaultdict
from typing import List, Callable, Union

import pandas as pd
import torch
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