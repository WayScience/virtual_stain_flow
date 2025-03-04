from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

class AbstractMetrics(nn.Module, ABC):
    """Abstract class for metrics"""

    def __init__(self, _metric_name: str):
        
        super(AbstractMetrics, self).__init__()

        self.__metric_name = _metric_name
        
        self.__train_metric_values = []
        self.__val_metric_values = []

    @property
    def metric_name(self):
        """Defines the mertic name returned by the class."""
        return self.__metric_name
    
    @property
    def train_metric_values(self):
        """Returns the training metric values."""
        return self.__train_metric_values
    
    @property
    def val_metric_values(self):
        """Returns the validation metric values."""
        return self.__val_metric_values

    @abstractmethod
    def forward(self, 
                _generated_outputs: torch.tensor, 
                _targets: torch.tensor
                ) -> torch.tensor:
        """Computes the metric given information about the data."""
        pass
    
    def update(self, 
               _generated_outputs: torch.tensor, 
               _targets: torch.tensor, 
               validation: bool=False
               ) -> None:
        """Updates the metric with the new data."""
        if validation:
            self.__val_metric_values.append(self.forward(_generated_outputs, _targets))
        else:
            self.__train_metric_values.append(self.forward(_generated_outputs, _targets))

    def reset(self):
        """Resets the metric."""
        self.__train_metric_values = []
        self.__val_metric_values = []

    def compute(self, **kwargs):
        """
        Calls the aggregate_metrics method to compute the metric value for now
        In future may be used for more complex computations
        """
        return self.aggregate_metrics(**kwargs)

    def aggregate_metrics(self, aggregation: Optional[str] = 'mean'):
        """
        Aggregates the metric value over batches

        :param aggregation: The aggregation method to use, by default 'mean'
        :type aggregation: Optional[str]
        :return: The aggregated metric value for training and validation
        :rtype: Tuple[torch.tensor, torch.tensor]
        """

        if aggregation == 'mean':
            return \
            torch.mean(torch.stack(self.__train_metric_values)) if len(self.__train_metric_values) > 0 else None , \
            torch.mean(torch.stack(self.__val_metric_values)) if len(self.__val_metric_values) > 0 else None
        
        elif aggregation == 'sum':
            return \
            torch.sum(torch.stack(self.__train_metric_values)) if len(self.__train_metric_values) > 0 else None , \
            torch.sum(torch.stack(self.__val_metric_values)) if len(self.__val_metric_values) > 0 else None
        
        elif aggregation is None:
            return \
            torch.stack(self.__train_metric_values) if len(self.__train_metric_values) > 0 else None , \
            torch.stack(self.__val_metric_values) if len(self.__val_metric_values) > 0 else None
        
        else:
            raise ValueError(f"Aggregation method {aggregation} is not supported.")