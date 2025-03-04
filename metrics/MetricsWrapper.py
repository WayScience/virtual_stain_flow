import torch

from .AbstractMetrics import AbstractMetrics

class MetricsWrapper(AbstractMetrics):
    """Metrics wrapper class that wraps a pytorch module
    and calls it forward pass function to accumulate the metric 
    values across batches
    """

    def __init__(self, _metric_name: str, module: torch.nn.Module):
        """
        Initialize the MetricsWrapper class with the metric name and the module.

        :param _metric_name: The name of the metric.
        :param module: The module to be wrapped. Needs to have a forward function.
        :type module: torch.nn.Module
        """
        
        super(MetricsWrapper, self).__init__(_metric_name)
        self._module = module

    def forward(self,_generated_outputs: torch.Tensor, _targets: torch.Tensor):
        return self._module(_generated_outputs, _targets).mean()