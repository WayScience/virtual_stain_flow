import torch
from .AbstractMetrics import AbstractMetrics

"""
Adapted from https://github.com/WayScience/nuclear_speckles_analysis
"""
class PSNR(AbstractMetrics):
    """Computes and tracks the Peak Signal-to-Noise Ratio (PSNR)."""

    def __init__(self, _metric_name: str, _max_pixel_value: int = 1):

        super(PSNR, self).__init__(_metric_name)
        
        self.__max_pixel_value = _max_pixel_value

    def forward(self, _generated_outputs: torch.Tensor, _targets: torch.Tensor):

        mse = torch.mean((_generated_outputs - _targets) ** 2, dim=[2, 3])
        psnr = torch.where(
            mse == 0,
            torch.tensor(0.0),
            10 * torch.log10((self.__max_pixel_value**2) / mse),
        )

        return psnr.mean()

    @property
    def metric_name(self):
        return self.__metric_name