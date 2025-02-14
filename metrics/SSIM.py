import torch
from .AbstractMetrics import AbstractMetrics

"""
Adapted from https://github.com/WayScience/nuclear_speckles_analysis
"""
class SSIM(AbstractMetrics):
    """Computes and tracks the Structural Similarity Index Measure (SSIM)."""

    def __init__(self, _metric_name: str, _max_pixel_value: int = 1):

        super(SSIM, self).__init__(_metric_name)
        
        self.__max_pixel_value = _max_pixel_value

    def forward(self, _generated_outputs: torch.Tensor, _targets: torch.Tensor):

        mu1 = _generated_outputs.mean(dim=[2, 3], keepdim=True)
        mu2 = _targets.mean(dim=[2, 3], keepdim=True)

        sigma1_sq = ((_generated_outputs - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma2_sq = ((_targets - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma12 = ((_generated_outputs - mu1) * (_targets - mu2)).mean(
            dim=[2, 3], keepdim=True
        )

        c1 = (self.__max_pixel_value * 0.01) ** 2
        c2 = (self.__max_pixel_value * 0.03) ** 2

        ssim_value = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        return ssim_value.mean()

    @property
    def metric_name(self):
        return self.__metric_name