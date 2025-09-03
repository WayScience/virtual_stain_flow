import torch

from .AbstractMetrics import AbstractMetrics

"""
Adapted from https://github.com/WayScience/nuclear_speckles_analysis
"""
class SSIM(AbstractMetrics):
    """Computes and tracks the Structural Similarity Index Measure (SSIM)."""

    def __init__(self, _metric_name: str, _max_pixel_value: int = 1):
        """
        Initializes the SSIM metric.

        :param _metric_name: The name of the metric.
        :param _max_pixel_value: The maximum possible pixel value of the images, by default 1.
        :type _max_pixel_value: int, optional
        """

        super(SSIM, self).__init__(_metric_name)
        
        self.__max_pixel_value = _max_pixel_value

    def forward(self, _generated_outputs: torch.Tensor, _targets: torch.Tensor):
        """
        Computes the Structural Similarity Index Measure (SSIM) between the generated outputs and the target images.

        :param _generated_outputs: The tensor containing the generated output images.
        :type _generated_outputs: torch.Tensor
        :param _targets: The tensor containing the target images.
        :type _targets: torch.Tensor
        :return: The computed SSIM value.
        :rtype: torch.Tensor
        """
        
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