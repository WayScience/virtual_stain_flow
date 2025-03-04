import torch

from .AbstractMetrics import AbstractMetrics

"""
Adapted from https://github.com/WayScience/nuclear_speckles_analysis
"""
class PSNR(AbstractMetrics):
    """Computes and tracks the Peak Signal-to-Noise Ratio (PSNR)."""

    def __init__(self, _metric_name: str, _max_pixel_value: int = 1):
        """
        Initializes the PSNR metric.

        :param _metric_name: The name of the metric.
        :param _max_pixel_value: The maximum possible pixel value of the images, by default 1.
        :type _max_pixel_value: int, optional
        """

        super(PSNR, self).__init__(_metric_name)
        
        self.__max_pixel_value = _max_pixel_value

    def forward(self, _generated_outputs: torch.Tensor, _targets: torch.Tensor):
        """
        Computes the Peak Signal-to-Noise Ratio (PSNR) between the generated outputs and the target images.

        :param _generated_outputs: The tensor containing the generated output images.
        :type _generated_outputs: torch.Tensor
        :param _targets: The tensor containing the target images.
        :type _targets: torch.Tensor
        :return: The computed PSNR value.
        :rtype: torch.Tensor
        """

        mse = torch.mean((_generated_outputs - _targets) ** 2, dim=[2, 3])
        psnr = torch.where(
            mse == 0,
            torch.tensor(0.0),
            10 * torch.log10((self.__max_pixel_value**2) / mse),
        )

        return psnr.mean()