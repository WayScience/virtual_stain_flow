"""
/transforms/normalizations.py

This module defines normalization transforms for image preprocessing using 
Albumentations. 

Classes:
- MaxScaleNormalize: Max Scale Normalization transform for converting
  gray-scale images to a [0, 1] range based on a specified normalization factor.
- ZScoreNormalize: Z-Score Normalization transform for normalizing images
  to have a mean of 0 and a standard deviation of 1 per image or with
  pre-specified mean and standard deviation values.
"""

from typing import Optional, Union, Literal

import numpy as np

from .base_transform import LoggableTransform

class MaxScaleNormalize(LoggableTransform):
    """
    Max Scale Normalization transform for converting gray-scale images to
    a [0, 1] range based on a specified normalization factor.
    """
    def __init__(
            self, 
            normalization_factor: Union[float, int, Literal['16bit', '8bit']],
            name: str = "MaxScaleNormalize", 
            p: float = 1.0,
            **kwargs
        ):

        super().__init__(
            name, 
            p=p, 
            **kwargs
        )

        if not isinstance(normalization_factor, (int, float)):
            raise TypeError(
                "Expected normalization factor to be a number (int or float), "
                f"got {type(normalization_factor).__name__} instead."
            )
        elif normalization_factor <= 0:
            raise ValueError(
                "Normalization factor must be greater than zero."
            )
        
        if isinstance(normalization_factor, int):
            normalization_factor = float(normalization_factor)
        elif isinstance(normalization_factor, float):
            pass
        elif isinstance(normalization_factor, str):
            if normalization_factor == '16bit':
                normalization_factor = 2 ** 16 - 1
            elif normalization_factor == '8bit':
                normalization_factor = 2 ** 8 - 1
            else:
                raise ValueError(
                    "Allowed literals of normalization_factor are '16bit' and '8bit'." 
                    f"Got {normalization_factor} instead."
                )
        else:
            raise TypeError(
                "Expected normalization factor to be a number (int or float) "
                "or a literal ('16bit', '8bit'), "
                f"got {type(normalization_factor).__name__} instead."
            )
        
        if normalization_factor <= 0:
            raise ValueError(
                "Normalization factor must be greater than zero."
            )
        
        self._normalization_factor = normalization_factor

    @property
    def normalization_factor(self) -> float:
        """
        Get the normalization factor.
        
        :return: Normalization factor.
        """
        return self._normalization_factor
    
    def __repr__(self) -> str:
        """
        String representation of the transform.
        
        :return: String representation of the transform.
        """
        return (f"{self.__class__.__name__}(name={self._name}, "
                f"normalization_factor={self._normalization_factor}, "
                f"p={self.p})"
        )

    def apply(
        self, 
        img: np.ndarray, 
        **params
    ) -> np.ndarray:
        
        if isinstance(img, np.ndarray):
            # Normalize the image using the normalization factor
            return img / self._normalization_factor
        else:
            raise TypeError(
                "Expected input image to be a NumPy array, "
                f"got {type(img).__name__} instead."
            )
        
    def to_config(self):
        return {
            "class": self.__class__.__name__,
            "name": self._name,
            "params": {
                "normalization_factor": self._normalization_factor,
                "p": self.p
            }
        }

class ZScoreNormalize(LoggableTransform):
    """
    Z-Score Normalization transform for normalizing images to have a
    mean of 0 and a standard deviation of 1 per image or with pre-specified
    mean and standard deviation values.
    """
    def __init__(
        self,
        name: str = "ZScoreNormalize",
        mean: Optional[float] = None,
        std: Optional[float] = None,
        p: float = 1.0,
    ):
        
        super().__init__(
            name="ZScoreNormalize",
            p=p
        )

        if mean is None:
            pass
        elif isinstance(mean, (int, float)):
            pass
        else:
            raise TypeError(
                "Expected mean to be a number (int or float), "
                f"got {type(mean).__name__} instead."
            )
        self._mean = mean
        
        if std is None:
            pass
        elif isinstance(std, (int, float)):
            if std <= 0:
                raise ValueError(
                    "Standard deviation must be greater than zero."
                )
        else:
            raise TypeError(
                "Expected std to be a number (int or float), "
                f"got {type(std).__name__} instead."
            )
        self._std = std

    @property
    def mean(self) -> Optional[float]:
        """
        Get the mean value used for normalization.
        
        :return: Mean value or None if not set.
        """
        return self._mean
    @property
    def std(self) -> Optional[float]:
        """
        Get the standard deviation value used for normalization.
        
        :return: Standard deviation value or None if not set.
        """
        return self._std
    
    def __repr__(self) -> str:
        """
        String representation of the transform.
        
        :return: String representation of the transform.
        """
        return (f"{self.__class__.__name__}(name={self._name}, "
                f"mean={self._mean}, std={self._std}, "
                f"p={self.p})"
        )

    def apply(self, img, **params):        

        if isinstance(img, np.ndarray):
            
            mean = self._mean or img.mean(axis=(1, 2), keepdims=True)
            std = self._std or img.std(axis=(1, 2), keepdims=True)

            if np.any(std == 0):
                raise ValueError(
                    "Standard deviation is zero, cannot normalize."
                )
            
            return (img - mean) / std

        else:
            raise TypeError(
                "Expected input image to be a NumPy array, "
                f"got {type(img).__name__} instead."
            )
        
    def to_config(self):
        return {
            "class": self.__class__.__name__,
            "name": self._name,
            "params": {
                "mean": self.mean,
                "std": self.std,
                "p": self.p
            }
        }
