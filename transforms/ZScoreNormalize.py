from albumentations import ImageOnlyTransform
import numpy as np

"""
Wrote this to get z score normalizae to work with albumentations
"""
class ZScoreNormalize(ImageOnlyTransform):
    """Z-score normalize each image"""

    def __init__(self, _mean=None, _std=None, _always_apply=False, _p=0.5):
        """
        Initializes the ZScoreNormalize transform.

        :param _mean: Precomputed mean for normalization (optional). If None, compute per-image mean.
        :type _mean: float, optional
        :param _std: Precomputed standard deviation for normalization (optional). If None, compute per-image std.
        :type _std: float, optional
        :param _always_apply: If True, always apply this transformation.
        :type _always_apply: bool
        :param _p: Probability of applying this transformation.
        :type _p: float
        """
        super(ZScoreNormalize, self).__init__(_always_apply, _p)
        self.__mean = _mean
        self.__std = _std

    @property
    def mean(self):
        return self.__mean

    @property
    def std(self):
        return self.__std

    def apply(self, _img, **params):
        """
        Apply z-score normalization to the image.

        :param _img: Input image as a numpy array.
        :type _img: np.ndarray
        :return: Z-score normalized image.
        :rtype: np.ndarray
        :raises TypeError: If the input image is not a numpy array.
        :raises ValueError: If the standard deviation is zero.
        """
        if not isinstance(_img, np.ndarray):
            raise TypeError("Unsupported image type for transform (Should be a numpy array)")

        mean = self.__mean if self.__mean is not None else _img.mean()
        std = self.__std if self.__std is not None else _img.std()

        if std == 0:
            raise ValueError("Standard deviation is zero; cannot perform z-score normalization.")

        return (_img - mean) / std
    
    def invert(self, _img, **kwargs):
        """
        Invert the z-score normalization.
        If this transform is applied on image basis (without global mean and std)
        Will simply return the z score transformed image back

        :param _img: Input image as a numpy array.
        :type _img: np.ndarray
        :return: Inverted image.
        :rtype: np.ndarray
        :raises TypeError: If the input image is not a numpy array.
        :raises ValueError: If the standard deviation is zero.
        """
        if not isinstance(_img, np.ndarray):
            raise TypeError("Unsupported image type for transform (Should be a numpy array)")

        if self.__mean is None or self.__std is None:
            mean = kwargs.get("mean", None)
            std = kwargs.get("std", None)
            if mean is None or std is None:
                return _img
            else:
                return (_img * std) + mean
        else:
            mean = self.__mean if self.__mean is not None else _img.mean()
            std = self.__std if self.__std is not None else _img.std()

            if std == 0:
                raise ValueError("Standard deviation is zero; cannot perform z-score normalization.")
            
            return (_img * std) + mean
