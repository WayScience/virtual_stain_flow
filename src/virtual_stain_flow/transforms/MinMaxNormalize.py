from albumentations import ImageOnlyTransform
import numpy as np

"""
Adapted from https://github.com/WayScience/nuclear_speckles_analysis
"""
class MinMaxNormalize(ImageOnlyTransform):
    """Min-Max normalize each image"""

    def __init__(self, 
                 _normalization_factor: float, 
                 _always_apply: bool=False, 
                 _p: float=0.5):
        """
        Initializes the MinMaxNormalize transform.

        :param _normalization_factor: The factor by which to normalize the image.
        :type _normalization_factor: float
        :param _always_apply: If True, always apply this transformation.
        :type _always_apply: bool
        :param _p: Probability of applying this transformation.
        :type _p: float
        """
        super(MinMaxNormalize, self).__init__(_always_apply, _p)
        self.__normalization_factor = _normalization_factor

    @property
    def normalization_factor(self):
        return self.__normalization_factor

    def apply(self, _img, **kwargs):
        """
        Apply min-max normalization to the image.

        :param _img: Input image as a numpy array.
        :type _img: np.ndarray
        :return: Min-max normalized image.
        :rtype: np.ndarray
        :raises TypeError: If the input image is not a numpy array.
        """
        if isinstance(_img, np.ndarray):
            return _img / self.normalization_factor

        else:
            raise TypeError("Unsupported image type for transform (Should be a numpy array)")
        
    def invert(self, _img, **kwargs):
        """
        Invert the min-max normalization.

        :param _img: Input image as a numpy array.
        :type _img: np.ndarray
        :return: Inverted image.
        :rtype: np.ndarray
        :raises TypeError: If the input image is not a numpy array.
        """
        if isinstance(_img, np.ndarray):            
            return _img * self.normalization_factor
        else:
            raise TypeError("Unsupported image type for transform (Should be a numpy array)")
