from albumentations import ImageOnlyTransform
import numpy as np

class PixelDepthTransform(ImageOnlyTransform):
    """
    Transform to convert images from a specified bit depth to another bit depth (e.g., 16-bit to 8-bit).
    Automatically scales pixel values up or down to the target bit depth.
    The only supported bit depths are 8, 16, and 32.
    """

    def __init__(self, 
                 src_bit_depth: int = 16, 
                 target_bit_depth: int = 8, 
                 _always_apply: bool = True, 
                 _p: float = 1.0):
        """
        Initializes the PixelDepthTransform.

        :param src_bit_depth: Bit depth of the input image (e.g., 16 for 16-bit).
        :type src_bit_depth: int
        :param target_bit_depth: Bit depth to scale the image to (e.g., 8 for 8-bit).
        :type target_bit_depth: int
        :param _always_apply: Whether to always apply the transform.
        :type _always_apply: bool
        :param _p: Probability of applying the transform.
        :type _p: float
        :raises ValueError: If the source or target bit depth is not supported.
        """
        if src_bit_depth not in [8, 16, 32]:
            raise ValueError("Unsupported source bit depth (should be 8 or 16)")
        if target_bit_depth not in [8, 16, 32]:
            raise ValueError("Unsupported target bit depth (should be 8 or 16)")

        super(PixelDepthTransform, self).__init__(_always_apply, _p)
        self.src_bit_depth = src_bit_depth
        self.target_bit_depth = target_bit_depth

    def apply(self, img, **kwargs):
        """
        Apply the bit depth transformation.

        :param img: Input image as a numpy array.
        :type img: np.ndarray
        :return: Transformed image scaled to the target bit depth.
        :rtype: np.ndarray
        :raises TypeError: If the input image is not a numpy array.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Unsupported image type for transform (should be a numpy array)")

        # Maximum pixel value based on source and target bit depth
        src_max_val = (2 ** self.src_bit_depth) - 1
        target_max_val = (2 ** self.target_bit_depth) - 1

        if self.target_bit_depth == 32:
            # Scale to the 32-bit integer range
            return ((img / src_max_val) * target_max_val).astype(np.uint32)
        else:
            # Standard conversion for 8-bit or 16-bit integers
            return ((img / src_max_val) * target_max_val).astype(
                np.uint8 if self.target_bit_depth == 8 else np.uint16
            )

    def invert(self, img, **kwargs):
        """
        Optionally invert the bit depth transformation (useful for debugging or preprocessing).

        :param img: Transformed image as a numpy array.
        :type img: np.ndarray
        :return: Image restored to the original bit depth.
        :rtype: np.ndarray
        :raises TypeError: If the input image is not a numpy array.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Unsupported image type for inversion (should be a numpy array)")

        target_max_val = (2 ** self.target_bit_depth) - 1
        src_max_val = (2 ** self.src_bit_depth) - 1

        # Invert scaling back to original bit depth
        img = (img / target_max_val) * src_max_val
        return img.astype(np.uint16) if self.src_bit_depth == 16 else img

    def __repr__(self):
        return f"PixelDepthTransform(src_bit_depth={self.src_bit_depth}, target_bit_depth={self.target_bit_depth})"