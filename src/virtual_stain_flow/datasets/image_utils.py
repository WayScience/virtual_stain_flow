"""
image_utils.py

This module centralizes image cropping and rotation operations so that
    the dataset can focus on data handling logic.
The primary method `crop_and_rotate_image` is intended to be used by datasets
    that need to crop and optionally rotate images based on bounding box annotations.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def crop_and_rotate_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    rcx: Optional[float] = None,
    rcy: Optional[float] = None,
    angle: float = 0.0,
    min_angle: float = 1e-3
) -> np.ndarray:
    """
    Crop and optionally rotate an image according to bounding box and rotation parameters.
    This is the primary image processing method to be used by datasets that need to
    crop and rotate images based on bounding box annotations.

    :param image: Input image as numpy array with shape (C, H, W) or (C, H, W, K)
    :param bbox: Bounding box coordinates (xmin, ymin, xmax, ymax)
    :param rcx: Rotation center x coordinate (optional)
    :param rcy: Rotation center y coordinate (optional)
    :param angle: Rotation angle in degrees
    :param min_angle: Minimum angle threshold for rotation
    :return: Cropped (and possibly rotated) image
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Fast path: no rotation needed
    if angle == 0.0 or abs(angle) < min_angle or rcx is None or rcy is None:
        return image[:, ymin:ymax, xmin:xmax]

    # Prepare image for cv2 (convert from CHW to HWC format)
    cv_image = _prepare_image_for_cv2(image)
    
    # Apply rotation
    M = cv2.getRotationMatrix2D(center=(rcx, rcy), angle=angle, scale=1.0)
    rotated_cv = cv2.warpAffine(
        cv_image, M, (cv_image.shape[1], cv_image.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    
    # Convert back to original format and crop
    rotated_image = _restore_image_format(rotated_cv, image.shape)
    return rotated_image[:, ymin:ymax, xmin:xmax]


def _prepare_image_for_cv2(image: np.ndarray) -> np.ndarray:
    """
    Convert image from (C, H, W) or (C, H, W, K) to OpenCV format.
    Internal helper method used by crop_and_rotate_image.
    """
    if image.ndim == 3:
        # (C, H, W) -> (H, W, C)
        return np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        # (C, H, W, K) -> (H, W, C*K)
        C, H, W, K = image.shape
        return image.transpose(1, 2, 0, 3).reshape(H, W, C * K)
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}. "
                        f"Expected 3D (C, H, W) or 4D (C, H, W, K).")


def _restore_image_format(cv_image: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Convert image back from OpenCV format to original format.
    Internal helper method used by crop_and_rotate_image.
    """
    # Handle single channel case
    if cv_image.ndim == 2:
        cv_image = cv_image[:, :, np.newaxis]
    
    if len(original_shape) == 3:
        # Convert back to (C, H, W)
        return np.transpose(cv_image, (2, 0, 1))
    else:  # len(original_shape) == 4
        # Convert back to (C, H, W, K)
        C, H, W, K = original_shape
        rotated_H, rotated_W = cv_image.shape[:2]
        return cv_image.reshape(rotated_H, rotated_W, C, K).transpose(2, 0, 1, 3)
