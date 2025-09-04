"""
/transforms/base_transform.py

This module defines a base class for image transforms using Albumentations.
It provides a serializable transform class for (albumentation based) image
transformations, facilitating easy logging.
"""

from abc import ABC, abstractmethod

from albumentations import ImageOnlyTransform
import numpy as np

class LoggableTransform(ABC, ImageOnlyTransform):
    """
    Base class for loggable image transforms using Albumentations.
    Serializable transform class for (albumentation based) image transformations,
    facilitating easy logging. 
    """
    def __init__(
            self, 
            name: str,
            p: float = 1.0,
            **kwargs
        ):

        self._name = name

        super().__init__(
            p=p,
            **kwargs
        )

    @property
    def name(self) -> str:
        """
        Get the name of the transform.
        
        :return: Name of the transform.
        """
        return self._name

    @abstractmethod
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Apply the transformation to the input image.
        
        :param img: Input image as a NumPy array.
        :return: Transformed image as a NumPy array.
        """
        pass

    def __repr__(self) -> str:
        """
        String representation of the transform.
        
        :return: String representation of the transform.
        """
        return (f"{self.__class__.__name__}(name={self._name}, "
                f"p={self.p})"
        )
    
    @abstractmethod
    def to_config(
            self, 
            **kwargs
        ) -> dict:
        """
        Serialize the transform to a dictionary exportable as JSON.

        Should create a dictionary with:
        - 'class': The class name of the transform.
        - 'name': The name of the transform.
        - 'params': A dictionary of parameters specific to the transform.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> 'LoggableTransform':
        return cls(
            name=config['name'],
            **config['params']
        )