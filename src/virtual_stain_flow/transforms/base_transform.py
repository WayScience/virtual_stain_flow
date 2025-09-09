"""
base_transform.py

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
        Abstract method to be overridden by subclasses.
        Needed for Albumentations transform __call__ method.
        Apply the transformation to a given numpy array image. 
        
        :param img: Input image as a NumPy array.
        :return: Transformed image as a NumPy array.
        """
        raise NotImplementedError

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
        Abstract method to be overridden by subclasses.
        Should return a JSON serializable dictionary that describes the transform
        configuration, and, more importantly, contain sufficient information for 
        recreation of the transform instance via the `from_config` class method. 
        Depending on how the subclass implementation of the `from_config`
        method is defined, the dictionary may or may not need to serve the purpose of kwargs.

        Should create a dictionary with:
        - 'class': The class name of the transform.
        - 'name': The name of the transform.
        - 'params': A dictionary of parameters specific to the transform.
        """

        # Example implementation:
        # return {
        #     'class': self.__class__.__name__,
        #     'name': self._name,
        #     'params': self.get_params()
        #     # add more parameters as needed
        # } 

        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> 'LoggableTransform':
        """
        Create an instance of the transform from a configuration dictionary.
        The config dictionary should match the structure defined in the `to_config` method.
        By default, this method assumes the config dictionary contains:
        - 'name': The name of the transform.
        - 'params': A dictionary of parameters specific to the transform.

        Depending on the subclass implementation of the `to_config` method, this method
        may need to be overridden to correctly instantiate the transform.
        """
        return cls(
            name=config['name'],
            **config['params']
        )
