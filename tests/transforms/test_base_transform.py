import pytest
import numpy as np
from virtual_stain_flow.transforms.base_transform import LoggableTransform
from albumentations import ImageOnlyTransform

class ConcreteTransform(LoggableTransform):
    """Concrete implementation for testing purposes."""
    
    def __init__(self, name: str, param1: int = 10, p: float = 1.0, **kwargs):
        self.param1 = param1
        super().__init__(name=name, p=p, **kwargs)
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Simple transformation for testing
        return img + self.param1
    
    def to_config(self, **kwargs) -> dict:
        return {
            'class': self.__class__.__name__,
            'name': self.name,
            'params': {
                'param1': self.param1,
                'p': self.p
            }
        }


class TestLoggableTransform:
    
    def test_init(self):
        transform = ConcreteTransform(name="test_transform", param1=5, p=0.8)
        assert transform.name == "test_transform"
        assert transform.param1 == 5
        assert transform.p == 0.8
    
    def test_name_property(self):
        transform = ConcreteTransform(name="my_transform")
        assert transform.name == "my_transform"
    
    def test_apply(self):
        transform = ConcreteTransform(name="test", param1=10)
        img = np.array([[1, 2], [3, 4]])
        result = transform.apply(img)
        expected = np.array([[11, 12], [13, 14]])
        np.testing.assert_array_equal(result, expected)
    
    def test_repr(self):
        transform = ConcreteTransform(name="test_transform", p=0.5)
        expected = "ConcreteTransform(name=test_transform, p=0.5)"
        assert repr(transform) == expected
    
    def test_to_config(self):
        transform = ConcreteTransform(name="test", param1=20, p=0.7)
        config = transform.to_config()
        expected = {
            'class': 'ConcreteTransform',
            'name': 'test',
            'params': {
                'param1': 20,
                'p': 0.7
            }
        }
        assert config == expected
    
    def test_from_config(self):
        config = {
            'name': 'restored_transform',
            'params': {
                'param1': 15,
                'p': 0.9
            }
        }
        transform = ConcreteTransform.from_config(config)
        assert transform.name == "restored_transform"
        assert transform.param1 == 15
        assert transform.p == 0.9
    
    def test_abstract_methods_raise_error(self):
        # Test that LoggableTransform cannot be instantiated directly
        with pytest.raises(TypeError):
            LoggableTransform(name="test")
    
    def test_inheritance_from_imageonlytransform(self):
        transform = ConcreteTransform(name="test")
        assert isinstance(transform, ImageOnlyTransform)