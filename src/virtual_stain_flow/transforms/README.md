# Transform Subpackage Documentation

This subpackage provides image transformation utilities for virtual staining workflows, handling preprocessing, augmentation, and standardization of microscopy images.
To facilitate reproducible experiments, the transformations are also made serializable and loggable. 

---

## Overview

This subpackage consists of three modules: 
1. **`transform_utils.py`**: Contains type definitions and validation utilities for transform objects, defining acceptable transform types and providing runtime type checking capabilities for both standard Albumentations transforms and custom LoggableTransform classes.

2. **`base_transform.py`**: Defines the abstract `LoggableTransform` base class that extends Albumentations' `ImageOnlyTransform`, adding serialization capabilities, naming conventions, and standardized logging interfaces for scientific reproducibility.

3. **`normalizations.py`**: Implements concrete normalization transforms including `MaxScaleNormalize` for scaling images to [0,1] range and `ZScoreNormalize` for statistical standardization, both inheriting from `LoggableTransform` to ensure proper integration with the package's logging and configuration systems.

---

## Usage:
See examples for in context use with datasets
```python
from normalizations import MaxScaleNormalize

scale_transform = MaxScaleNormalize(
    normalization_factor='16bit',
        name="Scale16BitImages",
        p=1.0
)
```
