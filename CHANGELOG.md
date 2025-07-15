# Changelog

All notable chagnes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


---

## [0.1.0] - 2025-03-03

### Added

#### Core Framework
- Introduced a minimal yet self-contained virtual staining framework structured around modular components for model training, dataset handling, transformations, metrics, and logging.

#### Models (`models`)
- Added `FNet`: Fully convolutional encoder-decoder for image-to-image translation.
- Added `UNet`: U-Net variant using bilinear interpolation for upsampling.
- Added GaN discriminators:
  - `PatchBasedDiscriminator`: Outputs a probability map.
  - `GlobalDiscriminator`: Outputs a global scalar probability.

#### Transforms (`transforms`)
- `MinMaxNormalize`: Albumentations transform for range-based normalization.
- `ZScoreNormalize`: Albumentations transform for z-score normalization.
- `PixelDepthTransform`: Converts between image bit depths (e.g., 16-bit to 8-bit).

#### Datasets (`datasets`)
- `ImageDataset`: Dynamically loads multi-channel microscopy images from a PE2LoadData-formatted CSV; supports input/target channel selection and Albumentations transforms.
- `PatchDataset`: Extends `ImageDataset` with configurable fixed-size cropping; supports object-centric patching and state retrieval (e.g., patch coordinates).
- `GenericImageDataset`: A simplified dataset for user-formatted directories using regex-based site/channel parsing.
- `CachedDataset`: Caches any of the above datasets in RAM to reduce I/O and speed up training.

#### Losses (`losses`)
- `AbstractLoss`: Base class defining standardized loss interface and trainer binding.
- `GeneratorLoss`: Combines image reconstruction and adversarial loss for training GaN generators.
- `WassersteinLoss`: Computes Wasserstein distance for GaN discriminator training.
- `GradientPenaltyLoss`: Adds gradient penalty to improve discriminator stability.

#### Metrics (`metrics`)
- `AbstractMetrics`: Base class for accumulating, aggregating, and resetting batch-wise metrics.
- `MetricsWrapper`: Wraps `torch.nn.Module` metrics with accumulation and aggregation logic.
- `PSNR`: Computes Peak Signal-to-Noise Ratio (PSNR) for image quality evaluation.

#### Callbacks (`callbacks`)
- `AbstractCallback`: Base class for trainer-stage hooks (`on_train_start`, `on_epoch_end`, etc.).
- `IntermediatePlot`: Visualizes model inference during training.
- `MlflowLogger`: Logs trainer metrics and losses to an MLflow server.

#### Training (`trainers`)
- `AbstractTrainer`: Defines a modular training loop with support for custom models, datasets, losses, metrics, and callbacks. Exposes extensible hooks for batch and epoch-level logic.

---

## [0.2.0] - 2025-06-19

### Added

#### Overhaul Phase 1/? - Logging Framework Overhaul
A minimal rework of the logging framework as the first step to a complete overhual of the `virtual_stain_flow` software. 

This version defines a new logging module that better integrates MLflow into the virtual staining model training process for a more comprehensive logging framework.

#### Introduced `logging.MlflowLogger` class
Notes: This class is simiar to the old `virtual_stain_flow.callback.MlflowLogger` class, but promoted to be an independent logger class, with ability to accept logger callbacks. Key design/functionality:
- Files/metrics/parameters produced by Logger callbacks gets automatically logged to MLflow appropriately instead of being independent products untracked. 
- Included some more pre-defined fine-grained run logging tags such as `experiment_type`, `model_architecture`, `target_channel_name`, `description` as logger class parameter.
- Has a `bind_trainer` and `unbind_trainer` methods to bind and unbind with the trainer instance during train step. 
- User controlled mlflow run cycle, no longer autoamtically ends with the train loop, so user can perform additional logging operation before explicity ending the run.
- Has exposed `log_artifact`, `log_metric`, and `log_param` methods for manual logging of artifacts, metrics, and parameters.
- Has some access point of trainer attributes for use by logger callbacks, but subject to optimization/change.

#### Introduced `trainers.trainerAbstractLoggingTrainer` class
Notes: This class subclasses the `virtual_stain_flow.trainers.AbstractTrainer` class and preserves most of its behavior and functionalities. 
Design/functionality change include:
- Binding of logger class moved from initialization to `train` method to reflect the design that logger instances should live with the training sessions.
- Early termination mode is now a parameter of the class to allow for selection of min/max optimzation mode.
- The `train` method loop invokes the logger life cycle methods:
    - `logger.on_train_start()`
    - `logger.on_epoch_start()`
    - `logger.on_epoch_end()`
    - `logger.on_train_end()` methods which in turn leads to logger's invocation of the logger callback methods.
- Requires child classes to implement the @abstract `save_model` method for unified handle for saving model weights that can be called by the logger.

#### Introduced `trainers.LoggingTrainer` class
Notes: This class is nearly identical to the old Trainer class, except that:
- It is the realization of the new `AbstractLoggingTrainer` class instead of the `AbstractTrainer` class.
- It overrides the parent class `save_model` method that defines saving of the model weight.

#### Introduced `logging.callbacks` module
Notes: This is a new module that is distinct from the existing `virtual_stain_flow.callbacks` module in that classes under this module are passed to `logging.MlflowLogger` instances as opposed to a `trainers.*` instances.     

##### The module currently contains:
- `AbstractLoggerCallback` class: A newly introduced abstract class that defines behavior for logger callbacks interacting with the `logging.MlflowLogger` class so product of callback gets logged appropriately as artifacts/metrics/parameters.
- `PlotPredictionCallback` class: A newly introduced class that is a realization of the `AbstractLoggerCallback` class. Serves as an example implementation of a logger callback. Similar to the `virtual_stain_flow.callbacks.intermediatePlotCallback`, plots predictions of the model on a subset of the dataset, but the additional interface with the `MlflowLogger` class ensures the 
plots produced are logged as mlflow artifacts.

### Refactored
- Internal function renames for clarity.
- Consistent attribute/property usage.
- Updated `__init__.py` directly exposing classes under modules.

---