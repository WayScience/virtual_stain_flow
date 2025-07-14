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