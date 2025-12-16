# Virtual Stain Flow Examples

This directory contains example notebooks and scripts demonstrating how to use the `virtual_stain_flow` library for training image-to-image translation models.

## Quick Start

### 1. Download Data

Download the JUMP pilot dataset from AWS S3 (public access, no credentials required):
```bash
python 0.download_data.py --outdir /YOUR/DATA/PATH/
```

This downloads the full 50GB of brightfield and fluorescence microscopy images for the default batch and plate hardcoded in the script.

### 2. Run Examples

Two example notebooks demonstrate core workflows:

- **`1.modular_unet_example.ipynb`** - Building and configuring UNet models.
Does not require dataset downloads.
- **`2.training_with_logging_example.ipynb`** - Training with MLflow logging and callbacks. 
Requires dataset and setting up of a mlflow tracking server.

## Requirements
See the project's `pyproject.toml`.
Note that for data access, AWS cli is additionally required.

## Data

Examples use the **JUMP Pilot** public dataset (CPJUMP1):
- **Source**: AWS S3 bucket (public access)
- **Content**: Multi-channel microscopy images (brightfield, Hoechst, GFP, etc.)
- **Reference**: [JUMP Pilot Project](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1)
