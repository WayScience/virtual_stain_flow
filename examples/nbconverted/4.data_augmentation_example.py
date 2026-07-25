#!/usr/bin/env python
# coding: utf-8

# # Examples for incorporating monai image augmentation suite for training

# ## Dependencies

# In[1]:


import pathlib

import pandas as pd
from monai.transforms import (
    Compose, 
    EnsureTyped, 
    RandFlipd, 
    RandRotate90d, 
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd
)

from virtual_stain_flow.datasets.base_dataset import BaseImageDataset
from virtual_stain_flow.datasets.crop_dataset import CropImageDataset
from virtual_stain_flow.datasets.monai_aug_adapter_dataset import MonaiAdapter
from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize
from virtual_stain_flow.evaluation.visualization import plot_dataset_grid


# ## Retrieve Demo Data
# The CPJUMP1 A549 dataset from notebook `0.download_example_dataset`

# In[ ]:


DATA_DOWNLOAD_DIR = pathlib.Path("/PATH/TO/WHERE/YOU/WANT/TO/DOWNLOAD/CPJUMP1")
if not DATA_DOWNLOAD_DIR.exists():
    raise FileNotFoundError(f"Data download directory not found: {DATA_DOWNLOAD_DIR}")
a549_data_dir = DATA_DOWNLOAD_DIR / "cpjump1_a549_48h"
if not a549_data_dir.exists():
    raise FileNotFoundError(f"A549 data directory not found: {a549_data_dir}")


# In[3]:


# for demo purpose just use the training split

file_index_file = a549_data_dir / "train" / "file_index.csv"
if not file_index_file.exists():
    raise FileNotFoundError("Train file index not found")

file_index = pd.read_csv(file_index_file)

print(len(file_index))

dataset = BaseImageDataset(
    file_index=file_index,
    check_exists=True,
    pil_image_mode="I;16",
    input_channel_keys=["BF"],
    target_channel_keys=["DNA"],
)
print(f"Dataset length: {len(dataset)}")
print(
    f"Input channels: {dataset.input_channel_keys}, target channels: {dataset._target_channel_keys}"
)    

cropped_dataset = CropImageDataset.from_base_dataset(
    dataset,
    crop_size=128,
    transforms=MaxScaleNormalize(
    normalization_factor='16bit')
)

_ = plot_dataset_grid(
    cropped_dataset,
    indices=[0,33,444,2000],
    wspace=0.025,
    hspace=0.05
)


# ## Transformation example

# In[4]:


monai_transform = Compose([
    EnsureTyped(keys=["input", "target"]),
    RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["input", "target"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["input", "target"], prob=0.5, max_k=3),
    RandAffined(
        keys=["input", "target"],
        prob=0.7,
        rotate_range=(0.0, 0.0, 0.15),
        translate_range=(0, 0), # no translate
        scale_range=(0.0, 0.0), # no scale
        padding_mode="border",
    ),
    RandGaussianSmoothd(
        keys=["input"],
        prob=0.2,
        sigma_x=(0.25, 0.5), # more aggressive smoothing to simulate out-of-focus
        sigma_y=(0.25, 0.5), 
    ),
    RandAdjustContrastd(
        keys=["input"],
        prob=0.2,
        gamma=(0.95, 1.05), # small variation to avoid unrealistic contrast change
        invert_image=False,
        retain_stats=True,
    ),
    RandGaussianNoised(
        keys=["input"],
        prob=0.2,
        mean=0.0, # no bias
        std=1e-4, # subtle salt and pepper
    ),
])

augmented_dataset = MonaiAdapter(cropped_dataset, transform=monai_transform)


# ## Visualize the same augmented dataset multiple times to see effects of augmentation
# Note that augmentation is only applied to the crop and the shown full FOV is always un-augmented

# In[5]:


for i in range(5):
    plot_dataset_grid(
        dataset=augmented_dataset,
        indices=[0], # only first sample to better see difference
        wspace=0.025,
        hspace=0.05
    )


# ## Use `MonaiAdapter` for training as would with any image dataset
# 
# e.g.
# ```python
# ...
# 
# # Make train loader from augmented adataset
# train_loader = DataLoader(
#     augmented_dataset, 
#     batch_size=batch_size, 
#     shuffle=True,
# )
# ...
# 
# # feed to trainer
# trainer = SingleGeneratorTrainer(
#     model=...,
#     optimizer=...,
#     losses=...,
#     loss_weights=...,
#     device='cuda',
#     train_loader=train_loader
# )
# 
# # optionally, if want to use plot prediction callback
# plot_callback = PlotPredictionCallback(
#     name="...",
#     dataset=crop_dataset, # non-augmented dataset recommended for consistency
#     # but augment datasets also work here
#     ...
# )
# ```
