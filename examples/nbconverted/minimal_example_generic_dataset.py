#!/usr/bin/env python
# coding: utf-8

# # A minimal example to demonstrate how the trainer for FNet and wGAN GP plus the callbacks works along with patched dataset
# 
# Is will not be dependent on the pe2loaddata generated index file from the ALSF pilot data repo unlike the other example notebook

# In[1]:


import sys
import pathlib

import pandas as pd
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(pathlib.Path('.').absolute().parent.parent))
print(str(pathlib.Path('.').absolute().parent.parent))

## Dataset
from virtual_stain_flow.datasets.GenericImageDataset import GenericImageDataset
from virtual_stain_flow.datasets.CachedDataset import CachedDataset

## FNet training
from virtual_stain_flow.models.fnet import FNet
from virtual_stain_flow.trainers.Trainer import Trainer

## wGaN training
from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.models.discriminator import GlobalDiscriminator
from virtual_stain_flow.trainers.WGANTrainer import WGANTrainer

## wGaN losses
from virtual_stain_flow.losses.GradientPenaltyLoss import GradientPenaltyLoss
from virtual_stain_flow.losses.DiscriminatorLoss import WassersteinLoss
from virtual_stain_flow.losses.GeneratorLoss import GeneratorLoss

from virtual_stain_flow.transforms.MinMaxNormalize import MinMaxNormalize

## Metrics
from virtual_stain_flow.metrics.MetricsWrapper import MetricsWrapper
from virtual_stain_flow.metrics.PSNR import PSNR
from virtual_stain_flow.metrics.SSIM import SSIM

## callback
from virtual_stain_flow.callbacks.MlflowLogger import MlflowLogger
from virtual_stain_flow.callbacks.IntermediatePlot import IntermediatePlot


# ## Specify train data and output paths

# In[ ]:


EXAMPLE_PATCH_DATA_EXPORT_PATH = '/REPLACE/WITH/PATH/TO/DATA'

EXAMPLE_DIR = pathlib.Path('.').absolute() / 'example_train_generic_dataset'
EXAMPLE_DIR.mkdir(exist_ok=True)


# In[3]:


get_ipython().system('rm -rf example_train_generic_dataset/*')


# In[4]:


PLOT_DIR = EXAMPLE_DIR / 'plot'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DIR =EXAMPLE_DIR / 'mlflow'
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)


# ## Configure channels

# In[5]:


channel_names = [
    "OrigBrightfield",
    "OrigDNA",
    "OrigER",
    "OrigMito",
    "OrigRNA",
    "OrigAGP",
]
input_channel_name = "OrigBrightfield"
target_channel_names = [ch for ch in channel_names if ch != input_channel_name]


# ## Prep Patch dataset and Cache

# In[6]:


ds = GenericImageDataset(
    image_dir=EXAMPLE_PATCH_DATA_EXPORT_PATH,
    site_pattern=r"^([^_]+_[^_]+_[^_]+)",
    channel_pattern=r"_([^_]+)\.tiff$",
    verbose=True
)

## Set input and target channels
ds.set_input_channel_keys([input_channel_name])
ds.set_target_channel_keys('OrigDNA')

## Cache for faster training 
cds = CachedDataset(
    ds,
    prefill_cache=True
)


# # FNet trainer

# ## Train model without callback and check logs

# In[7]:


model = FNet(depth=4)
lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

trainer = Trainer(
    model = model,
    optimizer = optimizer,
    backprop_loss = nn.L1Loss(),
    dataset = cds,
    batch_size = 16,
    epochs = 10,
    patience = 5,
    callbacks=None,
    metrics={'psnr': PSNR(_metric_name="psnr"), 'ssim': SSIM(_metric_name="ssim")},
    device = 'cuda',
    early_termination_metric = None
)

trainer.train()


# In[8]:


pd.DataFrame(trainer.log)


# ## Train model with alternative early termination metric

# In[9]:


model = FNet(depth=4)
lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

trainer = Trainer(
    model = model,
    optimizer = optimizer,
    backprop_loss = nn.L1Loss(),
    dataset = cds,
    batch_size = 16,
    epochs = 10,
    patience = 5,
    callbacks=None,
    metrics={'psnr': PSNR(_metric_name="psnr"), 'ssim': SSIM(_metric_name="ssim")},
    device = 'cuda',
    early_termination_metric = 'psnr' # set early termination metric as psnr for the sake of demonstration
)

trainer.train()


# ## Train with mlflow logger callbacks

# In[10]:


mlflow_logger_callback = MlflowLogger(
        name='mlflow_logger',
        mlflow_uri=MLFLOW_DIR / 'mlruns',
        mlflow_experiment_name='Default',
        mlflow_start_run_args={'run_name': 'example_train', 'nested': True},
        mlflow_log_params_args={
            'lr': 3e-4
        },
    )

del trainer

trainer = Trainer(
    model = model,
    optimizer = optimizer,
    backprop_loss = nn.L1Loss(),
    dataset = cds,
    batch_size = 16,
    epochs = 10,
    patience = 5,
    callbacks=[mlflow_logger_callback],
    metrics={'psnr': PSNR(_metric_name="psnr"), 'ssim': SSIM(_metric_name="ssim")},
    device = 'cuda'
)

trainer.train()


# # wGaN GP example with mlflow logger callback and plot callback

# In[11]:


generator = UNet(
    n_channels=1,
    n_classes=1
)

discriminator = GlobalDiscriminator(
    n_in_channels = 2,
    n_in_filters = 64,
    _conv_depth = 4,
    _pool_before_fc = True
)

generator_optimizer = optim.Adam(generator.parameters(), 
                                 lr=0.0002, 
                                 betas=(0., 0.9))
discriminator_optimizer = optim.Adam(discriminator.parameters(), 
                                     lr=0.00002, 
                                     betas=(0., 0.9),
                                     weight_decay=0.001)

gp_loss = GradientPenaltyLoss(
    _metric_name='gp_loss',
    discriminator=discriminator,
    weight=10.0,
)

gen_loss = GeneratorLoss(
    _metric_name='gen_loss'
)

disc_loss = WassersteinLoss(
    _metric_name='disc_loss'
)

mlflow_logger_callback = MlflowLogger(
        name='mlflow_logger',
        mlflow_uri=MLFLOW_DIR / 'mlruns',
        mlflow_experiment_name='Default',
        mlflow_start_run_args={'run_name': 'example_train_wgan', 'nested': True},
        mlflow_log_params_args={
            'gen_lr': 0.0002,
            'disc_lr': 0.00002
        },
    )

plot_callback = IntermediatePlot(
    name='plotter',
    path=PLOT_DIR,
    dataset=ds, # give it the patch dataset as opposed to the cached dataset
    indices=[1,3,5,7,9], # plot 5 selected patches images from the dataset
    plot_metrics=[SSIM(_metric_name='ssim'), PSNR(_metric_name='psnr')],
    figsize=(20, 25),
    show_plot=False,
)

wgan_trainer = WGANTrainer(
    dataset=cds,
    batch_size=16,
    epochs=20,
    patience=20, # setting this to prevent unwanted early termination here
    device='cuda',
    generator=generator,
    discriminator=discriminator,
    gen_optimizer=generator_optimizer,
    disc_optimizer=discriminator_optimizer,
    generator_loss_fn=gen_loss,
    discriminator_loss_fn=disc_loss,
    gradient_penalty_fn=gp_loss,
    discriminator_update_freq=1,
    generator_update_freq=2,
    callbacks=[mlflow_logger_callback, plot_callback],
    metrics={'ssim': SSIM(_metric_name='ssim'), 
             'psnr': PSNR(_metric_name='psnr')}
)

wgan_trainer.train()

del generator
del wgan_trainer


# ## # wGaN GP example with mlflow logger callback and alternative early termination loss

# In[12]:


generator = UNet(
    n_channels=1,
    n_classes=1
)

discriminator = GlobalDiscriminator(
    n_in_channels = 2,
    n_in_filters = 64,
    _conv_depth = 4,
    _pool_before_fc = True
)

generator_optimizer = optim.Adam(generator.parameters(), 
                                 lr=0.0002, 
                                 betas=(0., 0.9))
discriminator_optimizer = optim.Adam(discriminator.parameters(), 
                                     lr=0.00002, 
                                     betas=(0., 0.9),
                                     weight_decay=0.001)

gp_loss = GradientPenaltyLoss(
    _metric_name='gp_loss',
    discriminator=discriminator,
    weight=10.0,
)

gen_loss = GeneratorLoss(
    _metric_name='gen_loss'
)

disc_loss = WassersteinLoss(
    _metric_name='disc_loss'
)

mlflow_logger_callback = MlflowLogger(
        name='mlflow_logger',
        mlflow_uri=MLFLOW_DIR / 'mlruns',
        mlflow_experiment_name='Default',
        mlflow_start_run_args={'run_name': 'example_train_wgan_mae_early_term', 'nested': True},
        mlflow_log_params_args={
            'gen_lr': 0.0002,
            'disc_lr': 0.00002
        },
    )

wgan_trainer = WGANTrainer(
    dataset=cds,
    batch_size=16,
    epochs=20,
    patience=5, # lower patience here
    device='cuda',
    generator=generator,
    discriminator=discriminator,
    gen_optimizer=generator_optimizer,
    disc_optimizer=discriminator_optimizer,
    generator_loss_fn=gen_loss,
    discriminator_loss_fn=disc_loss,
    gradient_penalty_fn=gp_loss,
    discriminator_update_freq=1,
    generator_update_freq=2,
    callbacks=[mlflow_logger_callback],
    metrics={'ssim': SSIM(_metric_name='ssim'), 
             'psnr': PSNR(_metric_name='psnr'),
             'mae': MetricsWrapper(_metric_name='mae', module=nn.L1Loss()) # use a wrapper for torch nn L1Loss
             },
    early_termination_metric = 'mae' # update early temrination loss with the supplied L1Loss/mae metric instead of the default GaN generator loss
)

wgan_trainer.train()

