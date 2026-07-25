#!/usr/bin/env python
# coding: utf-8

# # Training wGAN (U-Net generator) with logging Example Notebook
# 
# This notebook demonstrates how to train `virtual_stain_flow.models` module,
# demoing with trainer and logging. 

# ## Dependencies

# In[1]:


import re
import pathlib
from typing import List


import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from mlflow.tracking import MlflowClient

from virtual_stain_flow.datasets.base_dataset import BaseImageDataset
from virtual_stain_flow.datasets.crop_dataset import CropImageDataset
from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize

from virtual_stain_flow.trainers.logging_gan_trainer import LoggingWGANTrainer
from virtual_stain_flow.vsf_logging.MlflowLogger import MlflowLogger
from virtual_stain_flow.vsf_logging.callbacks.PlotCallback import PlotPredictionCallback
from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.models.discriminator import PatchBasedDiscriminator
from virtual_stain_flow.losses.wgan_losses import (
    AdversarialLoss,
    GradientPenaltyLoss,
    WassersteinLoss
)
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


# ## Create dataset and crop center 128 by 128

# In[3]:


splits = ["train", "test"]

raw_datasets = {}
crop_datasets = {}
for split in splits:
    file_index_file = a549_data_dir / f"{split}" / "file_index.csv"
    if not file_index_file.exists():
        raise FileNotFoundError(f"{split} file index not found")

    file_index = pd.read_csv(file_index_file)
    print(f"Reading {split} dataset:")
    dataset = BaseImageDataset(
        file_index=file_index,
        check_exists=True,
        pil_image_mode="I;16",
        input_channel_keys=["BF"],
        target_channel_keys=["DNA"],
    )
    print(f"\tRaw dataset length: {len(dataset)}")
    print(
        f"\tInput channels: {dataset.input_channel_keys}, target channels: {dataset._target_channel_keys}"
    )    

    raw_datasets[split] = dataset

    cropped_dataset = CropImageDataset.from_base_dataset(
        dataset,
        crop_size=128,
        transforms=MaxScaleNormalize(
            normalization_factor='16bit'),
    )
    print(f"\tCropped dataset length: {len(cropped_dataset)}")
    crop_datasets[split] = cropped_dataset

    _ = plot_dataset_grid(
    cropped_dataset,
        indices=[0,1,2,3,4],
        wspace=0.025,
        hspace=0.05
    )


# ## Configure and train

# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[5]:


## Hyperparameters

# Arbitrary big number of epochs, mainly for demo purposes
epochs = 150

# Batch size arbitrarily chosen for demo purposes. 
# With a value of 16, the training fits comfortably into a nvidia RTX 3090 and utilizing <10GB of VRAM.
# Tune to your hardware capabilities.
batch_size = 32

# larger generating learning rate relative to discriminator for quick
# demo purposes, they are the same ones used in Cross-Zamirski et al. 2022
learning_rate = 2e-4
discriminator_learning_rate = 2e-4

# Batch with DataLoader
train_loader = DataLoader(crop_datasets['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(crop_datasets['test'], batch_size=batch_size, shuffle=False)

# Model & Optimizer
# Fully convolutional UNet as generator
fully_conv_unet = UNet(
    in_channels=1,
    out_channels=1,
    depth=4,
    encoder_down_block='conv',
    decoder_up_block='convt',
    act_type='sigmoid'
)
unet_optimizer = torch.optim.AdamW(
    fully_conv_unet.parameters(), 
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)

# Discriminator Model
discriminator = PatchBasedDiscriminator(
    in_channels=2, # matches the generator 1 input + 1 output channel stack
)

discriminator_optimizer = torch.optim.AdamW(
    discriminator.parameters(), 
    lr=learning_rate,
    betas=(0., 0.999),
    weight_decay=1e-5
)

# Plotting callback to visualize predictions during training
# At the end of every n epochs, the callback takes the most recent model
# weights and runs inference on the provided images (dataset indexed by sample indices).
# And plots the predictions alongside the inputs and targets to give a visual sense of training progress.
#
# This gets bounded to the logger instance below to automatically register
# plots to the training. 
plot_callback = PlotPredictionCallback(
    name="plot_callback_with_train_data",
    dataset=crop_datasets['train'],
    indices=[0,1,2,3,4], # first 5 samples
    plot_metrics=[torch.nn.L1Loss()],
    every_n_epochs=1, # plot predictions per epoch
    # kwargs passed to plotting backend
    show_plot=False, # don't show plot in notebook
    wspace=0.025, # small spacing between subplots
    hspace=0.05, # small spacing between subplots    
    tag="plot_train_predictions" # tag needed to plot train and val predictions separate
)

plot_callback_val = PlotPredictionCallback(
    name="plot_callback_with_val_data",
    dataset=crop_datasets['test'],
    indices=[0,1,2,3,4], # first 5 samples
    plot_metrics=[torch.nn.L1Loss()],
    every_n_epochs=1, # plot predictions per epoch
    # kwargs passed to plotting backend
    show_plot=False, # don't show plot in notebook
    wspace=0.025, # small spacing between subplots
    hspace=0.05, # small spacing between subplots
    tag="plot_heldout_predictions" # tag needed to plot train and val predictions separate
)

# MLflow Logger
# The logger that communicates with an MLflow tracking server.
# The Mlflow logger by default logs all metrics and losses specified to the
# trainer, plus any files (artifacts in mlflow terminology) generated by the callbacks.
#
# The logger by default saves the model weights at the end of every epoch and
# the best model weights according to validation loss (not applicable here since no val set).
# The only additional callback bound to the logger is plotting callback defined above. 
# Thus the only files being logged are the plots and the model weights.
logger = MlflowLogger(
    name="logger",
    experiment_name="vsf_examples",
    # Change to your MLflow tracking server/local file based tracking URI
    tracking_uri="http://127.0.0.1:5000", 
    run_name="Example training WGAN on unperturbed CPJUMP1 A549 @ 24h",
    description="Training for demo purposes",
    callbacks=[plot_callback, plot_callback_val],
    save_model_at_train_end=True,
    save_model_every_n_epochs=1,
    save_best_model=True
)

# Initialize Trainer and start training
trainer = LoggingWGANTrainer(
    # Generator
    generator=fully_conv_unet,
    generator_optimizer=unet_optimizer, 

    # Generator Losses (non-adversarial, depend only on generation and ground truth)
    generator_losses=[
        torch.nn.L1Loss(), # simple per pixel error
        MultiScaleStructuralSimilarityIndexMeasure( # MS-SSIM for faster convergence
            data_range=None, # use per batch empirical data range 
            kernel_size=11,
            sigma=1.5,
            # 4 instead of default 5 scales because 5 scales require images
            #   to be minimal 160 by 160 but we are using 128 by 128 crops
            betas=(0.0448, 0.2856, 0.3001, 0.2363) 
        )
    ],
    # maximize MS-SSIM while minimizing L1 distance
    # here the weight magnitudes are made larger to attribute sufficient weight
    # on the generation quality so that the wGAN adverserial
    # and wasserstein losses (both unbounded) don't outscale the other losses 
    generator_loss_weights=[1.0, -1.0], 

    # Generator Adverserial Loss
    generator_adverserial_loss=AdversarialLoss(),
    generator_adverserial_loss_weight=1.0,

    # Discriminator wasserstein Loss and Gradient Penalty
    discriminator=discriminator,
    discriminator_optimizer=discriminator_optimizer,
    discriminator_loss=WassersteinLoss(),
    discriminator_loss_weight=1.0,
    discriminator_gradient_penalty_loss=GradientPenaltyLoss(),
    discriminator_gradient_penalty_weight=10.0,

    # Training alternation
    n_discriminator_steps=5, # number of discriminator updates per generator update

    # Other parameters
    device=device,
    train_loader=train_loader, # training data loader
    val_loader=test_loader, # using the test loader as a stand-in for validation
    test_loader=None, # not used here in demo
)

# Start training
trainer.train(logger=logger, epochs=epochs)


# In[6]:


logger.end_run()


# ## Visualize training outcome through MLflow client

# ### Display the last logged prediction plot artifact

# In[7]:


# Create MLflow client
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

# Get the experiment by name
# should match the experiment name specified in the MlflowLogger above
experiment = client.get_experiment_by_name("vsf_examples")

if experiment is None:
    print("Experiment 'vsf_examples' not found")
else:
    # Search for runs with the specific run name
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'Example training WGAN on unperturbed CPJUMP1 A549 @ 24h'"
    )

    if len(runs) == 0:
        print("No runs found with name 'Example training WGAN on unperturbed CPJUMP1 A549 @ 24h'")
    else:
        # Get the most recent run (first in list)
        run = runs[0]
        print(f"Run ID: {run.info.run_id}")
        print(f"Run Name: {run.data.tags.get('mlflow.runName')}")

        # List artifacts (files) produced during training.
        plot_artifacts = client.list_artifacts(run.info.run_id, path='plots/epoch/plot_train_predictions/')

        # Filter for PNG files and sort by path (which includes epoch number)
        png_files = [artifact for artifact in plot_artifacts if artifact.path.endswith('.png')]

        # Get full paths and sort by epoch number
        png_files_sorted = sorted(png_files, key=lambda x: int(x.path.split('_')[-1].split('.')[0]))
        most_recent_png = png_files_sorted[-1]

        print(f"Last epoch plot: {most_recent_png.path}")

        # Download and display the image
        # (visualizing the most recent prediction plots, at the end of training)
        local_path = client.download_artifacts(run.info.run_id, most_recent_png.path)
        img = Image.open(local_path)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predictions - {most_recent_png.path.split('/')[-1]}")
        plt.tight_layout()
        plt.show()


# With the demo hyperparameters that quickly trains the generator with a 
# ~2000 patch dataset and small number of trianing epochs, 
# we shouldn't expect the model to train well.
# 
# For better/stable training results, consider:
# - lowering learning rate to allow more fine grained convergence to local/global optima
# - increase the training epochs
# - increase sample size
# - as GANs are inherently difficult to train, some hyper-parameter optimzation
#   (generator and discriminator lr, loss weigthing, update freqeuncy) is most likely
#   needed. Uncarefully tuned hyper-parameters can easily make wGAN generator
#   converge worse compared to training the generator alone. 

# ### Also visualize metrics from tracking

# In[8]:


metric_keys = list(run.data.metrics.keys()) or []

for metric_name in metric_keys:
    # 2. Get full history for each metric (all steps)
    history = client.get_metric_history(run.info.run_id, metric_name)
    if not history:
        continue

    steps = [m.step for m in history]
    values = [m.value for m in history]

    # 3. Plot each metric vs step
    plt.figure()
    plt.plot(steps, values, marker="o")
    plt.title(f"{metric_name} (run {run.info.run_id})")
    plt.xlabel("step")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()


# We observe decreasing L1 loss and increasing SSIM, indicating quick convergence of the discriminator,
# expected from the higher generator learning rate.
# 
# The adverserial loss (generator) falls for the first couple of epochs and then
# decreases much slower, while the wasserstein loss (discriminator) increases in the start 
# and the fluctutates around 0, indicating that the generator get good quickly at
# faking at the start, and the discriminator catches up slightly later. 
# 
# This is not the best trend to observe from wGAN training but understandable 
# since the hyper-parameters are selected to favor the discriminator for more stable
# training. In reality the discrminator needs to be made stronger to a point
# that the wasserstein loss has same order of magnitude as the adverserial loss
# without destroying the L1 and SSIM convergenance. 
