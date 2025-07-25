#!/usr/bin/env python
# coding: utf-8

# # A minimal example to demonstrate how prototype new MlflowloggerV2 (tentative name) class works to log artifacts in additional to train loss/metrics. 
# 
# Meant to serve as grounds to discuss design questions, and hopefully a real feature PR will follow this.
# 
# Is dependent on the files produced by 1.illumination_correction/0.create_loaddata_csvs ALSF pilot data repo https://github.com/WayScience/pediatric_cancer_atlas_profiling

# ## Software dependencies

# In[1]:


import pathlib
import sys
import yaml

import pandas as pd
import torch
import torch.optim as optim
import mlflow
from mlflow.tracking import MlflowClient


# In[2]:


sys.path.append(str(pathlib.Path('.').absolute().parent.parent))

## Dataset
from virtual_stain_flow.datasets.PatchDataset import PatchDataset

## FNet training
from virtual_stain_flow.models.fnet import FNet

from virtual_stain_flow.transforms.MinMaxNormalize import MinMaxNormalize

## Metrics
from virtual_stain_flow.metrics.MetricsWrapper import MetricsWrapper
from virtual_stain_flow.metrics.PSNR import PSNR
from virtual_stain_flow.metrics.SSIM import SSIM

from virtual_stain_flow.trainers.TrainerV2 import TrainerV2
from virtual_stain_flow.logging.MlflowLoggerV2 import MlflowLoggerV2
from virtual_stain_flow.logging.callbacks.PlotCallback import PlotPredictionCallback


# ## Data dependencies

# In[3]:


ANALYSIS_REPO_ROOT = pathlib.Path('.').absolute().parent.parent / 'pediatric_cancer_atlas_analysis'
CONFIG_PATH = ANALYSIS_REPO_ROOT / 'config.yml'
config = yaml.safe_load(CONFIG_PATH.read_text())

LOADDATA_FILE_PATH = ANALYSIS_REPO_ROOT / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_train.csv'
assert LOADDATA_FILE_PATH.exists(), f"File not found: {LOADDATA_FILE_PATH}" 

PROFILING_DIR = pathlib.Path(config['paths']['pediatric_cancer_atlas_profiling_path'])
assert PROFILING_DIR.exists(), f"Directory not found: {PROFILING_DIR}"

SC_FEATURES_DIR = pathlib.Path(config['paths']['sc_features_path'])
assert SC_FEATURES_DIR.exists(), f"Directory not found: {SC_FEATURES_DIR}"

INPUT_CHANNEL_NAMES = config['data']['input_channel_keys']
TARGET_CHANNEL_NAMES = config['data']['target_channel_keys']


# ## Example training

# ### Define some parameters for minimal training

# In[4]:


PATCH_SIZE = 256
CONV_DEPTH = 5
LR = 1e-4
BETAS = (0.5, 0.9)
BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 20 # no real early stopping here for demo purpose


# ### Load Dataset

# In[5]:


loaddata_df = pd.read_csv(LOADDATA_FILE_PATH)
sc_features = pd.DataFrame()
for plate in loaddata_df['Metadata_Plate'].unique():
    sc_features_parquet = SC_FEATURES_DIR / f'{plate}_sc_normalized.parquet'
    if not sc_features_parquet.exists():
        print(f'{sc_features_parquet} does not exist, skipping...')
        continue 
    else:
        sc_features = pd.concat([
            sc_features, 
            pd.read_parquet(
                sc_features_parquet,
                columns=['Metadata_Plate', 'Metadata_Well', 'Metadata_Site', 'Metadata_Cells_Location_Center_X', 'Metadata_Cells_Location_Center_Y']
            )
        ])

pds = PatchDataset(
        _loaddata_csv=loaddata_df,
        _sc_feature=sc_features,
        _input_channel_keys=INPUT_CHANNEL_NAMES,
        _target_channel_keys=TARGET_CHANNEL_NAMES,
        _input_transform=None,
        _target_transform=None,
        patch_size=PATCH_SIZE,
        verbose=False,
        patch_generation_method="random_cell",
        n_expected_patches_per_img=50,
        patch_generation_random_seed=42
    )


# In[6]:


pds.set_input_channel_keys(INPUT_CHANNEL_NAMES)that are like the old classes with 
pds.set_target_channel_keys(TARGET_CHANNEL_NAMES[0])


# In[7]:


## Configure the dataset with normalization methods
pds.set_input_transform(MinMaxNormalize(_normalization_factor=(2 ** 16) - 1, _always_apply=True, _p=1.0))
pds.set_target_transform(MinMaxNormalize(_normalization_factor=(2 ** 16) - 1, _always_apply=True, _p=1.0))


# ## Define where the test logging outptus go

# In[8]:


MLFLOW_DIR = pathlib.Path('.').absolute() / 'test_mlflow'
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = pathlib.Path('.').absolute() / 'test_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ## Initialize model, metrics

# In[9]:


metric_fns = {
        "mse_loss": MetricsWrapper(_metric_name='mse', module=torch.nn.MSELoss()),
        "ssim_loss": SSIM(_metric_name="ssim"),
        "psnr_loss": PSNR(_metric_name="psnr"),
    }


# In[10]:


model = FNet(
    depth=CONV_DEPTH,
    output_activation='sigmoid'
)


# In[11]:


params = {
            "lr": LR,
            "beta0": BETAS[0],
            "beta1": BETAS[1],
            "depth": CONV_DEPTH,
            "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "input_norm": 'min_max',
            "target_norm": 'min_max',
            "channel_name": TARGET_CHANNEL_NAMES[0],
        }


# ## Initialize logger and callback

# In[12]:


plot_callback = PlotPredictionCallback(
    name='plot_callback1',
    save_path=PLOT_DIR,
    dataset=pds,
    every_n_epochs=1,
    # kwargs passed to plotter
    show_plot=False
    )


# In[13]:


logger = MlflowLoggerV2(
    name='logger1',
    tracking_uri = str(MLFLOW_DIR / 'mlruns'),
    experiment_name='test_logging',
    run_name='test_run1',
    experiment_type='training',
    model_architecture='FNet',
    target_channel_name=TARGET_CHANNEL_NAMES[0],
    tags={},
    mlflow_log_params_args=params,
    callbacks=[plot_callback]    
)


# ## Finally, initialize trainer

# In[14]:


optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
backprop_loss = torch.nn.L1Loss()
trainer = TrainerV2(
    model=model,
    optimizer=optimizer,
    backprop_loss=backprop_loss,
    dataset=pds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    patience=PATIENCE,
    metrics=metric_fns,
    device='cuda',
    early_termination_metric=backprop_loss.__class__.__name__
)


# ## Bind logger to trainer during training

# In[15]:


trainer.train(logger=logger)


# ## Additionall Logging (concept, not yet Implemented)
# 
# Before explicitly telling mlflow to stop run, user gets to log additional stuff if applicable:

# ### Log dataset parameters

# In[ ]:


# logger.log_dataset(pds)


# ### Log other parameters

# In[16]:


# logger.log_artifact(...)
# logger.log_additional_stuff(...)


# ## Explicitly end run
# After logging everything needed

# In[17]:


mlflow.end_run()
"""
or
"""
# logger.end_run()
"""
or just
"""
del logger # end run is automatically done in the destructor


# ## View Mlflow experiment and run(s)

# In[18]:


mlflow.set_tracking_uri(str(MLFLOW_DIR / 'mlruns'))
client = MlflowClient()
experiments = client.search_experiments()
runs_df = mlflow.search_runs(experiment_ids=[experiments[0].experiment_id])
runs_df.head()


# With the current Implementation and single callback the only artifacts produced will be the best model weights and the prediction plots

# In[19]:


run_id = runs_df['run_id'].iloc[0]
def list_all_artifacts(client, run_id, path=""):
    all_paths = []
    items = client.list_artifacts(run_id, path)
    for item in items:
        if item.is_dir:
            all_paths.extend(list_all_artifacts(client, run_id, item.path))
        else:
            all_paths.append(item.path)
    return all_paths

# Usage
all_artifact_paths = list_all_artifacts(client, run_id)
for path in all_artifact_paths:
    print(path)


# ## Delete everything under the example plot and mlflow

# In[20]:


get_ipython().system('rm -rf ./test_mlflow/')
get_ipython().system('rm -rf ./test_plots/')

