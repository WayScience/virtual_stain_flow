# `virtual_stain_flow.logging`
(Prototype) Renovated Logger for Centralized MLflow Logging.

---

## Overview

This module contains a refactored and **isolated** logger class. It's predecessor, `virtual_stain_flow.callback.MlflowLogger` class, is in parallel with other callback classes for producing prediction plotting and saving model weights during the training progress, which makes logging of their outputs as mlflow artifacts difficult. This renovated module is now its independent class that can accept additional `virtual_stain_flow.logging.callback.LoggerCallbacks` to produce artifacts to be logged. 
* Note that the old `virtual_stain_flow.callback` classes are still retained for potential utility to allow for potential interactions with the training process without need for logging (though may be removed in the future).

---

## Module Structure

```bash
virtual_stain_flow/
└── logging/
    ├── MlflowLoggerV2.py        # Central logging controller class
    └──  callbacks/
        ├── LoggerCallback.py    # Abstract base class for logger-aware callbacks
        └── PlotCallback.py      # Example implementation to generate and prediction plots that the logger logs as artifacts
```

---

## Key Concepts

### 🔹 `MlflowLoggerV2`

* The **core logger** that:

  * Binds to a specific trainer instance only during training.
  * Automatically logs metrics and losses from the trainer.
  * Exposes control for MLflow run lifecycle (no longer ends with the training), thus ↵
  * allowing for manual logging by user on parameters, model weights, and artifacts.
  * Accepts a list of `LoggerCallback` objects to extend automated logging behavior.

### 🔹 `AbstractLoggerCallback`

Abstract class for logger-bound callbacks with defined lifecycle hooks at these stages of training:
  * `on_train_start`
  * `on_epoch_start`
  * `on_epoch_end`
  * `on_train_end`
  * batch-wise actions are not currently supported

and return signature:

  * Metrics (e.g., `"val_loss"`)
  * Parameters (e.g., dataset config)
  * Artifacts (e.g., images, `.pth` files, YAML/JSON configs)

### 🔹 `PlotPredictionCallback`

* A concrete example of `LoggerCallback` that:

  * Accepts a dataset during initialization and dynamically retrieves the model during training to generate prediction plots
  * Saves them to disk and returns paths for MLflow artifact logging
  * Configurable via sampling strategy, metrics, and plotting frequency

---
## Functionality
### Current 

* [x] **MLflow run management** (start, end, tag, etc.)
* [x] **Dynamic trainer binding** at training time
* [x] **Artifact logging routing** based on file type (`.png`, `.pth`, etc.)
* [x] **Flexible callback output handling** (`Path`, `dict`, `(metric_name, value)`)
* [x] **Plotting callback** with configurable dataset and metrics
* [x] **Trainer-agnostic weight saving** via `trainer.save_model()`

### Planned / TODOs

* [  ] Implement `log_param(...)` to support one of:
  * [  ] Flattened logging of structured parameters
  * [  ] Optional YAML/JSON serialization as artifacts
* [  ] Add `log_dict_as_yaml()` and `log_dict_as_json()` utility methods
* [  ] Define a log output exchange data class for richer logging (e.g. `LogEntry` dataclass)
    * to define more complex artifact logging paths and tags
* [  ] **`trainers`/`models` Module** Unify model weight saving across model architectures and trainer classes to faciliate weight logging as artifacts
* [  ] **`datasets` Module** Define interaction of dataset classes with the logger to log data processing steps as parameters
  * Perhaps something like `logger.log_dataset(dataset_obj)`
* [  ] Integrate with optuna optimization experiments to log these as artifacts as well. 

---

## Example Usage
```python
# ... import dataset/model/trainer modules
from virtual_stain_flow.logging import MlflowLoggerV2
from virtual_stain_flow.logging.callbacks import PlotPredictionCallback

"""
dataset = ...
model = ...
trainer = Trainer(
    model,
    dataset
)
"""

params = {
    'foo': 1,
    'bar': 2
}

plot_callback = PlotPredictionCallback(
    name='plot_callback1',
    save_path='.',
    dataset=dataset,
    every_n_epochs=5,
    # kwargs passed to plotter
    show_plot=False
    )

logger = MlflowLoggerV2(
    name='logger1',
    tracking_uri='path/to/mlruns',
    experiment_name='foo1',
    run_name='bar1',
    experiment_type='training',
    model_architecture='architecture1',
    target_channel_name='foobar',
    mlflow_log_params_args=params,
    callbacks=[plot_callback]
)

trainer.train(logger=logger)
```

Expected Logging Output:
```
mlruns/
└── 0/                                   
    └── <run_id>/                        
        ├── artifacts/
        │   ├── plots/
        │   │   └── epoch/
        │   │       └── plot/
        │   │           ├── epoch_5.png
        │   │           ├── epoch_10.png
        │   │           └── epoch_15.png
        │   └── weights/
        │       └── best/
        │           └── logger1_model_epoch_20.pth
        ├── meta.yaml
        ├── params/
        │   ├── foo
        │   └── bar
        ├── metrics/
        │   ├── val_loss
        │   ├── train_loss
        │   ├── train_ssim
        │   └── val_ssim
        └── tags/
            ├── analysis_type
            ├── model
            ├── channel
            └── run_name

```

See `examples/xx` for more detail