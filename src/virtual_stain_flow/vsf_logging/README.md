# `virtual_stain_flow.vsf_logging`
(Prototype) Renovated Logger for Centralized MLflow Logging.

---

## Overview

This subpackage contains a refactored and **isolated** logger that is its 
independent class with the capacity to be registered withl 
`virtual_stain_flow.vsf_logging.callback.LoggerCallbacks` that produces loggable
artifacts.

---

## Subpackage Structure

```bash
virtual_stain_flow/
└── logging/
    ├── MlflowLogger.py        # Central logging controller class
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

## Example Usage
```python
# ... import dataset/model/trainer subpackage
from virtual_stain_flow.vsf_logging import MlflowLogger
from virtual_stain_flow.vsf_logging.callbacks import PlotPredictionCallback

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