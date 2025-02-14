import os
import pathlib
import tempfile

import mlflow
import torch 

from .AbstractCallback import AbstractCallback

class MlflowLogger(AbstractCallback):
    """
    Callback to log metrics to MLflow.
    """

    def __init__(self, 
                 name: str,
                 artifact_name: str = 'best_model_weights.pth',
                 mlflow_uri: pathlib.Path | str = 'mlruns',
                 mlflow_experiment_name: str = 'default_experiment',
                 mlflow_start_run_args: dict = {},
                 mlflow_log_params_args: dict = {},

                 ):
        """
        :param name: Name of the callback.
        """
        super().__init__(name)

        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(mlflow_experiment_name)
        except Exception as e:
            print(f"Error setting MLflow tracking URI: {e}")

        self._artifact_name = artifact_name
        self._mlflow_start_run_args = mlflow_start_run_args
        self._mlflow_log_params_args = mlflow_log_params_args

    def on_train_start(self, trainer):
        """
        Called at the start of training.
        """
        mlflow.start_run(
            **self._mlflow_start_run_args
        )
        mlflow.log_params(
            self._mlflow_log_params_args
        )

    def on_epoch_end(self, trainer):
        """
        Called at the end of each epoch.
        """
        for key, values in trainer.log.items():
            if values is not None and len(values) > 0: 
                value = values[-1]
            else:
                value = None
            mlflow.log_metric(key, value, step=trainer.epoch)

    def on_train_end(self, trainer):
        """
        Called at the end of training.
        """
        # Save weights to a temporary directory and log artifacts
        with tempfile.TemporaryDirectory() as tmpdirname:
            weights_path = os.path.join(tmpdirname, self._artifact_name)
            torch.save(trainer.best_model, weights_path)
            mlflow.log_artifact(weights_path, artifact_path="models")

        mlflow.end_run()        