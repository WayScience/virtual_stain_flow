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
                 mlflow_experiment_name: str = 'Default',
                 mlflow_start_run_args: dict = {},
                 mlflow_log_params_args: dict = {},

                 ):
        """
        Initialize the MlflowLogger callback.

        :param name: Name of the callback.
        :type name: str
        :param artifact_name: Name of the artifact file to log, defaults to 'best_model_weights.pth'.
        :type artifact_name: str, optional
        :param mlflow_uri: URI for the MLflow tracking server, defaults to 'mlruns' under current wd.
        :type mlflow_uri: pathlib.Path or str, optional
        :param mlflow_experiment_name: Name of the MLflow experiment, defaults to 'Default'.
        :type mlflow_experiment_name: str, optional
        :param mlflow_start_run_args: Additional arguments for starting an MLflow run, defaults to {}.
        :type mlflow_start_run_args: dict, optional
        :param mlflow_log_params_args: Additional arguments for logging parameters to MLflow, defaults to {}.
        :type mlflow_log_params_args: dict, optional
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

    def on_train_start(self):
        """
        Called at the start of training.
        """
        mlflow.start_run(
            **self._mlflow_start_run_args
        )
        mlflow.log_params(
            self._mlflow_log_params_args
        )

    def on_epoch_end(self):
        """
        Called at the end of each epoch.
        """
        for key, values in self.trainer.log.items():
            if values is not None and len(values) > 0: 
                value = values[-1]
            else:
                value = None
            mlflow.log_metric(key, value, step=self.trainer.epoch)

    def on_train_end(self):
        """
        Called at the end of training.
        """
        # Save weights to a temporary directory and log artifacts
        with tempfile.TemporaryDirectory() as tmpdirname:
            weights_path = os.path.join(tmpdirname, self._artifact_name)
            torch.save(self.trainer.best_model, weights_path)
            mlflow.log_artifact(weights_path, artifact_path="models")

        mlflow.end_run()        