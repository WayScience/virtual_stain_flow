import os
import pathlib
import tempfile
from typing import Union, Dict, Optional

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
                 mlflow_uri: Union[pathlib.Path, str] = None,
                 mlflow_experiment_name: Optional[str] = None,
                 mlflow_start_run_args: dict = None,
                 mlflow_log_params_args: dict = None,

                 ):
        """
        Initialize the MlflowLogger callback.

        :param name: Name of the callback.
        :type name: str
        :param artifact_name: Name of the artifact file to log, defaults to 'best_model_weights.pth'.
        :type artifact_name: str, optional
        :param mlflow_uri: URI for the MLflow tracking server, defaults to None.
        If a path is specified, the logger class will call set_tracking_uri to that supplied path 
        thereby initiating a new tracking server. 
        If None (default), the logger class will not tamper with mlflow server to enable logging to a global server
        initialized outside of this class. 
        :type mlflow_uri: pathlib.Path or str, optional
        :param mlflow_experiment_name: Name of the MLflow experiment, defaults to None, which will not call the 
        set_experiment method of mlflow and will use whichever experiment name that is globally configured. If a 
        name is provided, the logger class will call set_experiment to that supplied name.
        :type mlflow_experiment_name: str, optional
        :param mlflow_start_run_args: Additional arguments for starting an MLflow run, defaults to None.
        :type mlflow_start_run_args: dict, optional
        :param mlflow_log_params_args: Additional arguments for logging parameters to MLflow, defaults to None.
        :type mlflow_log_params_args: dict, optional
        """
        super().__init__(name)

        if mlflow_uri is not None:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
            except Exception as e:
                raise RuntimeError(f"Error setting MLflow tracking URI: {e}")                
        
        if mlflow_experiment_name is not None:
            try:
                mlflow.set_experiment(mlflow_experiment_name)
            except Exception as e:
                raise RuntimeError(f"Error setting MLflow experiment: {e}")

        self._artifact_name = artifact_name
        self._mlflow_start_run_args = mlflow_start_run_args
        self._mlflow_log_params_args = mlflow_log_params_args

    def on_train_start(self):
        """
        Called at the start of training.

        Calls mlflow start run and logs params if provided
        """

        if self._mlflow_start_run_args is None:
            pass
        elif isinstance(self._mlflow_start_run_args, Dict):
            mlflow.start_run(
                **self._mlflow_start_run_args
            )
        else:
            raise TypeError("mlflow_start_run_args must be None or a dictionary.")
        
        if self._mlflow_log_params_args is None:
            pass
        elif isinstance(self._mlflow_log_params_args, Dict):
            mlflow.log_params(
                self._mlflow_log_params_args
            )
        else:
            raise TypeError("mlflow_log_params_args must be None or a dictionary.")

    def on_epoch_end(self):
        """
        Called at the end of each epoch.

        Iterate over the most recent log items in trainer and call mlflow log metric
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

        Saves trainer best model to a temporary directory and calls mlflow log artifact
        Then ends run
        """
        # Save weights to a temporary directory and log artifacts
        with tempfile.TemporaryDirectory() as tmpdirname:
            weights_path = os.path.join(tmpdirname, self._artifact_name)
            torch.save(self.trainer.best_model, weights_path)
            mlflow.log_artifact(weights_path, artifact_path="models")

        mlflow.end_run()        