"""
MlflowLogger.py
"""

import pathlib
import tempfile
from typing import Union, Dict, Optional, List, Any

import mlflow
from torch import nn

from ..trainers.trainer_protocol import TrainerProtocol
from .callbacks.LoggerCallback import (
    AbstractLoggerCallback,
    log_type
)

path_type = Union[pathlib.Path, str]

class MlflowLogger:
    """
    MLflow Logger for logging training runs, metrics, artifacts, and parameters, intended to be
        used with the TrainerProtocol class.
    
    This class is distinct and independent from the `virtual_stain_flow.callback` classes,
        in that it is no longer an optional callback to be supplied during trainer initialization 
        but a required parameter for the `train` function of the TrainerProtocol class,
        bound to a single train session.
    
    This class can accept a list of `AbstractLoggerCallback` subclasses so callback products are
        centrally logged appropriately as artifacts/parameters/metrics to MLflow. 
    """
    def __init__(
        self,
        name: str,
        experiment_name: str,
        tracking_uri: Optional[path_type] = None,
        run_name: Optional[str] = None,
        experiment_type: Optional[str] = None,
        model_architecture: Optional[str] = None,
        description: Optional[str] = None,
        target_channel_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        mlflow_start_run_args: dict = None,
        mlflow_log_params_args: dict = None,
        callbacks: Optional[List[Any]] = None,
        save_model_at_train_end: bool = True,
        save_model_every_n_epochs: Optional[int] = None,
        save_best_model: bool = True
    ):
        """
        Initialize the MLflowLoggerV2.

        :param name: Name of the logger.
        :param experiment_name: Name of the MLflow experiment.
        :param tracking_uri: URI for the MLflow tracking server.
        :param run_name: Name of the MLflow run, defaults to None.
        :param experiment_type: Type of the experiment, defaults to None.
        :param model_architecture: Model architecture used in the experiment, defaults to None.
        :param description: Description of the experiment, defaults to None.
        :param target_channel_name: Name of the target channel, defaults to None.
        :param tags: Additional tags to be logged with the run, defaults to None.
        :param mlflow_start_run_args: Additional arguments for starting an MLflow run, defaults to None.
        :param mlflow_log_params_args: Additional arguments for logging parameters to MLflow, defaults to None.
        :param callbacks: List of logger callbacks to be used for logging artifacts, metrics, and parameters, defaults to None.
        :param save_model_at_train_end: Whether to save the model at the end of training, defaults to True.
        :param save_model_every_n_epochs: Save the model every n epochs, defaults to None.
        :param save_best_model: Whether to save the best model during training, defaults to True.        
        :raises RuntimeError: If there is an error setting the MLflow tracking URI or experiment.
        :raises TypeError: If any callback is not an instance of AbstractLoggerCallback.
        :return: None
        """
        
        super().__init__()

        self.name = name

        if tracking_uri is not None:
            try:
                mlflow.set_tracking_uri(tracking_uri)
            except Exception as e:
                raise RuntimeError(f"Error setting MLflow tracking URI: {e}")
            
        # keep track of the run so we precisely terminate 
        # this one when the user explicitly calls end_run
        self.__run_id: Optional[str] = None

        # Tracking defaults to 'Default' experiment if none provided
        # Forces creation of non-existing experiment
        self._experiment_name = experiment_name or "Default"
        exp = mlflow.get_experiment_by_name(self._experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(self._experiment_name)
        else:
            exp_id = exp.experiment_id
        self.__experiment_id = exp_id

        # logged at run start
        self.run_name = run_name

        # logged as tags
        self.tags = {
            "experiment_type": experiment_type,
            "model_architecture": model_architecture,
            "target_channel_name": target_channel_name,
            "description": description
        }

        self.trainer: Optional[TrainerProtocol] = None

        if isinstance(tags, dict):
            for key, value in tags.items():

                # do not overwrite existing tags
                if value is not None and self.tags.get(key) is None:
                    self.tags[key] = value

        elif tags is not None:
            raise TypeError("tags must be a dictionary or None.")

        self._mlflow_start_run_args = mlflow_start_run_args or {}
        self._mlflow_log_params_args = mlflow_log_params_args or {}

        self.callbacks = callbacks or []
        for cb in self.callbacks:
            if not isinstance(cb, AbstractLoggerCallback):
                raise TypeError(
                    "All callbacks must be instances of AbstractLoggerCallback"
                )
            cb.bind_parent(self)

        self._run_id = None        

        self._save_model_at_train_end = save_model_at_train_end
        self._save_model_every_n_epochs = save_model_every_n_epochs
        self._save_best_model = save_best_model

        return None
    
    """
    Bind and unbind methods for trainer
    """

    def bind_trainer(self, trainer):
        """
        Bind the trainer to this callback.
        """
        self.trainer = trainer
        return None
    
    def unbind_trainer(self):
        """
        Unbind the trainer from this callback.
        """
        self.trainer = None
        return None
    
    """
    Lifecycle methods to be invoked by the Trainer class
    """
    
    def on_train_start(self):
        """
        Called at the start of training.

        Calls mlflow start run and logs params if provided
        """

        if not isinstance(self._mlflow_start_run_args, Dict):
            raise TypeError("mlflow_start_run_args must be None or a dictionary.")        
        
        _run = mlflow.start_run(
            experiment_id=self.__experiment_id,
            run_name=self.run_name,
            **self._mlflow_start_run_args
        )

        # keep track of the run id
        self.__run_id = _run.info.run_id
        print(f"MLflow run started with ID: {self.__run_id}")
        
        for key, value in self.tags.items():
            if value is not None:
                mlflow.set_tag(key, value)
        
        if self._mlflow_log_params_args is None:
            pass
        elif isinstance(self._mlflow_log_params_args, Dict):
            mlflow.log_params(
                self._mlflow_log_params_args
            )
        else:
            raise TypeError("mlflow_log_params_args must be None or a dictionary.")
        
        self._run_id = mlflow.active_run().info.run_id
        
        for callback in self.callbacks:
            # TODO consider if we want hasattr checks
            if hasattr(callback, 'on_train_start'):
                tag, callback_return = callback.on_train_start()
                self._log_callback_output(
                    tag=tag,
                    callback_return=callback_return,
                    stage='train',
                    step=0
                )
    
    def on_epoch_start(self):
        """
        Called at the start of each epoch.

        This method can be used to log any necessary information before the epoch begins.
        Currently, it does not perform any actions.
        """
        
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_start'):
                tag, callback_return = callback.on_epoch_start()
                self._log_callback_output(
                    tag=tag,
                    callback_return=callback_return,
                    stage='epoch',
                    step=self.trainer.epoch
                )
        
    def on_epoch_end(self):
        """
        Called at the end of each epoch.

        Iterate over the most recent log items in trainer and call mlflow log metric
        """

        if self._save_model_every_n_epochs is not None:
            if self.trainer.epoch % self._save_model_every_n_epochs == 0:
                self._save_model_weights(
                    artifact_path='weights',
                    best_model=False
                )

        if self._save_best_model:
            self._save_model_weights(
                artifact_path='weights',
                best_model=True
            )

        # Call on_epoch_end for all registered callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                tag, callback_return = callback.on_epoch_end()
                self._log_callback_output(
                    tag=tag,
                    callback_return=callback_return,
                    stage='epoch',
                    step=self.trainer.epoch
                )

    def on_train_end(self):
        """
        Called at the end of training.

        Saves trainer best model to a temporary directory and calls mlflow log artifact
        Then ends run
        """
        # Save weights to a temporary directory and log artifacts
        if self._save_model_at_train_end:
            self._save_model_weights(
                artifact_path='weights',
                best_model=False
            )

        if self._save_best_model:
            self._save_model_weights(
                artifact_path='weights',
                best_model=True
            )

        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                tag, callback_return = callback.on_train_end()
                self._log_callback_output(
                    tag=tag,
                    callback_return=callback_return,
                    stage='train',
                    step=self.trainer.epoch
                )

        self.end_run()

    """
    Run management methods
    """

    def end_run(self):
        """
        End the current MLflow run.
        """
        if self._run_id is None:
            raise RuntimeError("No mlflow run has been initiated with this logger.")

        active_run = mlflow.active_run()
        if active_run is not None:
            if active_run.info.run_id == self.__run_id:
                mlflow.end_run()
            else:
                print(f"Active run ({active_run.info.run_id}) does not match logger's run ({self.run_id}). Not ending.")
        else:
            print("No active MLflow run to end.")

    """
    Exposed? logging methods
    """    
    def log_artifact(
            self,
            tag: str,
            file_path: pathlib.Path,
            stage: Optional[str] = None
        ):
        """
        Log an artifact to MLflow.

        :param tag: The tag to associate with the logged artifact.
        :param file_path: The path to the file to log as an artifact.
        :param stage: Optional stage to categorize the artifact, defaults to None.
        :raises TypeError: If file_path is not a pathlib.Path instance.
        """

        if not isinstance(file_path, pathlib.Path):
            raise TypeError("file_path must be a pathlib.Path instance.")

        artifact_path = ''
        artifact_ext = file_path.suffix.lower()
        if artifact_ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
            # log as plot artifact
            artifact_path += 'plots/'
        elif artifact_ext in ['.pth', '.pt']:
            # log as model artifact
            artifact_path += 'weights/'
        else:
            # log as generic artifact
            artifact_path += 'artifacts/'
        
        if stage is not None:
            artifact_path += f"{stage}/"
        artifact_path += f"{tag}"

        mlflow.log_artifact(
            str(file_path), 
            artifact_path=artifact_path
        )

    def log_metric(
            self, 
            key: str, 
            value: Any, 
            step: int
        ):
        """
        Log a metric to MLflow.
        Meant to be invoked by the Trainer class

        :param key: The name of the metric to log.
        :param value: The value of the metric to log.
        :param step: The step or epoch at which the metric is logged.
        """
        
        mlflow.log_metric(
            key=key,
            value=value,
            step=step
        )

    def log_param(
            self,
            tag: str,
            param: Dict[str, Any],
            stage: Optional[str] = None,
        ):
        """
        Log parameter. Not yet implemented.

        :param tag: The tag to associate with the logged parameter.
        :param param: The parameter to log, which should be a dictionary.
        :param stage: Optional stage to categorize the parameter, defaults to None.
        :raises TypeError: If param is not a dictionary.
        """
        
        # TODO think about how to log parameters (see _log_dict_as_param)
        # alternatively log as yaml (see _log_dict_as_yaml)

        raise NotImplementedError(
            "log_data_param method is not implemented. "
            "Please implement this method to log data parameters."
        )
    

    def _log_callback_output(
        self,
        tag: str,
        callback_return: Optional[List[log_type]],
        # TODO consider whether the current 3 categories of logging (artifact, parameter, metric) 
        # has enough granularity
        stage: Optional[str] = None,
        step: Optional[int] = None
    ):
        """
        Log the output of a callback to MLflow based on its type.
        Mostly intended for use as an internal method to sort the incoming callback return 
         and log to MLflow accordingly. 
        
        TODO consider if we want to expose this in place of the `log_artifact`, `log_metric`, and `log_param` methods
        
        :param tag: The tag to associate with the logged output.
        :param callback_return: The output from the callback, which can be:
            - pathlib.Path: when a file is produced, log as an artifact
            - Dict[str, Any]: when a dictionary is produced, log as a parameter
            - Tuple[str, Any]: when a key value tuple is produced, log as a metric
        :param stage: Optional stage to categorize the logged output, defaults to None.
        :param step: Optional step or epoch at which the metric is logged, defaults to None.
        """

        if callback_return is None:
            return None
        
        for item in callback_return:
            if isinstance(item, pathlib.Path):
                self.log_artifact(
                    tag=tag,
                    file_path=item,
                    stage=stage
                )
            elif isinstance(item, dict):
                self.log_param(
                    tag=tag, 
                    param=item, 
                    stage=stage
                )
            elif isinstance(item, tuple) and len(item) == 2:
                if step is None:
                    step = 0
                key, value = item
                self.log_metric(
                    key=key,
                    value=value,
                    step=step
                )
            else:
                # TODO decide if we want to raise an error or just skip
                continue
                # raise TypeError("Unsupported callback return type for logging.")

    def _save_model_weights(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        artifact_path: str = "weights",
        best_model: bool = True
    ):
        with tempfile.TemporaryDirectory() as tmpdirname:
            
            tmpdirpath = pathlib.Path(tmpdirname)
            
            saved_file_paths = self.trainer.save_model(
                save_path=tmpdirpath,
                file_name_prefix=prefix,
                file_name_suffix=suffix,
                file_ext='.pth',
                best_model=best_model
            )

            for saved_file_path in saved_file_paths:
                mlflow.log_artifact(
                    str(saved_file_path), 
                    artifact_path=artifact_path
                )
    
    """
    Access point for callback to model
    """

    def __check_trainer_bound(
            self
        ) -> None:
        """
        Internal method for centralized check of trainer binding.
        Raises RuntimeError if no trainer is bound to the logger.
        """

        if self.trainer is None:
            raise RuntimeError("No trainer bound to logger. Cannot access trainer attributes.")


    def get_epoch(
            self
        ) -> int:

        try:
            self.__check_trainer_bound()
        except RuntimeError as e:
            raise RuntimeError(
                # some context for the error
                "Cannot access epoch. No trainer bound to logger."
            ) from e
        
        return self.trainer.epoch

    def get_model(
            self,
            best_model: bool = False
        ) -> Optional[nn.Module]:
        """
        Get the model associated with this logger.
        This is typically used by callbacks to access the model.
        
        :param best_model: Whether to return the best model (True) or the current model (False).
        :return: The model instance bound to the trainer.
        """
        
        try:
            self.__check_trainer_bound()
        except RuntimeError as e:
            raise RuntimeError(
                # some context for the error
                "Cannot access model. No trainer bound to logger."
            ) from e
        
        if best_model:
            return self.trainer.best_model
        else:
            return self.trainer.model    
    
    # might be useful access point for callbacks so they can 
    # embed these ids in the artifacts
    @property
    def run_id(self):
        return self._run_id
    
    """
    Unimplemented helper that might be useful
    """
    def _log_dict_as_yaml(
        self,
        dict: Dict[str, Any],
    ):
        """
        Log a dictionary as a YAML file in MLflow.
        
        :param dict: The dictionary to log.
        """
        
        # TODO implement this method to convert dict to YAML and log it
        raise NotImplementedError(
            "log_dict_as_yaml method is not implemented. "
            "Please implement this method to log dictionary as YAML."
        )
    
    def _log_dict_as_param(
        self
    ):
        """
        Log a dictionary as parameters in MLflow.
        
        :return: None
        :raises NotImplementedError: If the method is not implemented.
        
        This method is intended to log a dictionary as parameters in MLflow.
        It is currently not implemented and raises a NotImplementedError.
        """

        # 1. DO flatten dict

        # 2. DO mlflow log
        
        raise NotImplementedError(
            "log_dict_as_param method is not implemented. "
            "Please implement this method to log dictionary as parameters."
        )

    """
    Overridden destructor method to ensure MLflow run is ended
    """
    def __del__(self):
        """
        Destructor to ensure MLflow run is ended when the logger is deleted.
        """
        
        self.end_run()
        
        return None
