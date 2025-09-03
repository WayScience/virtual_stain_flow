from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any, TYPE_CHECKING
import pathlib

if TYPE_CHECKING:
    from ..MlflowLogger import MlflowLogger

# TODO is this good enough e.g. do we want richer callback return class
# that has more metadata/tags associated with them
#
# Define the types of log entries
log_type = Union[
    pathlib.Path, # when a file is produced, log as a artifact
    Dict[str, Any], # when a dictionary is produced, log as a parameter
    Tuple[str, Any], # when a key value tuple is produced, log as a metric
    None # in case the callback decides to not produce any entries
    ]
log_artifact_type = Union[
    pathlib.Path,
    None]
log_param_type = Union[
    Dict[str, Any],
    None]
log_metric_type = Union[
    Tuple[str, Any],
    None]

class AbstractLoggerCallback(ABC):
    """
    Abstract class for logger callbacks that defines the 
    behavior/signature/interaction with MlflowLogger.

    The subclasses of this class should do some external computation 
    granted access to the logger instance and the model, produce
    one or multiple log entries that fall under the category of:
    - pathlib.Path: when a file is produced, log as an artifact
    - Dict[str, Any]: when a dictionary is produced, log as a parameter
    - Tuple[str, Any]: when a key value tuple is produced, log as a metric
    The log entries will be logged to the MlflowLoggerV2 as artifact/parameter/metric accordingly
    """

    def __init__(
        self,
        name: str,
    ):
        
        self._name = name
        self._parent: Optional[MlflowLogger] = None        

    def bind_parent(
            self, 
            parent: 'MlflowLogger' = None
        ):
        """
        Bind the parent logger to this callback.
        """
        # TODO need a type check workaround
        # if not isinstance(parent, MlflowLoggerV2):
        #     raise TypeError("parent must be an instance of MlflowLoggerV2")

        self._parent = parent
        return None
    
    def unbind_parent(
            self
        ):
        """
        Unbind the parent logger from this callback.
        """
        self._parent = None
        return None
    
    def get_epoch(
        self
        ) -> Optional[int]:
        """
        Get the current epoch number from the parent logger.

        :return: The current epoch number, or None if the parent logger is not bound.
        """
        if self._parent is None:
            # TODO raise an error?
            return None
            # raise ValueError("Parent logger is not bound to this callback")
        
        return self._parent.get_epoch()
    
    def get_model(
        self,
        best_model: bool = False
        ):
        """
        Get the model associated with the parent logger

        :param best_model: If True, return the best model from the trainer,
            otherwise return the current model.
        """

        if self._parent is None:
            # TODO raise an error?
            return None
            # raise ValueError("Parent logger is not bound to this callback")
        
        return self._parent.get_model(best_model=best_model)
    
    # Abstract methods to be optionally implemented by subclasses
    def on_train_start(
            self
        ) -> Tuple[str, List[log_type]]:
        """
        Called at the start of training.

        :return: A tuple containing the name of the log entry,
          and a list of one or multiple log entries.
        """
        return ('', None)
    
    def on_epoch_start(
            self
        ) -> Tuple[str, List[log_type]]:
        """
        Called at the start of each epoch.

        :return: A tuple containing the name of the log entry,
          and a list of one or multiple log entries.
        """
        return ('', None)
    
    def on_epoch_end(
            self
        ) -> Tuple[str, List[log_type]]:
        """
        Called at the end of each epoch.

        :return: A tuple containing the name of the log entry,
          and a list of one or multiple log entries.
        """
        return ('', None)
    
    def on_train_end(
            self
        ) -> Tuple[str, List[log_type]]:
        """
        Called at the end of training.

        :return: A tuple containing the name of the log entry,
          and a list of one or multiple log entries.
        """
        return ('', None)    