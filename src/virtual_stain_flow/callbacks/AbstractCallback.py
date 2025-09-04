from abc import ABC

class AbstractCallback(ABC):
    """
    Abstract class for callbacks in the training process.
    Callbacks can be used to plot intermediate metrics, log contents, save checkpoints, etc.
    """
    
    def __init__(self, name: str):
        """
        :param name: Name of the callback.
        """        
        self._name = name
        self._trainer = None
    
    @property
    def name(self):
        """
        Getter for callback name
        """
        return self._name

    @property
    def trainer(self):
        """
        Allows for access of trainer
        """
        return self._trainer

    def _set_trainer(self, trainer):
        """
        Helper function called by trainer class to initialize trainer value field

        :param trainer: trainer object
        :type trainer: AbstractTrainer or subclass
        """
        
        self._trainer = trainer

    def on_train_start(self):
        """
        Called at the start of training.
        """
        pass

    def on_epoch_start(self):
        """
        Called at the start of each epoch.
        """
        pass

    def on_epoch_end(self):
        """
        Called at the end of each epoch.
        """
        pass
    
    def on_train_end(self):
        """
        Called at the end of training.
        """
        pass
