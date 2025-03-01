from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, random_split

from ..metrics.AbstractMetrics import AbstractMetrics
from ..callbacks.AbstractCallback import AbstractCallback


class AbstractTrainer(ABC):
    """
    Abstract trainer class for img2img translation models.
    Provides shared dataset handling and modular callbacks for logging and evaluation.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 16,
        epochs: int = 10,
        patience: int = 5,
        callbacks: List[AbstractCallback] = None,
        metrics: Dict[str, AbstractMetrics] = None,
        device: Optional[torch.device] = None,
        early_termination_metric: str = None,
        **kwargs,
    ):
        """
        :param dataset: The dataset to be used for training.
        :type dataset: torch.utils.data.Dataset
        :param batch_size: The batch size for training.
        :type batch_size: int
        :param epochs: The number of epochs for training.
        :type epochs: int
        :param patience: The number of epochs with no improvement after which training will be stopped.
        :type patience: int
        :param callbacks: List of callback functions to be executed
        at the end of each epoch.
        :type callbacks: list of callable
        :param metrics: Dictionary of metrics to be logged.
        :type metrics: dict
        :param device: (optional) The device to be used for training.
        :type device: torch.device
        :param early_termination_metric: (optional) The metric to be tracked and used to update early 
            termination count on the validation dataset. If not configured, will be using the value 
            computed by the first validation loss function
        :type early_termination_metric: str
        """

        self._batch_size = batch_size
        self._epochs = epochs
        self._patience = patience
        self.initialize_callbacks(callbacks)
        self._metrics = metrics if metrics else {}

        if isinstance(device, torch.device):
            self._device = device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._best_model = None
        self._best_loss = float("inf")
        self._early_termination = None # switch for early termination
        self._early_stop_counter = 0
        self._early_termination_metric = early_termination_metric

        # Customize data splits
        self._train_ratio = kwargs.get("train", 0.7)
        self._val_ratio = kwargs.get("val", 0.15)
        self._test_ratio = kwargs.get("test", 1.0 - self._train_ratio - self._val_ratio)

        if not (0 < self._train_ratio + self._val_ratio + self._test_ratio <= 1.0):
            raise ValueError("Data split ratios must sum to 1.0 or less.")

        train_size = int(self._train_ratio * len(dataset))
        val_size = int(self._val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Create DataLoaders
        self._train_loader = DataLoader(
            self._train_dataset, batch_size=self._batch_size, shuffle=True
        )
        self._val_loader = DataLoader(
            self._val_dataset, batch_size=self._batch_size, shuffle=False
        )

        # Epoch counter
        self._epoch = 0

        # Loss and metrics storage
        self._train_losses = defaultdict(list)
        self._val_losses = defaultdict(list)
        self._train_metrics = defaultdict(list)
        self._val_metrics = defaultdict(list)

    @abstractmethod
    def train_step(self, inputs: torch.tensor, targets: torch.tensor)->Dict[str, torch.Tensor]:
        """
        Abstract method for training the model on one batch
        Must be implemented by subclasses.
        This should be where the losses and metrics are calculated.
        Should return a dictionary with loss name as key and torch tensor loss as value.

        :param inputs: The input data.
        :type inputs: torch.Tensor
        :param targets: The target data.
        :type targets: torch.Tensor
        :return: A dictionary containing the loss values for the batch.
        :rtype: dict[str, torch.Tensor]
        """
        pass

    @abstractmethod
    def evaluate_step(self, inputs: torch.tensor, targets: torch.tensor)->Dict[str, torch.Tensor]:
        """
        Abstract method for evaluating the model on one batch
        Must be implemented by subclasses. 
        This should be where the losses and metrics are calculated.
        Should return a dictionary with loss name as key and torch tensor loss as value.

        :param inputs: The input data.
        :type inputs: torch.Tensor
        :param targets: The target data.
        :type targets: torch.Tensor
        :return: A dictionary containing the loss values for the batch.
        :rtype: dict[str, torch.Tensor]
        """
        pass
    
    @abstractmethod
    def train_epoch(self)->dict[str, torch.Tensor]:
        """
        Can be overridden by subclasses to implement custom training logic.
        Make calls to the train_step method for each batch 
        in the training DataLoader.

        Return a dictionary with loss name as key and 
        torch tensor loss as value. Multiple losses can be returned.

        :return: A dictionary containing the loss values for the epoch.
        :rtype: dict[str, torch.Tensor]
        """

        pass        

    @abstractmethod
    def evaluate_epoch(self)->dict[str, torch.Tensor]:
        """
        Can be overridden by subclasses to implement custom evaluation logic.
        Should make calls to the evaluate_step method for each batch 
        in the validation DataLoader.

        Should return a dictionary with loss name as key and
        torch tensor loss as value. Multiple losses can be returned.

        :return: A dictionary containing the loss values for the epoch.
        :rtype: dict[str, torch.Tensor]
        """
        
        pass

    def train(self):
        """
        Train the model for the specified number of epochs.
        Make calls to the train epoch and evaluate epoch methods.
        """

        self.model.to(self.device)

        # callbacks 
        for callback in self.callbacks:
            callback.on_train_start()

        for epoch in range(self.epochs):

            # Increment the epoch counter
            self.epoch += 1

            # callbacks
            for callback in self.callbacks:
                callback.on_epoch_start()

            # Access all the metrics and reset them
            for _, metric in self.metrics.items():
                metric.reset()

            # Train the model for one epoch
            train_loss = self.train_epoch()
            for loss_name, loss in train_loss.items():
                self._train_losses[loss_name].append(loss)

            # Evaluate the model for one epoch
            val_loss = self.evaluate_epoch()
            for loss_name, loss in val_loss.items():
                self._val_losses[loss_name].append(loss)

            # Access all the metrics and compute the final epoch metric value
            for metric_name, metric in self.metrics.items():
                train_metric, val_metric = metric.compute()
                self._train_metrics[metric_name].append(train_metric.item())
                self._val_metrics[metric_name].append(val_metric.item())

            # Invoke callback on epoch_end
            for callback in self.callbacks:
                callback.on_epoch_end()

            # Update early stopping
            if self._early_termination_metric is None:
                # Do not perform early stopping when no termination metric is specified
                self._early_termination = False
            else:
                self._early_termination = True
                # First look for the metric in validation loss
                if self._early_termination_metric in list(val_loss.keys()):
                    early_term_metric = val_loss[self._early_termination_metric]
                # Then look for the metric in validation metrics
                elif self._early_termination_metric in list(self._val_metrics.keys()):
                    early_term_metric = self._val_metrics[self._early_termination_metric][-1]
                else:
                    raise ValueError("Invalid early termination metric")                                                        
                
            self.update_early_stop(early_term_metric)

            # Check if early stopping is needed
            if self._early_termination and self.early_stop_counter >= self.patience:
                print(f"Early termination at epoch {epoch + 1} with best validation metric {self._best_loss}")
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def update_early_stop(self, val_loss: torch.Tensor):
        """
        Method to update the early stopping criterion

        :param val_loss: The loss value on the validation set
        :type val_loss: torch.Tensor
        """
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.early_stop_counter = 0
            self.best_model = self.model.state_dict().copy()
        else:
            self.early_stop_counter += 1

    def initialize_callbacks(self, callbacks):
        """
        Helper to iterate over all callbacks and set trainer property

        :param callbacks: List of callback objects that can be invoked 
            at epcoh start, epoch end, train start and train end
        :type callbacks: Callback class or subclass or list of Callback class  
        """

        if callbacks is None:
            self._callbacks = []
            return

        if not isinstance(callbacks, List):
            callbacks = [callbacks]
        for callback in callbacks:
            if not isinstance(callback, AbstractCallback):
                raise TypeError("Invalid callback object type")
            callback._set_trainer(self)
        
        self._callbacks = callbacks

    """
    Log property
    """
    @property
    def log(self):
        """
        Returns the training and validation losses and metrics.
        """
        log ={
            **{'epoch': list(range(1, self.epoch + 1))},
            **self._train_losses,
            **{f'val_{key}': val for key, val in self._val_losses.items()},
            **self._train_metrics,
            **{f'val_{key}': val for key, val in self._val_metrics.items()}
        }

        return log
    
    """
    Properties for accessing various attributes of the trainer.
    """
    @property
    def train_ratio(self):
        return self._train_ratio

    @property
    def val_ratio(self):
        return self._val_ratio

    @property
    def test_ratio(self):
        return self._test_ratio
    
    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def device(self):
        return self._device
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def epochs(self):
        return self._epochs
    
    @property
    def patience(self):
        return self._patience
    
    @property
    def callbacks(self):
        return self._callbacks
    
    @property
    def best_model(self):
        return self._best_model
    
    @property
    def best_loss(self):
        return self._best_loss
    
    @property
    def early_stop_counter(self):
        return self._early_stop_counter
    
    @property
    def metrics(self):
        return self._metrics
    
    @property
    def epoch(self):
        return self._epoch
    
    @property
    def train_losses(self):
        return self._train_losses
    
    @property
    def val_losses(self):
        return self._val_losses
    
    @property
    def train_metrics(self):
        return self._train_metrics
    
    @property
    def val_metrics(self):
        return self._val_metrics
    
    """
    Setters for best model and best loss and early stop counter
    Meant to be used by the subclasses to update the best model and loss
    """

    @best_model.setter
    def best_model(self, value: torch.nn.Module):
        self._best_model = value
    
    @best_loss.setter
    def best_loss(self, value):
        self._best_loss = value

    @early_stop_counter.setter
    def early_stop_counter(self, value: int):
        self._early_stop_counter = value

    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value

    """
    Update loss and metrics
    """

    def update_loss(self, 
                    loss: torch.Tensor, 
                    loss_name: str, 
                    validation: bool = False):
        if validation:
            self._val_losses[loss_name].append(loss)
        else:
            self._train_losses[loss_name].append(loss)

    def update_metrics(self, 
                       metric: torch.tensor, 
                       metric_name: str, 
                       validation: bool = False):
        if validation:
            self._val_metrics[metric_name].append(metric)
        else:
            self._train_metrics[metric_name].append(metric)
    
    """
    Properties for accessing the split datasets.
    """
    @property
    def train_dataset(self, loader=False):
        """
        Returns the training dataset or DataLoader if loader=True

        :param loader: (bool) whether to return a DataLoader or the dataset
        :type loader: bool
        """
        if loader:
            return self._train_loader
        else:
            return self._train_dataset
    
    @property
    def val_dataset(self, loader=False):
        """
        Returns the validation dataset or DataLoader if loader=True

        :param loader: (bool) whether to return a DataLoader or the dataset
        :type loader: bool
        """
        if loader:
            return self._val_loader
        else:
            return self._val_dataset
    
    @property
    def test_dataset(self, loader=False):
        """
        Returns the test dataset or DataLoader if loader=True
        Generates the DataLoader on the fly as the test data loader is not 
        pre-defined during object initialization

        :param loader: (bool) whether to return a DataLoader or the dataset
        :type loader: bool
        """
        if loader:
            return DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False)
        else:
            return self._test_dataset   
