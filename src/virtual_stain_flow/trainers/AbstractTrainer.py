"""
AbstractTrainer.py
"""

import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Literal, List, Union

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split

from .trainer_protocol import TrainerProtocol
from ..metrics.AbstractMetrics import AbstractMetrics
from ..vsf_logging import MlflowLogger
from ..datasets.data_split import default_random_split


class AbstractTrainer(TrainerProtocol, ABC):
    """
    Abstract trainer class for img2img translation models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Optional[torch.utils.data.Dataset] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        batch_size: int = 16,
        metrics: Dict[str, AbstractMetrics] = None,
        device: Optional[torch.device] = None,
        early_termination_metric: str = None,
        early_termination_mode: Literal['min', 'max'] = "min",
        **kwargs,
    ):
        """
        Initialize the trainer with the model, optimizer, dataset/loaders,

        :param model: The model to be trained. Should be supplied by subclasses
            to facilitate logging and checkpointing.
        :param optimizer: The optimizer to be used for training.
            to facilitate logging and checkpointing.
        :param dataset: (optional) The dataset to be used for training.
            Either dataset or train_loader, val_loader, test_loader
            must be provided.
        :param train_loader: (optional) DataLoader for training data.
        :param val_loader: (optional) DataLoader for validation data.
        :param test_loader: (optional) DataLoader for test data.
        :param batch_size: The batch size for training.
        :param metrics: Dictionary of metrics to be logged.
        :param device: (optional) The device to be used for training.
        :param early_termination_metric: (optional) The metric to update 
            early-termination count on the validation dataset. 
            If None, early termination is disabled and the
            training will run for the specified number of epochs.
        :param early_termination_mode: (optional)
        """

        self._model = model
        self._optimizer = optimizer

        self._batch_size = batch_size
        self._metrics = metrics if metrics else {}

        if isinstance(device, torch.device):
            self._device = device
        else:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._init_data(
            dataset, train_loader, val_loader, test_loader, **kwargs)
        self._init_state(
            early_termination_metric, early_termination_mode, **kwargs)

    def _init_state(
        self, 
        early_termination_metric,
        early_termination_mode,
        **kwargs
    ):
        # Early stopping state
        self._best_model = None
        self._best_loss = float("inf")
        self._early_stop_counter = 0
        self._early_termination_metric = early_termination_metric
        self._early_termination_mode = early_termination_mode
        self._early_termination = True if early_termination_metric else False

        # Epoch state
        self._epoch = 0

        # Loss and metrics state
        self._train_losses = defaultdict(list)
        self._val_losses = defaultdict(list)
        self._train_metrics = defaultdict(list)
        self._val_metrics = defaultdict(list)

        return None

    def _init_data(
        self, 
        dataset: Optional[torch.utils.data.Dataset] = None, 
        train_loader: Optional[DataLoader] = None, 
        val_loader: Optional[DataLoader] = None, 
        test_loader: Optional[DataLoader] = None, 
        **kwargs
    ):
        if train_loader is not None:

            self._train_loader = train_loader
            self._val_loader = val_loader if val_loader else []
            self._test_loader = test_loader if test_loader else []

        elif dataset is not None:
            (
                self._train_loader,
                self._val_loader,
                self._test_loader
            ) = default_random_split(dataset, **kwargs)
            
        else:
            raise ValueError(
                "Either provide dataset and specify datasplit parameters, "
                "or provide at least train_loader."
            )

        return None

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

    def train(
        self, 
        logger: MlflowLogger, 
        epochs: int, 
        patience: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Train the model for the specified number of epochs.
        Make calls to the train epoch and evaluate epoch methods.

        :param logger: The logger to be used for logging.
        :param epochs: The number of epochs to train the model.
        :param patience: The number of epochs with no improvement,
            after which training will be stopped. If None, early stopping is disabled.
        :param verbose: Whether to display the training progress bar
        """

        if not isinstance(logger, MlflowLogger):
            raise TypeError(f"Expected logger to be an instance of "
                            f"MlflowLogger, got {type(logger)}")

        logger.bind_trainer(self)        
        if hasattr(logger, "on_train_start"):
            logger.on_train_start()

        self._epochs = epochs
        self._patience = patience if patience else epochs # no early stopping
        self._epoch_pbar: Optional[tqdm] = tqdm(
            range(epochs), desc="Training", unit="epoch") if verbose else None
        iterable = self._epoch_pbar if self._epoch_pbar else range(epochs)

        for epoch in iterable:

            # Increment the epoch counter
            self.epoch += 1

            # Invoke the on_epoch_start method of the logge
            if hasattr(logger, "on_epoch_start"):
                logger.on_epoch_start()

            # Access all the metrics and reset them
            for _, metric in self.metrics.items():
                metric.reset()

            # Train the model for one epoch
            train_loss = self.train_epoch()
            for loss_name, loss in train_loss.items():
                self._train_losses[loss_name].append(loss)
                logger.log_metric(
                    f"train_{loss_name}", loss, epoch
                )

            # Evaluate the model for one epoch
            val_loss = self.evaluate_epoch()
            for loss_name, loss in val_loss.items():
                self._val_losses[loss_name].append(loss)
                logger.log_metric(
                    f"val_{loss_name}", loss, epoch
                )

            # Access all the metrics and compute the final epoch metric value
            for metric_name, metric in self.metrics.items():
                train_metric, val_metric = metric.compute()
                self._train_metrics[metric_name].append(train_metric)
                self._val_metrics[metric_name].append(val_metric)

                logger.log_metric(
                    f"train_{metric_name}", train_metric, epoch
                )
                logger.log_metric(
                    f"val_{metric_name}", val_metric, epoch
                )

            if hasattr(logger, "on_epoch_end"):
                logger.on_epoch_end()

            # Update early stopping
            if self.update_early_stop_counter():
                print(f"Early termination at epoch {epoch + 1} "
                      f"with best validation metric {self._best_loss}")
                break

        if hasattr(logger, "on_train_end"):
            logger.on_train_end()

    def _collect_early_stop_metric(self) -> float:
        if self._early_termination_metric is None:
            # Do not perform early stopping when no termination metric is specified
            early_term_metric = None
        else:
            # First look for the metric in validation loss
            if self._early_termination_metric in list(
                self._val_losses.keys()):
                early_term_metric = self._val_losses[
                    self._early_termination_metric][-1]
            # Then look for the metric in validation metrics
            elif self._early_termination_metric in list(
                self._val_metrics.keys()):
                early_term_metric = self._val_metrics[
                    self._early_termination_metric][-1]
            else:
                raise ValueError("Invalid early termination metric")
            
        return early_term_metric

    def update_early_stop_counter(self) -> bool:
        """
        Method to update the early stopping criterion

        :return: True if early stopping criterion is met, False otherwise.
        """
        
        early_term_metric = self._collect_early_stop_metric()

        # When early termination is disabled, 
        # the best model is updated with the current model
        if not self._early_termination and early_term_metric is None:
            self.best_model = self.model.state_dict().copy()
            return False
        
        reset_counter = (early_term_metric < self.best_loss) \
            if self._early_termination_mode == "min" \
                else (early_term_metric > self.best_loss)

        if reset_counter:
            self.best_loss = early_term_metric
            self.early_stop_counter = 0
            self.best_model = self.model.state_dict().copy()
        else:
            self.early_stop_counter += 1

        return self.early_stop_counter >= self.patience
    
    def _update_epoch_progress(
        self,
        batch_idx: int,
        num_batches: int,
        phase: Literal['Train', 'Val'] = 'Train'
    ) -> None:
        """
        Helper for richer progress bar updates during epoch.
        """
        if self._epoch_pbar is None:
            return
        self._epoch_pbar.set_postfix_str(
            f"{phase} Batch {batch_idx + 1}/{num_batches}"
        )

    @abstractmethod
    def save_model(
        self,
        save_path: pathlib.Path,
        file_name_prefix: Optional[str] = None,
        file_name_suffix: Optional[str] = None,
        file_ext: str = '.pth',
        best_model: bool = True
    ) -> Optional[List[pathlib.Path]]:
        pass

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
