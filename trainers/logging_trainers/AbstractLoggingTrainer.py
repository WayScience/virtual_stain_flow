from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Callable, Dict, Optional, Union
import pathlib

import torch
from torch.utils.data import DataLoader, random_split

from ..AbstractTrainer import AbstractTrainer
from ...metrics.AbstractMetrics import AbstractMetrics
from ...callbacks.AbstractCallback import AbstractCallback
from ...logging import MlflowLogger

path_type = Union[pathlib.Path, str]

class AbstractLoggingTrainer(AbstractTrainer):
    """
    Abstract class for trainers that support mlflow logging functionality.

    The callback support of the Trainer classes in general is still retained for potential
    functionality that is independent from the mlflow logging system. 
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
        early_termination_mode: str = "min",
    ):
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            callbacks=callbacks,
            metrics=metrics,
            device=device,
            early_termination_metric=early_termination_metric
        )

        if early_termination_mode not in ["min", "max"]:
            raise ValueError("early_termination_mode must be either 'min' or 'max'")
        self._early_termination_mode = early_termination_mode

    def update_early_stop_counter(self, val_loss: Optional[torch.Tensor]):
        """
        Method to update the early stopping criterion

        :param val_loss: The loss value on the validation set
        :type val_loss: torch.Tensor
        """
        
        # When early termination is disabled, the best model is updated with the current model
        if not self._early_termination and val_loss is None:
            self.best_model = self.model.state_dict().copy()
            return

        terminate = (val_loss < self.best_loss) if self._early_termination_mode == "min" else (val_loss > self.best_loss)

        if terminate:
            self.best_loss = val_loss
            self.early_stop_counter = 0
            self.best_model = self.model.state_dict().copy()
        else:
            self.early_stop_counter += 1

    """
    Re-worked train method to support new mlflow logging functionality.
    Note that in this implementation the logger is passed to the train method
    as opposed to the class during initialization.
    """
    def train(
        self,
        # TODO if we absolutely will do logging with mlflow no matter what
        # we might want to just construct the logger automatically for every trainer
        # and have it exposed
        logger: MlflowLogger, 
    ):
        """
        Train the model with the provided logger.
        Is largely similar to the parent class train method, 
        except it interfaces with the logger independently
        of the callback  invocations.

        :param logger: The logger to be used for logging training progress.
        :raises TypeError: If the logger is not an instance of MlflowLoggerV2.
        """

        if not isinstance(logger, MlflowLogger):
            raise TypeError("logger must be an instance of MlflowLoggerV2")
        
        self.model.to(self.device)

        # 1A) Bind the logger to the trainer, and
        # Invoke the on_train_start method of the logger
        logger.bind_trainer(self)        
        if hasattr(logger, "on_train_start"):
            logger.on_train_start()

        # 1B) Invoke the on_train_start method of callbacks
        for cb in self.callbacks:            
            if hasattr(cb, "on_train_start"):
                cb.on_train_start()

        for epoch in range(self.epochs):

            self.epoch += 1

            # 2A) Invoke the on_epoch_start method of the logger
            if hasattr(logger, "on_epoch_start"):
                logger.on_epoch_start()

            # 2B) Invoke the on_epoch_start method of callbacks
            for cb in self.callbacks:
                if hasattr(cb, "on_epoch_start"):
                    cb.on_epoch_start()
                
            # 2C) Reset all metrics if applicable
            for _, metric in self.metrics.items():
                if hasattr(metric, "reset"):
                    metric.reset()

            # 3A) Train 
            train_loss = self.train_epoch()

            # 3B) Log the training losses
            for loss_name, loss_value in train_loss.items():
                self._train_losses[loss_name].append(loss_value)
                logger.log_metric(
                    f"train_{loss_name}", loss_value, epoch
                )

            # 4A) Evaluate
            val_loss = self.evaluate_epoch()

            # 4B) Log the validation losses
            for loss_name, loss_value in val_loss.items():
                self._val_losses[loss_name].append(loss_value)
                logger.log_metric(
                    f"val_{loss_name}", loss_value, epoch
                )

            # 5) Compute additional metrics and log
            for metric_name, metric_fn in self.metrics.items():
                train_metric_value, val_metric_value = metric.compute()
                self._train_metrics[metric_name].append(train_metric_value)
                self._val_metrics[metric_name].append(val_metric_value)

                logger.log_metric(
                    f"train_{metric_name}", train_metric_value, epoch
                )
                logger.log_metric(
                    f"val_{metric_name}", val_metric_value, epoch
                )

            # 6A) Invoke the on_epoch_end method of the logger
            if hasattr(logger, "on_epoch_end"):
                logger.on_epoch_end()

            # 6B) Invoke the on_epoch_end method of callbacks
            for cb in self.callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end()

            # 7) Update early stopping criteria
            if self._early_termination_metric is None:
                # Do not perform any early stopping
                early_term_metric_value = None
            else:
                # First look for the metric in validation loss
                if self._early_termination_metric in list(val_loss.keys()):
                    early_term_metric_value = val_loss[self._early_termination_metric]
                # Then look for the metric in validation metrics
                elif self._early_termination_metric in list(self._val_metrics.keys()):
                    early_term_metric_value = self._val_metrics[self._early_termination_metric][-1]
                else:
                    raise ValueError("Invalid early termination metric")

            self.update_early_stop_counter(early_term_metric_value)

            # 8) Check if early stopping is needed
            if self._early_termination and self.early_stop_counter >= self.patience:
                print(f"Early termination at epoch {epoch + 1} with best validation metric {self.best_loss}")
                break

        # 9A) Invoke the on_train_end method of the logger
        if hasattr(logger, "on_train_end"):
            logger.on_train_end()

    @abstractmethod
    def save_model(
        self,
        save_path: path_type,
        file_name_prefix: Optional[str] = None,
        file_name_suffix: Optional[str] = None,
        file_ext: str = '.pth',
        best_model: bool = True
    ) -> Optional[List[pathlib.Path]]:

        raise NotImplementedError(
            "save_model method must be implemented in the subclass"
        )