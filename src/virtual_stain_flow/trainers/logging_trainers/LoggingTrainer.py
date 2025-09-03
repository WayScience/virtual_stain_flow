import pathlib
from collections import defaultdict
from typing import Optional, List, Union

import torch
from torch.utils.data import DataLoader, random_split

from .AbstractLoggingTrainer import AbstractLoggingTrainer 

path_type = Union[pathlib.Path, str]

class LoggingTrainer(AbstractLoggingTrainer):
    """
    Trainer class for single network CNN image-image models 
    Prototyping for the new mlflow logger
    """
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            backprop_loss: Union[torch.nn.Module, List[torch.nn.Module]],
            **kwargs                    
    ):
        """
        Initialize the trainer with the model, optimizer and loss function.

        :param model: The model to be trained.
        :param optimizer: The optimizer to be used for training.
        :param backprop_loss: The loss function to be used for training or a list of loss functions.
        :param kwargs: Additional arguments passed to the parent AbstractTrainer class.
        """

        super().__init__(**kwargs)

        self._model = model
        self._optimizer = optimizer
        self._backprop_loss = backprop_loss \
            if isinstance(backprop_loss, list) else [backprop_loss]
        
    """
    Overidden methods from the parent abstract class
    """
    def train_step(
            self, 
            inputs: torch.tensor, 
            targets: torch.tensor
        ):
        """
        Perform a single training step on batch.

        :param inputs: The input image data batch
        :param targets: The target image data batch
        """

        # move the data to the device
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # set the model to train
        self.model.train()
        # set the optimizer gradients to zero        
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs)

        # Back propagate the loss
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        for loss in self._backprop_loss:
            losses[type(loss).__name__] = loss(outputs, targets)
            total_loss += losses[type(loss).__name__]

        total_loss.backward()
        self.optimizer.step()

        # Calculate the metrics outputs and update the metrics
        for _, metric in self.metrics.items():
            metric.update(outputs, targets, validation=False)
        
        return {
            key: value.item() for key, value in losses.items()
        }
    
    def evaluate_step(
            self, 
            inputs: torch.tensor, 
            targets: torch.tensor
        ):
        """
        Perform a single evaluation step on batch.

        :param inputs: The input image data batch
        :param targets: The target image data batch
        """

        # move the data to the device
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # set the model to evaluation
        self.model.eval()

        with torch.no_grad():
            # Forward pass
            outputs = self.model(inputs)

            # calculate the loss
            losses = {}
            for loss in self._backprop_loss:
                losses[type(loss).__name__] = loss(outputs, targets)

            # Calculate the metrics outputs and update the metrics
            for _, metric in self.metrics.items():
                metric.update(outputs, targets, validation=True)
        
        return {
            key: value.item() for key, value in losses.items()
        }
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        losses = defaultdict(list)
        # Iterate over the train_loader
        for inputs, targets in self._train_loader:
            batch_loss = self.train_step(inputs, targets)
            for key, value in batch_loss.items():
                losses[key].append(value)

        # reduce loss
        return {
            key: sum(value) / len(value) for key, value in losses.items()
        }
    
    def evaluate_epoch(self):
        """
        Evaluate the model for one epoch.
        """

        self.model.eval()
        losses = defaultdict(list)
        # Iterate over the val_loader
        for inputs, targets in self._val_loader:
            batch_loss = self.evaluate_step(inputs, targets)
            for key, value in batch_loss.items():
                losses[key].append(value)

        # reduce loss
        return {
            key: sum(value) / len(value) for key, value in losses.items()
        }
    
    def save_model(
        self,
        save_path: path_type,
        file_name_prefix: Optional[str] = None,
        file_name_suffix: Optional[str] = None,
        file_ext: str = '.pth',
        best_model: bool = True
    ) -> Optional[List[pathlib.Path]]:        
        """
        Save the model to the specified path with optional prefix and suffix.

        :param save_path: The path where the model should be saved.
        :param file_name_prefix: Optional prefix for the file name, defaults to None.
        :param file_name_suffix: Optional suffix for the file name, defaults to None.
        :param file_ext: The file extension for the saved model, defaults to '.pth'.
        :param best_model: If True, saves the best model, otherwise saves the current model.
        """
        
        if isinstance(save_path, str):
            save_path = pathlib.Path(save_path)
        
        if isinstance(save_path, pathlib.Path):
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
        else:
            raise TypeError("save_path must be a string or pathlib.Path")
        
        model_file_path = save_path / f"{file_name_prefix}_model_{file_name_suffix}{file_ext}"

        torch.save(
            self.best_model if best_model else self.model,
            model_file_path
        )

        if model_file_path.exists():
            return [model_file_path]
        else:
            return []