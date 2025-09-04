from collections import defaultdict
from typing import Optional, List, Union

import torch
from torch.utils.data import DataLoader, random_split

from .AbstractTrainer import AbstractTrainer 

class Trainer(AbstractTrainer):
    """
    Trainer class for generator while backpropagating on single or multiple loss functions. 
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
        :type model: torch.nn.Module
        :param optimizer: The optimizer to be used for training.
        :type optimizer: torch.optim.Optimizer
        :param backprop_loss: The loss function to be used for training or a list of loss functions.
        :type backprop_loss: torch.nn.Module
        """

        super().__init__(**kwargs)

        self._model = model
        self._optimizer = optimizer
        self._backprop_loss = backprop_loss \
            if isinstance(backprop_loss, list) else [backprop_loss]

    """
    Overidden methods from the parent abstract class
    """
    def train_step(self, inputs: torch.tensor, targets: torch.tensor):
        """
        Perform a single training step on batch.

        :param inputs: The input image data batch
        :type inputs: torch.tensor
        :param targets: The target image data batch
        :type targets: torch.tensor
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
    
    def evaluate_step(self, inputs: torch.tensor, targets: torch.tensor):
        """
        Perform a single evaluation step on batch.

        :param inputs: The input image data batch
        :type inputs: torch.tensor
        :param targets: The target image data batch
        :type targets: torch.tensor
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
        self._model.train()
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

        self._model.eval()
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
