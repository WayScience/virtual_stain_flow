"""
Logging single generator trainer implementation.

This module defines the SingleGeneratorTrainer class, which extends the
AbstractTrainer to provide training and evaluation functionalities for a single
generator model using the engine subpackage for forward passes and loss
computations.
"""

import pathlib
from typing import Dict, List, Union, Optional
from collections import defaultdict

import torch

from .AbstractTrainer import AbstractTrainer
from ..engine.loss_group import LossGroup, LossItem
from ..engine.forward_groups import GeneratorForwardGroup


class SingleGeneratorTrainer(AbstractTrainer):
    """
    Trainer class for single generator model with logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        losses: Union[torch.nn.Module, List[torch.nn.Module]],
        device: torch.device,
        loss_weights: Optional[Union[float, List[float]]] = None,
        **kwargs
    ):
        """
        Initialize the trainer with the loss function and forward group.
        
        :param model: The generator model to be trained.
        :param optimizer: The optimizer to be used for training.
        :param losses: The loss function(s) to be used for training.
        :param device: The device to run the training on.
        :param loss_weights: Optional weights for each loss function.
        :kwargs: Additional arguments for the AbstractTrainer (for data/metric and more)
        """
        
        self._forward_group = GeneratorForwardGroup(
            generator=model,
            optimizer=optimizer,
            device=device
        )

        losses = losses if isinstance(losses, list) else [losses]
        if loss_weights is None:
            loss_weights = [1.0] * len(losses)
        elif isinstance(loss_weights, float):
            loss_weights = [loss_weights] * len(losses)
        elif len(loss_weights) != len(losses):
            raise ValueError(
                "Length of loss_weights must match length of losses."
            )

        self._loss_group = LossGroup(
            items=[
                LossItem(
                    module=loss,
                    weight=weight,
                    args=('preds', 'targets'),
                    device=device
                )
                for loss, weight in zip(losses, loss_weights)
            ]
        )

        super().__init__(
            model=self._forward_group.model,
            optimizer=self._forward_group.optimizer,
            **kwargs
        )

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step on batch.

        :param inputs: The input image data batch
        :param targets: The target image data batch
        :returns: A dictionary of loss values. 
        """

        ctx = self._forward_group(
            train=True,
            inputs=inputs,
            targets=targets
        )

        weighted_total, logs = self._loss_group(train=True, context=ctx)
        weighted_total.backward()
        self._forward_group.step()

        for _, metric in self.metrics.items():
            metric.update(*ctx.as_metric_args(), validation=False)

        return logs

    def evaluate_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single evaluation step on batch.

        :param inputs: The input image data batch
        :param targets: The target image data batch
        :returns: A dictionary of loss values.
        """
        
        ctx = self._forward_group(
            train=False,
            inputs=inputs,
            targets=targets
        )

        _, logs = self._loss_group(train=False, context=ctx)

        for _, metric in self.metrics.items():
            metric.update(*ctx.as_metric_args(), validation=True)

        return logs
    
    def train_epoch(self):
        """
        Perform a full training epoch over the training dataset.
        Primarily responsible for iterating over the data loader and
            invoking train_step, and then collecting the losses.

        :returns: A dictionary of average loss values for the epoch.
        """
        losses = defaultdict(list)

        batch_idx = 0
        for inputs, targets in self._train_loader:

            self._update_epoch_progress(
                batch_idx=batch_idx,
                num_batches=len(self._train_loader),
                phase="Train"
            )

            batch_loss = self.train_step(inputs, targets)
            for key, value in batch_loss.items():
                losses[key].append(value)

            batch_idx += 1            

        return {
            key: sum(values) / len(values) for key, values in losses.items()
        }
    
    def evaluate_epoch(self):
        """
        Perform a full evaluation epoch over the validation dataset.
        Primarily responsible for iterating over the data loader and
            invoking evaluate_step, and then collecting the losses.

        :returns: A dictionary of average loss values for the epoch.
        """
        losses = defaultdict(list)

        batch_idx = 0
        for inputs, targets in self._val_loader:

            self._update_epoch_progress(
                batch_idx=batch_idx,
                num_batches=len(self._val_loader),
                phase="Val"
            )

            batch_loss = self.evaluate_step(inputs, targets)
            for key, value in batch_loss.items():
                losses[key].append(value)

            batch_idx += 1

        return {
            key: sum(values) / len(values) for key, values in losses.items()
        }

    def save_model(
        self,
        save_path: pathlib.Path,
        file_name_prefix: Optional[str] = None,
        file_name_suffix: Optional[str] = None,
        file_ext: str = '.pth',
        best_model: bool = True
    ) -> Optional[List[pathlib.Path]]:
        pass


        if file_name_prefix is None:
            file_name_prefix = 'generator'

        if file_name_suffix is None:
            file_name_suffix = 'weights_' + (
                'best' if best_model else str(self.epoch)
            )

        path = self.model.save_weights(
            filename=f"{file_name_prefix}_{file_name_suffix}{file_ext}",
            dir=save_path
        )

        return [path]
