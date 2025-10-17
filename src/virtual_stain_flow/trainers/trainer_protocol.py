"""
trainer_protocol.py

Protocol for defining behavior and needed attributes of a trainer class.
"""

from typing import Protocol, Dict, runtime_checkable

import torch


@runtime_checkable
class TrainerProtocol(Protocol):
    """
    Protocol for defining a trainer class.
    """

    _batch_size: int
    _epochs: int
    _patience: int
    _device: torch.device

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]: ...

    def evaluate_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]: ...

    def train_epoch(self) -> Dict[str, float]: ...

    def evaluate_epoch(self) -> Dict[str, float]: ...

    def train(self, num_epochs: int) -> None: ...

    @property
    def epoch(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def metrics(self) -> Dict[str, torch.nn.Module]: ...

    @property
    def model(self) -> torch.nn.Module: ...

    @property
    def best_model(self) -> torch.nn.Module: ...
