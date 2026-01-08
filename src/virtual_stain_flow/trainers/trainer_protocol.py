"""
trainer_protocol.py

Protocol for defining behavior and needed attributes of a trainer class.
"""

from typing import Protocol, Dict, runtime_checkable, Any

import torch


@runtime_checkable
class TrainerProtocol(Protocol):
    """
    Protocol for defining the minimal behavior and attribute of a trainer class.
    This protocol is useful for type hinting and checking for trainer object
        in the `vsf_logging` subpackage to avoid circular imports.
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

    def train(self, *args: Any, **kwargs: Any) -> None: ...

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
