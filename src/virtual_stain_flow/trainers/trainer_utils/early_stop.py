from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from ...models.model import clone_model


@dataclass
class BestEpochState:
    """Track the best validation observation and its model snapshot."""

    best_mode: Literal["min", "max"] = "min"
    best_metric_value: Optional[float] = None
    best_epoch: int = 0
    best_model: Optional[torch.nn.Module] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.best_mode not in ("min", "max"):
            raise ValueError("best_mode must be either 'min' or 'max'.")

        if self.best_metric_value is None:
            self.best_metric_value = (
                float("inf") if self.best_mode == "min" else float("-inf")
            )

    def update(
        self,
        metric_value: float,
        epoch: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> bool:
        """Record an improved observation and return whether it improved."""
        metric_value = float(metric_value)
        is_better = (
            self.best_mode == "min" and metric_value < self.best_metric_value
        ) or (
            self.best_mode == "max" and metric_value > self.best_metric_value
        )
        if not is_better:
            return False

        self.best_metric_value = metric_value
        self.best_epoch = epoch if epoch is not None else self.best_epoch + 1
        if model is not None:
            self.best_model = clone_model(model)

        return True


class EarlyStopHelper:
    """Track validation improvements and decide when training should stop."""

    def __init__(
        self,
        best_mode: Literal["min", "max"] = "min",
        trainer_val_losses_ref: Optional[dict] = None,
        trainer_val_metrics_ref: Optional[dict] = None,
        best_metric_name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.best_epoch_state = BestEpochState(best_mode=best_mode)
        self.trainer_val_losses_ref = (
            trainer_val_losses_ref if trainer_val_losses_ref is not None else {}
        )
        self.trainer_val_metrics_ref = (
            trainer_val_metrics_ref if trainer_val_metrics_ref is not None else {}
        )
        self.best_metric_name = best_metric_name
        self.enabled = enabled
        self._patience: Optional[int] = None
        self._counter = 0

    def initialize_early_stop(self, patience: int) -> None:
        """Initialize a run's patience counter while preserving the best state."""
        if patience < 1:
            raise ValueError("patience must be at least 1.")

        self._patience = patience
        self._counter = 0

    def update(
        self,
        epoch: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> bool:
        """Update tracking state and return whether training should stop."""
        if not self.enabled:
            return False
        if self._patience is None:
            raise RuntimeError(
                "Early stopping has not been initialized. "
                "Call 'initialize_early_stop' first."
            )

        metric_value = _get_latest_metric_value(
            self.trainer_val_losses_ref,
            self.trainer_val_metrics_ref,
            self.best_metric_name,
        )
        if self.best_epoch_state.update(metric_value, epoch, model):
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self._patience

    @property
    def best_model(self) -> Optional[torch.nn.Module]:
        """Return the best model snapshot, if an observation was recorded."""
        return self.best_epoch_state.best_model

    @property
    def counter(self) -> int:
        """Return the number of consecutive observations without improvement."""
        return self._counter


def _get_latest_metric_value(
    val_losses: dict,
    val_metrics: dict,
    metric_name: Optional[str] = None,
) -> float:
    """Return the latest configured validation loss or metric value."""
    if metric_name is None:
        if not val_losses:
            raise RuntimeError("No validation losses are available for early stopping.")
        metric_name = next(iter(val_losses))

    if metric_name in val_losses:
        values = val_losses[metric_name]
    elif metric_name in val_metrics:
        values = val_metrics[metric_name]
    else:
        raise ValueError(
            f"Supplied early stop metric '{metric_name}' is not found in logs."
        )

    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"Expected the metric '{metric_name}' to be a list or tuple, "
            f"but got {type(values)} instead."
        )
    if not values:
        raise RuntimeError(f"Early stop metric '{metric_name}' has no recorded values.")

    return float(values[-1])
