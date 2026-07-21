from typing import Any, Dict, List, Optional

import mlflow
from torch.optim import Optimizer

from ...trainers.trainer_protocol import TrainerProtocol


class AutoOptimizerConfigLogger:
    """
    Auto-log optimizer metadata to MLflow.
    """

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    def discover_optimizers(
        self,
        trainer: Optional[TrainerProtocol],
    ) -> List[Optimizer]:
        if trainer is None:
            return []

        optimizers: List[Optimizer] = []

        explicit_optimizers = getattr(trainer, "optimizers", None)
        if isinstance(explicit_optimizers, list):
            optimizers.extend(explicit_optimizers)

        explicit_optimizer = getattr(trainer, "optimizer", None)
        if explicit_optimizer is not None:
            optimizers.append(explicit_optimizer)

        return optimizers

    def log_optimizer_configs(
        self,
        trainer: Optional[TrainerProtocol],
    ) -> None:
        optimizers = self.discover_optimizers(trainer)

        for idx, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, Optimizer):
                continue

            try:
                opt_config: Optional[Dict[str, Any]] = {
                    "class_path": (
                        f"{optimizer.__class__.__module__}."
                        f"{optimizer.__class__.__name__}"
                    ),
                    "defaults": dict(optimizer.defaults),
                }
            except Exception as e:
                print(f"Could not get optimizer config for logging: {e}")
                opt_config = None

            if opt_config is None:
                continue

            mlflow.set_tag(
                f"optimizer.{idx}.class_path",
                str(opt_config.get("class_path")),
            )

            try:
                self._logger.log_config(
                    tag=f"optimizer_{idx}",
                    config=opt_config,
                    stage=None,
                )
            except Exception as e:
                print(f"Fail to log optimizer config as artifact: {e}")
