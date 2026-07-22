from typing import Any, List, Optional

import mlflow

from ...models.model import BaseModel
from ...trainers.trainer_protocol import TrainerProtocol


class AutoModelConfigLogger:
    """
    Auto-log model configuration metadata to MLflow.
    """

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    def _discover_models(self, trainer: Optional[TrainerProtocol]) -> List[Any]:
        if trainer is None:
            return []

        explicit_models = getattr(trainer, "_models", None)
        if isinstance(explicit_models, list):
            return explicit_models

        model = getattr(trainer, "model", None)
        if model is not None:
            return [model]

        return []

    def log_model_configs(self, trainer: Optional[TrainerProtocol]) -> None:
        models = self._discover_models(trainer)

        for idx, model in enumerate(models):
            if not isinstance(model, BaseModel) or not hasattr(model, "to_config"):
                continue

            try:
                config = model.to_config()
            except Exception as e:
                print(f"Could not get model config for logging: {e}")
                config = None

            if not isinstance(config, dict):
                continue

            class_path = config.get("class_path")
            if class_path:
                mlflow.set_tag(
                    f"model.{idx}.class_path",
                    str(class_path),
                )

            try:
                self._logger.log_config(
                    tag=model.__class__.__name__,
                    config=config,
                    stage=None,
                )
            except Exception as e:
                print(f"Fail to log model config as artifact: {e}")
