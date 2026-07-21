from typing import Any, Dict, Optional

import mlflow

from ...trainers.trainer_protocol import TrainerProtocol


class AutoLossGroupConfigLogger:
    """
    Auto-log loss group metadata to MLflow.
    """

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    def discover_loss_groups(
        self,
        trainer: Optional[TrainerProtocol],
    ) -> Dict[str, Any]:
        if trainer is None:
            return {}

        loss_groups: Dict[str, Any] = {}

        explicit_groups = getattr(trainer, "loss_groups", None)
        if isinstance(explicit_groups, dict):
            for group_name, group in explicit_groups.items():
                if hasattr(group, "get_config"):
                    loss_groups[str(group_name)] = group

        fallback_attrs = {
            "main": "_loss_group",
            "generator": "_generator_loss_group",
            "discriminator": "_discriminator_loss_group",
        }
        for group_name, attr in fallback_attrs.items():
            if group_name in loss_groups:
                continue
            group = getattr(trainer, attr, None)
            if group is not None and hasattr(group, "get_config"):
                loss_groups[group_name] = group

        return loss_groups

    def log_loss_group_configs(
        self,
        trainer: Optional[TrainerProtocol],
    ) -> None:
        loss_groups = self.discover_loss_groups(trainer)
        if not loss_groups:
            return None

        for group_name, group in loss_groups.items():
            try:
                group_config = group.get_config()
            except Exception as e:
                print(
                    f"Could not get loss group config for logging "
                    f"({group_name}): {e}"
                )
                continue

            if not isinstance(group_config, list):
                continue

            for idx, item_cfg in enumerate(group_config):
                if not isinstance(item_cfg, dict):
                    continue

                if "key" in item_cfg and item_cfg["key"] is not None:
                    mlflow.set_tag(
                        f"loss.{group_name}.{idx}.name",
                        str(item_cfg["key"]),
                    )

                if "weight" in item_cfg and item_cfg["weight"] is not None:
                    mlflow.set_tag(
                        f"loss.{group_name}.{idx}.weight",
                        str(item_cfg["weight"]),
                    )

            try:
                self._logger.log_config(
                    tag=f"loss_group_{group_name}",
                    config={
                        "group_name": group_name,
                        "items": group_config,
                    },
                    stage=None,
                )
            except Exception as e:
                print(
                    f"Fail to log loss group config as artifact "
                    f"({group_name}): {e}"
                )

        return None
