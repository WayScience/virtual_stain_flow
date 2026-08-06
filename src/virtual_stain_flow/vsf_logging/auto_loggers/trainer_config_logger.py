from typing import Any, Optional

from ...trainers.trainer_protocol import TrainerProtocol


class AutoTrainerLogger:
    """
    Auto-log trainer metadata to MLflow.
    """

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    def log_trainer_config(self, trainer: Optional[TrainerProtocol]) -> None:
        
        if trainer is None:
            return

        config = {
            "class_path": f"{trainer.__class__.__module__}.{trainer.__class__.__name__}",
            "device": str(trainer.device), # device used for training
            "batch_size": trainer.batch_size, # batch size used for training
            "train_n": trainer.train_n,
            "val_n": trainer.val_n,
            "test_n": trainer.test_n,
        }

        try:
            self._logger.log_config(
                tag="trainer",
                config=config,
                stage=None,
            )
        except Exception as e:
            print(f"Could not log trainer config: {e}")
