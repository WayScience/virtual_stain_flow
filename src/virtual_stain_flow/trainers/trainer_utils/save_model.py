from pathlib import Path
from typing import Optional, List

from ..trainer_protocol import TrainerProtocol

def save_model(
    trainer: 'TrainerProtocol',
    save_path: Path,
    file_name_prefix: str = 'generator',
    file_name_suffix: Optional[str] = None,
    file_ext: str = '.pth',
    save_best_model: bool = True,
) -> List[Path]:

    if file_name_suffix is None:
        file_name_suffix = 'weights_' + (
            'best' if save_best_model else str(trainer.epoch)
        )

    model = trainer.best_model if save_best_model else trainer.model

    if model is None:
        return []

    path = model.save_weights(
        filename=f"{file_name_prefix}_{file_name_suffix}{file_ext}",
        dir=save_path
    )

    return [path]
