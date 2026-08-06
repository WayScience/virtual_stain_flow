from pathlib import Path
from typing import Optional, List

import torch

from ..trainer_protocol import TrainerProtocol


def save_optimizer_state(
    trainer: 'TrainerProtocol',
    save_path: Path,
    file_name_prefix: Optional[str] = None,
    file_name_suffix: Optional[str] = None,
    file_ext: str = '.pth',
) -> List[Path]:

    if file_name_prefix is None:
        file_name_prefix = 'optimizer'

    if file_name_suffix is None:
        file_name_suffix = f"{trainer.epoch}"

    optimizer = trainer.optimizer

    if optimizer is None:
        return []

    save_file = save_path / f"{file_name_prefix}_{file_name_suffix}{file_ext}"

    torch.save(
        optimizer.state_dict(), 
        save_file
    )

    if save_file.exists():
        return [save_file]

    return []
 
