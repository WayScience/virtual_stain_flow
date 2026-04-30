"""
progress.py

Progress tracking for loss weight scheduling.

Provides a centralized abstraction for training progress state that can be used
by loss schedulers operating at different granularities (epoch, step, etc.).
Designed to be minimal and extensible without overcomplicating the current API.
"""

from dataclasses import dataclass


@dataclass
class Progress:
    """
    Tracks training progress for loss weight scheduling.
    
    Provides centralized access to scheduling state including epoch and step,
    with room for future custom scheduling granularities.
    """
    epoch: int = 0
    step: int = 0
    
    def set_epoch(self, epoch: int) -> None:
        """
        Update the current epoch number.
        """
        self.epoch = epoch
    
    def set_step(self, step: int) -> None:
        """
        Update the current step number.
        Intended to be accumulated across epoch for a global step count 
            that can be used for step-based scheduling.
        """
        self.step = step
