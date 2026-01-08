"""
names.py

Defined names for context
"""

from typing import Final, FrozenSet

INPUTS: Final[str]  = "inputs" # always mean the input image tensor
TARGETS: Final[str] = "targets" # always mean the ground truth image tensor
PREDS: Final[str]   = "preds" # always mean the predicted image tensor predicting from inputs the targets

GENERATOR_MODEL: Final[str] = "generator"
DISCRIMINATOR_MODEL: Final[str] = "discriminator"

RESERVED_KEYS: FrozenSet[str] = frozenset({INPUTS, TARGETS, PREDS})
RESERVED_MODEL_KEYS: FrozenSet[str] = frozenset({GENERATOR_MODEL, DISCRIMINATOR_MODEL})
