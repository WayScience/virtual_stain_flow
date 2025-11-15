"""
forward_groups.py

ForwardGroup protocol and implementations for different model architectures.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict

import torch
import torch.optim as optim
import torch.nn as nn

from .names import INPUTS, TARGETS, PREDS, GENERATOR_MODEL
from .context import Context


class AbstractForwardGroup(ABC):
    """
    Abstract base for forward-pass orchestration.

    Concrete subclasses should define ordered key-tuples:
      - input_keys:  names pulled from batch/context and fed to model (*in order*)
      - target_keys: names required for training/metrics (kept as-is)
      - output_keys: names to map model outputs to (*in order*)
    """

    # Subclasses should override these with ordered tuples.
    input_keys:  Tuple[str, ...]
    target_keys: Tuple[str, ...]
    output_keys: Tuple[str, ...]

    def __init__(
        self,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize the forward group. Subclass should override as needed.
        """
        self.models: Dict[str, nn.Module] = {}
        self.optimizer: Dict[str, Optional[optim.Optimizer]] = {}
        self.device = device

    def _move_tensors(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move any tensors in the dict to the configured device."""
        return {
            k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    @staticmethod
    def _normalize_outputs(raw) -> Tuple[torch.Tensor, ...]:
        """
        Normalize model outputs to a tuple of tensors while preserving order.

        Acceptable:
          - Tensor -> (tensor,)
          - tuple/list[Tensor] -> tuple(list)
          - dict[str, Tensor] -> tuple(dict.values())  # preserves insertion order
        """
        if isinstance(raw, torch.Tensor):
            return (raw,)
        if isinstance(raw, (tuple, list)):
            return tuple(raw)
        if isinstance(raw, dict):
            return tuple(raw.values())
        raise TypeError(f"Unsupported model output type: {type(raw)}")
    
    @abstractmethod
    def __call__(self, train: bool, **inputs: torch.Tensor) -> Context:
        """
        Executes the forward pass, managing training/eval modes and optimizer steps.
        Subclasses may override this method if needed.

        :param train: Whether to run in training mode.
        :param inputs: Keyword arguments of input tensors.
        """
        pass


class GeneratorForwardGroup(AbstractForwardGroup):
    """
    Forward group for a simple single-generator workflow.
    Relevant context values are input_keys, target_keys, output_keys for a
        single-generator model, where:
    
    - the forward is called as:
        preds = generator(inputs)
    - and evaluation as:
        metric_value = metric_fn(preds, targets)
    """

    input_keys:  Tuple[str, ...] = (INPUTS,)
    target_keys: Tuple[str, ...] = (TARGETS,)
    output_keys: Tuple[str, ...] = (PREDS,)

    def __init__(
        self,
        generator: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device)

        self.models[GENERATOR_MODEL] = generator
        self.models[GENERATOR_MODEL].to(self.device)
        self.optimizer[GENERATOR_MODEL] = optimizer

    def __call__(self, train: bool, **inputs: torch.Tensor) -> Context:
        """
        Executes the forward pass, managing training/eval modes and optimizer steps.
        Subclasses may override this method if needed.

        :param train: Whether to run in training mode. Meant to be specified
            by the trainer to switch between train/eval modes and determine
            whether gradients should be computed.
        :param inputs: Keyword arguments of input tensors.
        """
        
        fp_model = self.models[GENERATOR_MODEL]
        fp_optimizer = self.optimizer[GENERATOR_MODEL]
        
        # 1) Stage and validate inputs/targets
        ctx = Context(**self._move_tensors(inputs), **{GENERATOR_MODEL: fp_model })
        ctx.require(self.input_keys)
        ctx.require(self.target_keys)

        # 2) Forward, with grad only when training
        fp_model.train(mode=train)
        train and fp_optimizer is not None and fp_optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            model_inputs = [ctx[k] for k in self.input_keys]  # ordered
            raw = fp_model(*model_inputs)
            y_tuple = self._normalize_outputs(raw)

        # 3) Arity check + map outputs to names
        if len(y_tuple) != len(self.output_keys):
            raise ValueError(
                f"Model returned {len(y_tuple)} outputs, "
                f"but output_keys expects {len(self.output_keys)}"
            )
        outputs = {k: v for k, v in zip(self.output_keys, y_tuple)}

        # 5) Return enriched context (preds available for losses/metrics)
        return ctx.add(**outputs)
