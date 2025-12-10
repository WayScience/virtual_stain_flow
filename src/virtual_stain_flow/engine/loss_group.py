"""
loss_group.py

LossGroup and LossItem classes for managing related loss computations.

This abstraction is motivated by the complexity of managing computation
    of losses from separate forward passes during the same training step,
    which happens in wGAN training. 

Also serves as a way to centralize device management and loss weight control.

The intended use of this module is by trainer classes, which should initialize
    LossItem and LossGroup objects as part of their setup, and then generate
    a dictionary of tensors containing all results of a forward pass and then
    call the LossGroup object directly to arrive at the total loss, and simpify
    the trainer logic substantially.

Note that this module is only responsible for loss computation and 
    weighted accumulation. Requires properly orchestrated forward passes
    and context management upstream by the forward_group module and
    trainer classes. See docstrings in those modules for details.
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Sequence, List

import torch

from .loss_utils import AbstractLoss, _get_loss_name, _scalar_from_ctx
from .context import Context
from .names import PREDS, TARGETS


@dataclass
class LossItem:
    """
    Wrapper data class around a torch.nn.Module loss.
    Additionally specifies:
    - loss weight
    - args for computing the loss (such as those used in GANs, require 
        non-standard inputs beyond (pred, target)). Here we abstract away
        the input definition to the loss module to avoid having complex
        logic in the trainer body. 
    - enabled state (used or not in current backpropagation)
    - compute at val (all loss items are by default also computed
        during validation stage and logged as additional metrics.
        However, some losses might not be suitable for validation time,
        e.g. gradient penalty. This flag allows certain losses to be
        skipped during train/validation.  
    - device (
        certain losses items may involve declaration of buffers/new tensors.
        The LossItem class interfaces between the trainer and the nn.Module
        losses and centralizes device management. 
    )
    """
    module: Union[torch.nn.Module, AbstractLoss]
    args: Union[str, Tuple[str, ...]] = (PREDS, TARGETS)
    key: Optional[str] = None
    weight: float = 1.0
    enabled: bool = True
    compute_at_val: bool = True
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        
        self.key = self.key or _get_loss_name(self.module)        
        self.args = (self.args,) if isinstance(self.args, str) else self.args
        
        try:
            self.module.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to move loss module {self.key} "
                f"to device {self.device}."
            ) from e
    
    def __call__(
        self,
        train: bool,
        context: Optional[Context] = None,
        **inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss value and the weighted loss value.
        Returns zero tensors if the loss is disabled or if compute should be 
        skipped during validation.

        :param train: Whether the model is in training mode.
        :param context: Optional Context object containing tensors.
        :param inputs: Keyword arguments containing all necessary inputs for the
            loss computation.
        :return: A tuple containing the raw loss and the weighted loss.
        """

        if context is not None:
            context.require(self.args)
            inputs = {arg: context[arg] for arg in self.args}
        
        if not self.enabled or (not train and not self.compute_at_val):
            zero = _scalar_from_ctx(0.0, inputs)
            return zero, zero
        
        missing = [kw for kw in self.args if kw not in inputs.keys()]
        if missing:
            raise ValueError(
                f"Missing required arguments {missing} for loss computation."
            )
        
        if hasattr(self.module, 'to'):
            self.module.to(self.device)
        
        raw = self.module(*[inputs[arg] for arg in self.args])

        return raw, raw * _scalar_from_ctx(self.weight, inputs)
    
@dataclass
class LossGroup:
    """
    Container class to manage the computation of multiple loss items.
    In this container class, each item is a LossItem object 
        (see LossItem class for details), and all items organized under the
        group are anticipated to be computed during the same forward pass.
        The __call__ method of the LossGroup class takes the superset of
        inputs required by all LossItems, iterates through the items and 
        distributes the inputs accordingly, and then returns the total loss.
    """
    items: Sequence[LossItem]

    @property
    def item_names(self) -> List[str]:
        return [item.key for item in self.items]
    
    def __call__(
        self,
        train: bool,
        context: Optional[Context] = None,
        **inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total loss and individual loss values.

        :param train: Whether the model is in training mode.
        :param context: Optional Context object containing tensors.
        :input inputs: Keyword arguments containing all necessary inputs for the
            loss computations.
        :return: A tuple containing the total loss and a dictionary of 
            individual loss values.
        """

        total = _scalar_from_ctx(0.0, context if context else inputs)

        logs: Dict[str, float] = {}

        for item in self.items:
            raw, weighted = item(train, context=context, **inputs)
            logs[item.key] = raw.item()
            total += weighted
        
        return total, logs
