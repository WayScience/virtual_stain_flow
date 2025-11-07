"""
loss_utils.py

Utility functions for loss handling.
"""

from typing import Union, Dict

import torch

from ..losses.AbstractLoss import AbstractLoss


def _get_loss_name(
        loss_fn: Union[torch.nn.Module, AbstractLoss]
) -> str:
    """
    Helper method to get the name of the loss function.
    """
    if isinstance(loss_fn, AbstractLoss) and hasattr(loss_fn, "metric_name"):
        return loss_fn.metric_name
    elif isinstance(loss_fn, torch.nn.Module):
        return type(loss_fn).__name__
    else:
        raise TypeError(
            "Expected loss_fn to be either a torch.nn.Module or "
            "an AbstractLoss instance."
            f"Got {type(loss_fn)} instead."
        )
    
def _scalar_from_device(
    value: float, 
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """
    Helper method to convert a scalar value to a tensor in the specified device.
    """
    return torch.tensor(value, device=device, dtype=dtype)

def _scalar_from_ctx(
    value: float,
    ctx: Dict[str, torch.Tensor]
):
    """
    Helper method to convert a scalar value on the same device and 
        of the same dtype as a given context object (dictionary of tensors)
    """
    t = next(iter(ctx.values()))
    if isinstance(t, torch.Tensor):
        pass
    elif isinstance(t, torch.nn.Module):
        t = next(t.parameters())
    else:
        raise TypeError(
            f"Unsupported context value type: {type(t)}"
        )
    
    return _scalar_from_device(value, t.device, t.dtype)
