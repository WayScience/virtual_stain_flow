"""
context.py

Context class for organizing tensors and torch modules relevant to a specific
    computation context (e.g. during a single forward pass), to facilitate
    isolated and modular computations.
"""

from typing import Dict, Iterable, Tuple, Union

import torch

from .names import TARGETS, PREDS, RESERVED_KEYS, RESERVED_MODEL_KEYS

ContextValue = Union[torch.Tensor, torch.nn.Module]


class ReservedKeyError(KeyError): ...
class ReservedKeyTypeError(TypeError): ...


class Context:
    """
    Simple context class for forward pass management.
    Behaves like a dictionary that maps string keys to torch.Tensor or torch.nn.Module values.
    """

    __slots__ = {"_store", }

    def __init__(self, **items: ContextValue):
        """
        Initializes the Context with optional initial tensors.

        :param items: Keyword arguments of context items,
            where keys are the names of the context items and 
            values the corresponding tensor/module.

        """
        self._store: Dict[str, ContextValue] = {}
        self.add(**items)
        
    def add(self, **items: ContextValue) -> "Context":
        """
        Adds new tensors to the context.

        :param tensors: Keyword arguments,
            where keys are the names of the tensors.
        """
        
        for k, v in items.items():
            if k in RESERVED_KEYS and not isinstance(v, torch.Tensor):
                raise ReservedKeyTypeError(
                    f"Reserved key '{k}' must be a torch.Tensor, got {type(v)}"
                )
            elif k in RESERVED_MODEL_KEYS and not isinstance(v, torch.nn.Module):
                raise ReservedKeyTypeError(
                    f"Reserved key '{k}' must be a torch.nn.Module, got {type(v)}"
                )
            
        self._store.update(items)
        return self
    
    def require(self, keys: Iterable[str]) -> None:
        """
        Called by forward groups to ensure all required inputs are present.
            Raises a ValueError if any key is missing.

        :param keys: An iterable of keys that are required to be present.
        """
        missing = [k for k in keys if k not in self._store]
        if missing:
            raise ValueError(
                f"Missing required inputs {missing} for forward group."
            )
        return None

    def as_kwargs(self) -> Dict[str, ContextValue]:
        """
        Returns the context as a dictionary of keyword arguments.
        Intended use: loss_group(train, **ctx.as_kwargs())
        """
        return self._store

    def as_metric_args(self) -> Tuple[ContextValue, ContextValue]:
        """
        Returns the predictions and targets tensors for 
            Image quality assessment metric computation.
        Intended use: metric.update(*ctx.as_metric_args())
        """
        self.require([PREDS, TARGETS])
        preds = self._store[PREDS]
        targs = self._store[TARGETS]
        return (preds, targs)

    def __repr__(self) -> str:
        if not self._store:
            return "Context()"
        lines = []
        for key, v in self._store.items():
            if isinstance(v, torch.Tensor):
                lines.append(f"  {key}: {tuple(v.shape)} {v.dtype} @ {v.device}")
            elif isinstance(v, torch.nn.Module):
                lines.insert(0, f"  {key}: nn.{v.__class__.__name__}")
            else:
                pass # should not happen due to type checks in add()

        return "Context(\n" + "\n".join(lines) + "\n)"
    
    # --- Methods for dict like behavior of context class ---
    
    def __getitem__(self, key: str) -> ContextValue:
        return self._store[key]
    
    def __iter__(self):
        return iter(self._store)
    
    def __len__(self):
        return len(self._store)
    
    def values(self):
        return self._store.values()

    def items(self):
        return self._store.items()

    def keys(self):
        return self._store.keys()
