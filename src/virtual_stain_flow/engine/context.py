"""
context.py

Context class for organizing tensors and torch modules relevant to a specific
    computation context (e.g. during a single forward pass), to facilitate
    isolated and modular computations.
"""

from typing import Dict, Iterable, Union, Optional

import torch
from torch import Tensor

from .names import INPUTS, TARGETS, PREDS, RESERVED_KEYS, RESERVED_MODEL_KEYS

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
        
        for key, value in items.items():
            self[key] = value

        return self
    
    def require(self, keys: Iterable[str]) -> None:
        """
        Called by forward groups to ensure all required inputs are present.
            Raises a ValueError if any key is missing.

        :param keys: An iterable of keys that are required to be present.
        """
        missing = [k for k in keys if k not in self]
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

    def as_metric_args(self) -> tuple[Tensor, Tensor]:
        """
        Returns the predictions and targets tensors for 
            Image quality assessment metric computation.
        Intended use: metric.update(*ctx.as_metric_args())

        :return: A tuple (preds, targets) of tensors.
        :raises ValueError: If either preds or targets is missing.
        """
        self.require(keys=[PREDS, TARGETS])
        preds: Tensor = self.preds
        targs: Tensor = self.targets
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
    
    def __setitem__(self, key: str, value: ContextValue) -> None:
        """
        Sets a context item, with checks for reserved keys.

        :param key: The name of the context item.
        :param value: The tensor/module to store.
        """
        # Only allow torch.Tensor or torch.nn.Module values
        if not isinstance(value, (torch.Tensor, torch.nn.Module)):
            raise TypeError(
                f"Context values must be torch.Tensor or torch.nn.Module, got {type(value)}"
            )
        
        # Further type check matching for reserved keys
        if key in RESERVED_KEYS and not isinstance(value, torch.Tensor):
            raise ReservedKeyTypeError(
                f"Reserved key '{key}' must be a torch.Tensor, got {type(value)}"
            )
        elif key in RESERVED_MODEL_KEYS and not isinstance(value, torch.nn.Module):
            raise ReservedKeyTypeError(
                f"Reserved key '{key}' must be a torch.nn.Module, got {type(value)}"
            )
            
        self._store[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __getitem__(self, key: str) -> ContextValue:
        return self._store[key]
    
    def __iter__(self):
        return iter(self._store)
    
    def __len__(self):
        return len(self._store)
    
    def get(self, key: str, default: Optional[ContextValue] = None) -> Optional[ContextValue]:
        return self._store.get(key, default)

    def values(self):
        return self._store.values()

    def items(self):
        return self._store.items()

    def keys(self):
        return self._store.keys()
    
    def pop(self, key: str) -> ContextValue:
        """
        Remove and return the value for key if key is in the context, 
            else raises a KeyError.
        """
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in Context.")
        return self._store.pop(key)
    
    def __or__(self, other: "Context") -> "Context":
        """
        Merge two Context objects using the | operator.
        Returns a new Context with items from both contexts.
        Items from the right operand (other) take precedence in case of key conflicts.
        
        :param other: Another Context object to merge with.
        :return: A new Context object containing items from both contexts.
        """
        if not isinstance(other, Context):
            raise NotImplementedError(
                "__or__ operation only supported between Context objects."
            )
        new_context = Context(**self._store)
        new_context.add(**other._store)
        return new_context
    
    def __ror__(self, other: "Context") -> "Context":
        """
        Reverse merge (right | operator) for Context objects.
        Called when the left operand doesn't support __or__ with Context.
        
        :param other: Another Context object to merge with.
        :return: A new Context object containing items from both contexts.
        """
        if not isinstance(other, Context):
            raise NotImplementedError(
                "__or__ operation only supported between Context objects."
            )
        new_context = Context(**other._store)
        new_context.add(**self._store)
        return new_context
    
    # --- Properties for robust typing for reserved keys ---
    # let fail if key is not present
    @property
    def inputs(self) -> Tensor:
        return self._store[INPUTS] # type: ignore
    
    @property
    def targets(self) -> Tensor:
        return self._store[TARGETS] # type: ignore
    
    @property
    def preds(self) -> Tensor:
        return self._store[PREDS] # type: ignore
