from typing import Optional, Sequence, Type, Union, Any, List, Tuple

from .blocks import AbstractBlock

BlockHandleSequence = Union[
    Type[AbstractBlock],
    Sequence[Type[AbstractBlock]]
]


def _is_block_handle(
    obj
) -> bool:
    """
    Check if the object is a block handle (a subclass of AbstractBlock).
    
    :param obj: The object to check.
    :return: True if obj is a block handle, False otherwise.
    """
    import inspect
    
    return inspect.isclass(obj) and issubclass(obj, AbstractBlock)


def _check_block_handle_sequence(
    obj
) -> bool:
    
    from collections.abc import Sequence as ABCSequence 

    """
    Check if the object is a sequence of block handles or a single block handle.
    
    :param obj: The object to check.
    :return: True if obj is a sequence of block handles or a single block handle, 
        False otherwise.
    """
    if _is_block_handle(obj):
        return [obj]
    elif isinstance(obj, ABCSequence) and not isinstance(obj, (str, bytes)):
        if all(_is_block_handle(item) for item in obj):
            return obj
        else:
            raise TypeError("Expected a sequence of block handles, "
                            f"got {type(obj).__name__}")
        
def _check_block_kwargs_sequence(
    obj: Optional[Any] = None
) -> bool:
    """
    Check if the object is a sequence of block kwargs or a single block kwargs.
    
    :param obj: The object to check.
    :return: True if obj is a sequence of block kwargs or a single block kwargs, 
        False otherwise.
    """
    from collections.abc import Sequence as ABCSequence 

    if obj is None:
        return [{}]
    elif isinstance(obj, dict):
        return [obj]
    elif isinstance(obj, ABCSequence) and not isinstance(obj, (str, bytes)):
        if all(isinstance(item, dict) for item in obj):
            return obj
        else:
            raise TypeError("Expected a sequence of block kwargs dictionaries, "
                            f"got {type(obj).__name__}")
        
def validate_block_configurations(
    in_block_handles: Any,
    comp_block_handles: Any,
    in_block_kwargs: Optional[Any] = None,
    comp_block_kwargs: Optional[Any] = None,
    depth: Optional[int] = None,
) -> Tuple[
            List[Type[AbstractBlock]],  # in_block_handles
            List[Type[AbstractBlock]],  # comp_block_handles
            List[dict],                 # in_block_kwargs
            List[dict],                 # comp_block_kwargs
            int                         # inferred_depth  
    ]:
    """
    Validate and prepare block handles and their configurations for a model.
    This function checks if the provided block handles and their configurations
    are valid, expands them if necessary, and returns them in a consistent format.

    :param in_block_handles: A single block handle or a sequence of block handles
        for the input blocks.
    :param comp_block_handles: A single block handle or a sequence of block handles
        for the computation blocks.
    :param in_block_kwargs: Optional sequence of dictionaries with additional
        keyword arguments for the input blocks. If not provided, defaults to a
        sequence of empty dictionaries.
    :param comp_block_kwargs: Optional sequence of dictionaries with additional
        keyword arguments for the computation blocks. If not provided, defaults
        to a sequence of empty dictionaries.
    :param depth: Optional depth of the model. If provided, it will expand the
        in_block_handles and comp_block_handles to match the specified depth.
        If not provided, it will be inferred from the lengths of in_block_handles
        and comp_block_handles.
    """

    in_block_handles = _check_block_handle_sequence(in_block_handles)
    comp_block_handles = _check_block_handle_sequence(comp_block_handles)
    in_block_kwargs = _check_block_kwargs_sequence(in_block_kwargs)
    comp_block_kwargs = _check_block_kwargs_sequence(comp_block_kwargs)

    # Expand if needed
    if len(in_block_handles) == 1 and len(comp_block_handles) == 1:        
        # when both in_block_handles and comp_block_handles are single handles
        # a specified depth is required and expansion will happen accordingly
        
        if depth is None:
            raise ValueError(
                "Expected `depth` when using a single in_block and comp_block handle."
            )
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"`depth` must be a positive integer, got {depth}")
        
        # this is the only case where the input depth is used
        # to control the handle expansion
        in_block_handles *= depth
        comp_block_handles *= depth
        inferred_depth = depth

    elif len(in_block_handles) != len(comp_block_handles):
        # when in_block_handles and comp_block_handles are not of the same length
        # there can be 3 cases:
        
        if len(in_block_handles) == 1:
            # one is a singleton, the other is a sequence
            # expand the singleton to match the length of the other
            # in this case we override the depth
            in_block_handles *= len(comp_block_handles)
            inferred_depth = len(comp_block_handles)
        elif len(comp_block_handles) == 1:
            # same as above
            comp_block_handles *= len(in_block_handles)
            inferred_depth = len(in_block_handles)
        else:
            # both are sequences of different lengths
            # raise an error
            raise ValueError(
                "Mismatch in lengths: in_block_handles and comp_block_handles "
                f"must match or be broadcastable (got {len(in_block_handles)} and {len(comp_block_handles)})"
            )
    else:
        # when both in_block_handles and comp_block_handles are sequences
        # and they match in length,
        # inferred_depth is also overridden irrespective of the input depth
        inferred_depth = len(in_block_handles)

    # in_block_kwargs is only checked against the post validation
    # in_block_handles and need to either be a singleton or a match
    # length sequence of dictionaries.
    if len(in_block_kwargs) == 1:
        in_block_kwargs *= inferred_depth
    elif len(in_block_kwargs) != inferred_depth:
        raise ValueError(
            "Expected in_block_kwargs to match depth or be a singleton. "
            f"Got {len(in_block_kwargs)} and depth {inferred_depth}"
        )

    # comp_block_kwargs is only checked against the post validation
    # comp_block_handles and need to either be a singleton or a match
    # length sequence of dictionaries.
    if len(comp_block_kwargs) == 1:
        comp_block_kwargs *= inferred_depth
    elif len(comp_block_kwargs) != inferred_depth:
        raise ValueError(
            "Expected comp_block_kwargs to match depth or be a singleton. "
            f"Got {len(comp_block_kwargs)} and depth {inferred_depth}"
        )

    return (
        in_block_handles, 
        comp_block_handles, 
        in_block_kwargs,
        comp_block_kwargs, 
        inferred_depth
    )
        
