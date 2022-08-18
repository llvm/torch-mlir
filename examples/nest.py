from typing import Union, Any, List, Optional, Generator, Callable, Tuple, TypeVar

import torch
from torch.utils._pytree import tree_flatten, _is_leaf

TScope = List[str]
TNested = Union[Any, list, tuple, dict]
TSelectFn = Callable[[Any], bool]

def visit_structure(
    data_structure: TNested,
    select_fn: TSelectFn,
    strict: bool = False,
    scope: Optional[TScope] = None,
) -> Generator[Tuple[TScope, Any], None, None]:
    """Recursively traverse nested structure and return the items accepted by
    the selector.

    Args:
        data_structure: A nested data structure to traverse recursively.
        select_fn: A callable that returns true if the item passed should be
            selected.
        strict: Strictly checks that an item in the nested structure is either
            a list/dict/tuple or selected by the select_fn. Otherwise, raises
            an error. Defaults to False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Yields:
        A tuples of (scope, item) for each item selected by the select_fn.
    """
    objs, spec = tree_flatten(data_structure)
    objs_iter = iter(objs)

    def recurse_spec(spec, scope):
        if spec.context:
            for key, val in zip(spec.context, spec.children_specs):
                yield from recurse_spec(val, scope + [str(key)])
        else:
            obj = next(objs_iter)
            if _is_leaf(obj):
                if select_fn(obj):
                    yield scope, obj
                elif strict:
                    raise ValueError(
                        f"Unknown data structure: {obj}"
                    )
            else:
                for i, val in enumerate(spec.children_specs):
                    yield from recurse_spec(val, scope + [str(i)])

    yield from recurse_spec(spec, scope or [])


def visit_torch_tensors(
    data_structure: TNested,
    strict: bool = False,
    scope: Optional[TScope] = None,
):
    """Recursively finds all torch tensors in the nested data structure.

    Args:
        data_structure: A nested data structure to traverse recursively.
        strict: Strictly checks that an item in the nested structure is one of
            a list/dict/tuple/tensor. Otherwise, raises an error. Defaults to
            False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Yields:
        A tuple of (scope, tensor) for each tensor in the nested structure.
    """
    yield from visit_structure(
        data_structure,
        select_fn=lambda item: isinstance(item, torch.Tensor),
        strict=strict,
        scope=scope,
    )


def visit_lazy_tensors(
    data_structure: TNested,
    strict: bool = False,
    scope: Optional[TScope] = None,
):
    """Recursively finds all Lazy tensors in the nested data structure.

    Args:
        data_structure: A nested data structure to traverse recursively.
        strict: Strictly checks that an item in the nested structure is one of
            a list/dict/tuple/Lazy tensor. Otherwise, raises an error. Defaults
            to False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Yields:
        A Tuple of (scope, Lazy tensor) for each tensor in the nested structure.
    """
    yield from visit_structure(
        data_structure,
        select_fn=lambda item: isinstance(item, torch.Tensor) and item.device.type == "lazy",
        strict=strict,
        scope=scope,
    )
