# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Any, List, Iterable, Optional, Callable

import torch
from torch import Tensor

# ==============================================================================
# Shape, dtype, and decomposition function testing infrastructure.
# ==============================================================================

# We expect all functions to be adequately tested. For functions
# implemented with upstream helpers, additional testing is usually not needed.
# But for functions that are authored/maintained by the Torch-MLIR
# project, we expect adequate testing.
#
# To do this, we provide decorators
# - `@check_shape_function`
# - `@check_dtype_function`
# - `@check_decomposition_function`
# which can be used to specify a series of operator invocations (such as "call
# this operator with two arguments -- a first tensor of size [2, 3] and a second
# tensor of size [3, 4]"). These tests are then run as part of this script, and
# any mismatches from the real op's behavior will be reported.
#
# A typical use of the decorator might look like:
# ```
# @check_shape_function([
#     Invocation(TensorOfShape(2, 3, 4)), # Basic case.
#     Invocation(TensorOfShape(2, 3, 4), dim=0), # Test explicit `dim`.
#     Invocation(TensorOfShape(2, 3, 4), dim=0, keepdim=True), # `keepdim`.
#     Invocation(TensorOfShape(2, 3, 4), dim=-3), # Negative `dim`.
#     Invocation(TensorOfShape(2, 3, 4), dim=2), # Maximum valid `dim`.
#     ErrorInvocation(TensorOfShape(2, 3, 4), dim=-4), # `dim` out of bounds.
#     ErrorInvocation(TensorOfShape(2, 3, 4), dim=3), # `dim` out of bounds.
# ])
# ```
# Each `Invocation` takes a list of args/kwargs which will be passed to both the
# shape function and the real op and the results compared.
# We expect both the successful and error cases to be tested.
#
# The typical iteration flow is to add invocations to the list and then re-run
# `build_tools/update_abstract_interp_lib.sh` to re-run the tests.


class TensorOfShape:
    """Symbolic placeholder for a tensor argument to an operation.

    Shape functions take tensor arguments as `List[int]`, whereas the real ops
    take them as `Tensor`, so we need a symbolic representation of a tensor
    argument to an op in order to represent an invocation that can drive both
    the shape function and the real op (see `Invocation`).

    A plain list doesn't work, because plain lists are actually legal arguments
    to a shape function (e.g. conv dilations), and we don't want them to receive
    this special treatment.

    This class also tracks a dtype of the tensor, since some ops require a
    specific dtype.
    """

    def __init__(
        self,
        *shape: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.shape = list(shape)
        self.dtype = dtype
        self.device = "meta" if device is None else device

    def __repr__(self):
        args_str = ", ".join(repr(x) for x in self.shape)
        return f"TensorOfShape({args_str}, dtype={self.dtype}, device={self.device})"


def LongTensorOfShape(*args, **kwargs):
    """Helper for indicating a TensorOfShape with integer type."""
    return TensorOfShape(*args, **kwargs, dtype=torch.long)


def NonZeroDTensorWithDtype(dtype, device: Optional[torch.device] = None):
    """Helper for indicating a non-zero dim tensor with custom type."""
    return TensorOfShape(1, dtype=dtype, device=device)


def ZeroDTensorWithDtype(dtype, device: Optional[torch.device] = None):
    """Helper for indicating a zero dim tensor with custom type."""
    return TensorOfShape(dtype=dtype, device=device)


def _recursively_transform_tensor_args(
    o: Any, tensor_transformer: Callable[[TensorOfShape], Any]
) -> Any:
    """Replace `TensorOfShape` with the result of `tensor_transformer`"""
    if o is None or isinstance(o, (float, int, str)):
        return o
    if isinstance(o, TensorOfShape):
        return tensor_transformer(o)
    if isinstance(o, list):
        return [_recursively_transform_tensor_args(x, tensor_transformer) for x in o]
    if isinstance(o, tuple):
        return tuple(
            _recursively_transform_tensor_args(x, tensor_transformer) for x in o
        )
    raise Exception(f"Unhandled type {type(o)}")


class Invocation:
    """Representation of a single op invocation (i.e. list of args to the op).

    This class is used to represent a single invocation of an op in a way that
    we can use to both invoke the abstract interpretation function and invoke
    the actual op, which have slightly different signatures.

    Specifically, this class has special knowledge of `TensorOfShape` and
    translates it appropriately to either a tensor (for the real op), a
    `List[int]` for the shape function, and a tuple with two `int`s
    representing the tensor rank and dtype in the case of a dtype function.

    This class also tracks whether the invocation is expected to raise an
    exception for greater precision when interpreting errors raised during
    testing.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = list(args)
        # We assume kwargs don't contain tensors, so they don't need any
        # special handling.
        self.kwargs = kwargs

    def is_expected_to_raise_exception(self) -> bool:
        """Returns true if the invocation is expected to raise an exception.

        The subclass ErrorInvocation overrides this to indicate an Invocation
        that is expected to raise an exception.
        """
        return False

    def to_shape_function_args(self):
        """Gets positional arguments appropriate for a shape function."""
        # Make a copy of the size list, since a shape function might
        # modify it in-place. In the compiler, the lowering always
        # produces a new list via a fresh invocation of `AtenSizeOp`,
        # which allocates a new, unaliased list. So in-place mutations
        # are ok since they make it a bit easier to write some shape
        # functions.
        tensor_transformer = lambda o: list(o.shape)
        return _recursively_transform_tensor_args(self.args, tensor_transformer)

    def to_dtype_function_args(self):
        """Gets positional arguments appropriate for a dtype function."""
        tensor_transformer = lambda o: (len(o.shape), o.dtype)
        return _recursively_transform_tensor_args(self.args, tensor_transformer)

    def to_real_op_args(self):
        """Gets positional arguments appropriate for the real op."""
        tensor_transformer = lambda o: torch.ones(o.shape, dtype=o.dtype).to(o.device)
        return _recursively_transform_tensor_args(self.args, tensor_transformer)

    def __repr__(self) -> str:
        args_str = ", ".join(repr(x) for x in self.args)
        kwargs_str = ""
        if self.kwargs:
            kwargs_str = ", " + ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"Invocation({args_str}{kwargs_str})"


class ErrorInvocation(Invocation):
    """An Invocation that raises an exception.

    Explicitly knowing that an invocation is expected to raise an exception
    avoids certain failure modes of the test infrastructure where a bug
    slips through when both the abstract interpretation function and the real
    op raise exceptions due to independent bugs (that cancel each other out and
    spurioiusly make the two appear to "agree" that an exception needs to be
    raised).
    """

    def is_expected_to_raise_exception(self) -> bool:
        return True


def _normalize_multiple_results_to_list(t: Any):
    """Returns a flat list of results.

    This normalizes the fact that Python represents multiple returns with a
    tuple, but single returns as a single value. We just want a list with
    N elements for N results.
    """
    if isinstance(t, tuple):
        return list(t)
    # Shape functions return List[int] instead of tensors.
    if isinstance(t, (Tensor, list, torch.dtype, int, float)):
        return [t]
    raise ValueError(f"Unexpected type {type(t)}")


def _report(f, invocation: Invocation, error_message: str):
    fn_type = f.__name__.split("〡")[-1]
    raise ValueError(
        f"For {fn_type} function {f.__name__!r} with invocation {invocation}: {error_message}"
    )


def _get_fn_and_golden_results(f, invocation: List[Invocation]):
    """Run the invocation on the library function and torch op.

    If no unexpected errors are detected, returns a tuple wth the first
    element being the results from the library function and the second
    element being the results from the torch op. The results will be `None`
    if the library function and torch op expectedly result in errors.
    """
    fn_name_without_fn_type, fn_type = f.__name__.split("〡")
    fn_name_parts = fn_name_without_fn_type.split("〇")
    ns, unqual = fn_name_parts[:2]
    overload = "default" if len(fn_name_parts) != 3 else fn_name_parts[-1]
    op = getattr(getattr(getattr(torch.ops, ns), unqual), overload)
    fn_error, op_error, fn_results, golden_results = None, None, None, None
    try:
        fn_results = _normalize_multiple_results_to_list(
            f(
                *(getattr(invocation, f"to_{fn_type}_function_args")()),
                **invocation.kwargs,
            )
        )
    except Exception as e:
        fn_error = f"{e}"
    try:
        golden_results = _normalize_multiple_results_to_list(
            op(*invocation.to_real_op_args(), **invocation.kwargs)
        )
    except Exception as e:
        op_error = f"{e}"

    # Check for error behavior.
    if invocation.is_expected_to_raise_exception():
        if fn_error is None and op_error is None:
            _report(
                f,
                invocation,
                f"Expected to raise an exception, but neither {fn_type} function or op raised an exception",
            )
        if fn_error is None:
            _report(
                f,
                invocation,
                f"Op raised error {op_error!r}, but shape/dtype function did not.",
            )
        if op_error is None:
            _report(
                f,
                invocation,
                f"{fn_type} function raised error {fn_error!r}, but op did not.",
            )
    else:
        if fn_error is not None and op_error is not None:
            _report(
                f,
                invocation,
                f"Both {fn_type} function and op raised errors, but were not expected to. {fn_type} function raised error {fn_error!r} and op raised error {op_error!r}.",
            )
        if fn_error is not None:
            _report(
                f,
                invocation,
                f"{fn_type} function raised error {fn_error!r} but op did not raise any error.",
            )
        if op_error is not None:
            _report(
                f,
                invocation,
                f"Op raised error {op_error!r} but {fn_type} function did not raise any error.",
            )

    return fn_results, golden_results


def check_shape_function(invocations: List[Invocation]):
    """Decorator that automatically tests a shape function.

    The shape function, which is expected to be named systematically with
    `〇` instead of `.`, is tested against the corresponding op in
    `torch.ops.*` function using the given invocations.
    """

    def decorator(f):
        for invocation in invocations:
            result_shapes, golden_results = _get_fn_and_golden_results(f, invocation)
            if invocation.is_expected_to_raise_exception():
                continue
            # Check for matching results.
            if len(result_shapes) != len(golden_results):
                _report(
                    f,
                    invocation,
                    f"Expected {len(golden_results)} result shapes, got {len(result_shapes)}",
                )
            for result_shape, golden_result in zip(result_shapes, golden_results):
                result_rank = len(result_shape)
                golden_rank = len(golden_result.shape)
                if result_rank != golden_rank:
                    _report(
                        f,
                        invocation,
                        f"Expected result rank {golden_rank}, got {result_rank}",
                    )
                for dimension_size, golden_dimension_size in zip(
                    result_shape, golden_result.shape
                ):
                    if dimension_size != golden_dimension_size:
                        _report(
                            f,
                            invocation,
                            f"Expected result shape {golden_result.shape}, got {result_shape}",
                        )
        return f

    return decorator


@torch.jit.script
def _convert_dtype_to_int(dtype: torch.dtype) -> int:
    """Convert a PyTorch `dtype` into its underlying `int` representation.

    This works because in TorchScript there is no special type for `dtypes`;
    they are simply `int`s.
    """
    return dtype


def check_dtype_function(invocations: List[Invocation]):
    """Decorator that automatically tests a dtype function.

    The dtype function, which is expected to be named systematically with
    `〇` instead of `.`, is tested against the corresponding op in
    `torch.ops.*` function using the given invocations.
    """

    def decorator(f):
        for invocation in invocations:
            result_dtypes, golden_results = _get_fn_and_golden_results(f, invocation)
            if invocation.is_expected_to_raise_exception():
                continue

            if len(result_dtypes) != len(golden_results):
                _report(
                    f,
                    invocation,
                    f"Expected {len(golden_results)} result dtypes, got {len(result_dtypes)}",
                )
            for result_dtype, golden_result in zip(result_dtypes, golden_results):
                if isinstance(golden_result, torch.Tensor):
                    golden_dtype = golden_result.dtype
                elif isinstance(golden_result, (int, float)):
                    # Turn Python type to PyTorch dtype
                    golden_dtype = torch.tensor([]).to(type(golden_result)).dtype
                else:
                    raise ValueError(f"Unhandled return type {type(golden_result)}")
                # Some dtype funtions have default `dtype` parameters, which are
                # represented as `int` values in the registry. In order to
                # support returning the default `int` value, the comparisons of
                # the result and golden dtypes are done using their underlying
                # `int` representation.
                if _convert_dtype_to_int(result_dtype) != _convert_dtype_to_int(
                    golden_dtype
                ):
                    _report(
                        f,
                        invocation,
                        f"Expected result dtype {golden_dtype}, got {result_dtype}",
                    )
        return f

    return decorator
