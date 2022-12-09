# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import string
from typing import List, Optional, Any, Tuple, Union

import argparse
import importlib
import inspect
import os
import re

import torch
from torch import device, Tensor
import torch.jit._shape_functions as upstream_shape_functions

from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder
from torch_mlir.passmanager import PassManager

from .registry import Registry


# ==============================================================================
# Shape function testing infrastructure.
# ==============================================================================

# We expect all shape functions to be adequately tested. For shape functions
# implemented with upstream helpers, additional testing is usually not needed.
# But for shape functions that are authored/maintained by the Torch-MLIR
# project, we expect adequate testing.
#
# To do this, we provide a decorator `@check_shape_function` which can be used
# to specify a series of operator invocations (such as "call this operator with
# two arguments -- a first tensor of size [2, 3] and a second tensor of size
# [3, 4]"). These tests are then run as part of this script, and any mismatches
# from the real op's behavior will be reported.
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
# `build_tools/update_shape_lib.sh` to re-run the tests.

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
    def __init__(self, *shape: int, dtype: torch.dtype = torch.float32):
        self.shape = list(shape)
        self.dtype = dtype
    def __repr__(self):
        args_str = ", ".join(repr(x) for x in self.shape)
        if self.dtype is torch.float32:
            return f"TensorOfShape({args_str})"
        else:
            return f"TensorOfShape({args_str}, dtype={self.dtype})"

def _embedding_bag_helper(weight: List[int], indices: List[int], offsets: List[int], include_last_offset: bool, mode: int):
    assert len(weight) == 2
    assert len(indices) == 1
    assert len(offsets) == 1
    output_bag_shape: List[int] = []
    out_dim0 = offsets[0]
    if (include_last_offset):
        out_dim0 = out_dim0 - 1
    out_dim1 = weight[1]
    output_bag_shape.append(out_dim0)
    output_bag_shape.append(out_dim1)

    offset2bag_shape: List[int] = []
    if mode == 1:
        offset2bag_shape.append(0)
    else:
        offset2bag_shape = upstream_shape_functions._copy(indices)

    bag_size_shape = upstream_shape_functions._copy(offsets)

    max_indices_shape: List[int] = []
    if mode == 2:
        max_indices_shape = upstream_shape_functions._copy(output_bag_shape)
    else:
        max_indices_shape = upstream_shape_functions._copy(offsets)

    return output_bag_shape, offset2bag_shape, bag_size_shape, max_indices_shape

def LongTensorOfShape(*args, **kwargs):
    """Helper for indicating a TensorOfShape with integer type."""
    return TensorOfShape(*args, **kwargs, dtype=torch.long)

def _recursively_convert_to_shape_function_args(o: Any) -> Any:
    """Converts an Invocation argument to a shape function argument.

    TensorOfShape is replaced with a List[int] for the shape.
    """
    if o is None:
        return None
    if isinstance(o, TensorOfShape):
        # Make a copy of the size list, since a shape function might
        # modify it in-place. In the compiler, the lowering always
        # produces a new list via a fresh invocation of `AtenSizeOp`,
        # which allocates a new, unaliased list. So in-place mutations
        # are ok since they make it a bit easier to write some shape
        # functions.
        return list(o.shape)
    if isinstance(o, list):
        return [_recursively_convert_to_shape_function_args(x) for x in o]
    if isinstance(o, tuple):
        return tuple(_recursively_convert_to_shape_function_args(x) for x in o)
    if isinstance(o, (float, int)):
        return o
    raise Exception(f"Unhandled type {type(o)}")

def _recursively_convert_to_real_op_args(o: Any) -> Any:
    """Converts a shape function argument to a real op argument.

    TensorOfShape is replaced with a tensor of the given shape (and dtype).
    """
    if o is None:
        return None
    if isinstance(o, TensorOfShape):
        return torch.ones(o.shape, dtype=o.dtype)
    if isinstance(o, list):
        return [_recursively_convert_to_real_op_args(x) for x in o]
    if isinstance(o, tuple):
        return tuple(_recursively_convert_to_real_op_args(x) for x in o)
    if isinstance(o, (float, int)):
        return o
    raise Exception(f"Unhandled type {type(o)}")

class Invocation:
    """Representation of a single op invocation (i.e. list of args to the op).

    This class is used to represent a single invocation of an op in a way that
    we can use to both invoke the shape function and invoke the actual op,
    which have slightly different signatures.

    Specifically, this class has special knowledge of `TensorOfShape` and
    translates it appropriately to either a tensor (for the real op) or a
    `List[int]` (for the shape function).

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
        return _recursively_convert_to_shape_function_args(self.args)

    def to_real_op_args(self):
        """Gets positional arguments appropriate for the real op."""
        return _recursively_convert_to_real_op_args(self.args)

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
    slips through when both the shape function and the real op raise exceptions
    due to independent bugs (that cancel each other out and spurioiusly make the
    two appear to "agree" that an exception needs to be raised).
    """
    def is_expected_to_raise_exception(self) -> bool:
        return True

def _normalize_multiple_results_to_list(t: Union[Tensor, Tuple]):
    """Returns a flat list of tensor results.

    This normalizes the fact that Python represents multiple returns with a
    tuple, but single returns as a single value. We just want a list with
    N elements for N results.
    """
    if isinstance(t, tuple):
        return list(t)
    if isinstance(t, Tensor):
        return [t]
    # Shape functions return List[int] instead of tensors.
    if isinstance(t, list):
        return [t]
    raise ValueError(f"Unexpected type {type(t)}")


def check_shape_function(invocations: List[Invocation]):
    """Decorator that automatically tests a shape function.
    
    The shape function, which is expected to be named systematically with
    `〇` instead of `.`, is tested against the corresponding op in
    `torch.ops.*` function using the given invocations.
    """
    def decorator(f):
        # `torch.ops.*` functions are overloaded already, so we don't need
        # to pass in the overload name.
        ns, unqual = f.__name__.split("〇")[:2]
        op = getattr(getattr(torch.ops, ns), unqual)
        for invocation in invocations:
            shape_fn_error, op_error = None, None
            try:
                result_shapes = _normalize_multiple_results_to_list(f(
                    *invocation.to_shape_function_args(),
                    **invocation.kwargs))
            except Exception as e:
                shape_fn_error = f"{e}"
            try:
                golden_results = _normalize_multiple_results_to_list(op(
                    *invocation.to_real_op_args(),
                    **invocation.kwargs))
            except Exception as e:
                op_error = f"{e}"

            def report(error_message: str):
                raise ValueError(f"For shape function {f.__name__!r} with invocation {invocation}: {error_message}")

            # Check for error behavior.
            if invocation.is_expected_to_raise_exception():
                if shape_fn_error is None and op_error is None:
                    report(f"Expected to raise an exception, but neither shape function nor op raised an exception")
                if shape_fn_error is None:
                    report(f"Op raised error {op_error!r}, but shape function did not.")
                if op_error is None:
                    report(f"Shape function raised error {shape_fn_error!r}, but op did not.")
            else:
                if shape_fn_error is not None and op_error is not None:
                    report(f"Both shape function and op raised errors, but were not expected to. Shape function raised error {shape_fn_error!r} and op raised error {op_error!r}.")
                if shape_fn_error is not None:
                    report(f"Shape function raised error {shape_fn_error!r} but op did not raise any error.")
                if op_error is not None:
                    report(f"Op raised error {op_error!r} but shape function did not raise any error.")

            if shape_fn_error is not None or op_error is not None:
                # If both raised errors, then that is good -- the shape function
                # and the real op should agree on the erroneous cases.
                # The exact error message might differ though.
                if shape_fn_error is not None and op_error is not None:
                    continue


            # Check for matching results.
            if len(result_shapes) != len(golden_results):
                report(f"Expected {len(golden_results)} result shapes, got {len(result_shapes)}")
            for result_shape, golden_result in zip(result_shapes, golden_results):
                for dimension_size, golden_dimension_size in zip(result_shape, golden_result.shape):
                    if dimension_size != golden_dimension_size:
                        report(f"Expected result shape {golden_result.shape}, got {result_shape}")
        return f
    return decorator


def not_present_in_registry(f):
    """Decorator for shape functions not present in the shape registry.

    This can happen for "valsem" ops that we have in Torch-MLIR, such as
    torch.valsem.aten.bernoulli.float, which are consistent with PyTorch conventions
    (e.g. being the value-semantic correspondent of torch.aten.bernoulli_.float),
    but that for whatever reason are not present in PyTorch. Such ops are useful
    to keep certain passes within Torch-MLIR as consistent as possible.
    For such ops, in the shape library generator, we treat them as if they
    were registered torch ops (so we don't put "valsem" on them), to keep the
    generator consistent.

    To check if this decorator has been applied, use
    `hasattr(f, "_not_present_in_registry")`.
    """
    f._not_present_in_registry = None
    return f

# ==============================================================================
# Shape functions
# ==============================================================================

def aten〇triu(self: List[int], diagonal: int = 0) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇tanh(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇erf(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇sigmoid(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇hardsigmoid(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇softplus(self: List[int], beta: float = 1, threshold: float = 20) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇square(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇hardswish(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇silu(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇exp(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇expm1(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇sin(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇cos(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇hardtanh(self: List[int], min_val: float = -1, max_val: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇sqrt(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇neg(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇floor(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇detach(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇log2(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇log1p(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇rsqrt(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇abs(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇reciprocal(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇tanh_backward(grad_output: List[int], output: List[int]) -> List[int]:
    return upstream_shape_functions.unary(grad_output)

def aten〇gelu_backward(grad_output: List[int], self: List[int], approximate: str = "none") -> List[int]:
    return upstream_shape_functions.unary(grad_output)

def aten〇ceil(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇log(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇mish(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇relu(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇relu6(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇round(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇_softmax(self: List[int], dim: int, half_to_float: bool) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇softmax〇int(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇_log_softmax(self: List[int], dim: int, half_to_float: bool) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇log_softmax〇int(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇clamp(self: List[int], min: Optional[float] = None, max: Optional[float] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇clamp_min(self: List[int], min: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇clamp_max(self: List[int], max: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇rsub〇Scalar(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇to〇dtype(self: List[int], dtype: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def prims〇convert_element_type(a: List[int], dtype: int) -> List[int]:
    return upstream_shape_functions.unary(a)

def aten〇to〇dtype_layout(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return self

def aten〇to〇device(self: List[int], device: device, dtype: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇to〇other(self: List[int], other: List[int], non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇type_as(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇dropout(input: List[int], p: float, train: bool) -> List[int]:
    return upstream_shape_functions.unary(input)

def aten〇gelu(self: List[int], approximate: str = "none") -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇contiguous(self: List[int], memory_format: int = 0) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇clone(self: List[int], memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇lift_fresh_copy(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇_log_softmax_backward_data(grad_output: List[int], output: List[int], dim: int, input_dtype: int) -> List[int]:
    return upstream_shape_functions.unary(grad_output)

def aten〇eq〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇ne〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇gt〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇ge〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇le〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇lt〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇add〇Scalar(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇sub〇Scalar(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇mul〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇div〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇remainder〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇floor_divide〇Scalar(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇pow〇Tensor_Scalar(self: List[int], exponent: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇pow〇Tensor_Tensor(self: List[int], exponent: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, exponent)

def aten〇rsub〇Scalar(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇leaky_relu(self: List[int], negative_slope: float = 0.01) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇gather(self: List[int], dim: int, index: List[int], sparse_grad: bool = False) -> List[int]:
    return upstream_shape_functions.unary(index)

def aten〇layer_norm(input: List[int], normalized_shape: List[int], weight: Optional[List[int]] = None, bias: Optional[List[int]] = None, eps: float = 1.0000000000000001e-05, cudnn_enable: bool = True) -> List[int]:
    return upstream_shape_functions.unary(input)

def aten〇_softmax_backward_data(grad_output: List[int], output: List[int], dim: int, input_dtype: int) -> List[int]:
    return upstream_shape_functions.unary(output)

def aten〇any(self: List[int]) -> List[int]:
    return []

def aten〇all(self: List[int]) -> List[int]:
    return []

def aten〇max(self: List[int]) -> List[int]:
    return []

def aten〇sum(self: List[int], dtype: Optional[int] = None) -> List[int]:
    return []

def aten〇mean(self: List[int], dtype: Optional[int] = None) -> List[int]:
    return []

def aten〇var(self: List[int], unbiased: bool = True) -> List[int]:
    return []

def aten〇var〇dim(self: List[int], dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇var〇correction(self: List[int], dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇var_mean〇correction(self: List[int], dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False) -> Tuple[List[int], List[int]]:
    out = upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)
    return out, out

def aten〇var_mean(self: List[int], unbiased: bool = True) -> Tuple[List[int], List[int]]:
    return [], []

def aten〇std(self: List[int], unbiased: bool = True) -> List[int]:
    return []

def aten〇std〇dim(self: List[int], dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def _reduce_along_dim(self: List[int], dim: int, keepdim: bool):
    dim = upstream_shape_functions.maybe_wrap_dim(dim, len(self))
    out: List[int] = []
    for i, self_dim in enumerate(self):
        if i == dim:
            if keepdim:
                out.append(1)
        else:
            out.append(self_dim)
    return out

@check_shape_function([
    Invocation(TensorOfShape(2, 3, 4)), # Basic case.
    Invocation(TensorOfShape(2, 3, 4), dim=0), # Test explicit `dim`.
    Invocation(TensorOfShape(2, 3, 4), dim=0, keepdim=True), # `keepdim`.
    Invocation(TensorOfShape(2, 3, 4), dim=-3), # Negative `dim`.
    Invocation(TensorOfShape(2, 3, 4), dim=2), # Maximum valid `dim`.
    ErrorInvocation(TensorOfShape(2, 3, 4), dim=-4), # `dim` out of bounds.
    ErrorInvocation(TensorOfShape(2, 3, 4), dim=3), # `dim` out of bounds.
])
def aten〇argmax(self: List[int], dim: Optional[int] = None, keepdim: bool = False) -> List[int]:
    if dim is None:
        return []
    return _reduce_along_dim(self, dim, keepdim)

def aten〇any〇dim(self: List[int], dim: int, keepdim: bool = False) -> List[int]:
    return _reduce_along_dim(self, dim, keepdim)

def aten〇max〇dim(self: List[int], dim: int, keepdim: bool = False) -> Tuple[List[int], List[int]]:
    reduced_shape = _reduce_along_dim(self, dim, keepdim)
    return reduced_shape, reduced_shape

def aten〇amax(self: List[int], dim: List[int] = (), keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇mean〇dim(self: List[int], dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, dtype)

def aten〇sum〇dim_IntList(self: List[int], dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, dtype)

def aten〇permute(self: List[int], dims: List[int]) -> List[int]:
    return upstream_shape_functions.permute(self, dims)

def aten〇transpose〇int(self: List[int], dim0: int, dim1: int) -> List[int]:
    return upstream_shape_functions.transpose(self, dim0, dim1)

def aten〇t(self: List[int]) -> List[int]:
    return upstream_shape_functions.transpose(self, 0, 1)

def aten〇numpy_T(self: List[int]) -> List[int]:
    result_shape: List[int] = []
    for i in self:
        result_shape.insert(0, i)
    return result_shape

def aten〇matmul(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.matmul(self, other)

def aten〇mv(self: List[int], vec: List[int]) -> List[int]:
    return upstream_shape_functions.mv(self, vec)

def aten〇mm(self: List[int], mat2: List[int]) -> List[int]:
    return upstream_shape_functions.mm(self, mat2)

def aten〇addmm(self: List[int], mat1: List[int], mat2: List[int], beta: float = 1, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.addmm(self, mat1, mat2, beta, alpha)

@check_shape_function([
    Invocation(TensorOfShape(2, 3, 4), TensorOfShape(2, 4, 5)), # Basic case.
    ErrorInvocation(TensorOfShape(2, 3, 7), TensorOfShape(2, 4, 5)), # mismatching contracting dimension.
    ErrorInvocation(TensorOfShape(7, 3, 4), TensorOfShape(2, 4, 5)), # mismatching batch dimension.
    ErrorInvocation(TensorOfShape(7, 3), TensorOfShape(2, 4, 5)), # LHS is not rank 3.
    ErrorInvocation(TensorOfShape(2, 3, 4), TensorOfShape(2, 4)), # RHS is not rank 3.
])
def aten〇bmm(self: List[int], mat2: List[int]) -> List[int]:
    assert len(self) == 3, "bmm only supports 3D tensors"
    assert len(mat2) == 3, "bmm only supports 3D tensors"
    assert self[0] == mat2[0], "mismatching batch dimension"
    assert self[2] == mat2[1], "mismatching contracting dimension"
    return [self[0], self[1], mat2[2]]

def aten〇baddbmm(self: List[int], batch1: List[int], batch2: List[int], beta: float = 1, alpha: float = 1) -> List[int]:
    assert len(batch1) == 3, "baddbmm only supports 3D tensors"
    assert len(batch2) == 3, "baddbmm only supports 3D tensors"
    assert batch1[0] == batch2[0], "mismatching batch dimension"
    assert batch1[2] == batch2[1], "mismatching contracting dimension"
    return [batch1[0], batch1[1], batch2[2]]

def aten〇embedding(weight: List[int], indices: List[int], padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> List[int]:
    return upstream_shape_functions.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

def aten〇repeat(self: List[int], repeats: List[int]) -> List[int]:
    assert len(repeats) >= len(self)
    ndim = len(repeats)
    tensor_dim = len(self)
    if ndim == 0:
        return upstream_shape_functions._copy(self)
    out: List[int] = []
    leading_rank = ndim - tensor_dim
    for i in range(leading_rank):
        out.append(repeats[i])
    for i in range(tensor_dim):
        out.append(self[i] * repeats[i + leading_rank])
    return out

def aten〇roll(self: List[int], shifts: List[int], dims: List[int] = ()) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇expand(self: List[int], size: List[int], implicit: bool = False) -> List[int]:
    return upstream_shape_functions.expand(self, size)

def aten〇expand_as(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.unary(other)

def aten〇broadcast_to(self: List[int], size: List[int]) -> List[int]:
    return upstream_shape_functions.expand(self, size)

def aten〇view(self: List[int], size: List[int]) -> List[int]:
    return upstream_shape_functions.view(self, size)

def aten〇reshape(self: List[int], shape: List[int]) -> List[int]:
    return upstream_shape_functions.view(self, shape)

def aten〇_reshape_alias(self: List[int], size: List[int], stride: List[int]) -> List[int]:
    return upstream_shape_functions.view(self, size)

def aten〇_unsafe_view(self: List[int], size: List[int]) -> List[int]:
    return size

def aten〇resize_(self: List[int], size: List[int], memory_format: Optional[int] = None) -> List[int]:
    return size

def aten〇max_pool2d(self: List[int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False) -> List[int]:
    return upstream_shape_functions.max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode)

def aten〇max_pool2d_with_indices(self: List[int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False) -> Tuple[List[int], List[int]]:
    maxpool2d = indices = upstream_shape_functions.max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode)
    return maxpool2d, indices

def aten〇max_pool2d_with_indices_backward(grad_output: List[int], self: List[int], kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, indices: List[int]) -> List[int]:
    return self

def aten〇upsample_nearest2d_backward(grad_output: List[int], output_size: List[int], input_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None) -> List[int]:
    return input_size

# TODO: This should be upstreamed.
# See https://github.com/pytorch/pytorch/pull/76889 for an example.
def avg_pool2d(input: List[int], kernel_size: List[int], stride: List[int], padding: List[int], ceil_mode: bool, count_include_pad: bool, divisor_override: Optional[int]):
  assert len(kernel_size) == 1 or len(kernel_size) == 2, "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints"
  kH = kernel_size[0]
  kW = kH if len(kernel_size) == 1 else kernel_size[1]

  assert len(stride) == 0 or len(stride) == 1 or len(stride) == 2, "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
  dH = kH if len(stride) == 0 else stride[0]
  if len(stride) == 0:
    dW = kW
  elif len(stride) == 1:
    dW = dH
  else:
    dW = stride[1]

  assert len(padding) == 1 or len(padding) == 2, "avg_pool2d: padding must be either be a single int, or a tuple of two ints"
  padH = padding[0]
  padW = padH if len(padding) == 1 else padding[1]

  dilationH = 1
  dilationW = 1

  assert len(input) == 3 or len(input) == 4

  nbatch = input[-4] if len(input) == 4 else 1
  nInputPlane = input[-3]
  inputHeight = input[-2]
  inputWidth = input[-1]

  outputHeight = upstream_shape_functions.pooling_output_shape(
      inputHeight, kH, padH, dH, dilationH, ceil_mode)
  outputWidth = upstream_shape_functions.pooling_output_shape(
      inputWidth, kW, padW, dW, dilationW, ceil_mode)

  upstream_shape_functions.pool2d_shape_check(
      input, kH, kW, dH, dW, padH, padW, dilationH, dilationW, nInputPlane,
      inputHeight, inputWidth, outputHeight, outputWidth)

  if len(input) == 3:
    return [nInputPlane, outputHeight, outputWidth]
  else:
    return [nbatch, nInputPlane, outputHeight, outputWidth]

def aten〇avg_pool2d(self: List[int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None) -> List[int]:
    return avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

def aten〇adaptive_avg_pool2d(self: List[int], output_size: List[int]) -> List[int]:
    return upstream_shape_functions.adaptive_avg_pool2d(self, output_size)

def aten〇flatten〇using_ints(self: List[int], start_dim: int = 0, end_dim: int = -1) -> List[int]:
    return upstream_shape_functions.flatten(self, start_dim, end_dim)

def aten〇linear(input: List[int], weight: List[int], bias: Optional[List[int]] = None) -> List[int]:
    return upstream_shape_functions.linear(input, weight, bias)

@check_shape_function([
    Invocation([2, 3]),
])
def aten〇zeros(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇ones(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇empty〇memory_format(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return size

def aten〇full(size: List[int], fill_value: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇full_like(self: List[int], fill_value: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return self

def aten〇zeros_like(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇ones_like(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇empty_like(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇new_zeros(self: List[int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇new_ones(self: List[int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇new_empty(self: List[int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇_to_copy(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇masked_fill〇Scalar(self: List[int], mask: List[int], value: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇masked_fill〇Tensor(self: List[int], mask: List[int], value: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇zero(self: List[int]) -> List[int]:
    return self

def aten〇fill〇Tensor(self: List[int], value: List[int]) -> List[int]:
    return self

def aten〇fill〇Scalar(self: List[int], value: float) -> List[int]:
    return self

def aten〇copy(self: List[int], src: List[int], non_blocking: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇uniform(self: List[int], from_: float = 0., to: float = 1., generator: Any = None) -> List[int]:
    return self

@not_present_in_registry
def aten〇bernoulli〇float(self: List[int], p: float = 0.5, generator: Any = None) -> List[int]:
    return self

def aten〇bernoulli〇Tensor(self: List[int], p: List[int], generator: Any = None) -> List[int]:
    return self

def aten〇_index_put_impl(self: List[int], indices: List[Optional[List[int]]], values: List[int], accumulate: bool = False, unsafe: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇bernoulli(self: List[int], generator: Any = None) -> List[int]:
    return self

def aten〇cumsum(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return self

def aten〇rand_like(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return self

def aten〇randint〇low(low: int, high: int, size: List[int], dtype: Optional[int] = 4, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇randn(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇randn〇generator(size: List[int], generator: Any, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇arange〇start_step(start: float, end: float, step: float = 1, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return upstream_shape_functions.arange_start_step(start, end, step, dtype, layout, device, pin_memory)

def aten〇arange〇start(start: float, end: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return upstream_shape_functions.arange_start(start, end, dtype, layout, device, pin_memory)

def aten〇arange(end: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return upstream_shape_functions.arange_end(end, dtype, layout, device, pin_memory)

@check_shape_function([
    Invocation(TensorOfShape(2, 3), TensorOfShape(2, 3)), # Basic case.
    Invocation(TensorOfShape(2, 3), TensorOfShape(3)), # Rank broadcasting.
    Invocation(TensorOfShape(2, 3), TensorOfShape(1, 3)), # Size-1 broadcasting.
    ErrorInvocation(TensorOfShape(2, 3), TensorOfShape(4, 3)), # Non-size-1 dimension size mismatch.
])
def aten〇add〇Tensor(self: List[int], other: List[int], alpha: float = 1) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇sub〇Tensor(self: List[int], other: List[int], alpha: float = 1) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇mul〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇div〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇div〇Tensor_mode(self: List[int], other: List[int], rounding_mode: Optional[str]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇floor_divide(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇atan2(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇__and__〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇minimum(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇maximum(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇bitwise_or〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇bitwise_and〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇bitwise_not(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇logical_or(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇threshold(self: List[int], threshold: float, value: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇threshold_backward(grad_output: List[int], self: List[int], threshold: float) -> List[int]:
    return upstream_shape_functions.broadcast(grad_output, self)

def aten〇eq〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇gt〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇ge〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇lt〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇le〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇unsqueeze(self: List[int], dim: int) -> List[int]:
    return upstream_shape_functions.unsqueeze(self, dim)

def aten〇squeeze(self: List[int]) -> List[int]:
    return upstream_shape_functions.squeeze_nodim(self)

def aten〇squeeze〇dim(self: List[int], dim: int) -> List[int]:
    return upstream_shape_functions.squeeze(self, dim)

def prim〇NumToTensor〇Scalar(a: float) -> List[int]:
    return []

def aten〇tensor〇float(t: float, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> List[int]:
    return []

def aten〇tensor〇int(t: int, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> List[int]:
    return []

def aten〇tensor〇bool(t: bool, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> List[int]:
    return []

@check_shape_function([
    Invocation(TensorOfShape()),
    Invocation(TensorOfShape(2, 3)),
])
def aten〇_shape_as_tensor(self: List[int]) -> List[int]:
    return [len(self)]

def aten〇where〇self(condition: List[int], self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(condition, upstream_shape_functions.broadcast(self, other))

def aten〇where〇Scalar(condition: List[int], self: float, other: float) -> List[int]:
    return upstream_shape_functions.unary(condition)

def aten〇where〇ScalarOther(condition: List[int], self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.broadcast(condition, self)

def aten〇where〇ScalarSelf(condition: List[int], self: float, other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(condition, other)

def aten〇lerp〇Tensor(self: List[int], end: List[int], weight: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, upstream_shape_functions.broadcast(end, weight))

def aten〇addcmul(self: List[int], tensor1: List[int], tensor2: List[int], value: float = 1) -> List[int]:
    return upstream_shape_functions.broadcast(self, upstream_shape_functions.broadcast(tensor1, tensor2))

def aten〇addcdiv(self: List[int], tensor1: List[int], tensor2: List[int], value: float = 1) -> List[int]:
    return upstream_shape_functions.broadcast(self, upstream_shape_functions.broadcast(tensor1, tensor2))

@check_shape_function([
    Invocation(TensorOfShape(2, 3), 1), # Basic case.
    Invocation(TensorOfShape(2, 3), 2, dim=0), # Test explicit `dim`.
    ErrorInvocation(TensorOfShape(2, 3), 10), # `k` too big.
    ErrorInvocation(TensorOfShape(2, 3), 2, dim=100), # `dim` out of bounds.
])
def aten〇topk(self: List[int], k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple[List[int], List[int]]:
    assert k <= self[dim], f"k ({k}) is too big for dimension {dim} of size {self[dim]}"
    # All lists which represent tensor shapes are expected to be the result
    # of a fresh invocation of `AtenSizeOp`, which allocates a new, unaliased
    # list. So in-place mutations are ok.
    self[dim] = k
    return self, self

def aten〇conv2d(input: List[int], weight: List[int], bias: Optional[List[int]] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), groups: int = 1) -> List[int]:
    return upstream_shape_functions.conv2d(input, weight, bias, stride, padding, dilation, groups)

def aten〇conv_transpose2d〇input(input: List[int], weight: List[int], bias: Optional[List[int]] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), output_padding: List[int] = (0, 0), groups: int = 1, dilation: List[int] = (1, 1)) -> List[int]:
    return upstream_shape_functions.conv_transpose2d_input(input, weight, bias, stride, padding, output_padding, groups, dilation)

def aten〇convolution(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int) -> List[int]:
    return upstream_shape_functions.conv_forwards(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

def aten〇_convolution(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, allow_tf32: bool) -> List[int]:
    return aten〇convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

def aten〇_convolution〇deprecated(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool) -> List[int]:
    return aten〇convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

def aten〇flip(self: List[int], dims: List[int]) -> List[int]:
    return self

def aten〇convolution_backward(grad_output: List[int], input: List[int], weight: List[int], bias_sizes: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]) -> Tuple[List[int], List[int], List[int]]:
    return upstream_shape_functions.conv_backwards(grad_output, input, weight, bias_sizes)

def aten〇convolution_backward_overrideable(grad_output: List[int], input: List[int], weight: List[int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]) -> Tuple[List[int], List[int], List[int]]:
    return upstream_shape_functions.conv_backwards(grad_output, input, weight, None)

def aten〇batch_norm(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool, momentum: float, eps: float, cudnn_enabled: bool) -> List[int]:
    # Torch's symbolic shape analysis is a bit looser about optional
    # arguments than we are, so their batch_norm helper function works
    # even though the `weight` is not `Optional`.
    # Upstream is working to make this more consistent.
    # For now, since this function is so trivial, just write it ourselves.
    #return upstream_shape_functions.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
    return input

def aten〇slice〇Tensor(self: List[int], dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> List[int]:
    return upstream_shape_functions.slice(self, dim, start, end, step)

def aten〇narrow(self: List[int], dim: int, start: int, length: int) -> List[int]:
    return upstream_shape_functions.slice(self, dim, start, start + length, 1)

def aten〇slice_scatter(self: List[int], src: List[int], dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> List[int]:
    return self

def aten〇select〇int(self: List[int], dim: int, index: int) -> List[int]:
    return upstream_shape_functions.select(self, dim, index)

def aten〇select_scatter(self: List[int], src: List[int], dim: int, index: int) -> List[int]:
    return self

def aten〇index_select(self: List[int], dim: int, index: List[int]) -> List[int]:
    return upstream_shape_functions.index_select(self, dim, index)

def aten〇index_put(self: List[int], indices: List[Optional[List[int]]], values: List[int], accumulate: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇index_put〇hacked_twin(self: List[int], indices: List[List[int]], values: List[int], accumulate: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇embedding(weight: List[int], indices: List[int], padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> List[int]:
    return upstream_shape_functions.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

def aten〇embedding_bag〇padding_idx(weight: List[int], indices: List[int], offsets: List[int], scale_grad_by_freq: bool, mode: int, sparse: bool, per_sample_weights: Optional[List[int]], include_last_offset: bool, padding_idx: Optional[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    return _embedding_bag_helper(weight, indices, offsets, include_last_offset, mode)

def aten〇_embedding_bag(weight: List[int], indices: List[int], offsets: List[int], scale_grad_by_freq: bool = False, mode: int = 0, sparse: bool = False, per_sample_weights: Optional[List[int]] = None, include_last_offset: bool = False, padding_idx: int = -1) -> Tuple[List[int], List[int], List[int], List[int]]:
    return _embedding_bag_helper(weight, indices, offsets, include_last_offset, mode)

@check_shape_function([
    Invocation(TensorOfShape(2, 3), LongTensorOfShape(2), None, 1, -100), # Basic case.
    Invocation(TensorOfShape(3), LongTensorOfShape(), None, 1, -100), # No batch dim.
    Invocation(TensorOfShape(2, 3), LongTensorOfShape(2), None, 0, -100), # No reduction.
    ErrorInvocation(TensorOfShape(2, 3), LongTensorOfShape(7), None, 1, -100), # Mismatched batch dimension.
])
def aten〇nll_loss_forward(self: List[int], target: List[int], weight: Optional[List[int]], reduction: int, ignore_index: int) -> Tuple[List[int], List[int]]:
    # This is taken shamelessly from the meta function in LossNLL.cpp
    self_dim = len(self)
    target_dim = len(target)
    assert 0 < self_dim <= 2
    assert target_dim <= 1
    no_batch_dim = self_dim == 1 and target_dim == 0
    assert no_batch_dim or (self[0] == target[0])
    n_classes = self[-1]
    scalar_shape: List[int] = []
    assert weight is None or (len(weight) == 1 and weight[0] == n_classes)
    if reduction == 0 and self_dim == 2:
        return [self[0]], scalar_shape
    else:
        return scalar_shape, scalar_shape

def aten〇nll_loss_backward(grad_output: List[int], self: List[int], target: List[int], weight: Optional[List[int]], reduction: int, ignore_index: int, total_weight: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇mse_loss(self: List[int], target: List[int], reduction: int = 1) -> List[int]:
    if reduction == 0:
        return upstream_shape_functions.unary(self)
    return []

@check_shape_function([
    Invocation(TensorOfShape(2, 5, 2, 2, 3), [2, 2, 3], None, None, 1e-6), # Basic case.
])
def aten〇native_layer_norm(input: List[int], normalized_shape: List[int], weight: Optional[List[int]], bias: Optional[List[int]], eps: float) -> Tuple[List[int], List[int], List[int]]:
    reduction_shape: List[int] = []
    num_unreduced_dimensions = len(input) - len(normalized_shape)
    assert num_unreduced_dimensions >= 0
    for i in range(num_unreduced_dimensions):
        reduction_shape.append(input[i])
    for i in range(num_unreduced_dimensions, len(input)):
        reduction_shape.append(1)
    return input, reduction_shape, reduction_shape

@check_shape_function([
    Invocation(TensorOfShape(2, 3), None, None, None, None, True, 1e-4, 1e-6), # Training basic case.
    Invocation(TensorOfShape(2, 3), None, None, TensorOfShape(3), TensorOfShape(3), False, 1e-4, 1e-6), # Inference basic case.
    Invocation(TensorOfShape(2, 3, 4, 5, 6), None, None, None, None, True, 1e-4, 1e-6), # Training high-D case.
    Invocation(TensorOfShape(2, 3, 4, 5, 6), None, None, TensorOfShape(3), TensorOfShape(3), False, 1e-4, 1e-6), # Inference high-D case.
    ErrorInvocation(TensorOfShape(2), None, None, None, None, True, 1e-4, 1e-6) # Dimensionality too low.
])
def aten〇native_batch_norm(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool, momentum: float, eps: float) -> Tuple[List[int], List[int], List[int]]:
    if training:
        return input, [input[1]], [input[1]]
    return input, [0], [0]

# TODO: This should be upstreamed.
# See https://github.com/pytorch/pytorch/pull/76889 for an example.
def pad_shape_fn(input: List[int], pad: List[int]):
    assert len(pad) % 2 == 0, "Must have paired low-high pad amount values"
    assert len(pad) // 2 <= len(input), "Number of padded dimensions must be less than or equal to the input dimension"
    # The `pad` list takes the form of Low-high pairs starting at the
    # *rightmost* dimension of `self`.
    for i in range(len(pad) // 2):
        input[-(i + 1)] += pad[2 * i] + pad[2 * i + 1]
    return input

@check_shape_function([
    Invocation(TensorOfShape(2), [1, 2]), # Basic case.
    Invocation(TensorOfShape(2, 3), [1, 2, 3, 4]), # More dimensions.
    Invocation(TensorOfShape(2, 3, 4), [1, 2, 3, 4]), # More dimensions than padded dimensions.
    ErrorInvocation(TensorOfShape(2), [1, 2, 3, 4]), # Too many pad values.
    ErrorInvocation(TensorOfShape(2), [1]), # Unpaired pad value.
])
def aten〇constant_pad_nd(self: List[int], pad: List[int], value: float = 0) -> List[int]:
    return pad_shape_fn(self, pad)

def aten〇pad(self: List[int], pad: List[int], mode: str = "constant", value: Optional[float] = None) -> List[int]:
    return pad_shape_fn(self, pad)

def index_tensor_like(self: List[int], indices: List[Optional[List[int]]]) -> List[int]:
    assert len(indices) <= len(self), "More indices than dimensions to index"
    broadcasted_shape: List[int] = []
    unused_dim_sizes: List[int] = []
    for i in range(len(self)):
        if i >= len(indices):
            unused_dim_sizes.append(self[i])
        else:
            index_tensor_shape = indices[i]
            if index_tensor_shape is not None:
                broadcasted_shape = upstream_shape_functions.broadcast(broadcasted_shape, index_tensor_shape)
            else:
                unused_dim_sizes.append(self[i])

    if len(unused_dim_sizes) == 0:
        return broadcasted_shape

    first_index_tensor_location = -1
    index_tensors_are_together = True
    for e, index_tensor_shape in enumerate(indices):
        if index_tensor_shape is not None:
            if first_index_tensor_location == -1:
                first_index_tensor_location = e
            elif e - first_index_tensor_location != 1:
                index_tensors_are_together = False

    if not index_tensors_are_together:
        return broadcasted_shape + unused_dim_sizes

    # If index tensors are all in consecutive dimensions, the broadcasted
    # shape is inserted in the location of the consecutive index tensors.
    result_shape: List[int] = []
    for i in range(first_index_tensor_location):
        result_shape.append(unused_dim_sizes[i])
    for broadcasted_size in broadcasted_shape:
        result_shape.append(broadcasted_size)
    for i in range(first_index_tensor_location, len(unused_dim_sizes)):
        result_shape.append(unused_dim_sizes[i])
    return result_shape

# See https://numpy.org/doc/stable/user/basics.indexing.html
@check_shape_function([
    Invocation(TensorOfShape(2), [LongTensorOfShape(4)]), # Basic case.
    Invocation(TensorOfShape(2, 3), [LongTensorOfShape(4), LongTensorOfShape(4)]), # More dimensions.
    Invocation(TensorOfShape(2, 3), [LongTensorOfShape(4), LongTensorOfShape(6, 4)]), # Multidimensional index tensor along a dimension.
    Invocation(TensorOfShape(2, 3), [LongTensorOfShape(4), None]), # Explicit None value.
    Invocation(TensorOfShape(2, 3, 4, 5), [None, LongTensorOfShape(4), LongTensorOfShape(4)]), # Indexing tensors on consecutive dimensions.
    Invocation(TensorOfShape(2, 3, 4, 5), [None, LongTensorOfShape(4), None, LongTensorOfShape(4)]), # Indexing tensors on non-consecutive dimensions.
    Invocation(TensorOfShape(2, 3, 4, 5), [LongTensorOfShape(4, 2), None, LongTensorOfShape(2)]), # Indexing tensors on non-consecutive dimensions.
    Invocation(TensorOfShape(2, 3), [LongTensorOfShape(4, 5, 6), LongTensorOfShape(1, 5, 1)]), # Broadcasting of index tensors.
    Invocation(TensorOfShape(2, 3), [LongTensorOfShape(4)]), # Fewer index tensors than dimensions.
    ErrorInvocation(TensorOfShape(2, 3), [LongTensorOfShape(4), LongTensorOfShape(4), LongTensorOfShape(4)]), # More index tensors than dimensions.
])
def aten〇index〇Tensor(self: List[int], indices: List[Optional[List[int]]]) -> List[int]:
    return index_tensor_like(self, indices)

def aten〇index〇Tensor_hacked_twin(self: List[int], indices: List[List[int]]) -> List[int]:
    optional_indices: List[Optional[List[int]]] = [x for x in indices]
    return index_tensor_like(self, optional_indices)

def aten〇cat(tensors: List[List[int]], dim: int = 0) -> List[int]:
    return upstream_shape_functions.cat(tensors, dim)

class DummyClassType:
    def __init__(self):
        pass

def hacky_get_unknown_dimension_size():
    """Gets a value which symbolically represents an unknown dimension size.

    Note that this is a pretty gross hack, because it breaks the invariant
    that shape functions are executable code that calculates a correct shape.

    We use this for ops that have data-dependent shapes, such as
    `aten::bincount` -- for those, all we need is that
    `torch-shape-refinement-pipeline` sees an opaque integer, but at least we
    can return a shape with a known rank. The way we hackily accomplish that is
    by calling `id` on a freshly allocated class type object, which isn't
    something the compiler can easily reason about.

    TODO: Fix this properly.
    There are 4 main approaches I can think of for fixing this properly:
    1. Add another mechanism in the compiler to allow writing symbolic shape
       functions in C++, which only work for deducing e.g. ranks. The hard part
       here is for this refinement to run to a fixed-point together with
       the rest of the shape functions (i.e., it somehow needs to run in tandem
       with torch-simplify-shape-calculations).
    2. Teach the shape library mechanism how to handle data-dependent shapes,
       such as by allowing passing the actual tensor value (not just its shape)
       to the shape function. This could work for cases like bincount, but
       for other ops the work of determining the size is equivalent to
       actually executing the op (like `torch.unique`), so this gets recursive.
    3. Teach the shape library mechanism about a new type of shape function
       that only returns the rank of the tensor.
    4. Teach the shape library mechanism how to properly indicate a symbolic
       unknown dimension size, along with some sort of way of annotating that
       such a shape function is "not executable".
    Approach 4 seems the most promising, and could probably be implemented by
    registering a custom Torch-MLIR-specific operator in the registry.
    """
    return id(DummyClassType())

def aten〇bincount(self: List[int], weights: Optional[List[int]] = None, minlength: int = 0) -> List[int]:
    return [hacky_get_unknown_dimension_size()]

def aten〇linalg_vector_norm(self: List[int], ord: float = 2, dim: Optional[List[int]] = None, keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, dtype)

def aten〇frobenius_norm〇dim(self: List[int], dim: List[int], keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, 0)

def aten〇upsample_nearest2d(self: List[int], output_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None) -> List[int]:
    return [self[0], self[1], output_size[0], output_size[1]]

# ==============================================================================
# Shape library generator main().
# ==============================================================================

def _verify_signature_matches_registry(f, registry: Registry):
    source = inspect.getsource(f)
    signature = None
    for line in source.splitlines():
        if line.startswith("def "):
            signature = line
            break
    assert signature is not None, f"Could not find signature for {f.__name__}"
    atoms = f.__name__.split("〇")
    if len(atoms) == 2:
        atoms += [""]
    operator = registry.get_by_triple(tuple(atoms))
    expected_signature = operator.get_shape_function_signature()
    if signature != expected_signature:
        raise ValueError(f"Signature mismatch for {f.__name__!r}: expected {expected_signature!r}, got {signature!r}")

def _maybe_import_op_extensions(args: argparse.Namespace):
    extension_string = str.strip(args.pytorch_op_extensions)
    if len(extension_string) > 0:
        extension_names = extension_string.split(",")
        for name in extension_names:
            # Registration of new PyTorch ops should be a side-effect of
            # importing these modules, so we don't need the return value.
            importlib.import_module(name)

def main(args):
    _maybe_import_op_extensions(args)
    mb = ModuleBuilder()
    # We use the registry to ensure that the shape functions are consistent
    # with the ops.
    registry = Registry.load()
    for k, v in globals().items():
        if "〇" not in k:
            continue
        if not hasattr(v, "_not_present_in_registry"):
            _verify_signature_matches_registry(v, registry)
        # Add it to the compilation unit.
        torch.jit.script(v)
    for function in torch.jit._state._python_cu.get_functions():
        mb.import_function(function)
    # Clean up the IR a bit before writing it out.
    pm = PassManager.parse("builtin.module(canonicalize)", context=mb.module.context)
    pm.run(mb.module)
    # Munge the IR a bit to make it more systematically accessible.
    asm = mb.module.operation.get_asm()
    # We'd like a unique function prefix to avoid collisions with user-
    # defined symbols. Since all of our shape functions conveniently have
    # a `〇` in them, we replace the torch namespace with our prefix. E.g.:
    # __torch__.aten〇add〇Scalar -> __torch_mlir_shape_fn.aten〇add〇Scalar
    asm = re.sub(r"__torch__\.([^.(]+)\\E3\\80\\87",
                 r"__torch_mlir_shape_fn.\1\\E3\\80\\87",
                 asm) 
    # Put the `〇` back to a regular `.`.
    asm = asm.replace("\\E3\\80\\87", ".")

    # We're about to put quotes around the string, so escape the `"` characters.
    asm = asm.replace("\"", "\\\"")

    # Instead of dumping one big chunk of text that is several thousand lines
    # long (and which causes MSVC to error out), split it into multiple lines.
    # See MSVC Compiler Error C2026
    # [https://docs.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026?view=msvc-170]
    # for details.
    multiple_lines = asm.replace("\n", "\\n\"\n\"")
    asm = f"\"{multiple_lines}\""

    # Write out the shape library .cpp file.
    shape_lib_cpp_file = os.path.join(
        args.torch_transforms_cpp_dir, "ShapeLibrary.cpp")
    with open(shape_lib_cpp_file, "w") as f:
        p = lambda *args: print(*args, file=f)
        p(
f"""//===-------------------------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// This file is auto-generated! Do not edit!!!
// Generated with the script `build_tools/update_shape_lib.sh`.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;

StringRef mlir::torch::Torch::getShapeLibrary() {{
#ifndef _MSC_VER
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
#endif
  // clang-format off
  return {asm};
  // clang-format on
#ifndef _MSC_VER
#pragma clang diagnostic pop
#endif
}}""")

def _create_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_ods")
    parser.add_argument(
        "--torch_transforms_cpp_dir",
        required=True,
        help="Directory containing the Torch transforms cpp files")
    parser.add_argument(
        "--pytorch_op_extensions",
        type=str,
        default="",
        help="An optional, comma-separated list of Python modules which register additional PyTorch operators upon being imported. These modules can be used to build a torch-mlir which supports PyTorch extensions.")
    return parser

if __name__ == "__main__":
    main(_create_argparse().parse_args())
