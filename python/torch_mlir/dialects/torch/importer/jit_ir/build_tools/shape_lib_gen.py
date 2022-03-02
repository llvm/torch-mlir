# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List, Optional, Any, Tuple, Union

import os
import argparse
import inspect

import torch
from torch import device, Tensor

from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder
from torch_mlir.passmanager import PassManager
import torch_mlir.all_passes_registration

from .registry import Registry
import torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_helpers as shape_helpers


class TensorOfShape:
    """Symbolic placeholder for a tensor argument to an operation.

    Shape functions take tensor arguments as `List[int]`, so we need a symbolic
    representation of a tensor argument to an op in order to reprsent an
    invocation that can drive both the shape function and the real op
    (see `Invocation`).

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

def IntTensorOfShape(*args, **kwargs):
    """Helper for indicating a TensorOfShape with integer type."""
    return TensorOfShape(*args, **kwargs, dtype=torch.long)

def _recursively_convert_to_shape_function_args(o: Any) -> Any:
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
    slips through when both the shape function and the real op fail due to
    independent bugs (that cancel each other out and spurioiusly make the two
    appear to "agree").
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

@check_shape_function([
    Invocation(TensorOfShape(2, 3)),
])
def aten〇tanh(self: List[int]) -> List[int]:
    return shape_helpers.unary(self)

def aten〇relu(self: List[int]) -> List[int]:
    return shape_helpers.unary(self)

def aten〇_softmax(self: List[int], dim: int, half_to_float: bool) -> List[int]:
    return shape_helpers.unary(self)

def aten〇softmax〇int(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return shape_helpers.unary(self)

def aten〇log_softmax〇int(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return shape_helpers.unary(self)

def aten〇clamp(self: List[int], min: Optional[float] = None, max: Optional[float] = None) -> List[int]:
    return shape_helpers.unary(self)

def aten〇rsub〇Scalar(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return shape_helpers.unary(self)

def aten〇to〇dtype(self: List[int], dtype: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return shape_helpers.unary(self)

def aten〇to〇other(self: List[int], other: List[int], non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return shape_helpers.unary(self)

def aten〇type_as(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.unary(self)

def aten〇dropout(input: List[int], p: float, train: bool) -> List[int]:
    return shape_helpers.unary(input)

def aten〇gelu(self: List[int]) -> List[int]:
    return shape_helpers.unary(self)

def aten〇contiguous(self: List[int], memory_format: int = 0) -> List[int]:
    return shape_helpers.unary(self)

def aten〇eq〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇ne〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇gt〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇ge〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇le〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇lt〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇add〇Scalar(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return shape_helpers.unary(self)

def aten〇sub〇Scalar(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return shape_helpers.unary(self)

def aten〇mul〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇div〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇floor_divide〇Scalar(self: List[int], other: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇pow〇Tensor_Scalar(self: List[int], exponent: float) -> List[int]:
    return shape_helpers.unary(self)

def aten〇rsub〇Scalar(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return shape_helpers.unary(self)

def aten〇leaky_relu(self: List[int], negative_slope: float = 0.01) -> List[int]:
    return shape_helpers.unary(self)

def aten〇gather(self: List[int], dim: int, index: List[int], sparse_grad: bool = False) -> List[int]:
    return shape_helpers.unary(index)

def aten〇any(self: List[int]) -> List[int]:
    return []

def aten〇all(self: List[int]) -> List[int]:
    return []

def aten〇sum(self: List[int], dtype: Optional[int] = None) -> List[int]:
    return []

def aten〇mean(self: List[int], dtype: Optional[int] = None) -> List[int]:
    return []

def _reduce_along_dim(self: List[int], dim: int, keepdim: bool):
    dim = shape_helpers.maybe_wrap_dim(dim, len(self))
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

def aten〇mean〇dim(self: List[int], dim: List[int], keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return shape_helpers.mean_dim(self, dim, keepdim, dtype)

def aten〇sum〇dim_IntList(self: List[int], dim: List[int], keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return shape_helpers.mean_dim(self, dim, keepdim, dtype)


def aten〇permute(self: List[int], dims: List[int]) -> List[int]:
    return shape_helpers.permute(self, dims)

def aten〇transpose〇int(self: List[int], dim0: int, dim1: int) -> List[int]:
    return shape_helpers.transpose(self, dim0, dim1)

def aten〇t(self: List[int]) -> List[int]:
    return shape_helpers.transpose(self, 0, 1)

def aten〇matmul(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.matmul(self, other)

def aten〇mm(self: List[int], mat2: List[int]) -> List[int]:
    return shape_helpers.mm(self, mat2)

def aten〇addmm(self: List[int], mat1: List[int], mat2: List[int], beta: float = 1, alpha: float = 1) -> List[int]:
    return shape_helpers.addmm(self, mat1, mat2, beta, alpha)

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

def aten〇embedding(weight: List[int], indices: List[int], padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> List[int]:
    return shape_helpers.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

def aten〇expand(self: List[int], size: List[int], implicit: bool = False) -> List[int]:
    return shape_helpers.expand(self, size)

def aten〇broadcast_to(self: List[int], size: List[int]) -> List[int]:
    return shape_helpers.expand(self, size)

def aten〇view(self: List[int], size: List[int]) -> List[int]:
    return shape_helpers.view(self, size)

def aten〇resize_(self: List[int], size: List[int], memory_format: Optional[int] = None) -> List[int]:
    return size

def aten〇layer_norm(input: List[int], normalized_shape: List[int], weight: Optional[List[int]] = None, bias: Optional[List[int]] = None, eps: float = 1.0000000000000001e-05, cudnn_enable: bool = True) -> List[int]:
    return shape_helpers.unary(input)

def aten〇max_pool2d(self: List[int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False) -> List[int]:
    return shape_helpers.max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode)

def aten〇adaptive_avg_pool2d(self: List[int], output_size: List[int]) -> List[int]:
    return shape_helpers.adaptive_avg_pool2d(self, output_size)

def aten〇flatten〇using_ints(self: List[int], start_dim: int = 0, end_dim: int = -1) -> List[int]:
    return shape_helpers.flatten(self, start_dim, end_dim)

def aten〇linear(input: List[int], weight: List[int], bias: Optional[List[int]] = None) -> List[int]:
    return shape_helpers.linear(input, weight, bias)

@check_shape_function([
    Invocation([2, 3]),
])
def aten〇zeros(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇ones(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇empty〇memory_format(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return size

def aten〇zeros_like(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return shape_helpers.unary(self)

def aten〇ones_like(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return shape_helpers.unary(self)

def aten〇empty_like(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return shape_helpers.unary(self)

def aten〇arange〇start_step(start: float, end: float, step: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return shape_helpers.arange_start_step(start, end, step, dtype, layout, device, pin_memory)

def aten〇arange〇start(start: float, end: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return shape_helpers.arange_start(start, end, dtype, layout, device, pin_memory)

def aten〇arange(end: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return shape_helpers.arange_end(end, dtype, layout, device, pin_memory)

@check_shape_function([
    Invocation(TensorOfShape(2, 3), TensorOfShape(2, 3)), # Basic case.
    Invocation(TensorOfShape(2, 3), TensorOfShape(3)), # Rank broadcasting.
    Invocation(TensorOfShape(2, 3), TensorOfShape(1, 3)), # Size-1 broadcasting.
    ErrorInvocation(TensorOfShape(2, 3), TensorOfShape(4, 3)), # Non-size-1 dimension size mismatch.
])
def aten〇add〇Tensor(self: List[int], other: List[int], alpha: float = 1) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇sub〇Tensor(self: List[int], other: List[int], alpha: float = 1) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇mul〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇div〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇__and__〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇minimum(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇maximum(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇bitwise_and〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇threshold_backward(grad_output: List[int], self: List[int], threshold: float) -> List[int]:
    return shape_helpers.broadcast(grad_output, self)

def aten〇eq〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇gt〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇lt〇Tensor(self: List[int], other: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, other)

def aten〇unsqueeze(self: List[int], dim: int) -> List[int]:
    return shape_helpers.unsqueeze(self, dim)

def aten〇squeeze(self: List[int]) -> List[int]:
    return shape_helpers.squeeze_nodim(self)

def aten〇squeeze〇dim(self: List[int], dim: int) -> List[int]:
    return shape_helpers.squeeze(self, dim)


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
    return shape_helpers.broadcast(condition, shape_helpers.broadcast(self, other))

def aten〇lerp〇Tensor(self: List[int], end: List[int], weight: List[int]) -> List[int]:
    return shape_helpers.broadcast(self, shape_helpers.broadcast(end, weight))

def aten〇addcmul(self: List[int], tensor1: List[int], tensor2: List[int], value: float = 1) -> List[int]:
    return shape_helpers.broadcast(self, shape_helpers.broadcast(tensor1, tensor2))

def aten〇addcdiv(self: List[int], tensor1: List[int], tensor2: List[int], value: float = 1) -> List[int]:
    return shape_helpers.broadcast(self, shape_helpers.broadcast(tensor1, tensor2))

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
    return shape_helpers.conv2d(input, weight, bias, stride, padding, dilation, groups)

def aten〇batch_norm(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool, momentum: float, eps: float, cudnn_enabled: bool) -> List[int]:
    # Torch's symbolic shape analysis is a bit looser about optional
    # arguments than we are, so their batch_norm helper function works
    # even though the `weight` is not `Optional`.
    # Upstream is working to make this more consistent.
    # For now, since this function is so trivial, just write it ourselves.
    #return shape_helpers.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
    return input

def aten〇slice〇Tensor(self: List[int], dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> List[int]:
    return shape_helpers.slice(self, dim, start, end, step)

def aten〇select〇int(self: List[int], dim: int, index: int) -> List[int]:
    return shape_helpers.select(self, dim, index)

def aten〇index_select(self: List[int], dim: int, index: List[int]) -> List[int]:
    return shape_helpers.index_select(self, dim, index)

def aten〇embedding(weight: List[int], indices: List[int], padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> List[int]:
    return shape_helpers.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

@check_shape_function([
    Invocation(TensorOfShape(2, 3), IntTensorOfShape(2), None, 1, -100), # Basic case.
    Invocation(TensorOfShape(3), IntTensorOfShape(), None, 1, -100), # No batch dim.
    Invocation(TensorOfShape(2, 3), IntTensorOfShape(2), None, 0, -100), # No reduction.
    ErrorInvocation(TensorOfShape(2, 3), IntTensorOfShape(7), None, 1, -100), # Mismatched batch dimension.
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

# TODO: Fix shape function (see body).
# @check_shape_function([
#     Invocation(TensorOfShape(2, 5, 2, 2, 3), [2, 2, 3], None, None, 1e-6), # Basic case.
# ])
def aten〇native_layer_norm(input: List[int], normalized_shape: List[int], weight: Optional[List[int]], bias: Optional[List[int]], eps: float) -> Tuple[List[int], List[int], List[int]]:
    reduction_shape: List[int] = []
    # TODO: Fix buggy behavior. TorchToLinalg needs to properly handle the
    # correctly inferred shapes.
    # With input=[2, 5, 2, 2, 3] and normalized_shape=[2, 2, 3], we should get
    # [[2, 5, 2, 2, 3], [2, 5, 1, 1, 1], [2, 5, 1, 1, 1]]
    for i in range(len(normalized_shape), len(input)):
        reduction_shape.append(input[i])
    # Correct code:
    # num_unreduced_dimensions = len(input) - len(normalized_shape)
    # assert num_unreduced_dimensions >= 0
    # for i in range(num_unreduced_dimensions):
    #     reduction_shape.append(input[i])
    # for i in range(num_unreduced_dimensions, len(input)):
    #     reduction_shape.append(1)
    return input, reduction_shape, reduction_shape

@check_shape_function([
    Invocation(TensorOfShape(2), [1, 2]), # Basic case.
    Invocation(TensorOfShape(2, 3), [1, 2, 3, 4]), # More dimensions.
    Invocation(TensorOfShape(2, 3, 4), [1, 2, 3, 4]), # More dimensions than padded dimensions.
    ErrorInvocation(TensorOfShape(2), [1, 2, 3, 4]), # Too many pad values.
    ErrorInvocation(TensorOfShape(2), [1]), # Unpaired pad value.
])
def aten〇constant_pad_nd(self: List[int], pad: List[int], value: float = 0) -> List[int]:
    assert len(pad) % 2 == 0, "Must have paired low-high pad amount values"
    assert len(pad) // 2 <= len(self), "Number of padded dimensions must be less than or equal to the input dimension"
    # The `pad` list takes the form of Low-high pairs starting at the
    # *rightmost* dimension of `self`.
    for i in range(len(pad) // 2):
        self[-(i + 1)] += pad[2 * i] + pad[2 * i + 1]
    return self

@check_shape_function([
    Invocation(TensorOfShape(2), [IntTensorOfShape(4)]), # Basic case.
    Invocation(TensorOfShape(2, 3), [IntTensorOfShape(4), IntTensorOfShape(4)]), # More dimensions.
    Invocation(TensorOfShape(2, 3), [IntTensorOfShape(4), IntTensorOfShape(6, 4)]), # Multidimensional index tensor along a dimension.
    Invocation(TensorOfShape(2, 3), [IntTensorOfShape(4), None]), # Explicit None value.
    Invocation(TensorOfShape(2, 3), [IntTensorOfShape(4, 5, 6), IntTensorOfShape(1, 5, 1)]), # Broadcasting of index tensors.
    Invocation(TensorOfShape(2, 3), [IntTensorOfShape(4)]), # Fewer index tensors than dimensions.
    ErrorInvocation(TensorOfShape(2, 3), [IntTensorOfShape(4), IntTensorOfShape(4), IntTensorOfShape(4)]), # More index tensors than dimensions.
])
def aten〇index〇Tensor(self: List[int], indices: List[Optional[List[int]]]) -> List[int]:
    assert len(indices) <= len(self), "More indices than dimensions to index"
    broadcasted_shape: List[int] = []
    for index_tensor_shape in indices:
        if index_tensor_shape is not None:
            broadcasted_shape = shape_helpers.broadcast(broadcasted_shape, index_tensor_shape)
    return broadcasted_shape

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

def main(args):
    mb = ModuleBuilder()
    # We use the registry to ensure that the shape functions are consistent
    # with the ops.
    registry = Registry.load()
    for k, v in globals().items():
        if "〇" not in k:
            continue
        _verify_signature_matches_registry(v, registry)
        # Add it to the compilation unit.
        torch.jit.script(v)
    for function in torch.jit._state._python_cu.get_functions():
        mb.import_function(function)
    # Clean up the IR a bit before writing it out.
    pm = PassManager.parse("canonicalize", context=mb.module.context)
    pm.run(mb.module)
    # Munge the IR a bit to make it more systematically accessible.
    asm = mb.module.operation.get_asm()
    # Put the `〇` back to a regular `.`.
    asm = asm.replace("\\E3\\80\\87", ".")
    # Use a unique prefix on functon names to avoid collisions with
    # user-defined symbols.
    asm = asm.replace("__torch__.aten", "__torch_mlir_shape_fn.aten")
    asm = asm.replace("__torch__.prim", "__torch_mlir_shape_fn.prim")

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
  constexpr StringLiteral shapeLib(R"mlir(
{asm})mlir");
  return shapeLib;
}}""")

def _create_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_ods")
    parser.add_argument(
        "--torch_transforms_cpp_dir",
        required=True,
        help="Directory containing the Torch transforms cpp files")
    return parser

if __name__ == "__main__":
    main(_create_argparse().parse_args())
