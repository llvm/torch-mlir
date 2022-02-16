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
    """
    def __init__(self, *shape: int):
        self.shape = list(shape)
    def __repr__(self):
        args_str = ", ".join(repr(x) for x in self.shape)
        return f"TensorOfShape({args_str})"

class Invocation:
    """Representation of a single op invocation (i.e. list of args to the op).

    This class is used to represent a single invocation of an op in a way that
    we can use to both invoke the shape function and invoke the actual op,
    which have slightly different signatures.

    Specifically, this class has special knowledge of `TensorOfShape` and
    translates it appropriately to either a tensor (for the real op) or a
    `List[int]` (for the shape function).
    """
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = list(args)
        # We assume kwargs don't contain tensors, so they don't need any
        # special handling.
        self.kwargs = kwargs

    def to_shape_function_args(self):
        """Gets positional arguments appropriate for a shape function."""
        args = []
        for arg in self.args:
            if isinstance(arg, TensorOfShape):
                args.append(arg.shape)
            else:
                args.append(arg)
        return args

    def to_real_op_args(self):
        """Gets positional arguments appropriate for the real op."""
        args = []
        for arg in self.args:
            if isinstance(arg, TensorOfShape):
                args.append(torch.ones(arg.shape))
            else:
                args.append(arg)
        return args

    def __repr__(self) -> str:
        args_str = ", ".join(repr(x) for x in self.args)
        kwargs_str = ""
        if self.kwargs:
            kwargs_str = ", " + ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"Invocation({args_str}{kwargs_str})"

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

            # Check for matching error behavior.
            if shape_fn_error is not None or op_error is not None:
                # If both raised errors, then that is good -- the shape function
                # and the real op should agree on the erroneous cases.
                # The exact error message might differ though.
                if shape_fn_error is not None and op_error is not None:
                    continue
                if shape_fn_error is not None:
                    report(f"Shape function raised error {shape_fn_error!r} but op did not raise any error.")
                if op_error is not None:
                    report(f"Op raised error {op_error!r} but shape function did not raise any error.")

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

def aten〇embedding(weight: List[int], indices: List[int], padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> List[int]:
    return shape_helpers.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

def aten〇expand(self: List[int], size: List[int], implicit: bool = False) -> List[int]:
    return shape_helpers.expand(self, size)

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


@check_shape_function([
    Invocation(TensorOfShape(2, 3), TensorOfShape(2, 3)), # Basic case.
    Invocation(TensorOfShape(2, 3), TensorOfShape(3)), # Rank broadcasting.
    Invocation(TensorOfShape(2, 3), TensorOfShape(1, 3)), # Size-1 broadcasting.
    # Error cases.
    Invocation(TensorOfShape(2, 3), TensorOfShape(4, 3)), # Non-size-1 dimension size mismatch.
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

def aten〇unsqueeze(self: List[int], dim: int) -> List[int]:
    return shape_helpers.unsqueeze(self, dim)

def aten〇squeeze(self: List[int]) -> List[int]:
    return shape_helpers.squeeze_nodim(self)

def prim〇NumToTensor〇Scalar(a: float) -> List[int]:
    return []

@check_shape_function([
    Invocation(TensorOfShape(2, 3), 1), # Basic case.
    Invocation(TensorOfShape(2, 3), 2, dim=0), # Test explicit `dim`.
    # Error cases.
    Invocation(TensorOfShape(2, 3), 10), # `k` too big.
    Invocation(TensorOfShape(2, 3), 2, dim=100), # `dim` out of bounds.
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
