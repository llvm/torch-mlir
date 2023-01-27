# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List, Optional, Any, Tuple, Union
import argparse
import os

import torch
from torch import device
import torch.jit._shape_functions as upstream_shape_functions

from .testing_framework import Invocation, ErrorInvocation, TensorOfShape, LongTensorOfShape, NonZeroDTensorWithDtype, ZeroDTensorWithDtype, check_shape_function, check_dtype_function
from .library_generator import generate_library, not_present_in_registry, promote_dtypes, get_dtype_of_scalar, is_integer_dtype, is_float_dtype, is_complex_dtype

# ==============================================================================
# Shape Functions
# ==============================================================================

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

def aten〇triu〡shape(self: List[int], diagonal: int = 0) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇tanh〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇erf〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇sigmoid〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇hardsigmoid〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇softplus〡shape(self: List[int], beta: float = 1, threshold: float = 20) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇square〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇hardswish〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇silu〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇exp〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇expm1〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇sin〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇cos〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇hardtanh〡shape(self: List[int], min_val: float = -1, max_val: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇sqrt〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def prims〇sqrt〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇neg〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇floor〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇detach〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇log2〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇log1p〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇rsqrt〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇abs〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇reciprocal〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇tanh_backward〡shape(grad_output: List[int], output: List[int]) -> List[int]:
    return upstream_shape_functions.unary(grad_output)

def aten〇gelu_backward〡shape(grad_output: List[int], self: List[int], approximate: str = "none") -> List[int]:
    return upstream_shape_functions.unary(grad_output)

def aten〇leaky_relu_backward〡shape(grad_output: List[int], self: List[int], negative_slope: float, self_is_result: bool) -> List[int]:
    return upstream_shape_functions.unary(grad_output)

def aten〇ceil〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇log〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇mish〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇relu〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇relu6〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇round〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇_softmax〡shape(self: List[int], dim: int, half_to_float: bool) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇softmax〇int〡shape(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇_log_softmax〡shape(self: List[int], dim: int, half_to_float: bool) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇log_softmax〇int〡shape(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇clamp〡shape(self: List[int], min: Optional[float] = None, max: Optional[float] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇clamp_min〡shape(self: List[int], min: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇clamp_max〡shape(self: List[int], max: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇rsub〇Scalar〡shape(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def prims〇convert_element_type〡shape(a: List[int], dtype: int) -> List[int]:
    return upstream_shape_functions.unary(a)

def aten〇to〇dtype〡shape(self: List[int], dtype: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇to〇dtype_layout〡shape(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return self

def aten〇to〇device〡shape(self: List[int], device: device, dtype: int, non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇to〇other〡shape(self: List[int], other: List[int], non_blocking: bool = False, copy: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇type_as〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇dropout〡shape(input: List[int], p: float, train: bool) -> List[int]:
    return upstream_shape_functions.unary(input)

def aten〇gelu〡shape(self: List[int], approximate: str = "none") -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇contiguous〡shape(self: List[int], memory_format: int = 0) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇clone〡shape(self: List[int], memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇lift_fresh_copy〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇_log_softmax_backward_data〡shape(grad_output: List[int], output: List[int], dim: int, input_dtype: int) -> List[int]:
    return upstream_shape_functions.unary(grad_output)

def aten〇eq〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇ne〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇gt〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇ge〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇le〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇lt〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇add〇Scalar〡shape(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇sub〇Scalar〡shape(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇mul〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇div〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇remainder〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇floor_divide〇Scalar〡shape(self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇pow〇Tensor_Scalar〡shape(self: List[int], exponent: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇pow〇Tensor_Tensor〡shape(self: List[int], exponent: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, exponent)

def aten〇rsub〇Scalar〡shape(self: List[int], other: float, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇leaky_relu〡shape(self: List[int], negative_slope: float = 0.01) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇prelu〡shape(self: List[int], weight: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇gather〡shape(self: List[int], dim: int, index: List[int], sparse_grad: bool = False) -> List[int]:
    return upstream_shape_functions.unary(index)

def aten〇layer_norm〡shape(input: List[int], normalized_shape: List[int], weight: Optional[List[int]] = None, bias: Optional[List[int]] = None, eps: float = 1.0000000000000001e-05, cudnn_enable: bool = True) -> List[int]:
    return upstream_shape_functions.unary(input)

def aten〇_softmax_backward_data〡shape(grad_output: List[int], output: List[int], dim: int, input_dtype: int) -> List[int]:
    return upstream_shape_functions.unary(output)

def aten〇any〡shape(self: List[int]) -> List[int]:
    return []

def aten〇all〡shape(self: List[int]) -> List[int]:
    return []

def aten〇max〡shape(self: List[int]) -> List[int]:
    return []

def aten〇sum〡shape(self: List[int], dtype: Optional[int] = None) -> List[int]:
    return []

def aten〇mean〡shape(self: List[int], dtype: Optional[int] = None) -> List[int]:
    return []

def aten〇var〡shape(self: List[int], unbiased: bool = True) -> List[int]:
    return []

def prims〇var〡shape(inp: List[int], dims: Optional[List[int]], correction: int, output_dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(inp, dims, False, None)

def aten〇var〇dim〡shape(self: List[int], dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇var〇correction〡shape(self: List[int], dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇var_mean〇correction〡shape(self: List[int], dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False) -> Tuple[List[int], List[int]]:
    out = upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)
    return out, out

def aten〇var_mean〡shape(self: List[int], unbiased: bool = True) -> Tuple[List[int], List[int]]:
    return [], []

def aten〇std〡shape(self: List[int], unbiased: bool = True) -> List[int]:
    return []

def aten〇std〇dim〡shape(self: List[int], dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇std〇correction〡shape(self: List[int], dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False) -> List[int]:
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
def aten〇argmax〡shape(self: List[int], dim: Optional[int] = None, keepdim: bool = False) -> List[int]:
    if dim is None:
        return []
    return _reduce_along_dim(self, dim, keepdim)

def aten〇any〇dim〡shape(self: List[int], dim: int, keepdim: bool = False) -> List[int]:
    return _reduce_along_dim(self, dim, keepdim)

def aten〇max〇dim〡shape(self: List[int], dim: int, keepdim: bool = False) -> Tuple[List[int], List[int]]:
    reduced_shape = _reduce_along_dim(self, dim, keepdim)
    return reduced_shape, reduced_shape

def aten〇amax〡shape(self: List[int], dim: List[int] = (), keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇mean〇dim〡shape(self: List[int], dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, dtype)

def aten〇sum〇dim_IntList〡shape(self: List[int], dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, dtype)

def aten〇permute〡shape(self: List[int], dims: List[int]) -> List[int]:
    return upstream_shape_functions.permute(self, dims)

def aten〇transpose〇int〡shape(self: List[int], dim0: int, dim1: int) -> List[int]:
    return upstream_shape_functions.transpose(self, dim0, dim1)

def aten〇t〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.transpose(self, 0, 1)

def aten〇numpy_T〡shape(self: List[int]) -> List[int]:
    result_shape: List[int] = []
    for i in self:
        result_shape.insert(0, i)
    return result_shape

def aten〇matmul〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.matmul(self, other)

def aten〇mv〡shape(self: List[int], vec: List[int]) -> List[int]:
    return upstream_shape_functions.mv(self, vec)

def aten〇mm〡shape(self: List[int], mat2: List[int]) -> List[int]:
    return upstream_shape_functions.mm(self, mat2)

def aten〇addmm〡shape(self: List[int], mat1: List[int], mat2: List[int], beta: float = 1, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.addmm(self, mat1, mat2, beta, alpha)

@check_shape_function([
    Invocation(TensorOfShape(2, 3, 4), TensorOfShape(2, 4, 5)), # Basic case.
    ErrorInvocation(TensorOfShape(2, 3, 7), TensorOfShape(2, 4, 5)), # mismatching contracting dimension.
    ErrorInvocation(TensorOfShape(7, 3, 4), TensorOfShape(2, 4, 5)), # mismatching batch dimension.
    ErrorInvocation(TensorOfShape(7, 3), TensorOfShape(2, 4, 5)), # LHS is not rank 3.
    ErrorInvocation(TensorOfShape(2, 3, 4), TensorOfShape(2, 4)), # RHS is not rank 3.
])
def aten〇bmm〡shape(self: List[int], mat2: List[int]) -> List[int]:
    assert len(self) == 3, "bmm only supports 3D tensors"
    assert len(mat2) == 3, "bmm only supports 3D tensors"
    assert self[0] == mat2[0], "mismatching batch dimension"
    assert self[2] == mat2[1], "mismatching contracting dimension"
    return [self[0], self[1], mat2[2]]

def aten〇baddbmm〡shape(self: List[int], batch1: List[int], batch2: List[int], beta: float = 1, alpha: float = 1) -> List[int]:
    assert len(batch1) == 3, "baddbmm only supports 3D tensors"
    assert len(batch2) == 3, "baddbmm only supports 3D tensors"
    assert batch1[0] == batch2[0], "mismatching batch dimension"
    assert batch1[2] == batch2[1], "mismatching contracting dimension"
    return [batch1[0], batch1[1], batch2[2]]

def aten〇embedding〡shape(weight: List[int], indices: List[int], padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> List[int]:
    return upstream_shape_functions.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

def aten〇repeat〡shape(self: List[int], repeats: List[int]) -> List[int]:
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

def aten〇roll〡shape(self: List[int], shifts: List[int], dims: List[int] = ()) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇expand〡shape(self: List[int], size: List[int], implicit: bool = False) -> List[int]:
    return upstream_shape_functions.expand(self, size)

def aten〇expand_as〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.unary(other)

def aten〇broadcast_to〡shape(self: List[int], size: List[int]) -> List[int]:
    return upstream_shape_functions.expand(self, size)

def aten〇view〡shape(self: List[int], size: List[int]) -> List[int]:
    return upstream_shape_functions.view(self, size)

def aten〇reshape〡shape(self: List[int], shape: List[int]) -> List[int]:
    return upstream_shape_functions.view(self, shape)

def aten〇_reshape_alias〡shape(self: List[int], size: List[int], stride: List[int]) -> List[int]:
    return upstream_shape_functions.view(self, size)

def aten〇_unsafe_view〡shape(self: List[int], size: List[int]) -> List[int]:
    return size

def aten〇resize_〡shape(self: List[int], size: List[int], memory_format: Optional[int] = None) -> List[int]:
    return size

def aten〇max_pool2d〡shape(self: List[int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False) -> List[int]:
    return upstream_shape_functions.max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode)

def aten〇max_pool2d_with_indices〡shape(self: List[int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False) -> Tuple[List[int], List[int]]:
    maxpool2d = indices = upstream_shape_functions.max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode)
    return maxpool2d, indices

def aten〇max_pool2d_with_indices_backward〡shape(grad_output: List[int], self: List[int], kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, indices: List[int]) -> List[int]:
    return self

def aten〇upsample_nearest2d_backward〡shape(grad_output: List[int], output_size: List[int], input_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None) -> List[int]:
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

def aten〇avg_pool2d〡shape(self: List[int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None) -> List[int]:
    return avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

def aten〇adaptive_avg_pool2d〡shape(self: List[int], output_size: List[int]) -> List[int]:
    return upstream_shape_functions.adaptive_avg_pool2d(self, output_size)

def aten〇flatten〇using_ints〡shape(self: List[int], start_dim: int = 0, end_dim: int = -1) -> List[int]:
    return upstream_shape_functions.flatten(self, start_dim, end_dim)

def aten〇linear〡shape(input: List[int], weight: List[int], bias: Optional[List[int]] = None) -> List[int]:
    return upstream_shape_functions.linear(input, weight, bias)

@check_shape_function([
    Invocation([2, 3]),
])
def aten〇zeros〡shape(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇ones〡shape(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇empty〇memory_format〡shape(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return size

def aten〇full〡shape(size: List[int], fill_value: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇full_like〡shape(self: List[int], fill_value: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return self

def aten〇zeros_like〡shape(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇ones_like〡shape(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇empty_like〡shape(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇new_zeros〡shape(self: List[int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇new_ones〡shape(self: List[int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇new_empty〡shape(self: List[int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇_to_copy〡shape(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, memory_format: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇masked_fill〇Scalar〡shape(self: List[int], mask: List[int], value: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇masked_fill〇Tensor〡shape(self: List[int], mask: List[int], value: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇zero〡shape(self: List[int]) -> List[int]:
    return self

def aten〇fill〇Tensor〡shape(self: List[int], value: List[int]) -> List[int]:
    return self

def aten〇fill〇Scalar〡shape(self: List[int], value: float) -> List[int]:
    return self

def aten〇copy〡shape(self: List[int], src: List[int], non_blocking: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇uniform〡shape(self: List[int], from_: float = 0., to: float = 1., generator: Any = None) -> List[int]:
    return self

@not_present_in_registry
def aten〇bernoulli〇float〡shape(self: List[int], p: float = 0.5, generator: Any = None) -> List[int]:
    return self

def aten〇bernoulli〇Tensor〡shape(self: List[int], p: List[int], generator: Any = None) -> List[int]:
    return self

def aten〇_index_put_impl〡shape(self: List[int], indices: List[Optional[List[int]]], values: List[int], accumulate: bool = False, unsafe: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇bernoulli〡shape(self: List[int], generator: Any = None) -> List[int]:
    return self

def aten〇cumsum〡shape(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return self

def aten〇rand_like〡shape(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return self

def aten〇randint〇low〡shape(low: int, high: int, size: List[int], dtype: Optional[int] = 4, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇randn〡shape(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇randn〇generator〡shape(size: List[int], generator: Any, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return size

def aten〇arange〇start_step〡shape(start: float, end: float, step: float = 1, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return upstream_shape_functions.arange_start_step(start, end, step, dtype, layout, device, pin_memory)

def aten〇arange〇start〡shape(start: float, end: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return upstream_shape_functions.arange_start(start, end, dtype, layout, device, pin_memory)

def aten〇arange〡shape(end: float, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
    return upstream_shape_functions.arange_end(end, dtype, layout, device, pin_memory)

@check_shape_function([
    Invocation(TensorOfShape(2, 3), TensorOfShape(2, 3)), # Basic case.
    Invocation(TensorOfShape(2, 3), TensorOfShape(3)), # Rank broadcasting.
    Invocation(TensorOfShape(2, 3), TensorOfShape(1, 3)), # Size-1 broadcasting.
    ErrorInvocation(TensorOfShape(2, 3), TensorOfShape(4, 3)), # Non-size-1 dimension size mismatch.
])
def aten〇add〇Tensor〡shape(self: List[int], other: List[int], alpha: float = 1) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇sub〇Tensor〡shape(self: List[int], other: List[int], alpha: float = 1) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇mul〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇div〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇div〇Tensor_mode〡shape(self: List[int], other: List[int], rounding_mode: Optional[str]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇floor_divide〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇atan2〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇__and__〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇minimum〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇maximum〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇bitwise_or〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇bitwise_and〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇bitwise_xor〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇bitwise_not〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇logical_or〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇logical_and〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇logical_xor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇logical_not〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇threshold〡shape(self: List[int], threshold: float, value: float) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇threshold_backward〡shape(grad_output: List[int], self: List[int], threshold: float) -> List[int]:
    return upstream_shape_functions.broadcast(grad_output, self)

def aten〇eq〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇gt〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇ge〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇lt〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇le〇Tensor〡shape(self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, other)

def aten〇unsqueeze〡shape(self: List[int], dim: int) -> List[int]:
    return upstream_shape_functions.unsqueeze(self, dim)

def aten〇squeeze〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.squeeze_nodim(self)

def aten〇squeeze〇dim〡shape(self: List[int], dim: int) -> List[int]:
    return upstream_shape_functions.squeeze(self, dim)

def prim〇NumToTensor〇Scalar〡shape(a: float) -> List[int]:
    return []

def aten〇tensor〇float〡shape(t: float, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> List[int]:
    return []

def aten〇tensor〇int〡shape(t: int, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> List[int]:
    return []

def aten〇tensor〇bool〡shape(t: bool, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> List[int]:
    return []

@check_shape_function([
    Invocation(TensorOfShape()),
    Invocation(TensorOfShape(2, 3)),
])
def aten〇_shape_as_tensor〡shape(self: List[int]) -> List[int]:
    return [len(self)]

def aten〇where〇self〡shape(condition: List[int], self: List[int], other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(condition, upstream_shape_functions.broadcast(self, other))

def aten〇where〇Scalar〡shape(condition: List[int], self: float, other: float) -> List[int]:
    return upstream_shape_functions.unary(condition)

def aten〇where〇ScalarOther〡shape(condition: List[int], self: List[int], other: float) -> List[int]:
    return upstream_shape_functions.broadcast(condition, self)

def aten〇where〇ScalarSelf〡shape(condition: List[int], self: float, other: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(condition, other)

def aten〇lerp〇Tensor〡shape(self: List[int], end: List[int], weight: List[int]) -> List[int]:
    return upstream_shape_functions.broadcast(self, upstream_shape_functions.broadcast(end, weight))

def aten〇addcmul〡shape(self: List[int], tensor1: List[int], tensor2: List[int], value: float = 1) -> List[int]:
    return upstream_shape_functions.broadcast(self, upstream_shape_functions.broadcast(tensor1, tensor2))

def aten〇addcdiv〡shape(self: List[int], tensor1: List[int], tensor2: List[int], value: float = 1) -> List[int]:
    return upstream_shape_functions.broadcast(self, upstream_shape_functions.broadcast(tensor1, tensor2))

@check_shape_function([
    Invocation(TensorOfShape(2, 3), 1), # Basic case.
    Invocation(TensorOfShape(2, 3), 2, dim=0), # Test explicit `dim`.
    ErrorInvocation(TensorOfShape(2, 3), 10), # `k` too big.
    ErrorInvocation(TensorOfShape(2, 3), 2, dim=100), # `dim` out of bounds.
])
def aten〇topk〡shape(self: List[int], k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple[List[int], List[int]]:
    assert k <= self[dim], f"k ({k}) is too big for dimension {dim} of size {self[dim]}"
    # All lists which represent tensor shapes are expected to be the result
    # of a fresh invocation of `AtenSizeOp`, which allocates a new, unaliased
    # list. So in-place mutations are ok.
    self[dim] = k
    return self, self

def aten〇conv2d〡shape(input: List[int], weight: List[int], bias: Optional[List[int]] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), groups: int = 1) -> List[int]:
    return upstream_shape_functions.conv2d(input, weight, bias, stride, padding, dilation, groups)

def aten〇conv_transpose2d〇input〡shape(input: List[int], weight: List[int], bias: Optional[List[int]] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), output_padding: List[int] = (0, 0), groups: int = 1, dilation: List[int] = (1, 1)) -> List[int]:
    return upstream_shape_functions.conv_transpose2d_input(input, weight, bias, stride, padding, output_padding, groups, dilation)

def aten〇convolution〡shape(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int) -> List[int]:
    return upstream_shape_functions.conv_forwards(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

def aten〇_convolution〡shape(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, allow_tf32: bool) -> List[int]:
    return aten〇convolution〡shape(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

def aten〇_convolution〇deprecated〡shape(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool) -> List[int]:
    return aten〇convolution〡shape(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)

def aten〇flip〡shape(self: List[int], dims: List[int]) -> List[int]:
    return self

def aten〇convolution_backward〡shape(grad_output: List[int], input: List[int], weight: List[int], bias_sizes: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]) -> Tuple[List[int], List[int], List[int]]:
    return upstream_shape_functions.conv_backwards(grad_output, input, weight, bias_sizes)

def aten〇convolution_backward_overrideable〡shape(grad_output: List[int], input: List[int], weight: List[int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]) -> Tuple[List[int], List[int], List[int]]:
    return upstream_shape_functions.conv_backwards(grad_output, input, weight, None)

def aten〇batch_norm〡shape(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool, momentum: float, eps: float, cudnn_enabled: bool) -> List[int]:
    # Torch's symbolic shape analysis is a bit looser about optional
    # arguments than we are, so their batch_norm helper function works
    # even though the `weight` is not `Optional`.
    # Upstream is working to make this more consistent.
    # For now, since this function is so trivial, just write it ourselves.
    #return upstream_shape_functions.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
    return input

def aten〇slice〇Tensor〡shape(self: List[int], dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> List[int]:
    return upstream_shape_functions.slice(self, dim, start, end, step)

def aten〇narrow〡shape(self: List[int], dim: int, start: int, length: int) -> List[int]:
    return upstream_shape_functions.slice(self, dim, start, start + length, 1)

def aten〇slice_scatter〡shape(self: List[int], src: List[int], dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> List[int]:
    return self

def aten〇select〇int〡shape(self: List[int], dim: int, index: int) -> List[int]:
    return upstream_shape_functions.select(self, dim, index)

def aten〇select_scatter〡shape(self: List[int], src: List[int], dim: int, index: int) -> List[int]:
    return self

def aten〇index_select〡shape(self: List[int], dim: int, index: List[int]) -> List[int]:
    return upstream_shape_functions.index_select(self, dim, index)

def aten〇index_put〡shape(self: List[int], indices: List[Optional[List[int]]], values: List[int], accumulate: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇index_put〇hacked_twin〡shape(self: List[int], indices: List[List[int]], values: List[int], accumulate: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇embedding〡shape(weight: List[int], indices: List[int], padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> List[int]:
    return upstream_shape_functions.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

def aten〇embedding_bag〇padding_idx〡shape(weight: List[int], indices: List[int], offsets: List[int], scale_grad_by_freq: bool, mode: int, sparse: bool, per_sample_weights: Optional[List[int]], include_last_offset: bool, padding_idx: Optional[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    return _embedding_bag_helper(weight, indices, offsets, include_last_offset, mode)

def aten〇_embedding_bag〡shape(weight: List[int], indices: List[int], offsets: List[int], scale_grad_by_freq: bool = False, mode: int = 0, sparse: bool = False, per_sample_weights: Optional[List[int]] = None, include_last_offset: bool = False, padding_idx: int = -1) -> Tuple[List[int], List[int], List[int], List[int]]:
    return _embedding_bag_helper(weight, indices, offsets, include_last_offset, mode)

@check_shape_function([
    Invocation(TensorOfShape(2, 3), LongTensorOfShape(2), None, 1, -100), # Basic case.
    Invocation(TensorOfShape(3), LongTensorOfShape(), None, 1, -100), # No batch dim.
    Invocation(TensorOfShape(2, 3), LongTensorOfShape(2), None, 0, -100), # No reduction.
    ErrorInvocation(TensorOfShape(2, 3), LongTensorOfShape(7), None, 1, -100), # Mismatched batch dimension.
])
def aten〇nll_loss_forward〡shape(self: List[int], target: List[int], weight: Optional[List[int]], reduction: int, ignore_index: int) -> Tuple[List[int], List[int]]:
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

def aten〇nll_loss_backward〡shape(grad_output: List[int], self: List[int], target: List[int], weight: Optional[List[int]], reduction: int, ignore_index: int, total_weight: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇mse_loss〡shape(self: List[int], target: List[int], reduction: int = 1) -> List[int]:
    if reduction == 0:
        return upstream_shape_functions.unary(self)
    return []

@check_shape_function([
    Invocation(TensorOfShape(2, 5, 2, 2, 3), [2, 2, 3], None, None, 1e-6), # Basic case.
])
def aten〇native_layer_norm〡shape(input: List[int], normalized_shape: List[int], weight: Optional[List[int]], bias: Optional[List[int]], eps: float) -> Tuple[List[int], List[int], List[int]]:
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
def aten〇native_batch_norm〡shape(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool, momentum: float, eps: float) -> Tuple[List[int], List[int], List[int]]:
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
def aten〇constant_pad_nd〡shape(self: List[int], pad: List[int], value: float = 0) -> List[int]:
    return pad_shape_fn(self, pad)

def aten〇pad〡shape(self: List[int], pad: List[int], mode: str = "constant", value: Optional[float] = None) -> List[int]:
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
def aten〇index〇Tensor〡shape(self: List[int], indices: List[Optional[List[int]]]) -> List[int]:
    return index_tensor_like(self, indices)

def aten〇index〇Tensor_hacked_twin〡shape(self: List[int], indices: List[List[int]]) -> List[int]:
    optional_indices: List[Optional[List[int]]] = [x for x in indices]
    return index_tensor_like(self, optional_indices)

def aten〇cat〡shape(tensors: List[List[int]], dim: int = 0) -> List[int]:
    return upstream_shape_functions.cat(tensors, dim)

def aten〇fft_fft〡shape(self: List[int], n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> List[int]:
    return self

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

def aten〇bincount〡shape(self: List[int], weights: Optional[List[int]] = None, minlength: int = 0) -> List[int]:
    return [hacky_get_unknown_dimension_size()]

def aten〇linalg_vector_norm〡shape(self: List[int], ord: float = 2, dim: Optional[List[int]] = None, keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, dtype)

def aten〇frobenius_norm〇dim〡shape(self: List[int], dim: List[int], keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, 0)

def aten〇upsample_nearest2d〡shape(self: List[int], output_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None) -> List[int]:
    return [self[0], self[1], output_size[0], output_size[1]]

# ==============================================================================
# Dtype Functions
# ==============================================================================

# All the torch types sorted in decreasing order of priority during type promotion.
_SORTED_TORCH_TYPES = [
    torch.complex128, torch.complex64,
    torch.float64, torch.float32, torch.float16, torch.bfloat16,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool
]

def _check_tensors_with_the_same_dtype(
        num_of_tensors: Optional[int] = None,
        tensor_shapes: Optional[list[tuple[int]]] = None,
        error_types: Optional[set[int]] = None, *args, **kwargs):
    """Create invocations where all tensors have the same dtype.

    This function generates invocations with `num_of_tensors` tensors
    that all have the same dtype. It creates an invocation for every
    possible dtype. For dtypes in `error_types`, the invocations are
    error invocations.

    One can also specify the shapes of the tensors. Either `num_of_tensors`
    or `tensor_shapes` must be specified whenever this function is called.

    The extra *args and **kwargs arguments are passed to the invocations.
    """
    invocations = []
    for type_ in _SORTED_TORCH_TYPES:
        tensors = []
        if tensor_shapes is None and num_of_tensors is not None:
            tensors = [NonZeroDTensorWithDtype(type_)] * num_of_tensors
        elif tensor_shapes is not None and num_of_tensors is None:
            for tensor_shape in tensor_shapes:
                tensors.append(TensorOfShape(*tensor_shape, dtype=type_))
        else:
            assert False, \
                "Either `num_of_tensors` or `tensor_shapes` must be specified"

        if error_types is not None and type_ in error_types:
            invocations.append(ErrorInvocation(*tensors, *args, **kwargs))
        else:
            invocations.append(Invocation(*tensors, *args, **kwargs))
    return invocations

def _check_two_tensor_op(
        input_error_types: Optional[set[int]] = None,
        output_error_types: Optional[set[int]] = None, **kwargs):
    """Generate invocations for basic two-tensor dtype functions.

    This helper function is meant to be used to check dtype functions that
    take two tensor operands and either return the promoted result or
    return a constant dtype based on the tensor dtypes.

    The testing performed is thorough enough to be able to detect if dtypes
    are invalid as inputs or as outputs to the PyTorch op. Invalid dtypes
    must be specified in `input_error_types` and `output_error_types` to
    ensure the invocations are error invocations.
    """
    if input_error_types is not None and output_error_types is not None:
        assert len(input_error_types.intersection(output_error_types)) == 0, \
            "An invalid input type implies an invalid output type, " \
            "so there is no need to repeat the type in the `output_error_types` set"
    all_error_types = set()
    all_error_types |= set() if input_error_types is None else input_error_types
    all_error_types |= set() if output_error_types is None else output_error_types

    def check_two_tensors_with_one_varying_dtype_at_a_time(**kwargs):
        """Create invocations where one tensor varies its dtype.

        This helper function creates invocations with two tensors where one
        tensor varies its dtype while the other one stays constant. The varying
        is done for both tensors and the varying is performed over every possible
        dtype.

        This function helps identify when a dtype is an invalid input dtype
        for dtype functions that do promotion.
        """
        # We will only create invocations for dtypes with priorities less than
        # or equal to the highest priority valid type. By setting the non-varying
        # tensor dtype to be the highest priority valid type, we ensure that
        # every promotion results in a valid dtype. This allows the invocations
        # to test in isolation assertions on input types.
        constant_type = None
        constant_type_index = None
        for e, type_ in enumerate(_SORTED_TORCH_TYPES):
            if type_ not in all_error_types:
                constant_type = type_
                constant_type_index = e
                break
        assert constant_type is not None, \
            "Unable to find a constant type. Make sure the union of " \
            "`input_error_types` and `output_error_types` is not all possible types."

        invocations = []
        for type_ in _SORTED_TORCH_TYPES[constant_type_index:]:
            tensor_1 = NonZeroDTensorWithDtype(type_)
            tensor_2 = NonZeroDTensorWithDtype(constant_type)
            if input_error_types is not None and type_ in input_error_types:
                invocations += [ErrorInvocation(tensor_1, tensor_2, **kwargs),
                                ErrorInvocation(tensor_2, tensor_1, **kwargs)]
            else:
                invocations += [Invocation(tensor_1, tensor_2, **kwargs),
                                Invocation(tensor_2, tensor_1, **kwargs)]
        return invocations

    same_dtype_invocations = _check_tensors_with_the_same_dtype(
        num_of_tensors=2, error_types=all_error_types, **kwargs)

    varying_dtype_invocations = \
        check_two_tensors_with_one_varying_dtype_at_a_time(**kwargs)
    return same_dtype_invocations + varying_dtype_invocations

def _get_dtype_of_floating_point_op(input_dtype: int) -> int:
    if (is_float_dtype(input_dtype) and input_dtype != torch.float32) \
       or is_complex_dtype(input_dtype):
        return input_dtype
    return torch.float32

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇tanh〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇exp〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(
    num_of_tensors=1, error_types={torch.float16, torch.complex64, torch.complex128}))
def aten〇expm1〡dtype(self_rank: int, self_dtype: int) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    assert self_dtype != torch.float16, "`self` cannot have float16 dtype"
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇sin〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇cos〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇sigmoid〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇reciprocal〡dtype(self_rank: int, self_dtype: int) -> int:
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇sqrt〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇log〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇log2〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇log1p〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def aten〇rsqrt〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16, torch.complex64, torch.complex128}))
def aten〇erf〡dtype(self_rank: int, self_dtype: int) -> int:
    assert not is_complex_dtype(self_dtype) and self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(
    num_of_tensors=1, error_types={torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
                                   torch.int64, torch.float16, torch.complex64, torch.complex128}))
def aten〇softplus〡dtype(self_rank: int, self_dtype: int, beta: Union[int, float] = 1, threshold: Union[int, float] = 20) -> int:
    assert not is_complex_dtype(self_dtype) and not is_integer_dtype(self_dtype) and self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(
    num_of_tensors=1, error_types={torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}, dim=[0]))
def aten〇frobenius_norm〇dim〡dtype(self_rank: int, self_dtype: int, dim: List[int], keepdim: bool = False) -> int:
    assert not is_integer_dtype(self_dtype)
    if self_dtype == torch.complex128:
        return torch.float64
    elif self_dtype == torch.complex64:
        return torch.float32
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.float16}))
def prims〇sqrt〡dtype(self_rank: int, self_dtype: int) -> int:
    assert self_dtype != torch.float16
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇all〡dtype(self_rank: int, self_dtype: int) -> int:
    return torch.uint8 if self_dtype == torch.uint8 else torch.bool

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇any〡dtype(self_rank: int, self_dtype: int) -> int:
    return torch.uint8 if self_dtype == torch.uint8 else torch.bool

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇eq〇Scalar〡dtype(self_rank: int, self_dtype: int, other: Union[int, float]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇eq〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=0.0) +
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=0))
def aten〇ge〇Scalar〡dtype(self_rank: int, self_dtype: int, other: Union[int, float]) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=0.0) +
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=0))
def aten〇gt〇Scalar〡dtype(self_rank: int, self_dtype: int, other: Union[int, float]) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    return torch.bool

@check_dtype_function(
    _check_two_tensor_op(input_error_types={torch.complex64, torch.complex128}))
def aten〇gt〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    assert not is_complex_dtype(other_dtype), "`self` cannot be complex"
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=0.0) +
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=0))
def aten〇le〇Scalar〡dtype(self_rank: int, self_dtype: int, other: Union[int, float]) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇logical_and〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    return torch.bool

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇logical_not〡dtype(self_rank: int, self_dtype: int) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇logical_or〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇logical_xor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=0.0) +
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=0))
def aten〇lt〇Scalar〡dtype(self_rank: int, self_dtype: int, other: Union[int, float]) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    return torch.bool

@check_dtype_function(
    _check_two_tensor_op(input_error_types={torch.complex64, torch.complex128}))
def aten〇lt〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    assert not is_complex_dtype(other_dtype), "`self` cannot be complex"
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇ne〇Scalar〡dtype(self_rank: int, self_dtype: int, other: Union[int, float]) -> int:
    return torch.bool

@check_dtype_function([
    Invocation(0.0, 0.0), # float, float
    Invocation(0.0, 0), # float, int
    Invocation(0, 0.0), # int, float
    Invocation(0, 0), # int, int
])
def aten〇add〡dtype(a: Union[int, float], b: Union[int, float]) -> int:
    ranks: List[Optional[int]] = [None, None]
    dtypes = [get_dtype_of_scalar(a), get_dtype_of_scalar(b)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.float16, torch.bfloat16}))
def aten〇fft_fft〡dtype(self_rank: int, self_dtype: int, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> int:
    if is_complex_dtype(self_dtype):
        return self_dtype
    elif self_dtype == torch.float:
        return torch.complex64
    elif self_dtype == torch.double:
        return torch.complex128
    elif is_integer_dtype(self_dtype):
        return torch.complex64
    else:
        assert False, "Unsupported dtype"

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.bool}, other=0.0) +
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1, error_types={torch.bool}, other=0))
def aten〇rsub〇Scalar〡dtype(self_rank: int, self_dtype: int, other: Union[int, float], alpha: Union[int, float] = 1) -> int:
    assert self_dtype != torch.bool, "`self` cannot have bool dtype"
    return promote_dtypes([self_rank, None], [self_dtype, get_dtype_of_scalar(other)])

@check_dtype_function(
    _check_two_tensor_op(input_error_types={torch.float16, torch.bfloat16, torch.float32,
                                            torch.float64, torch.complex64, torch.complex128}))
def aten〇__and__〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert is_integer_dtype(self_dtype), "Expected `self` to have integer dtype"
    assert is_integer_dtype(other_dtype), "Expected `other` to have integer dtype"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op())
def aten〇add〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int, alpha: Union[int, float] = 1) -> int:
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_two_tensor_op(input_error_types={torch.float16, torch.bfloat16, torch.float32,
                                            torch.float64, torch.complex64, torch.complex128}))
def aten〇bitwise_and〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert is_integer_dtype(self_dtype), "Expected `self` to have integer dtype"
    assert is_integer_dtype(other_dtype), "Expected `other` to have integer dtype"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_two_tensor_op(input_error_types={torch.float16, torch.bfloat16, torch.float32,
                                            torch.float64, torch.complex64, torch.complex128}))
def aten〇bitwise_or〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert is_integer_dtype(self_dtype), "Expected `self` to have integer dtype"
    assert is_integer_dtype(other_dtype), "Expected `other` to have integer dtype"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_two_tensor_op(input_error_types={torch.float16, torch.bfloat16, torch.float32,
                                            torch.float64, torch.complex64, torch.complex128}))
def aten〇bitwise_xor〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert is_integer_dtype(self_dtype), "Expected `self` to have integer dtype"
    assert is_integer_dtype(other_dtype), "Expected `other` to have integer dtype"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        tensor_shapes=[(2, 3, 4), (2, 4, 3)], error_types={torch.float16, torch.bool}) +
    # Different width
    [ErrorInvocation(TensorOfShape(2, 3, 4, dtype=torch.float64),
                     TensorOfShape(2, 4, 3, dtype=torch.float32)),
     # Different type
     ErrorInvocation(TensorOfShape(2, 3, 4, dtype=torch.float32),
                     TensorOfShape(2, 4, 3, dtype=torch.int32))])
def aten〇bmm〡dtype(self_rank: int, self_dtype: int, mat2_rank: int, mat2_dtype: int) -> int:
    assert self_dtype not in [torch.float16, torch.bool], \
        "Expected dtype of `self` to not be float16 or bool"
    assert mat2_dtype not in [torch.float16, torch.bool], \
        "Expected dtype of `mat2` to not be float16 or bool"
    assert self_dtype == mat2_dtype, "`self` and `mat2` must have the same dtype"
    return self_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇div〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    if is_complex_dtype(promoted_dtype) or \
       (is_float_dtype(promoted_dtype) and promoted_dtype != torch.float32):
        return promoted_dtype
    else:
        return torch.float32

@check_dtype_function(_check_two_tensor_op(rounding_mode=None))
def aten〇div〇Tensor_mode〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int, rounding_mode: Optional[str]) -> int:
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    if is_complex_dtype(promoted_dtype) or \
       (is_float_dtype(promoted_dtype) and promoted_dtype != torch.float32):
        return promoted_dtype
    else:
        return torch.float32

@check_dtype_function(_check_two_tensor_op(input_error_types={torch.complex64, torch.complex128}, output_error_types={torch.bool}))
def aten〇floor_divide〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    assert not is_complex_dtype(other_dtype), "`other` cannot be complex"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    assert promoted_dtype != torch.bool, "Result dtype for aten.floor_divide bool"
    return promoted_dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        tensor_shapes=[(2, 3, 4), (2, 4, 3)], error_types={torch.float16, torch.bool}) +
    # Different width
    [ErrorInvocation(TensorOfShape(2, 3, 4, dtype=torch.float64),
                     TensorOfShape(2, 4, 3, dtype=torch.float32)),
     # Different type
     ErrorInvocation(TensorOfShape(2, 3, 4, dtype=torch.float32),
                     TensorOfShape(2, 4, 3, dtype=torch.int32))])
def aten〇matmul〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert self_dtype not in [torch.float16, torch.bool], \
        "Expected dtype of `self` to not be float16 or bool"
    assert other_dtype not in [torch.float16, torch.bool], \
        "Expected dtype of `other` to not be float16 or bool"
    assert self_dtype == other_dtype, "`self` and `other` must have the same dtype"
    return self_dtype

@check_dtype_function(_check_two_tensor_op(input_error_types={torch.complex64, torch.complex128}))
def aten〇maximum〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    assert not is_complex_dtype(other_dtype), "`other` cannot be complex"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op(input_error_types={torch.complex64, torch.complex128}))
def aten〇minimum〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    assert not is_complex_dtype(other_dtype), "`other` cannot be complex"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        tensor_shapes=[(3, 4), (4, 3)], error_types={torch.float16, torch.bool}) +
    # Different width
    [ErrorInvocation(TensorOfShape(3, 4, dtype=torch.float64),
                     TensorOfShape(4, 3, dtype=torch.float32)),
     # Different type
     ErrorInvocation(TensorOfShape(3, 4, dtype=torch.float32),
                     TensorOfShape(4, 3, dtype=torch.int32))])
def aten〇mm〡dtype(self_rank: int, self_dtype: int, mat2_rank: int, mat2_dtype: int) -> int:
    assert self_dtype not in [torch.float16, torch.bool], \
        "Expected dtype of `self` to not be float16 or bool"
    assert mat2_dtype not in [torch.float16, torch.bool], \
        "Expected dtype of `mat2` to not be float16 or bool"
    assert self_dtype == mat2_dtype, "`self` and `mat2` must have the same dtype"
    return self_dtype

@check_dtype_function(_check_two_tensor_op(input_error_types={torch.complex64, torch.complex128},
                                           output_error_types={torch.bool, torch.uint8, torch.int8, torch.int16,
                                                               torch.int32, torch.int64, torch.bfloat16}))
def aten〇mse_loss〡dtype(self_rank: int, self_dtype: int, target_rank: int, target_dtype: int, reduction: int = 1) -> int:
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    assert not is_complex_dtype(target_dtype), "`target` cannot be complex"
    ranks: List[Optional[int]] = [self_rank, target_rank]
    dtypes = [self_dtype, target_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    assert is_float_dtype(promoted_dtype) and promoted_dtype != torch.bfloat16, \
        "Expected promoted dtype to be float but not `bfloat16`"
    return promoted_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇mul〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int) -> int:
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        tensor_shapes=[(3, 4), (4,)], error_types={torch.float16, torch.bool}) +
    # Different width
    [ErrorInvocation(TensorOfShape(3, 4, dtype=torch.float64),
                     TensorOfShape(4, dtype=torch.float32)),
     # Different type
     ErrorInvocation(TensorOfShape(3, 4, dtype=torch.float32),
                     TensorOfShape(4, dtype=torch.int32))])
def aten〇mv〡dtype(self_rank: int, self_dtype: int, vec_rank: int, vec_dtype: int) -> int:
    assert self_dtype not in [torch.float16, torch.bool], \
        "Expected dtype of `self` to not be float16 or bool"
    assert vec_dtype not in [torch.float16, torch.bool], \
        "Expected dtype of `vec` to not be float16 or bool"
    assert self_dtype == vec_dtype, "`self` and `vec` must have the same dtype"
    ranks: List[Optional[int]] = [self_rank, vec_rank]
    dtypes = [self_dtype, vec_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op(input_error_types={torch.bool}))
def aten〇sub〇Tensor〡dtype(self_rank: int, self_dtype: int, other_rank: int, other_dtype: int, alpha: Union[int, float] = 1) -> int:
    assert self_dtype != torch.bool, "`self` cannot be of bool dtype"
    assert other_dtype != torch.bool, "`other` cannot be of bool dtype"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op(input_error_types={torch.complex64, torch.complex128}, output_error_types={torch.bool, torch.float16}, threshold=0))
def aten〇threshold_backward〡dtype(grad_output_rank: int, grad_output_dtype: int, self_rank: int, self_dtype: int, threshold: Union[int, float]) -> int:
    assert not is_complex_dtype(grad_output_dtype), "`grad_output` cannot be complex"
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    ranks: List[Optional[int]] = [grad_output_rank, self_rank]
    dtypes = [grad_output_dtype, self_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    assert promoted_dtype not in [torch.bool, torch.float16], \
        "Result dtype for aten.threshold_backward cannot be bool or float16"
    return promoted_dtype

# ==============================================================================
# Main
# ==============================================================================

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
    asm = generate_library(globals())
    # We're about to put quotes around the string, so escape the `"` characters.
    asm = asm.replace("\"", "\\\"")

    # Instead of dumping one big chunk of text that is several thousand lines
    # long (and which causes MSVC to error out), split it into multiple lines.
    # See MSVC Compiler Error C2026
    # [https://docs.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026?view=msvc-170]
    # for details.
    multiple_lines = asm.replace("\n", "\\n\"\n\"")
    asm = f"\"{multiple_lines}\""

    # Write out the library .cpp file.
    abstract_interp_lib_cpp_file = os.path.join(
        args.torch_transforms_cpp_dir, "AbstractInterpLibrary.cpp")
    with open(abstract_interp_lib_cpp_file, "w") as f:
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
// Generated with the script `build_tools/update_abstract_interp_lib.sh`.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;

StringRef mlir::torch::Torch::getAbstractInterpLibrary() {{
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

