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
from .library_generator import generate_library, not_present_in_registry, promote_dtypes, get_dtype_of_scalar, is_integer_dtype, is_float_dtype, is_complex_dtype, get_priority_of_dtype, all_integer_dtypes, all_float_dtypes, all_complex_dtypes

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

def aten〇hardtanh_backward〡shape(grad_output: List[int], self: List[int], min_val: float, max_val: float) -> List[int]:
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

def aten〇bucketize〇Tensor〡shape(self: List[int], boundaries: List[int], out_int32: bool = False, right: bool = False) -> List[int]:
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

def prims〇var〡shape(inp: List[int], dims: Optional[List[int]], correction: float, output_dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(inp, dims, False, None)

def aten〇var〇dim〡shape(self: List[int], dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇var〇correction〡shape(self: List[int], dim: Optional[List[int]] = None, correction: Optional[float] = None, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇var_mean〇correction〡shape(self: List[int], dim: Optional[List[int]] = None, correction: Optional[float] = None, keepdim: bool = False) -> Tuple[List[int], List[int]]:
    out = upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)
    return out, out

def aten〇var_mean〡shape(self: List[int], unbiased: bool = True) -> Tuple[List[int], List[int]]:
    return [], []

def aten〇std〡shape(self: List[int], unbiased: bool = True) -> List[int]:
    return []

def aten〇std〇dim〡shape(self: List[int], dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇std〇correction〡shape(self: List[int], dim: Optional[List[int]] = None, correction: Optional[float] = None, keepdim: bool = False) -> List[int]:
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

def aten〇new_empty_strided〡shape(self: List[int], size: List[int], stride: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
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

def aten〇bernoulli〇p〡shape(self: List[int], p: float, generator: Any = None) -> List[int]:
    return self

def aten〇_index_put_impl〡shape(self: List[int], indices: List[Optional[List[int]]], values: List[int], accumulate: bool = False, unsafe: bool = False) -> List[int]:
    return upstream_shape_functions.unary(self)

def aten〇bernoulli〡shape(self: List[int], generator: Any = None) -> List[int]:
    return self

def aten〇cumsum〡shape(self: List[int], dim: int, dtype: Optional[int] = None) -> List[int]:
    return self

def aten〇rand_like〡shape(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
    return self

def aten〇randn_like〡shape(self: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> List[int]:
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

def aten〇scatter_reduce〇two〡shape(self: List[int], dim: int, index: List[int], src: List[int], reduce: str, include_self: bool = True) -> List[int]:
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
    Invocation(TensorOfShape(2), None, None, None, None, True, 1e-4, 1e-6) # 1D input.
])
def aten〇native_batch_norm〡shape(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool, momentum: float, eps: float) -> Tuple[List[int], List[int], List[int]]:
    if training:
        if len(input) >= 2:
            return input, [input[1]], [input[1]]
        else:
            return input, [], []
    running_mean_list: List[int] = [0] if running_mean is None else running_mean
    running_var_list: List[int] = [0] if running_var is None else running_var
    return input, running_mean_list, running_var_list

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

def aten〇stack〡shape(tensors: List[List[int]], dim: int = 0) -> List[int]:
    return upstream_shape_functions.stack(tensors, dim)

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

def aten〇norm〇ScalarOpt_dim〡shape(self: List[int], p: Optional[float], dim: List[int], keepdim: bool = False) -> List[int]:
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
        tensor_shapes: Optional[list[tuple[int]]] = None,
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
    if tensor_shapes is None:
        tensor_shapes = [(1,), (1,)]
    shape_1, shape_2 = tensor_shapes

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
            if input_error_types is not None and type_ in input_error_types:
                invocation_type = ErrorInvocation
            else:
                invocation_type = Invocation
            invocations += [invocation_type(TensorOfShape(*shape_1, dtype=type_), TensorOfShape(*shape_2, dtype=constant_type), **kwargs),
                            invocation_type(TensorOfShape(*shape_1, dtype=constant_type), TensorOfShape(*shape_2, dtype=type_), **kwargs)]
        return invocations

    same_dtype_invocations = _check_tensors_with_the_same_dtype(
        tensor_shapes=tensor_shapes, error_types=all_error_types, **kwargs)

    varying_dtype_invocations = \
        check_two_tensors_with_one_varying_dtype_at_a_time(**kwargs)
    return same_dtype_invocations + varying_dtype_invocations

def _get_dtype_of_floating_point_op(input_dtype: int) -> int:
    if (is_float_dtype(input_dtype) and input_dtype != torch.float32) \
       or is_complex_dtype(input_dtype):
        return input_dtype
    return torch.float32

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇tanh〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇exp〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇expm1〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇sin〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇cos〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇sigmoid〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇reciprocal〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇sqrt〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇log〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇log2〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇log1p〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇rsqrt〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇erf〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇softplus〡dtype(self_rank_dtype: Tuple[int, int], beta: Union[int, float] = 1, threshold: Union[int, float] = 20) -> int:
    self_rank, self_dtype = self_rank_dtype
    if is_integer_dtype(self_dtype):
        return self_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(
    num_of_tensors=1,
    error_types={torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}, dim=[0]))
def aten〇frobenius_norm〇dim〡dtype(self_rank_dtype: Tuple[int, int], dim: List[int], keepdim: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    assert not is_integer_dtype(self_dtype)
    if self_dtype == torch.complex128:
        return torch.float64
    elif self_dtype == torch.complex64:
        return torch.float32
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def prims〇sqrt〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    if is_integer_dtype(self_dtype):
        return self_dtype
    return _get_dtype_of_floating_point_op(self_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇abs〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.complex128:
        return torch.float64
    elif self_dtype == torch.complex64:
        return torch.float32
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(2, 3, 5, 7)], output_size=[2, 2]))
def aten〇adaptive_avg_pool2d〡dtype(self_rank_dtype: Tuple[int, int], output_size: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(2, 3, 5, 7)], kernel_size=[2, 2]))
def aten〇avg_pool2d〡dtype(self_rank_dtype: Tuple[int, int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(
    tensor_shapes=[(2, 3, 5), (3,), (3,), (3,), (3,)], training=False, momentum=0.1, eps=1e-5, cudnn_enabled=True))
def aten〇batch_norm〡dtype(input_rank_dtype: Tuple[int, int], weight_rank_dtype: Optional[Tuple[int, int]], bias_rank_dtype: Optional[Tuple[int, int]], running_mean_rank_dtype: Optional[Tuple[int, int]], running_var_rank_dtype: Optional[Tuple[int, int]], training: bool, momentum: float, eps: float, cudnn_enabled: bool) -> int:
    input_rank, input_dtype = input_rank_dtype
    return input_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇bernoulli_〇float〡dtype(self_rank_dtype: Tuple[int, int], p: float = 0.5, generator: Any = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇bernoulli〡dtype(self_rank_dtype: Tuple[int, int], generator: Any = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=2))
def aten〇bernoulli〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], p_rank_dtype: Tuple[int, int], generator: Any = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇bitwise_not〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, size=[2, 2]))
def aten〇broadcast_to〡dtype(self_rank_dtype: Tuple[int, int], size: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇ceil〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, max=0))
def aten〇clamp_max〡dtype(self_rank_dtype: Tuple[int, int], max: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.bool:
        return torch.int64
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, min=0))
def aten〇clamp_min〡dtype(self_rank_dtype: Tuple[int, int], min: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.bool:
        return torch.int64
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, min=-1, max=1))
def aten〇clamp〡dtype(self_rank_dtype: Tuple[int, int], min: Optional[Union[int, float]] = None, max: Optional[Union[int, float]] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.bool:
        return torch.int64
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇clone〡dtype(self_rank_dtype: Tuple[int, int], memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, pad=[1, 1]))
def aten〇constant_pad_nd〡dtype(self_rank_dtype: Tuple[int, int], pad: List[int], value: Union[int, float] = 0) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇contiguous〡dtype(self_rank_dtype: Tuple[int, int], memory_format: int = 0) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇copy〡dtype(self_rank_dtype: Tuple[int, int], src_rank_dtype: Tuple[int, int], non_blocking: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

# TODO: Cannot call .cpu() on meta tensor due to: Cannot copy out of meta tensor; no data!
def aten〇cpu〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, dim=0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, dim=0, dtype=torch.float32))
def aten〇cumsum〡dtype(self_rank_dtype: Tuple[int, int], dim: int, dtype: Optional[int] = None) -> int:
    if dtype is not None:
        return dtype
    self_rank, self_dtype = self_rank_dtype
    if is_integer_dtype(self_dtype):
        return torch.int64
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇detach〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, p=0.5, train=False))
def aten〇dropout〡dtype(input_rank_dtype: Tuple[int, int], p: float, train: bool) -> int:
    input_rank, input_dtype = input_rank_dtype
    return input_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇expand_as〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, size=[2, 2]))
def aten〇expand〡dtype(self_rank_dtype: Tuple[int, int], size: List[int], implicit: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, value=0))
def aten〇fill〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], value: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(1,), ()]))
def aten〇fill〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], value_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇flatten〇using_ints〡dtype(self_rank_dtype: Tuple[int, int], start_dim: int = 0, end_dim: int = -1) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dims=[0]))
def aten〇flip〡dtype(self_rank_dtype: Tuple[int, int], dims: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇floor〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(None, [(5,)], None, 0, TensorOfShape(1, dtype=torch.int64)))
def aten〇gather〡dtype(self_rank_dtype: Tuple[int, int], dim: int, index_rank_dtype: Tuple[int, int], sparse_grad: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇gelu_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], self_rank_dtype: Tuple[int, int], approximate: str = "none") -> int:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [grad_output_rank, self_rank]
    dtypes = [grad_output_dtype, self_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    return promoted_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇gelu〡dtype(self_rank_dtype: Tuple[int, int], approximate: str = "none") -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇hardsigmoid〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇hardswish〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_two_tensor_op(min_val=0.2, max_val=0.5))
def aten〇hardtanh_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], self_rank_dtype: Tuple[int, int], min_val: Union[int, float], max_val: Union[int, float]) -> int:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    if is_integer_dtype(grad_output_dtype):
        return torch.float32
    return grad_output_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.uint8, torch.bool}))
def aten〇hardtanh〡dtype(self_rank_dtype: Tuple[int, int], min_val: Union[int, float] = -1, max_val: Union[int, float] = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    assert self_dtype not in [torch.uint8, torch.bool]
    return self_dtype

_index_put_invocations = [
    # same dtype
    Invocation(TensorOfShape(3, dtype=dtype), [TensorOfShape(3, dtype=torch.int64)], TensorOfShape(3, dtype=dtype)) for dtype in _SORTED_TORCH_TYPES
] + [
    # different dtypes
    Invocation(TensorOfShape(3, dtype=dtype), [TensorOfShape(3, dtype=torch.int64)], TensorOfShape(3, dtype=torch.float32)) for dtype in _SORTED_TORCH_TYPES
] + [
    # index dtype
    Invocation(TensorOfShape(3, dtype=torch.float32), [TensorOfShape(3, dtype=dtype)], TensorOfShape(3, dtype=torch.float32)) for dtype in _SORTED_TORCH_TYPES
]
@check_dtype_function(_index_put_invocations)
def aten〇index_put〇hacked_twin〡dtype(self_rank_dtype: Tuple[int, int], indices_rank_dtype: List[Tuple[int, int]], values_rank_dtype: Tuple[int, int], accumulate: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_index_put_invocations)
def aten〇_index_put_impl〡dtype(self_rank_dtype: Tuple[int, int], indices_rank_dtype: List[Optional[Tuple[int, int]]], values_rank_dtype: Tuple[int, int], accumulate: bool = False, unsafe: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_index_put_invocations)
def aten〇index_put〡dtype(self_rank_dtype: Tuple[int, int], indices_rank_dtype: List[Optional[Tuple[int, int]]], values_rank_dtype: Tuple[int, int], accumulate: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(None, [(5,)], None, 0, TensorOfShape(1, dtype=torch.int64)))
def aten〇index_select〡dtype(self_rank_dtype: Tuple[int, int], dim: int, index_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(None, [(5,)], None, [TensorOfShape(1, dtype=torch.int64)]))
def aten〇index〇Tensor_hacked_twin〡dtype(self_rank_dtype: Tuple[int, int], indices_rank_dtype: List[Tuple[int, int]]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(None, [(5,)], None, [TensorOfShape(1, dtype=torch.int64)]))
def aten〇index〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], indices_rank_dtype: List[Optional[Tuple[int, int]]]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(
    num_of_tensors=1, error_types={*all_integer_dtypes()}, normalized_shape=[1]))
def aten〇layer_norm〡dtype(input_rank_dtype: Tuple[int, int], normalized_shape: List[int], weight_rank_dtype: Optional[Tuple[int, int]] = None, bias_rank_dtype: Optional[Tuple[int, int]] = None, eps: float = 1.0000000000000001e-05, cudnn_enable: bool = True) -> int:
    input_rank, input_dtype = input_rank_dtype
    assert not is_integer_dtype(input_dtype)
    return input_dtype

@check_dtype_function(_check_two_tensor_op(negative_slope=0.1, self_is_result=False))
def aten〇leaky_relu_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], self_rank_dtype: Tuple[int, int], negative_slope: Union[int, float], self_is_result: bool) -> int:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [grad_output_rank, self_rank]
    dtypes = [grad_output_dtype, self_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    return promoted_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇lift_fresh_copy〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(
    _check_two_tensor_op(dim=0, input_dtype=torch.float32) +
    _check_two_tensor_op(dim=0, input_dtype=torch.float64))
def aten〇_log_softmax_backward_data〡dtype(grad_output_rank_dtype: Tuple[int, int], output_rank_dtype: Tuple[int, int], dim: int, input_dtype: int) -> int:
    return input_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(None, [(3,)], None, TensorOfShape(1, dtype=torch.bool), 0))
def aten〇masked_fill〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], mask_rank_dtype: Tuple[int, int], value: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(None, [(3,)], None, TensorOfShape(1, dtype=torch.bool), 0))
def aten〇masked_fill_〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], mask_rank_dtype: Tuple[int, int], value: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(None, [(3,)], None, TensorOfShape(1, dtype=torch.bool), TensorOfShape(dtype=torch.float32)))
def aten〇masked_fill〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], mask_rank_dtype: Tuple[int, int], value_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

# TODO: This op cannot be run on the Meta backend. We should have the test suite default on CPU when the Meta backend does not work.
def aten〇masked_select〡dtype(self_rank_dtype: Tuple[int, int], mask_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(2, 3, 5, 7)], kernel_size=[2, 2]))
def aten〇max_pool2d〡dtype(self_rank_dtype: Tuple[int, int], kernel_size: List[int], stride: List[int] = (), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), ceil_mode: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇mish〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=0, start=0, length=1))
def aten〇narrow〡dtype(self_rank_dtype: Tuple[int, int], dim: int, start: int, length: int) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.bool}))
def aten〇neg〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    assert self_dtype != torch.bool
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇numpy_T〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, pad=[1, 1]))
def aten〇pad〡dtype(self_rank_dtype: Tuple[int, int], pad: List[int], mode: str = "constant", value: Optional[float] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dims=[0]))
def aten〇permute〡dtype(self_rank_dtype: Tuple[int, int], dims: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇pow〇Tensor_Tensor〡dtype(self_rank_dtype: Tuple[int, int], exponent_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    exponent_rank, exponent_dtype = exponent_rank_dtype
    ranks: List[Optional[int]] = [self_rank, exponent_rank]
    dtypes = [self_dtype, exponent_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    if promoted_dtype == torch.bool:
        return torch.int64
    return promoted_dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=2) +
    [ErrorInvocation(TensorOfShape(1, dtype=torch.float32), TensorOfShape(1, dtype=torch.float64)),
     ErrorInvocation(TensorOfShape(1, dtype=torch.float64), TensorOfShape(1, dtype=torch.float32))])
def aten〇prelu〡dtype(self_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    weight_rank, weight_dtype = weight_rank_dtype
    assert self_dtype == weight_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.bool}))
def aten〇relu6〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    assert self_dtype != torch.bool
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇relu〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, repeats=[1]))
def aten〇repeat〡dtype(self_rank_dtype: Tuple[int, int], repeats: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], stride=[1]))
def aten〇_reshape_alias〡dtype(self_rank_dtype: Tuple[int, int], size: List[int], stride: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, shape=[1]))
def aten〇reshape〡dtype(self_rank_dtype: Tuple[int, int], shape: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1]))
def aten〇resize_〡dtype(self_rank_dtype: Tuple[int, int], size: List[int], memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, shifts=[0], dims=[0]))
def aten〇roll〡dtype(self_rank_dtype: Tuple[int, int], shifts: List[int], dims: List[int] = ()) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇round〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(
    [Invocation(TensorOfShape(3, dtype=dtype), 0, TensorOfShape(3, dtype=torch.int64), TensorOfShape(3, dtype=dtype), "sum") for dtype in _SORTED_TORCH_TYPES])
def aten〇scatter_reduce〇two〡dtype(self_rank_dtype: Tuple[int, int], dim: int, index_rank_dtype: Tuple[int, int], src_rank_dtype: Tuple[int, int], reduce: str, include_self: bool = True) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=0, index=0))
def aten〇select〇int〡dtype(self_rank_dtype: Tuple[int, int], dim: int, index: int) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_two_tensor_op(tensor_shapes=[(1, 1), (1,)], dim=0, index=0))
def aten〇select_scatter〡dtype(self_rank_dtype: Tuple[int, int], src_rank_dtype: Tuple[int, int], dim: int, index: int) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇silu〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_two_tensor_op(dim=0))
def aten〇slice_scatter〡dtype(self_rank_dtype: Tuple[int, int], src_rank_dtype: Tuple[int, int], dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇slice〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], dim: int = 0, start: Optional[int] = None, end: Optional[int] = None, step: int = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=2, dim=0, input_dtype=torch.float32) +
    _check_tensors_with_the_same_dtype(num_of_tensors=2, dim=0, input_dtype=torch.float64) +
    [Invocation(TensorOfShape(1, dtype=torch.float32), TensorOfShape(1, dtype=torch.float64), dim=0, input_dtype=torch.float32),
     Invocation(TensorOfShape(1, dtype=torch.float64), TensorOfShape(1, dtype=torch.float32), dim=0, input_dtype=torch.float32)])
def aten〇_softmax_backward_data〡dtype(grad_output_rank_dtype: Tuple[int, int], output_rank_dtype: Tuple[int, int], dim: int, input_dtype: int) -> int:
    return input_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇square〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.bool:
        return torch.int64
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=0))
def aten〇squeeze〇dim〡dtype(self_rank_dtype: Tuple[int, int], dim: int) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇squeeze〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇tanh_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], output_rank_dtype: Tuple[int, int]) -> int:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    output_rank, output_dtype = output_rank_dtype
    ranks: List[Optional[int]] = [grad_output_rank, output_rank]
    dtypes = [grad_output_dtype, output_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    return promoted_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, threshold=0, value=0))
def aten〇threshold〡dtype(self_rank_dtype: Tuple[int, int], threshold: Union[int, float], value: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇t〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, device=torch.device("meta")))
def aten〇to〇prim_Device〡dtype(self_rank_dtype: Tuple[int, int], device: Optional[device], dtype: Optional[int] = None, non_blocking: bool = False, copy: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(2, 3)], dim0=0, dim1=1))
def aten〇transpose〇int〡dtype(self_rank_dtype: Tuple[int, int], dim0: int, dim1: int) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(2, 3)]))
def aten〇triu〡dtype(self_rank_dtype: Tuple[int, int], diagonal: int = 0) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇uniform〡dtype(self_rank_dtype: Tuple[int, int], from_: float = 0., to: float = 1., generator: Any = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1]))
def aten〇_unsafe_view〡dtype(self_rank_dtype: Tuple[int, int], size: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=0))
def aten〇unsqueeze〡dtype(self_rank_dtype: Tuple[int, int], dim: int) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(1, 1, 4, 8)], output_size=[4, 8], input_size=[1, 1, 2, 3]))
def aten〇upsample_nearest2d_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], output_size: List[int], input_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None) -> int:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    return grad_output_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(2, 3, 5, 7)], output_size=[11, 13]))
def aten〇upsample_nearest2d〡dtype(self_rank_dtype: Tuple[int, int], output_size: List[int], scales_h: Optional[float] = None, scales_w: Optional[float] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1]))
def aten〇view〡dtype(self_rank_dtype: Tuple[int, int], size: List[int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇zero〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇zero_〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function([Invocation(-1), Invocation(-1.0)])
def prim〇abs〇Scalar〡dtype(a: Union[int, float]) -> int:
    return get_dtype_of_scalar(a)

@check_dtype_function(_check_tensors_with_the_same_dtype(
    None, [(3,), (3, 4)], None,
    TensorOfShape(3, dtype=torch.int64), None, 0, 10, TensorOfShape(1, dtype=torch.float32)) +
    [Invocation(TensorOfShape(3, dtype=torch.float32), TensorOfShape(3, 4, dtype=torch.float64), TensorOfShape(3, dtype=torch.int64), None, 0, 10, TensorOfShape(1, dtype=torch.float32)),
     Invocation(TensorOfShape(3, dtype=torch.float64), TensorOfShape(3, 4, dtype=torch.float32), TensorOfShape(3, dtype=torch.int64), None, 0, 10, TensorOfShape(1, dtype=torch.float32))])
def aten〇nll_loss_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], self_rank_dtype: Tuple[int, int], target_rank_dtype: Tuple[int, int], weight_rank_dtype: Optional[Tuple[int, int]], reduction: int, ignore_index: int, total_weight_rank_dtype: Tuple[int, int]) -> int:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, grad_output_rank]
    dtypes = [self_dtype, grad_output_dtype]
    result = promote_dtypes(ranks, dtypes)
    if result == torch.bool:
        return torch.int64
    return result

@check_dtype_function(_check_tensors_with_the_same_dtype(
    None, [(2, 4, 7, 6), (2, 4, 6, 5)], None,
    [2, 2], [1, 1], [1, 1], [1, 1], False, TensorOfShape(2, 4, 7, 6, dtype=torch.int64)) +
    [ErrorInvocation(TensorOfShape(2, 4, 7, 6, dtype=torch.float32), TensorOfShape(2, 4, 6, 5, dtype=torch.float64), [2, 2], [1, 1], [1, 1], [1, 1], False, TensorOfShape(2, 4, 7, 6, dtype=torch.int64)),
     ErrorInvocation(TensorOfShape(2, 4, 7, 6, dtype=torch.float64), TensorOfShape(2, 4, 6, 5, dtype=torch.float32), [2, 2], [1, 1], [1, 1], [1, 1], False, TensorOfShape(2, 4, 7, 6, dtype=torch.int64))])
def aten〇max_pool2d_with_indices_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], self_rank_dtype: Tuple[int, int], kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, indices_rank_dtype: Tuple[int, int]) -> int:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    assert grad_output_dtype == self_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇all〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return torch.uint8 if self_dtype == torch.uint8 else torch.bool

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇any〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return torch.uint8 if self_dtype == torch.uint8 else torch.bool

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇eq〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇eq〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇ge〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇gt〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇gt〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇ge〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇le〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇logical_and〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇logical_not〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇logical_or〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇logical_xor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇lt〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇lt〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(_check_two_tensor_op())
def aten〇le〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    return torch.bool

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇ne〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
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
    _check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.bfloat16}))
def aten〇fft_fft〡dtype(self_rank_dtype: Tuple[int, int], n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    if is_complex_dtype(self_dtype):
        return self_dtype
    elif self_dtype == torch.float16:
        return torch.complex32
    elif self_dtype == torch.float32:
        return torch.complex64
    elif self_dtype == torch.float64:
        return torch.complex128
    elif is_integer_dtype(self_dtype):
        return torch.complex64
    else:
        assert False, "Unsupported dtype"

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0.0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=0))
def aten〇rsub〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float], alpha: Union[int, float] = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    return promote_dtypes([self_rank, None], [self_dtype, get_dtype_of_scalar(other)])

@check_dtype_function(_check_two_tensor_op())
def aten〇__and__〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op())
def aten〇add〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int], alpha: Union[int, float] = 1) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op())
def aten〇bitwise_and〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op())
def aten〇bitwise_or〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op())
def aten〇bitwise_xor〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(2, 3, 4), (2, 4, 3)]) +
    # Different width
    [Invocation(TensorOfShape(2, 3, 4, dtype=torch.float64),
                TensorOfShape(2, 4, 3, dtype=torch.float32)),
     # Two f16 types
     Invocation(TensorOfShape(2, 3, 4, dtype=torch.float16),
                TensorOfShape(2, 4, 3, dtype=torch.bfloat16)),
     # Different type
     Invocation(TensorOfShape(2, 3, 4, dtype=torch.float32),
                TensorOfShape(2, 4, 3, dtype=torch.int32))])
def aten〇bmm〡dtype(self_rank_dtype: Tuple[int, int], mat2_rank_dtype: Tuple[int, int]) -> int:
    mat2_rank, mat2_dtype = mat2_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    mat2_priority = get_priority_of_dtype(mat2_dtype)
    self_priority = get_priority_of_dtype(self_dtype)
    return mat2_dtype if mat2_priority < self_priority else self_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇div〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    if is_complex_dtype(promoted_dtype) or \
       (is_float_dtype(promoted_dtype) and promoted_dtype != torch.float32):
        return promoted_dtype
    else:
        return torch.float32

@check_dtype_function(_check_two_tensor_op(rounding_mode=None))
def aten〇div〇Tensor_mode〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int], rounding_mode: Optional[str]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    if is_complex_dtype(promoted_dtype) or \
       (is_float_dtype(promoted_dtype) and promoted_dtype != torch.float32):
        return promoted_dtype
    else:
        return torch.float32

@check_dtype_function(_check_two_tensor_op(input_error_types={torch.complex64, torch.complex128}, output_error_types={torch.bool}))
def aten〇floor_divide〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    assert not is_complex_dtype(self_dtype), "`self` cannot be complex"
    assert not is_complex_dtype(other_dtype), "`other` cannot be complex"
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    assert promoted_dtype != torch.bool, "Result dtype for aten.floor_divide bool"
    return promoted_dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(2, 3, 4), (2, 4, 3)]) +
    # Different width
    [Invocation(TensorOfShape(2, 3, 4, dtype=torch.float64),
                TensorOfShape(2, 4, 3, dtype=torch.float32)),
     # Two f16 types
     Invocation(TensorOfShape(2, 3, 4, dtype=torch.float16),
                TensorOfShape(2, 4, 3, dtype=torch.bfloat16)),
     # Different type
     Invocation(TensorOfShape(2, 3, 4, dtype=torch.float32),
                TensorOfShape(2, 4, 3, dtype=torch.int32))])
def aten〇matmul〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    other_priority = get_priority_of_dtype(other_dtype)
    self_priority = get_priority_of_dtype(self_dtype)
    return other_dtype if other_priority < self_priority else self_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇maximum〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op())
def aten〇minimum〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(3, 4), (4, 3)]) +
    # Different width
    [Invocation(TensorOfShape(3, 4, dtype=torch.float64),
                TensorOfShape(4, 3, dtype=torch.float32)),
     # Two f16 types
     Invocation(TensorOfShape(3, 4, dtype=torch.float16),
                TensorOfShape(4, 3, dtype=torch.bfloat16)),
     # Different type
     Invocation(TensorOfShape(3, 4, dtype=torch.float32),
                TensorOfShape(4, 3, dtype=torch.int32))])
def aten〇mm〡dtype(self_rank_dtype: Tuple[int, int], mat2_rank_dtype: Tuple[int, int]) -> int:
    mat2_rank, mat2_dtype = mat2_rank_dtype
    self_rank, self_dtype = self_rank_dtype

    float16_types = [torch.bfloat16, torch.float16]
    if self_dtype in float16_types and mat2_dtype in float16_types and self_dtype != mat2_dtype:
        return torch.float16

    ranks: List[Optional[int]] = [self_rank, mat2_rank]
    dtypes = [self_dtype, mat2_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op(
    output_error_types={torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}))
def aten〇mse_loss〡dtype(self_rank_dtype: Tuple[int, int], target_rank_dtype: Tuple[int, int], reduction: int = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    target_rank, target_dtype = target_rank_dtype
    ranks: List[Optional[int]] = [self_rank, target_rank]
    dtypes = [self_dtype, target_dtype]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    assert not is_integer_dtype(promoted_dtype)
    return promoted_dtype

@check_dtype_function(_check_two_tensor_op())
def aten〇mul〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_tensors_with_the_same_dtype(tensor_shapes=[(3, 4), (4,)]) +
    # Different width
    [Invocation(TensorOfShape(3, 4, dtype=torch.float64),
                TensorOfShape(4, dtype=torch.float32)),
     # Two f16 types
     Invocation(TensorOfShape(3, 4, dtype=torch.float16),
                TensorOfShape(4, dtype=torch.bfloat16)),
     # Different type
     Invocation(TensorOfShape(3, 4, dtype=torch.float32),
                TensorOfShape(4, dtype=torch.int32))])
def aten〇mv〡dtype(self_rank_dtype: Tuple[int, int], vec_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    vec_rank, vec_dtype = vec_rank_dtype
    ranks: List[Optional[int]] = [self_rank, vec_rank]
    dtypes = [self_dtype, vec_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_two_tensor_op())
def aten〇sub〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int], alpha: Union[int, float] = 1) -> int:
    other_rank, other_dtype = other_rank_dtype
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

# TODO: This op has incosistent behavior when run on CPU vs META
# If the following check is made to pass, e2e tests fail
# @check_dtype_function(_check_two_tensor_op(threshold=0))
def aten〇threshold_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], self_rank_dtype: Tuple[int, int], threshold: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    ranks: List[Optional[int]] = [self_rank, grad_output_rank]
    dtypes = [self_dtype, grad_output_dtype]
    return promote_dtypes(ranks, dtypes)

# TODO: This op fails when using meta backend with error:
# Op raised error 'convolution_overrideable not implemented.
# You are likely triggering this with tensor backend other than
# CPU/CUDA/MKLDNN, if this is intended, please use TORCH_LIBRARY_IMPL
# to override this function ' but dtype function did not raise any error.
#
# This is similar to https://github.com/pytorch/pytorch/issues/97481
def aten〇_convolution〡dtype(input_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int], bias_rank_dtype: Optional[Tuple[int, int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, allow_tf32: bool) -> int:
    input_rank, input_dtype = input_rank_dtype
    weight_rank, weight_dtype = weight_rank_dtype
    assert not is_complex_dtype(input_dtype) and input_dtype not in [torch.bool, torch.float16]
    assert not is_complex_dtype(weight_dtype) and weight_dtype not in [torch.bool, torch.float16]
    ranks: List[Optional[int]] = [input_rank, weight_rank]
    dtypes = [input_dtype, weight_dtype]
    return promote_dtypes(ranks, dtypes)

# TODO: This op fails when using meta backend with error:
# Op raised error 'convolution_overrideable not implemented.
# You are likely triggering this with tensor backend other than
# CPU/CUDA/MKLDNN, if this is intended, please use TORCH_LIBRARY_IMPL
# to override this function ' but dtype function did not raise any error.
#
# This is similar to https://github.com/pytorch/pytorch/issues/97481
def aten〇_convolution〇deprecated〡dtype(input_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int], bias_rank_dtype: Optional[Tuple[int, int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool) -> int:
    input_rank, input_dtype = input_rank_dtype
    weight_rank, weight_dtype = weight_rank_dtype
    assert not is_complex_dtype(input_dtype) and input_dtype not in [torch.bool, torch.float16]
    assert not is_complex_dtype(weight_dtype) and weight_dtype not in [torch.bool, torch.float16]
    ranks: List[Optional[int]] = [input_rank, weight_rank]
    dtypes = [input_dtype, weight_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(1, 1, 1, 1), (1, 1, 1, 1)]) +
    [Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.bool), TensorOfShape(1, 1, 1, 1, dtype=torch.float32)),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.bool)),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float16), TensorOfShape(1, 1, 1, 1, dtype=torch.float32)),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float16))
])
def aten〇conv2d〡dtype(input_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int], bias_rank_dtype: Optional[Tuple[int, int]] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), dilation: List[int] = (1, 1), groups: int = 1) -> int:
    input_rank, input_dtype = input_rank_dtype
    return input_dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(1, 1, 1, 1), (1, 1, 1, 1)]) +
    [Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.bool), TensorOfShape(1, 1, 1, 1, dtype=torch.float32)),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.bool)),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float16), TensorOfShape(1, 1, 1, 1, dtype=torch.float32)),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float16))
])
def aten〇conv_transpose2d〇input〡dtype(input_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int], bias_rank_dtype: Optional[Tuple[int, int]] = None, stride: List[int] = (1, 1), padding: List[int] = (0, 0), output_padding: List[int] = (0, 0), groups: int = 1, dilation: List[int] = (1, 1)) -> int:
    input_rank, input_dtype = input_rank_dtype
    return input_dtype

convolution_kwargs = {
    "stride" : [1, 1], "padding" : [0, 0], "dilation" : [1, 1], "transposed" : False, "output_padding" : [0, 0], "groups" : 1}
@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        tensor_shapes=[(1, 1, 1, 1), (1, 1, 1, 1), (1,)], **convolution_kwargs) +
    [Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.bool), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                TensorOfShape(1, dtype=torch.float32), **convolution_kwargs),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.bool),
                TensorOfShape(1, dtype=torch.float32), **convolution_kwargs),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float16), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                TensorOfShape(1, dtype=torch.float32), **convolution_kwargs),
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float16),
                TensorOfShape(1, dtype=torch.float32), **convolution_kwargs)
])
def aten〇convolution〡dtype(input_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int], bias_rank_dtype: Optional[Tuple[int, int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int) -> int:
    input_rank, input_dtype = input_rank_dtype
    return input_dtype

convolution_backward_kwargs = {
    "bias_sizes" : [1], "stride" : [1, 1], "padding" : [0, 0], "dilation" : [1, 1], "transposed" : False, "output_padding" : [0, 0], "groups" : 1, "output_mask" : [True, True, True]}
@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        tensor_shapes=[(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)],
        **convolution_backward_kwargs) +
    # dtype of first three tensors must be float
    [Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.int32), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                TensorOfShape(1, 1, 1, 1, dtype=torch.float32), **convolution_backward_kwargs),
     # dtype of first three tensors must be float
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.int32),
                TensorOfShape(1, 1, 1, 1, dtype=torch.float32), **convolution_backward_kwargs),
     # dtype of first three tensors must be float
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                TensorOfShape(1, 1, 1, 1, dtype=torch.int32), **convolution_backward_kwargs),
     # dtype of first three tensors must be float
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                TensorOfShape(1, 1, 1, 1, dtype=torch.float32), **convolution_backward_kwargs),
     # grad_output, input, and weight must have same dtype
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float64), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                TensorOfShape(1, 1, 1, 1, dtype=torch.float32), **convolution_backward_kwargs),
     # grad_output, input, and weight must have same dtype
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float64),
                TensorOfShape(1, 1, 1, 1, dtype=torch.float32), **convolution_backward_kwargs),
     # grad_output, input, and weight must have same dtype
     Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                TensorOfShape(1, 1, 1, 1, dtype=torch.float64), **convolution_backward_kwargs),
])
def aten〇convolution_backward〡dtype(grad_output_rank_dtype: Tuple[int, int], input_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int], bias_sizes: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]) -> Tuple[int, int, int]:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    input_rank, input_dtype = input_rank_dtype
    weight_rank, weight_dtype = weight_rank_dtype
    return grad_output_dtype, grad_output_dtype, grad_output_dtype

# TODO: Currently unable to test because the op `torch.ops.aten.convolution_backward_overrideable`
# fails to run on the CPU backend. A bug has been filed upstream: https://github.com/pytorch/pytorch/issues/97481
# The dtype function for this op is unlikely to be different from `torch.ops.aten.convolution_backward`
# which is tested.
def aten〇convolution_backward_overrideable〡dtype(grad_output_rank_dtype: Tuple[int, int], input_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]) -> Tuple[int, int, int]:
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    input_rank, input_dtype = input_rank_dtype
    weight_rank, weight_dtype = weight_rank_dtype
    assert grad_output_dtype == input_dtype
    assert weight_dtype == input_dtype
    assert is_float_dtype(input_dtype) and input_dtype != torch.float16
    grad_output_rank, grad_output_dtype = grad_output_rank_dtype
    return grad_output_dtype, grad_output_dtype, grad_output_dtype

# TODO: This op cannot be run on the Meta backend. We should have the test suite default on CPU when the Meta backend does not work.
def aten〇bincount〡dtype(self_rank_dtype: Tuple[int, int], weights_rank_dtype: Optional[Tuple[int, int]] = None, minlength: int = 0) -> int:
    self_rank, self_dtype = self_rank_dtype
    assert is_integer_dtype(self_dtype) and self_dtype != torch.bool
    if weights_rank_dtype is None:
        return torch.int64
    return torch.float64

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(1, 1), (1, 1), (1, 1)]) +
    # Different width
    [Invocation(TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 4, dtype=torch.float64),
                TensorOfShape(4, 3, dtype=torch.float32)),
     # Different type
     Invocation(TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 4, dtype=torch.float32),
                TensorOfShape(4, 3, dtype=torch.int32)),
     Invocation(TensorOfShape(3, 3, dtype=torch.int32),
                TensorOfShape(3, 4, dtype=torch.float32),
                TensorOfShape(4, 3, dtype=torch.float32))])
def aten〇addmm〡dtype(self_rank_dtype: Tuple[int, int], mat1_rank_dtype: Tuple[int, int], mat2_rank_dtype: Tuple[int, int], beta: Union[int, float] = 1, alpha: Union[int, float] = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    mat1_rank, mat1_dtype = mat1_rank_dtype
    mat2_rank, mat2_dtype = mat2_rank_dtype

    ranks: List[Optional[int]] = [self_rank, mat1_rank, mat2_rank]
    dtypes = [self_dtype, mat1_dtype, mat2_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(1, 1), (1, 1), (1, 1)]) +
    # Different width
    [Invocation(TensorOfShape(4, 3, dtype=torch.float32),
                TensorOfShape(4, 3, dtype=torch.float64),
                TensorOfShape(4, 3, dtype=torch.float32)),
     # Different type
     Invocation(TensorOfShape(4, 3, dtype=torch.float32),
                TensorOfShape(4, 3, dtype=torch.float32),
                TensorOfShape(4, 3, dtype=torch.int32)),
     Invocation(TensorOfShape(4, 3, dtype=torch.int32),
                TensorOfShape(4, 3, dtype=torch.float32),
                TensorOfShape(4, 3, dtype=torch.float32))])
def aten〇lerp〇Tensor〡dtype(self_rank_dtype: Tuple[int, int], end_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    end_rank, end_dtype = end_rank_dtype
    weight_rank, weight_dtype = weight_rank_dtype

    ranks: List[Optional[int]] = [self_rank, end_rank, weight_rank]
    dtypes = [self_dtype, end_dtype, weight_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(1, 1), (1, 1), (1, 1)], error_types={torch.bool}) +
    # Different width
    [Invocation(TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 3, dtype=torch.float64),
                TensorOfShape(3, 3, dtype=torch.float32)),
     # Different type
     Invocation(TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 3, dtype=torch.int32)),
     Invocation(TensorOfShape(3, 3, dtype=torch.int32),
                TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 3, dtype=torch.float32))])
def aten〇addcmul〡dtype(self_rank_dtype: Tuple[int, int], tensor1_rank_dtype: Tuple[int, int], tensor2_rank_dtype: Tuple[int, int], value: Union[int, float] = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    tensor1_rank, tensor1_dtype = tensor1_rank_dtype
    tensor2_rank, tensor2_dtype = tensor2_rank_dtype

    assert self_dtype != torch.bool
    assert tensor1_dtype != torch.bool
    assert tensor2_dtype != torch.bool

    ranks: List[Optional[int]] = [self_rank, tensor1_rank, tensor2_rank]
    dtypes = [self_dtype, tensor1_dtype, tensor2_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(1, 1), (1, 1), (1, 1)]) +
    # Different width
    [Invocation(TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 3, dtype=torch.float64),
                TensorOfShape(3, 3, dtype=torch.float32)),
     # Different type
     Invocation(TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 3, dtype=torch.int32)),
     Invocation(TensorOfShape(3, 3, dtype=torch.int32),
                TensorOfShape(3, 3, dtype=torch.float32),
                TensorOfShape(3, 3, dtype=torch.float32))])
def aten〇addcdiv〡dtype(self_rank_dtype: Tuple[int, int], tensor1_rank_dtype: Tuple[int, int], tensor2_rank_dtype: Tuple[int, int], value: Union[int, float] = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    tensor1_rank, tensor1_dtype = tensor1_rank_dtype
    tensor2_rank, tensor2_dtype = tensor2_rank_dtype

    ranks: List[Optional[int]] = [self_rank, tensor1_rank, tensor2_rank]
    dtypes = [self_dtype, tensor1_dtype, tensor2_dtype]
    result = promote_dtypes(ranks, dtypes)
    if is_integer_dtype(result):
        return torch.float32
    return result

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, other=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1.0))
def aten〇add〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float], alpha: Union[int, float] = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(other)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1.0))
def aten〇sub〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float], alpha: Union[int, float] = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(other)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, other=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1.0))
def aten〇mul〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(other)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, other=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1.0))
def aten〇div〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(other)]
    promoted_dtype = promote_dtypes(ranks, dtypes)
    if is_integer_dtype(promoted_dtype):
        return torch.float32
    return promoted_dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1.0))
def aten〇fmod〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(other)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=1) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.complex64, torch.complex128}, other=1.0))
def aten〇floor_divide〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    assert not is_complex_dtype(self_dtype)
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(other)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, exponent=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, exponent=1.0))
def aten〇pow〇Tensor_Scalar〡dtype(self_rank_dtype: Tuple[int, int], exponent: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(exponent)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.bool}, negative_slope=1) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, error_types={torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}, negative_slope=1.0))
def aten〇leaky_relu〡dtype(self_rank_dtype: Tuple[int, int], negative_slope: Union[int, float] = 0.01) -> int:
    self_rank, self_dtype = self_rank_dtype
    assert self_dtype != torch.bool
    ranks: List[Optional[int]] = [self_rank, None]
    negative_slope_dtype = get_dtype_of_scalar(negative_slope)
    if is_float_dtype(negative_slope_dtype):
        assert not is_integer_dtype(self_dtype)
    dtypes = [self_dtype, negative_slope_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, other=1.0))
def aten〇remainder〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(other)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(tensor_shapes=[(1, 1, 1), (1, 1, 1), (1, 1, 1)]) +
    [Invocation(TensorOfShape(
        1, 1, 1, dtype=torch.float64), TensorOfShape(1, 1, 1, dtype=torch.int16), TensorOfShape(1, 1, 1, dtype=torch.int32)),
    Invocation(
        TensorOfShape(1, 1, 1, dtype=torch.float64), TensorOfShape(1, 1, 1, dtype=torch.int64), TensorOfShape(1, 1, 1, dtype=torch.float16)),
    Invocation(
        TensorOfShape(1, 1, 1, dtype=torch.float64), TensorOfShape(1, 1, 1, dtype=torch.float16), TensorOfShape(1, 1, 1, dtype=torch.int64)),
    Invocation(
        TensorOfShape(1, 1, 1, dtype=torch.float64), TensorOfShape(1, 1, 1, dtype=torch.bfloat16), TensorOfShape(1, 1, 1, dtype=torch.float16))])
def aten〇baddbmm〡dtype(self_rank_dtype: Tuple[int, int], batch1_rank_dtype: Tuple[int, int], batch2_rank_dtype: Tuple[int, int], beta: Union[int, float] = 1, alpha: Union[int, float] = 1) -> int:
    batch2_rank, batch2_dtype = batch2_rank_dtype
    return batch2_dtype

@check_dtype_function([
    Invocation(NonZeroDTensorWithDtype(torch.bool), NonZeroDTensorWithDtype(torch.int16), NonZeroDTensorWithDtype(torch.int32)),
    Invocation(NonZeroDTensorWithDtype(torch.bool), NonZeroDTensorWithDtype(torch.int64), NonZeroDTensorWithDtype(torch.float16)),
    Invocation(NonZeroDTensorWithDtype(torch.bool), NonZeroDTensorWithDtype(torch.float16), NonZeroDTensorWithDtype(torch.int64)),
    Invocation(NonZeroDTensorWithDtype(torch.bool), NonZeroDTensorWithDtype(torch.bfloat16), NonZeroDTensorWithDtype(torch.float16))])
def aten〇where〇self〡dtype(condition_rank_dtype: Tuple[int, int], self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    other_rank, other_dtype = other_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function([Invocation(NonZeroDTensorWithDtype(torch.bool), 0, 0),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), 0, 0.0),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), 0.0, 0),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), 0.0, 0.0)])
def aten〇where〇Scalar〡dtype(condition_rank_dtype: Tuple[int, int], self: Union[int, float], other: Union[int, float]) -> int:
    if is_integer_dtype(get_dtype_of_scalar(self)) and is_integer_dtype(get_dtype_of_scalar(other)):
        return torch.int64
    return torch.float32

@check_dtype_function([Invocation(NonZeroDTensorWithDtype(torch.bool), NonZeroDTensorWithDtype(torch.int16), 0),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), NonZeroDTensorWithDtype(torch.int64), 0.0),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), NonZeroDTensorWithDtype(torch.float16), 0),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), NonZeroDTensorWithDtype(torch.float64), 0.0)])
def aten〇where〇ScalarOther〡dtype(condition_rank_dtype: Tuple[int, int], self_rank_dtype: Tuple[int, int], other: Union[int, float]) -> int:
    self_rank, self_dtype = self_rank_dtype
    ranks: List[Optional[int]] = [self_rank, None]
    dtypes = [self_dtype, get_dtype_of_scalar(other)]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function([Invocation(NonZeroDTensorWithDtype(torch.bool), 0, NonZeroDTensorWithDtype(torch.int16)),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), 0.0, NonZeroDTensorWithDtype(torch.int64)),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), 0, NonZeroDTensorWithDtype(torch.float16)),
                       Invocation(NonZeroDTensorWithDtype(torch.bool), 0.0, NonZeroDTensorWithDtype(torch.float64))])
def aten〇where〇ScalarSelf〡dtype(condition_rank_dtype: Tuple[int, int], self: Union[int, float], other_rank_dtype: Tuple[int, int]) -> int:
    other_rank, other_dtype = other_rank_dtype
    ranks: List[Optional[int]] = [None, other_rank]
    dtypes = [get_dtype_of_scalar(self), other_dtype]
    return promote_dtypes(ranks, dtypes)

@check_dtype_function(
    [Invocation(TensorOfShape(2, 3, dtype=torch.float32), TensorOfShape(2, dtype=torch.int64),
                TensorOfShape(3, dtype=torch.float32), reduction=0, ignore_index=0),
     ErrorInvocation(TensorOfShape(2, 3, dtype=torch.float32), TensorOfShape(2, dtype=torch.int32), # target must be int64
                     TensorOfShape(3, dtype=torch.float32), reduction=0, ignore_index=0),
     ErrorInvocation(TensorOfShape(2, 3, dtype=torch.float32), TensorOfShape(2, dtype=torch.float64), # target must be int64
                     TensorOfShape(3, dtype=torch.float32), reduction=0, ignore_index=0),
     Invocation(TensorOfShape(2, 3, dtype=torch.float64), TensorOfShape(2, dtype=torch.int64), # self and weight must have same dtype
                TensorOfShape(3, dtype=torch.float32), reduction=0, ignore_index=0),
     Invocation(TensorOfShape(2, 3, dtype=torch.int32), TensorOfShape(2, dtype=torch.int64), # self and weight must be float
                TensorOfShape(3, dtype=torch.int32), reduction=0, ignore_index=0),
     Invocation(TensorOfShape(2, 3, dtype=torch.complex64), TensorOfShape(2, dtype=torch.int64), # self and weight must be float
                TensorOfShape(3, dtype=torch.complex64), reduction=0, ignore_index=0)])
def aten〇nll_loss_forward〡dtype(self_rank_dtype: Tuple[int, int], target_rank_dtype: Tuple[int, int], weight_rank_dtype: Optional[Tuple[int, int]], reduction: int, ignore_index: int) -> Tuple[int, int]:
    self_rank, self_dtype = self_rank_dtype
    target_rank, target_dtype = target_rank_dtype
    assert target_dtype == torch.int64
    return self_dtype, self_dtype

@check_dtype_function(
    [Invocation(TensorOfShape(2, 3, dtype=torch.float32), [3], TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), eps=0.0),
     Invocation(TensorOfShape(2, 3, dtype=torch.float64), [3], TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), eps=0.0),
     Invocation(TensorOfShape(2, 3, dtype=torch.float32), [3], TensorOfShape(3, dtype=torch.float64),
                TensorOfShape(3, dtype=torch.float32), eps=0.0),
     Invocation(TensorOfShape(2, 3, dtype=torch.float32), [3], TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float64), eps=0.0),
     # Input must be float or complex
     ErrorInvocation(TensorOfShape(2, 3, dtype=torch.int32), [3], TensorOfShape(3, dtype=torch.int32),
                     TensorOfShape(3, dtype=torch.int32), eps=0.0),
     Invocation(TensorOfShape(2, 3, dtype=torch.complex64), [3], TensorOfShape(3, dtype=torch.complex64),
                TensorOfShape(3, dtype=torch.complex64), eps=0.0),
     Invocation(TensorOfShape(2, 3, dtype=torch.complex128), [3], TensorOfShape(3, dtype=torch.complex64),
                TensorOfShape(3, dtype=torch.complex64), eps=0.0),
     ])
def aten〇native_layer_norm〡dtype(input_rank_dtype: Tuple[int, int], normalized_shape: List[int], weight_rank_dtype: Optional[Tuple[int, int]], bias_rank_dtype: Optional[Tuple[int, int]], eps: float) -> Tuple[int, int, int]:
    input_rank, input_dtype = input_rank_dtype
    assert not is_integer_dtype(input_dtype)
    result_dtype = input_dtype
    if input_dtype == torch.complex64:
        result_dtype = torch.float32
    if input_dtype == torch.complex128:
        result_dtype = torch.float64
    return input_dtype, input_dtype, result_dtype

@check_dtype_function(
    [Invocation(TensorOfShape(3, 3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), training=False, momentum=0.0, eps=0.0),
     # Tensors with different dtype
     Invocation(TensorOfShape(3, 3, dtype=torch.float64), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), training=False, momentum=0.0, eps=0.0),
     Invocation(TensorOfShape(3, 3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float64),
                TensorOfShape(3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), training=False, momentum=0.0, eps=0.0),
     Invocation(TensorOfShape(3, 3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float64), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), training=False, momentum=0.0, eps=0.0),
     Invocation(TensorOfShape(3, 3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float64),
                TensorOfShape(3, dtype=torch.float32), training=False, momentum=0.0, eps=0.0),
     Invocation(TensorOfShape(3, 3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float32), TensorOfShape(3, dtype=torch.float32),
                TensorOfShape(3, dtype=torch.float64), training=False, momentum=0.0, eps=0.0),
     # Non-float tensors
     Invocation(TensorOfShape(3, 3, dtype=torch.int32), TensorOfShape(3, dtype=torch.int32),
                TensorOfShape(3, dtype=torch.int32), TensorOfShape(3, dtype=torch.int32),
                TensorOfShape(3, dtype=torch.int32), training=False, momentum=0.0, eps=0.0),
     Invocation(TensorOfShape(3, 3, dtype=torch.complex64), TensorOfShape(3, dtype=torch.complex64),
                TensorOfShape(3, dtype=torch.complex64), TensorOfShape(3, dtype=torch.complex64),
                TensorOfShape(3, dtype=torch.complex64), training=False, momentum=0.0, eps=0.0),
     ])
def aten〇native_batch_norm〡dtype(input_rank_dtype: Tuple[int, int], weight_rank_dtype: Optional[Tuple[int, int]], bias_rank_dtype: Optional[Tuple[int, int]], running_mean_rank_dtype: Optional[Tuple[int, int]], running_var_rank_dtype: Optional[Tuple[int, int]], training: bool, momentum: float, eps: float) -> Tuple[int, int, int]:
    input_rank, input_dtype = input_rank_dtype
    result_dtype = input_dtype
    if is_integer_dtype(input_dtype):
        result_dtype = torch.float32
    return input_dtype, input_dtype, result_dtype

@check_dtype_function([Invocation(end=0, dtype=None), # No floats
                       Invocation(end=0.0, dtype=None), # One float
                       ErrorInvocation(end=0, dtype=torch.complex64), # Dtype specified
                       Invocation(end=0, dtype=torch.float16), # Dtype specified
                       Invocation(end=0, dtype=torch.int16)]) # Dtype specified
def aten〇arange〡dtype(end: Union[int, float], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    if dtype is not None:
        assert not is_complex_dtype(dtype)
        return dtype
    if is_float_dtype(get_dtype_of_scalar(end)):
        return torch.float32
    return torch.int64

@check_dtype_function([Invocation(start=0, end=10, dtype=None), # No floats
                       Invocation(start=0.0, end=10, dtype=None), # One float
                       Invocation(start=0, end=10.0, dtype=None), # One float
                       ErrorInvocation(start=0, end=10, dtype=torch.complex64), # Dtype specified
                       Invocation(start=0, end=10, dtype=torch.float16), # Dtype specified
                       Invocation(start=0, end=10, dtype=torch.int16)]) # Dtype specified
def aten〇arange〇start〡dtype(start: Union[int, float], end: Union[int, float], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    if dtype is not None:
        assert not is_complex_dtype(dtype)
        return dtype
    if is_float_dtype(get_dtype_of_scalar(start)) or \
       is_float_dtype(get_dtype_of_scalar(end)):
        return torch.float32
    return torch.int64

@check_dtype_function([Invocation(start=0, end=10, step=1, dtype=None), # No floats
                       Invocation(start=0.0, end=10, step=1, dtype=None), # One float
                       Invocation(start=0, end=10.0, step=1, dtype=None), # One float
                       Invocation(start=0, end=10, step=1.0, dtype=None), # One float
                       ErrorInvocation(start=0, end=10, step=1, dtype=torch.complex64), # Dtype specified
                       Invocation(start=0, end=10, step=1, dtype=torch.float16), # Dtype specified
                       Invocation(start=0, end=10, step=1, dtype=torch.int16)]) # Dtype specified
def aten〇arange〇start_step〡dtype(start: Union[int, float], end: Union[int, float], step: Union[int, float] = 1, dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    if dtype is not None:
        assert not is_complex_dtype(dtype)
        return dtype
    if is_float_dtype(get_dtype_of_scalar(start)) or \
       is_float_dtype(get_dtype_of_scalar(end)) or \
       is_float_dtype(get_dtype_of_scalar(step)):
        return torch.float32
    return torch.int64

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.float32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.complex64))
def aten〇sum〡dtype(self_rank_dtype: Tuple[int, int], dtype: Optional[int] = None) -> int:
    if dtype is not None:
        return dtype
    self_rank, self_dtype = self_rank_dtype
    if is_integer_dtype(self_dtype):
        return torch.int64
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=None) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dim=None, dtype=torch.float32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dim=None, dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dim=None, dtype=torch.complex64))
def aten〇sum〇dim_IntList〡dtype(self_rank_dtype: Tuple[int, int], dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None) -> int:
    return aten〇sum〡dtype(self_rank_dtype, dtype)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1,
        error_types={torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}, dim=None) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, dim=None, dtype=torch.float32) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, dim=None, dtype=torch.complex64) +
    [ErrorInvocation(NonZeroDTensorWithDtype(torch.float32), dim=None, dtype=torch.int32)])
def aten〇mean〇dim〡dtype(self_rank_dtype: Tuple[int, int], dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    result = aten〇sum〡dtype(self_rank_dtype, dtype)
    assert not is_integer_dtype(result)
    return result

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇argmax〡dtype(self_rank_dtype: Tuple[int, int], dim: Optional[int] = None, keepdim: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    return torch.int64

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=0))
def aten〇any〇dim〡dtype(self_rank_dtype: Tuple[int, int], dim: int, keepdim: bool = False) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.uint8:
        return self_dtype
    return torch.bool

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇max〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇amax〡dtype(self_rank_dtype: Tuple[int, int], dim: List[int] = (), keepdim: bool = False) -> int:
    return aten〇max〡dtype(self_rank_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=0))
def aten〇max〇dim〡dtype(self_rank_dtype: Tuple[int, int], dim: int, keepdim: bool = False) -> Tuple[int, int]:
    return aten〇max〡dtype(self_rank_dtype), torch.int64

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1,
        error_types={torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.float32) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.complex64) +
    [ErrorInvocation(NonZeroDTensorWithDtype(torch.float32), dtype=torch.int32)])
def aten〇mean〡dtype(self_rank_dtype: Tuple[int, int], dtype: Optional[int] = None) -> int:
    return aten〇mean〇dim〡dtype(self_rank_dtype, dim=None, dtype=dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇std〡dtype(self_rank_dtype: Tuple[int, int], unbiased: bool = True) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.complex64:
        return torch.float32
    if self_dtype == torch.complex128:
        return torch.float64
    return self_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=None))
def aten〇std〇dim〡dtype(self_rank_dtype: Tuple[int, int], dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False) -> int:
    return aten〇std〡dtype(self_rank_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇std〇correction〡dtype(self_rank_dtype: Tuple[int, int], dim: Optional[List[int]] = None, correction: Optional[Union[int, float]] = None, keepdim: bool = False) -> int:
    return aten〇std〡dtype(self_rank_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇var〡dtype(self_rank_dtype: Tuple[int, int], unbiased: bool = True) -> int:
    return aten〇std〡dtype(self_rank_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dim=None))
def aten〇var〇dim〡dtype(self_rank_dtype: Tuple[int, int], dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False) -> int:
    return aten〇std〡dtype(self_rank_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1))
def aten〇var〇correction〡dtype(self_rank_dtype: Tuple[int, int], dim: Optional[List[int]] = None, correction: Optional[Union[int, float]] = None, keepdim: bool = False) -> int:
    return aten〇std〡dtype(self_rank_dtype)

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, dims=[], correction=0.0))
def prims〇var〡dtype(inp_rank_dtype: Tuple[int, int], dims: Optional[List[int]], correction: float, output_dtype: Optional[int] = None) -> int:
    return aten〇std〡dtype(inp_rank_dtype)

@check_dtype_function(
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1,
        error_types={torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}) +
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1,
        error_types={torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.complex64, torch.complex128}, dtype=torch.float64) +
    _check_tensors_with_the_same_dtype(
        num_of_tensors=1,
        error_types={torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.float16, torch.float32, torch.float64}, dtype=torch.complex128) +
    [ErrorInvocation(NonZeroDTensorWithDtype(torch.float32), dtype=torch.int32)])
def aten〇linalg_vector_norm〡dtype(self_rank_dtype: Tuple[int, int], ord: Union[int, float] = 2, dim: Optional[List[int]] = None, keepdim: bool = False, dtype: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    assert not is_integer_dtype(self_dtype)
    if dtype is not None:
        assert not is_integer_dtype(dtype)
        if is_complex_dtype(self_dtype):
            assert is_complex_dtype(dtype)
            return aten〇std〡dtype((self_rank, dtype))
        assert not is_complex_dtype(dtype)
        return dtype
    return aten〇std〡dtype(self_rank_dtype)

@check_dtype_function([Invocation(0.0),
                       Invocation(0.0, dtype=torch.int32),
                       Invocation(0.0, dtype=torch.float16),
                       Invocation(0.0, dtype=torch.complex64)])
def aten〇tensor〇float〡dtype(t: float, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> int:
    if dtype is None:
        return torch.float32
    return dtype

@check_dtype_function([Invocation(0),
                       Invocation(0, dtype=torch.int32),
                       Invocation(0, dtype=torch.float16),
                       Invocation(0, dtype=torch.complex64)])
def aten〇tensor〇int〡dtype(t: int, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> int:
    if dtype is None:
        return torch.int64
    return dtype

@check_dtype_function([Invocation(True),
                       Invocation(True, dtype=torch.int32),
                       Invocation(True, dtype=torch.float16),
                       Invocation(True, dtype=torch.complex64)])
def aten〇tensor〇bool〡dtype(t: bool, dtype: Optional[int] = None, device: Optional[device] = None, requires_grad: bool = False) -> int:
    if dtype is None:
        return torch.bool
    return dtype

@check_dtype_function([Invocation([1]),
                       Invocation([1], dtype=torch.int32),
                       Invocation([1], dtype=torch.float16),
                       Invocation([1], dtype=torch.complex64)])
def aten〇zeros〡dtype(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    return torch.float32 if dtype is None else dtype

@check_dtype_function([Invocation([1]),
                       Invocation([1], dtype=torch.int32),
                       Invocation([1], dtype=torch.float16),
                       Invocation([1], dtype=torch.complex64)])
def aten〇ones〡dtype(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    return torch.float32 if dtype is None else dtype

@check_dtype_function([Invocation([1]),
                       Invocation([1], dtype=torch.int32),
                       Invocation([1], dtype=torch.float16),
                       Invocation([1], dtype=torch.complex64)])
def aten〇empty〇memory_format〡dtype(size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> int:
    return torch.float32 if dtype is None else dtype

@check_dtype_function([Invocation([1], 0.0),
                       Invocation([1], 0),
                       Invocation([1], 0.0, dtype=torch.int32),
                       Invocation([1], 0.0, dtype=torch.float16),
                       Invocation([1], 0.0, dtype=torch.complex64)])
def aten〇full〡dtype(size: List[int], fill_value: Union[int, float], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    if dtype is not None:
        return dtype
    fill_value_dtype = get_dtype_of_scalar(fill_value)
    if is_float_dtype(fill_value_dtype):
        return torch.float32
    return fill_value_dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.float16) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.complex64))
def aten〇zeros_like〡dtype(self_rank_dtype: Tuple[int, int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.float16) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.complex64))
def aten〇ones_like〡dtype(self_rank_dtype: Tuple[int, int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.float16) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.complex64))
def aten〇empty_like〡dtype(self_rank_dtype: Tuple[int, int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, fill_value=0.0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, fill_value=0) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, fill_value=0.0, dtype=torch.float16) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, fill_value=0.0, dtype=torch.int32) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, fill_value=0.0, dtype=torch.complex64))
def aten〇full_like〡dtype(self_rank_dtype: Tuple[int, int], fill_value: Union[int, float], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1]) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.float16) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.complex64))
def aten〇new_zeros〡dtype(self_rank_dtype: Tuple[int, int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1]) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.float16) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.complex64))
def aten〇new_ones〡dtype(self_rank_dtype: Tuple[int, int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1]) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.float16) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.int32) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], dtype=torch.complex64))
def aten〇new_empty〡dtype(self_rank_dtype: Tuple[int, int], size: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(
    _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], stride=[1]) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], stride=[1], dtype=torch.float16) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], stride=[1], dtype=torch.int32) +
    _check_tensors_with_the_same_dtype(num_of_tensors=1, size=[1], stride=[1], dtype=torch.complex64))
def aten〇new_empty_strided〡dtype(self_rank_dtype: Tuple[int, int], size: List[int], stride: List[int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.float16) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.complex64))
def aten〇rand_like〡dtype(self_rank_dtype: Tuple[int, int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.float16) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.complex64))
def aten〇randn_like〡dtype(self_rank_dtype: Tuple[int, int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

@check_dtype_function(_check_tensors_with_the_same_dtype(num_of_tensors=1) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.float16) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.int32) +
                      _check_tensors_with_the_same_dtype(num_of_tensors=1, dtype=torch.complex64))
def aten〇_to_copy〡dtype(self_rank_dtype: Tuple[int, int], dtype: Optional[int] = None, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None, non_blocking: bool = False, memory_format: Optional[int] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    return self_dtype if dtype is None else dtype

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

