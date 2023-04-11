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
from .library_generator import generate_library, not_present_in_registry, promote_dtypes, get_dtype_of_scalar

# TODO: upstream this
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

@check_dtype_function([
    Invocation(NonZeroDTensorWithDtype(torch.float32)),
    Invocation(NonZeroDTensorWithDtype(torch.float64)),
    Invocation(NonZeroDTensorWithDtype(torch.bfloat16)),
    Invocation(NonZeroDTensorWithDtype(torch.int64)),
    Invocation(NonZeroDTensorWithDtype(torch.int32)),
    Invocation(NonZeroDTensorWithDtype(torch.bool)),
    Invocation(ZeroDTensorWithDtype(torch.float32)),
    Invocation(ZeroDTensorWithDtype(torch.float64)),
    Invocation(ZeroDTensorWithDtype(torch.bfloat16)),
    Invocation(ZeroDTensorWithDtype(torch.int64)),
    Invocation(ZeroDTensorWithDtype(torch.int32)),
    Invocation(ZeroDTensorWithDtype(torch.bool)),
])
def aten〇expm1〡dtype(self_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.float64 or self_dtype == torch.bfloat16 or self_dtype == torch.float16:
        return self_dtype
    else:
        return torch.float32

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

@check_dtype_function([
    Invocation(NonZeroDTensorWithDtype(torch.float32), other=0),
    Invocation(NonZeroDTensorWithDtype(torch.int64), other=0.0),
    Invocation(NonZeroDTensorWithDtype(torch.float16), other=0.0),
    Invocation(ZeroDTensorWithDtype(torch.float32), other=0),
    Invocation(ZeroDTensorWithDtype(torch.int64), other=0.0),
    Invocation(ZeroDTensorWithDtype(torch.float16), other=0.0)
])
def aten〇rsub〇Scalar〡dtype(self_rank_dtype: Tuple[int, int], other: Union[int, float], alpha: Union[int, float] = 1) -> int:
    self_rank, self_dtype = self_rank_dtype
    return promote_dtypes([self_rank, None], [self_dtype, get_dtype_of_scalar(other)])

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
    return upstream_shape_functions.argmax(self, dim, keepdim)

# TODO: The result shape when num_classes=-1 depends on the runtime values of the input tensor,
# making it impossible to add support for it using the current design of the shape library.
def aten〇one_hot〡shape(self: List[int], num_classes: int = -1) -> List[int]:
    assert num_classes != -1, "getting num_classes from tensor contents is not supported"
    return self + [num_classes]

def aten〇any〇dim〡shape(self: List[int], dim: int, keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.argmax(self, dim, keepdim)

def aten〇max〇dim〡shape(self: List[int], dim: int, keepdim: bool = False) -> Tuple[List[int], List[int]]:
    reduced_shape = upstream_shape_functions.argmax(self, dim, keepdim)
    return reduced_shape, reduced_shape

def aten〇amax〡shape(self: List[int], dim: List[int] = (), keepdim: bool = False) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, None)

def aten〇mean〇dim〡shape(self: List[int], dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, dtype)

def aten〇sum〇dim_IntList〡shape(self: List[int], dim: Optional[List[int]], keepdim: bool = False, dtype: Optional[int] = None) -> List[int]:
    return upstream_shape_functions.sum_mean_dim(self, dim, keepdim, dtype)

def aten〇permute〡shape(self: List[int], dims: List[int]) -> List[int]:
    return upstream_shape_functions.permute(self, dims)

def aten〇movedim〇int〡shape(self: List[int], source: int, destination: int) -> List[int]:
    return upstream_shape_functions.movedim(self, [source], [destination])

def aten〇movedim〇int〡dtype(self_rank_dtype: Tuple[int, int], source: int, destination: int) -> int:
    _, self_dtype = self_rank_dtype
    return self_dtype

def aten〇transpose〇int〡shape(self: List[int], dim0: int, dim1: int) -> List[int]:
    return upstream_shape_functions.transpose(self, dim0, dim1)

def aten〇t〡shape(self: List[int]) -> List[int]:
    return upstream_shape_functions.transpose(self, 0, 1)

# TODO: upstream this
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
    return upstream_shape_functions.bmm(self, mat2)

def aten〇baddbmm〡shape(self: List[int], batch1: List[int], batch2: List[int], beta: float = 1, alpha: float = 1) -> List[int]:
    return upstream_shape_functions.bmm(batch1, batch2)

def aten〇embedding〡shape(weight: List[int], indices: List[int], padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False) -> List[int]:
    return upstream_shape_functions.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)

# TODO: upstream this
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

def aten〇randint〡shape(high: int, size: List[int], dtype: Optional[int] = 4, layout: Optional[int] = None, device: Optional[device] = None, pin_memory: Optional[bool] = None) -> List[int]:
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

@check_dtype_function([
    Invocation(NonZeroDTensorWithDtype(torch.float32), NonZeroDTensorWithDtype(torch.float32)),
    Invocation(ZeroDTensorWithDtype(torch.float64), NonZeroDTensorWithDtype(torch.float32)),
    Invocation(ZeroDTensorWithDtype(torch.float32), NonZeroDTensorWithDtype(torch.float64)),
    Invocation(NonZeroDTensorWithDtype(torch.float32), NonZeroDTensorWithDtype(torch.int32)),
])
def aten〇floor_divide〡dtype(self_rank_dtype: Tuple[int, int], other_rank_dtype: Tuple[int, int]) -> int:
    self_rank, self_dtype = self_rank_dtype
    other_rank, other_dtype = other_rank_dtype
    ranks: List[Optional[int]] = [self_rank, other_rank]
    dtypes = [self_dtype, other_dtype]
    return promote_dtypes(ranks, dtypes)

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

def prims〇squeeze〡shape(a: List[int], dimensions: List[int]) -> List[int]:
    return upstream_shape_functions.squeeze_dims(a, dimensions)

def prims〇view_of〡shape(a: List[int]) -> List[int]:
    return a

def prims〇view_of〡dtype(a_rank_dtype: Tuple[int, int]) -> int:
    _, a_dtype = a_rank_dtype
    return a_dtype

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
    return upstream_shape_functions.topk(self, k, dim)

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

_convolution_deprecated_kwargs = {
    "stride" : [1, 1], "padding" : [0, 0], "dilation" : [1, 1], "transposed" : False, "output_padding" : [0, 0],
    "groups" : 1, "benchmark" : False, "deterministic" : False, "cudnn_enabled" : False}
@check_dtype_function(
    [Invocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float32), # Same type
                TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.int32), TensorOfShape(1, 1, 1, 1, dtype=torch.float32), # Different type
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.bfloat16), TensorOfShape(1, 1, 1, 1, dtype=torch.float32), # Different width
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.bfloat16), TensorOfShape(1, 1, 1, 1, dtype=torch.int32), # Different type and width
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.complex64), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.complex128),
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.bool), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.bool),
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float16), TensorOfShape(1, 1, 1, 1, dtype=torch.float32),
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs),
     ErrorInvocation(TensorOfShape(1, 1, 1, 1, dtype=torch.float32), TensorOfShape(1, 1, 1, 1, dtype=torch.float16),
                     TensorOfShape(1, dtype=torch.float32), **_convolution_deprecated_kwargs)
])
def aten〇_convolution〇deprecated〡dtype(input_rank_dtype: Tuple[int, int], weight_rank_dtype: Tuple[int, int], bias_rank_dtype: Optional[Tuple[int, int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool) -> int:
    input_rank, input_dtype = input_rank_dtype
    weight_rank, weight_dtype = weight_rank_dtype
    assert input_dtype == weight_dtype
    assert input_dtype not in [torch.bool, torch.float16, torch.complex64, torch.complex128]
    ranks: List[Optional[int]] = [input_rank, weight_rank]
    dtypes = [input_dtype, weight_dtype]
    return promote_dtypes(ranks, dtypes)

def aten〇flip〡shape(self: List[int], dims: List[int]) -> List[int]:
    return self

def aten〇convolution_backward〡shape(grad_output: List[int], input: List[int], weight: List[int], bias_sizes: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool]) -> Tuple[List[int], List[int], List[int]]:
    return upstream_shape_functions.conv_backwards(grad_output, input, weight, bias_sizes)

def aten〇batch_norm〡shape(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool, momentum: float, eps: float, cudnn_enabled: bool) -> List[int]:
    return upstream_shape_functions.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)

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
    return upstream_shape_functions.nll_loss_forward(self, target, weight, reduction)

def aten〇nll_loss_backward〡shape(grad_output: List[int], self: List[int], target: List[int], weight: Optional[List[int]], reduction: int, ignore_index: int, total_weight: List[int]) -> List[int]:
    return upstream_shape_functions.unary(self)

# TODO: upstream this
def aten〇mse_loss〡shape(self: List[int], target: List[int], reduction: int = 1) -> List[int]:
    if reduction == 0:
        return upstream_shape_functions.unary(self)
    return []

@check_shape_function([
    Invocation(TensorOfShape(2, 5, 2, 2, 3), [2, 2, 3], None, None, 1e-6), # Basic case.
])
def aten〇native_layer_norm〡shape(input: List[int], normalized_shape: List[int], weight: Optional[List[int]], bias: Optional[List[int]], eps: float) -> Tuple[List[int], List[int], List[int]]:
    return upstream_shape_functions.native_layer_norm(input, normalized_shape)

@check_shape_function([
    Invocation(TensorOfShape(2, 3), None, None, None, None, True, 1e-4, 1e-6), # Training basic case.
    Invocation(TensorOfShape(2, 3), None, None, TensorOfShape(3), TensorOfShape(3), False, 1e-4, 1e-6), # Inference basic case.
    Invocation(TensorOfShape(2, 3, 4, 5, 6), None, None, None, None, True, 1e-4, 1e-6), # Training high-D case.
    Invocation(TensorOfShape(2, 3, 4, 5, 6), None, None, TensorOfShape(3), TensorOfShape(3), False, 1e-4, 1e-6), # Inference high-D case.
    ErrorInvocation(TensorOfShape(2), None, None, None, None, True, 1e-4, 1e-6) # Dimensionality too low.
])
def aten〇native_batch_norm〡shape(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool, momentum: float, eps: float) -> Tuple[List[int], List[int], List[int]]:
    return upstream_shape_functions.native_batch_norm(input, weight, bias, running_mean, running_var, training)

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

# TODO: upstream this
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

@check_dtype_function([
    Invocation(NonZeroDTensorWithDtype(torch.complex64)),
    Invocation(NonZeroDTensorWithDtype(torch.complex128)),
    Invocation(NonZeroDTensorWithDtype(torch.float)),
    Invocation(NonZeroDTensorWithDtype(torch.double)),
    Invocation(NonZeroDTensorWithDtype(torch.bool)),
    Invocation(NonZeroDTensorWithDtype(torch.uint8)),
    Invocation(NonZeroDTensorWithDtype(torch.int8)),
    Invocation(NonZeroDTensorWithDtype(torch.int16)),
    Invocation(NonZeroDTensorWithDtype(torch.int32)),
    Invocation(NonZeroDTensorWithDtype(torch.int64)),
    ErrorInvocation(NonZeroDTensorWithDtype(torch.float16)),
    ErrorInvocation(NonZeroDTensorWithDtype(torch.bfloat16)),
])
def aten〇fft_fft〡dtype(self_rank_dtype: Tuple[int, int], n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> int:
    self_rank, self_dtype = self_rank_dtype
    if self_dtype == torch.complex64 or self_dtype == torch.complex128:
        return self_dtype
    elif self_dtype == torch.float:
        return torch.complex64
    elif self_dtype == torch.double:
        return torch.complex128
    elif self_dtype == torch.bool or self_dtype == torch.uint8 or \
         self_dtype == torch.int8 or self_dtype == torch.int16 or \
         self_dtype == torch.int32 or self_dtype == torch.int64:
        return torch.complex64
    else:
        assert False, "Unsupported dtype"

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

