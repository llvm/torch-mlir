//===- tensor_aten_ops.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/lazy_tensor_core/csrc/tensor_aten_ops.h
//===----------------------------------------------------------------------===//

#pragma once

#include <torch/csrc/lazy/core/tensor.h>

namespace torch_lazy_tensors {
namespace lazy_tensor_aten_ops {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
// Takes a slice from the input as R1 at the specified offset and reshapes it
// into the provided size.
torch::lazy::LazyTensorPtr as_strided(
    const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size,
    std::vector<int64_t> stride, c10::optional<int64_t> storage_offset);

// In-place version of the method above.
void as_strided_(
    torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size,
    std::vector<int64_t> stride, c10::optional<int64_t> storage_offset);

torch::lazy::LazyTensorPtr
expand(const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size);

// Fills the input with the given value.
void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value);

// Returns a new tensor that is a narrowed view of the input in the given
// dimension.
torch::lazy::LazyTensorPtr narrow(
    const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
    int64_t length);

// Permute the dimensions of this tensor according to the given permutation.
torch::lazy::LazyTensorPtr
permute(const torch::lazy::LazyTensorPtr& input, c10::ArrayRef<int64_t> dims);

// Repeats the input tensor along each dimension by the given number of
// repeats.
torch::lazy::LazyTensorPtr
repeat(const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> repeats);

void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src);

torch::lazy::LazyTensorPtr slice(
    const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
    int64_t end, int64_t step);

std::tuple<
    torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr,
    torch::lazy::LazyTensorPtr>
svd(const torch::lazy::LazyTensorPtr& input, bool some, bool compute_uv);

// Swap given dimensions of the input.
torch::lazy::LazyTensorPtr
transpose(const torch::lazy::LazyTensorPtr& input, int64_t dim0, int64_t dim1);

// In-place version of the method above.
void transpose_(torch::lazy::LazyTensorPtr& input, int64_t dim0, int64_t dim1);

// Like reshape, but it returns a view into the original tensor.
torch::lazy::LazyTensorPtr view(
    const torch::lazy::LazyTensorPtr& input,
    c10::ArrayRef<int64_t> output_size);

} // namespace lazy_tensor_aten_ops
} // namespace torch_lazy_tensors
