//===- tensor.h -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <torch/csrc/lazy/core/tensor.h>

namespace torch {
namespace lazy {

// Ops like torch.ones/zeros etc. which produce new tensor as an output
// should have explicit tensor functinoalization. Otherwise we can get
// unfanctionalized primitives or in the worst case if we apply inplace
// operations to unfunctionalized tensor it won't be captured in LTC graph.
TORCH_API at::Tensor
CreateFunctionalizedAtenFromLtcTensor(const LazyTensorPtr &ltc_tensor);

} // namespace lazy
} // namespace torch
