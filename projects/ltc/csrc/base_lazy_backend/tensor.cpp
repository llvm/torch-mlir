//===- tensor.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <ATen/FunctionalTensorWrapper.h>

#include "tensor.h"

namespace torch {
namespace lazy {

at::Tensor
CreateFunctionalizedAtenFromLtcTensor(const LazyTensorPtr &ltc_tensor) {
  at::Tensor tensor = CreateAtenFromLtcTensor(ltc_tensor);
  if (!c10::impl::tls_is_dispatch_key_excluded(
          c10::DispatchKey::Functionalize) &&
      !at::functionalization::impl::isFunctionalTensor(tensor)) {
    return at::functionalization::impl::to_functional_tensor(tensor);
  }
  return tensor;
}

} // namespace lazy
} // namespace torch
