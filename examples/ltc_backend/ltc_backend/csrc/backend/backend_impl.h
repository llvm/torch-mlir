//===- backend_impl.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <torch/csrc/lazy/backend/backend_interface.h>

namespace at {
// This function is defined in the codegenerated RegisterLazy.cpp file.
TORCH_API void RegisterTorchMlirLazyNativeFunctions();
} // namespace at

namespace torch {
namespace lazy {

torch::lazy::BackendImplInterface *GetExampleMlirBackendImpl();

void InitExampleMlirBackend();

} // namespace lazy
} // namespace torch
