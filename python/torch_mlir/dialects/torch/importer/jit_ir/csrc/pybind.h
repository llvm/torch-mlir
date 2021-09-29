//===- module_builder.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// Includes Torch-specific pybind and associated helpers.
// Depend on this for access to all Torch types (versus depending on pybind11
// directly).
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_PYBIND_H
#define TORCHMLIRJITIRIMPORTER_CSRC_PYBIND_H

#include <torch/csrc/utils/pybind.h>

namespace torch_mlir {

/// Thrown on failure when details are in MLIR emitted diagnostics.
class mlir_diagnostic_emitted : public std::runtime_error {
public:
  mlir_diagnostic_emitted(const char *what) : std::runtime_error(what) {}
  mlir_diagnostic_emitted() : std::runtime_error("see diagnostics") {}
};

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_PYBIND_H
