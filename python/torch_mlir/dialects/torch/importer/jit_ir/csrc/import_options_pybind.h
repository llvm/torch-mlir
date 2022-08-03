//===- import_options_pybind.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_IMPORT_OPTIONS_PYBIND_H
#define TORCHMLIRJITIRIMPORTER_CSRC_IMPORT_OPTIONS_PYBIND_H

#include <torch/csrc/utils/pybind.h>

namespace torch_mlir {
void initImportOptionsBindings(pybind11::module &m);
} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_IMPORT_OPTIONS_PYBIND_H
