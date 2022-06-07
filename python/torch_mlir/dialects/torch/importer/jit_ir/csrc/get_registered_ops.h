//===- get_registered_ops.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// Listing of the JIT operator registry, for use in generating the `torch`
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_GETREGISTEREDOPS_H
#define TORCHMLIRJITIRIMPORTER_CSRC_GETREGISTEREDOPS_H

#include <torch/csrc/utils/pybind.h>

namespace torch_mlir {

void initGetRegisteredOpsBindings(py::module &m);

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_GETREGISTEREDOPS_H
