//===- get_registered_ops.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE.pytorch for license information.
//
//===----------------------------------------------------------------------===//
//
// Listing of the JIT operator registry, for use in generating the `torch`
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_GETREGISTEREDOPS_H
#define TORCHMLIRJITIRIMPORTER_CSRC_GETREGISTEREDOPS_H

#include "pybind.h"

namespace torch_mlir {

void initGetRegisteredOpsBindings(py::module &m);

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_GETREGISTEREDOPS_H
