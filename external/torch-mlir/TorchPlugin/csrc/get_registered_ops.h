//===- get_registered_ops.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// Listing of the JIT operator registry, for use in generating the `torch`
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRPLUGIN_CSRC_GETREGISTEREDOPS_H
#define TORCHMLIRPLUGIN_CSRC_GETREGISTEREDOPS_H

#include "pybind.h"

namespace torch_mlir {

void initGetRegisteredOpsBindings(py::module &m);

} // namespace torch_mlir

#endif // TORCHMLIRPLUGIN_CSRC_GETREGISTEREDOPS_H
