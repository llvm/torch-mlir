//===- init_python_bindings.h -----------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef INIT_PYTHON_BINDINGS_H
#define INIT_PYTHON_BINDINGS_H

#include "torch/csrc/jit/pybind.h"

namespace torch_mlir {

// Initialize bindings for torch_mlir functions
void InitBindings(py::module m);

} // namespace torch_mlir

#endif
