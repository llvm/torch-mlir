//===- init_python_bindings.h -----------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef INIT_PYTHON_BINDINGS_H
#define INIT_PYTHON_BINDINGS_H

#include "pybind.h"

namespace torch_mlir {

// Perform top-level initialization for the module.
void InitBindings(pybind11::module &m);

// Adds bindings related to building modules.
void InitBuilderBindings(pybind11::module &m);

} // namespace torch_mlir

#endif
