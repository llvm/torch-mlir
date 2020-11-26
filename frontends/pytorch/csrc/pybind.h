//===- module_builder.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//
// Includes Torch-specific pybind and associated helpers.
// Depend on this for access to all Torch types (versus depending on pybind11
// directly).
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_PYBIND_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_PYBIND_H

#include <torch/csrc/utils/pybind.h>

namespace torch_mlir {

namespace py = pybind11;

/// Thrown on failure when details are in MLIR emitted diagnostics.
class mlir_diagnostic_emitted : public std::runtime_error {
public:
  mlir_diagnostic_emitted(const char *what) : std::runtime_error(what) {}
  mlir_diagnostic_emitted() : std::runtime_error("see diagnostics") {}
};

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_PYBIND_H
