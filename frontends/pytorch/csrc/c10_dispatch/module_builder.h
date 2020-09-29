//===- module_builder.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_MODULE_BUILDER_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_MODULE_BUILDER_H

#include <pybind11/pybind11.h>

#include "acap_dispatch.h"

#include "mlir-c/IR.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_mlir {

/// Main entry-point for constructing an MLIR module from some combination
/// of PyTorch programs/execution.
class ModuleBuilder {
public:
  ModuleBuilder();
  ~ModuleBuilder();

  /// Creates Python bindings for the class.
  static void bind(pybind11::module &m);

  // TODO: Remove this once the MLIR Python objects are exposed directly.
  pybind11::str getAsm();

  // Starts a device-capture based function.
  // TODO: Add inputs.
  std::shared_ptr<AcapController> startCaptureFunction(std::string &name);

private:
  // Creates a new top-level function and returns its operation.
  MlirOperation createFunction(std::string &name,
                               llvm::SmallVectorImpl<MlirType> &inputTypes,
                               llvm::SmallVectorImpl<MlirType> &resultTypes);

  MlirContext context;
  MlirLocation unknownLoc;
  MlirModule module;
};

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_MODULE_BUILDER_H
