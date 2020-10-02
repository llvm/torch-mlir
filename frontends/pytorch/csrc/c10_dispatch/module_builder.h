//===- module_builder.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_MODULE_BUILDER_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_MODULE_BUILDER_H

// TODO: Remove this dep once the getAsm() method is removed.
#include "../pybind.h"

#include "acap_dispatch.h"

#include "mlir-c/IR.h"
#include "llvm/ADT/SmallVector.h"

#include <ATen/Tensor.h>

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
  std::shared_ptr<AcapController>
  startCaptureFunction(std::string &name, std::vector<at::Tensor> args);

private:
  MlirBlock getBodyBlock();

  MlirContext context;
  MlirLocation unknownLoc;
  MlirModule module;
  MlirOperation terminator;

  TypeMapper typeMapper;
};

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_MODULE_BUILDER_H
