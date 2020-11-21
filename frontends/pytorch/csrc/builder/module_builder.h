//===- module_builder.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_BUILDER_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_BUILDER_H

#include "../pybind.h"

#include "acap_dispatch.h"

#include "mlir-c/IR.h"
#include "llvm/ADT/SmallVector.h"

#include <ATen/Tensor.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

/// Main entry-point for constructing an MLIR module from some combination
/// of PyTorch programs/execution.
class ModuleBuilder {
public:
  ModuleBuilder(pybind11::object contextObj);

  /// Creates Python bindings for the class.
  static void bind(pybind11::module &m);

  pybind11::object getContextObj() { return contextObj; }
  pybind11::object getModuleObj() { return moduleObj; }

  // Starts a device-capture based function.
  std::shared_ptr<AcapController>
  startCaptureFunction(std::string &name, std::vector<at::Tensor> args);

  // Imports a traced function. Note that the python type
  // torch.jit.ScriptFunction is the C++ type torch::jit::StrongFunctionPtr.
  // Just a bit of naming cruft.
  void importFunction(torch::jit::StrongFunctionPtr function);

private:
  FuncBuilder::Inserter createInserter();
  MlirBlock getBodyBlock();

  // Capture references to the python-owned context and module. Ownership
  // is delegated to python for these, and the C-API types are extracted via
  // the capsule API.
  pybind11::object contextObj;
  MlirContext context;
  MlirModule module;
  pybind11::object moduleObj;
  MlirOperation terminator;
  MlirLocation unknownLoc;

  TypeMapper typeMapper;
};

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_MODULE_BUILDER_H
