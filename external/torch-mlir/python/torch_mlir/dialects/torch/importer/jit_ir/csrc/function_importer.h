//===- function_importer.h --------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE.pytorch for license information.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_FUNCTION_IMPORTER_H
#define TORCHMLIRJITIRIMPORTER_CSRC_FUNCTION_IMPORTER_H

#include <memory>

#include "node_importer.h"
#include "pybind.h"

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

/// Main entry-point for importing torch::jit::Function instances.
///
/// This code doesn't handle importing of torch::jit::Module's. See
/// IValueImporter for that.
///
/// A torch::jit::Function holds a c10::FunctionSchema along with a
/// c10::QualifiedName and a torch::jit::Graph.
///
/// The torch::jit::Graph is a combination of an MLIR context, function, and
/// builder. See NodeImporter for importing of the core IR Node/Block
/// structure that is analogous to MLIR's Operation/Region/Block core structure.
///
/// If the `getArgAttribute` function is present, then it will be called for
/// each function argument index `i` and should return an MlirAttribute which
/// will be attached as an argument attribute to the func op's argument. If a
/// null MlirAttribute is returned, no attribute will be attached to that
/// argument.
MlirOperation importJitFunctionAsFuncOp(
    MlirContext context, torch::jit::Function *function,
    std::function<MlirAttribute(int)> getArgAttribute =
        [](int) -> MlirAttribute { return {nullptr}; });

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_FUNCTION_IMPORTER_H
