//===- graph_importer.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_GRAPH_IMPORTER_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_GRAPH_IMPORTER_H

#include <memory>

#include "../pybind.h"
#include "func_builder.h"
#include "node_importer.h"

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

/// Main entry-point for importing torch::jit::Graph instances.
///
/// This code doesn't handle importing of torch::jit::Module's. See
/// IValueImporter for that.
///
/// A Graph is a combination of an MLIR context, function, and builder.
/// See NodeImporter for importing of the core IR Node/Block
/// structure that is analogous to MLIR's Operation/Region/Block core structure.
MlirOperation importGraphAsFuncOp(MlirContext context, torch::jit::Graph *graph,
                                  const std::string &name);

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_GRAPH_IMPORTER_H
