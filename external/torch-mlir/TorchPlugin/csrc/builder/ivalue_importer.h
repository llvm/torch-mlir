//===- ivalue_importer.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRPLUGIN_CSRC_IVALUE_IMPORTER_H
#define TORCHMLIRPLUGIN_CSRC_IVALUE_IMPORTER_H

#include <memory>

#include "../pybind.h"
#include "class_annotator.h"

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

/// Main entry-point for importing torch IValue's .
/// Recursively imports `ivalue`, inserting operations at the end of `block`.
void importIValue(c10::IValue ivalue, MlirBlock block, MlirContext context,
                  ClassAnnotator &annotator);

} // namespace torch_mlir

#endif // TORCHMLIRPLUGIN_CSRC_IVALUE_IMPORTER_H
