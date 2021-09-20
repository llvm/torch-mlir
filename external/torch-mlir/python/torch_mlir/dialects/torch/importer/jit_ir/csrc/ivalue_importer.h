//===- ivalue_importer.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE.pytorch for license information.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H
#define TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H

#include <memory>

#include "class_annotator.h"
#include "pybind.h"

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

#endif // TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H
