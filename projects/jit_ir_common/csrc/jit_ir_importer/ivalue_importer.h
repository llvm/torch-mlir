//===- ivalue_importer.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H
#define TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H

#include <memory>

#include "class_annotator.h"
#include "import_options.h"

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

/// Main entry-point for importing torch IValue's .
/// Recursively imports `ivalue`, inserting operations at the end of `block`.
MlirValue importIValue(c10::IValue ivalue, MlirBlock block, MlirContext context,
                       ClassAnnotator &annotator,
                       const ImportOptions &importOptions);

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H
