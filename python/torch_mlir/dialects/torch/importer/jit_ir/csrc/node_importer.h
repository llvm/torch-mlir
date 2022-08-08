//===- node_importer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_NODE_IMPORTER_H
#define TORCHMLIRJITIRIMPORTER_CSRC_NODE_IMPORTER_H

#include "import_options.h"

#include <memory>

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

using CreateTerminatorFn =
    std::function<void(c10::ArrayRef<MlirValue>, MlirBlock)>;

/// Import `jitBlock` into a corresponding `MlirBlock`.
///
/// Because `jit::Block` does not have a concept of terminator in the MLIR sense
/// (it is kind of "built-in" to the block, and not a free op chosen by the
/// enclosing op), the `createTerminator` function will be used to create the
/// terminator for the created block. Type adjustments like handling
/// derefinement can be handled there as well.
///
/// `blockArgTypes`, if present, gives a set of types that the block arguments
/// are required to be for correctness. The code will internally attempt to
/// adjust the types to the block argument types.
/// TODO: Formalize what type conversions are allowed here.
MlirBlock importBlock(
    MlirContext context, torch::jit::Block *jitBlock,
    CreateTerminatorFn createTerminator,
    c10::optional<c10::ArrayRef<MlirType>> blockArgTypes = c10::nullopt,
    const ImportOptions &importOptions = {});

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_NODE_IMPORTER_H
