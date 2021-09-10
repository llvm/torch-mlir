//===- op_builder.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRPLUGIN_CSRC_BUILDER_OP_BUILDER_H
#define TORCHMLIRPLUGIN_CSRC_BUILDER_OP_BUILDER_H

#include "mlir_utils.h"
#include "torch_to_mlir_utils.h"

#include "mlir-c/IR.h"

#include <ATen/Tensor.h>
#include <ATen/core/function_schema.h>

namespace torch_mlir {

/// Convenience class for centralizing creation of some operations.
///
/// For many operations, they are created in only one place in the code, and so
/// the functionality in mlir_utils.h is enough.
///
/// TODO: Add insertion point like the normal MLIR builder?
class OpBuilder {
public:
  OpBuilder(MlirContext context);
  MlirOperation createNoneConstant(MlirLocation loc);
  MlirOperation createBoolConstant(MlirLocation loc, bool value);
  MlirOperation createStdConstant(MlirLocation loc, MlirAttribute value);

private:
  MlirContext context;
};

} // namespace torch_mlir

#endif // TORCHMLIRPLUGIN_CSRC_BUILDER_OP_BUILDER_H
