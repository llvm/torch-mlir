//===- op_builder.cpp -----------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "op_builder.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "torch-mlir-c/TorchTypes.h"

using namespace torch_mlir;

OpBuilder::OpBuilder(MlirContext context) : context(context) {}

MlirOperation OpBuilder::createNoneConstant(MlirLocation loc) {
  return createMlirOperation("torch.constant.none", loc,
                             torchMlirTorchNoneTypeGet(context));
}

MlirOperation OpBuilder::createBoolConstant(MlirLocation loc, bool value) {
  return createMlirOperation(
      "torch.constant.bool", loc, torchMlirTorchBoolTypeGet(context),
      toMlirNamedAttribute("value", mlirBoolAttrGet(context, value)));
}
