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
#include "npcomp-c/Types.h"

using namespace torch_mlir;

OpBuilder::OpBuilder(MlirContext context) : context(context) {}

MlirOperation OpBuilder::createNoneConstant(MlirLocation loc) {
  return createMlirOperation("basicpy.singleton", loc,
                             npcompNoneTypeGet(context));
}

MlirOperation OpBuilder::createBoolConstant(MlirLocation loc, bool value) {
  return createMlirOperation(
      "basicpy.bool_constant", loc, npcompBoolTypeGet(context),
      toMlirNamedAttribute("value", mlirBoolAttrGet(context, value)));
}

MlirOperation OpBuilder::createStdConstant(MlirLocation loc,
                                           MlirAttribute value) {
  return createMlirOperation("std.constant", loc, mlirAttributeGetType(value),
                             toMlirNamedAttribute("value", value));
}
