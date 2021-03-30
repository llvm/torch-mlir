//===- function_importer.cpp ----------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "function_importer.h"

#include <unordered_map>

#include "mlir_utils.h"
#include "torch_to_mlir_utils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"

namespace py = pybind11;
using namespace torch_mlir;

MlirOperation torch_mlir::importJitFunctionAsFuncOp(
    MlirContext context, torch::jit::Function *function,
    std::function<MlirAttribute(int)> getArgAttribute) {
  // Useful for debugging:
  // graph->dump();
  MlirLocation loc = mlirLocationUnknownGet(context);
  MlirType functionType =
      getFunctionTypeFromSchema(context, function->getSchema());
  // Use the function's qualified name from the compilation unit.
  // This is a stable linkage name that matches Python module lookup
  // conventions (see compilation unit import in IValueImporter for more details
  // on qualified names).
  MlirAttribute symNameAttr = mlirStringAttrGet(
      context, toMlirStringRef(function->qualname().qualifiedName()));
  MlirOperation func = createMlirOperation(
      "func", loc, mlirRegionCreate(),
      toMlirNamedAttribute("type", mlirTypeAttrGet(functionType)),
      toMlirNamedAttribute("sym_name", symNameAttr));
  for (int i = 0, e = mlirFunctionTypeGetNumInputs(functionType); i != e; i++) {
    MlirAttribute argAttr = getArgAttribute(i);
    if (mlirAttributeIsNull(argAttr)) {
      continue;
    }
    mlirOperationSetAttributeByName(
        func, toMlirStringRef("arg" + std::to_string(i)), argAttr);
  }
  MlirRegion bodyRegion = mlirOperationGetRegion(func, 0);
  std::vector<MlirType> resultTypes;
  for (int i = 0, e = mlirFunctionTypeGetNumResults(functionType); i != e;
       i++) {
    resultTypes.push_back(mlirFunctionTypeGetResult(functionType, i));
  }
  auto createTerminator = [&](c10::ArrayRef<MlirValue> yieldedValues,
                              MlirBlock appendToBlock) {
    createMlirOperationAtEnd(
        appendToBlock, "std.return", loc,
        derefineValues(yieldedValues, resultTypes, loc, appendToBlock));
  };
  MlirBlock block =
      importBlock(context, function->graph()->block(), createTerminator);
  mlirRegionAppendOwnedBlock(bodyRegion, block);
  return func;
}
