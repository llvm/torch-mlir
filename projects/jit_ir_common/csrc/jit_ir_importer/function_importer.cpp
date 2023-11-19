//===- function_importer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "function_importer.h"

#include <unordered_map>

#include "mlir_utils.h"
#include "torch_to_mlir_utils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"

using namespace torch_mlir;

MlirOperation torch_mlir::importJitFunctionAsFuncOp(
    MlirContext context, torch::jit::Function *function,
    std::function<MlirAttribute(int)> getArgAttribute,
    const ImportOptions &importOptions) {
  // Useful for debugging:
  // graph->dump();
  MlirLocation loc = mlirLocationUnknownGet(context);
  MlirType functionType =
      getFunctionTypeFromSchema(context, function->getSchema(), importOptions);
  // Use the function's qualified name from the compilation unit.
  // This is a stable linkage name that matches Python module lookup
  // conventions (see compilation unit import in IValueImporter for more details
  // on qualified names).
  MlirAttribute symNameAttr = mlirStringAttrGet(
      context, toMlirStringRef(function->qualname().qualifiedName()));
  MlirOperation func = createMlirOperation(
      "func.func", loc, mlirRegionCreate(),
      toMlirNamedAttribute("function_type", mlirTypeAttrGet(functionType)),
      toMlirNamedAttribute("sym_name", symNameAttr));
  std::vector<MlirAttribute> argAttrDicts;
  for (int i = 0, e = mlirFunctionTypeGetNumInputs(functionType); i != e; i++) {
    MlirAttribute argAttrDict = getArgAttribute(i);
    if (mlirAttributeIsNull(argAttrDict)) {
      argAttrDicts.push_back(mlirDictionaryAttrGet(context, 0, nullptr));
    } else {
      argAttrDicts.push_back(argAttrDict);
    }
  }
  mlirOperationSetAttributeByName(
      func, toMlirStringRef("arg_attrs"),
      mlirArrayAttrGet(context, argAttrDicts.size(), argAttrDicts.data()));
  MlirRegion bodyRegion = mlirOperationGetRegion(func, 0);
  std::vector<MlirType> resultTypes;
  for (int i = 0, e = mlirFunctionTypeGetNumResults(functionType); i != e;
       i++) {
    resultTypes.push_back(mlirFunctionTypeGetResult(functionType, i));
  }
  std::vector<MlirType> inputTypes;
  for (int i = 0, e = mlirFunctionTypeGetNumInputs(functionType); i != e; i++) {
    inputTypes.push_back(mlirFunctionTypeGetInput(functionType, i));
  }
  auto createTerminator = [&](c10::ArrayRef<MlirValue> yieldedValues,
                              MlirBlock appendToBlock) {
    createMlirOperationAtEnd(appendToBlock, "func.return", loc,
                             adjustStaticInformationForValues(
                                 appendToBlock, loc, yieldedValues, resultTypes,
                                 /*userAllowsRefinement=*/false));
  };
  MlirBlock block = importBlock(
      context, torch::jit::toGraphFunction(*function).graph()->block(),
      createTerminator, inputTypes, importOptions);
  mlirRegionAppendOwnedBlock(bodyRegion, block);
  return func;
}
