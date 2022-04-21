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

namespace py = pybind11;
using namespace torch_mlir;

MlirOperation torch_mlir::importJitFunctionAsFuncOp(
    MlirContext context, torch::jit::Function *function,
    std::function<MlirAttribute(int)> getArgAttribute) {
  MlirLocation loc = mlirLocationUnknownGet(context);

  // Extract the schema return types. These are the types that we need to
  // derefine the block's return values to.
  auto mapType = [&](const c10::Argument &arg) {
    const c10::TypePtr &torchType = arg.type();
    MlirType type = getMlirTypeFromTorchType(loc, torchType);
    if (mlirTypeIsNull(type)) {
      std::stringstream msg;
      msg << "unsupported type in function schema: '"
          << c10::toString(torchType) << "'";
      throw std::invalid_argument(msg.str());
    }
    return type;
  };
  std::vector<MlirType> resultTypes =
      c10::fmap(function->getSchema().returns(), mapType);
  auto createTerminator = [&](c10::ArrayRef<MlirValue> yieldedValues,
                              MlirBlock appendToBlock) {
    createMlirOperationAtEnd(
        appendToBlock, "func.return", loc,
        derefineValues(yieldedValues, resultTypes, loc, appendToBlock));
  };

  // Import the operations in the function.
  MlirBlock block = importBlock(
      context, torch::jit::toGraphFunction(*function).graph()->block(),
      createTerminator);

  // Now, create the func op itself.

  // Extract the function input types directly from the block.
  //
  // `torch.jit.trace` will create functions where the schema has type-erased
  // tensors, but the block will have precise tensor types (and everywhere in
  // the code where the function is called the types will be precise).
  std::vector<MlirType> inputTypes;
  for (int i = 0, e = mlirBlockGetNumArguments(block); i < e; ++i) {
    inputTypes.push_back(mlirValueGetType(mlirBlockGetArgument(block, i)));
  }
  MlirType functionType =
      mlirFunctionTypeGet(context, inputTypes.size(), inputTypes.data(),
                          resultTypes.size(), resultTypes.data());

  // Use the function's qualified name from the compilation unit.
  // This is a stable linkage name that matches Python module lookup
  // conventions (see compilation unit import in IValueImporter for more details
  // on qualified names).
  MlirAttribute symNameAttr = mlirStringAttrGet(
      context, toMlirStringRef(function->qualname().qualifiedName()));
  MlirOperation func = createMlirOperation(
      "builtin.func", loc, mlirRegionCreate(),
      toMlirNamedAttribute("type", mlirTypeAttrGet(functionType)),
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
  mlirRegionAppendOwnedBlock(bodyRegion, block);
  return func;
}
