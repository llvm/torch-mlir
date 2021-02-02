//===- ivalue_importer.cpp ------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "ivalue_importer.h"
#include "graph_importer.h"

#include <unordered_map>

#include "mlir_utils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "npcomp-c/Types.h"

using namespace torch_mlir;

namespace {
/// Helper class for holding state during recursive IValue import.
///
/// The intended usage pattern of this class is to construct it then call
/// `importIValue` exactly once. Calling `importIValue` more than once
/// is likely to produce unexpected results since the same in-memory IValue
/// can be reimported more than once. That is, subsequent calls to
/// `importIValue` will not properly unify IValue's with already-imported
/// IValue's.
///
/// TODO: Support unifying repeated IValue's.
/// This already is an issue when importing a single IValue through the current
/// API, because the same underlying Tensor (or List/Dict) can be referenced by
/// multiple properties of a module. There is an extra complication with tensors
/// because they can alias each other in fairly arbitrary ways, which we will
/// need to model with slice ops.
class IValueImporter {
public:
  IValueImporter(MlirBlock importBlock, MlirContext context)
      : importBlock(importBlock), context(context), typeMapper(context) {}

  MlirValue importIValue(c10::IValue value);

private:
  MlirValue importModule(torch::jit::Module jitModule);
  void importMethod(torch::jit::Function *function, MlirBlock nnModuleBody);

  MlirBlock importBlock;
  MlirContext context;
  TypeMapper typeMapper;
};
} // namespace

MlirValue IValueImporter::importModule(torch::jit::Module currentModule) {
  // TODO: Can we do better?
  MlirLocation loc = mlirLocationUnknownGet(context);

  MlirOperation nnModule =
      createMlirOperation("torch.nn_module", loc,
                          npcompNnModuleTypeGet(context), mlirRegionCreate());
  MlirRegion nnModuleRegion = mlirOperationGetRegion(nnModule, 0);
  mlirRegionAppendOwnedBlock(nnModuleRegion, mlirBlockCreate(0, nullptr));
  MlirBlock nnModuleBody = mlirRegionGetFirstBlock(nnModuleRegion);

  const std::vector<c10::IValue> &slots = currentModule._ivalue()->slots();
  const std::vector<c10::ClassAttribute> &classAttributes =
      currentModule.type()->getAttributes();
  assert(slots.size() == classAttributes.size() &&
         "mismatch between object and type!");
  for (int i = 0, e = slots.size(); i < e; i++) {
    const c10::ClassAttribute &classAttribute = classAttributes[i];
    MlirValue slotValue = importIValue(slots[i]);
    // TODO: Is it necessary to track whether an attribute is a "parameter"?
    createMlirOperationAtEnd(
        nnModuleBody, "torch.attr", loc, slotValue,
        toMlirNamedAttribute(
            "name", mlirStringAttrGet(
                        context, toMlirStringRef(classAttribute.getName()))));
  }

  for (torch::jit::Function *function : currentModule.type()->methods()) {
    importMethod(function, nnModuleBody);
  }

  createMlirOperationAtEnd(nnModuleBody, "torch.nn_module_terminator", loc);
  mlirBlockInsertOwnedOperationBefore(
      importBlock, mlirBlockGetTerminator(importBlock), nnModule);
  return mlirOperationGetResult(nnModule, 0);
}

MlirValue IValueImporter::importIValue(c10::IValue ivalue) {
  // TODO: Can we do better?
  MlirLocation loc = mlirLocationUnknownGet(context);

  if (ivalue.isBool()) {
    MlirType type = npcompBoolTypeGet(context);
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "basicpy.bool_constant", loc, type,
        toMlirNamedAttribute("value",
                             mlirBoolAttrGet(context, ivalue.toBool())));
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isDouble()) {
    MlirType type = mlirF64TypeGet(context);
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "basicpy.numeric_constant", loc, type,
        toMlirNamedAttribute(
            "value", mlirFloatAttrDoubleGet(context, type, ivalue.toDouble())));
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isInt()) {
    MlirType type = mlirIntegerTypeGet(context, 64);
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "basicpy.numeric_constant", loc, type,
        toMlirNamedAttribute("value",
                             mlirIntegerAttrGet(type, ivalue.toInt())));
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isTensor()) {
    at::Tensor tensor = ivalue.toTensor().contiguous();
    MlirAttribute denseElements = converTensorToMlirElementsAttr(tensor, loc);
    MlirOperation constant = createMlirOperationAtEnd(
        importBlock, "std.constant", loc, mlirAttributeGetType(denseElements),
        toMlirNamedAttribute("value", denseElements));
    MlirOperation ndarray = createMlirOperationAtEnd(
        importBlock, "numpy.create_array_from_tensor", loc,
        npcompNdArrayTypeGetUnranked(npcompAnyDtypeTypeGet(context)),
        mlirOperationGetResult(constant, 0));
    return mlirOperationGetResult(ndarray, 0);
  }
  if (ivalue.isModule()) {
    return importModule(ivalue.toModule());
  }
  std::stringstream msg;
  msg << "Unsupported ivalue: " << ivalue;
  throw std::invalid_argument(msg.str());
}

void IValueImporter::importMethod(torch::jit::Function *function,
                                  MlirBlock nnModuleBody) {
  // We make an effort for the func op's symbol name to be useful for debugging,
  // but still clearly non-load-bearing.
  std::string symName =
      "__npcomp_priv_fn." + function->qualname().qualifiedName();
  MlirOperation func =
      importGraphAsFuncOp(context, function->graph().get(), symName);
  mlirOperationSetAttributeByName(
      func, toMlirStringRef("sym_visibility"),
      mlirStringAttrGet(context, toMlirStringRef("private")));
  mlirBlockInsertOwnedOperationBefore(
      importBlock, mlirBlockGetTerminator(importBlock), func);
  createMlirOperationAtEnd(
      nnModuleBody, "torch.method", mlirLocationUnknownGet(context),
      toMlirNamedAttribute(
          "name",
          mlirStringAttrGet(context, toMlirStringRef(function->name()))),
      toMlirNamedAttribute("function", mlirFlatSymbolRefAttrGet(
                                           context, toMlirStringRef(symName))));
}
void torch_mlir::importIValue(c10::IValue ivalue, MlirBlock block,
                              MlirContext context) {
  // When debugging module importing, it can be useful to dump as so:
  // if (ivalue.isModule())
  //   ivalue.toModule().dump(true, true, true);
  IValueImporter importer(block, context);
  importer.importIValue(ivalue);
}
