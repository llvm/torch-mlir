//===- ivalue_importer.cpp ------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "ivalue_importer.h"
#include "class_annotator.h"
#include "graph_importer.h"

#include <unordered_map>

#include "mlir_utils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "npcomp-c/Types.h"

using namespace torch_mlir;

// Hashing functionality for IValue's.
//
// What we want here is a strict object identity hash. This is different from
// what Python usually treats as hashing, which is a deep equality hash. In
// Python terms, what we want here is a hash of `id(x)` -- unfortunately, IValue
// is not uniformly heap allocated the way a `PyObject*` is, so special handling
// is needed. At the time of this writing, there seem to be two different
// implementations, neither of which is exactly what we want.
//
// - c10::IValue::hash static method
//   - Problem: Doesn't handle certain data types, in particular objects (which
//   modules are a special case of) and lists/dicts. This makes sense when
//   reflecting the Python semantics.
// - c10::WeakIValue::hash method
//   - Problem: it literally just returns the bits of the "union" as an int.
//   This seems to read uninitialized bits for the bool variant.
//
// We use the `c10::IValue::hash` static method with special cases for data
// types that need their identity to be handled specially. `c10::IValue::hash`
// seems to be implemented in a principled way following the Python semantics,
// which is compatible with the semantics we want (for the subset it doesn't
// throw an error on).
namespace {
struct IValueHasher {
  size_t operator()(const c10::IValue &ivalue) const {
    if (ivalue.isObject() || ivalue.isList()) {
      return std::hash<const void *>()(
          static_cast<const void *>(ivalue.internalToPointer()));
    }

    return c10::IValue::hash(ivalue);
  }
};
} // namespace

// TODO: The implementation of isSameIdentity looks vulnerable to malloc reusing
// the same memory block (if this hash function is used in an online setting,
// such as when tracing). Can we do better?
namespace {
struct IValueEq {
  bool operator()(const c10::IValue &lhs, const c10::IValue &rhs) const {
    return lhs.isSameIdentity(rhs);
  }
};
} // namespace

namespace {
/// Helper class for holding state during recursive IValue import.
///
/// The intended usage pattern of this class is to construct it then call
/// `importIValue`.
///
/// The `importIValue` method can be called more than once, and values are
/// unified *by object identity*. For types isomorphic to Python builtin types
/// the behavior is what you would expect from `id(x)`.
///
/// For tensors, object identity is a little tricky. As background, at::Tensor
/// basically has 4 parts:
/// - at::Tensor which is a smart pointer to at::TensorImpl
/// - at::TensorImpl which holds sizes/strides/etc. and points to at::Storage
///   - the address of the at::TensorImpl is the identity of the tensor.
/// - at::Storage which is a smart pointer to at::StorageImpl
/// - at::StorageImpl which is a low-level buffer
///   - the address of the at::StorageImpl is the identity of the "storage".
///
/// Multiple different tensors can share the same underlying storage. We
/// currently import tensors by identity and emit errors in the case of tensors
/// with different identity but sharing the same storage. This is done because
/// correctly modeling the many ways that tensors can overlap and alias when
/// they share storage is difficult. Example hard cases are weird
/// strides/offsets that overlap, and even cases where the data types mismatch
/// (PyTorch allows this!).
class IValueImporter {
public:
  IValueImporter(MlirBlock importBlock, MlirContext context,
                 ClassAnnotator &annotator)
      : importBlock(importBlock), context(context), typeMapper(context),
        annotator(annotator) {}

  MlirValue importIValue(c10::IValue value);

private:
  MlirValue rawImportIValue(c10::IValue value);
  MlirValue importModule(torch::jit::Module jitModule);
  void importMethod(torch::jit::Function *function, MlirBlock classTypeBody,
                    const MethodAnnotation &methodAnnotation);
  void importClassType(c10::ClassType *classType);

  MlirBlock importBlock;
  MlirContext context;
  TypeMapper typeMapper;
  ClassAnnotator &annotator;

  // Map tracking already-imported values.
  std::unordered_map<c10::IValue, MlirValue, IValueHasher, IValueEq> valueMap;
  // Used to detect potentially aliasing tensors.
  std::unordered_set<c10::StorageImpl *> seenStorageImpls;
  // The set of ClassType's that have already been imported.
  //
  // ClassType's are referenced via their `classType->name()->qualifiedName()`
  // string (as an MLIR symbol name) so we don't need to keep a map associating
  // them with the MlirOperation that they import into.
  std::unordered_set<c10::ClassType *> classTypes;
  // The stack of attribute names we have traversed to reach the current IValue.
  // Used for diagnostics.
  std::vector<std::string> attributeNameStack;
  // The root module encountered during recursive IValue traversal.
  // Used for diagnostics.
  // Note that the "top-level" object being imported can in theory be a list
  // of modules, so this is populated when our recursive traversal enters a
  // module not enclosed in any other module, and unset after our recursive
  // traversal exits the module.
  c10::optional<std::string> rootModuleName;
};
} // namespace

MlirValue IValueImporter::importModule(torch::jit::Module currentModule) {
  // TODO: Can we do better?
  MlirLocation loc = mlirLocationUnknownGet(context);

  c10::optional<c10::QualifiedName> maybeName = currentModule.type()->name();
  if (!maybeName) {
    throw std::invalid_argument("cannot import unnamed module");
  }
  std::string moduleTypeName = maybeName->qualifiedName();

  // Ensure the class type has been imported.
  importClassType(currentModule.type().get());

  MlirOperation nnModule = createMlirOperation(
      "torch.nn_module", loc,
      npcompNnModuleTypeGet(context, toMlirStringRef(moduleTypeName)),
      mlirRegionCreate());
  MlirRegion nnModuleRegion = mlirOperationGetRegion(nnModule, 0);
  mlirRegionAppendOwnedBlock(nnModuleRegion, mlirBlockCreate(0, nullptr));
  MlirBlock nnModuleBody = mlirRegionGetFirstBlock(nnModuleRegion);

  if (!rootModuleName.has_value()) {
    rootModuleName = moduleTypeName;
  }

  const std::vector<c10::IValue> &slots = currentModule._ivalue()->slots();
  const std::vector<c10::ClassAttribute> &classAttributes =
      currentModule.type()->getAttributes();
  assert(slots.size() == classAttributes.size() &&
         "mismatch between object and type!");
  for (int i = 0, e = slots.size(); i < e; i++) {
    const c10::ClassAttribute &classAttribute = classAttributes[i];
    attributeNameStack.push_back(classAttribute.getName());
    MlirValue slotValue = importIValue(slots[i]);
    // TODO: Is it necessary to track whether an attribute is a "parameter"?
    createMlirOperationAtEnd(
        nnModuleBody, "torch.slot", loc, slotValue,
        toMlirNamedAttribute(
            "name", mlirStringAttrGet(
                        context, toMlirStringRef(classAttribute.getName()))));
    attributeNameStack.pop_back();
  }

  if (rootModuleName.has_value()) {
    rootModuleName = c10::nullopt;
  }

  createMlirOperationAtEnd(nnModuleBody, "torch.nn_module_terminator", loc);
  mlirBlockInsertOwnedOperationBefore(
      importBlock, mlirBlockGetTerminator(importBlock), nnModule);
  return mlirOperationGetResult(nnModule, 0);
}

MlirValue IValueImporter::importIValue(c10::IValue ivalue) {
  auto it = valueMap.find(ivalue);
  if (it != valueMap.end()) {
    return it->second;
  }
  // Reject potentially aliased tensors.
  if (ivalue.isTensor()) {
    c10::StorageImpl *storageImpl =
        ivalue.toTensor().storage().unsafeGetStorageImpl();
    if (!seenStorageImpls.insert(storageImpl).second) {
      std::stringstream msg;
      msg << "Unhandled tensor that shares storage with another tensor.";
      if (rootModuleName) {
        msg << "\nFound at path '<root>."
            << c10::QualifiedName(attributeNameStack).qualifiedName()
            << "' from root object '" << *rootModuleName << "'";
      }
      throw std::invalid_argument(msg.str());
    }
  }
  MlirValue value = rawImportIValue(ivalue);
  valueMap[ivalue] = value;
  return value;
}

MlirValue IValueImporter::rawImportIValue(c10::IValue ivalue) {
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
  if (ivalue.isList()) {
    c10::List<c10::IValue> list = ivalue.toList();
    std::vector<MlirValue> elems;
    for (const c10::IValue &elem : list) {
      elems.push_back(importIValue(elem));
    }
    MlirOperation operation =
        createMlirOperationAtEnd(importBlock, "basicpy.build_list", loc,
                                 npcompListTypeGet(context), elems);
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
  if (ivalue.isNone()) {
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "basicpy.singleton", loc, npcompNoneTypeGet(context));
    return mlirOperationGetResult(operation, 0);
  }
  std::stringstream msg;
  msg << "Unsupported ivalue: " << ivalue;
  throw std::invalid_argument(msg.str());
}

void IValueImporter::importMethod(torch::jit::Function *function,
                                  MlirBlock classTypeBody,
                                  const MethodAnnotation &methodAnnotation) {
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
  c10::optional<MlirNamedAttribute> isPrivate;
  if (!methodAnnotation.isExported) {
    isPrivate = toMlirNamedAttribute("isPrivate", mlirUnitAttrGet(context));
  }
  createMlirOperationAtEnd(
      classTypeBody, "torch.method", mlirLocationUnknownGet(context),
      toMlirNamedAttribute(
          "name",
          mlirStringAttrGet(context, toMlirStringRef(function->name()))),
      toMlirNamedAttribute("function", mlirFlatSymbolRefAttrGet(
                                           context, toMlirStringRef(symName))),
      isPrivate);
}

void IValueImporter::importClassType(c10::ClassType *classType) {
  if (!classTypes.insert(classType).second) {
    return;
  }

  // TODO: Can we do better?
  MlirLocation loc = mlirLocationUnknownGet(context);

  MlirOperation op = createMlirOperationAtEnd(
      importBlock, "torch.class_type", loc, mlirRegionCreate(),
      toMlirNamedAttribute(
          "sym_name",
          mlirStringAttrGet(
              context, toMlirStringRef(classType->name()->qualifiedName()))));
  MlirRegion region = mlirOperationGetRegion(op, 0);
  mlirRegionAppendOwnedBlock(region, mlirBlockCreate(0, nullptr));
  MlirBlock classTypeBody = mlirRegionGetFirstBlock(region);

  ClassAnnotation &classAnnotation =
      annotator.getOrCreateClassAnnotation(classType);

  const auto &attributeAnnotations = classAnnotation.getAttributeAnnotations();
  const auto &classAttributes = classType->getAttributes();
  for (int i = 0, e = classAttributes.size(); i != e; i++) {
    const c10::ClassAttribute &classAttribute = classAttributes[i];
    c10::optional<MlirNamedAttribute> isPrivate;
    if (!attributeAnnotations[i].isExported) {
      isPrivate = toMlirNamedAttribute("isPrivate", mlirUnitAttrGet(context));
    }
    createMlirOperationAtEnd(
        classTypeBody, "torch.attr", loc,
        toMlirNamedAttribute(
            "name", mlirStringAttrGet(
                        context, toMlirStringRef(classAttribute.getName()))),
        toMlirNamedAttribute("type",
                             mlirTypeAttrGet(typeMapper.mapFromTorchType(
                                 loc, classAttribute.getType()))),
        isPrivate);
  }

  const auto &methodAnnotations = classAnnotation.getMethodAnnotations();
  const auto &methods = classType->methods();
  for (int i = 0, e = methods.size(); i != e; i++) {
    importMethod(methods[i], classTypeBody, methodAnnotations[i]);
  }

  createMlirOperationAtEnd(classTypeBody, "torch.class_type_terminator", loc);
}

void torch_mlir::importIValue(c10::IValue ivalue, MlirBlock block,
                              MlirContext context, ClassAnnotator &annotator) {
  // When debugging module importing, it can be useful to dump as so:
  // if (ivalue.isModule())
  //   ivalue.toModule().dump(true, false, false);
  IValueImporter importer(block, context, annotator);
  importer.importIValue(ivalue);
}
