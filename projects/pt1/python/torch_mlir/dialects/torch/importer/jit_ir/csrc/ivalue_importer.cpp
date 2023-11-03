//===- ivalue_importer.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "ivalue_importer.h"
#include "class_annotator.h"
#include "function_importer.h"
#include "torch_to_mlir_utils.h"

#include <unordered_map>

#include "mlir_utils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "torch-mlir-c/TorchTypes.h"

#include "ATen/native/quantized/PackedParams.h"

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
    if (ivalue.isObject() || ivalue.isList() || ivalue.isGenericDict()) {
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
                 ClassAnnotator &annotator, const ImportOptions &importOptions)
      : importBlock(importBlock), context(context), annotator(annotator),
        importOptions(importOptions) {}

  MlirValue importIValue(c10::IValue ivalue);

private:
  MlirValue rawImportIValue(c10::IValue ivalue);
  MlirValue importTensor(c10::IValue ivalue);
  MlirValue importModule(torch::jit::Module jitModule);
  void importMethod(torch::jit::Function *function, MlirBlock classTypeBody,
                    const MethodAnnotation &methodAnnotation);
  void importClassType(c10::ClassType *classType);
  void importCompilationUnit(torch::jit::CompilationUnit *cu);

  MlirBlock importBlock;
  MlirContext context;
  ClassAnnotator &annotator;
  const ImportOptions &importOptions;

  // Map tracking already-imported values.
  std::unordered_map<c10::IValue, MlirValue, IValueHasher, IValueEq> valueMap;

  // The unique compilation unit that is shared by all modules reachable
  // from the root ivalue being imported.
  // It basically contains a symbol table of functions which are referenced from
  // e.g. methods (the function names are meaningful and match with Python's
  // module hierarchy, with the exception of `__main__` being replaced with
  // `__torch__`).
  torch::jit::CompilationUnit *compilationUnit = nullptr;

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

// RAII pattern to insert an operation before going out of scope.
class InserterGuard {
private:
  MlirBlock importBlock;
  MlirOperation nnModule;

public:
  InserterGuard(MlirBlock importBlock, MlirOperation nnModule)
      : importBlock(importBlock), nnModule(nnModule) {}

  ~InserterGuard() {
    mlirBlockInsertOwnedOperationBefore(
        importBlock, mlirBlockGetTerminator(importBlock), nnModule);
  }
};

MlirValue IValueImporter::importModule(torch::jit::Module currentModule) {
  // TODO: Can we do better?
  MlirLocation loc = mlirLocationUnknownGet(context);

  c10::optional<c10::QualifiedName> maybeName = currentModule.type()->name();
  if (!maybeName) {
    throw std::invalid_argument("cannot import unnamed module");
  }
  std::string moduleTypeName = maybeName->qualifiedName();

  // If this is the first time we are encountering a module, import the
  // compilation unit.
  importCompilationUnit(currentModule._ivalue()->compilation_unit().get());

  // Ensure the class type has been imported.
  importClassType(currentModule.type().get());

  MlirOperation nnModule = createMlirOperation(
      "torch.nn_module", loc,
      torchMlirTorchNnModuleTypeGet(context, toMlirStringRef(moduleTypeName)),
      mlirRegionCreate());
  MlirRegion nnModuleRegion = mlirOperationGetRegion(nnModule, 0);
  mlirRegionAppendOwnedBlock(nnModuleRegion, mlirBlockCreate(0, nullptr, nullptr));
  MlirBlock nnModuleBody = mlirRegionGetFirstBlock(nnModuleRegion);
  InserterGuard inserterGuard(importBlock, nnModule);

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
    MlirType type = torchMlirTorchBoolTypeGet(context);
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "torch.constant.bool", loc, type,
        toMlirNamedAttribute("value",
                             mlirBoolAttrGet(context, ivalue.toBool())));
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isDouble()) {
    MlirType type = torchMlirTorchFloatTypeGet(context);
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "torch.constant.float", loc, type,
        toMlirNamedAttribute(
            "value", mlirFloatAttrDoubleGet(context, mlirF64TypeGet(context),
                                            ivalue.toDouble())));
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isInt()) {
    MlirType type = torchMlirTorchIntTypeGet(context);
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "torch.constant.int", loc, type,
        toMlirNamedAttribute("value",
                             mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64),
                                                ivalue.toInt())));
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isList()) {
    c10::List<c10::IValue> list = ivalue.toList();
    std::vector<MlirValue> elems;
    for (const c10::IValue &elem : list) {
      elems.push_back(importIValue(elem));
    }
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "torch.prim.ListConstruct", loc,
        torchMlirTorchListTypeGet(
            getMlirTypeFromTorchType(loc, list.elementType(), importOptions)),
        elems);
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isGenericDict()) {
    c10::Dict<c10::IValue, c10::IValue> dict = ivalue.toGenericDict();
    std::vector<MlirValue> keys;
    std::vector<MlirValue> values;
    for (auto it = dict.begin(); it != dict.end(); it++) {
      keys.push_back(importIValue(it->key()));
      values.push_back(importIValue(it->value()));
    }
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "torch.prim.DictConstruct", loc,
        torchMlirTorchDictTypeGet(
            getMlirTypeFromTorchType(loc, dict.keyType(), importOptions),
            getMlirTypeFromTorchType(loc, dict.valueType(), importOptions)),
        keys, values);
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isTuple()) {
    auto list = ivalue.toTuple()->elements();
    std::vector<MlirValue> operands;
    std::vector<MlirType> types;
    for (const c10::IValue &elem : list) {
      MlirValue operand = importIValue(elem);
      operands.push_back(operand);
      types.push_back(mlirValueGetType(operand));
    }
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "torch.prim.TupleConstruct", loc,
        torchMlirTorchTupleTypeGet(context, types.size(), types.data()),
        operands);
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isTensor()) {
    return importTensor(ivalue);
  }
  if (ivalue.isModule()) {
    return importModule(ivalue.toModule());
  }
  if (ivalue.isString()) {
    MlirOperation operation = createMlirOperationAtEnd(
        importBlock, "torch.constant.str", loc,
        torchMlirTorchStringTypeGet(context),
        toMlirNamedAttribute(
            "value",
            mlirStringAttrGet(context,
                              toMlirStringRef(ivalue.toString()->string()))));
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isNone()) {
    MlirOperation operation =
        createMlirOperationAtEnd(importBlock, "torch.constant.none", loc,
                                 torchMlirTorchNoneTypeGet(context));
    return mlirOperationGetResult(operation, 0);
  }
  if (ivalue.isCustomClass()) {
    if (ivalue.type().get() ==
        c10::getCustomClassType<c10::intrusive_ptr<LinearPackedParamsBase>>()
            .get()) {
      c10::intrusive_ptr<LinearPackedParamsBase> linearParams =
          ivalue.toCustomClass<LinearPackedParamsBase>();
      at::Tensor weight;
      c10::optional<at::Tensor> bias;
      std::tie(weight, bias) = linearParams->unpack();
      MlirValue weightValue = importIValue(c10::IValue(weight));
      c10::optional<MlirValue> biasValue = c10::nullopt;
      if (bias.has_value()) {
        biasValue = importIValue(c10::IValue(*bias));
      }
      MlirOperation operation = createMlirOperationAtEnd(
          importBlock, "torch.linear_params.create", loc,
          torchMlirTorchLinearParamsTypeGet(context), weightValue, biasValue);
      return mlirOperationGetResult(operation, 0);
    }
  }
  std::stringstream msg;
  msg << "Unsupported ivalue: " << ivalue;
  throw std::invalid_argument(msg.str());
}

MlirValue IValueImporter::importTensor(c10::IValue ivalue) {
  assert(ivalue.isTensor() && "expected a tensor!");

  // TODO: Can we do better?
  MlirLocation loc = mlirLocationUnknownGet(context);

  // Import the bulk tensor representation.
  at::Tensor tensor = ivalue.toTensor().contiguous();
  MlirAttribute denseElements = convertTensorToMlirElementsAttr(tensor, loc);

  MlirOperation tensorOp;

  if (importOptions.assumeTensorsHaveValueSemantics) {
    tensorOp = createMlirOperationAtEnd(
        importBlock, "torch.vtensor.literal", loc,
        torchMlirTorchValueTensorTypeGetFromAttribute(denseElements),
        toMlirNamedAttribute("value", denseElements));
  } else {
    tensorOp = createMlirOperationAtEnd(
        importBlock, "torch.tensor.literal", loc,
        torchMlirTorchNonValueTensorTypeGetFromAttribute(denseElements),
        toMlirNamedAttribute("value", denseElements));
  }

  MlirValue tensorReprValue = mlirOperationGetResult(tensorOp, 0);

  // Construct the complete tensor value. This is trivial for most tensors, but
  // for quantized tensors (and probably sparse too, TBD) there is more for us
  // to do.
  MlirValue tensorValue;
  if (tensor.is_quantized()) {
    // Note that Torch models quantization in a type-erased way. So we don't
    // make an effort here to do any special static modeling. If desired, later
    // compiler stages that are building a statically modeled quantization
    // representation will need to convert this to their representation.
    std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
    MlirType quantizedTensorType;
    if (importOptions.assumeTensorsHaveValueSemantics) {
      quantizedTensorType = torchMlirTorchValueTensorTypeGet(
          context, shape.size(), shape.data(),
          getMlirTypeForTorchScalarType(loc, tensor.scalar_type()));
    } else {
      quantizedTensorType = torchMlirTorchNonValueTensorTypeGet(
          context, shape.size(), shape.data(),
          getMlirTypeForTorchScalarType(loc, tensor.scalar_type()));
    }
    if (tensor.qscheme() == c10::kPerTensorAffine) {
      MlirValue qScale = importIValue(c10::IValue(tensor.q_scale()));
      MlirValue zeroPoint = importIValue(c10::IValue(tensor.q_zero_point()));
      MlirOperation quantizedTensor = createMlirOperationAtEnd(
          importBlock, "torch.per_tensor_affine.create", loc,
          quantizedTensorType, tensorReprValue, qScale, zeroPoint);
      tensorValue = mlirOperationGetResult(quantizedTensor, 0);
    } else {
      std::stringstream msg;
      msg << "Unsupported quantization scheme '"
          << c10::toString(tensor.qscheme()) << "' for tensor: " << ivalue;
      throw std::invalid_argument(msg.str());
    }
  } else {
    tensorValue = tensorReprValue;
  }

  return tensorValue;
}

void IValueImporter::importMethod(torch::jit::Function *function,
                                  MlirBlock classTypeBody,
                                  const MethodAnnotation &methodAnnotation) {
  // The function's name becomes the MLIR symbol table name of the imported func
  // when we import the compilation unit.
  const std::string &symName = function->qualname().qualifiedName();
  MlirAttribute functionSymbolRef =
      mlirFlatSymbolRefAttrGet(context, toMlirStringRef(symName));

  c10::optional<MlirNamedAttribute> isPrivate;
  if (!methodAnnotation.isExported) {
    isPrivate = toMlirNamedAttribute("isPrivate", mlirUnitAttrGet(context));
  }
  createMlirOperationAtEnd(
      classTypeBody, "torch.method", mlirLocationUnknownGet(context),
      toMlirNamedAttribute(
          "name",
          mlirStringAttrGet(context, toMlirStringRef(function->name()))),
      toMlirNamedAttribute("function", functionSymbolRef), isPrivate);
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
  mlirRegionAppendOwnedBlock(region, mlirBlockCreate(0, nullptr, nullptr));
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
        toMlirNamedAttribute("type", mlirTypeAttrGet(getMlirTypeFromTorchType(
                                         loc, classAttribute.getType(), importOptions))),
        isPrivate);
  }

  const auto &methodAnnotations = classAnnotation.getMethodAnnotations();
  const auto &methods = classType->methods();
  for (int i = 0, e = methods.size(); i != e; i++) {
    importMethod(methods[i], classTypeBody, methodAnnotations[i]);
  }

  createMlirOperationAtEnd(classTypeBody, "torch.class_type_terminator", loc);
}

void IValueImporter::importCompilationUnit(torch::jit::CompilationUnit *cu) {
  if (compilationUnit == nullptr) {
    compilationUnit = cu;
  } else {
    // All sorts of stuff is connected to the compilation unit, such as
    // c10::ClassType's (owned by the compilation unit), c10::FunctionType
    // (which holds a pointer to a torch::jit::Function in the compilation
    // unit), load-bearing symbol table names of functions, etc.
    //
    // It doesn't seem to be defined how multiple compilation units semantically
    // connect with each other, and it doesn't seem to happen either (though
    // structurally at the C++ level nothing prevents it), so make it an error.
    if (compilationUnit != cu) {
      throw std::invalid_argument(
          "found two compilation units while importing");
    }
    return;
  }

  for (torch::jit::Function *function : cu->get_functions()) {
    // Useful for debugging errors in free functions that end up being
    // unused. These can be missing when round-tripping through the on-disk
    // format, even though they still cause import issues when importing
    // through the larger Python session where they originate.
    // std::cerr << "NAME: " << function->qualname().qualifiedName() << "\n";
    // std::cerr << *torch::jit::toGraphFunction(function).graph();
    MethodAnnotation *annotation =
        annotator.getMethodAnnotationForFunction(function);
    MlirOperation func = importJitFunctionAsFuncOp(
        context, function,
        [&](int argIndex) -> MlirAttribute {
          if (!annotation || !annotation->argAnnotations.has_value()) {
            return {nullptr};
          }
          c10::optional<std::vector<int64_t>> &maybeShape =
              annotation->argAnnotations.value()[argIndex].shape;
          c10::optional<c10::ScalarType> &maybeDtype =
              annotation->argAnnotations.value()[argIndex].dtype;
          bool hasValueSemantics =
              annotation->argAnnotations.value()[argIndex].hasValueSemantics;

          // TODO: Handle unranked tensors and tensors with unknown dtype (but
          // possibly known ranks/sizes).
          if (!maybeShape || !maybeDtype) {
            return {nullptr};
          }

          std::vector<int64_t> shape = *maybeShape;
          MlirType dtype = getMlirTypeForTorchScalarType(
              mlirLocationUnknownGet(context), *maybeDtype);
          MlirType typeBound;
          // `std::vector`'s `.data()` method can return nullptr when the
          // size is 0. This triggers the "nothing known about sizes" case in
          // the C API constructor, when we want the "we know we have 0 sizes"
          // case. So use a dummy data pointer.
          int64_t dummy;
          int64_t *shapeData = shape.size() == 0 ? &dummy : shape.data();
          if (hasValueSemantics) {
            typeBound = torchMlirTorchValueTensorTypeGet(context, shape.size(),
                                                         shapeData, dtype);
          } else {
            typeBound = torchMlirTorchNonValueTensorTypeGet(
                context, shape.size(), shapeData, dtype);
          }

          MlirNamedAttribute typeBoundAttr = toMlirNamedAttribute(
              "torch.type_bound", mlirTypeAttrGet(typeBound));
          return mlirDictionaryAttrGet(context, 1, &typeBoundAttr);
        },
        importOptions);
    // For IValue importing, the logical linkage structure of the module
    // is determined by the object graph.
    //
    // The functions' symbol names are thus irrelevant to the module's
    // externally visible characteristics, so mark them all as private.
    //
    // These functions may be referenced by the object graph, which can make
    // them reachable from the exernally visible characteristics of the module,
    // but they cannot be intrinsically externally visible.
    mlirOperationSetAttributeByName(
        func, toMlirStringRef("sym_visibility"),
        mlirStringAttrGet(context, toMlirStringRef("private")));
    mlirBlockInsertOwnedOperationBefore(
        importBlock, mlirBlockGetTerminator(importBlock), func);
  }
}

MlirValue torch_mlir::importIValue(c10::IValue ivalue, MlirBlock block,
                                   MlirContext context,
                                   ClassAnnotator &annotator,
                                   const ImportOptions &importOptions) {
  // When debugging module importing, it can be useful to dump as so:
  // if (ivalue.isModule())
  //   ivalue.toModule().dump(true, false, false);
  IValueImporter importer(block, context, annotator, importOptions);
  return importer.importIValue(ivalue);
}
