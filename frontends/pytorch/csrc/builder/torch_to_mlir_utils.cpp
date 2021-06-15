//===- ivalue_importer.cpp ------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "function_importer.h"
#include "ivalue_importer.h"

#include <unordered_map>

#include "mlir_utils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "npcomp-c/BasicpyTypes.h"
#include "npcomp-c/TorchTypes.h"

using namespace torch_mlir;

MlirType TypeMapper::mapFromTorchScalarType(c10::ScalarType scalarType) {
  auto type = rawMapFromTorchScalarType(scalarType);
  if (mlirTypeIsNull(type)) {
    std::stringstream message;
    message << "unsupported PyTorch scalar type: " << c10::toString(scalarType);
    throw std::invalid_argument(message.str());
  }
  return type;
}

MlirType TypeMapper::mapFromTorchScalarType(MlirLocation loc,
                                            c10::ScalarType scalarType) {
  auto type = rawMapFromTorchScalarType(scalarType);
  if (mlirTypeIsNull(type)) {
    std::stringstream message;
    message << "unsupported PyTorch scalar type: " << c10::toString(scalarType);
    mlirEmitError(loc, message.str().c_str());
  }
  return type;
}

MlirType TypeMapper::rawMapFromTorchScalarType(c10::ScalarType scalarType) {
  using c10::ScalarType;
  switch (scalarType) {
  case ScalarType::Byte:
    return mlirIntegerTypeUnsignedGet(context, 8);
  case ScalarType::Char:
    return mlirIntegerTypeSignedGet(context, 8);
  case ScalarType::Short:
    return mlirIntegerTypeSignedGet(context, 16);
  case ScalarType::Int:
    return mlirIntegerTypeSignedGet(context, 32);
  case ScalarType::Long:
    return mlirIntegerTypeSignedGet(context, 64);
  case ScalarType::Bool:
    return npcompBasicpyBoolTypeGet(context);
  case ScalarType::Double:
    return mlirF64TypeGet(context);
  case ScalarType::Float:
    return mlirF32TypeGet(context);
  case ScalarType::BFloat16:
    return mlirBF16TypeGet(context);
  case ScalarType::Half:
    return mlirF16TypeGet(context);
  case ScalarType::QInt8:
    return npcompTorchQInt8TypeGet(context);
  default: {
    return {nullptr};
  }
  }
}

// Types (such as `LinearPackedParamsBase`) implemented with the
// `torch::CustomClassHolder` mechanism described at
// https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html
// are modeled with ordinary c10::ClassType's, but require special handling
// for importing.
//
// These types correspond to c10::IValue's with `isCustomClass() == true`.
//
// Under the hood, Torch represents such "custom classes" using the
// "object" variant of c10::IValue and a class type with one slot holding a
// type-erased c10::intrusive_ptr to the custom type. One the side, it keeps
// a registry of custom classes which is used to implement `isCustomClass()`
// by checking names against the registry.
//
// There is no generic way to import custom classes (or their types), so we
// have to name match them here (and the relevant code in the ivalue
// importer) and create special IR constructs for them.
static MlirType mapCustomClassType(MlirContext context, MlirLocation loc,
                                   const c10::ClassTypePtr &classType) {
  // If the type is unnamed, it cannot be a custom class.
  if (!classType->name().has_value()) {
    return {nullptr};
  }
  std::string name = classType->name()->qualifiedName();
  // If the type is not stored in the custom class registry, it cannot be a
  // custom class.
  if (!torch::getCustomClass(name)) {
    return {nullptr};
  }

  // Individually handle the custom classes that we know about.
  if (name == "__torch__.torch.classes.quantized.LinearPackedParamsBase") {
    return npcompTorchLinearParamsTypeGet(context);
  }

  // At this point, we know that the type is indeed a custom class type, but
  // that we don't know how to specially import it. We cannot proceed, so emit a
  // diagnostic and halt compilation.
  std::stringstream message;
  message << "unable to import Torch CustomClass type '" << classType
          << "' to MLIR type";
  mlirEmitError(loc, message.str().c_str());
  throw mlir_diagnostic_emitted();
}

MlirType TypeMapper::mapFromTorchType(MlirLocation loc,
                                      const c10::TypePtr &torchType) {
  using c10::TypeKind;
  auto kind = torchType->kind();
  switch (kind) {
  case TypeKind::TensorType: {
    auto tensorType = torchType->cast<c10::TensorType>();
    // Element type.
    MlirType elementType = {nullptr};
    if (tensorType->scalarType()) {
      elementType = mapFromTorchScalarType(loc, *tensorType->scalarType());
      if (mlirTypeIsNull(elementType))
        return {nullptr};
    }
    // Sizes.
    auto &sizes = tensorType->symbolic_sizes();
    if (!sizes.rank()) {
      // Unranked.
      return npcompTorchNonValueTensorTypeGet(context,
                                              /*numSizes=*/0,
                                              /*optionalSizes=*/nullptr,
                                              /*optionalDtype=*/
                                              elementType);
    }
    // Ranked with possibly dynamic dims.
    auto &symbolicShape = tensorType->symbolic_sizes();
    std::vector<int64_t> dims;
    dims.resize(*sizes.rank());
    for (size_t i = 0; i < dims.size(); ++i) {
      auto shapeSymbol = symbolicShape[i];
      dims[i] = shapeSymbol.is_static() ? shapeSymbol.static_size() : -1;
    }
    return npcompTorchNonValueTensorTypeGet(context, dims.size(),
                                            /*optionalSizes=*/dims.data(),
                                            /*optionalDtype=*/
                                            elementType);
  }
  case TypeKind::ClassType: {
    const c10::ClassTypePtr &classType = torchType->cast<c10::ClassType>();
    MlirType customClassType = mapCustomClassType(context, loc, classType);
    if (!mlirTypeIsNull(customClassType)) {
      return customClassType;
    }
    auto maybeName = classType->name();
    std::string name = maybeName ? maybeName->qualifiedName() : "unnamed class";
    return npcompTorchNnModuleTypeGet(context, toMlirStringRef(name));
  }
  case TypeKind::FloatType: {
    return mlirF64TypeGet(context);
  }
  case TypeKind::OptionalType: {
    return npcompTorchOptionalTypeGet(mapFromTorchType(
        loc, torchType->cast<c10::OptionalType>()->getElementType()));
  }
  case TypeKind::IntType: {
    return mlirIntegerTypeGet(context, 64);
  }
  case TypeKind::NoneType: {
    return npcompTorchNoneTypeGet(context);
  }
  case TypeKind::BoolType: {
    return npcompBasicpyBoolTypeGet(context);
  }
  case TypeKind::ListType: {
    return npcompTorchListTypeGet(mapFromTorchType(
        loc, torchType->cast<c10::ListType>()->getElementType()));
  }
  case TypeKind::TupleType: {
    std::vector<MlirType> containedTypes;
    for (const c10::TypePtr &type :
         torchType->cast<c10::TupleType>()->containedTypes()) {
      containedTypes.push_back(mapFromTorchType(loc, type));
    }
    return npcompTorchTupleTypeGet(context, containedTypes.size(),
                                   containedTypes.data());
  }
  case TypeKind::StringType: {
    return npcompTorchStringTypeGet(context);
  }
  case TypeKind::DeviceObjType: {
    return npcompTorchDeviceTypeGet(context);
  }
  default: {
    std::stringstream message;
    message << "unable to map Torch type '" << *torchType << "' to MLIR type";
    mlirEmitError(loc, message.str().c_str());
    return {nullptr};
  }
  }
}

MlirType TypeMapper::forwardTensorToType(at::Tensor tensor) {
  if (!tensor.defined()) {
    // Undefined tensors are equivalent to None.
    // This may need to be re-evaluated at some point.
    return npcompTorchNoneTypeGet(context);
  }

  MlirType elementType = mapFromTorchScalarType(tensor.scalar_type());
  // TODO: Decide when it is necessary to take strides into account. Right now,
  // just erase them and let the compiler decide.

  auto sizes = tensor.sizes();
  return npcompTorchNonValueTensorTypeGet(context, sizes.size(), sizes.data(),
                                          elementType);
}

MlirType
torch_mlir::getFunctionTypeFromSchema(MlirContext context,
                                      const c10::FunctionSchema &schema) {
  MlirLocation loc = mlirLocationUnknownGet(context);
  TypeMapper typeMapper(context);
  auto mapType = [&](const c10::TypePtr &torchType) {
    MlirType type = typeMapper.mapFromTorchType(loc, torchType);
    if (mlirTypeIsNull(type)) {
      std::stringstream msg;
      msg << "unsupported type in function schema: '"
              << c10::toString(torchType) << "'";
      throw std::invalid_argument(msg.str());
    }
    return type;
  };

  std::vector<MlirType> inputTypes =
      c10::fmap(schema.arguments(),
                [&](const c10::Argument &arg) { return mapType(arg.type()); });
  std::vector<MlirType> outputTypes =
      c10::fmap(schema.returns(),
                [&](const c10::Argument &arg) { return mapType(arg.type()); });
  return mlirFunctionTypeGet(context, inputTypes.size(), inputTypes.data(),
                             outputTypes.size(), outputTypes.data());
}

MlirAttribute torch_mlir::convertTensorToMlirElementsAttr(at::Tensor tensor,
                                                          MlirLocation loc) {
  MlirContext context = mlirLocationGetContext(loc);
  TypeMapper typeMapper(context);
  using at::ScalarType;

  auto throwUnsupportedTensorError = [&]() {
    std::stringstream msg;
    msg << "Unsupported import tensor type: " << tensor;
    throw std::invalid_argument(msg.str());
  };

  // Get a C-contiguous form as we can bulk-load that into a DenseElementsAttr.
  if (!tensor.is_contiguous())
    tensor = tensor.contiguous();

  // The flat number of bytes throws an exception for tensors that are not
  // dense and accessible as such.
  at::checkLayout(at::CheckedFrom("accessing contiguous"), tensor,
                  c10::Layout::Strided);

  // Construct the ShapedType.
  MlirType elementType;
  if (tensor.scalar_type() == ScalarType::Bool) {
    // Bool is a special case. When used as an element type, it must be i1.
    // The generalized (non-Tensor) conversion, assumes that Bool is the
    // Basicpy bool type.
    elementType = mlirIntegerTypeGet(context, 1);
  } else if (tensor.scalar_type() == ScalarType::QInt8) {
    // This function returns the underlying integer representation of the tensor
    // as an elements attr. That underlying representation is of type si8
    // for a torch.qint8 tensor.
    // Caller code is responsible for materializing the proper op that
    // incorporates the quantization scheme to create a tensor of `!torch.qint8`
    // element type.
    elementType = mlirIntegerTypeSignedGet(context, 8);
  } else {
    elementType = typeMapper.mapFromTorchScalarType(tensor.scalar_type());
  }
  std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
  MlirType shapedType = mlirRankedTensorTypeGetChecked(
      loc, shape.size(), shape.data(), elementType, {nullptr});
  if (mlirTypeIsNull(shapedType)) {
    throwUnsupportedTensorError();
  }

  // Import DenseElementsAttr data.
  // TODO: Support bool tensors.
  // TODO: More import formats in C-API.
  auto numElements = tensor.numel();
  auto tensorData = tensor.data_ptr();
  switch (tensor.scalar_type()) {
  case ScalarType::Int:
    return mlirDenseElementsAttrInt32Get(
        shapedType, numElements, static_cast<const int32_t *>(tensorData));
    break;
  case ScalarType::Long:
    return mlirDenseElementsAttrInt64Get(
        shapedType, numElements, static_cast<const int64_t *>(tensorData));
    break;
  case ScalarType::Float:
    return mlirDenseElementsAttrFloatGet(
        shapedType, numElements, static_cast<const float *>(tensorData));
    break;
  case ScalarType::Double:
    return mlirDenseElementsAttrDoubleGet(
        shapedType, numElements, static_cast<const double *>(tensorData));
    break;
  case ScalarType::Bool:
    // TODO: Add a test specifically for bool and ensure consistency between
    // storage format and load format
    // (https://github.com/llvm/mlir-npcomp/issues/100).
    return mlirDenseElementsAttrBoolGet(shapedType, numElements,
                                        static_cast<const int *>(tensorData));
    break;
  case ScalarType::QInt8:
    return mlirDenseElementsAttrInt8Get(
        shapedType, numElements, static_cast<const int8_t *>(tensorData));
  default:
    throwUnsupportedTensorError();
  }
  return {nullptr}; // Unreachable.
}

MlirAttribute torch_mlir::importAttribute(MlirLocation loc,
                                          torch::jit::Node *node,
                                          c10::Symbol symbol) {
  MlirContext context = mlirLocationGetContext(loc);
  auto kind = node->kindOf(symbol);
  switch (kind) {
  case torch::jit::AttributeKind::i:
    // TODO: This should be a signed int once we have a constant op that can
    // do that.
    return mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), node->i(symbol));
  case torch::jit::AttributeKind::f:
    return mlirFloatAttrDoubleGet(context, mlirF64TypeGet(context),
                                  node->f(symbol));
  case torch::jit::AttributeKind::s:
    return mlirStringAttrGet(context, toMlirStringRef(node->s(symbol)));
  case torch::jit::AttributeKind::t:
    return convertTensorToMlirElementsAttr(node->t(symbol), loc);
  default: {
    std::stringstream msg;
    msg << "unhandled: value attribute kind " << toString(kind);
    mlirEmitError(loc, msg.str().c_str());
    throw mlir_diagnostic_emitted();
  }
  }
}

MlirLocation torch_mlir::getMlirLocationFromNode(MlirContext context,
                                                 torch::jit::Node *node) {
  auto flc = node->sourceRange().file_line_col();
  if (flc) {
    const std::string &file = std::get<0>(*flc);
    int line = std::get<1>(*flc);
    int col = std::get<2>(*flc);
    return mlirLocationFileLineColGet(context, toMlirStringRef(file), line,
                                      col);
  }
  return mlirLocationUnknownGet(context);
}

std::vector<MlirType>
torch_mlir::getMlirTypesFromValues(MlirLocation loc,
                                   c10::ArrayRef<torch::jit::Value *> values) {
  TypeMapper typeMapper(mlirLocationGetContext(loc));
  std::vector<MlirType> ret;
  for (auto value : values) {
    MlirType t = typeMapper.mapFromTorchType(loc, value->type());
    if (mlirTypeIsNull(t))
      throw mlir_diagnostic_emitted("unsupported type");
    ret.push_back(t);
  }
  return ret;
}

std::vector<MlirValue>
torch_mlir::derefineValues(c10::ArrayRef<MlirValue> values,
                           c10::ArrayRef<MlirType> expectedTypes,
                           MlirLocation loc, MlirBlock appendToBlock) {
  std::vector<MlirValue> ret;
  assert(values.size() == expectedTypes.size());
  for (int i = 0, e = values.size(); i != e; i++) {
    MlirValue value = values[i];
    MlirType expectedType = expectedTypes[i];
    MlirType type = mlirValueGetType(value);
    if (mlirTypeEqual(expectedType, type)) {
      // No need to derefine.
      ret.push_back(value);
    } else {
      MlirOperation operation = createMlirOperationAtEnd(
          appendToBlock, "torch.derefine", loc, expectedType, value);
      ret.push_back(mlirOperationGetResult(operation, 0));
    }
  }
  return ret;
}

MlirOperation
torch_mlir::createOperationFromSchema(MlirBlock appendToBlock, MlirLocation loc,
                                      const c10::FunctionSchema &schema,
                                      c10::ArrayRef<MlirType> resultTypes,
                                      c10::ArrayRef<MlirValue> operands) {
  MlirContext context = mlirLocationGetContext(loc);

  // Munge the name into the appropriate MLIR operation name.
  // See torch_ods_gen.py:JitOperator for the logic used to construct the MLIR
  // op name from the schema. This logic must be kept in sync with that logic.
  std::string opNameSuffix = schema.name();
  auto separatorPosition = opNameSuffix.find_first_of("::");
  assert(separatorPosition != std::string::npos);
  opNameSuffix.replace(separatorPosition, 2, ".");
  const std::string &overloadName = schema.overload_name();
  if (!overloadName.empty()) {
    opNameSuffix = opNameSuffix + "." + overloadName;
  }
  std::string opName = "torch." + opNameSuffix;
  // If we have a registered op, use it!
  if (mlirContextIsRegisteredOperation(context, toMlirStringRef(opName))) {
    return createMlirOperationAtEnd(appendToBlock, opName, loc, resultTypes,
                                    operands);
  }
  // Oops, no registered op -- create an opaque wrapper so that import can
  // still succeed. This helps a common use case of filling out registered ops
  // support, where it is easier to iterate on an MLIR file with
  // unregistered ops in it than to rerun import repeatedly.
  // The alternative here would be to allow unregistered ops in the `torch`
  // dialect, but that has the following disadvantages:
  // - Makes the dialect overall less strict
  // - Makes it hard to see exactly which ops from a model are registered or
  //   not.
  return createMlirOperationAtEnd(
      appendToBlock, "torch.operator", loc, resultTypes, operands,
      toMlirNamedAttribute(
          "name", mlirStringAttrGet(context, toMlirStringRef(opNameSuffix))));
}
