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
#include "npcomp-c/Types.h"

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
    // TODO: convert to mlirIntegerTypeUnsignedGet once supported.
    return mlirIntegerTypeGet(context, 8);
  case ScalarType::Char:
    return mlirIntegerTypeGet(context, 8);
  case ScalarType::Short:
    // TODO: convert to mlirIntegerTypeSignedGet once supported.
    return mlirIntegerTypeGet(context, 16);
  case ScalarType::Int:
    // TODO: convert to mlirIntegerTypeSignedGet once supported.
    return mlirIntegerTypeGet(context, 32);
  case ScalarType::Long:
    // TODO: convert to mlirIntegerTypeSignedGet once supported.
    return mlirIntegerTypeGet(context, 64);
  case ScalarType::Bool:
    return npcompBoolTypeGet(context);
  case ScalarType::Double:
    return mlirF64TypeGet(context);
  case ScalarType::Float:
    return mlirF32TypeGet(context);
  case ScalarType::BFloat16:
    return mlirBF16TypeGet(context);
  case ScalarType::Half:
    return mlirF16TypeGet(context);
  default: {
    return {nullptr};
  }
  }
}

MlirType TypeMapper::mapFromTorchType(MlirLocation loc,
                                      const c10::TypePtr &torchType) {
  using c10::TypeKind;
  auto kind = torchType->kind();
  switch (kind) {
  case TypeKind::TensorType: {
    auto tensorType = torchType->cast<c10::TensorType>();
    // Element type.
    MlirType elementType;
    if (tensorType->scalarType()) {
      elementType = mapFromTorchScalarType(loc, *tensorType->scalarType());
      if (mlirTypeIsNull(elementType))
        return {nullptr};
    } else {
      elementType = npcompAnyDtypeTypeGet(context);
    }
    // Sizes.
    auto &sizes = tensorType->symbolic_sizes();
    if (!sizes.rank()) {
      // Unranked.
      return npcompNdArrayTypeGetUnranked(elementType);
    }
    // Ranked with possibly dynamic dims.
    auto &symbolicShape = tensorType->symbolic_sizes();
    std::vector<int64_t> dims;
    dims.resize(*sizes.rank());
    for (size_t i = 0; i < dims.size(); ++i) {
      auto shapeSymbol = symbolicShape[i];
      dims[i] = shapeSymbol.is_static() ? shapeSymbol.static_size() : -1;
    }
    return npcompNdArrayTypeGetRanked(dims.size(), dims.data(), elementType);
  }
  case TypeKind::ClassType: {
    auto maybeName = torchType->cast<c10::ClassType>()->name();
    return npcompNnModuleTypeGet(
        context, toMlirStringRef(maybeName ? maybeName->qualifiedName()
                                           : "unnamed class"));
  }
  case TypeKind::FloatType: {
    return mlirF64TypeGet(context);
  }
  case TypeKind::OptionalType: {
    return npcompOptionalTypeGet(
        mapFromTorchType(
            loc, torchType->cast<c10::OptionalType>()->getElementType()));
  }
  case TypeKind::IntType: {
    return mlirIntegerTypeGet(context, 64);
  }
  case TypeKind::NoneType: {
    return npcompNoneTypeGet(context);
  }
  case TypeKind::BoolType: {
    return npcompBoolTypeGet(context);
  }
  case TypeKind::ListType: {
    // TODO: Don't lose the element type information.
    return npcompListTypeGet(context);
  }
  case TypeKind::TupleType: {
    // TODO: Don't lose the element type information.
    return npcompTupleTypeGet(context);
  }
  case TypeKind::StringType: {
    return npcompBytesTypeGet(context);
  }
  case TypeKind::DeviceObjType: {
    return npcompDeviceTypeGet(context);
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
    return npcompNoneTypeGet(context);
  }

  MlirType elementType = mapFromTorchScalarType(tensor.scalar_type());
  // TODO: Decide when it is necessary to take strides into account. Right now,
  // just erase them and let the compiler decide.

  auto sizes = tensor.sizes();
  return npcompNdArrayTypeGetRanked(sizes.size(), sizes.data(), elementType);
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

MlirAttribute torch_mlir::converTensorToMlirElementsAttr(at::Tensor tensor,
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
    return converTensorToMlirElementsAttr(node->t(symbol), loc);
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
