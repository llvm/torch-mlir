//===- ir.cpp ---------------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/ATen/ATenDialect.h"

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "ir.h"

#include <c10/util/ArrayRef.h>

#define DEBUG_TYPE "torch_mlir"

using namespace mlir;

namespace torch_mlir {
namespace ir {

void RegisterAtenIR() {
  mlir::registerDialect<mlir::NPCOMP::aten::ATenDialect>();
  mlir::registerDialect<mlir::StandardOpsDialect>();
  mlir::registerDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerDialect<mlir::AffineDialect>();
  mlir::registerDialect<mlir::scf::SCFDialect>();
}

std::vector<int64_t> Value::sizes() const { return node->sizes(index); }

std::vector<int64_t> Value::strides() const { return node->strides(index); }

Node::Node(OpKind op) : op_(std::move(op)) {}

Node::Node(OpKind op, OpList operands, std::vector<int64_t> sizes)
    : op_(std::move(op)), operands_(std::move(operands)) {
  for (auto &oper : operands)
    operands_.push_back(oper);
  sizes_[0] = sizes;
}

Node::Node(OpKind op, OpList operands, at::IntArrayRef sizes)
    : op_(std::move(op)), operands_(std::move(operands)) {
  for (auto &oper : operands)
    operands_.push_back(oper);
  for (auto &size : sizes)
    sizes_[0].push_back(size);
}

std::vector<int64_t> Node::strides(std::vector<int64_t> sz) const {
  auto dim = sz.size();
  std::vector<int64_t> ret(dim);
  int64_t n = 1;
  for (int i = dim - 1; i >= 0; i--) {
    ret[i] = n;
    n = n * sz[i];
  }
  return ret;
}

mlir::Operation *
Node::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
              mlir::MLIRContext &context,
              std::map<const ir::Value, mlir::Value> &symbolTable) {
  std::cout << "unsupported node type in Node::genMLIR" << op() << std::endl;
  assert(0);
}

mlir::Operation *
ConstantNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                      mlir::MLIRContext &context,
                      std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  // the type of the mlir value
  mlir::Type mlirTy;

  // the attribuite attached to the mlir value
  std::vector<mlir::NamedAttribute> attrs;
  auto typeId = mlir::Identifier::get("type", &context);
  auto valueId = mlir::Identifier::get("value", &context);

  if (scalar) {
    if (scalar->isIntegral(false)) {
      mlirTy = mlir::IntegerType::get(32, &context);
      attrs.emplace_back(typeId, mlir::StringAttr::get("i32", &context));
      attrs.emplace_back(valueId,
                         mlir::IntegerAttr::get(mlirTy, scalar->to<int32_t>()));
    } else if (scalar->isFloatingPoint()) {
      mlirTy = mlir::FloatType::getF32(&context);
      attrs.emplace_back(typeId, mlir::StringAttr::get("f32", &context));
      attrs.emplace_back(valueId,
                         mlir::FloatAttr::get(mlirTy, scalar->to<float>()));
    } else if (scalar->isBoolean()) {
      mlirTy = mlir::IntegerType::get(1, &context);
      attrs.emplace_back(typeId, mlir::StringAttr::get("bool", &context));
      attrs.emplace_back(
          valueId, mlir::IntegerAttr::get(mlirTy, (int)scalar->to<bool>()));
    } else {
      assert(0 && "unhandled scalar type in ir::ConstantNode");
    }
  } else if (array.size() > 0) {
    auto iTy = mlir::IntegerType::get(32, &context);
    mlirTy = mlir::NPCOMP::aten::ATenListType::get(iTy);
    auto vecTy =
        mlir::VectorType::get(llvm::ArrayRef<int64_t>(array.size()), iTy);
    attrs.emplace_back(typeId, mlir::StringAttr::get("List[i32]", &context));
    std::vector<int32_t> values;
    for (auto a : array)
      values.push_back((int32_t)a);
    attrs.emplace_back(
        valueId, DenseElementsAttr::get(vecTy, ArrayRef<int32_t>(values)));
  } else if (bool_) {
    mlirTy = mlir::IntegerType::get(1, &context);
    attrs.emplace_back(typeId, mlir::StringAttr::get("bool", &context));
    attrs.emplace_back(valueId, mlir::IntegerAttr::get(mlirTy, (int)*bool_));
  } else if (int_) {
    mlirTy = mlir::IntegerType::get(32, &context);
    attrs.emplace_back(typeId, mlir::StringAttr::get("i32", &context));
    attrs.emplace_back(valueId, mlir::IntegerAttr::get(mlirTy, *int_));
  } else if (double_) {
    mlirTy = mlir::FloatType::getF64(&context);
    attrs.emplace_back(typeId, mlir::StringAttr::get("f64", &context));
    attrs.emplace_back(valueId, mlir::FloatAttr::get(mlirTy, *double_));
  } else if (float_) {
    mlirTy = mlir::FloatType::getF32(&context);
    attrs.emplace_back(typeId, mlir::StringAttr::get("f32", &context));
    attrs.emplace_back(valueId, mlir::FloatAttr::get(mlirTy, *float_));
  } else {
    auto iTy = mlir::IntegerType::get(32, &context);
    mlirTy = mlir::NPCOMP::aten::ATenListType::get(iTy);
    auto vecTy =
        mlir::VectorType::get(llvm::ArrayRef<int64_t>(array.size()), iTy);
    attrs.emplace_back(typeId, mlir::StringAttr::get("List[i32]", &context));
    std::vector<int32_t> values;
    for (auto a : array)
      values.push_back((int32_t)a);
    attrs.emplace_back(
        valueId, DenseElementsAttr::get(vecTy, ArrayRef<int32_t>(values)));
  }
  // else {
  //  assert(0 && "unhandled type in ir::ConstantNode");
  // }
  return builder->create<mlir::NPCOMP::aten::ConstantOp>(
      loc, ArrayRef<mlir::Type>{mlirTy}, ArrayRef<mlir::Value>{}, attrs);
}

mlir::Operation *AdaptiveAvgPool2dNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::_adaptive_avg_pool2d"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type selfTy = self.getType();
  auto elemTy = ((mlir::ShapedType *)&selfTy)->getElementType();

  mlir::Type mlirTy = mlir::RankedTensorType::get(sizes(), elemTy);

  return builder->create<mlir::NPCOMP::aten::AdaptiveAvgPool2dOp>(
      loc, mlirTy, self, symbolTable[operand(1)]);
}

mlir::Operation *AdaptiveAvgPool2dBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::_adaptive_avg_pool2d_backward"));

  mlir::Value self = symbolTable[operand(1)];
  mlir::Type selfTy = self.getType();
  auto elemTy = ((mlir::ShapedType *)&selfTy)->getElementType();

  mlir::Type mlirTy = mlir::RankedTensorType::get(sizes(), elemTy);

  return builder->create<mlir::NPCOMP::aten::AdaptiveAvgPool2dBackwardOp>(
      loc, mlirTy, symbolTable[operand(1)], self);
}

mlir::Operation *
AddNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                 mlir::MLIRContext &context,
                 std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::add"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::AddOp>(loc, retTy, arg0, arg1,
                                                    arg2);
}

mlir::Operation *
AddInPlaceNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                        mlir::MLIRContext &context,
                        std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::add_"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::AddUnderOp>(loc, retTy, arg0, arg1,
                                                         arg2);
}

mlir::Operation *
AddmmNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                   mlir::MLIRContext &context,
                   std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::addmm"));

  mlir::Type tensorTy = symbolTable[operand(0)].getType();
  auto elemTy = ((mlir::ShapedType *)&tensorTy)->getElementType();
  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];
  auto arg3 = symbolTable[operand(3)];
  auto arg4 = symbolTable[operand(4)];

  return builder->create<mlir::NPCOMP::aten::AddmmOp>(loc, retTy, arg0, arg1,
                                                      arg2, arg3, arg4);
}

mlir::Operation *
AsStridedNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                       mlir::MLIRContext &context,
                       std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::as_strided"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type selfTy = self.getType();
  auto elemTy = ((mlir::ShapedType *)&selfTy)->getElementType();

  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  return builder->create<mlir::NPCOMP::aten::AsStridedOp>(
      loc, retTy, self, symbolTable[operand(1)], symbolTable[operand(2)]);
}

std::vector<int64_t> AsStridedNode::sizes() const {

  auto input_size = operand(0).sizes();

  // XXX
  // std::cout << "TODO: handle stride!\n";

  LLVM_DEBUG(llvm::dbgs() << "as strided input size: ");
  for (int64_t n : input_size)
    LLVM_DEBUG(llvm::dbgs() << n << " ");
  LLVM_DEBUG(llvm::dbgs() << "\n");

  LLVM_DEBUG(llvm::dbgs() << "view size: ");
  for (int64_t n : size)
    LLVM_DEBUG(llvm::dbgs() << n << " ");
  LLVM_DEBUG(llvm::dbgs() << "\n");

  std::vector<int64_t> output_size;
  output_size.resize(size.size());

  int64_t numel = 1;
  for (int64_t n : input_size)
    numel *= n;

  int64_t numel_view = 1;
  for (int i = size.size() - 1; i >= 0; i--) {
    int64_t n = size[i];
    if (n == -1)
      n = numel / numel_view;
    else if (n <= 0)
      assert(n && "unhandled size in AsStridedNode::sizes()");
    output_size[i] = n;
    numel_view *= n;
  }

  LLVM_DEBUG(llvm::dbgs() << "output size: ");
  for (int64_t n : output_size)
    LLVM_DEBUG(llvm::dbgs() << n << " ");
  LLVM_DEBUG(llvm::dbgs() << "\n");

  assert(numel == numel_view && "bad size in AsStridedNode::sizes()");
  return output_size;
}

mlir::Operation *
BatchNormNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                       mlir::MLIRContext &context,
                       std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::native_batch_norm"));

  mlir::Type tensorTy = symbolTable[operand(0)].getType();
  auto elemTy = ((mlir::ShapedType *)&tensorTy)->getElementType();
  mlir::Type mlirTy = mlir::RankedTensorType::get(sizes(), elemTy);

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::NativeBatchNormOp>(
      loc,
      ArrayRef<mlir::Type>(
          std::vector<mlir::Type>{mlirTy, symbolTable[operand(2)].getType(),
                                  symbolTable[operand(3)].getType()}),
      ArrayRef<mlir::Value>(mlirOperands), attrs);
}

mlir::Operation *BatchNormBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::native_batch_norm_backward"));

  mlir::TensorType tensorTy =
      symbolTable[operand(0)].getType().cast<TensorType>();
  mlir::Type elemTy = tensorTy.getElementType();
  mlir::Type ret0Ty = mlir::RankedTensorType::get(sizes(0), elemTy);
  mlir::Type ret1Ty = mlir::RankedTensorType::get(sizes(1), elemTy);
  mlir::Type ret2Ty = mlir::RankedTensorType::get(sizes(2), elemTy);

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::NativeBatchNormBackwardOp>(
      loc,
      ArrayRef<mlir::Type>(std::vector<mlir::Type>{ret0Ty, ret1Ty, ret2Ty}),
      ArrayRef<mlir::Value>(mlirOperands), attrs);
}

std::vector<int64_t> BatchNormBackwardNode::sizes(size_t i) const {
  if (i == 0)
    return operand(0).sizes();
  if (i == 1)
    return {operand(1).sizes()[1]};
  if (i == 2)
    return {operand(1).sizes()[1]};

  assert(0 && "bad operand index");
}

mlir::Operation *
Conv2dNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                    mlir::MLIRContext &context,
                    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::_convolution"));

  mlir::Type tensorTy = symbolTable[operand(0)].getType();
  auto elemTy = ((mlir::ShapedType *)&tensorTy)->getElementType();
  mlir::Type mlirTy = mlir::RankedTensorType::get(sizes(), elemTy);

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::ConvolutionOverrideableOp>(
      loc, ArrayRef<mlir::Type>{mlirTy}, ArrayRef<mlir::Value>(mlirOperands),
      attrs);
}

std::vector<int64_t> Conv2dNode::sizes() const {
  auto isize = operand(0).sizes();
  auto wsize = operand(1).sizes();
  int64_t osize0 = isize[0];
  int64_t osize1 = wsize[0];
  int64_t osize2 = 1 + ((isize[2] - wsize[2] + 2 * padding[0]) / stride[0]);
  int64_t osize3 = 1 + ((isize[3] - wsize[3] + 2 * padding[1]) / stride[1]);

  std::vector<int64_t> osize{osize0, osize1, osize2, osize3};
  return osize;
}

mlir::Operation *Conv2dBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::_convolution_backward"));

  mlir::Type tensorTy = symbolTable[operand(0)].getType();
  auto elemTy = ((mlir::ShapedType *)&tensorTy)->getElementType();
  mlir::Type retTy0 = mlir::RankedTensorType::get(sizes(0), elemTy);
  mlir::Type retTy1 = mlir::RankedTensorType::get(sizes(1), elemTy);
  mlir::Type retTy2 = mlir::RankedTensorType::get(sizes(2), elemTy);

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::ConvolutionBackwardOverrideableOp>(
      loc, ArrayRef<mlir::Type>{retTy0, retTy1, retTy2},
      ArrayRef<mlir::Value>(mlirOperands), attrs);
}

std::vector<int64_t> Conv2dBackwardNode::sizes(size_t index) const {
  if (index == 0)
    return operand(1).sizes();
  if (index == 1)
    return operand(2).sizes();
  else if (index == 2)
    return {operand(2).sizes()[0]};
  else
    assert(0 && "bad index");
}

mlir::Operation *
DivNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                 mlir::MLIRContext &context,
                 std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::div"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::DivOp>(loc, retTy, arg0, arg1);
}

mlir::Operation *
DivInPlaceNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                        mlir::MLIRContext &context,
                        std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::div_"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::DivUnderOp>(loc, retTy, arg0,
                                                         arg1);
}

mlir::Operation *
ExpandNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                    mlir::MLIRContext &context,
                    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::expand"));

  mlir::Value input = symbolTable[operand(0)];
  mlir::Type elemTy = input.getType().cast<TensorType>().getElementType();

  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  auto size = symbolTable[operand(1)];
  auto implicit = symbolTable[operand(2)];

  return builder->create<mlir::NPCOMP::aten::ExpandOp>(loc, retTy, input, size,
                                                       implicit);
}

mlir::Operation *
GatherNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                    mlir::MLIRContext &context,
                    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::gather"));

  mlir::Value input = symbolTable[operand(0)];
  mlir::Type elemTy = input.getType().cast<TensorType>().getElementType();

  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  auto dim = symbolTable[operand(1)];
  auto index = symbolTable[operand(2)];
  auto sparse_grad = symbolTable[operand(3)];

  return builder->create<mlir::NPCOMP::aten::GatherOp>(loc, retTy, input, dim,
                                                       index, sparse_grad);
}

mlir::Operation *
HardtanhNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                      mlir::MLIRContext &context,
                      std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::hardtanh"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::HardtanhOp>(loc, retTy, arg0, arg1,
                                                         arg2);
}

mlir::Operation *HardtanhInPlaceNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::hardtanh_"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::HardtanhUnderOp>(loc, retTy, arg0,
                                                              arg1, arg2);
}

mlir::Operation *HardtanhBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::hardtanh_backward"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];
  auto arg3 = symbolTable[operand(3)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::HardtanhBackwardOp>(
      loc, retTy, arg0, arg1, arg2, arg3);
}

mlir::Operation *
LogSoftmaxNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                        mlir::MLIRContext &context,
                        std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::_log_softmax"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type selfTy = self.getType();
  auto elemTy = ((mlir::ShapedType *)&selfTy)->getElementType();

  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  auto dim = symbolTable[operand(1)];
  auto half_to_float = symbolTable[operand(2)];

  return builder->create<mlir::NPCOMP::aten::LogSoftmaxOp>(loc, retTy, self,
                                                           dim, half_to_float);
}

mlir::Operation *LogSoftmaxBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::_log_softmax_backward_data"));

  mlir::Value arg0 = symbolTable[operand(0)];
  mlir::Value arg1 = symbolTable[operand(1)];
  mlir::Value arg2 = symbolTable[operand(2)];
  mlir::Value arg3 = symbolTable[operand(3)];

  mlir::Type retTy = arg1.getType();

  return builder->create<mlir::NPCOMP::aten::LogSoftmaxBackwardDataOp>(
      loc, retTy, arg0, arg1, arg2, arg3);
}

mlir::Operation *MaxPool2dWithIndicesNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::max_pool2d_with_indices"));

  mlir::Type tensorTy = symbolTable[operand(0)].getType();
  auto elemTy = ((mlir::ShapedType *)&tensorTy)->getElementType();
  mlir::Type retTy = mlir::RankedTensorType::get(sizes(0), elemTy);
  mlir::Type idxTy = mlir::RankedTensorType::get(
      sizes(0), mlir::IntegerType::get(64, &context));

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::MaxPool2dWithIndicesOp>(
      loc, ArrayRef<mlir::Type>{retTy, idxTy},
      ArrayRef<mlir::Value>(mlirOperands), attrs);
}

std::vector<int64_t> MaxPool2dWithIndicesNode::sizes(size_t index) const {
  auto isize = operand(0).sizes();
  int64_t osize0 = isize[0];
  int64_t osize1 = isize[1];
  // stride can be empty. the default is kernel_size
  int64_t stride0 = stride.size() == 2 ? stride[0] : kernel_size[0];
  int64_t stride1 = stride.size() == 2 ? stride[1] : kernel_size[1];
  int64_t osize2 = 1 + ((isize[2] - kernel_size[0] + 2 * padding[0]) / stride0);
  int64_t osize3 = 1 + ((isize[3] - kernel_size[1] + 2 * padding[1]) / stride1);

  std::vector<int64_t> osize{osize0, osize1, osize2, osize3};
  return osize;
}

mlir::Operation *MaxPool2dWithIndicesBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::max_pool2d_with_indices_backward"));

  mlir::Type retTy = symbolTable[operand(1)].getType();

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::MaxPool2dWithIndicesBackwardOp>(
      loc, ArrayRef<mlir::Type>{retTy}, ArrayRef<mlir::Value>(mlirOperands),
      attrs);
}

mlir::Operation *
MeanNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                  mlir::MLIRContext &context,
                  std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::mean"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type selfTy = self.getType();
  auto elemTy = ((mlir::ShapedType *)&selfTy)->getElementType();

  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  return builder->create<mlir::NPCOMP::aten::MeanOp>(loc, retTy, self);
}

std::vector<int64_t> MeanNode::sizes() const {

  std::vector<int64_t> input_size = operand(0).sizes();
  std::vector<int64_t> output_dims;
  std::vector<int64_t> result;

  if (dim.size() == 0)
    return {1};

  for (int64_t n : input_size) {
    output_dims.push_back(n);
  }
  for (int64_t d : dim) {
    if (d < 0)
      d += output_dims.size();

    if (keepdim)
      output_dims[d] = 1;
    else
      output_dims[d] = 0;
  }
  for (int64_t n : output_dims) {
    if (n > 0)
      result.push_back(n);
  }
  return result;
}

mlir::Operation *
MMNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                mlir::MLIRContext &context,
                std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::mm"));

  mlir::Type tensorTy = symbolTable[operand(0)].getType();
  auto elemTy = ((mlir::ShapedType *)&tensorTy)->getElementType();
  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];

  return builder->create<mlir::NPCOMP::aten::MmOp>(loc, retTy, arg0, arg1);
}

mlir::Operation *
MulNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                 mlir::MLIRContext &context,
                 std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::mul"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::MulOp>(loc, retTy, arg0, arg1);
}

mlir::Operation *
MulInPlaceNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                        mlir::MLIRContext &context,
                        std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::mul_"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::MulUnderOp>(loc, retTy, arg0,
                                                         arg1);
}

mlir::Operation *
NegNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                 mlir::MLIRContext &context,
                 std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);
  assert(op() == ir::OpKind::Get("aten::neg"));

  auto arg0 = symbolTable[operand(0)];
  return builder->create<mlir::NPCOMP::aten::NegOp>(loc, arg0.getType(), arg0);
}

mlir::Operation *NllLoss2dForwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::nll_loss2d_forward"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto input = symbolTable[operand(0)];

  mlir::TensorType tensorTy = input.getType().cast<TensorType>();
  mlir::Type elemTy = tensorTy.getElementType();
  mlir::Type retTy = mlir::RankedTensorType::get(1, elemTy);

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::NllLoss2dForwardOp>(
      loc, ArrayRef<mlir::Type>{retTy, retTy},
      ArrayRef<mlir::Value>(mlirOperands), attrs);
}

mlir::Operation *NllLoss2dBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::nll_loss2d_backward"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto input = symbolTable[operand(1)];

  mlir::Type retTy = input.getType();

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::NllLoss2dBackwardOp>(
      loc, ArrayRef<mlir::Type>{retTy}, ArrayRef<mlir::Value>(mlirOperands),
      attrs);
}

mlir::Operation *NllLossForwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::nll_loss_forward"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto input = symbolTable[operand(0)];

  mlir::TensorType tensorTy = input.getType().cast<TensorType>();
  mlir::Type elemTy = tensorTy.getElementType();
  mlir::Type retTy = mlir::RankedTensorType::get(1, elemTy);

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::NllLossForwardOp>(
      loc, ArrayRef<mlir::Type>{retTy, retTy},
      ArrayRef<mlir::Value>(mlirOperands), attrs);
}

mlir::Operation *NllLossBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::nll_loss_backward"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto input = symbolTable[operand(1)];

  mlir::Type retTy = input.getType();

  std::vector<mlir::NamedAttribute> attrs;
  std::vector<mlir::Value> mlirOperands;

  for (auto &op : operands())
    mlirOperands.push_back(symbolTable[op]);

  return builder->create<mlir::NPCOMP::aten::NllLossBackwardOp>(
      loc, ArrayRef<mlir::Type>{retTy}, ArrayRef<mlir::Value>(mlirOperands),
      attrs);
}

mlir::Operation *
SumNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                 mlir::MLIRContext &context,
                 std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::sum"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type selfTy = self.getType();
  auto elemTy = ((mlir::ShapedType *)&selfTy)->getElementType();

  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  auto dim = symbolTable[operand(1)];
  auto keepdim = symbolTable[operand(2)];

  return builder->create<mlir::NPCOMP::aten::SumOp>(loc, retTy, self, dim,
                                                    keepdim);
}

std::vector<int64_t> SumNode::sizes() const {

  std::vector<int64_t> input_size = operand(0).sizes();
  std::vector<int64_t> output_dims;
  std::vector<int64_t> result;

  for (int64_t n : input_size) {
    output_dims.push_back(n);
  }
  for (int64_t d : dim) {
    if (d < 0)
      d += output_dims.size();

    if (keepdim)
      output_dims[d] = 1;
    else
      output_dims[d] = 0;
  }
  for (int64_t n : output_dims) {
    if (n > 0)
      result.push_back(n);
  }

  return result;
}

mlir::Operation *
ReLUNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                  mlir::MLIRContext &context,
                  std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::relu"));

  auto input = symbolTable[operand(0)];
  return builder->create<mlir::NPCOMP::aten::ReluOp>(loc, input.getType(),
                                                     input);
}

mlir::Operation *
ReLUInPlaceNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                         mlir::MLIRContext &context,
                         std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::relu_"));

  auto input = symbolTable[operand(0)];
  return builder->create<mlir::NPCOMP::aten::ReluUnderOp>(loc, input.getType(),
                                                          input);
}

mlir::Operation *
SizeNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                  mlir::MLIRContext &context,
                  std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::size"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type tTy = self.getType().cast<TensorType>();
  mlir::Type retTy = mlir::IntegerType::get(32, &context);
  std::vector<mlir::NamedAttribute> attrs;
  auto typeId = mlir::Identifier::get("type", &context);
  auto valueId = mlir::Identifier::get("value", &context);
  attrs.emplace_back(typeId, mlir::StringAttr::get("i32", &context));
  attrs.emplace_back(valueId, mlir::IntegerAttr::get(retTy, sizes()[dim]));
  return builder->create<mlir::NPCOMP::aten::ConstantOp>(
      loc, ArrayRef<mlir::Type>{retTy}, ArrayRef<mlir::Value>{}, attrs);
}

mlir::Operation *
SqueezeNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                     mlir::MLIRContext &context,
                     std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::squeeze"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type elemTy = self.getType().cast<TensorType>().getElementType();
  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  return builder->create<mlir::NPCOMP::aten::SqueezeOp>(
      loc, retTy, self, symbolTable[operand(1)]);
}

std::vector<int64_t> SqueezeNode::sizes() const {
  std::vector<int64_t> input_size = operand(0).sizes();
  std::vector<int64_t> output_size;

  int input_dim = input_size.size();
  int arg_dim = dim;
  assert(arg_dim <= input_dim + 1);
  assert(arg_dim >= -input_dim - 1);

  if (arg_dim < 0)
    arg_dim = arg_dim + input_dim + 1;

  int i = 1;
  for (int64_t n : input_size) {
    if (i++ == dim && n == 1)
      continue;
    output_size.push_back(n);
  }
  return output_size;
}

mlir::Operation *
SubNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                 mlir::MLIRContext &context,
                 std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::sub"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::SubOp>(loc, retTy, arg0, arg1,
                                                    arg2);
}

mlir::Operation *
SubInPlaceNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                        mlir::MLIRContext &context,
                        std::map<const ir::Value, mlir::Value> &symbolTable) {
  assert(op() == ir::OpKind::Get("aten::sub_"));

  auto loc = mlir::UnknownLoc::get(&context);

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];
  auto retTy = arg0.getType();

  return builder->create<mlir::NPCOMP::aten::SubUnderOp>(loc, retTy, arg0, arg1,
                                                         arg2);
}

mlir::Operation *ThresholdBackwardNode::genMLIR(
    std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
    std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::threshold_backward"));

  auto arg0 = symbolTable[operand(0)];
  auto arg1 = symbolTable[operand(1)];
  auto arg2 = symbolTable[operand(2)];

  return builder->create<mlir::NPCOMP::aten::ThresholdBackwardOp>(
      loc, arg0.getType(), arg0, arg1, arg2);
}

mlir::Operation *
TransposeNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                       mlir::MLIRContext &context,
                       std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::t"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type elemTy = self.getType().cast<TensorType>().getElementType();
  mlir::Type mlirTy = mlir::RankedTensorType::get(sizes(), elemTy);

  return builder->create<mlir::NPCOMP::aten::TOp>(loc, mlirTy, self);
}

mlir::Operation *
UnsqueezeNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                       mlir::MLIRContext &context,
                       std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::unsqueeze"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type elemTy = self.getType().cast<TensorType>().getElementType();
  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  return builder->create<mlir::NPCOMP::aten::UnsqueezeOp>(
      loc, retTy, self, symbolTable[operand(1)]);
}

std::vector<int64_t> UnsqueezeNode::sizes() const {
  std::vector<int64_t> input_size = operand(0).sizes();
  std::vector<int64_t> output_size;

  int input_dim = input_size.size();
  int arg_dim = dim;
  assert(arg_dim <= input_dim + 1);
  assert(arg_dim >= -input_dim - 1);

  if (arg_dim < 0)
    arg_dim = arg_dim + input_dim + 1;

  int i = 1;
  for (int64_t n : input_size) {
    if (i++ == dim)
      output_size.push_back(1);
    output_size.push_back(n);
  }
  return output_size;
}

mlir::Operation *
ViewNode::genMLIR(std::unique_ptr<mlir::OpBuilder> &builder,
                  mlir::MLIRContext &context,
                  std::map<const ir::Value, mlir::Value> &symbolTable) {
  auto loc = mlir::UnknownLoc::get(&context);

  assert(op() == ir::OpKind::Get("aten::view"));

  mlir::Value self = symbolTable[operand(0)];
  mlir::Type elemTy = self.getType().cast<TensorType>().getElementType();
  mlir::Type retTy = mlir::RankedTensorType::get(sizes(), elemTy);

  return builder->create<mlir::NPCOMP::aten::ViewOp>(loc, retTy, self,
                                                     symbolTable[operand(1)]);
}

std::vector<int64_t> ViewNode::sizes() const {

  auto input_size = operand(0).sizes();

#if 0
  std::cout << "view input size: ";
  for (int64_t n : input_size)
    std::cout << n << " ";
  std::cout << std::endl;

  std::cout << "view size: ";
  for (int64_t n : view_size)
    std::cout << n << " ";
  std::cout << std::endl;
#endif

  std::vector<int64_t> output_size;
  output_size.resize(view_size.size());

  int64_t numel = 1;
  for (int64_t n : input_size)
    numel *= n;

  int64_t numel_view = 1;
  for (int i = view_size.size() - 1; i >= 0; i--) {
    int64_t n = view_size[i];
    if (n == -1)
      n = numel / numel_view;
    else if (n <= 0)
      assert(n && "unhandled size in ViewNode::sizes()");
    output_size[i] = n;
    numel_view *= n;
  }

  assert(numel == numel_view && "bad size in ViewNode::sizes()");
  return output_size;
}

} // namespace ir
} // namespace torch_mlir
