//===- ir.h -----------------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

// This file defines an intermediate IR generated from a pytorch model.
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class OpBuilder;
class Value;
class Operation;
class MLIRContext;
} // namespace mlir

#include <map>
#include <vector>

#include <ATen/Tensor.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/Scalar.h>
#include <c10/util/ArrayRef.h>

namespace torch_mlir {
namespace ir {

class Node;

void RegisterAtenIR();

using NodePtr = std::shared_ptr<Node>;

struct Value {
  Value() = default;
  Value(NodePtr node, size_t index = 0) : node(std::move(node)), index(index) {}

  operator bool() const { return node != nullptr; }

  bool operator==(const Value &rhs) const {
    return node == rhs.node && index == rhs.index;
  }

  bool operator<(const Value &rhs) const {
    if (node == rhs.node)
      return index < rhs.index;
    return node < rhs.node;
  }

  std::vector<int64_t> sizes() const;
  std::vector<int64_t> strides() const;

  NodePtr node;
  size_t index = 0;
};

struct OpKind {
  OpKind() = default;
  explicit OpKind(c10::Symbol op) : op(std::move(op)) {}

  bool operator==(const OpKind &rhs) const { return op == rhs.op; }
  bool operator!=(const OpKind &rhs) const { return !operator==(rhs); }
  bool operator<(const OpKind &rhs) const {
    return c10::unique_t(op) < c10::unique_t(rhs.op);
  }

  // size_t hash() const;

  std::string ToString() const { return op.toQualString(); }

  static OpKind Get(const std::string &name) {
    return OpKind(c10::Symbol::fromQualString(name));
  }

  c10::Symbol op;
};

inline std::ostream &operator<<(std::ostream &stream, const OpKind &op) {
  stream << op.ToString();
  return stream;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &stream,
                                     const OpKind &op) {
  stream << op.ToString();
  return stream;
}

using OpList = std::vector<Value>;

class Node {

public:
  Node(OpKind op);
  Node(OpKind op, OpList operands, std::vector<int64_t> sizes);
  Node(OpKind op, OpList operands, at::IntArrayRef sizes);

  const OpKind &op() const { return op_; }

  virtual std::vector<int64_t> sizes() const { return sizes_[0]; }
  virtual std::vector<int64_t> sizes(size_t i) const { return sizes_[0]; }

  virtual std::vector<int64_t> strides() const { return strides(sizes()); }
  virtual std::vector<int64_t> strides(size_t i) const {
    return strides(sizes(i));
  }

  OpList &operands() { return operands_; }
  Value operand(size_t i) const { return operands_.at(i); }

  virtual mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable);

private:
  std::vector<int64_t> strides(std::vector<int64_t> sz) const;

  OpKind op_;
  OpList operands_;
  std::array<std::vector<int64_t>, 3> sizes_;
  // std::array<std::vector<int64_t>, 3> strides_;
};

class ConstantNode : public Node {
public:
  ConstantNode(at::Scalar scalar)
      : Node(OpKind::Get("aten::constant")), scalar(scalar) {}

  ConstantNode(at::IntArrayRef array)
      : Node(OpKind::Get("aten::constant")), array(array.begin(), array.end()) {
  }

  ConstantNode(bool bool_)
      : Node(OpKind::Get("aten::constant")), bool_(bool_) {}

  ConstantNode(int int_) : Node(OpKind::Get("aten::constant")), int_(int_) {}

  ConstantNode(int64_t int_)
      : Node(OpKind::Get("aten::constant")), int_(int_) {}

  ConstantNode(float float_)
      : Node(OpKind::Get("aten::constant")), float_(float_) {}

  ConstantNode(double double_)
      : Node(OpKind::Get("aten::constant")), double_(double_) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override { return {1}; }
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

private:
  c10::optional<at::Scalar> scalar;
  std::vector<int64_t> array;
  c10::optional<bool> bool_;
  c10::optional<int> int_;
  c10::optional<float> float_;
  c10::optional<double> double_;
};

class AdaptiveAvgPool2dNode : public Node {
public:
  AdaptiveAvgPool2dNode(Value input, at::IntArrayRef kernel_size)
      : Node(OpKind::Get("aten::_adaptive_avg_pool2d"),
             OpList{input,
                    ir::Value(std::make_shared<ir::ConstantNode>(kernel_size))},
             std::vector<int64_t>{input.sizes()[0], input.sizes()[1],
                                  kernel_size[0], kernel_size[1]}) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class AdaptiveAvgPool2dBackwardNode : public Node {
public:
  AdaptiveAvgPool2dBackwardNode(Value grad_output, Value self)
      : Node(OpKind::Get("aten::_adaptive_avg_pool2d_backward"),
             OpList{grad_output, self}, self.sizes()) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class AddNode : public Node {
public:
  AddNode(Value rhs, Value lhs, Value alpha)
      : Node(OpKind::Get("aten::add"), OpList{rhs, lhs, alpha}, rhs.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class AddInPlaceNode : public Node {
public:
  AddInPlaceNode(Value self, Value other, Value alpha)
      : Node(OpKind::Get("aten::add_"), OpList{self, other, alpha},
             self.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class AddmmNode : public Node {
public:
  AddmmNode(Value input, Value mat1, Value mat2, Value beta, Value alpha)
      : Node(OpKind::Get("aten::addmm"), OpList{input, mat1, mat2, beta, alpha},
             std::vector<int64_t>{mat1.sizes()[0], mat2.sizes()[1]}){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class AsStridedNode : public Node {
public:
  AsStridedNode(Value input, at::IntArrayRef size, at::IntArrayRef stride,
                c10::optional<int64_t> storage_offset)
      : Node(OpKind::Get("aten::as_strided"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(size)),
                    ir::Value(std::make_shared<ir::ConstantNode>(stride))},
             input.sizes()),
        size(size.begin(), size.end()), stride(stride.begin(), stride.end()),
        storage_offset(storage_offset) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override;
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

  std::vector<int64_t> strides() const override { return stride; }
  std::vector<int64_t> strides(size_t i) const override { return strides(); }

  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  c10::optional<int64_t> storage_offset;
};

class BatchNormNode : public Node {
public:
  BatchNormNode(Value input, Value weight, Value bias, Value running_mean,
                Value running_var, bool training, double momentum, double eps)
      : Node(OpKind::Get("aten::native_batch_norm"),
             OpList{
                 input, weight, bias, running_mean, running_var,
                 ir::Value(std::make_shared<ir::ConstantNode>(training)),
                 ir::Value(std::make_shared<ir::ConstantNode>((float)momentum)),
                 ir::Value(std::make_shared<ir::ConstantNode>((float)eps))},
             input.sizes()),
        training(training), momentum(momentum), eps(eps) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  bool training;
  double momentum;
  double eps;
};

class BatchNormBackwardNode : public Node {
public:
  BatchNormBackwardNode(Value grad_out, Value input, Value weight,
                        Value running_mean, Value running_var, Value save_mean,
                        Value save_invstd, bool train, double eps,
                        std::array<bool, 3> output_mask)
      : Node(OpKind::Get("aten::native_batch_norm_backward"),
             OpList{grad_out, input, weight, running_mean, running_var,
                    save_mean, save_invstd,
                    ir::Value(std::make_shared<ir::ConstantNode>(train)),
                    ir::Value(std::make_shared<ir::ConstantNode>((float)eps))},
             input.sizes()),
        train(train), eps(eps), output_mask(output_mask) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override {
    assert(0 && "Cannot call sizes() for multiple outputs");
  }
  std::vector<int64_t> sizes(size_t i) const override;

private:
  bool train;
  double eps;
  std::array<bool, 3> output_mask;
};

class Conv2dNode : public Node {
public:
  Conv2dNode(Value input, Value weight, Value bias, at::IntArrayRef stride,
             at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
             at::IntArrayRef output_padding, int64_t groups)
      : Node(OpKind::Get("aten::_convolution"),
             OpList{
                 input, weight, bias,
                 ir::Value(std::make_shared<ir::ConstantNode>(stride)),
                 ir::Value(std::make_shared<ir::ConstantNode>(padding)),
                 ir::Value(std::make_shared<ir::ConstantNode>(dilation)),
                 ir::Value(std::make_shared<ir::ConstantNode>(transposed)),
                 ir::Value(std::make_shared<ir::ConstantNode>(output_padding)),
                 ir::Value(std::make_shared<ir::ConstantNode>(groups))},
             input.sizes()),
        stride(stride.begin(), stride.end()),
        padding(padding.begin(), padding.end()),
        dilation(dilation.begin(), dilation.end()), transposed(transposed),
        output_padding(output_padding.begin(), output_padding.end()),
        groups(groups), has_bias(true) {}

  Conv2dNode(Value input, Value weight, at::IntArrayRef stride,
             at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
             at::IntArrayRef output_padding, int64_t groups)
      : Node(OpKind::Get("aten::_convolution"),
             OpList{
                 input, weight,
                 ir::Value(std::make_shared<ir::ConstantNode>(stride)),
                 ir::Value(std::make_shared<ir::ConstantNode>(padding)),
                 ir::Value(std::make_shared<ir::ConstantNode>(dilation)),
                 ir::Value(std::make_shared<ir::ConstantNode>(transposed)),
                 ir::Value(std::make_shared<ir::ConstantNode>(output_padding)),
                 ir::Value(std::make_shared<ir::ConstantNode>(groups))},
             input.sizes()),
        stride(stride.begin(), stride.end()),
        padding(padding.begin(), padding.end()),
        dilation(dilation.begin(), dilation.end()), transposed(transposed),
        output_padding(output_padding.begin(), output_padding.end()),
        groups(groups), has_bias(false) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override;
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

private:
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int64_t groups;
  bool has_bias;
};

class Conv2dBackwardNode : public Node {
public:
  Conv2dBackwardNode(Value grad_output, Value input, Value weight,
                     at::IntArrayRef stride, at::IntArrayRef padding,
                     at::IntArrayRef dilation, bool transposed,
                     at::IntArrayRef output_padding, int64_t groups)
      : Node(OpKind::Get("aten::_convolution_backward"),
             OpList{
                 grad_output, input, weight,
                 ir::Value(std::make_shared<ir::ConstantNode>(stride)),
                 ir::Value(std::make_shared<ir::ConstantNode>(padding)),
                 ir::Value(std::make_shared<ir::ConstantNode>(dilation)),
                 ir::Value(std::make_shared<ir::ConstantNode>(transposed)),
                 ir::Value(std::make_shared<ir::ConstantNode>(output_padding)),
                 ir::Value(std::make_shared<ir::ConstantNode>(groups))},
             input.sizes()),
        stride(stride.begin(), stride.end()),
        padding(padding.begin(), padding.end()),
        dilation(dilation.begin(), dilation.end()), transposed(transposed),
        output_padding(output_padding.begin(), output_padding.end()),
        groups(groups) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override {
    assert(0 && "Cannot call sizes() for multiple outputs");
  }
  std::vector<int64_t> sizes(size_t i) const override;

private:
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int64_t groups;
};

class DivNode : public Node {
public:
  DivNode(Value rhs, Value lhs)
      : Node(OpKind::Get("aten::div"), OpList{rhs, lhs}, rhs.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class DivInPlaceNode : public Node {
public:
  DivInPlaceNode(Value self, Value other)
      : Node(OpKind::Get("aten::div_"), OpList{self, other}, self.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class ExpandNode : public Node {
public:
  ExpandNode(Value input, at::IntArrayRef size, bool implicit)
      : Node(OpKind::Get("aten::expand"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(size)),
                    ir::Value(std::make_shared<ir::ConstantNode>(implicit))},
             input.sizes()),
        output_size(size.begin(), size.end()), implicit(implicit) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override { return output_size; }
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

private:
  std::vector<int64_t> output_size;
  bool implicit;
};

class GatherNode : public Node {
public:
  GatherNode(Value input, int64_t dim, Value index, bool sparse_grad)
      : Node(OpKind::Get("aten::gather"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(dim)),
                    index,
                    ir::Value(std::make_shared<ir::ConstantNode>(sparse_grad))},
             input.sizes()) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class HardtanhNode : public Node {
public:
  HardtanhNode(Value self, Value min_val, Value max_val)
      : Node(OpKind::Get("aten::hardtanh"), OpList{self, min_val, max_val},
             self.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class HardtanhInPlaceNode : public Node {
public:
  HardtanhInPlaceNode(Value self, Value min_val, Value max_val)
      : Node(OpKind::Get("aten::hardtanh_"), OpList{self, min_val, max_val},
             self.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class HardtanhBackwardNode : public Node {
public:
  HardtanhBackwardNode(Value grad_output, Value self, Value min_val,
                       Value max_val)
      : Node(OpKind::Get("aten::hardtanh_backward"),
             OpList{grad_output, self, min_val, max_val}, self.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class LogSoftmaxNode : public Node {
public:
  LogSoftmaxNode(Value input, int64_t dim, bool half_to_float)
      : Node(OpKind::Get("aten::_log_softmax"),
             OpList{
                 input, ir::Value(std::make_shared<ir::ConstantNode>(dim)),
                 ir::Value(std::make_shared<ir::ConstantNode>(half_to_float))},
             input.sizes()),
        dim(dim), half_to_float(half_to_float) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  int64_t dim;
  bool half_to_float;
};

class LogSoftmaxBackwardNode : public Node {
public:
  LogSoftmaxBackwardNode(Value grad_output, Value output, int64_t dim,
                         Value input)
      : Node(OpKind::Get("aten::_log_softmax_backward_data"),
             OpList{grad_output, output,
                    ir::Value(std::make_shared<ir::ConstantNode>(dim)), input},
             input.sizes()),
        dim(dim) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  int64_t dim;
};

class MaxPool2dWithIndicesNode : public Node {
public:
  MaxPool2dWithIndicesNode(Value input, at::IntArrayRef kernel_size,
                           at::IntArrayRef stride, at::IntArrayRef padding,
                           at::IntArrayRef dilation, bool ceil_mode)
      : Node(OpKind::Get("aten::max_pool2d_with_indices"),
             OpList{input,
                    ir::Value(std::make_shared<ir::ConstantNode>(kernel_size)),
                    ir::Value(std::make_shared<ir::ConstantNode>(stride)),
                    ir::Value(std::make_shared<ir::ConstantNode>(padding)),
                    ir::Value(std::make_shared<ir::ConstantNode>(dilation)),
                    ir::Value(std::make_shared<ir::ConstantNode>(ceil_mode))},
             input.sizes()),
        kernel_size(kernel_size.begin(), kernel_size.end()),
        stride(stride.begin(), stride.end()),
        padding(padding.begin(), padding.end()),
        dilation(dilation.begin(), dilation.end()), ceil_mode(ceil_mode){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override {
    assert(0 && "Cannot call sizes() for multiple outputs");
  }
  std::vector<int64_t> sizes(size_t i) const override;

private:
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
};

class MaxPool2dWithIndicesBackwardNode : public Node {
public:
  MaxPool2dWithIndicesBackwardNode(Value grad_output, Value input,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode,
                                   Value indices)
      : Node(OpKind::Get("aten::max_pool2d_with_indices_backward"),
             OpList{grad_output, input,
                    ir::Value(std::make_shared<ir::ConstantNode>(kernel_size)),
                    ir::Value(std::make_shared<ir::ConstantNode>(stride)),
                    ir::Value(std::make_shared<ir::ConstantNode>(padding)),
                    ir::Value(std::make_shared<ir::ConstantNode>(dilation)),
                    ir::Value(std::make_shared<ir::ConstantNode>(ceil_mode)),
                    indices},
             input.sizes()),
        kernel_size(kernel_size.begin(), kernel_size.end()),
        stride(stride.begin(), stride.end()),
        padding(padding.begin(), padding.end()),
        dilation(dilation.begin(), dilation.end()), ceil_mode(ceil_mode){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
};

class MeanNode : public Node {
public:
  MeanNode(Value input, at::IntArrayRef dim, bool keepdim,
           c10::optional<at::ScalarType> dtype)
      : Node(OpKind::Get("aten::mean"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(dim)),
                    ir::Value(std::make_shared<ir::ConstantNode>(keepdim))},
             input.sizes()),
        dim(dim.begin(), dim.end()), keepdim(keepdim), dtype(dtype) {}

  MeanNode(Value input, c10::optional<at::ScalarType> dtype)
      : Node(OpKind::Get("aten::mean"), OpList{input}, input.sizes()),
        dtype(dtype) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override;
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

private:
  std::vector<int64_t> dim;
  bool keepdim;
  c10::optional<at::ScalarType> dtype;
};

class MMNode : public Node {
public:
  MMNode(Value input, Value mat2)
      : Node(OpKind::Get("aten::mm"), OpList{input, mat2},
             std::vector<int64_t>{input.sizes()[0], mat2.sizes()[1]}){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class MulNode : public Node {
public:
  MulNode(Value rhs, Value lhs)
      : Node(OpKind::Get("aten::mul"), OpList{rhs, lhs}, rhs.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class MulInPlaceNode : public Node {
public:
  MulInPlaceNode(Value self, Value other)
      : Node(OpKind::Get("aten::mul_"), OpList{self, other}, self.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class NegNode : public Node {
public:
  NegNode(Value input)
      : Node(OpKind::Get("aten::neg"), OpList{input}, input.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class NllLoss2dForwardNode : public Node {
public:
  NllLoss2dForwardNode(Value self, Value target, Value weight,
                       int64_t reduction, int64_t ignore_index)
      : Node(
            OpKind::Get("aten::nll_loss2d_forward"),
            OpList{self, target, weight,
                   ir::Value(std::make_shared<ir::ConstantNode>(reduction)),
                   ir::Value(std::make_shared<ir::ConstantNode>(ignore_index))},
            1 /*target.sizes()*/),
        reduction(reduction), ignore_index(ignore_index) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  int64_t reduction;
  int64_t ignore_index;
};

class NllLoss2dBackwardNode : public Node {
public:
  NllLoss2dBackwardNode(Value grad_output, Value self, Value target,
                        Value weight, int64_t reduction, int64_t ignore_index,
                        Value total_weight)
      : Node(OpKind::Get("aten::nll_loss2d_backward"),
             OpList{grad_output, self, target, weight,
                    ir::Value(std::make_shared<ir::ConstantNode>(reduction)),
                    ir::Value(std::make_shared<ir::ConstantNode>(ignore_index)),
                    total_weight},
             self.sizes()),
        reduction(reduction), ignore_index(ignore_index) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  int64_t reduction;
  int64_t ignore_index;
};

class NllLossForwardNode : public Node {
public:
  NllLossForwardNode(Value self, Value target, Value weight, int64_t reduction,
                     int64_t ignore_index)
      : Node(
            OpKind::Get("aten::nll_loss_forward"),
            OpList{self, target, weight,
                   ir::Value(std::make_shared<ir::ConstantNode>(reduction)),
                   ir::Value(std::make_shared<ir::ConstantNode>(ignore_index))},
            1 /*target.sizes()*/),
        reduction(reduction), ignore_index(ignore_index) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  int64_t reduction;
  int64_t ignore_index;
};

class NllLossBackwardNode : public Node {
public:
  NllLossBackwardNode(Value grad_output, Value self, Value target, Value weight,
                      int64_t reduction, int64_t ignore_index,
                      Value total_weight)
      : Node(OpKind::Get("aten::nll_loss_backward"),
             OpList{grad_output, self, target, weight,
                    ir::Value(std::make_shared<ir::ConstantNode>(reduction)),
                    ir::Value(std::make_shared<ir::ConstantNode>(ignore_index)),
                    total_weight},
             self.sizes()),
        reduction(reduction), ignore_index(ignore_index) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  int64_t reduction;
  int64_t ignore_index;
};

class SumNode : public Node {
public:
  SumNode(Value input, at::IntArrayRef dim, bool keepdim,
          c10::optional<at::ScalarType> dtype)
      : Node(OpKind::Get("aten::sum"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(dim)),
                    ir::Value(std::make_shared<ir::ConstantNode>(keepdim))},
             input.sizes()),
        dim(dim.begin(), dim.end()), keepdim(keepdim), dtype(dtype) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override;
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

private:
  std::vector<int64_t> dim;
  bool keepdim;
  c10::optional<at::ScalarType> dtype;
};

class ReLUNode : public Node {
public:
  ReLUNode(Value input)
      : Node(OpKind::Get("aten::relu"), OpList{input}, input.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class ReLUInPlaceNode : public Node {
public:
  ReLUInPlaceNode(Value input)
      : Node(OpKind::Get("aten::relu_"), OpList{input}, input.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class ThresholdBackwardNode : public Node {
public:
  ThresholdBackwardNode(Value grad_output, Value input, Value threshold)
      : Node(OpKind::Get("aten::threshold_backward"),
             OpList{grad_output, input, threshold}, input.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class TransposeNode : public Node {
public:
  TransposeNode(Value input)
      : Node(OpKind::Get("aten::t"), OpList{input},
             std::vector<int64_t>{input.sizes()[1], input.sizes()[0]}){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class SizeNode : public Node {
public:
  SizeNode(Value input, int64_t dim)
      : Node(OpKind::Get("aten::size"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(dim))},
             1),
        dim(dim) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

private:
  int64_t dim;
};

class SqueezeNode : public Node {
public:
  SqueezeNode(Value input, int64_t dim)
      : Node(OpKind::Get("aten::squeeze"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(dim))},
             input.sizes()),
        dim(dim) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override;
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

private:
  int64_t dim;
};

class SubNode : public Node {
public:
  SubNode(Value rhs, Value lhs, Value alpha)
      : Node(OpKind::Get("aten::sub"), OpList{rhs, lhs, alpha}, rhs.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class SubInPlaceNode : public Node {
public:
  SubInPlaceNode(Value self, Value other, Value alpha)
      : Node(OpKind::Get("aten::sub_"), OpList{self, other, alpha},
             self.sizes()){};

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;
};

class UnsqueezeNode : public Node {
public:
  UnsqueezeNode(Value input, int64_t dim)
      : Node(OpKind::Get("aten::unsqueeze"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(dim))},
             input.sizes()),
        dim(dim) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override;
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

private:
  int64_t dim;
};

class ViewNode : public Node {
public:
  ViewNode(Value input, at::IntArrayRef size)
      : Node(OpKind::Get("aten::view"),
             OpList{input, ir::Value(std::make_shared<ir::ConstantNode>(size))},
             input.sizes()),
        view_size(size.begin(), size.end()) {}

  mlir::Operation *
  genMLIR(std::unique_ptr<mlir::OpBuilder> &builder, mlir::MLIRContext &context,
          std::map<const ir::Value, mlir::Value> &symbolTable) override;

  std::vector<int64_t> sizes() const override;
  std::vector<int64_t> sizes(size_t i) const override { return sizes(); }

private:
  std::vector<int64_t> view_size;
};

class TorchDataNode : public Node {

public:
  TorchDataNode(at::Tensor tensor)
      : Node(ir::OpKind::Get("aten::torch_data"), {}, tensor.sizes()),
        tensor_(std::move(tensor)) {}

  at::Tensor tensor() { return tensor_; }

private:
  at::Tensor tensor_;
};

} // namespace ir
} // namespace torch_mlir
