//===- tensor.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "ATen/ArrayRef.h"
namespace at {
template <typename T> using ArrayRef = c10::ArrayRef<T>;
}
#include "ATen/Tensor.h"

#include "jit.h"
#include "tensor.h"

#include <atomic>

#define DEBUG_TYPE "torch_mlir"

namespace torch_mlir {

MLIRTensor MLIRTensor::Create(const at::Tensor &tensor, const Device &device) {
  assert(tensor.device().type() == at::kCPU);
  MLIRTensor device_tensor(tensor, device);
  return device_tensor;
}

MLIRTensor
MLIRTensor::Create(ir::Value ir_value, const Device &device,
                   c10::optional<at::ScalarType> logical_element_type) {
  MLIRTensor device_tensor(std::move(ir_value), device, logical_element_type);
  return device_tensor;
}

MLIRTensor::MLIRTensor(const at::Tensor &tensor, const Device &device)
    : data_(std::make_shared<Data>(tensor, device)) {}

MLIRTensor::MLIRTensor(ir::Value ir_value, const Device &device,
                       c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(ir_value), device,
                                   logical_element_type)) {}

MLIRTensor::Data *MLIRTensor::data() const {
  assert(data_ != nullptr && "Trying to access null data");
  return data_.get();
}

at::ScalarType MLIRTensor::dtype() const {
  return data()->logical_element_type ? *data()->logical_element_type
                                      : at::ScalarType::Float;
}

const Device &MLIRTensor::GetDevice() const { return data()->device; }

uint64_t MLIRTensor::GetNextTensorId() {
  static std::atomic<uint64_t> *id_generator = new std::atomic<uint64_t>(1);
  return id_generator->fetch_add(1);
}

void MLIRTensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

ir::Value MLIRTensor::GetIrValue() const {
  ir::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (tensor_data) {
    at::Tensor tensor = *tensor_data;
    if (!tensor.dim()) {
      auto dtype = tensor.dtype();
      if (dtype == at::kFloat) {
        auto d = tensor.data_ptr<float>();
        return ir::Value(std::make_shared<ir::ConstantNode>(d[0]));
      } else if (dtype == at::kDouble) {
        auto d = tensor.data_ptr<double>();
        return ir::Value(std::make_shared<ir::ConstantNode>(d[0]));
      } else if (dtype == at::kLong) {
        auto d = tensor.data_ptr<int64_t>();
        return ir::Value(std::make_shared<ir::ConstantNode>(d[0]));
      } else if (dtype == at::kInt) {
        auto d = tensor.data_ptr<int32_t>();
        return ir::Value(std::make_shared<ir::ConstantNode>(d[0]));
      } else if (dtype == at::kShort) {
        auto d = tensor.data_ptr<int16_t>();
        return ir::Value(std::make_shared<ir::ConstantNode>(d[0]));
      } else if (dtype == at::kChar || dtype == at::kByte) {
        auto d = tensor.data_ptr<int8_t>();
        return ir::Value(std::make_shared<ir::ConstantNode>(d[0]));
      }
      // fall through to TorchDataNode below
    }
    return ir::Value(std::make_shared<ir::TorchDataNode>(*tensor_data));
  }
  assert(0 && "Could not create ir value from leaf tensor");
  return ir::Value();
}

ir::Value MLIRTensor::CurrentIrValue() const { return data()->ir_value; }

void MLIRTensor::SetIrValue(ir::Value ir_value) {
  data()->generation += 1;
  data()->ir_value = std::move(ir_value);
}

c10::optional<at::Tensor> MLIRTensor::CurrentTensorData() const {
  return data()->tensor_data;
}

void MLIRTensor::SetTensor(at::Tensor tensor) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  SetTensorData(tensor);
  data()->generation += 1;
}

at::Tensor MLIRTensor::ToTensor() const {
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data)
    tensor_data = CompileAndRun();
  assert(tensor_data);
  return *tensor_data;
}

void MLIRTensor::ShallowCopyTo(MLIRTensor *dest) const {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");

  auto data = CurrentTensorData();
  if (data)
    dest->SetTensor(*data);
  else
    dest->SetIrValue(CurrentIrValue());

  dest->SetScalarType(dtype());
  assert(GetDevice() == dest->GetDevice());
}

void MLIRTensor::SetScalarType(
    c10::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

std::vector<int64_t> MLIRTensor::sizes() const {
  if (data()->ir_value) {
    return data()->ir_value.sizes();
  }
  assert(data()->tensor_data && "tensor has no shape information");
  if (data()->tensor_data) {
    auto s = data()->tensor_data->sizes();
    return {s.begin(), s.end()};
  }
  return {};
}

std::vector<int64_t> MLIRTensor::strides() const {
  if (data()->ir_value) {
    return data()->ir_value.strides();
  }
  assert(data()->tensor_data && "tensor has no shape information");
  if (data()->tensor_data) {
    auto s = data()->tensor_data->strides();
    return {s.begin(), s.end()};
  }
  return {};
}

MLIRTensor MLIRTensor::CreateFrom(ir::Value ir_value) const {
  return Create(std::move(ir_value), GetDevice(), dtype());
}

////////////////////////////////////////////
// aten tensor methods
////////////////////////////////////////////

MLIRTensor MLIRTensor::_adaptive_avg_pool2d(const MLIRTensor &self,
                                            at::IntArrayRef output_size) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::AdaptiveAvgPool2dNode>(
      self.GetIrValue(), output_size);
  return self.CreateFrom(node);
}

MLIRTensor
MLIRTensor::_adaptive_avg_pool2d_backward(const MLIRTensor &grad_output,
                                          const MLIRTensor &self) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::AdaptiveAvgPool2dBackwardNode>(
          grad_output.GetIrValue(), self.GetIrValue());
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::add(const MLIRTensor &self, const MLIRTensor &other,
                           at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::AddNode>(
      self.GetIrValue(), other.GetIrValue(),
      ir::Value(std::make_shared<ir::ConstantNode>(alpha)));
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::add_(MLIRTensor &self, const MLIRTensor &other,
                            at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::AddInPlaceNode>(
      self.GetIrValue(), other.GetIrValue(),
      ir::Value(std::make_shared<ir::ConstantNode>(alpha)));
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::addmm(const MLIRTensor &input, const MLIRTensor &mat1,
                             const MLIRTensor &mat2, at::Scalar beta,
                             at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::AddmmNode>(
      input.GetIrValue(), mat1.GetIrValue(), mat2.GetIrValue(),
      ir::Value(std::make_shared<ir::ConstantNode>(beta)),
      ir::Value(std::make_shared<ir::ConstantNode>(alpha)));
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::as_strided(const MLIRTensor &input, at::IntArrayRef size,
                                  at::IntArrayRef stride,
                                  c10::optional<int64_t> storage_offset) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::AsStridedNode>(
      input.GetIrValue(), size, stride, storage_offset);
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::clone(const MLIRTensor &input) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  return MLIRTensor::Create(std::move(input.ToTensor()), input.GetDevice());
}

MLIRTensor MLIRTensor::convolution(
    const MLIRTensor &input, const MLIRTensor &weight, const MLIRTensor &bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::Conv2dNode>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(), stride,
      padding, dilation, transposed, output_padding, groups);
  return input.CreateFrom(node);
}

std::tuple<MLIRTensor, MLIRTensor, MLIRTensor> MLIRTensor::convolution_backward(
    const MLIRTensor &grad_output, const MLIRTensor &input,
    const MLIRTensor &weight, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::Conv2dBackwardNode>(
      grad_output.GetIrValue(), input.GetIrValue(), weight.GetIrValue(), stride,
      padding, dilation, transposed, output_padding, groups /*, output_mask*/);
  auto result0 = input.CreateFrom(ir::Value(node, 0));
  auto result1 = input.CreateFrom(ir::Value(node, 1));
  auto result2 = input.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(result0, result1, result2);
}

void MLIRTensor::copy_(MLIRTensor &self, MLIRTensor &src) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  src.ShallowCopyTo(&self);
}

MLIRTensor MLIRTensor::div(const MLIRTensor &self, at::Scalar other) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::DivNode>(
      self.GetIrValue(), ir::Value(std::make_shared<ir::ConstantNode>(other)));
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::div(const MLIRTensor &self, const MLIRTensor &other) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::DivNode>(self.GetIrValue(), other.GetIrValue());
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::div_(MLIRTensor &self, const MLIRTensor &other) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::DivInPlaceNode>(
      self.GetIrValue(), other.GetIrValue());
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::expand(const MLIRTensor &self, at::IntArrayRef size,
                              bool implicit) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::ExpandNode>(self.GetIrValue(), size, implicit);
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::gather(const MLIRTensor &self, int64_t dim,
                              const MLIRTensor &index, bool sparse_grad) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::GatherNode>(
      self.GetIrValue(), dim, index.GetIrValue(), sparse_grad);
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::hardtanh(const MLIRTensor &self, at::Scalar min_val,
                                at::Scalar max_val) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::HardtanhNode>(
      self.GetIrValue(), ir::Value(std::make_shared<ir::ConstantNode>(min_val)),
      ir::Value(std::make_shared<ir::ConstantNode>(max_val)));
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::hardtanh_(MLIRTensor &self, at::Scalar min_val,
                                 at::Scalar max_val) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::HardtanhInPlaceNode>(
      self.GetIrValue(), ir::Value(std::make_shared<ir::ConstantNode>(min_val)),
      ir::Value(std::make_shared<ir::ConstantNode>(max_val)));
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::hardtanh_backward(const MLIRTensor &grad_output,
                                         const MLIRTensor &self,
                                         at::Scalar min_val,
                                         at::Scalar max_val) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::HardtanhBackwardNode>(
      grad_output.GetIrValue(), self.GetIrValue(),
      ir::Value(std::make_shared<ir::ConstantNode>(min_val)),
      ir::Value(std::make_shared<ir::ConstantNode>(max_val)));
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::_log_softmax(const MLIRTensor &input, int64_t dim,
                                    bool half_to_float) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::LogSoftmaxNode>(
      input.GetIrValue(), dim, half_to_float);
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::_log_softmax_backward_data(const MLIRTensor &grad_output,
                                                  const MLIRTensor &output,
                                                  int64_t dim,
                                                  const MLIRTensor &input) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::LogSoftmaxBackwardNode>(
      grad_output.GetIrValue(), output.GetIrValue(), dim, input.GetIrValue());
  return input.CreateFrom(node);
}

std::tuple<MLIRTensor, MLIRTensor> MLIRTensor::max_pool2d_with_indices(
    const MLIRTensor &input, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::MaxPool2dWithIndicesNode>(
          input.GetIrValue(), kernel_size, stride, padding, dilation,
          ceil_mode);
  auto result0 = input.CreateFrom(ir::Value(node, 0));
  auto result1 = input.CreateFrom(ir::Value(node, 1));
  return std::make_tuple(result0, result1);
}

MLIRTensor MLIRTensor::max_pool2d_with_indices_backward(
    const MLIRTensor &grad_output, const MLIRTensor &input,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const MLIRTensor &indices) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::MaxPool2dWithIndicesBackwardNode>(
          grad_output.GetIrValue(), input.GetIrValue(), kernel_size, stride,
          padding, dilation, ceil_mode, indices.GetIrValue());
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::mean(const MLIRTensor &input,
                            c10::optional<at::ScalarType> dtype) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::MeanNode>(input.GetIrValue(), dtype);
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::mean(const MLIRTensor &input, at::IntArrayRef dim,
                            bool keepdim, c10::optional<at::ScalarType> dtype) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::MeanNode>(input.GetIrValue(), dim, keepdim, dtype);
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::mm(const MLIRTensor &input, const MLIRTensor &mat1) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::MMNode>(input.GetIrValue(), mat1.GetIrValue());
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::mul(const MLIRTensor &self, const MLIRTensor &other) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::MulNode>(self.GetIrValue(), other.GetIrValue());
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::mul_(MLIRTensor &self, const MLIRTensor &other) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::MulInPlaceNode>(
      self.GetIrValue(), other.GetIrValue());
  return self.CreateFrom(node);
}

std::tuple<MLIRTensor, MLIRTensor, MLIRTensor> MLIRTensor::native_batch_norm(
    const MLIRTensor &self, const MLIRTensor &weight, const MLIRTensor &bias,
    const MLIRTensor &running_mean, const MLIRTensor &running_var,
    bool training, double momentum, double eps) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::BatchNormNode>(
      self.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      running_mean.GetIrValue(), running_var.GetIrValue(), training, momentum,
      eps);
  auto result0 = self.CreateFrom(ir::Value(node, 0));
  auto result1 = self.CreateFrom(ir::Value(node, 1));
  auto result2 = self.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(result0, result1, result2);
}

std::tuple<MLIRTensor, MLIRTensor, MLIRTensor>
MLIRTensor::native_batch_norm_backward(
    const MLIRTensor &grad_out, const MLIRTensor &input,
    const MLIRTensor &weight, const MLIRTensor &running_mean,
    const MLIRTensor &running_var, const MLIRTensor &save_mean,
    const MLIRTensor &save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::BatchNormBackwardNode>(
      grad_out.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
      running_mean.GetIrValue(), running_var.GetIrValue(),
      save_mean.GetIrValue(), save_invstd.GetIrValue(), train, eps,
      output_mask);
  auto result0 = input.CreateFrom(ir::Value(node, 0));
  auto result1 = input.CreateFrom(ir::Value(node, 1));
  auto result2 = input.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(result0, result1, result2);
}

MLIRTensor MLIRTensor::neg(const MLIRTensor &input) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::NegNode>(input.GetIrValue());
  return input.CreateFrom(node);
}

std::tuple<MLIRTensor, MLIRTensor>
MLIRTensor::nll_loss2d_forward(const MLIRTensor &self, const MLIRTensor &target,
                               const MLIRTensor &weight, int64_t reduction,
                               int64_t ignore_index) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::NllLoss2dForwardNode>(
      self.GetIrValue(), target.GetIrValue(), weight.GetIrValue(), reduction,
      ignore_index);
  auto result0 = self.CreateFrom(ir::Value(node, 0));
  auto result1 = self.CreateFrom(ir::Value(node, 1));
  return std::make_tuple(result0, result1);
}

MLIRTensor MLIRTensor::nll_loss2d_backward(
    const MLIRTensor &grad_output, const MLIRTensor &self,
    const MLIRTensor &target, const MLIRTensor &weight, int64_t reduction,
    int64_t ignore_index, const MLIRTensor &total_weight) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::NllLoss2dBackwardNode>(
      grad_output.GetIrValue(), self.GetIrValue(), target.GetIrValue(),
      weight.GetIrValue(), reduction, ignore_index, total_weight.GetIrValue());
  return self.CreateFrom(node);
}

std::tuple<MLIRTensor, MLIRTensor>
MLIRTensor::nll_loss_forward(const MLIRTensor &self, const MLIRTensor &target,
                             const MLIRTensor &weight, int64_t reduction,
                             int64_t ignore_index) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::NllLossForwardNode>(
      self.GetIrValue(), target.GetIrValue(), weight.GetIrValue(), reduction,
      ignore_index);
  auto result0 = self.CreateFrom(ir::Value(node, 0));
  auto result1 = self.CreateFrom(ir::Value(node, 1));
  return std::make_tuple(result0, result1);
}

MLIRTensor MLIRTensor::nll_loss_backward(
    const MLIRTensor &grad_output, const MLIRTensor &self,
    const MLIRTensor &target, const MLIRTensor &weight, int64_t reduction,
    int64_t ignore_index, const MLIRTensor &total_weight) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::NllLossBackwardNode>(
      grad_output.GetIrValue(), self.GetIrValue(), target.GetIrValue(),
      weight.GetIrValue(), reduction, ignore_index, total_weight.GetIrValue());
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::sum(const MLIRTensor &input, at::IntArrayRef dim,
                           bool keepdim, c10::optional<at::ScalarType> dtype) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::SumNode>(input.GetIrValue(), dim, keepdim, dtype);
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::relu(const MLIRTensor &input) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::ReLUNode>(input.GetIrValue());
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::relu_(MLIRTensor &input) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::ReLUInPlaceNode>(input.GetIrValue());
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::size(const MLIRTensor &input, int64_t dim) {
  assert(0);
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::SizeNode>(input.GetIrValue(), dim);
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::squeeze(const MLIRTensor &input, int64_t dim) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::SqueezeNode>(input.GetIrValue(), dim);
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::sub(const MLIRTensor &self, const MLIRTensor &other,
                           at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::SubNode>(
      self.GetIrValue(), other.GetIrValue(),
      ir::Value(std::make_shared<ir::ConstantNode>(alpha)));
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::sub_(MLIRTensor &self, const MLIRTensor &other,
                            at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::SubInPlaceNode>(
      self.GetIrValue(), other.GetIrValue(),
      ir::Value(std::make_shared<ir::ConstantNode>(alpha)));
  return self.CreateFrom(node);
}

MLIRTensor MLIRTensor::t(const MLIRTensor &input) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::TransposeNode>(input.GetIrValue());
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::threshold_backward(const MLIRTensor &grad_output,
                                          const MLIRTensor &input,
                                          at::Scalar threshold) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node = std::make_shared<ir::ThresholdBackwardNode>(
      grad_output.GetIrValue(), input.GetIrValue(),
      ir::Value(std::make_shared<ir::ConstantNode>(threshold)));
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::to(MLIRTensor &input, c10::optional<Device> device,
                          c10::optional<at::ScalarType> scalar_type) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  if (!device) {
    device = input.GetDevice();
  }
  if (!scalar_type) {
    scalar_type = input.dtype();
  }

  MLIRTensor new_tensor = Create(input.ToTensor(), *device);

  if (input.dtype() != *scalar_type) {
    new_tensor.SetScalarType(*scalar_type);
  }
  return new_tensor;
}

MLIRTensor MLIRTensor::unsqueeze(const MLIRTensor &input, int64_t dim) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::UnsqueezeNode>(input.GetIrValue(), dim);
  return input.CreateFrom(node);
}

MLIRTensor MLIRTensor::view(const MLIRTensor &input, at::IntArrayRef size) {
  LLVM_DEBUG(llvm::dbgs() << "MLIRTensor::" << __func__ << "\n");
  std::shared_ptr<ir::Node> node =
      std::make_shared<ir::ViewNode>(input.GetIrValue(), size);
  return input.CreateFrom(node);
}

} // namespace torch_mlir
