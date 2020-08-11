//===- aten_mlir_type.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// Structured similarly to code from git@github.com:pytorch/xla.git

#include "llvm/Support/Debug.h"

#include "aten_mlir_bridge.h"
#include "aten_mlir_type.h"
#include "aten_mlir_type_default.h"
#include "ir.h"
#include "tensor_impl.h"
#include "torch_util.h"

#include <mutex>

#define DEBUG_TYPE "torch_mlir"

namespace torch_mlir {
namespace {

struct MLIROptions {
  MLIROptions(const at::TensorOptions &options,
              c10::optional<Device> device_opt = c10::nullopt,
              c10::optional<at::ScalarType> scalar_type_opt = c10::nullopt)
      : device(std::move(device_opt)), scalar_type(std::move(scalar_type_opt)) {
    if (options.has_device()) {
      device = bridge::AtenDeviceToMLIRDevice(options.device());
    }
    if (options.has_dtype()) {
      scalar_type = c10::typeMetaToScalarType(options.dtype());
    }
  }

  Device get_device() const { return device ? *device : *GetDefaultDevice(); }

  at::ScalarType
  get_scalar_type(at::ScalarType defval = at::ScalarType::Float) const {
    return scalar_type ? *scalar_type : defval;
  }

  c10::optional<Device> device;
  c10::optional<at::ScalarType> scalar_type;
};

std::tuple<MLIRTensor, MLIRTensor>
GetPromotedMLIRTensorsForBinaryOp(const at::Tensor &self,
                                  const at::Tensor &other) {
  // this requires slightly newer than pytorch 1.3.0, disable for now.
  // at::ScalarType dtype = at::result_type(self, other);
  MLIRTensor tensor1 = bridge::GetMLIRTensor(self);
  MLIRTensor tensor2 =
      bridge::GetOrCreateMLIRTensor(other, tensor1.GetDevice());
  // tensor1.SetScalarType(dtype);
  // tensor2.SetScalarType(dtype);
  return std::make_tuple(tensor1, tensor2);
}

void AtenInitialize() {
  RegisterAtenTypeFunctions();
  ir::RegisterAtenIR();
}

} // namespace

void ATenMLIRType::InitializeAtenBindings() {
  static std::once_flag once;
  std::call_once(once, []() { AtenInitialize(); });
}

at::Tensor ATenMLIRType::_adaptive_avg_pool2d(const at::Tensor &self,
                                              at::IntArrayRef output_size) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(MLIRTensor::_adaptive_avg_pool2d(
      bridge::GetMLIRTensor(self), output_size));
}

at::Tensor
ATenMLIRType::_adaptive_avg_pool2d_backward(const at::Tensor &grad_output,
                                            const at::Tensor &self) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto grad_output_tensor =
      bridge::GetOrCreateMLIRTensor(grad_output, input_tensor.GetDevice());

  return bridge::AtenFromMLIRTensor(MLIRTensor::_adaptive_avg_pool2d_backward(
      grad_output_tensor, input_tensor));
}

at::Tensor ATenMLIRType::add(const at::Tensor &self, const at::Tensor &other,
                             at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensors = GetPromotedMLIRTensorsForBinaryOp(self, other);
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::add(std::get<0>(tensors), std::get<1>(tensors), alpha));
}

at::Tensor &ATenMLIRType::add_(at::Tensor &self, const at::Tensor &other,
                               at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensors = GetPromotedMLIRTensorsForBinaryOp(self, other);
  auto result = bridge::AtenFromMLIRTensor(
      MLIRTensor::add_(std::get<0>(tensors), std::get<1>(tensors), alpha));
  MLIRTensorImpl *self_impl =
      dynamic_cast<MLIRTensorImpl *>(self.unsafeGetTensorImpl());
  self_impl->shallow_copy_from(result.getIntrusivePtr());
  return self;
}

at::Tensor ATenMLIRType::addmm(const at::Tensor &self, const at::Tensor &mat1,
                               const at::Tensor &mat2, at::Scalar beta,
                               at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensor = bridge::GetMLIRTensor(self);
  return bridge::AtenFromMLIRTensor(MLIRTensor::addmm(
      tensor, bridge::GetOrCreateMLIRTensor(mat1, tensor.GetDevice()),
      bridge::GetOrCreateMLIRTensor(mat2, tensor.GetDevice()), beta, alpha));
}

at::Tensor ATenMLIRType::as_strided(const at::Tensor &self,
                                    at::IntArrayRef size,
                                    at::IntArrayRef stride,
                                    c10::optional<int64_t> storage_offset) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(MLIRTensor::as_strided(
      bridge::GetMLIRTensor(self), size, stride, storage_offset));
}

at::Tensor ATenMLIRType::clone(const at::Tensor &self) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");

  return bridge::AtenFromMLIRTensor(
      MLIRTensor::clone(bridge::GetMLIRTensor(self)));
}

at::Tensor &ATenMLIRType::copy_(at::Tensor &self, const at::Tensor &src,
                                bool non_blocking) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");

  auto self_tensor = bridge::TryGetMLIRTensor(self);
  auto src_tensor = bridge::TryGetMLIRTensor(src);

  if (!src_tensor) {
    assert(self_tensor);
    self_tensor->SetTensor(util::CopyTensor(src, self.scalar_type()));
  } else if (!self_tensor) {
    at::Tensor t = src_tensor->ToTensor();
    const_cast<at::Tensor &>(self).unsafeGetTensorImpl()->shallow_copy_from(
        t.getIntrusivePtr());
  } else {
    MLIRTensor::copy_(*self_tensor, *src_tensor);
  }
  return self;
}

at::Tensor ATenMLIRType::_copy_from(const at::Tensor &self,
                                    const at::Tensor &dst, bool non_blocking) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");

  std::vector<at::Tensor> tensors = {self};
  auto device_tensors = bridge::MLIRCreateTensorList(tensors);
  // Hack in an overwrite of a const tensor.
  at::Tensor t = util::CopyTensor(device_tensors.front(), dst.scalar_type());
  const_cast<at::Tensor &>(dst).unsafeGetTensorImpl()->shallow_copy_from(
      t.getIntrusivePtr());
  return dst;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ATenMLIRType::convolution_backward_overrideable(
    const at::Tensor &grad_output, const at::Tensor &input,
    const at::Tensor &weight, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(input);
  auto weight_tensor =
      bridge::GetOrCreateMLIRTensor(weight, input_tensor.GetDevice());
  auto grad_output_tensor =
      bridge::GetOrCreateMLIRTensor(grad_output, input_tensor.GetDevice());

  auto ret = MLIRTensor::convolution_backward(
      grad_output_tensor, input_tensor, weight_tensor, stride, padding,
      dilation, transposed, output_padding, groups, output_mask);
  return std::make_tuple(bridge::AtenFromMLIRTensor(std::get<0>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<1>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<2>(ret)));
}

at::Tensor ATenMLIRType::convolution_overrideable(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(input);
  auto weight_tensor =
      bridge::GetOrCreateMLIRTensor(weight, input_tensor.GetDevice());

  auto bias_tensor =
      bias.defined()
          ? bridge::GetOrCreateMLIRTensor(bias, input_tensor.GetDevice())
          : bridge::GetOrCreateMLIRTensor(
                at::zeros(at::IntArrayRef{weight.sizes()[0]}),
                input_tensor.GetDevice());

  return bridge::AtenFromMLIRTensor(MLIRTensor::convolution(
      input_tensor, weight_tensor, bias_tensor, stride, padding, dilation,
      transposed, output_padding, groups));
}

at::Tensor ATenMLIRType::div(const at::Tensor &self, at::Scalar other) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  return bridge::AtenFromMLIRTensor(MLIRTensor::div(input_tensor, other));
}

at::Tensor ATenMLIRType::div(const at::Tensor &self, const at::Tensor &other) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensors = GetPromotedMLIRTensorsForBinaryOp(self, other);
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::div(std::get<0>(tensors), std::get<1>(tensors)));
}

at::Tensor &ATenMLIRType::div_(at::Tensor &self, const at::Tensor &other) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensors = GetPromotedMLIRTensorsForBinaryOp(self, other);
  auto result = bridge::AtenFromMLIRTensor(
      MLIRTensor::div_(std::get<0>(tensors), std::get<1>(tensors)));
  MLIRTensorImpl *self_impl =
      dynamic_cast<MLIRTensorImpl *>(self.unsafeGetTensorImpl());
  self_impl->shallow_copy_from(result.getIntrusivePtr());
  return self;
}

at::Tensor ATenMLIRType::expand(const at::Tensor &self, at::IntArrayRef size,
                                bool implicit) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::expand(input_tensor, size, implicit));
}

at::Tensor ATenMLIRType::gather(const at::Tensor &self, int64_t dim,
                                const at::Tensor &index, bool sparse_grad) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto index_tensor =
      bridge::GetOrCreateMLIRTensor(index, input_tensor.GetDevice());
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::gather(input_tensor, dim, index_tensor, sparse_grad));
}

at::Tensor ATenMLIRType::hardtanh(const at::Tensor &self, at::Scalar min_val,
                                  at::Scalar max_val) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto result = bridge::AtenFromMLIRTensor(
      MLIRTensor::hardtanh(input_tensor, min_val, max_val));
  MLIRTensorImpl *self_impl =
      dynamic_cast<MLIRTensorImpl *>(self.unsafeGetTensorImpl());
  self_impl->shallow_copy_from(result.getIntrusivePtr());
  return self;
}

at::Tensor &ATenMLIRType::hardtanh_(at::Tensor &self, at::Scalar min_val,
                                    at::Scalar max_val) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto result = bridge::AtenFromMLIRTensor(
      MLIRTensor::hardtanh_(input_tensor, min_val, max_val));
  MLIRTensorImpl *self_impl =
      dynamic_cast<MLIRTensorImpl *>(self.unsafeGetTensorImpl());
  self_impl->shallow_copy_from(result.getIntrusivePtr());
  return self;
}

at::Tensor ATenMLIRType::hardtanh_backward(const at::Tensor &grad_output,
                                           const at::Tensor &self,
                                           at::Scalar min_val,
                                           at::Scalar max_val) {
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto grad_output_tensor =
      bridge::GetOrCreateMLIRTensor(grad_output, input_tensor.GetDevice());
  return bridge::AtenFromMLIRTensor(MLIRTensor::hardtanh_backward(
      grad_output_tensor, input_tensor, min_val, max_val));
}

at::Tensor ATenMLIRType::_log_softmax(const at::Tensor &self, int64_t dim,
                                      bool half_to_float) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::_log_softmax(input_tensor, dim, half_to_float));
}

at::Tensor
ATenMLIRType::_log_softmax_backward_data(const at::Tensor &grad_output,
                                         const at::Tensor &output, int64_t dim,
                                         const at::Tensor &self) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto output_tensor =
      bridge::GetOrCreateMLIRTensor(output, input_tensor.GetDevice());
  auto grad_output_tensor =
      bridge::GetOrCreateMLIRTensor(grad_output, input_tensor.GetDevice());
  return bridge::AtenFromMLIRTensor(MLIRTensor::_log_softmax_backward_data(
      grad_output_tensor, output_tensor, dim, input_tensor));
}

std::tuple<at::Tensor, at::Tensor> ATenMLIRType::max_pool2d_with_indices(
    const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto ret = MLIRTensor::max_pool2d_with_indices(
      input_tensor, kernel_size, stride, padding, dilation, ceil_mode);
  return std::make_tuple(bridge::AtenFromMLIRTensor(std::get<0>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<1>(ret)));
}

at::Tensor ATenMLIRType::max_pool2d_with_indices_backward(
    const at::Tensor &grad_output, const at::Tensor &self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor &indices) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto grad_output_tensor =
      bridge::GetOrCreateMLIRTensor(grad_output, input_tensor.GetDevice());
  auto indices_tensor =
      bridge::GetOrCreateMLIRTensor(indices, input_tensor.GetDevice());

  return bridge::AtenFromMLIRTensor(
      MLIRTensor::max_pool2d_with_indices_backward(
          grad_output_tensor, input_tensor, kernel_size, stride, padding,
          dilation, ceil_mode, indices_tensor));
}

at::Tensor ATenMLIRType::mean(const at::Tensor &self,
                              c10::optional<at::ScalarType> dtype) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::mean(bridge::GetMLIRTensor(self), dtype));
}

at::Tensor ATenMLIRType::mean(const at::Tensor &self, at::IntArrayRef dim,
                              bool keepdim,
                              c10::optional<at::ScalarType> dtype) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::mean(bridge::GetMLIRTensor(self), dim, keepdim, dtype));
}

at::Tensor ATenMLIRType::mm(const at::Tensor &input, const at::Tensor &mat2) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(input);
  auto mat2_tensor =
      bridge::GetOrCreateMLIRTensor(mat2, input_tensor.GetDevice());
  return bridge::AtenFromMLIRTensor(MLIRTensor::mm(input_tensor, mat2_tensor));
}

at::Tensor ATenMLIRType::mul(const at::Tensor &self, const at::Tensor &other) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensors = GetPromotedMLIRTensorsForBinaryOp(self, other);
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::mul(std::get<0>(tensors), std::get<1>(tensors)));
}

at::Tensor &ATenMLIRType::mul_(at::Tensor &self, const at::Tensor &other) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensors = GetPromotedMLIRTensorsForBinaryOp(self, other);
  auto result = bridge::AtenFromMLIRTensor(
      MLIRTensor::mul_(std::get<0>(tensors), std::get<1>(tensors)));
  MLIRTensorImpl *self_impl =
      dynamic_cast<MLIRTensorImpl *>(self.unsafeGetTensorImpl());
  self_impl->shallow_copy_from(result.getIntrusivePtr());
  return self;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ATenMLIRType::native_batch_norm(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    const at::Tensor &running_mean, const at::Tensor &running_var,
    bool training, double momentum, double eps) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(input);
  auto weight_tensor =
      bridge::GetOrCreateMLIRTensor(weight, input_tensor.GetDevice());
  auto bias_tensor =
      bridge::GetOrCreateMLIRTensor(bias, input_tensor.GetDevice());
  auto running_mean_tensor =
      bridge::GetOrCreateMLIRTensor(running_mean, input_tensor.GetDevice());
  auto running_var_tensor =
      bridge::GetOrCreateMLIRTensor(running_var, input_tensor.GetDevice());

  auto ret = MLIRTensor::native_batch_norm(
      input_tensor, weight_tensor, bias_tensor, running_mean_tensor,
      running_var_tensor, training, momentum, eps);

  return std::make_tuple(bridge::AtenFromMLIRTensor(std::get<0>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<1>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<2>(ret)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ATenMLIRType::native_batch_norm_backward(
    const at::Tensor &grad_out, const at::Tensor &input,
    const at::Tensor &weight, const at::Tensor &running_mean,
    const at::Tensor &running_var, const at::Tensor &save_mean,
    const at::Tensor &save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(input);
  auto grad_out_tensor =
      bridge::GetOrCreateMLIRTensor(grad_out, input_tensor.GetDevice());
  auto weight_tensor =
      bridge::GetOrCreateMLIRTensor(weight, input_tensor.GetDevice());
  auto running_mean_tensor =
      bridge::GetOrCreateMLIRTensor(running_mean, input_tensor.GetDevice());
  auto running_var_tensor =
      bridge::GetOrCreateMLIRTensor(running_var, input_tensor.GetDevice());
  auto save_mean_tensor =
      bridge::GetOrCreateMLIRTensor(save_mean, input_tensor.GetDevice());
  auto save_invstd_tensor =
      bridge::GetOrCreateMLIRTensor(save_invstd, input_tensor.GetDevice());

  auto ret = MLIRTensor::native_batch_norm_backward(
      grad_out_tensor, input_tensor, weight_tensor, running_mean_tensor,
      running_var_tensor, save_mean_tensor, save_invstd_tensor, train, eps,
      output_mask);

  return std::make_tuple(bridge::AtenFromMLIRTensor(std::get<0>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<1>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<2>(ret)));
}

at::Tensor ATenMLIRType::neg(const at::Tensor &self) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  return bridge::AtenFromMLIRTensor(MLIRTensor::neg(input_tensor));
}

std::tuple<at::Tensor, at::Tensor> ATenMLIRType::nll_loss2d_forward(
    const at::Tensor &self, const at::Tensor &target, const at::Tensor &weight,
    int64_t reduction, int64_t ignore_index) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto target_tensor =
      bridge::GetOrCreateMLIRTensor(target, input_tensor.GetDevice());

  auto weight_tensor =
      weight.defined()
          ? bridge::GetOrCreateMLIRTensor(weight, input_tensor.GetDevice())
          : bridge::GetOrCreateMLIRTensor(at::ones(self.sizes()[1]),
                                          input_tensor.GetDevice());

  auto ret = MLIRTensor::nll_loss2d_forward(
      input_tensor, target_tensor, weight_tensor, reduction, ignore_index);

  return std::make_tuple(bridge::AtenFromMLIRTensor(std::get<0>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<1>(ret)));
}

at::Tensor ATenMLIRType::nll_loss2d_backward(
    const at::Tensor &grad_output, const at::Tensor &self,
    const at::Tensor &target, const at::Tensor &weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor &total_weight) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto grad_output_tensor =
      bridge::GetOrCreateMLIRTensor(grad_output, input_tensor.GetDevice());
  auto target_tensor =
      bridge::GetOrCreateMLIRTensor(target, input_tensor.GetDevice());

  auto weight_tensor =
      weight.defined()
          ? bridge::GetOrCreateMLIRTensor(weight, input_tensor.GetDevice())
          : bridge::GetOrCreateMLIRTensor(at::ones(self.sizes()[1]),
                                          input_tensor.GetDevice());
  auto total_weight_tensor =
      bridge::GetOrCreateMLIRTensor(total_weight, input_tensor.GetDevice());

  return bridge::AtenFromMLIRTensor(MLIRTensor::nll_loss2d_backward(
      grad_output_tensor, input_tensor, target_tensor, weight_tensor, reduction,
      ignore_index, total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor>
ATenMLIRType::nll_loss_forward(const at::Tensor &self, const at::Tensor &target,
                               const at::Tensor &weight, int64_t reduction,
                               int64_t ignore_index) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto target_tensor =
      bridge::GetOrCreateMLIRTensor(target, input_tensor.GetDevice());

  auto weight_tensor =
      weight.defined()
          ? bridge::GetOrCreateMLIRTensor(weight, input_tensor.GetDevice())
          : bridge::GetOrCreateMLIRTensor(at::ones(self.sizes()[1]),
                                          input_tensor.GetDevice());

  auto ret = MLIRTensor::nll_loss_forward(
      input_tensor, target_tensor, weight_tensor, reduction, ignore_index);

  return std::make_tuple(bridge::AtenFromMLIRTensor(std::get<0>(ret)),
                         bridge::AtenFromMLIRTensor(std::get<1>(ret)));
}

at::Tensor ATenMLIRType::nll_loss_backward(
    const at::Tensor &grad_output, const at::Tensor &self,
    const at::Tensor &target, const at::Tensor &weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor &total_weight) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto grad_output_tensor =
      bridge::GetOrCreateMLIRTensor(grad_output, input_tensor.GetDevice());
  auto target_tensor =
      bridge::GetOrCreateMLIRTensor(target, input_tensor.GetDevice());

  auto weight_tensor =
      weight.defined()
          ? bridge::GetOrCreateMLIRTensor(weight, input_tensor.GetDevice())
          : bridge::GetOrCreateMLIRTensor(at::ones(self.sizes()[1]),
                                          input_tensor.GetDevice());
  auto total_weight_tensor =
      bridge::GetOrCreateMLIRTensor(total_weight, input_tensor.GetDevice());

  return bridge::AtenFromMLIRTensor(MLIRTensor::nll_loss_backward(
      grad_output_tensor, input_tensor, target_tensor, weight_tensor, reduction,
      ignore_index, total_weight_tensor));
}

at::Tensor ATenMLIRType::relu(const at::Tensor &self) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::relu(bridge::GetMLIRTensor(self)));
}

at::Tensor &ATenMLIRType::relu_(at::Tensor &self) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto result = bridge::AtenFromMLIRTensor(MLIRTensor::relu_(input_tensor));
  MLIRTensorImpl *self_impl =
      dynamic_cast<MLIRTensorImpl *>(self.unsafeGetTensorImpl());
  self_impl->shallow_copy_from(result.getIntrusivePtr());
  return self;
}

int64_t ATenMLIRType::size(const at::Tensor &self, int64_t dim) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::GetMLIRTensor(self).sizes()[dim];
}

at::Tensor ATenMLIRType::squeeze(const at::Tensor &self, int64_t dim) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::squeeze(bridge::GetMLIRTensor(self), dim));
}

at::Tensor ATenMLIRType::sub(const at::Tensor &self, const at::Tensor &other,
                             at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensors = GetPromotedMLIRTensorsForBinaryOp(self, other);
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::sub(std::get<0>(tensors), std::get<1>(tensors), alpha));
}

at::Tensor &ATenMLIRType::sub_(at::Tensor &self, const at::Tensor &other,
                               at::Scalar alpha) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto tensors = GetPromotedMLIRTensorsForBinaryOp(self, other);
  auto result = bridge::AtenFromMLIRTensor(
      MLIRTensor::sub_(std::get<0>(tensors), std::get<1>(tensors), alpha));
  MLIRTensorImpl *self_impl =
      dynamic_cast<MLIRTensorImpl *>(self.unsafeGetTensorImpl());
  self_impl->shallow_copy_from(result.getIntrusivePtr());
  return self;
}

at::Tensor ATenMLIRType::sum(const at::Tensor &self, at::IntArrayRef dim,
                             bool keepdim,
                             c10::optional<at::ScalarType> dtype) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::sum(bridge::GetMLIRTensor(self), dim, keepdim, dtype));
}

at::Tensor ATenMLIRType::t(const at::Tensor &self) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(MLIRTensor::t(bridge::GetMLIRTensor(self)));
}

at::Tensor ATenMLIRType::threshold_backward(const at::Tensor &grad_output,
                                            const at::Tensor &self,
                                            at::Scalar threshold) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  auto input_tensor = bridge::GetMLIRTensor(self);
  auto grad_output_tensor =
      bridge::GetOrCreateMLIRTensor(grad_output, input_tensor.GetDevice());
  return bridge::AtenFromMLIRTensor(MLIRTensor::threshold_backward(
      grad_output_tensor, input_tensor, threshold));
}

at::Tensor ATenMLIRType::to(const at::Tensor &self,
                            const at::TensorOptions &options,
                            bool /* non_blocking */, bool /* copy */) {

  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");

  auto self_tensor = bridge::TryGetMLIRTensor(self);
  if (!self_tensor) {
    assert(options.has_device());
    at::ScalarType dtype = options.has_dtype()
                               ? c10::typeMetaToScalarType(options.dtype())
                               : self.scalar_type();
    MLIRTensor xtensor =
        MLIRTensor::Create(util::CopyTensor(self, dtype),
                           bridge::AtenDeviceToMLIRDevice(options.device()));
    return bridge::AtenFromMLIRTensor(xtensor);
  }
  if (options.has_device() && options.device().type() != at::kXLA) {
    return bridge::MLIRToAtenTensor(*self_tensor, options);
  }
  MLIROptions mlir_options(options, self_tensor->GetDevice(),
                           self_tensor->dtype());
  return bridge::AtenFromMLIRTensor(MLIRTensor::to(
      *self_tensor, mlir_options.device, mlir_options.scalar_type));
}

at::Tensor ATenMLIRType::to(const at::Tensor &self, c10::Device device,
                            at::ScalarType dtype, bool non_blocking,
                            bool copy) {
  return to(self, self.options().device(device).dtype(dtype), non_blocking,
            copy);
}

at::Tensor ATenMLIRType::to(const at::Tensor &self, at::ScalarType dtype,
                            bool non_blocking, bool copy) {
  return to(self, self.options().dtype(dtype), non_blocking, copy);
}

at::Tensor ATenMLIRType::to(const at::Tensor &self, const at::Tensor &other,
                            bool non_blocking, bool copy) {
  return to(self, other.options(), non_blocking, copy);
}

at::Tensor ATenMLIRType::_unsafe_view(const at::Tensor &self,
                                      at::IntArrayRef size) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::view(bridge::GetMLIRTensor(self), size));
}

at::Tensor ATenMLIRType::unsqueeze(const at::Tensor &self, int64_t dim) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::unsqueeze(bridge::GetMLIRTensor(self), dim));
}

at::Tensor ATenMLIRType::view(const at::Tensor &self, at::IntArrayRef size) {
  LLVM_DEBUG(llvm::dbgs() << "ATenMLIRType::" << __func__ << "\n");
  return bridge::AtenFromMLIRTensor(
      MLIRTensor::view(bridge::GetMLIRTensor(self), size));
}
} // namespace torch_mlir
