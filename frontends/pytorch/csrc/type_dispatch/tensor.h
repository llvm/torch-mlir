//===- tensor.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "device.h"
#include "ir.h"

#include <cstdint>

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace torch_mlir {

class MLIRTensor {
  struct Data;

public:
  static MLIRTensor Create(const at::Tensor &tensor, const Device &device);
  static MLIRTensor Create(ir::Value ir_value, const Device &device,
                           c10::optional<at::ScalarType> logical_element_type);

  MLIRTensor() = default;

  bool is_null() const { return data_ptr() == nullptr; }

  void ShallowCopyTo(MLIRTensor *dest) const;

  void SetTensor(at::Tensor tensor);
  void SetIrValue(ir::Value ir_value);

  at::ScalarType dtype() const;

  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<at::ScalarType> logical_element_type);

  std::vector<int64_t> sizes() const;
  std::vector<int64_t> strides() const;

  at::Tensor ToTensor() const;

  const Device &GetDevice() const;

  size_t generation() const { return data()->generation; }

  std::string GetMLIR() const;

  // Retrieves the IR Node representing this MLIRTensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state of the object.
  ir::Value GetIrValue() const;

  at::Tensor CompileAndRun() const;

  uint64_t id() const { return data()->unique_id; }

private:
  struct Data {
    Data(at::Tensor tensor_data, const Device &device)
        : logical_element_type(tensor_data.scalar_type()),
          tensor_data(std::move(tensor_data)), device(device),
          unique_id(GetNextTensorId()) {}

    Data(ir::Value ir_value, const Device &device,
         c10::optional<at::ScalarType> logical_element_type)
        : logical_element_type(logical_element_type),
          ir_value(std::move(ir_value)), device(device),
          unique_id(GetNextTensorId()) {}

    ~Data(){};

    c10::optional<at::ScalarType> logical_element_type;
    c10::optional<at::Tensor> tensor_data;
    ir::Value ir_value;

    const Device device;
    const uint64_t unique_id = 0;
    size_t generation = 1;
  };

  MLIRTensor(const at::Tensor &tensor, const Device &device);

  MLIRTensor(ir::Value ir_value, const Device &device,
             c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  void SetTensorData(at::Tensor tensor_data);

  c10::optional<at::Tensor> CurrentTensorData() const;

  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  ir::Value CurrentIrValue() const;

  Data *data() const;

  std::shared_ptr<Data> data_ptr() const { return data_; }

  MLIRTensor CreateFrom(ir::Value ir_value) const;

  static uint64_t GetNextTensorId();

  std::shared_ptr<Data> data_;

  //////////////////////////////////////////////////////////////////////////////
  // ATEN operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
public:
  static MLIRTensor _adaptive_avg_pool2d(const MLIRTensor &self,
                                         at::IntArrayRef output_size);

  static MLIRTensor _adaptive_avg_pool2d_backward(const MLIRTensor &grad_output,
                                                  const MLIRTensor &self);

  static MLIRTensor add(const MLIRTensor &input, const MLIRTensor &other,
                        at::Scalar alpha);

  static MLIRTensor add_(MLIRTensor &input, const MLIRTensor &other,
                         at::Scalar alpha);

  static MLIRTensor addmm(const MLIRTensor &input, const MLIRTensor &mat1,
                          const MLIRTensor &mat2, at::Scalar beta,
                          at::Scalar alpha);

  static MLIRTensor as_strided(const MLIRTensor &self, at::IntArrayRef size,
                               at::IntArrayRef stride,
                               c10::optional<int64_t> storage_offset);

  static MLIRTensor clone(const MLIRTensor &self);

  static MLIRTensor convolution(const MLIRTensor &input,
                                const MLIRTensor &weight,
                                const MLIRTensor &bias, at::IntArrayRef stride,
                                at::IntArrayRef padding,
                                at::IntArrayRef dilation, bool transposed,
                                at::IntArrayRef output_padding, int64_t groups);

  static std::tuple<MLIRTensor, MLIRTensor, MLIRTensor>
  convolution_backward(const MLIRTensor &grad_output, const MLIRTensor &input,
                       const MLIRTensor &weight, at::IntArrayRef stride,
                       at::IntArrayRef padding, at::IntArrayRef dilation,
                       bool transposed, at::IntArrayRef output_padding,
                       int64_t groups, std::array<bool, 3> output_mask);

  static void copy_(MLIRTensor &input, MLIRTensor &src);

  static MLIRTensor div(const MLIRTensor &self, at::Scalar other);

  static MLIRTensor div(const MLIRTensor &self, const MLIRTensor &other);

  static MLIRTensor div_(MLIRTensor &self, const MLIRTensor &other);

  static MLIRTensor expand(const MLIRTensor &self, at::IntArrayRef size,
                           bool implicit);

  static MLIRTensor gather(const MLIRTensor &self, int64_t dim,
                           const MLIRTensor &index, bool sparse_grad);

  static MLIRTensor hardtanh(const MLIRTensor &self, at::Scalar min_val,
                             at::Scalar max_val);

  static MLIRTensor hardtanh_(MLIRTensor &self, at::Scalar min_val,
                              at::Scalar max_val);

  static MLIRTensor hardtanh_backward(const MLIRTensor &grad_output,
                                      const MLIRTensor &self,
                                      at::Scalar min_val, at::Scalar max_val);

  static MLIRTensor _log_softmax(const MLIRTensor &input, int64_t dim,
                                 bool half_to_float);

  static MLIRTensor _log_softmax_backward_data(const MLIRTensor &grad_output,
                                               const MLIRTensor &output,
                                               int64_t dim,
                                               const MLIRTensor &self);

  static std::tuple<MLIRTensor, MLIRTensor>
  max_pool2d_with_indices(const MLIRTensor &input, at::IntArrayRef kernel_size,
                          at::IntArrayRef stride, at::IntArrayRef padding,
                          at::IntArrayRef dilation, bool ceil_mode);

  static MLIRTensor max_pool2d_with_indices_backward(
      const MLIRTensor &grad_output, const MLIRTensor &self,
      at::IntArrayRef kernel_size, at::IntArrayRef stride,
      at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
      const MLIRTensor &indices);

  static MLIRTensor mean(const MLIRTensor &input,
                         c10::optional<at::ScalarType> dtype);

  static MLIRTensor mean(const MLIRTensor &input, at::IntArrayRef dim,
                         bool keepdim, c10::optional<at::ScalarType> dtype);

  static MLIRTensor mm(const MLIRTensor &input, const MLIRTensor &mat1);

  static MLIRTensor mul(const MLIRTensor &self, const MLIRTensor &other);

  static MLIRTensor mul_(MLIRTensor &self, const MLIRTensor &other);

  static std::tuple<MLIRTensor, MLIRTensor, MLIRTensor>
  native_batch_norm(const MLIRTensor &input, const MLIRTensor &weight,
                    const MLIRTensor &bias, const MLIRTensor &running_mean,
                    const MLIRTensor &running_var, bool training,
                    double momentum, double eps);

  static std::tuple<MLIRTensor, MLIRTensor, MLIRTensor>
  native_batch_norm_backward(const MLIRTensor &grad_out,
                             const MLIRTensor &input, const MLIRTensor &weight,
                             const MLIRTensor &running_mean,
                             const MLIRTensor &running_var,
                             const MLIRTensor &save_mean,
                             const MLIRTensor &save_invstd, bool train,
                             double eps, std::array<bool, 3> output_mask);

  static MLIRTensor neg(const MLIRTensor &input);

  static std::tuple<MLIRTensor, MLIRTensor>
  nll_loss2d_forward(const MLIRTensor &self, const MLIRTensor &target,
                     const MLIRTensor &weight, int64_t reduction,
                     int64_t ignore_index);

  static MLIRTensor nll_loss2d_backward(const MLIRTensor &grad_output,
                                        const MLIRTensor &self,
                                        const MLIRTensor &target,
                                        const MLIRTensor &weight,
                                        int64_t reduction, int64_t ignore_index,
                                        const MLIRTensor &total_weight);

  static std::tuple<MLIRTensor, MLIRTensor>
  nll_loss_forward(const MLIRTensor &self, const MLIRTensor &target,
                   const MLIRTensor &weight, int64_t reduction,
                   int64_t ignore_index);

  static MLIRTensor nll_loss_backward(const MLIRTensor &grad_output,
                                      const MLIRTensor &self,
                                      const MLIRTensor &target,
                                      const MLIRTensor &weight,
                                      int64_t reduction, int64_t ignore_index,
                                      const MLIRTensor &total_weight);

  static MLIRTensor size(const MLIRTensor &self, int64_t dim);

  static MLIRTensor squeeze(const MLIRTensor &self, int64_t dim);

  static MLIRTensor sub(const MLIRTensor &input, const MLIRTensor &other,
                        at::Scalar alpha);

  static MLIRTensor sub_(MLIRTensor &input, const MLIRTensor &other,
                         at::Scalar alpha);

  static MLIRTensor sum(const MLIRTensor &self, at::IntArrayRef dim,
                        bool keepdim, c10::optional<at::ScalarType> dtype);

  static MLIRTensor relu(const MLIRTensor &input);

  static MLIRTensor relu_(MLIRTensor &input);

  static MLIRTensor t(const MLIRTensor &input);

  static MLIRTensor threshold_backward(const MLIRTensor &grad_output,
                                       const MLIRTensor &self,
                                       at::Scalar threshold);

  static MLIRTensor to(MLIRTensor &input, c10::optional<Device> device,
                       c10::optional<at::ScalarType> scalar_type);

  static MLIRTensor unsqueeze(const MLIRTensor &self, int64_t dim);

  static MLIRTensor view(const MLIRTensor &input, at::IntArrayRef size);
};
} // namespace torch_mlir
