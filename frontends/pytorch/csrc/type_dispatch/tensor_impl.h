//===- tensor_impl.h --------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tensor.h"

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

namespace torch_mlir {

class MLIRTensorImpl : public c10::TensorImpl {
public:
  explicit MLIRTensorImpl(MLIRTensor tensor);

  MLIRTensor &tensor() { return tensor_; }

  c10::intrusive_ptr<TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) override;

  at::IntArrayRef sizes() const override;

  at::IntArrayRef strides() const override;

  int64_t dim() const override;

  int64_t numel() const override;

  bool is_contiguous(at::MemoryFormat memory_format) const override;

  int64_t size(int64_t d) const override;

  static c10::Device GetCurrentAtenDevice();

  static c10::Device SetCurrentAtenDevice(c10::Device device);

  static void AtenInitialize();

  const at::Storage &storage() const override;

  bool has_storage() const override;

private:
  static caffe2::TypeMeta GetTypeMeta(const MLIRTensor &tensor);

  void SetupSizeProperties();

  MLIRTensor tensor_;
  size_t generation_ = 0;
};
} // namespace torch_mlir
