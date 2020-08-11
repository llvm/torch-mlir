//===- aten_ops.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// This file implements C libraries that are targetted by MLIR code generation
// from the ATen dialect.  This library is intended to support a functional
// proof of concept rather than optimized for high performance.  Most of the
// functions are implemented by calling back into the torch libraries.

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "nnpack.h"
#include <ATen/CPUType.h>

namespace {

template <typename T, int N> struct tensor_t {
  T *d;
  T *aligned;
  size_t offset;
  size_t shape[N];
  size_t stride[N];

  size_t index(size_t n, size_t channel, size_t row, size_t col) const {
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    return n * height * width * channels + channel * height * width +
           row * width + col;
  }

  tensor_t() {
    d = aligned = nullptr;
    offset = 0;
    for (int i = 0; i < N; i++)
      shape[i] = stride[i] = 0;
  }
};

template <typename T, int N>
std::vector<int64_t> translate_shape(tensor_t<T, N> *t) {
  std::vector<int64_t> shape;
  for (int i = 0; i < N; i++) {
    shape.push_back(t->shape[i]);
    // std::cout << i << " shape " << t->shape[i] << std::endl;
  }
  return shape;
}

template <typename T, int N>
std::vector<int64_t> translate_stride(tensor_t<T, N> *t) {
  std::vector<int64_t> stride;
  for (int i = 0; i < N; i++) {
    stride.push_back(t->stride[i]);
    // std::cout << i << " stride " << t->stride[i] << std::endl;
  }
  return stride;
}

template <int N> void dumpTensor(std::ostream &o, tensor_t<float, N> *t) {
  o << "Shape:";
  for (int i = 0; i < N; i++)
    o << t->shape[i] << " ";
  o << "Stride:";
  for (int i = 0; i < N; i++)
    o << t->stride[i] << " ";
  o << "\n";
}

template <typename T, int N>
at::Tensor to_torch(tensor_t<T, N> *t,
                    const at::TensorOptions &options = at::TensorOptions()) {
  // std::cout << "to_torch\n";
  return torch::from_blob((void *)t->d, translate_shape(t), translate_stride(t),
                          options);
}

template <typename T>
void mm_out(tensor_t<T, 2> *a, tensor_t<T, 2> *b, tensor_t<T, 2> *r);

template <typename T, int N>
void add_out(tensor_t<T, N> *a, tensor_t<T, N> *b, T alpha, tensor_t<T, N> *r) {
  at::Tensor torch_a = to_torch(a);
  at::Tensor torch_b = to_torch(b);
  at::Tensor result = at::native::add(torch_a, torch_b, alpha).clone();

  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T>
void addmm_out(tensor_t<T, 1> *a, tensor_t<T, 2> *b, tensor_t<T, 2> *c,
               int32_t alpha, int32_t beta, tensor_t<T, 2> *r) {
  at::Tensor torch_a = to_torch(a);
  at::Tensor torch_b = to_torch(b);
  at::Tensor torch_c = to_torch(c);
  at::Tensor result =
      at::native::addmm(torch_a, torch_b, torch_c, alpha, beta).clone();

  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T, int N, int M>
void as_strided_out(tensor_t<float, M> *a,
                    /*size*/ int32_t sz0, int32_t sz1, int32_t sz2, int32_t sz3,
                    /*stride*/ int32_t sd0, int32_t sd1, int32_t sd2,
                    int32_t sd3, int32_t offset, tensor_t<T, N> *r) {
  at::Tensor input = to_torch(a);

  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  c10::optional<int64_t> storage_offset;

  if (offset != 0)
    storage_offset = offset;
  if (N > 0) {
    size.push_back(sz0);
    stride.push_back(sd0);
  }
  if (N > 1) {
    size.push_back(sz1);
    stride.push_back(sd1);
  }
  if (N > 2) {
    size.push_back(sz2);
    stride.push_back(sd2);
  }
  if (N > 3) {
    size.push_back(sz3);
    stride.push_back(sd3);
  }

  std::vector<int64_t> sizeRef{size};
  std::vector<int64_t> strideRef{stride};

  // for (int i = 0; i<N; i++)
  //  std::cout << "STRIDE " << i << " " << stride[i] << std::endl;
  at::Tensor result =
      at::native::as_strided_tensorimpl(input, size, stride, storage_offset)
          .clone();

  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

// FIXME: stride, padding, dilaection, output_padding should be IntArrayRef
template <typename T>
void conv2d_out(tensor_t<T, 4> *t, tensor_t<T, 4> *weight, tensor_t<T, 1> *bias,
                int32_t stride, int32_t pad, int32_t dilation,
                tensor_t<T, 4> *r) {
  at::Tensor torch_t = to_torch(t);
  at::Tensor torch_w = to_torch(weight);
  at::Tensor torch_b = to_torch(bias);
  int64_t groups = 1;

  at::Tensor result = at::native::conv2d(torch_t, torch_w, torch_b, stride, pad,
                                         dilation, groups)
                          .clone();

  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T>
void conv2d_backward_out(tensor_t<T, 4> *grad_output, tensor_t<T, 4> *input,
                         tensor_t<T, 4> *weight, int32_t stride, int32_t pad,
                         int32_t dilation, tensor_t<T, 4> *r0,
                         tensor_t<T, 4> *r1, tensor_t<T, 1> *r2) {
  const at::Tensor &arg_grad = to_torch(grad_output);
  const at::Tensor &arg_input = to_torch(input);
  const at::Tensor &arg_weight = to_torch(weight);

  std::vector<int64_t> p{pad, pad};
  std::vector<int64_t> s{stride, stride};
  std::vector<int64_t> d{dilation, dilation};

  std::array<bool, 3> output_mask{true, true, true};

  std::tuple<at::Tensor, at::Tensor, at::Tensor> grads =
      at::native::mkldnn_convolution_backward(arg_input, arg_grad, arg_weight,
                                              p, s, d, 1, output_mask);

  auto result0 = std::get<0>(grads);
  auto result1 = std::get<1>(grads);
  auto result2 = std::get<2>(grads);

  memcpy(r0->d, result0.data_ptr(), result0.numel() * sizeof(T));
  memcpy(r1->d, result1.data_ptr(), result1.numel() * sizeof(T));
  memcpy(r2->d, result2.data_ptr(), result2.numel() * sizeof(T));
}

template <typename T, int N>
void log_softmax_out(tensor_t<T, N> *t, int32_t dim, bool half_to_float,
                     tensor_t<T, N> *r) {
  at::Tensor input = to_torch(t);
  at::Tensor result = at::native::log_softmax_cpu(input, dim, half_to_float);
  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T, int N>
void log_softmax_backward_data_out(tensor_t<T, N> *a, tensor_t<T, N> *b,
                                   int32_t c, tensor_t<T, N> *d,
                                   tensor_t<T, N> *r) {
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);
  at::Tensor inputD = to_torch(d);

  at::Tensor result =
      at::native::log_softmax_backward_cpu(inputA, inputB, c, inputD);
  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T>
void max_pool2d_with_indices_out(tensor_t<T, 4> *t, int32_t c, int32_t d,
                                 int32_t e, int32_t f, bool ceil_mode,
                                 tensor_t<T, 4> *r0, tensor_t<int64_t, 4> *r1) {
  at::Tensor input = to_torch(t);

  std::vector<int64_t> kernel{c, c};
  std::vector<int64_t> stride{d, d};
  std::vector<int64_t> padding{e, e};
  std::vector<int64_t> dilation{f, f};

  auto result = at::native::max_pool2d_with_indices_cpu(
      input, kernel, stride, padding, dilation, ceil_mode);
  at::Tensor outTensor = std::get<0>(result);
  at::Tensor idxTensor = std::get<1>(result);
  memcpy(r0->d, outTensor.data_ptr(), outTensor.numel() * sizeof(T));
  memcpy(r1->d, idxTensor.data_ptr(), idxTensor.numel() * sizeof(T));
}

template <typename T>
void max_pool2d_with_indices_backward_out(tensor_t<T, 4> *a, tensor_t<T, 4> *b,
                                          int32_t c, int32_t d, int32_t e,
                                          int32_t f, bool g,
                                          tensor_t<int64_t, 4> *h,
                                          tensor_t<T, 4> *r) {
  const at::Tensor &inputA = to_torch(a);
  const at::Tensor &inputB = to_torch(b);
  at::TensorOptions options(at::ScalarType::Long);
  const at::Tensor &inputH = to_torch(h, options);

  std::vector<int64_t> kernel{c, c};
  std::vector<int64_t> stride{d, d};
  std::vector<int64_t> padding{e, e};
  std::vector<int64_t> dilation{f, f};

  at::Tensor result = at::native::max_pool2d_with_indices_backward_cpu(
      inputA, inputB, kernel, stride, padding, dilation, g, inputH);
  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T>
void mm_out(tensor_t<T, 2> *a, tensor_t<T, 2> *b, tensor_t<T, 2> *r) {
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);

  at::Tensor result = inputA.matmul(inputB);
  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T, int N>
void mul_out(tensor_t<T, N> *a, tensor_t<T, N> *b, tensor_t<T, N> *r) {
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);

  at::Tensor result = at::native::mul(inputA, inputB);
  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T, int N>
void relu_out(tensor_t<T, N> *a, tensor_t<T, N> *r) {
  at::Tensor inputA = to_torch(a);

  at::Tensor result = at::native::relu(inputA);
  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T> void t_out(tensor_t<T, 2> *a, tensor_t<T, 2> *r) {
  size_t h = a->shape[0];
  size_t w = a->shape[1];

  for (size_t i = 0; i < h; i++)
    for (size_t j = 0; j < w; j++)
      r->d[j * h + i] = a->d[i * w + j];
}

template <typename T, int N>
void threshold_backward_out(tensor_t<T, N> *a, tensor_t<T, N> *b, int32_t c,
                            tensor_t<T, N> *r) {
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);

  at::Tensor result = at::native::threshold_backward(inputA, inputB, c);
  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

template <typename T, int N, int M>
void view_out(tensor_t<T, M> *a, int32_t b, int32_t c, int32_t d, int32_t e,
              tensor_t<T, N> *r) {
  tensor_t<T, N> result;
  size_t numel = 1;
  for (size_t d = 0; d < M; d++)
    numel *= a->shape[d];

  if (N == 1)
    c = d = e = 1;
  if (N == 2)
    d = e = 1;
  if (N == 3)
    e = 1;

  int inferred = 0;
  if (b == -1)
    inferred++;
  if (c == -1)
    inferred++;
  if (d == -1)
    inferred++;
  if (e == -1)
    inferred++;
  assert(inferred <= 1 &&
         "aten.view Error: only one dimension can be inferred");

  if (b == -1)
    b = numel / (c * d * e);
  if (c == -1)
    c = numel / (b * d * e);
  if (d == -1)
    d = numel / (b * c * e);
  if (e == -1)
    e = numel / (b * c * d);

  if (N > 0)
    r->shape[0] = b;
  if (N > 1)
    r->shape[1] = c;
  if (N > 2)
    r->shape[2] = d;
  if (N > 3)
    r->shape[3] = e;

  memcpy(r->d, a->d, numel * sizeof(T));
}

} // namespace

extern "C" {

// add_out

void _mlir_ciface_add_1F32_1F32_1F32_out(tensor_t<float, 1> *a,
                                         tensor_t<float, 1> *b, int32_t i,
                                         tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  add_out<float, 1>(a, b, i, r);
}

void _mlir_ciface_add_2F32_2F32_2F32_out(tensor_t<float, 2> *a,
                                         tensor_t<float, 2> *b, int32_t i,
                                         tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  add_out<float, 2>(a, b, i, r);
}

void _mlir_ciface_add_3F32_3F32_3F32_out(tensor_t<float, 3> *a,
                                         tensor_t<float, 3> *b, int32_t i,
                                         tensor_t<float, 3> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  add_out<float, 3>(a, b, i, r);
}

void _mlir_ciface_add_4F32_4F32_4F32_out(tensor_t<float, 4> *a,
                                         tensor_t<float, 4> *b, int32_t i,
                                         tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  add_out<float, 4>(a, b, i, r);
}

// addmm_out

void _mlir_ciface_addmm_2F32_1F32_2F32_2F32_out(tensor_t<float, 1> *a,
                                                tensor_t<float, 2> *b,
                                                tensor_t<float, 2> *c,
                                                int32_t alpha, int32_t beta,
                                                tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  addmm_out<float>(a, b, c, alpha, beta, r);
}

// as_strided_out

void _mlir_ciface_as_strided_1F32_1F32_out(tensor_t<float, 1> *a,
                                           /*size*/ int32_t sz0, int32_t sz1,
                                           int32_t sz2, int32_t sz3,
                                           /*stride*/ int32_t sd0, int32_t sd1,
                                           int32_t sd2, int32_t sd3,
                                           int32_t offset,
                                           tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  as_strided_out<float, 1, 1>(a, sz0, sz1, sz2, sz3, sd0, sd1, sd2, sd3, offset,
                              r);
}

void _mlir_ciface_as_strided_4F32_2F32_out(tensor_t<float, 2> *a,
                                           /*size*/ int32_t sz0, int32_t sz1,
                                           int32_t sz2, int32_t sz3,
                                           /*stride*/ int32_t sd0, int32_t sd1,
                                           int32_t sd2, int32_t sd3,
                                           int32_t offset,
                                           tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  // std::cout << sz0 << " "
  //           << sz1 << " "
  //           << sz2 << " "
  //           << sz3 << "\n";
  // std::cout << sd0 << " "
  //           << sd1 << " "
  //           << sd2 << " "
  //           << sd3 << "\n";
  as_strided_out<float, 4, 2>(a, sz0, sz1, sz2, sz3, sd0, sd1, sd2, sd3, offset,
                              r);
}

// conv2d_out

void _mlir_ciface_conv2d_4F32_4F32_4F32_1F32_out(
    tensor_t<float, 4> *t, tensor_t<float, 4> *weight, tensor_t<float, 1> *bias,
    int32_t stride, int32_t padding, int32_t dilation, tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  conv2d_out<float>(t, weight, bias, stride, padding, dilation, r);
}

void _mlir_ciface_conv2d_relu_4F32_4F32_4F32_1F32_out(
    tensor_t<float, 4> *t, tensor_t<float, 4> *weight, tensor_t<float, 1> *bias,
    int32_t stride, int32_t padding, int32_t dilation, tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  conv2d_out<float>(t, weight, bias, stride, padding, dilation, r);
  relu_out<float, 4>(r, r);
}

// conv2d_backward_out

void _mlir_ciface_conv2d_backward_4F32_4F32_1F32_4F32_4F32_4F32_out(
    tensor_t<float, 4> *grad_output, tensor_t<float, 4> *t,
    tensor_t<float, 4> *weight, int32_t stride, int32_t padding,
    int32_t dilation, tensor_t<float, 4> *r0, tensor_t<float, 4> *r1,
    tensor_t<float, 1> *r2) {
  // std::cout << "aten_ops " << __func__ << "\n";
  conv2d_backward_out<float>(grad_output, t, weight, stride, padding, dilation,
                             r0, r1, r2);
}

// div
float *div_0F32_0F32_0F32(float *a, float *b) {
  // std::cout << "aten_ops " << __func__ << "\n";
  float *ret = (float *)malloc(sizeof(float));
  *ret = *a / *b;
  return ret;
}

// log_softmax_out

void _mlir_ciface_log_softmax_1F32_1F32_out(tensor_t<float, 1> *t, int32_t dim,
                                            bool half_to_float,
                                            tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_out<float, 1>(t, dim, half_to_float, r);
}
void _mlir_ciface_log_softmax_2F32_2F32_out(tensor_t<float, 2> *t, int32_t dim,
                                            bool half_to_float,
                                            tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_out<float, 2>(t, dim, half_to_float, r);
}
void _mlir_ciface_log_softmax_3F32_3F32_out(tensor_t<float, 3> *t, int32_t dim,
                                            bool half_to_float,
                                            tensor_t<float, 3> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_out<float, 3>(t, dim, half_to_float, r);
}
void _mlir_ciface_log_softmax_4F32_4F32_out(tensor_t<float, 4> *t, int32_t dim,
                                            bool half_to_float,
                                            tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_out<float, 4>(t, dim, half_to_float, r);
}

// log_softmax_backward_data_out

void _mlir_ciface_log_softmax_backward_data_2F32_2F32_2F32_2F32_out(
    tensor_t<float, 2> *a, tensor_t<float, 2> *b, int32_t c,
    tensor_t<float, 2> *d, tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_backward_data_out<float, 2>(a, b, c, d, r);
}

void _mlir_ciface_log_softmax_backward_data_4F32_4F32_4F32_4F32_out(
    tensor_t<float, 4> *a, tensor_t<float, 4> *b, int32_t c,
    tensor_t<float, 4> *d, tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  log_softmax_backward_data_out<float, 4>(a, b, c, d, r);
}

// max_pool2d_out

void _mlir_ciface_max_pool2d_with_indices_4F32_4I64_4F32_out(
    tensor_t<float, 4> *t, int32_t kernel, int32_t pad, int32_t stride,
    int32_t dilation, bool ceil_mode, tensor_t<float, 4> *r0,
    tensor_t<int64_t, 4> *r1) {
  // std::cout << "aten_ops " << __func__ << "\n";
  max_pool2d_with_indices_out<float>(t, kernel, pad, stride, dilation,
                                     ceil_mode, r0, r1);
}

// max_pool2d backward_out

void _mlir_ciface_max_pool2d_with_indices_backward_4F32_4F32_4F32_4I64_out(
    tensor_t<float, 4> *a, tensor_t<float, 4> *b, int32_t c, int32_t d,
    int32_t e, int32_t f, bool g, tensor_t<int64_t, 4> *h,
    tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  max_pool2d_with_indices_backward_out<float>(a, b, c, d, e, f, g, h, r);
}

// mm_out

void _mlir_ciface_mm_2F32_2F32_2F32_out(tensor_t<float, 2> *a,
                                        tensor_t<float, 2> *b,
                                        tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  mm_out<float>(a, b, r);
}

// mul_out

void _mlir_ciface_mul_1F32_1F32_1F32_out(tensor_t<float, 1> *a,
                                         tensor_t<float, 1> *b,
                                         tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  mul_out<float, 1>(a, b, r);
}

void _mlir_ciface_mul_2F32_2F32_2F32_out(tensor_t<float, 2> *a,
                                         tensor_t<float, 2> *b,
                                         tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  mul_out<float, 2>(a, b, r);
}

void _mlir_ciface_mul_3F32_3F32_3F32_out(tensor_t<float, 3> *a,
                                         tensor_t<float, 3> *b,
                                         tensor_t<float, 3> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  mul_out<float, 3>(a, b, r);
}

void _mlir_ciface_mul_4F32_4F32_4F32_out(tensor_t<float, 4> *a,
                                         tensor_t<float, 4> *b,
                                         tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  mul_out<float, 4>(a, b, r);
}

// nll_loss2d_forward_out

void _mlir_ciface_nll_loss2d_forward_1F32_1F32_4F32_3I64_1F32_out(
    tensor_t<float, 4> *a, tensor_t<uint64_t, 3> *b, tensor_t<float, 1> *c,
    int64_t d, int64_t e, tensor_t<float, 1> *r0, tensor_t<float, 1> *r1) {
  // std::cout << "aten_ops " << __func__ << "\n";
  using T = float;
  at::Tensor inputA = to_torch(a);
  at::TensorOptions options(at::ScalarType::Long);
  at::Tensor inputB = to_torch(b, options);
  at::Tensor inputC = to_torch(c);

  std::tuple<at::Tensor, at::Tensor> result =
      at::CPUType::nll_loss2d_forward(inputA, inputB, inputC, d, e);

  at::Tensor result0 = std::get<0>(result);
  at::Tensor result1 = std::get<1>(result);
  memcpy(r0->d, result0.data_ptr(), result0.numel() * sizeof(T));
  memcpy(r1->d, result1.data_ptr(), result1.numel() * sizeof(T));
}

// nll_loss2d_backward_out

void _mlir_ciface_nll_loss2d_backward_4F32_1F32_4F32_3I64_1F32_1F32_out(
    tensor_t<float, 1> *a, tensor_t<float, 4> *b, tensor_t<uint64_t, 3> *c,
    tensor_t<float, 1> *d, int32_t e, int32_t f, tensor_t<float, 1> *g,
    tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  using T = float;
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);
  at::TensorOptions options(at::ScalarType::Long);
  at::Tensor inputC = to_torch(c, options);
  at::Tensor inputD = to_torch(d);
  at::Tensor inputG = to_torch(g);

  at::Tensor result = at::CPUType::nll_loss2d_backward(inputA, inputB, inputC,
                                                       inputD, e, f, inputG);
  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

void _mlir_ciface_nll_loss_backward_2F32_1F32_2F32_1I64_1F32_1F32_out(
    tensor_t<float, 1> *a, tensor_t<float, 2> *b, tensor_t<uint64_t, 1> *c,
    tensor_t<float, 1> *d, int32_t e, int32_t f, tensor_t<float, 1> *g,
    tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  using T = float;
  at::Tensor inputA = to_torch(a);
  at::Tensor inputB = to_torch(b);
  at::TensorOptions options(at::ScalarType::Long);
  at::Tensor inputC = to_torch(c, options);
  at::Tensor inputD = to_torch(d);
  at::Tensor inputG = to_torch(g);

  at::Tensor result = at::CPUType::nll_loss_backward(inputA, inputB, inputC,
                                                     inputD, e, f, inputG);

  memcpy(r->d, result.data_ptr(), result.numel() * sizeof(T));
}

// nll_loss_forward_out

void _mlir_ciface_nll_loss_forward_1F32_1F32_2F32_1I64_1F32_out(
    tensor_t<float, 2> *a, tensor_t<uint64_t, 1> *b, tensor_t<float, 1> *c,
    int64_t d, int64_t e, tensor_t<float, 1> *r0, tensor_t<float, 1> *r1) {
  // std::cout << "aten_ops " << __func__ << "\n";
  using T = float;
  at::Tensor inputA = to_torch(a);
  at::TensorOptions options(at::ScalarType::Long);
  at::Tensor inputB = to_torch(b, options);
  at::Tensor inputC = to_torch(c);

  std::tuple<at::Tensor, at::Tensor> result =
      at::CPUType::nll_loss_forward(inputA, inputB, inputC, d, e);

  at::Tensor result0 = std::get<0>(result);
  at::Tensor result1 = std::get<1>(result);

  memcpy(r0->d, result0.data_ptr(), result0.numel() * sizeof(T));
  memcpy(r1->d, result1.data_ptr(), result1.numel() * sizeof(T));
}

// relu_out

void _mlir_ciface_relu_1F32_1F32_out(tensor_t<float, 1> *a,
                                     tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  relu_out<float, 1>(a, r);
}

void _mlir_ciface_relu_2F32_2F32_out(tensor_t<float, 2> *a,
                                     tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  relu_out<float, 2>(a, r);
}

void _mlir_ciface_relu_3F32_3F32_out(tensor_t<float, 3> *a,
                                     tensor_t<float, 3> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  relu_out<float, 3>(a, r);
}

void _mlir_ciface_relu_4F32_4F32_out(tensor_t<float, 4> *a,
                                     tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  relu_out<float, 4>(a, r);
}

// t_out

void _mlir_ciface_t_2F32_2F32_out(tensor_t<float, 2> *a,
                                  tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  t_out<float>(a, r);
}

// threshold_backward_out

void _mlir_ciface_threshold_backward_1F32_1F32_1F32_out(tensor_t<float, 1> *a,
                                                        tensor_t<float, 1> *b,
                                                        int32_t c,
                                                        tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  threshold_backward_out<float, 1>(a, b, c, r);
}

void _mlir_ciface_threshold_backward_2F32_2F32_2F32_out(tensor_t<float, 2> *a,
                                                        tensor_t<float, 2> *b,
                                                        int32_t c,
                                                        tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  threshold_backward_out<float, 2>(a, b, c, r);
}

void _mlir_ciface_threshold_backward_3F32_3F32_3F32_out(tensor_t<float, 3> *a,
                                                        tensor_t<float, 3> *b,
                                                        int32_t c,
                                                        tensor_t<float, 3> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  threshold_backward_out<float, 3>(a, b, c, r);
}

void _mlir_ciface_threshold_backward_4F32_4F32_4F32_out(tensor_t<float, 4> *a,
                                                        tensor_t<float, 4> *b,
                                                        int32_t c,
                                                        tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  threshold_backward_out<float, 4>(a, b, c, r);
}

// view_out

void _mlir_ciface_view_1F32_4F32_out(tensor_t<float, 4> *a, int32_t b,
                                     int32_t c, int32_t d, int32_t e,
                                     tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  view_out<float, 1, 4>(a, b, c, d, e, r);
}

void _mlir_ciface_view_1F32_3F32_out(tensor_t<float, 3> *a, int32_t b,
                                     int32_t c, int32_t d, int32_t e,
                                     tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  view_out<float, 1, 3>(a, b, c, d, e, r);
}

void _mlir_ciface_view_1F32_2F32_out(tensor_t<float, 2> *a, int32_t b,
                                     int32_t c, int32_t d, int32_t e,
                                     tensor_t<float, 1> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  view_out<float, 1, 2>(a, b, c, d, e, r);
}

void _mlir_ciface_view_2F32_4F32_out(tensor_t<float, 4> *a, int32_t b,
                                     int32_t c, int32_t d, int32_t e,
                                     tensor_t<float, 2> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  view_out<float, 2, 4>(a, b, c, d, e, r);
}

void _mlir_ciface_view_4F32_1F32_out(tensor_t<float, 1> *a, int32_t b,
                                     int32_t c, int32_t d, int32_t e,
                                     tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  view_out<float, 4, 1>(a, b, c, d, e, r);
}

void _mlir_ciface_view_4F32_2F32_out(tensor_t<float, 2> *a, int32_t b,
                                     int32_t c, int32_t d, int32_t e,
                                     tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  view_out<float, 4, 2>(a, b, c, d, e, r);
}

void _mlir_ciface_view_4F32_3F32_out(tensor_t<float, 3> *a, int32_t b,
                                     int32_t c, int32_t d, int32_t e,
                                     tensor_t<float, 4> *r) {
  // std::cout << "aten_ops " << __func__ << "\n";
  view_out<float, 4, 3>(a, b, c, d, e, r);
}
}
