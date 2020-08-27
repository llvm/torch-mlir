//===- ATenOpStatisticsUtils.h ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_OPSTATISTICSUTILS_H
#define NPCOMP_DIALECT_ATEN_OPSTATISTICSUTILS_H

#include "npcomp/Dialect/ATen/ATenDialect.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aten-op-stats"

/// This file generally contains utility methods that factor common code
/// out from operations that implement Statisticsopinterface.

namespace mlir {
namespace NPCOMP {
namespace aten {

// Return the op statistics for conv2d-like operations.
template <class T>
std::map<std::string, uint64_t> getConv2dStatistics(T *o, uint64_t groups) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = o->getResult().getType().template cast<TensorType>();
  TensorType inputTy = o->input().getType().template cast<TensorType>();
  TensorType weightTy = o->weight().getType().template cast<TensorType>();
  TensorType biasTy = o->bias().getType().template cast<TensorType>();

  uint64_t ofm_volume = getTensorVolume(resultTy);
  uint64_t ofm_depth = resultTy.getShape()[1];

  uint64_t ifm_depth = inputTy.getShape()[1];
  uint64_t kernel_height = weightTy.getShape()[2];
  uint64_t kernel_width = weightTy.getShape()[3];

  // Number of forward MACs per pixel =
  //  kernel_width * kernel_height * ifm_depth / groups
  uint64_t MACs_per_OFM = (ifm_depth / groups) * kernel_height * kernel_width;
  uint64_t total_MACs = ofm_volume * MACs_per_OFM;

  uint64_t ifm_volume = getTensorVolume(inputTy);
  uint64_t weight_volume = getTensorVolume(weightTy);
  uint64_t bias_volume = getTensorVolume(biasTy);

  // Should be gated on whether there is bias at all
  toReturn["ops:+"] = ofm_volume;
  toReturn["ops:MAC"] = total_MACs;

  toReturn["operand:0:activation_in"] = ifm_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  toReturn["operand:1:parameters_in:weight"] = weight_volume;
  toReturn["operand:2:parameters_in:bias"] = bias_volume;

  toReturn["reads"] = weight_volume + bias_volume + ifm_volume;
  toReturn["writes"] = ofm_volume;

  return toReturn;
}

// Return the op statistics for conv2dBackward-like operations.
template <typename T>
std::map<std::string, uint64_t> getConv2dBackwardStatistics(T op,
                                                            uint64_t groups) {

  std::map<std::string, uint64_t> toReturn;
  TensorType dx_out_resultTy =
      op.getResult(0).getType().template cast<TensorType>();
  uint64_t dx_out_volume = getTensorVolume(dx_out_resultTy);

  TensorType weightTy = op.getOperand(2).getType().template cast<TensorType>();
  uint64_t weight_volume = getTensorVolume(weightTy);
  uint64_t loss_in_depth = weightTy.getShape()[0];
  uint64_t kernel_width = weightTy.getShape()[2];
  uint64_t kernel_height = weightTy.getShape()[3];

  uint64_t MACs_per_loss =
      (loss_in_depth / groups) * kernel_height * kernel_width;

  uint64_t total_MACs = dx_out_volume * MACs_per_loss;

  TensorType ifmTy = op.getOperand(1).getType().template cast<TensorType>();
  uint64_t ifm_volume = getTensorVolume(ifmTy);
  auto ifm_shape = ifmTy.getShape();

  uint64_t ifm_bwh = ifm_shape[0] * ifm_shape[2] *
                     ifm_shape[3]; // Batch * height * width: the depth is in
                                   // the weight shape already
  total_MACs += ifm_bwh * weight_volume;

  TensorType dx_inTy = op.getOperand(0).getType().template cast<TensorType>();
  uint64_t dx_in_volume = getTensorVolume(dx_inTy);
  toReturn["ops:+"] = dx_in_volume;

  // Reads: Conv_backward reads 3 tensors: the loss in, the activation in and
  // the transposed weights
  toReturn["reads"] = dx_in_volume + ifm_volume + weight_volume;

  // Writes: Conv_backward writes 3 tensors: the loss out, gradients for the
  // weights, and gradients for the biases
  TensorType biasTy = op.getResult(2).getType().template cast<TensorType>();
  uint64_t bias_volume = getTensorVolume(biasTy);
  toReturn["writes"] = dx_out_volume + weight_volume + bias_volume;

  toReturn["ops:MAC"] = total_MACs;
  toReturn["operand:0:activation_in"] = dx_in_volume;
  toReturn["operand:1:activation_in"] = ifm_volume;
  toReturn["operand:2:parameters_in:weight"] = weight_volume;

  toReturn["result:0:grad:dx"] = dx_out_volume;
  toReturn["result:1:grad:dw"] = weight_volume;
  toReturn["result:2:grad:db"] = bias_volume;

  return toReturn;
}

// Return a model of the number of bytes needed to represent the operand of
// the given convolution-like operation with the given index.  The shape is
// assumed to be in NCHW order with a simple tiled model of data reuse.  TODO:
// convert this to a target-specific interface.
template <class T>
uint64_t getConv2dOperandTransferVolume(T *o, unsigned int idx, bool read) {

  if (!read)
    return 0;

  double vol = getTensorVolume(o->getOperand(idx).getType());

  TensorType inputTy = o->input().getType().template cast<TensorType>();
  TensorType weightTy = o->weight().getType().template cast<TensorType>();
  TensorType resultTy = o->getResult().getType().template cast<TensorType>();

  float filter_width = weightTy.getShape()[2];
  float filter_height = weightTy.getShape()[3];

  float batch_sw = inputTy.getShape()[0];
  float ifm_depth_sw = inputTy.getShape()[1];
  float ih = inputTy.getShape()[2];
  float iw = inputTy.getShape()[3];

  float ofm_depth_sw = resultTy.getShape()[1];

  const float batch_hw = 4;
  const float ifm_depth_hw = 32;
  const float ofm_depth_hw = 32;

  const float ifm_tile_height = 4;
  const float ifm_tile_width = 4;
  const float ofm_tile_height = 4;
  const float ofm_tile_width = 4;

  float ifm_aperture = ifm_tile_height - ceilf(filter_height / 2.0f);
  float ifm_overlap = ceilf(filter_height / 2.0f);

  float bl = ceilf(batch_sw / batch_hw);
  float ol = ceilf(ofm_depth_sw / ofm_depth_hw);
  float il = ceilf(ifm_depth_sw / ifm_depth_hw);

  float ifm_overhead = 1.0f;
  float weight_overhead = 1.0f;
  if (filter_width > 1) {
    ifm_overhead =
        ol * ifm_tile_height * ((ih - ifm_overlap) / (ih * ifm_aperture));
    weight_overhead = bl;
  } else {
    ifm_overhead = ol;
  }

  if (idx == 0) {
    LLVM_DEBUG(llvm::outs() << "ifm_overhead:" << ifm_overhead << "\n");
    return vol * ifm_overhead;
  }
  if (idx == 1) {
    LLVM_DEBUG(llvm::outs() << "weight_overhead:" << weight_overhead << "\n");
    return vol * weight_overhead;
  }
  return vol;
}

// Return a model of the number of bytes needed to represent the result of
// the given convolution-like operation with the given index.  The shape is
// assumed to be in NCHW order with a simple tiled model of data reuse.  TODO:
// convert this to a target-specific interface.
template <class T>
uint64_t getConv2dResultTransferVolume(T *o, unsigned int idx, bool write) {

  TensorType inputTy = o->input().getType().template cast<TensorType>();
  TensorType resultTy = o->getResult().getType().template cast<TensorType>();
  TensorType weightTy = o->weight().getType().template cast<TensorType>();
  float filter_width = weightTy.getShape()[2];
  // float filter_height = weightTy.getShape()[3];

  float ifm_depth_sw = inputTy.getShape()[1];
  const float ifm_depth_hw = 32;

  float il = ceilf(ifm_depth_sw / ifm_depth_hw);

  float write_output_overhead = 1.0f;
  float read_output_cost = 1.0f;

  if (filter_width > 1) {
    write_output_overhead = il;
    read_output_cost = il;
  }

  double vol = getTensorVolume(resultTy);

  if (write) {
    LLVM_DEBUG(llvm::outs()
               << "write_output_overhead:" << write_output_overhead << "\n");
    return vol * write_output_overhead;
  } else {
    LLVM_DEBUG(llvm::outs() << "read_output_cost:" << read_output_cost << "\n");
    return vol * read_output_cost;
  }
}

// Return the op statistics for matrixmultiply-like operations.
template <typename T> std::map<std::string, uint64_t> getMMOpStatistics(T op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType resultTy = op.getResult().getType().template cast<TensorType>();
  uint64_t ofm_volume = getTensorVolume(resultTy);

  // Use the weight tensor to find the number of input neurons
  TensorType lossTy = op.getOperand(0).getType().template cast<TensorType>();
  TensorType weightTy = op.getOperand(1).getType().template cast<TensorType>();
  uint64_t num_input_neurons = weightTy.getShape()[0];
  uint64_t total_MACs = ofm_volume * num_input_neurons;
  toReturn["ops:MAC"] = total_MACs;

  uint64_t loss_in_volume = getTensorVolume(lossTy);
  uint64_t weight_volume = getTensorVolume(weightTy);
  toReturn["reads"] = loss_in_volume + weight_volume;
  toReturn["writes"] = ofm_volume;

  toReturn["operand:0:activation_in"] = loss_in_volume;
  toReturn["operand:1:activation_in"] = weight_volume;
  toReturn["result:0:activation_out"] = ofm_volume;
  return toReturn;
}

// Return the op statistics for ReLU-like operations.
template <typename T>
std::map<std::string, uint64_t> getReLUOpStatistics(T op) {

  std::map<std::string, uint64_t> toReturn;

  TensorType inputTy = op.getOperand().getType().template cast<TensorType>();
  TensorType resultTy = op.getResult().getType().template cast<TensorType>();

  uint64_t in_volume = getTensorVolume(inputTy);
  uint64_t out_volume = getTensorVolume(resultTy);

  toReturn["operand:0:activation_in"] = in_volume;
  toReturn["result:0:activation_out"] = out_volume;
  toReturn["reads"] = in_volume;
  toReturn["writes"] = out_volume;
  toReturn["ops:>"] = out_volume;

  return toReturn;
}

} // namespace aten
} // namespace NPCOMP
} // namespace mlir

#endif
